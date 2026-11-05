import * as THREE from "three";
import { GaitController } from "./sim/GaitController.js";
import { Steering } from "./sim/Steering.js";
import { requestRender, scene, updateSkeleton } from "./scene.js";
import { isLocked, state } from "./state.js";

const _force = new THREE.Vector3();
const _jointWorldPos = new THREE.Vector3();
const _bonePos = new THREE.Vector3();
const _boneEuler = new THREE.Euler(0, 0, 0, "ZYX");
const _boneQEuler = new THREE.Quaternion();
const _boneQPre = new THREE.Quaternion();
const _boneScale = new THREE.Vector3();

/**
 * Unified Actor class representing any character entity in the scene
 * (both the anchor actor #0 and autonomous spawned agents).
 */
export class Actor {
  constructor(id, name, options = {}) {
    this.id = Number(id);
    this.name = name || `Actor #${this.id}`;
    this.isAnchor = (this.id === 0);
    this.isGpuAgent = false;

    // Three.js scene components
    this.group = options.group || new THREE.Group();
    this.group.name = `actor-${this.id}`;
    this.mesh = options.mesh || null;
    this.rootBone = options.rootBone || null;
    this.skeleton = options.skeleton || null;
    this.material = options.material || null;
    this.boneIndexMap = options.boneIndexMap || null;
    this.rigMapping = options.rigMapping || null;

    // Character Parameter Vectors
    const poseCount = options.poseCount || 204;
    const idCount = options.idCount || 0;
    const exprCount = options.exprCount || 0;
    this.poseValues = new Float32Array(poseCount);
    this.identityValues = new Float32Array(idCount);
    this.expressionValues = new Float32Array(exprCount);

    if (options.initialPoseValues) {
      this.poseValues.set(options.initialPoseValues);
    }
    if (options.initialIdentityValues) {
      this.identityValues.set(options.initialIdentityValues);
    }
    if (options.initialExpressionValues) {
      this.expressionValues.set(options.initialExpressionValues);
    }

    // Dynamic Joint Positions in World Space (for Gizmo raycasting and spheres)
    const jointCount = options.jointCount || 127;
    this.jointCount = jointCount;
    this.jointPositions = new Float32Array(jointCount * 3);

    // Kinematics & Physical State
    this.position = new THREE.Vector3();
    if (options.initialPosition) {
      this.position.copy(options.initialPosition);
      this.group.position.copy(this.position);
    }
    this.velocity = new THREE.Vector3();
    this.acceleration = new THREE.Vector3();
    this.maxSpeed = options.maxSpeed || 1.3;
    this.maxForce = options.maxForce || 4.5;
    this.scale = options.scale || 1.0;
    this.group.scale.setScalar(this.scale);
    this.radius = 0.45;

    // Behavioral State
    this.behavior = options.behavior || (this.isAnchor ? "manual" : "wander");
    this.heldPose = null;
    this.currentSpeed = 0;
    this.wanderAngle = Math.random() * Math.PI * 2;
    this.gaitPhase = options.gaitPhase !== undefined ? options.gaitPhase : (this.id * 1.618) % (Math.PI * 2);
    this.cadence = options.cadence || 3.3;
    this.activeClip = null; // Performing Mixamo ClipPlayer, or null for gait.

    // Joint channel buffer for rig evaluations
    this.thetaJ = new Float32Array(jointCount * 7);
    // Authored-pose base layer: each tick copies this, then gait is added on
    // top — so manual edits persist instead of snapping back to pure gait.
    this._baseThetaJ = new Float32Array(jointCount * 7);
    if (this.rigMapping) this.computeThetaJFromParams();

    // Initial appearance
    this.color = options.color || this._generateColor(this.id);
    this.pupil = options.pupil ?? 0.45;
    if (this.material && !this.isAnchor) {
      this.material.color.copy(this.color);
    }
  }

  _generateColor(id) {
    if (id === 0) return new THREE.Color(0xb48c46); // Default body color
    const goldenRatio = 0.618033988749895;
    const hue = (id * goldenRatio) % 1.0;
    return new THREE.Color().setHSL(hue, 0.65, 0.60);
  }

  /**
   * Gets parameter value by group and index.
   */
  getParamValue(group, index) {
    if (group === "pose") return this.poseValues[index] ?? 0;
    if (group === "identity") return this.identityValues[index] ?? 0;
    if (group === "expression") return this.expressionValues[index] ?? 0;
    return 0;
  }

  /**
   * Sets parameter value by group and index.
   */
  setParamValue(group, index, value) {
    const param = state.metadata?.groups?.[group]?.[index];
    if (param && isLocked(param)) return;
    if (group === "pose") {
      this.poseValues[index] = value;
      this.evaluatePose();
    } else if (group === "identity") {
      this.identityValues[index] = value;
    } else if (group === "expression") {
      this.expressionValues[index] = value;
    }
  }

  /**
   * Evaluates the full MHR linear rig mapping for this actor's pose parameter vector.
   */
  evaluatePose(rigMapping = this.rigMapping) {
    if (!rigMapping || !this.skeleton || !this.boneIndexMap) return;
    this.computeThetaJFromParams(rigMapping);
    this.applyThetaJ(rigMapping);
    if (actorManager.activeActor === this) {
      syncDisplayedSkeleton();
    }
  }

  computeThetaJFromParams(rigMapping = this.rigMapping) {
    const { TpMatrix, jointCount, tpCols } = rigMapping;

    // Compute thetaJ = TpMatrix @ poseValues
    for (let row = 0; row < jointCount * 7; row++) {
      let sum = 0;
      const rowOffset = row * tpCols;
      for (let col = 0; col < tpCols; col++) {
        const val = this.poseValues[col];
        if (val !== 0) {
          sum += TpMatrix[rowOffset + col] * val;
        }
      }
      this.thetaJ[row] = sum;
    }
    // Every pose edit flows through here: keep the tick's base layer in sync.
    this._baseThetaJ.set(this.thetaJ);
  }

  applyThetaJ(rigMapping = this.rigMapping) {
    if (!this.skeleton || !this.boneIndexMap) return;
    const { jointOffsets, jointPreRotations, jointCount } = rigMapping;

    // Apply joint channels to bones
    const bones = this.skeleton.bones;
    for (let j = 0; j < jointCount; j++) {
      const base = j * 7;
      const tx = this.thetaJ[base + 0];
      const ty = this.thetaJ[base + 1];
      const tz = this.thetaJ[base + 2];
      const rx = this.thetaJ[base + 3];
      const ry = this.thetaJ[base + 4];
      const rz = this.thetaJ[base + 5];
      const s = 2.0 ** this.thetaJ[base + 6];

      _bonePos.set(
        (jointOffsets[j * 3 + 0] + tx) / 100.0,
        (jointOffsets[j * 3 + 1] + ty) / 100.0,
        (jointOffsets[j * 3 + 2] + tz) / 100.0,
      );

      _boneEuler.set(rx, ry, rz);
      _boneQEuler.setFromEuler(_boneEuler);
      _boneQPre.set(
        jointPreRotations[j * 4 + 0],
        jointPreRotations[j * 4 + 1],
        jointPreRotations[j * 4 + 2],
        jointPreRotations[j * 4 + 3],
      );
      _boneQPre.multiply(_boneQEuler);

      _boneScale.set(s, s, s);

      const bone = bones[this.boneIndexMap[j]];
      if (bone) {
        bone.position.copy(_bonePos);
        bone.quaternion.copy(_boneQPre);
        bone.scale.copy(_boneScale);
      }
    }

    if (this.rootBone) {
      this.rootBone.updateWorldMatrix(true, true);
    }
    this.skeleton.update();
    this.updateWorldJointPositions();
  }

  /**
   * Updates world-space joint positions for 3D picking and Gizmo alignment.
   */
  updateWorldJointPositions() {
    if (!this.skeleton || !this.boneIndexMap) return;
    const bones = this.skeleton.bones;
    for (let j = 0; j < this.jointCount; j++) {
      const bone = bones[this.boneIndexMap[j]];
      if (bone) {
        bone.getWorldPosition(_jointWorldPos);
        this.jointPositions[j * 3 + 0] = _jointWorldPos.x;
        this.jointPositions[j * 3 + 1] = _jointWorldPos.y;
        this.jointPositions[j * 3 + 2] = _jointWorldPos.z;
      }
    }
  }

  /**
   * Applies an external steering force to this actor.
   */
  applyForce(force) {
    this.acceleration.add(force);
  }

  setColor(color) {
    if (typeof color === "string") {
      this.color.set(color);
    } else {
      this.color.copy(color);
    }
    if (this.material) {
      this.material.color.copy(this.color);
    }
  }

  setScale(scale) {
    this.scale = Math.max(0.5, Math.min(2.5, scale));
    this.group.scale.setScalar(this.scale);
    this.updateWorldJointPositions();
    // The joint spheres read actor joints (fresh above) but the skeleton
    // lines read the shared display buffer: re-sync it or the lines stay
    // at the old scale while the body and spheres move.
    if (actorManager.activeActor === this) {
      syncDisplayedSkeleton();
    }
  }

  setWireframe(enabled) {
    if (this.material) {
      this.material.wireframe = enabled;
    }
  }

  setMaxSpeed(speed) {
    this.maxSpeed = Math.max(0.1, Math.min(6.0, speed));
  }

  setCadence(cadence) {
    this.cadence = Math.max(1.0, Math.min(8.0, cadence));
  }

  setBehavior(behavior) {
    this.behavior = behavior;
  }

  /**
   * Simulation physics and locomotion tick (for autonomous movement).
   */
  update(deltaTime, environment, allActors) {
    if (this.activeClip && this.behavior === "manual") {
      // Performed clip on a stationary actor: pose only, no steering.
      this.applyClipPose(deltaTime);
      return;
    }
    // Manual actors need no per-tick hold (nothing overwrites them), so a
    // preset sticks from its one-shot apply while pose sliders keep working.
    if (this.behavior === "manual") {
      // Stationary manual posing mode
      this.velocity.set(0, 0, 0);
      this.currentSpeed = 0;
      this.updateWorldJointPositions();
      return;
    }

    // 1. Steering
    this._computeSteering(environment, allActors, deltaTime);

    // 2. Physics integration
    this.velocity.addScaledVector(this.acceleration, deltaTime);
    this.velocity.clampLength(0, this.maxSpeed);
    this.position.addScaledVector(this.velocity, deltaTime);
    this.acceleration.set(0, 0, 0);

    // Ground constraint
    this.position.y = 0;
    this.group.position.copy(this.position);

    // Heading rotation
    this.currentSpeed = this.velocity.length();
    if (this.currentSpeed > 0.05) {
      const targetAngle = Math.atan2(this.velocity.x, this.velocity.z);
      let diff = targetAngle - this.group.rotation.y;
      while (diff > Math.PI) diff -= Math.PI * 2;
      while (diff < -Math.PI) diff += Math.PI * 2;
      this.group.rotation.y += diff * Math.min(1.0, 7.0 * deltaTime);
    }

    // 3. Pose: a performed clip replaces locomotion gait while active.
    if (this.activeClip) {
      this.applyClipPose(deltaTime);
      return;
    }
    if (this.heldPose) {
      this.applyHeldPose();
      return;
    }
    const speedFactor = this.currentSpeed / this.maxSpeed;
    if (speedFactor > 0.06) {
      this.gaitPhase = (this.gaitPhase + this.currentSpeed * this.cadence * deltaTime) % (Math.PI * 2);
    } else {
      this.gaitPhase = (this.gaitPhase + 1.6 * deltaTime) % (Math.PI * 2);
    }

    this.applyLocomotionGait(speedFactor);
  }

  /**
   * Poses this actor from its gait phase (locomotion). The default path
   * evaluates on the CPU into the actor's own skeleton; GpuAgent overrides it
   * to fill thetaJ only (posing happens in the compute dispatch).
   */
  applyLocomotionGait(speedFactor) {
    if (this.rigMapping) {
      this.thetaJ.set(this._baseThetaJ);
      GaitController.evaluateAndApply(
        this.gaitPhase,
        speedFactor,
        this.skeleton.bones,
        this.boneIndexMap,
        this.rigMapping,
        this.thetaJ,
        true,
      );
    }

    if (this.rootBone) {
      this.rootBone.updateWorldMatrix(true, true);
    }
    this.skeleton.update();
    this.updateWorldJointPositions();
  }

  /** Restore the authored base pose (clip stop on an unticked actor). */
  restoreBasePose() {
    this.thetaJ.set(this._baseThetaJ);
    this.applyThetaJ(this.rigMapping);
    this.updateWorldJointPositions();
  }

  /**
   * Poses this actor from its performed Mixamo clip (replaces gait).
   * The default path fills thetaJ and poses the CPU skeleton; GpuAgent
   * overrides it to fill thetaJ only (posing happens in the dispatch).
   */
  applyClipPose(deltaTime) {
    if (!this.activeClip || !this.rigMapping) return;
    this.thetaJ.set(this._baseThetaJ);
    this.activeClip.update(deltaTime, this.thetaJ, this.group.quaternion);
    this.applyThetaJ(this.rigMapping);
  }

  /**
   * Holds a BVH preset pose (static retarget snapshot). Mirrors
   * applyClipPose: base layer first, then held rotations + root-Y bob.
   * GpuAgent overrides it to fill thetaJ only (see below).
   */
  applyHeldPose() {
    if (!this.heldPose || !this.rigMapping) return;
    this._mergeHeldPose();
    this.applyThetaJ(this.rigMapping);
  }

  _mergeHeldPose() {
    this.thetaJ.set(this._baseThetaJ);
    const { theta, joints, rootIdx } = this.heldPose;
    for (const j of joints) {
      const base = j * 7;
      this.thetaJ[base + 3] = theta[base + 3];
      this.thetaJ[base + 4] = theta[base + 4];
      this.thetaJ[base + 5] = theta[base + 5];
    }
    this.thetaJ[rootIdx * 7 + 1] += theta[rootIdx * 7 + 1];
  }

  _computeSteering(environment, allActors, deltaTime) {
    const { bounds, centerObstacle } = environment;
    // Grid neighborhood when the sim provides one (superset of every force
    // radius — each force still checks exact distance), else the full list.
    const neighbors = environment.neighborsOf ? environment.neighborsOf(this) : allActors;

    if (this.behavior === "idle") {
      this.velocity.multiplyScalar(0.85);
      return;
    } else if (this.behavior === "flock") {
      const sep = Steering.separation(this, neighbors, 1.4);
      const ali = Steering.alignment(this, neighbors, 3.5);
      const coh = Steering.cohesion(this, neighbors, 3.5);
      const wander = Steering.wander(this, deltaTime, 2.0, 1.0, 2.0);

      this.applyForce(_force.copy(sep).multiplyScalar(1.6));
      this.applyForce(_force.copy(ali).multiplyScalar(1.0));
      this.applyForce(_force.copy(coh).multiplyScalar(0.8));
      this.applyForce(_force.copy(wander).multiplyScalar(0.5));
    } else if (this.behavior === "circle") {
      const circ = Steering.circleCenter(this, centerObstacle.position, 3.2);
      const sep = Steering.separation(this, neighbors, 1.2);
      this.applyForce(_force.copy(circ).multiplyScalar(1.5));
      this.applyForce(_force.copy(sep).multiplyScalar(1.8));
    } else if (this.behavior === "scatter") {
      const flee = Steering.flee(this, centerObstacle.position, 5.5);
      const sep = Steering.separation(this, neighbors, 1.5);
      const wander = Steering.wander(this, deltaTime, 1.5, 1.0, 3.0);
      this.applyForce(_force.copy(flee).multiplyScalar(2.0));
      this.applyForce(_force.copy(sep).multiplyScalar(1.5));
      this.applyForce(_force.copy(wander).multiplyScalar(0.5));
    } else {
      // Default: wander
      const wander = Steering.wander(this, deltaTime, 2.2, 1.2, 2.5);
      const sep = Steering.separation(this, neighbors, 1.3);
      this.applyForce(_force.copy(wander).multiplyScalar(1.2));
      this.applyForce(_force.copy(sep).multiplyScalar(1.6));
    }

    // Avoid center model
    if (!this.isAnchor) {
      const avoid = Steering.avoidCenter(this, centerObstacle.position, centerObstacle.radius);
      this.applyForce(_force.copy(avoid).multiplyScalar(2.2));
    }

    // Boundary containment
    const contain = Steering.contain(this, bounds, 1.4);
    this.applyForce(_force.copy(contain).multiplyScalar(2.5));
  }

  dispose() {
    if (this.group.parent) {
      this.group.parent.remove(this.group);
    }
    if (this.material && !this.isAnchor) {
      this.material.dispose();
    }
    if (this.skeleton && !this.isAnchor) {
      this.skeleton.dispose();
    }
  }
}

/**
 * Global Actor Manager managing character registry, active actor target, and selection changes.
 */
export class ActorManager {
  constructor() {
    this.actors = new Map();
    this.activeActorId = 0;
    this._listeners = new Set();
  }

  get activeActor() {
    return this.actors.get(this.activeActorId) || this.actors.get(0) || null;
  }

  addChangeListener(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  _notifyChange() {
    for (const fn of this._listeners) fn(this.activeActor);
  }

  registerActor(actor) {
    this.actors.set(actor.id, actor);
    this._notifyChange();
    return actor;
  }

  unregisterActor(id) {
    const actor = this.actors.get(id);
    if (!actor || actor.isAnchor) return;
    actor.dispose();
    this.actors.delete(id);
    if (this.activeActorId === id) {
      this.setActiveActor(0);
    } else {
      this._notifyChange();
    }
  }

  setActiveActor(id) {
    const targetId = Number(id);
    if (this.activeActorId === targetId && this.actors.has(targetId)) return;
    if (this.actors.has(targetId)) {
      this.activeActorId = targetId;
      this._notifyChange();
      requestRender();
    }
  }

  getActor(id) {
    return this.actors.get(Number(id)) || null;
  }

  getAllActors() {
    return Array.from(this.actors.values());
  }

  clearNonAnchorActors() {
    for (const [id, actor] of this.actors) {
      if (!actor.isAnchor) {
        actor.dispose();
        this.actors.delete(id);
      }
    }
    this.setActiveActor(0);
  }
}

export const actorManager = new ActorManager();

// Gizmo sphere refresh, registered by gizmo.js at init (a static import
// here would cycle: gizmo imports this module).
let sphereSyncFn = null;

export function registerSphereSync(fn) {
  sphereSyncFn = fn;
}

/**
 * Points the skeleton overlay + bone tooltip at the active actor. The
 * display buffer always mirrors whoever is active: an agent's live joints
 * (GPU agents read the probe, posed per tick while running and on select
 * while paused), else the anchor's last evaluated pose.
 */
export function syncDisplayedSkeleton() {
  if (!state.jointPositions || !state.metadata) return;
  const active = actorManager.activeActor;
  if (active && !active.isAnchor && active.jointPositions.length === state.jointPositions.length) {
    active.updateWorldJointPositions();
    state.jointPositions.set(active.jointPositions);
  } else {
    const anchor = actorManager.getActor(0);
    if (anchor && anchor.jointPositions.length === state.jointPositions.length) {
      state.jointPositions.set(anchor.jointPositions);
    }
  }
  updateSkeleton(state.jointPositions);
  sphereSyncFn?.();
}
