import * as THREE from "three";
import * as TSL from "three/tsl";
import { state } from "./state.js";
import { scene } from "./scene.js";
import { createEyeballGeometry, createEyeMaterial, eyesOral } from "./eyesOral.js";

// Instanced eyeballs for every crowd agent (CPU + GPU), one draw call.
//
// eyesOral.js parents meshes to per-agent bones; GPU agents have no
// per-agent bones and N agents x M meshes would explode draw calls. Instead
// this fills one InstancedMesh (2 instances per agent) from per-agent head
// matrices: CPU agents expose their head bone directly, GPU agents get a
// CPU FK walk over the root->head chain from the thetaJ they already fill
// each tick (same inputs + composition as the compute FK, so the matrices
// match the rendered meshes by construction).
//
// Life: every agent gets slow gaze wander baked into its instance matrices.
// GPU agents additionally blink through their own eyesClosed expression
// weights, which provider.update() uploads every tick anyway (zero extra
// cost). CPU agents share the anchor's geometry, so per-agent blinking is
// impossible for them and they only wander.

const BLINK_MS = 200;

const _groupM = new THREE.Matrix4();
const _headM = new THREE.Matrix4();
const _localM = new THREE.Matrix4();
const _eyeM = new THREE.Matrix4();
const _pos = new THREE.Vector3();
const _scl = new THREE.Vector3();
const _euler = new THREE.Euler(0, 0, 0, "ZYX");
const _qEuler = new THREE.Quaternion();
const _qPre = new THREE.Quaternion();
const _qGaze = new THREE.Quaternion();
const _mAnchor = new THREE.Matrix4();

function gazeQuat(yawDeg, pitchDeg, out) {
  _euler.set(
    THREE.MathUtils.degToRad(-pitchDeg),
    THREE.MathUtils.degToRad(yawDeg),
    0,
  );
  return out.setFromEuler(_euler);
}

export class CrowdEyes {
  constructor() {
    this.enabled = true;
    this.blinkEnabled = true;
    this.mesh = null;
    this.capacity = 0;
    this.count = 0;
    this._pupilArray = null;
    this._pupilAttr = null;
    this.lastMs = 0;
    this._headJoint = -1;
    this._chain = null;
    this._anchorL = new THREE.Vector3();
    this._anchorR = new THREE.Vector3();
    this._modQuat = new THREE.Quaternion();
    this._ready = false;
    this._closedL = -1;
    this._closedR = -1;
    this._jawDrop = -1;
    this._gazeOverride = new Map();
  }

  jawDropIndex() {
    if (!this._ready && !this._ensureReady()) return -1;
    return this._jawDrop;
  }

  // Manual per-agent gaze (Actor card Face group). Null yaw clears back to
  // wander. Tissue blendshapes are deliberately untouched — instance
  // rotation only, invisible-difference at crowd distance, zero upload.
  setAgentGaze(id, yaw, pitch) {
    const key = Number(id);
    if (yaw === null || yaw === undefined) this._gazeOverride.delete(key);
    else this._gazeOverride.set(key, { yaw: Number(yaw), pitch: Number(pitch) });
  }

  getAgentGaze(id) {
    return this._gazeOverride.get(Number(id)) || null;
  }

  // Starts a blink on one GPU agent; the next update() picks it up. Returns
  // false for CPU agents (shared anchor mesh cannot blink per-agent).
  triggerAgentBlink(agent) {
    if (!agent || !agent.isGpuAgent) return false;
    if (!this._ready && !this._ensureReady()) return false;
    if (this._closedL < 0) return false;
    agent._crowdBlink = { next: 0, t0: -1, baseL: 0, baseR: 0 };
    return true;
  }

  _ensureReady() {
    if (this._ready) return true;
    const names = state.metadata?.jointNames;
    const rig = state.rigMapping;
    if (!names || !rig) return false;
    const head = names.indexOf("c_head");
    const lEye = names.indexOf("l_eye");
    const rEye = names.indexOf("r_eye");
    if (head < 0 || lEye < 0 || rEye < 0) return false;
    // Root->head chain for the GPU-agent FK walk.
    const parents = state.metadata.jointParents;
    const chain = [];
    for (let j = head; j >= 0; j = parents[j]) {
      chain.push(j);
      if (j === 0) break;
    }
    chain.reverse();
    this._chain = chain;
    this._headJoint = head;
    const off = rig.jointOffsets;
    this._anchorL.set(off[lEye * 3] / 100, off[lEye * 3 + 1] / 100, off[lEye * 3 + 2] / 100);
    this._anchorR.set(off[rEye * 3] / 100, off[rEye * 3 + 1] / 100, off[rEye * 3 + 2] / 100);
    // Module->head basis: reuse actor #0's when attached (guaranteed
    // consistent), else derive from offsets if the eye joints parent
    // directly under the head.
    if (eyesOral.attached) {
      this._modQuat.copy(eyesOral.modQuat);
    } else if (parents[lEye] === head && parents[rEye] === head) {
      const crown = names.indexOf("c_head_null");
      if (crown < 0 || parents[crown] !== head) return false;
      const up = new THREE.Vector3(off[crown * 3], off[crown * 3 + 1], off[crown * 3 + 2]).normalize();
      const lat = this._anchorL.clone().sub(this._anchorR).normalize();
      const fwd = new THREE.Vector3().crossVectors(lat, up).normalize();
      const mid = this._anchorL.clone().add(this._anchorR).multiplyScalar(0.5);
      if (mid.dot(fwd) < 0) {
        lat.negate();
        fwd.negate();
      }
      this._modQuat.setFromRotationMatrix(_localM.makeBasis(lat, up, fwd));
    } else {
      return false;
    }
    const expr = state.metadata.groups?.expression || [];
    this._closedL = expr.findIndex((e) => e.name === "eyesClosed_L");
    this._closedR = expr.findIndex((e) => e.name === "eyesClosed_R");
    this._jawDrop = expr.findIndex((e) => e.name === "jawDrop");
    this._ready = true;
    return true;
  }

  _ensureMesh(need) {
    if (this.mesh && need <= this.capacity) return;
    const capacity = Math.max(need, this.capacity * 2, 256);
    if (this.mesh) {
      scene.remove(this.mesh);
      this.mesh.dispose();
    }
    // Own material: pupils are per-instance here (actor #0 keeps a uniform),
    // so the shader reads the aPupil instanced attribute instead.
    const material = createEyeMaterial({ pupil: TSL.attribute("aPupil") });
    const geo = createEyeballGeometry();
    this._pupilArray = new Float32Array(capacity).fill(0.45);
    this._pupilAttr = new THREE.InstancedBufferAttribute(this._pupilArray, 1);
    this._pupilAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("aPupil", this._pupilAttr);
    this.mesh = new THREE.InstancedMesh(geo, material, capacity);
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.mesh.frustumCulled = false;
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.capacity = capacity;
    scene.add(this.mesh);
  }

  // Head matrix (rig space) from a GPU agent's CPU-side thetaJ. Replicates
  // Actor.applyThetaJ composition over the root->head chain only.
  _headFromThetaJ(agent, out) {
    const { jointOffsets, jointPreRotations } = agent.rigMapping;
    const thetaJ = agent.thetaJ;
    out.identity();
    for (const j of this._chain) {
      const base = j * 7;
      _pos.set(
        (jointOffsets[j * 3] + thetaJ[base]) / 100,
        (jointOffsets[j * 3 + 1] + thetaJ[base + 1]) / 100,
        (jointOffsets[j * 3 + 2] + thetaJ[base + 2]) / 100,
      );
      _euler.set(thetaJ[base + 3], thetaJ[base + 4], thetaJ[base + 5]);
      _qEuler.setFromEuler(_euler);
      _qPre.set(
        jointPreRotations[j * 4],
        jointPreRotations[j * 4 + 1],
        jointPreRotations[j * 4 + 2],
        jointPreRotations[j * 4 + 3],
      );
      _qPre.multiply(_qEuler);
      const s = 2.0 ** thetaJ[base + 6];
      _localM.compose(_pos, _qPre, _scl.set(s, s, s));
      out.multiply(_localM);
    }
    agent.group.updateMatrix();
    return out.premultiply(agent.group.matrix);
  }

  _blinkValue(agent, timeMs) {
    // Returns the eyesClosed weight for this tick, or -1 when idle.
    if (!agent._crowdBlink) {
      agent._crowdBlink = {
        next: timeMs + 1500 + ((agent.id * 7919) % 3000),
        t0: -1,
        baseL: 0,
        baseR: 0,
      };
    }
    const st = agent._crowdBlink;
    if (st.t0 < 0) {
      if (timeMs < st.next) return -1;
      st.t0 = timeMs;
      st.baseL = agent.expressionValues[this._closedL] || 0;
      st.baseR = agent.expressionValues[this._closedR] || 0;
    }
    const t = (timeMs - st.t0) / BLINK_MS;
    if (t >= 1) {
      st.t0 = -1;
      st.next = timeMs + 2200 + ((agent.id * 104729 + timeMs) % 2600);
      return 0;
    }
    const raw = t < 0.35 ? t / 0.35 : 1 - (t - 0.35) / 0.65;
    return raw * raw * (3 - 2 * raw);
  }

  update(agents, simTimeSec, timeMs) {
    const t0 = performance.now();
    if (!this.enabled || !this._ensureReady()) {
      if (this.mesh) this.mesh.count = 0;
      this.count = 0;
      this.lastMs = performance.now() - t0;
      return;
    }
    // Over-allocate check first so the mesh exists before filling.
    this._ensureMesh(agents.length * 2);
    let n = 0;
    const canBlink = this.blinkEnabled && this._closedL >= 0 && this._closedR >= 0;
    for (const agent of agents) {
      if (!agent.mesh || agent.mesh.visible === false) continue;
      if (agent.isGpuAgent) {
        if (canBlink && agent.expressionValues.length > this._closedR) {
          const v = this._blinkValue(agent, timeMs);
          if (v >= 0) {
            const st = agent._crowdBlink;
            agent.expressionValues[this._closedL] = Math.max(st.baseL, v);
            agent.expressionValues[this._closedR] = Math.max(st.baseR, v);
          }
        }
        this._headFromThetaJ(agent, _headM);
      } else {
        if (!agent.skeleton || !agent.boneIndexMap) continue;
        const bone = agent.skeleton.bones[agent.boneIndexMap[this._headJoint]];
        if (!bone) continue;
        _headM.copy(bone.matrixWorld);
      }
      // Manual override wins; otherwise slow wander (deterministic phases).
      const manual = this._gazeOverride.get(agent.id);
      const yaw = manual ? manual.yaw
        : 11 * Math.sin(simTimeSec * 0.45 + agent.id * 2.399)
          + 5 * Math.sin(simTimeSec * 0.21 + agent.id * 1.307);
      const pitch = manual ? manual.pitch
        : 7 * Math.sin(simTimeSec * 0.53 + agent.id * 3.733)
          + 3 * Math.sin(simTimeSec * 0.17 + agent.id * 0.911);
      gazeQuat(yaw, pitch, _qGaze);
      _eyeM.copy(_headM);
      const pupil = agent.pupil ?? 0.45;
      for (const anchor of [this._anchorL, this._anchorR]) {
        _mAnchor.compose(anchor, this._modQuat, _scl.set(1, 1, 1));
        _localM.copy(_eyeM).multiply(_mAnchor);
        // Module-space gaze rotation about the eyeball center.
        _mAnchor.makeRotationFromQuaternion(_qGaze);
        _localM.multiply(_mAnchor);
        this.mesh.setMatrixAt(n, _localM);
        this._pupilArray[n] = pupil;
        n++;
      }
    }
    this.mesh.count = n;
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this._pupilAttr) this._pupilAttr.needsUpdate = true;
    this.count = n;
    this.lastMs = performance.now() - t0;
  }

  // World head matrix for a GPU agent (FK walk over root->head chain).
  // Shared with crowdMouth so mouths ride the same matrices as the eyes.
  headMatrixFor(agent, out) {
    if (!this._ready && !this._ensureReady()) return false;
    if (!agent.isGpuAgent) return false;
    this._headFromThetaJ(agent, out);
    return true;
  }

  // Mouth frame: eye midpoint (head space) + module orientation.
  mouthFrame() {
    if (!this._ready && !this._ensureReady()) return null;
    return {
      mid: this._anchorL.clone().add(this._anchorR).multiplyScalar(0.5),
      quat: this._modQuat,
    };
  }

  setEnabled(on) {
    this.enabled = Boolean(on);
    if (!this.enabled && this.mesh) this.mesh.count = 0;
    this.count = this.enabled ? this.count : 0;
  }

  setBlinkEnabled(on) {
    this.blinkEnabled = Boolean(on);
  }
}

export const crowdEyes = new CrowdEyes();
