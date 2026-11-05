import * as THREE from "three";
import { Actor } from "../actor.js";
import { GaitController } from "../sim/GaitController.js";

/**
 * GPU-resident crowd agent. The mesh is a plain (non-skinned) Mesh whose
 * position/normal attributes are compute-shader outputs; posing happens in
 * the GpuCrowd provider from this agent's thetaJ + shape weights, so per-tick
 * CPU work is steering + gait values only. The shared probe skeleton
 * (posed per-frame for whichever GPU agent is active) backs gizmo/picking
 * reads; see provider.poseProbe.
 */
export class GpuAgent extends Actor {
  constructor(id, pool, lod, sourceMesh, rigMapping, boneIndexMap, options = {}) {
    const goldenRatio = 0.618033988749895;
    const color = options.color || new THREE.Color().setHSL((id * goldenRatio) % 1.0, 0.65, 0.60);
    const record = pool.spawnSlot({
      color,
      wireframe: sourceMesh.material.wireframe,
      baseMaterial: sourceMesh.material,
    });

    const group = new THREE.Group();
    group.name = `agent-${id}`;
    group.add(record.mesh);

    const scale = options.scale || (0.94 + ((id * 37) % 13) / 100);
    const maxSpeed = options.maxSpeed || (1.1 + ((id * 17) % 5) * 0.1);

    super(id, `Actor #${id}`, {
      group,
      mesh: record.mesh,
      rootBone: pool.probe.rootBone,
      skeleton: pool.probe.skeleton,
      material: record.material,
      boneIndexMap,
      rigMapping,
      poseCount: rigMapping.tpCols,
      idCount: options.idCount || 0,
      exprCount: options.exprCount || 0,
      jointCount: rigMapping.jointCount,
      initialPosition: options.initialPosition,
      maxSpeed,
      maxForce: options.maxForce || 4.5,
      scale,
      behavior: options.behavior || "wander",
      color,
      cadence: 3.3,
      gaitPhase: (id * 1.61803) % (Math.PI * 2),
    });

    this.isGpuAgent = true;
    this._pool = pool;
    this._record = record;
    this.lod = lod;
    this.lodPin = options.lodPin ?? null;
  }

  /**
   * Moves this agent to another LOD pool (auto-LOD migration). Pose state
   * (thetaJ/identity/expression/phase) is LOD-independent and untouched;
   * only the slot + mesh change. The new mesh pays first-render init on
   * its first frame (~65ms, same as spawn) — callers budget migrations.
   */
  migrateTo(pool, lod) {
    const oldRecord = this._record;
    const oldPool = this._pool;
    const record = pool.spawnSlot({
      color: this.color,
      wireframe: oldRecord.material.wireframe,
      baseMaterial: oldRecord.material,
    });
    this.group.remove(oldRecord.mesh);
    oldPool.releaseSlot(oldRecord);
    this.group.add(record.mesh);
    this.mesh = record.mesh;
    this.material = record.material;
    this.rootBone = pool.probe.rootBone;
    this.skeleton = pool.probe.skeleton;
    this._record = record;
    this._pool = pool;
    this.lod = lod;
  }

  applyLocomotionGait(speedFactor) {
    if (!this.rigMapping) return;
    this.thetaJ.set(this._baseThetaJ);
    const values = GaitController.gaitValues(this.gaitPhase, speedFactor);
    GaitController.accumulateThetaJ(values, this.thetaJ, this.rigMapping, true);
  }

  applyClipPose(deltaTime) {
    if (!this.activeClip || !this.rigMapping) return;
    this.thetaJ.set(this._baseThetaJ);
    this.activeClip.update(deltaTime, this.thetaJ, this.group.quaternion);
  }

  applyHeldPose() {
    if (!this.heldPose || !this.rigMapping) return;
    this._mergeHeldPose();
  }

  dispose() {
    if (this._disposed) return;
    this._disposed = true;
    if (this.group.parent) {
      this.group.parent.remove(this.group);
    }
    // The probe skeleton is provider-owned: never disposed here (it may be
    // parented under this group; it is reparented on the next probe pass).
    if (this._pool && this._record) {
      this._pool.releaseSlot(this._record);
      this._record = null;
      this._pool = null;
    }
  }
}
