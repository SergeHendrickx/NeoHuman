// GPU crowd compute provider (Stage C: batched FK + skinning on GPU).
//
// Browser-only: needs a WebGPURenderer + three/tsl. Owns all sim-lifetime GPU
// resources (static rig/body/LOD buffers, dynamic theta/weight/shape buffers,
// mega bone/output buffers, the two batched compute nodes, and the CPU probe
// skeleton). Per tick the whole population costs exactly two dispatches; each
// agent's mesh renders its slot slice of the shared mega outputs via a biased
// index copy. The CPU agent path stays untouched as the fallback; this
// provider never runs on WebGL.
import * as THREE from "three";
import { Actor } from "../actor.js";
import {
  GPU_MAX_AGENTS,
  allocBoneMega,
  allocOutputMega,
  allocShapeBuffer,
  allocThetaBuffer,
  allocWeightBuffer,
  biasSlotIndex,
  buildBodyBuffer,
  buildLodBuffer,
  buildRigBuffer,
  computeBodyLayout,
  computeLodLayout,
  computeRigLayout,
  neutralOutputNormals,
  neutralOutputPositions,
} from "./layouts.js";
import { buildBatchedKernels } from "./kernels.js";

function flattenInverses(boneInverses, jointCount) {
  const out = new Float32Array(jointCount * 16);
  for (let j = 0; j < jointCount; j++) {
    out.set(boneInverses[j].elements, j * 16);
  }
  return out;
}

export function computeNeutralNormals(neutral4, srcVertCount, indexArray) {
  const positions = new Float32Array(srcVertCount * 3);
  for (let v = 0; v < srcVertCount; v++) {
    positions[v * 3] = neutral4[v * 4];
    positions[v * 3 + 1] = neutral4[v * 4 + 1];
    positions[v * 3 + 2] = neutral4[v * 4 + 2];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indexArray, 1));
  geometry.computeVertexNormals();
  const normals = geometry.getAttribute("normal").array.slice();
  geometry.dispose();
  return normals;
}

export class GpuCrowd {
  constructor() {
    this.ready = false;
    this.renderer = null;
    this.maxSlots = GPU_MAX_AGENTS;
    this.lod = null;
    this.TSL = null;
    this.StorageBufferAttribute = null;
    this.key = null;
    this.dims = null;
    this.layouts = null;
    this.static = null;
    this.dynamic = null;
    this.records = new Map();
    this.freeSlots = [];
    this.probe = null;
    this.uvAttribute = null;
    this.lastFillMs = 0;
    this.lastQueueMs = 0;
    this.mega = null;
    this.fkAll = null;
    this.skinAll = null;
    this.uFkTotal = null;
    this.uSkinRowW = null;
    this.uSkinTotal = null;
    this.compiled = false;
    this._anchorIndexArray = null;
    this._slotIndices = [];
  }

  static async create(renderer, source, maxSlots = GPU_MAX_AGENTS) {
    const gpu = new GpuCrowd();
    gpu.renderer = renderer;
    gpu.maxSlots = maxSlots;
    gpu.TSL = await import("three/tsl");
    ({ StorageBufferAttribute: gpu.StorageBufferAttribute } = await import("three/webgpu"));
    gpu._buildStatic(source);
    gpu._buildProbe(source);
    gpu.ready = true;
    console.info(
      "[gpu] crowd compute ready (LOD %d, %d joints, %d slots).",
      source.lod, source.jointCount, maxSlots,
    );
    return gpu;
  }

  static sourceKey(source) {
    return [source.lod, source.jointCount, source.vertCount, source.srcVertCount,
      source.shapeCount, source.topologyVersion].join(":");
  }

  _buildStatic(source) {
    const { jointCount: J, vertCount: V, srcVertCount: V1, shapeCount: B } = source;
    const S = this.maxSlots;
    this.lod = source.lod;
    const rigLayout = computeRigLayout({ jointCount: J, vertCount: V, srcVertCount: V1 });
    const bodyLayout = computeBodyLayout({ srcVertCount: V1, shapeCount: B });
    const lodLayout = computeLodLayout({ targetCount: V });
    const SBA = this.StorageBufferAttribute;
    const rigArray = buildRigBuffer(rigLayout, { jointCount: J, vertCount: V, srcVertCount: V1 }, source);
    const bodyArray = buildBodyBuffer(bodyLayout, { srcVertCount: V1, shapeCount: B }, source);
    const lodArray = buildLodBuffer(lodLayout, {
      targetCount: V,
      sourceCount: V1,
      mapping: source.lodMapping,
    });
    const thetaArray = allocThetaBuffer(S, J);
    const weightArray = allocWeightBuffer(S, B);
    const shapeArray = allocShapeBuffer(S);
    const thetaAttr = new SBA(thetaArray, 1);
    thetaAttr.setUsage(THREE.DynamicDrawUsage);
    const weightAttr = new SBA(weightArray, 1);
    weightAttr.setUsage(THREE.DynamicDrawUsage);
    const shapeAttr = new SBA(shapeArray, 1);
    shapeAttr.setUsage(THREE.DynamicDrawUsage);
    this.static = {
      rigAttr: new SBA(rigArray, 1),
      bodyAttr: new SBA(bodyArray, 1),
      lodAttr: new SBA(lodArray, 1),
    };
    this.dynamic = { thetaAttr, weightAttr, shapeAttr, thetaArray, weightArray, shapeArray };
    this.layouts = { rigLayout, bodyLayout, lodLayout };
    this.dims = { J, V, V1, B };
    this.key = GpuCrowd.sourceKey(source);
    this.uvAttribute = source.uvAttribute || null;
    this._anchorIndexArray = source.indexAttribute.array;
    // Mega outputs, one slice per slot, neutral-filled so a first render is
    // never garbage (every active slot is overwritten by skinAll each tick).
    const neutralPos = neutralOutputPositions({
      targetCount: V,
      sourceCount: V1,
      mapping: source.lodMapping,
      neutral4: source.neutral4,
    });
    const neutralNrm = neutralOutputNormals({
      targetCount: V,
      sourceCount: V1,
      mapping: source.lodMapping,
      neutralNrm: source.neutralNrm,
    });
    const posMegaArray = allocOutputMega(S, V);
    const nrmMegaArray = allocOutputMega(S, V);
    for (let s = 0; s < S; s++) {
      posMegaArray.set(neutralPos, s * V * 3);
      nrmMegaArray.set(neutralNrm, s * V * 3);
    }
    const bonesMega = new SBA(allocBoneMega(S, J), 1);
    const posMegaAttr = new SBA(posMegaArray, 3);
    const nrmMegaAttr = new SBA(nrmMegaArray, 3);
    this.mega = { bonesMega, posMegaAttr, nrmMegaAttr };
    // The two batched kernels, built once: per-tick uniforms only change
    // the guarded thread totals, never the graph.
    this.uFkTotal = this.TSL.uniform(0);
    this.uSkinRowW = this.TSL.uniform(0);
    this.uSkinTotal = this.TSL.uniform(0);
    const { fkAll, skinAll } = buildBatchedKernels(this.TSL, {
      rig: this.static.rigAttr,
      rigCount: this.layouts.rigLayout.totalFloats,
      body: this.static.bodyAttr,
      bodyCount: this.layouts.bodyLayout.totalFloats,
      theta: this.dynamic.thetaAttr,
      thetaCount: this.dynamic.thetaArray.length,
      weights: this.dynamic.weightAttr,
      weightCount: this.dynamic.weightArray.length,
      shape: this.dynamic.shapeAttr,
      bones: this.mega.bonesMega,
      outPos: this.mega.posMegaAttr,
      outNormal: this.mega.nrmMegaAttr,
      lod: this.static.lodAttr,
      lodCount: this.layouts.lodLayout.totalFloats,
      uFkTotal: this.uFkTotal,
      uSkinRowW: this.uSkinRowW,
      uSkinTotal: this.uSkinTotal,
      dims: {
        S,
        J,
        V,
        V1,
        B,
        rig: this.layouts.rigLayout.offsets,
        body: this.layouts.bodyLayout.offsets,
        lod: this.layouts.lodLayout.offsets,
      },
    });
    this.fkAll = fkAll;
    this.skinAll = skinAll;
  }

  _buildProbe(source) {
    const rootBone = source.rootBone.clone(true);
    const byName = new Map();
    rootBone.traverse((obj) => {
      if (obj.isBone) byName.set(obj.name, obj);
    });
    const bones = source.skeleton.bones.map((bone) => {
      const cloned = byName.get(bone.name);
      if (!cloned) throw new Error(`GPU probe: missing bone "${bone.name}".`);
      return cloned;
    });
    const skeleton = new THREE.Skeleton(bones, source.skeleton.boneInverses);
    this.probe = new Actor(-1, "gpu-probe", {
      skeleton,
      rootBone,
      boneIndexMap: source.boneIndexMap,
      rigMapping: source.rigMapping,
      jointCount: source.jointCount,
      poseCount: source.poseCount,
    });
  }

  _destroyAttribute(attr) {
    try {
      const backend = this.renderer && this.renderer.backend;
      if (backend && typeof backend.destroyAttribute === "function") {
        backend.destroyAttribute(attr);
      }
    } catch {
      // Best effort: dropping the reference still frees the CPU side.
    }
  }

  reset() {
    for (const record of this.records.values()) this._releaseRecord(record, false);
    this.records.clear();
    this.freeSlots = [];
    if (this.static) {
      this._destroyAttribute(this.static.rigAttr);
      this._destroyAttribute(this.static.bodyAttr);
      this._destroyAttribute(this.static.lodAttr);
      this.static = null;
    }
    if (this.dynamic) {
      this._destroyAttribute(this.dynamic.thetaAttr);
      this._destroyAttribute(this.dynamic.weightAttr);
      this._destroyAttribute(this.dynamic.shapeAttr);
      this.dynamic = null;
    }
    if (this.mega) {
      this._destroyAttribute(this.mega.bonesMega);
      this._destroyAttribute(this.mega.posMegaAttr);
      this._destroyAttribute(this.mega.nrmMegaAttr);
      this.mega = null;
    }
    for (const attr of this._slotIndices) {
      if (attr) this._destroyAttribute(attr);
    }
    this._slotIndices = [];
    this._anchorIndexArray = null;
    if (this.fkAll) {
      this.fkAll.dispose();
      this.fkAll = null;
    }
    if (this.skinAll) {
      this.skinAll.dispose();
      this.skinAll = null;
    }
    this.uFkTotal = null;
    this.uSkinRowW = null;
    this.uSkinTotal = null;
    this.compiled = false;
    if (this.probe) {
      if (this.probe.rootBone.parent) this.probe.rootBone.parent.remove(this.probe.rootBone);
      this.probe.skeleton.dispose();
      this.probe = null;
    }
    this.ready = false;
    this.key = null;
  }

  /**
   * Whether a GPU slot is free (a freed slot or room below the cap).
   */
  hasFreeSlot() {
    return this.freeSlots.length > 0 || this.records.size < this.maxSlots;
  }

  /**
   * Biased index attribute for a slot, built once and shared by every agent
   * that ever occupies the slot (provider-owned: never destroyed on release).
   */
  _slotIndex(slot) {
    let attr = this._slotIndices[slot];
    if (!attr) {
      attr = new THREE.BufferAttribute(
        biasSlotIndex(this._anchorIndexArray, slot, this.dims.V), 1,
      );
      this._slotIndices[slot] = attr;
    }
    return attr;
  }

  /**
   * Allocates a GPU slot + mesh for one agent. The mesh renders its slot
   * slice of the shared mega outputs; no per-agent buffers or kernels.
   */
  spawnSlot({ color, wireframe, baseMaterial }) {
    if (!this.ready) throw new Error("GPU crowd is not ready.");
    let slot;
    if (this.freeSlots.length > 0) {
      slot = this.freeSlots.pop();
    } else {
      slot = this.records.size;
    }
    if (slot >= this.maxSlots) throw new Error("GPU crowd is full.");
    const geometry = new THREE.BufferGeometry();
    geometry.setIndex(this._slotIndex(slot));
    geometry.setAttribute("position", this.mega.posMegaAttr);
    geometry.setAttribute("normal", this.mega.nrmMegaAttr);
    // Pool-owned read-only uv (from this pool's own GLB, like its index
    // cache): keeps node materials that sample `uv` quiet without
    // duplicating the buffer or touching the anchor mesh.
    if (this.uvAttribute) geometry.setAttribute("uv", this.uvAttribute);
    const material = baseMaterial.clone();
    material.color.copy(color);
    material.wireframe = Boolean(wireframe);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.frustumCulled = false;
    const record = { slot, mesh, geometry, material };
    this.records.set(slot, record);
    return record;
  }

  _releaseRecord(record, freeSlot = true) {
    // Never dispose() the geometry: three's dispose listener rebuilds the
    // attribute cache from the geometry and destroys every attached GPU
    // buffer — all of ours are provider-owned and shared (mega
    // position/normal, cached slot index, anchor uv). Detaching first does
    // not help: the rebuild reads `.version` off the detached (undefined)
    // attributes and throws. So drop the shell for GC and dispose only the
    // per-agent material clone (shared programs are refcounted); GPU
    // buffers die explicitly in reset().
    record.material.dispose();
    if (freeSlot) {
      this.records.delete(record.slot);
      this.freeSlots.push(record.slot);
    }
  }

  releaseSlot(record) {
    if (record) this._releaseRecord(record, true);
  }

  warmedCount() {
    return this.compiled ? this.records.size : 0;
  }

  /**
   * Uploads per-agent thetaJ + shape weights and dispatches the two batched
   * kernels over every occupied slot. pairs: [{ record, agent }]. forceAll is
   * kept for API compat; the batch covers all pairs every tick by design.
   */
  update(pairs, forceAll = false) {
    if (!this.ready || pairs.length === 0) return;
    void forceAll;
    const { J, V, B } = this.dims;
    const t0 = performance.now();
    let maxSlot = -1;
    for (const { record, agent } of pairs) {
      const tOff = record.slot * J * 7;
      this.dynamic.thetaArray.set(agent.thetaJ, tOff);
      const wOff = record.slot * B;
      const weights = this.dynamic.weightArray;
      let shaped = 0;
      const idCount = agent.identityValues.length;
      for (let b = 0; b < B; b++) {
        const w = b < idCount ? agent.identityValues[b] : agent.expressionValues[b - idCount];
        weights[wOff + b] = w;
        if (w !== 0) shaped = 1;
      }
      this.dynamic.shapeArray[record.slot] = shaped;
      if (record.slot > maxSlot) maxSlot = record.slot;
    }
    this.dynamic.thetaAttr.needsUpdate = true;
    this.dynamic.weightAttr.needsUpdate = true;
    this.dynamic.shapeAttr.needsUpdate = true;
    const t1 = performance.now();
    // Exactly two dispatches per tick, sized to the highest occupied slot
    // (freed slots below it compute harmlessly; nothing reads them).
    // Explicit [x, y, 1] sizes keep dispatch independent of the backend's
    // 1D-count overflow spillover; the uTotal guards park edge threads.
    // Two submits (fk strictly before skin) cost O(1) CPU — this is what
    // replaced the 2N-per-agent dispatch loop (~64ms at 256 agents).
    const S = maxSlot + 1;
    const fkTotal = S * J;
    this.uFkTotal.value = fkTotal;
    this.renderer.compute(this.fkAll, [Math.ceil(fkTotal / 64), 1, 1]);
    const skinTotal = S * V;
    const skinWg = Math.ceil(skinTotal / 64);
    const nx = Math.min(skinWg, 65535);
    this.uSkinRowW.value = nx * 64;
    this.uSkinTotal.value = skinTotal;
    this.renderer.compute(this.skinAll, [nx, Math.ceil(skinWg / nx), 1]);
    this.compiled = true;
    this.lastFillMs = t1 - t0;
    this.lastQueueMs = performance.now() - t1;
  }

  /**
   * Poses the shared CPU probe skeleton from the active GPU agent so the
   * gizmo and joint spheres track it. The probe root is reparented under the
   * agent's group so matrixWorld (and hence jointPositions) is world-correct.
   */
  poseProbe(agent) {
    const probe = this.probe;
    if (!probe) return;
    if (probe.rootBone.parent !== agent.group) {
      agent.group.add(probe.rootBone);
    }
    probe.thetaJ.set(agent.thetaJ);
    probe.applyThetaJ();
    agent.jointPositions.set(probe.jointPositions);
  }
}

export { flattenInverses };
