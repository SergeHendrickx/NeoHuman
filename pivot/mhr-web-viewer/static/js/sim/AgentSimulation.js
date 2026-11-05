import * as THREE from "three";
import { Agent } from "./Agent.js";
import { SpatialHash } from "./SpatialHash.js";
import {
  CPU_DEBUG_COLOR,
  LOD_DEBUG_COLORS,
  LOD_MIGRATION_BUDGET,
  LOD_POOL_SLOTS,
  POOL_LODS,
  bandForDistance,
  migrateTarget,
  overflowLods,
  sortNearestFirst,
} from "./LodPolicy.js";
import { actorManager, syncDisplayedSkeleton } from "../actor.js";
import { activeCamera, bodyMaterial, controls, renderer, requestRender, scene, skeletonOverlay } from "../scene.js";
import { state } from "../state.js";
import { GpuCrowd, computeNeutralNormals, flattenInverses } from "../gpu/provider.js";
import { GpuAgent } from "../gpu/GpuAgent.js";
import { loadRiggedModel, parseLodMapping } from "../rig.js";
import { fetchBinaryAsset } from "../cache.js";
import { readStageTimes } from "../perf.js";
import { crowdEyes } from "../crowdEyes.js";
import { crowdMouth } from "../crowdMouth.js";

export const AGENT_LIMITS = { maxAgents: Infinity };

// Reused frustum-culling scratch (no per-agent allocation in the hot loop).
const _cullFrustum = new THREE.Frustum();
const _cullMatrix = new THREE.Matrix4();
const _cullSphere = new THREE.Sphere();

const _pickSeg0 = new THREE.Vector3();
const _pickSeg1 = new THREE.Vector3();
const _pickOnRay = new THREE.Vector3();
const _pickOnSegment = new THREE.Vector3();

/**
 * Analytic ray-vs-vertical-capsule test for GPU agents (their mesh positions
 * live on the GPU, so mesh raycasting would read stale buffers). Returns the
 * along-ray distance of a hit, or null.
 */
export function rayCapsuleHit(ray, position, height, radius) {
  _pickSeg0.set(position.x, 0, position.z);
  _pickSeg1.set(position.x, height, position.z);
  const distSq = ray.distanceSqToSegment(_pickSeg0, _pickSeg1, _pickOnRay, _pickOnSegment);
  if (distSq > radius * radius) return null;
  return _pickOnRay.distanceTo(ray.origin);
}

export class AgentSimulation {
  constructor() {
    this.agents = [];
    this.isRunning = false;
    this.behaviorMode = "wander";
    this.timeScale = 1.0;
    this.frameIndex = 0;
    this.simTime = 0.0;
    this.selectedAgentId = 0; // 0 = anchor actor #0, >0 = Agent ID
    this.bounds = { minX: -5.5, maxX: 5.5, minZ: -5.5, maxZ: 5.5 };
    this.centerObstacle = { position: new THREE.Vector3(0, 0, 0), radius: 1.5 };
    this._rafId = null;
    this._lastTime = 0;
    this._listeners = new Set();
    this._onTick = this._tick.bind(this);
    this._raycaster = new THREE.Raycaster();
    // LOD pools: lod -> { gpu, status, error, failed }. Pools own their GLB
    // geometry + buffers and are anchor-independent; see ensurePool.
    this.pools = new Map();
    this._poolPromises = new Map();
    this._poolGeomCache = new Map();
    this._poolNeutralNrmCache = null;
    this._poolEpoch = 0;
    this.poolTopologyVersion = null;
    this.autoLod = true;
    this.frustumCulling = true;
    this.showLodColors = false;
    this._lodColorCache = new Map();
    this._prefetchQueue = [];
    this._pendingMigrations = 0;
    this._blockedMigrations = 0;
    this._migratedThisTick = 0;
    this._movedTotal = 0;
    this._migrateFailsTotal = 0;
    this._culledCount = 0;
    this._lastSpawn = null;
    this._agentMs = 0;
    this._gpuMs = 0;
    this._probeMs = 0;
    this._fillMs = 0;
    this._queueMs = 0;
    this._uiMs = 0;
    this._lastPairCount = 0;
    this._gpuExecMs = null;
    this._gpuExecResolveFrame = 0;
    this._gpuTimePending = false;
    this._lastPerfLogFrame = -1000;
    this._lastDeltaMs = 0;
    // Cell 3.5m = largest boid radius (alignment/cohesion); rebuilt per tick.
    this.spatialHash = new SpatialHash(3.5);

    this._initSelectionRing();
  }

  /**
   * Rebuilds the neighbor grid and returns the per-tick environment shared
   * by every agent update (live tick and paused stepping alike).
   */
  _tickEnvironment() {
    this.spatialHash.build(this.agents);
    const grid = this.spatialHash;
    return {
      bounds: this.bounds,
      centerObstacle: this.centerObstacle,
      neighborsOf: (agent) => grid.query(agent.position.x, agent.position.z),
    };
  }

  /**
   * Whether the model data needed by every pool is present (LOD-agnostic:
   * pools load their own GLB geometry and never touch the anchor mesh).
   */
  _modelReadyForPools() {
    if (!state.isWebGPU || !renderer) return false;
    const md = state.metadata;
    if (!md || !state.rigMapping || !state.blendshapes || !state.skinnedMesh) return false;
    return true;
  }

  /**
   * Parsed GLB bundle for a LOD pool, cached per topology. The meshes stay
   * detached (never added to the scene); pools keep geometry + skeleton.
   */
  async _poolGeometry(lod) {
    const topo = state.metadata?.model?.topologyVersion ?? 0;
    const cached = this._poolGeomCache.get(lod);
    if (cached && cached.topo === topo) return cached.geom;
    const { skinnedMesh, rootBone } = await loadRiggedModel(lod);
    const geometry = skinnedMesh.geometry;
    const skinIdx = geometry.getAttribute("skinIndex");
    const skinW = geometry.getAttribute("skinWeight");
    const indexAttr = geometry.index;
    if (!skinIdx || !skinW || !indexAttr) {
      throw new Error(`LOD ${lod} GLB lacks skinning/index attributes.`);
    }
    // Per-pool bone map: GLB bone order is exporter accident, names are
    // the contract (same mapping rig.js builds for the anchor).
    const bonesByName = new Map(skinnedMesh.skeleton.bones.map((bone) => [bone.name, bone]));
    const boneIndexMap = state.metadata.jointNames.map((name) => {
      const bone = bonesByName.get(name);
      if (!bone) throw new Error(`LOD ${lod} model is missing joint "${name}".`);
      return skinnedMesh.skeleton.bones.indexOf(bone);
    });
    const geom = {
      geometry,
      skeleton: skinnedMesh.skeleton,
      rootBone,
      boneIndexMap,
      indexAttr,
      skinIdx,
      skinW,
      uvAttr: geometry.getAttribute("uv") || null,
      vertCount: geometry.getAttribute("position").count,
    };
    this._poolGeomCache.set(lod, { topo, geom });
    return geom;
  }

  /**
   * Source-to-target transfer for a LOD pool (null identity on LOD 1).
   * Shares the anchor path's parsed-mapping cache.
   */
  async _poolMapping(lod) {
    if (lod === 1) return null;
    if (!state.lodMappings[lod]) {
      const buffer = await fetchBinaryAsset(
        `/api/lod-mapping?lod=${lod}`,
        `Could not load LOD ${lod} mapping.`,
      );
      const mapping = parseLodMapping(buffer);
      if (!mapping) throw new Error(`LOD ${lod} mapping is empty.`);
      state.lodMappings[lod] = mapping;
    }
    return state.lodMappings[lod];
  }

  /**
   * Source-resolution neutral normals, computed once from the LOD 1 faces
   * and shared by every pool (reduced LODs gather through it; the kernel
   * never needs target-res neutral normals).
   */
  async _poolNeutralNrm() {
    const topo = state.metadata?.model?.topologyVersion ?? 0;
    if (this._poolNeutralNrmCache && this._poolNeutralNrmCache.topo === topo) {
      return this._poolNeutralNrmCache.nrm;
    }
    const bs = state.blendshapes;
    const V1 = bs.neutralVertices.length / 4;
    const lod1 = await this._poolGeometry(1);
    const nrm = computeNeutralNormals(bs.neutralVertices, V1, lod1.indexAttr.array);
    this._poolNeutralNrmCache = { topo, nrm };
    return nrm;
  }

  /**
   * Anchor LOD mapped onto the pool set (LOD 0 is excluded system-wide:
   * detail-free subdivision, so it shares LOD 1's pool).
   */
  _anchorPoolLod() {
    const lod = state.metadata?.model?.lod ?? 1;
    return lod === 0 ? 1 : lod;
  }

  /**
   * Builds (or returns) the GPU pool for a LOD. Resolves to the GpuCrowd
   * or null when the pool failed; concurrent callers share one build.
   * A scratch dispatch warms pipelines + the mega first-upload so the
   * first real agent skips Dawn compilation (~250-350ms once per pool).
   */
  ensurePool(lod) {
    if (!POOL_LODS.includes(lod)) return Promise.resolve(null);
    const rec = this.pools.get(lod);
    if (rec && rec.status !== "loading") return Promise.resolve(rec.gpu || null);
    if (this._poolPromises.has(lod)) return this._poolPromises.get(lod);
    const p = this._buildPool(lod);
    this._poolPromises.set(lod, p);
    const forget = () => {
      if (this._poolPromises.get(lod) === p) this._poolPromises.delete(lod);
    };
    p.then(forget, forget);
    return p;
  }

  async _buildPool(lod) {
    const epoch = this._poolEpoch;
    const rec = { gpu: null, status: "loading", error: null, failed: false };
    this.pools.set(lod, rec);
    this._notifyChange();
    try {
      if (!this._modelReadyForPools()) throw new Error("model still loading");
      const topo = state.metadata.model?.topologyVersion ?? 0;
      const geom = await this._poolGeometry(lod);
      const mapping = await this._poolMapping(lod);
      if (lod !== 1 && !mapping) throw new Error(`LOD ${lod} mapping unavailable`);
      if ((state.metadata.model?.topologyVersion ?? 0) !== topo) {
        throw new Error("topology changed mid-build");
      }
      const bs = state.blendshapes;
      const rig = state.rigMapping;
      const source = {
        lod,
        jointCount: rig.jointCount,
        vertCount: geom.vertCount,
        srcVertCount: bs.neutralVertices.length / 4,
        shapeCount: bs.idCount + bs.feCount,
        topologyVersion: topo,
        jointOffsets: rig.jointOffsets,
        jointPreRots: rig.jointPreRotations,
        jointParents: state.metadata.jointParents,
        inverses: flattenInverses(geom.skeleton.boneInverses, rig.jointCount),
        skinIdx: geom.skinIdx.array,
        skinW: geom.skinW.array,
        neutralNrm: await this._poolNeutralNrm(),
        neutral4: bs.neutralVertices,
        deltas4: bs.blendshapeDeltas,
        lodMapping: mapping,
        indexAttribute: geom.indexAttr,
        uvAttribute: geom.uvAttr,
        rootBone: geom.rootBone,
        skeleton: geom.skeleton,
        boneIndexMap: geom.boneIndexMap,
        rigMapping: rig,
        poseCount: rig.tpCols,
      };
      const gpu = await GpuCrowd.create(renderer, source, LOD_POOL_SLOTS[lod]);
      const scratch = gpu.spawnSlot({
        color: new THREE.Color(0x000000),
        wireframe: false,
        baseMaterial: state.skinnedMesh.material,
      });
      gpu.update([{
        record: scratch,
        agent: {
          thetaJ: new Float32Array(gpu.dims.J * 7),
          identityValues: new Float32Array(bs.idCount),
          expressionValues: new Float32Array(bs.feCount),
        },
      }], true);
      gpu.releaseSlot(scratch);
      if (epoch !== this._poolEpoch) {
        gpu.reset();
        return null;
      }
      rec.gpu = gpu;
      rec.status = "ready";
      if (this.poolTopologyVersion === null) this.poolTopologyVersion = topo;
      this._notifyChange();
      return gpu;
    } catch (error) {
      if (epoch !== this._poolEpoch) return null;
      rec.status = "failed";
      rec.error = error?.message || String(error);
      console.warn(`[gpu] LOD ${lod} pool unavailable:`, rec.error);
      this._notifyChange();
      return null;
    }
  }

  /**
   * Fire-and-forget GPU warmup: anchor pool first (spawn-ready fast), the
   * rest staggered in the background. Safe to call any time; never throws.
   */
  prewarmGpu() {
    if (!this._modelReadyForPools()) return;
    const anchorLod = this._anchorPoolLod();
    void this.ensurePool(anchorLod).then((gpu) => {
      if (!gpu) return;
      for (const l of POOL_LODS) {
        if (l !== anchorLod && !this.pools.has(l) && !this._prefetchQueue.includes(l)) {
          this._prefetchQueue.push(l);
        }
      }
    });
  }

  /**
   * Picks a ready pool for a spawn at `distance` from the camera: the band
   * pool (auto) or anchor pool (manual), overflowing coarser, else the
   * nearest ready pool as a parking spot (migration fixes it up), else
   * null for the CPU fallback. Kicks the preferred build when missing.
   */
  _poolForSpawn(distance) {
    const anchorLod = this._anchorPoolLod();
    const preferred = this.autoLod ? bandForDistance(distance) : anchorLod;
    for (const lod of overflowLods(preferred)) {
      const rec = this.pools.get(lod);
      if (rec && rec.status === "ready" && !rec.failed && rec.gpu.hasFreeSlot()) {
        return { pool: rec.gpu, lod };
      }
    }
    void this.ensurePool(preferred);
    let fallback = null;
    for (const [lod, rec] of this.pools) {
      if (rec.status === "ready" && !rec.failed && rec.gpu.hasFreeSlot()) {
        if (!fallback || Math.abs(lod - preferred) < Math.abs(fallback.lod - preferred)) {
          fallback = { pool: rec.gpu, lod };
        }
      }
    }
    return fallback;
  }

  /**
   * Display-only GPU/CPU status snapshot for the Crowd Stats panel.
   * dims comes from any ready pool (J/B are identical everywhere; V is
   * that pool's) so the upload figure stays valid across pools.
   */
  gpuStatus() {
    const pools = [];
    for (const l of POOL_LODS) {
      const rec = this.pools.get(l);
      pools.push({
        lod: l,
        status: rec ? rec.status : "missing",
        used: rec && rec.gpu ? rec.gpu.records.size : 0,
        slots: LOD_POOL_SLOTS[l],
        error: rec ? rec.error : null,
        failed: Boolean(rec && rec.failed),
        verts: rec && rec.gpu ? rec.gpu.dims.V : null,
      });
    }
    const extra = {
      pools,
      autoLod: this.autoLod,
      pendingMigrations: this._pendingMigrations,
      blockedMigrations: this._blockedMigrations,
      migratedThisTick: this._migratedThisTick,
      movedTotal: this._movedTotal,
      migrateFails: this._migrateFailsTotal,
      frustumEnabled: this.frustumCulling,
      culled: this._culledCount,
      lastSpawn: this._lastSpawn,
    };
    const ready = [...this.pools.values()].find((rec) => rec.status === "ready" && !rec.failed);
    if (ready) {
      return { mode: "gpu", key: ready.gpu.key, dims: ready.gpu.dims, ...extra };
    }
    if ([...this.pools.values()].some((rec) => rec.failed)) {
      return { mode: "cpu", reason: "compute dispatch failed — GPU agents frozen", ...extra };
    }
    const failed = pools.find((p) => p.status === "failed");
    if (failed) {
      return { mode: "cpu", reason: `LOD ${failed.lod} pool failed: ${failed.error}`, ...extra };
    }
    if (!state.isWebGPU || !renderer) {
      return { mode: "cpu", reason: "WebGL renderer (WebGPU off)", ...extra };
    }
    if (!this._modelReadyForPools()) {
      return { mode: "cpu", reason: "model still loading", ...extra };
    }
    if ([...this.pools.values()].some((rec) => rec.status === "loading")) {
      return { mode: "cpu", reason: "warming up GPU…", ...extra };
    }
    return { mode: "cpu", reason: "GPU not warmed up yet", ...extra };
  }

  /**
   * Display-only last-tick timings (ms) for the Crowd Stats panel.
   * gpuExecMs is true GPU execution time per frame (null until the first
   * timestamp resolve lands, or when the backend lacks support).
   */
  get timings() {
    return {
      agentMs: this._agentMs,
      gpuMs: this._gpuMs,
      probeMs: this._probeMs,
      fillMs: this._fillMs,
      queueMs: this._queueMs,
      uiMs: this._uiMs,
      pairCount: this._lastPairCount,
      gpuExecMs: this._gpuExecMs,
    };
  }

  /**
   * Resolves one timestamp batch for the compute queue (true GPU execution
   * time). At most one resolve in flight; the pool coalesces anyway. No-op
   * without timestamp-query support or dispatches.
   */
  _maybeResolveGpuTime() {
    if (this._lastPairCount === 0 || this._gpuTimePending) return;
    const backend = renderer?.backend;
    if (!backend || backend.trackTimestamp !== true) return;
    this._gpuTimePending = true;
    const fromFrame = this._gpuExecResolveFrame;
    renderer.resolveTimestampsAsync("compute").then(
      (ms) => {
        this._gpuTimePending = false;
        if (typeof ms !== "number" || !Number.isFinite(ms)) return;
        const frames = Math.max(1, this.frameIndex - fromFrame);
        this._gpuExecMs = ms / frames;
        this._gpuExecResolveFrame = this.frameIndex;
      },
      () => {
        this._gpuTimePending = false;
      },
    );
  }

  get count() {
    return this.agents.length;
  }

  /**
   * Initializes visual ground indicator ring for the selected actor.
   */
  _initSelectionRing() {
    const ringGeom = new THREE.RingGeometry(0.38, 0.46, 32);
    ringGeom.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x818cf8,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
    });
    this.selectionRing = new THREE.Mesh(ringGeom, ringMat);
    this.selectionRing.renderOrder = 3;
    this._ringVisible = true;
    scene.add(this.selectionRing);
    // Derive initial visibility from the actual selection (id 0 = anchor).
    this._updateSelectionRing();
  }

  /**
   * Updates position and visibility of the ground selection ring.
   */
  setSelectionRingVisible(on) {
    this._ringVisible = on !== false;
    this._updateSelectionRing();
  }

  _updateSelectionRing() {
    if (!this.selectionRing) return;
    if (this._ringVisible === false) {
      this.selectionRing.visible = false;
      return;
    }
    if (this.selectedAgentId === 0) {
      this.selectionRing.position.set(0, 0.02, 0);
      this.selectionRing.visible = true;
    } else {
      const agent = this.getSelectedAgent();
      if (agent) {
        this.selectionRing.position.set(agent.position.x, 0.02, agent.position.z);
        this.selectionRing.visible = true;
      } else {
        this.selectionRing.visible = false;
      }
    }
  }

  /**
   * Subscribes a listener to simulation events.
   */
  addChangeListener(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  _notifyChange() {
    for (const fn of this._listeners) fn();
  }

  /**
   * Spawns independent autonomous agents into the scene, a few per frame so
   * the browser stays responsive (agents pop in progressively with live
   * totals). A newer spawn/clear supersedes an in-flight one via the token.
   */
  async spawn(count) {
    if (!state.skinnedMesh || !state.rootBone || !state.skeleton || !state.rigMapping) {
      throw new Error("Model is still loading — try again in a moment.");
    }

    const token = (this._spawnToken = (this._spawnToken || 0) + 1);
    const available = AGENT_LIMITS.maxAgents - this.agents.length;
    const n = Math.max(0, Math.min(count, available));
    if (n === 0) {
      return { added: 0, total: this.agents.length, capped: count > 0, gpuAdded: 0, cpuAdded: 0, ms: 0 };
    }

    const wantGpu = this._modelReadyForPools();
    if (wantGpu) this.prewarmGpu();
    const t0 = performance.now();
    const BATCH = 4;
    let added = 0;
    let gpuAdded = 0;
    const lodCounts = new Map();
    let batches = 0;
    let constructMs = 0;
    let slowestBatchMs = 0;
    let notifyMs = 0;
    let rafWaitMs = 0;

    for (let done = 0; done < n; done += BATCH) {
      if (token !== this._spawnToken) break;
      const end = Math.min(n, done + BATCH);
      const b0 = performance.now();
      for (let i = done; i < end; i++) {
        const id = this.agents.length + 1;

        // Spawn distributed in a ring around center
        const angle = (this.agents.length * 1.9 + Math.random() * 0.4);
        const dist = 2.4 + Math.random() * 2.2;
        const initialPos = new THREE.Vector3(
          Math.cos(angle) * dist,
          0,
          Math.sin(angle) * dist,
        );

        // GPU while pool slots last, then CPU overflow: total agent
        // count is uncapped. Placement re-resolves per agent (a pool may
        // land mid-spawn); migration fixes up parking spots later.
        const placement = wantGpu
          ? this._poolForSpawn(initialPos.distanceTo(activeCamera.position))
          : null;
        const agent = placement
          ? new GpuAgent(
              id,
              placement.pool,
              placement.lod,
              state.skinnedMesh,
              state.rigMapping,
              this._poolGeomCache.get(placement.lod)?.geom.boneIndexMap || state.boneIndexMap,
              {
                initialPosition: initialPos,
                behavior: this.behaviorMode,
                idCount: state.blendshapes.idCount,
                exprCount: state.blendshapes.feCount,
              },
            )
          : new Agent(
              id,
              state.skinnedMesh,
              state.rootBone,
              state.skeleton,
              state.rigMapping,
              state.boneIndexMap,
              {
                initialPosition: initialPos,
                behavior: this.behaviorMode,
              },
            );
        if (placement) {
          gpuAdded++;
          lodCounts.set(placement.lod, (lodCounts.get(placement.lod) || 0) + 1);
        }

        scene.add(agent.group);
        this.agents.push(agent);
        actorManager.registerActor(agent);
        if (this.showLodColors && agent.material) {
          this._lodDisplayColor(agent, agent.material.color);
        }
        added++;
      }
      const batchMs = performance.now() - b0;
      constructMs += batchMs;
      if (batchMs > slowestBatchMs) slowestBatchMs = batchMs;
      batches++;
      const n0 = performance.now();
      if (!this.isRunning) {
        this.start();
      } else {
        this._updateSelectionRing();
        this._notifyChange();
      }
      requestRender();
      notifyMs += performance.now() - n0;
      if (end < n && token === this._spawnToken) {
        const r0 = performance.now();
        await new Promise((resolve) => requestAnimationFrame(resolve));
        rafWaitMs += performance.now() - r0;
      }
    }

    const ms = performance.now() - t0;
    const mix = [...lodCounts.entries()].sort((a, b) => a[0] - b[0])
      .map(([lod, n]) => `L${lod}:${n}`).join(" ");
    // Persisted for the Crowd Stats Spawn row (same numbers as the log).
    this._lastSpawn = {
      added, gpu: gpuAdded, cpu: added - gpuAdded, mix,
      wallMs: ms, constructMs, notifyMs, rafWaitMs, batches, slowestBatchMs,
    };
    console.info(
      `[gpu] spawning ${added} agent(s): ${gpuAdded} GPU${mix ? ` (${mix})` : ""} + ` +
      `${added - gpuAdded} CPU, ` +
      `${ms.toFixed(0)}ms wall / ${constructMs.toFixed(0)}ms construct / ` +
      `${notifyMs.toFixed(0)}ms notify / ${rafWaitMs.toFixed(0)}ms raf-wait / ` +
      `${batches} batches, slowest batch ${slowestBatchMs.toFixed(1)}ms.`,
    );

    // Spawn never steals the selection: the camera only follows explicit
    // selection changes (click, dropdown, removal fallback).
    if (!this.isRunning && this.agents.length > 0) {
      this.start();
    }

    requestRender();
    return { added, total: this.agents.length, capped: count > n, gpuAdded, cpuAdded: added - gpuAdded, ms };
  }

  /**
   * Cleans up and removes all simulated agents.
   */
  /**
   * Drops eye instances of removed agents when no tick will do it for us:
   * removal while paused would leave a stale InstancedMesh (floating
   * eyeballs). (When running, the tick covers it.)
   */
  _syncEyesAfterRemoval() {
    if (!this.isRunning) {
      crowdEyes.update(this.agents, this.simTime, performance.now());
      crowdMouth.update(this.agents);
    }
  }

  clear() {
    this._spawnToken = (this._spawnToken || 0) + 1;
    for (let i = 0; i < this.agents.length; i++) {
      this.agents[i].dispose();
    }
    this.agents.length = 0;
    actorManager.clearNonAnchorActors();
    // Time keeps flowing with zero agents; selectAgent(0) notifies, and the
    // running tick re-shows the ring on actor #0 (the honest selection).
    this.selectAgent(0);
    this._syncEyesAfterRemoval();
    requestRender();
  }

  /**
   * Disposes CPU agents only; GPU agents survive (their pools own every
   * buffer they render). Returns the removed count.
   */
  clearCpuAgents() {
    let removed = 0;
    for (let i = this.agents.length - 1; i >= 0; i--) {
      const agent = this.agents[i];
      if (agent.isGpuAgent) continue;
      agent.dispose();
      actorManager.unregisterActor(agent.id);
      this.agents.splice(i, 1);
      removed++;
    }
    if (this.selectedAgentId !== 0 && !this.agents.some((a) => a.id === this.selectedAgentId)) {
      this.selectAgent(0);
    } else {
      this._notifyChange();
    }
    this._syncEyesAfterRemoval();
    if (removed > 0) requestRender();
    return removed;
  }

  /**
   * Drops every LOD pool (GPU buffers destroyed): only valid when no GPU
   * agent holds a slot — callers clear them first.
   */
  resetPools() {
    this._poolEpoch++;
    for (const rec of this.pools.values()) {
      if (rec.gpu) rec.gpu.reset();
    }
    this.pools.clear();
    this._poolPromises.clear();
    this._poolGeomCache.clear();
    this._poolNeutralNrmCache = null;
    this.poolTopologyVersion = null;
    this._prefetchQueue = [];
    this._pendingMigrations = 0;
    this._blockedMigrations = 0;
  }

  /**
   * Anchor LOD switches replace the shared mesh. Clones/instances always go
   * (crowd.js); here CPU agents go too (they share anchor geometry), while
   * GPU agents survive — unless the topology itself changed (seal cut),
   * which invalidates every pool buffer and drops everything.
   */
  handleLodSwitch() {
    const topo = state.metadata?.model?.topologyVersion ?? 0;
    if (this.poolTopologyVersion !== null && topo !== this.poolTopologyVersion) {
      this.clear();
      this.resetPools();
      this.poolTopologyVersion = topo;
      return "full";
    }
    this.poolTopologyVersion = topo;
    return this.clearCpuAgents() > 0 ? "partial" : "kept";
  }

  /**
   * Sets the global behavior mode for all agents.
   */
  setBehaviorMode(mode) {
    this.behaviorMode = mode;
    for (let i = 0; i < this.agents.length; i++) {
      this.agents[i].behavior = mode;
    }
    this._notifyChange();
  }

  /**
   * Selects an actor by ID (0 = anchor actor #0, 1+ = Simulated Agent).
   */
  selectAgent(id) {
    this.selectedAgentId = Number(id);
    this._updateSelectionRing();
    const selected = this.getSelectedAgent();
    if (selected?.isGpuAgent && !this.isRunning && selected._record && selected._pool?.ready) {
      selected._pool.poseProbe(selected);
    }
    actorManager.setActiveActor(this.selectedAgentId);
    requestRender();
    this._notifyChange();
  }

  /**
   * Keeps the skeleton overlay + joint spheres on a moving selected agent.
   * Spheres ride along via syncDisplayedSkeleton's registered callback.
   */
  _trackSelectionOverlays() {
    if (!skeletonOverlay.visible) return;
    const active = actorManager.activeActor;
    if (!active || active.isAnchor) return;
    syncDisplayedSkeleton();
  }

  /**
   * Retrieves the currently selected Agent, or null if anchor actor #0 is selected.
   */
  getSelectedAgent() {
    if (this.selectedAgentId === 0) return null;
    return this.agents.find((a) => a.id === this.selectedAgentId) || null;
  }

  /**
   * Removes a specific agent by ID.
   */
  removeAgent(id) {
    const index = this.agents.findIndex((a) => a.id === id);
    if (index !== -1) {
      this.agents[index].dispose();
      this.agents.splice(index, 1);
      actorManager.unregisterActor(id);
      if (this.selectedAgentId === id) {
        this.selectAgent(0);
      } else {
        this._notifyChange();
      }
      this._syncEyesAfterRemoval();
      requestRender();
    }
  }

  /**
   * Smoothly aligns OrbitControls target onto the currently selected actor.
   */
  focusSelectedAgent() {
    if (!controls) return;
    if (this.selectedAgentId === 0) {
      controls.target.set(0, 1.0, 0);
      controls.update();
      requestRender();
      return;
    }
    const agent = this.getSelectedAgent();
    if (agent) {
      controls.target.set(agent.position.x, 1.0 * agent.scale, agent.position.z);
      controls.update();
      requestRender();
    }
  }

  /**
   * Performs raycasting against agent meshes and main character to pick actors in 3D.
   */
  pickAgentAtNdc(ndcPoint, camera) {
    this._raycaster.setFromCamera(ndcPoint, camera);
    let gpuBest = null;
    let gpuBestDist = Infinity;
    for (const agent of this.agents) {
      if (!agent.isGpuAgent) continue;
      const t = rayCapsuleHit(
        this._raycaster.ray, agent.position, 1.9 * agent.scale, 0.5 * agent.scale,
      );
      if (t !== null && t < gpuBestDist) {
        gpuBest = agent;
        gpuBestDist = t;
      }
    }
    const meshes = [];
    for (const agent of this.agents) {
      if (!agent.isGpuAgent) meshes.push(agent.mesh);
    }
    const hits = this._raycaster.intersectObjects(meshes, false);
    if (hits.length > 0 && (gpuBest === null || hits[0].distance < gpuBestDist)) {
      const hitAgent = this.agents.find((a) => a.mesh === hits[0].object);
      if (hitAgent) {
        this.selectAgent(hitAgent.id);
        return hitAgent;
      }
    }
    if (gpuBest) {
      this.selectAgent(gpuBest.id);
      return gpuBest;
    }
    if (state.skinnedMesh) {
      const mainHits = this._raycaster.intersectObject(state.skinnedMesh, false);
      if (mainHits.length > 0) {
        this.selectAgent(0);
        return null;
      }
    }
    return undefined;
  }

  /**
   * Sets the global simulation time scale (e.g. 0.25x for slow motion, 2.0x for fast forward).
   */
  setTimeScale(scale) {
    this.timeScale = Math.max(0.1, Math.min(4.0, Number(scale) || 1.0));
    this._notifyChange();
  }

  /**
   * Steps simulation forward by an exact number of discrete frames (useful when paused).
   */
  step(frames = 1, dt = 1 / 60) {
    for (let f = 0; f < frames; f++) {
      const environment = this._tickEnvironment();
      const effectiveDt = dt * this.timeScale;
      for (let i = 0; i < this.agents.length; i++) {
        this.agents[i].update(effectiveDt, environment, this.agents);
      }
      this._migrateLodAgents();
      if (this.showLodColors) this._applyLodColors();
      this._updateFrustumCulling();
      this.frameIndex += 1;
      this.simTime += effectiveDt;
    }
    crowdEyes.update(this.agents, this.simTime, performance.now());
    crowdMouth.update(this.agents);
    this._postSimUpdate();
    this._updateSelectionRing();
    this._trackSelectionOverlays();
    requestRender();
    this._notifyChange();
  }

  /**
   * Auto-LOD migration: moves GPU agents whose camera distance left their
   * band (hysteresis in LodPolicy), budgeted per tick — each move swaps
   * the mesh and pays first-render init on its next frame. Nearest agents
   * go first (a camera swing invalidates hundreds of bands at once; id
   * order would stall the visible crowd behind far agents). The selected
   * agent never migrates (stable while inspecting); pins override bands.
   * Never demotes GPU agents to CPU; full band pools overflow toward the
   * move direction (spawn rules) instead of wedging the queue.
   */
  _migrateLodAgents() {
    this._migratedThisTick = 0;
    let budget = LOD_MIGRATION_BUDGET;
    const camPos = activeCamera.position;
    const anchorLod = this._anchorPoolLod();
    const cands = [];
    for (const agent of this.agents) {
      if (!agent.isGpuAgent || !agent._record || !agent._pool?.ready) continue;
      if (agent.id === this.selectedAgentId) continue;
      const dist = agent.position.distanceTo(camPos);
      const desired = agent.lodPin ?? (this.autoLod ? migrateTarget(agent.lod, dist) : anchorLod);
      if (desired === agent.lod) continue;
      cands.push({ agent, desired, dist });
    }
    sortNearestFirst(cands);
    let pending = cands.length;
    let blocked = 0;
    for (const { agent, desired } of cands) {
      if (budget <= 0) break;
      if (this.frameIndex < (agent._migrateRetryAt || 0)) {
        blocked++;
        continue;
      }
      // Overflow like spawn (a full band pool must not wedge the queue),
      // but only toward the move direction: finer moves take finer pools,
      // coarser moves take coarser ones. Progress is monotonic, so moves
      // converge on the band instead of oscillating.
      const movingFiner = desired < agent.lod;
      let target = null;
      for (const lod of overflowLods(desired)) {
        if (lod === agent.lod) continue;
        if (movingFiner ? lod > agent.lod : lod < agent.lod) continue;
        const rec = this.pools.get(lod);
        if (rec && rec.status === "ready" && !rec.failed && rec.gpu.hasFreeSlot()) {
          target = { pool: rec.gpu, lod };
          break;
        }
      }
      if (!target) {
        blocked++;
        void this.ensurePool(desired);
        continue;
      }
      try {
        agent.migrateTo(target.pool, target.lod);
        budget--;
        pending--;
        this._migratedThisTick++;
        this._movedTotal++;
      } catch (error) {
        agent._migrateRetryAt = this.frameIndex + 60;
        this._migrateFailsTotal++;
        blocked++;
        console.warn(`[gpu] LOD migration failed (agent ${agent.id}):`, error?.message || error);
      }
    }
    this._pendingMigrations = pending;
    this._blockedMigrations = blocked;
  }

  /**
   * Debug view: paint every agent by LOD pool (CPU gray). Runs every tick
   * so spawns and mesh-swapping migrations can never show a stale color;
   * per-agent colors restore from agent.color when toggled off.
   */
  setShowLodColors(enabled) {
    this.showLodColors = Boolean(enabled);
    this._applyLodColors();
    this._notifyChange();
    requestRender();
  }

  _lodDisplayColor(agent, target) {
    if (!this.showLodColors) return target.copy(agent.color);
    const hex = agent.isGpuAgent ? (LOD_DEBUG_COLORS[agent.lod] ?? 0xffffff) : CPU_DEBUG_COLOR;
    let cached = this._lodColorCache.get(hex);
    if (!cached) {
      cached = new THREE.Color(hex);
      this._lodColorCache.set(hex, cached);
    }
    return target.copy(cached);
  }

  _applyLodColors() {
    for (const agent of this.agents) {
      if (agent.material) this._lodDisplayColor(agent, agent.material.color);
    }
    // Actor #0 is not a sim agent but paints along at the server LOD (0 shares 1).
    const anchor = actorManager.getActor(0);
    if (anchor && bodyMaterial) {
      if (!this.showLodColors) {
        bodyMaterial.color.copy(anchor.color);
      } else {
        const lod = state.metadata?.model?.lod || 1;
        const hex = LOD_DEBUG_COLORS[lod] ?? 0xffffff;
        let cached = this._lodColorCache.get(hex);
        if (!cached) {
          cached = new THREE.Color(hex);
          this._lodColorCache.set(hex, cached);
        }
        bodyMaterial.color.copy(cached);
      }
    }
  }

  /**
   * Manual frustum culling (three's built-in can't run: agent geometries
   * share mega buffers, so per-agent bounds don't exist). Skips raster +
   * shadow draws for off-screen agents; steering/gait/compute still run
   * for everyone (slot compaction would be needed to skip those too).
   */
  setFrustumCulling(enabled) {
    this.frustumCulling = Boolean(enabled);
    this._updateFrustumCulling();
    requestRender();
  }

  _updateFrustumCulling() {
    if (!this.frustumCulling) {
      for (const agent of this.agents) {
        if (agent.mesh) agent.mesh.visible = true;
      }
      this._culledCount = 0;
      return;
    }
    _cullMatrix.multiplyMatrices(activeCamera.projectionMatrix, activeCamera.matrixWorldInverse);
    _cullFrustum.setFromProjectionMatrix(_cullMatrix);
    let culled = 0;
    for (const agent of this.agents) {
      if (!agent.mesh) continue;
      const s = agent.scale || 1;
      _cullSphere.center.set(agent.position.x, agent.position.y + 1.2 * s, agent.position.z);
      _cullSphere.radius = 2.0 * s;
      const visible = _cullFrustum.intersectsSphere(_cullSphere);
      agent.mesh.visible = visible;
      if (!visible) culled++;
    }
    this._culledCount = culled;
  }

  /**
   * Poses the probe from the active GPU agent (gizmo tracking) and dispatches
   * FK + skin compute for every pool with agents. Fill/queue timings sum
   * across pools (honest CPU total); a throw freezes that pool only.
   */
  _postSimUpdate(forceAll = false) {
    this._gpuMs = 0;
    this._probeMs = 0;
    this._fillMs = 0;
    this._queueMs = 0;
    this._lastPairCount = 0;
    const pairsByPool = new Map();
    for (const agent of this.agents) {
      if (agent.isGpuAgent && agent._record && agent._pool?.ready) {
        let entry = pairsByPool.get(agent._pool);
        if (!entry) {
          entry = { gpu: agent._pool, pairs: [] };
          pairsByPool.set(agent._pool, entry);
        }
        entry.pairs.push({ record: agent._record, agent });
      }
    }
    if (pairsByPool.size === 0) return;
    const t0 = performance.now();
    const active = actorManager.activeActor;
    if (active && active.isGpuAgent && active._record && active._pool?.ready) {
      active._pool.poseProbe(active);
    }
    this._probeMs = performance.now() - t0;
    let fillMs = 0;
    let queueMs = 0;
    let pairs = 0;
    for (const entry of pairsByPool.values()) {
      const rec = this.pools.get(entry.gpu.lod);
      if (rec && rec.failed) continue;
      try {
        entry.gpu.update(entry.pairs, forceAll);
        fillMs += entry.gpu.lastFillMs;
        queueMs += entry.gpu.lastQueueMs;
        pairs += entry.pairs.length;
      } catch (error) {
        // Freeze this pool at its last good pose rather than killing the
        // sim loop; other pools and the CPU fallback are unaffected.
        if (rec) rec.failed = true;
        console.error(`[gpu] LOD ${entry.gpu.lod} dispatch failed, freezing pool:`, error?.message || error);
      }
    }
    this._fillMs = fillMs;
    this._queueMs = queueMs;
    this._lastPairCount = pairs;
    this._gpuMs = performance.now() - t0;
    this._maybeResolveGpuTime();
  }

  /**
   * Re-runs GPU posing outside the tick loop (paused manual edits of the
   * active GPU agent). Callers still own requestRender().
   */
  refreshGpuAgents() {
    this._postSimUpdate(true);
  }

  /**
   * Starts the simulation update loop.
   */
  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this._lastTime = performance.now();
    this._rafId = requestAnimationFrame(this._onTick);
    this._notifyChange();
  }

  /**
   * Stops/pauses the simulation update loop.
   */
  stop() {
    if (!this.isRunning) return;
    this.isRunning = false;
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
    this._notifyChange();
  }

  /**
   * Toggles between play and pause.
   */
  toggle() {
    if (this.isRunning) {
      this.stop();
    } else {
      this.start();
    }
  }

  /**
   * Copy-paste diagnostics: one line every ~2s, any tick over 50ms, and any
   * frame gap over 50ms (rAF stall with cheap ticks — max one per 30
   * frames each). Filter the console for "[perf]".
   */
  _maybeLogPerf() {
    const tickTotal = this._agentMs + this._gpuMs + this._uiMs;
    // The loop runs from boot, so the idle scene stays quiet: the periodic
    // line only fires with agents around. Over-budget/hitch lines still log.
    const periodic = this.frameIndex % 120 === 0 && this.agents.length > 0;
    const overBudget = tickTotal > 50 && this.frameIndex - this._lastPerfLogFrame >= 30;
    const hitched = this._lastDeltaMs > 50 && this.frameIndex - this._lastPerfLogFrame >= 30;
    if (!periodic && !overBudget && !hitched) return;
    this._lastPerfLogFrame = this.frameIndex;
    const gpu = this.agents.reduce((n, a) => n + (a.isGpuAgent ? 1 : 0), 0);
    let warmed = 0;
    for (const rec of this.pools.values()) {
      if (rec.status === "ready" && rec.gpu) warmed += rec.gpu.warmedCount();
    }
    const render = readStageTimes().render;
    console.info(
      `[perf] frame=${this.frameIndex} agents=${this.agents.length}(${gpu}G+${this.agents.length - gpu}C) ` +
      `tick=${tickTotal.toFixed(1)}[agents=${this._agentMs.toFixed(1)},post=${this._gpuMs.toFixed(1)}` +
      `(probe=${this._probeMs.toFixed(1)},fill=${this._fillMs.toFixed(1)},queue=${this._queueMs.toFixed(1)}),` +
      `ui=${this._uiMs.toFixed(1)}]ms dt=${this._lastDeltaMs.toFixed(0)}ms render=${render.toFixed(1)}ms ` +
      `gpuExec=${this._gpuExecMs === null ? "n/a" : `${this._gpuExecMs.toFixed(2)}ms`} ` +
      `warmed=${warmed}/${this._lastPairCount} ` +
      `culled=${this._culledCount} moves=+${this._migratedThisTick}/${this._movedTotal}/F${this._migrateFailsTotal} pending=${this._pendingMigrations}(B${this._blockedMigrations})`,
    );
  }

  /**
   * Simulation frame tick.
   */
  _tick(currentTime) {
    if (!this.isRunning) return;

    const deltaMs = currentTime - this._lastTime;
    this._lastTime = currentTime;
    this._lastDeltaMs = deltaMs;

    // Clamp delta to prevent huge leaps on tab defocus
    const rawDelta = Math.min(Math.max(deltaMs / 1000.0, 0.001), 0.05);
    const effectiveDt = rawDelta * this.timeScale;

    const environment = this._tickEnvironment();

    // Update all autonomous agents
    const tickStart = performance.now();
    for (let i = 0; i < this.agents.length; i++) {
      this.agents[i].update(effectiveDt, environment, this.agents);
    }
    this._migrateLodAgents();
    if (this.showLodColors) this._applyLodColors();
    this._updateFrustumCulling();
    this._agentMs = performance.now() - tickStart;

    this.frameIndex += 1;
    this.simTime += effectiveDt;

    // Background pool prefetch, staggered (one pool per ~2s).
    if (this._prefetchQueue.length > 0 && this.frameIndex % 120 === 0) {
      const lod = this._prefetchQueue.shift();
      if (!this.pools.has(lod)) void this.ensurePool(lod);
    }

    // Crowd eyes before the GPU upload so this tick's blinks ride along.
    crowdEyes.update(this.agents, this.simTime, currentTime);
    crowdMouth.update(this.agents);
    this._postSimUpdate();
    this._updateSelectionRing();
    this._trackSelectionOverlays();
    requestRender();
    const ui0 = performance.now();
    this._notifyChange();
    this._uiMs = performance.now() - ui0;
    this._maybeLogPerf();
    this._rafId = requestAnimationFrame(this._onTick);
  }
}

// Global simulation singleton instance
export const simulation = new AgentSimulation();
