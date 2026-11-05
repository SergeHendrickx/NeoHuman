# GPU Crowd: What Was Built, What's Next, Where It's Going

The autonomous-agents crowd sim now poses and skins its agents with WebGPU
compute (TSL) instead of per-agent CPU skeletons — on LOD1, with automatic
CPU fallback everywhere else. This doc covers the design, the exact CPU/GPU
split, every behavior change, the bugs found on the way, verification status
with honest caveats, next steps, and the vision.

Companion docs: [arch.md](arch.md) (whole-app architecture, §4.10 for the
actor system) and [gpu.md](gpu.md) (WebGPU inventory, §9 for the crowd
compute summary).

## 1. TL;DR

- Spawns on WebGPU become `GpuAgent`s: steering, waypoints, and gait phase
  run on the CPU per tick; FK (joint matrices) and skinning (blendshapes +
  LBS + normals) run as two batched TSL compute dispatches per LOD pool.
- One GPU pool per LOD (1–6; LOD 0 excluded — detail-free subdivision),
  each with its own GLB geometry, buffers, and kernels; agents auto-migrate
  between pools by camera distance (bands + hysteresis, budgeted), with
  per-agent LOD pins and an auto/manual toggle.
- Per-agent pose AND per-agent identity/expression shapes — the old
  shared-geometry limitation (all clones look identical) is gone for GPU
  agents.
- Agent count is uncapped. GPU slots are capped at 256 per pool;
  overflow sheds to coarser pools, then CPU. The Crowd Stats panel shows
  the live mix, pools, and pending migrations.
- The anchor (Main Character) is always CPU. GPU agents survive anchor LOD
  switches (pools are anchor-independent); CPU agents still clear.
- Manual pose edits on agents persist as a base layer with gait added on
  top (no more snapback).

## 2. Scope Decisions (Locked)

These were explicit calls, not accidents:

1. **~~LOD1 only~~ LIFTED: all 7 LODs on GPU.** Each pool feeds the skin
   kernel its real LOD table (identity only on LOD1). Anchor LOD is now
   just the default spawn band, not a GPU gate.
2. **Anchor stays CPU.** The Main Character keeps the full `rig.js` path (FK
   + correctives + blendshapes + LOD + normals) and remains the export truth.
   No correctives run on agents (they never had them on CPU either).
3. **No agent-count cap, but per-pool GPU slot caps.** Total agents:
   unlimited. GPU residency: 256 slots per pool, then coarser-pool
   overflow, then CPU. VRAM math is in §4.3.
4. **~~Stage C deferred~~ DONE.** Batched 2-dispatch-per-pool submit (see
   `gpu.md` §9); per-agent nodes and the warmup ramp are gone.

## 3. Architecture

### 3.1 Selection and fallback (`sim/AgentSimulation.js`)

Each spawn decides per agent: GPU iff the WebGPU renderer is active, the
model is loaded, **and** a ready pool on the overflow chain (band pool,
then coarser) has a free slot — otherwise the original CPU `Agent`. Mixed
populations are normal and expected (pool overflow, spawns during warmup).

Pools build lazily (`ensurePool`: GLB fetch + parse, buffers, scratch
warmup ~250–350ms once each). `prewarmGpu` warms the anchor pool first,
then prefetches the rest staggered (~1 per 2s); migration and spawn also
kick builds on demand and wait. `gpuStatus()` reports per-pool state
(missing/loading/ready/failed + used/slots + verts) plus the old CPU
reasons; reason strings display verbatim in the UI.

A compute throw mid-run freezes that pool only (`rec.failed`, agents hold
last pose); other pools and the CPU fallback are unaffected.

### 3.2 What runs where

CPU per tick, per agent (both modes):

- Boids/waypoint steering, physics integration, heading, ground clamp.
- Gait phase advance (speed-driven, per-agent cadence).
- `thetaJ` assembly: authored base pose + sparse gait entries (13 matrix
  entries via the sparse Tp lookup).

CPU per tick, whole crowd (GPU mode only):

- One upload of all slots' theta (J×7 floats each) + shape weights (B
  floats each) + shaped flags into the dynamic buffers, one `needsUpdate`
  each.
- Probe posing for the active GPU agent (see §3.5).
- Exactly 2 `renderer.compute()` calls for the whole population (batched
  FK, then batched skin) — O(1) CPU, the Stage C fix for the old 2N
  per-agent dispatch loop (~64 ms queue CPU at 256 agents).

GPU per tick, whole crowd (Stage C batched; slot = thread id / N):

- **`fkAll`** (S×J threads): reads each slot's theta, walks each joint's
  ancestor chain (up to 16 links), composes local TRS (ZYX Euler →
  quaternion × pre-rotation → 2^s-scaled matrix, cm→m translations), and
  writes row-major bone matrices (chain-world × bone-inverse) to the
  slot's slice of the mega bone buffer.
- **`skinAll`** (S×V threads): gathers neutral position/normal through the
  LOD table, adds the slot's weighted blendshape deltas (skipped entirely
  when the shape buffer reads 0), runs 4-bone LBS against the slot's bones,
  normalizes the analytic normal, and writes the slot's slice of the mega
  position/normal buffers — which are bound directly as the mesh's vertex
  attributes (zero-copy compute→raster handoff, exactly the v2 pattern).

Never on the GPU path: pose correctives (anchor-only feature),
`computeVertexNormals` (agents use analytic normals), LOD switching itself,
picking math, gizmo/proxy solving.

### 3.3 Buffers (`gpu/layouts.js`, `gpu/provider.js`)

All float32, one type per buffer (indices/parents stored as exact floats).
Sizes at LOD1 (J=127 joints, V=18,439 verts, B=117 shapes, 256 slots):

- **Static rig** (~823 KB): joint offsets | pre-rotations | parents |
  bone inverses | skin indices | skin weights | neutral normals.
- **Static body** (~26 MB, the big one): neutral positions + all 117
  blendshape deltas, w-component stripped (vec4→vec3).
- **Static LOD** (~442 KB): per-target-vertex source triplet + barycentric
  weights (identity transfer on LOD1).
- **Dynamic theta** (~910 KB): 256 slots × 889 floats, rewritten every tick.
- **Dynamic weights** (~120 KB): 256 slots × 117 floats, rewritten every tick.
- **Dynamic shape flags** (1 KB): 256 shaped/not-shaped words.
- **Mega bones** (~2 MB): 256 slots × 127 joints × 16 floats, FK-written.
- **Mega outputs** (~113 MB): 256 slots × 18,439 verts × 3 floats × 2
  (position + normal), skin-written, neutral-filled at build.
- **Per slot** (cached, provider-owned): one biased index copy (F×4 B)
  reading the slot's output slice; uv stays shared read-only with the
  anchor. Agent dispose detaches everything and frees only the geometry
  shell + material.

Per-tick logical upload per agent: (889 + 117) × 4 B + 4 B ≈ 4 KB.
Worst-case residency ≈ 145 MB + 256 biased index copies, dominated by the
mega outputs and the static blendshape table (~26 MB, paid once).

### 3.4 Render and lifetime

Each GPU agent is a plain `THREE.Mesh` (no `Skeleton`, no skinning) whose
position/normal attributes are the shared mega outputs, sliced by a cached
biased index copy, in a `THREE.Group` that carries placement/heading/scale
like any actor. Materials are clones of the anchor's with per-agent color.
Mega outputs initialize to the neutral pose so the first frame is never
garbage. `releaseSlot` frees the slot for reuse; spawn reuses freed slots
before growing.

### 3.5 Picking, gizmo, probe

GPU vertex positions are unreadable on the CPU, so:

- **Picking** is analytic: ray vs. vertical capsule per agent, no mesh
  raycast.
- **Gizmo/joint spheres** track a shared CPU probe skeleton: each tick the
  active GPU agent's thetaJ is composed on the CPU (`poseProbe`) into the
  probe, reparented under that agent's group so matrices are world-correct.
  Only the active agent pays this (one 127-joint FK, ~microseconds).

## 4. Changes Outside `static/js/gpu/` (Full List)

The `gpu/` folder holds only `layouts.js`, `kernels.js`, `provider.js`,
`GpuAgent.js`. Everything else it needed:

- `sim/GaitController.js`: split the old monolithic gait application into
  `gaitValues(phase, speed)` (pure) + `accumulateThetaJ(values, thetaJ,
  mapping, additive=false)` (sparse Tp scatter, shared by CPU and GPU).
- `actor.js`: `computeThetaJFromParams` (full matvec), `applyThetaJ`,
  `applyLocomotionGait` (CPU override point), `isGpuAgent` flag, and the
  authored-base-pose layer (`_baseThetaJ`, see §5.1).
- `sim/AgentSimulation.js`: GPU selection + CPU overflow, provider
  lifecycle, `_postSimUpdate` dispatch, tick instrumentation (agent/gpu ms,
  pair count), `gpuStatus()`, `timings`, capsule picking, selection ring.
- `sim/Agent.js`: unchanged behavior (still the CPU fallback), shares the
  refactored gait + layering.
- `crowd.js`: re-exports (`prewarm/refreshGpuAgents`, extended
  `getSimulationStats` with gpu status + timings).
- `app.js`: GPU prewarm/refresh hooks on boot/configure/slider/reset,
  CPU/GPU labels (totals, dropdown, inspector title, badge), uncapped
  spawn handler, `refreshCrowdStats`.
- `gizmo.js`: refresh GPU dispatch after non-anchor drags; spheres read the
  probe-backed `jointPositions`.
- `index.html`: Crowd Stats panel markup, live pipeline-bar IDs, camera +
  appearance cards open by default.
- `styles.css`: stats panel styles, pipeline segment rename (`.seg-skin` →
  `.seg-mesh`).
- `state.js`: element refs for the above.
- `perf.js` / `rig.js` / `scene.js`: the Client Compute Stage Breakdown bar
  is live (`recordStageTimes`) with a real mesh stage
  (normals/bounds/transfer) instead of the old hardcoded skinning segment.
- Tests: `tests/gpu/*.test.mjs` (Node: layouts, kernel construction,
  gait incl. additive/default paths) and two new pytest wiring checks
  (stats panel wiring, FK quaternion pin).
- Stage C follow-up: `layouts.js` (mega allocators + `biasSlotIndex`),
  `kernels.js` (`buildBatchedKernels`), `provider.js` (mega buffers, two
  dispatches, slot release without per-agent destroys), `app.js` dispatch
  row (2 batched submits + shape byte in the upload figure),
  `AgentSimulation.js` prewarm comment; tests updated to the batched
  builder + a slot-bias case + the dispatch-row pin.
- Multi-LOD pools + auto-LOD: `sim/LodPolicy.js` (bands, hysteresis,
  overflow; Node-tested), `sim/SpatialHash.js` (neighbor grid, feeds
  steering today, LOD/frustum tomorrow), `provider.js` (`maxSlots` per
  pool), `GpuAgent.js` (`lod`/`lodPin`/`migrateTo`), `AgentSimulation.js`
  (pool manager, budgeted migration, selective LOD-switch clear),
  `rig.js` (exported `loadRiggedModel`), `crowd.js` (auto toggle + pin
  re-exports, listener keeps GPU agents), `app.js`/`index.html`/`state.js`
  (toggle, pools row, LOD column + pin select, honest switch summaries).

## 5. Behavior Notes

### 5.1 No more snapback (pose layering)

Previously every tick refilled `thetaJ` from gait alone, so a manual slider
edit on a wandering agent rendered for one frame and snapped back. Now each
actor keeps `_baseThetaJ` (synced inside `computeThetaJFromParams`, which
every pose edit flows through); each tick copies the base and adds gait on
top. Edits persist as the walk's foundation. Untouched agents (zero base)
render bit-identically to before, and CPU agents get the same fix through
the shared path.

### 5.2 Per-agent shapes

GPU agents upload their own identity/expression weights per tick
(`uShape`-gated skip when all zero), so the identity slider visibly reshapes
only the targeted agent — verified live (40% of viewport pixels changed on a
maxed identity edit while paused). CPU agents keep the old limitation
(values stored, shared geometry unreshaped).

### 5.3 Uncapped count with GPU overflow

The 40-agent cap is gone everywhere (input, sim, handler). Spawns take GPU
slots while they last (freed slots are reused), then overflow to CPU
agents — same steering/gait, CPU skinning. The spawn console line and the
stats panel always report the split, e.g. `300 agents: 256 GPU + 44 CPU`.

### 5.4 Auto-LOD (bands, hysteresis, pins)

Distance bands (m): L1 <5, L2 <7.5, L3 <10, L4 <13, L5 <17, L6 beyond
(LOD 0 excluded system-wide). `migrateTarget` adds 10% hysteresis at edges
and jumps multi-band
directly (one mesh swap either way). Migration runs 8/tick nearest-first
(a camera swing invalidates hundreds of bands at once; id order would
stall the visible crowd behind far agents), overflowing toward the move
direction when the band pool is full (never wedges, never oscillates);
each move swaps the mesh and pays first-render init on its next frame
(measured: saturated drain at 600 agents holds render at 3-13ms with
occasional spikes, which is why the budget went 2→8). The selected agent
never migrates; per-agent pins (`lodPin`, inspector LOD select) override
bands; auto-off drifts everyone back to the anchor pool.
Spawns land in their band pool (or overflow/parking + later migration).

### 5.5 Crowd Stats panel (open by default)

`#crowd-stats-*`, refreshed on demand (Refresh button) or ~4Hz with the
auto-refresh checkbox (off by default — the panel is the dominant `ui`
cost at scale); the per-agent table additionally skips rebuilds unless
its sampling key changed:

- Headline: agent count, GPU/CPU split, tick ms.
- Engine: WebGPU/WebGL + LOD. Provider: ready dims or the exact fallback
  reason. Last tick: total ms as agents (steering+gait CPU) + post
  (probe+upload+dispatch-queue CPU), with a tooltip stating GPU execution
  itself is async. Dispatches: count + upload KB. GPU exec: true GPU
  execution ms/frame via timestamp queries (`n/a` without adapter
  support). The row shows compute + whole-scene render; both pools are
  drained periodically (compute from the sim tick, render from the render
  path) so neither overflows.
- LOD mix: live agents per LOD pool (`L1:60 …`) + CPU overflow.
- LOD moves: pending moves split into queued (waiting for budget) vs
  blocked (no free slot in range, or retry-waiting), moves last tick,
  total moves, failed moves (retry in 60 ticks). Diagnoses migration
  lag: draining `queued` is healthy; stuck `blocked` with full band
  pools means demand exceeds supply (e.g. 300 agents want L1's 256
  slots); growing `failed` means mesh swaps are throwing.
- Frustum: culling on/off + visible/culled split.
- Spawn: last `spawn()` breakdown (wall/construct/notify/raf-wait,
  batches); LOD mix + slowest batch in the tooltip.
- Per-agent table: id, mode, GPU slot, behavior, speed, gait phase,
  theta-sum (pose-diversity fingerprint), position.
- The `[perf]` console line carries the same migration/culling state in
  compact form: `culled=N moves=+thisTick/total/Ffails pending=N(Bblocked)`.

Nothing is sampled or estimated — every number is measured or counted live,
except GPU execution time, which WebGPU timestamp queries would be needed
for (see §8.3).

## 6. Bugs Found on the Way

### 6.1 The crouch: euler-`w` instead of premultiplied-`qw` (fixed)

Every GPU agent rendered as the same crouch with arms raised — not a walk
frame at all. Root cause in the FK kernel's quaternion→matrix conversion:
`q = pre * euler` computed `qx/qy/qz` correctly but never computed `qw`,
and the matrix's w-terms used the euler-only `ew`. With non-identity
pre-rotations on nearly every joint, all bone matrices were systematically
wrong (and non-rigid, which collapses the mesh toward the joints — the
crouch). One-line-class fix: compute
`qw = pw*ew − px*ex − py*ey − pz*ez` and use it in `wx/wy/wz`. Post-fix
agents stand and stride correctly. Pinned by
`test_fk_kernel_uses_full_premultiplied_quaternion` (static pin — TSL
bodies execute only on a GPU; behavior is covered by the browser probe).

### 6.2 The apparent sync (explained, no second bug)

With the distortion above, all agents looked locked to one stride frame on
LOD1 (LOD2/CPU always walked fine). The per-agent chain was audited end to
end and every link checked out: distinct gait phases, distinct thetaJ,
per-slot uploads, per-record `uAgent` values 0..N, per-record bone/output
buffers — plus a decisive experiment: editing agent #2's shape weights
changed only agent #2's mesh, which is impossible unless slots, uniforms,
and bindings are all per-agent correct. Verdict: the "sync" was the
systematic qw distortion dominating every pose, not lost per-agent data.
Post-fix screenshots show correct, diverse mid-stride frames.

### 6.3 Console `uv` warning (open, cosmetic)

`AttributeNode: Vertex attribute "uv" not found on geometry` appears once
per spawn batch. Ruled out: anchor geometry (GLB ships no uv, anchor
material is mapless), selection ring (has uv), skeleton points (point-coord
sampling), overlay visibility. The uv-sharing added to the provider is a
correct no-op (nothing to share). It does not affect rendering; the exact
emitting node was never isolated. See §8.6.

## 7. Verification Status (Read the Caveats)

Proven:

- `pytest` 53 passed (incl. 2 new wiring/pin checks), `node --test`
  13 passed (layouts, kernel construction, gait incl. additive paths).
- Browser probes (headless Chromium + SwiftShader WebGPU, real backend
  bytes served from disk): 6×GPU spawn with motion, dropdown select +
  badge/inspector labels, pause-freeze, focus, paused per-agent shape edit,
  resume, LOD2→CPU fallback + LOD1→GPU rebuild, flocking strides,
  multi-frame stats responsiveness, panel screenshot read-back.
- Pose layering: Node harness over the REAL rig mapping bytes
  (889×204×127) through `setParamValue` → `computeThetaJFromParams` →
  `evaluatePose` → `applyLocomotionGait` → `accumulateThetaJ` →
  `evaluateAndApply` on `Actor`, plus the `GpuAgent` override: zero-base
  identity exact, base+gait composition within 1e-5, base untouched by
  ticks, 6-frame stability (peak 0.863, animating), single-param edit
  equals the exact Tp column.

Caveats — what is NOT proven:

- All timing numbers (tick 0.2–1.0 ms, queue costs) were measured under
  SwiftShader CPU emulation, not a real GPU. Treat them as smoke signals,
  not benchmarks.
- The >256 CPU-overflow path has never been exercised (no 257-agent run).
- Stride diversity + the qw fix are verified headless only; confirmation on
  the local real-GPU run is still pending.
- No timestamp queries: true GPU execution time is unmeasured by design.

## 8. Next Steps (Ordered)

1. **Real-GPU confirmation (you, 5 minutes).** Refresh the local run, spawn
   6–30 agents on LOD1, check: upright diverse strides (not crouch/sync),
   stats panel values sane, no console errors. This closes the qw/sync
   saga for good.
2. **Exercise the overflow.** Spawn 300 once: expect 256 GPU + 44 CPU, no
   errors, stats mix correct. If the CPU tail tanks the frame, consider a
   soft warning in the stats panel (not a cap).
3. **True GPU timings.** DONE in code, awaiting real-GPU confirmation:
   `trackTimestamp: true` on the renderer (zero-cost no-op without adapter
   support), periodic `resolveTimestampsAsync("compute")` (one in flight
   max, per-frame average), surfaced as the GPU exec row. Confirm on the
   local run that it reports sane ms/frame instead of `n/a`.
4. **Dispatch hygiene.** DONE (Stage C merged kernels): the two batched
   kernels compile once (scratch dispatch at prewarm primes the Dawn
   program cache), every tick costs exactly 2 dispatches / 2 submits no
   matter the population, and the old per-agent warmup ramp is deleted.
   This fixed ~64 ms queue CPU at 256 agents. Still open, only if queue
   CPU ever dominates again: partial-range uploads (today the full
   theta/weight buffers re-upload every tick), explicit `computeAsync`
   await ordering, and a validation-layer stress pass.
5. **Dawn-in-Node kernel tests.** Replace the static qw pin with real
   execute-and-readback tests (`node-webgpu` + Darwin/Windows Dawn):
   upload thetas → dispatch FK → compare bones vs CPU `applyThetaJ`
   within tolerance. The layouts tests already run in Node; this extends
   the same harness to the math that actually matters.
6. **Resolve the `uv` warning.** Capture the emitting material with a
   `customProgramCacheKey`/build hook or a debug-group sweep; either fix
   the geometry or delete the dead uv sampling. Cosmetic, but it's noise
   in an otherwise clean console.
7. **Perf at scale.** Re-measure tick CPU vs agent count on the real
   machine now that dispatch is O(1) (steering is O(n²) via neighbor
   checks — the likely first bottleneck past ~100 agents, for BOTH modes)
   and record the curve in `gpu.md`.
8. **Parked (migration tuning follow-ups).** Nearest-first + budget 8 +
   queued/blocked split are implemented, tested (node 24/24, pytest 55),
   and awaiting browser confirmation: swing the camera over a 600 crowd,
   near agents should snap LODs in ~1–2s with `queued` draining ~8/frame.
   If render spikes grow with 8 moves/tick, dial the budget back or
   stagger the per-move first-render init. Still open after that:
   first-render init hunt (`renderer.info.programs.length` across a move
   — confirm shader/program compile), slot compaction so culled agents
   skip compute and not just draws, pooled neighbor arrays (GC churn at
   scale), CPU agents onto pooled geometries.

Explicitly NOT next: any anchor-on-GPU work. (The old LOD1-only park —
real LOD tables for other topologies — is done: every pool feeds its own
table. Remaining LOD-adjacent work: frustum visibility gating off the
spatial hash, and killing the ~65ms first-render mesh init that spawn and
migration both pay.)

## 9. Vision

Keep the split the hardware wants: **decisions on the CPU, deformation on
the GPU.** Steering, behavior, gait phase, and authored-pose layering stay
cheap scalar/vector CPU work (plus the one upload per tick); every
per-joint and per-vertex flop — FK chains, blendshapes, LBS, normals —
lives in compute, where 18k verts × 117 shapes is a rounding error.

Concretely, the end state I see:

- 100+ agents at a flat, boring frame budget: CPU tick ~1–2 ms
  (dominated by steering, fixable with spatial hashing), GPU deform async
  and unblocking, rasterization the only real cost.
- The stats panel as the permanent contract: any perf question answered in
  the UI (per-agent mode/slot/phase, tick split, dispatch count, upload
  volume, and — after §8.3 — true GPU ms), never by re-instrumenting.
- The anchor path converging only where it pays: the anchor keeps its
  correctives on CPU (the released model is a global MLP — no clean
  per-vertex port), but the day the anchor needs crowds of *itself*, the
  agent kernels already speak its exact math (same Tp, same ZYX, same
  exponential scale, same row-major bones).
- Tests that execute, not just pin: Dawn-backed kernel tests so the next
  qw-class bug fails in CI instead of in a screenshot.

The through-line: no magic numbers, no hidden state, no "trust me" —
everything measured, everything attributed, everything visible.

## 10. Known Issues Left Standing (Pre-Existing, Untouched)

Carried over from the architecture analysis; none introduced by the GPU
work, none touched by it:

- FK exists three times (`rig.js`, `Actor`, GPU kernel) and can drift.
- The global param vector aliases the last-displayed state; crowd
  randomize can disagree with panel/mesh.
- Dual selection state (`selectedAgentId` vs `activeActorId`) is
  convention-only.
- `nogeometry` syncs still run full TorchScript inference; reset POSTs
  download discarded bodies.
- `renderTime` has two writers; weight cache is unversioned; several perf
  badges/marketing rows are static copy.

