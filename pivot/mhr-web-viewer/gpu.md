# WebGPU in the MHR Web Viewer

## TL;DR

WebGPU is used as a **drop-in rasterizer backend only**. At startup the app
tries `WebGPURenderer` and falls back to `WebGLRenderer`; after that, one
shared, renderer-agnostic code path drives the whole app. There are no compute
shaders, no TSL, no storage buffers, and no GPU-side deformation — FK,
blendshapes, pose correctives, LOD transfer, and normal recomputation all run
on the CPU in TypedArrays. The only deformation work the GPU does is the
automatic hardware skinning in three.js's vertex shader, which works
identically under both renderers.

## 1. Complete inventory: every WebGPU touchpoint

The full list fits in a table — anything not listed here does not know or care
which renderer is active:

| Location | What |
| --- | --- |
| `static/index.html` (import map) | Maps `three/webgpu` → `/vendor/three/build/three.webgpu.js` and `three/tsl` → `three.tsl.js` |
| `static/js/scene.js:2` | Sole code import: `import { WebGPURenderer } from "three/webgpu"` |
| `static/js/scene.js` (`initRenderer`) | Tries WebGPU when `navigator.gpu` exists, `await gpuRenderer.init()`, otherwise WebGL; records `state.isWebGPU` |
| `static/js/state.js` | Declares the `isWebGPU: false` flag |
| `static/app.js` (`initialize`) | Only consumer of the flag: picks the "Ready (WebGPU Rendering…)" vs "(Client-side…)" status string |
| `static/index.html` (perf card copy) | Static marketing text: "Client TypedArray (WebGPU renderer)" and a pipeline segment titled "Three.js WebGL/WebGPU Render" |
| `static/js/perf.js` | Defensive `renderer.info` reads (`calls ?? drawCalls`) so draw-call stats work under either backend |

Notably, `three/tsl` is mapped in the import map but **never imported** — a
dead entry (harmless: nothing is downloaded until imported).

## 2. Initialization

`app.js` top-level flow awaits `initRenderer()` before wiring events or
loading the model, so the renderer always exists first:

```text
navigator.gpu present?
  yes -> new WebGPURenderer({canvas, antialias, alpha:false,
                             preserveDrawingBuffer, powerPreference})
         await init()
         success -> WebGPU active (state.isWebGPU = true)
         throw   -> fall through to WebGL (console.warn)
  no  -> new THREE.WebGLRenderer(same options)
shared: pixelRatio = min(devicePixelRatio, 2), SRGBColorSpace,
        ACESFilmicToneMapping @ 1.05, clear color from the UI picker
```

Both constructors receive **identical parameters**; all post-creation
configuration is shared. There is no adapter request, no device/limits
inspection, and no per-backend tuning anywhere — capability detection begins
and ends with `navigator.gpu`.

## 3. What actually runs on the GPU

Identical under WebGPU and WebGL — this is all stock three.js behavior, not
app code:

- **Rasterization + lighting.** One `MeshStandardMaterial` scene (hemisphere +
  key + rim lights), tone mapping, sRGB output.
- **Hardware skinning.** The persistent `SkinnedMesh`/`Skeleton` pair means
  per-frame posing uploads only ~127 bone matrices; vertex blending runs in
  the vertex shader. This is the app's one real GPU deformation stage, and it
  is free in the sense that no app shader code exists for it.
- **Helpers.** Skeleton lines/points, joint-sphere `InstancedMesh`,
  grid/axes, gizmo helper — all regular scene-graph objects.

Everything the MHR math produces reaches the GPU as plain `BufferAttribute`
uploads (`position.needsUpdate`, bone matrices, instance matrices). The app
never touches GPU buffers directly.

## 4. What stays on the CPU (and why "WebGPU" changes nothing about it)

All of `static/js/rig.js` is CPU TypedArray code that runs per parameter edit,
regardless of renderer:

1. `θⱼ = T_p · θ_p` dense multiply → bone FK composition (offset + ZYX Euler
   × pre-rotation quaternion, `s = 2^d`).
2. Pose-corrective MLP: 6D rotation features → sparse Layer-0 → ReLU →
   CSR Layer-2 SpMV (~4.5M non-zeros) → per-vertex deltas.
3. Blendshape accumulation over active weights only (`|w| > 1e-6`).
4. Barycentric LOD transfer for LOD ≠ 1.
5. Full-mesh `computeVertexNormals()` + bounding box/sphere.

Step 5 is the dominant main-thread cost per edit and is unaffected by the
renderer choice. The `v2/v2.md` design (TSL compute shader evaluating steps
2–4 into a `StorageBufferAttribute` bound directly as the `position`
attribute) was never implemented — see §6.

## 5. Renderer-agnostic by construction

Everything downstream of `initRenderer()` is backend-blind:

- **Scene graph**: meshes, skeleton, lights, controls (`OrbitControls`,
  `TransformControls`), raycasting (gizmo spheres, agent picking), and the
  agent simulation's per-tick bone writes use only three.js object APIs.
- **Render loop**: `requestRender()` coalescing, on-demand rAF, idle timer,
  `resizeRenderer()` (`setPixelRatio`/`setSize`), and `renderNow()` for
  screenshots behave the same; `preserveDrawingBuffer: true` (kept for
  `canvas.toDataURL` screenshots) applies to both.
- **Telemetry**: `renderer.info.render.calls ?? drawCalls` and
  `renderer.info.memory.geometries` tolerate both backends' info shapes. The
  "VRAM" figure is a CPU-side estimate (summed typed-array byte lengths), not
  a GPU memory query — there is no adapter-level memory API use.
- **On-demand idling** ("0% idle GPU spin") is a CPU-loop property (no rAF
  when nothing changes), not a GPU feature; it holds under both renderers.
- **The actor system adds no GPU interaction.** Per-actor posing is just more
  independent skeletons through the same hardware-skinning path, per-actor
  colors are cloned materials, and selection is scene-graph raycasting plus a
  DOM badge. No new renderer calls, no per-actor GPU resources beyond the
  meshes the crowd/sim systems already created.

## 6. What WebGPU is *not* used for (proposal vs reality)

`v2/v2.md` §B/§2 sketches a TSL compute pipeline. None of it exists in the app
*for the anchor* (crowd agents are the exception — see §9):

| Proposal | Reality |
| --- | --- |
| `three/tsl` (`Fn`, `storage`, `instanceIndex`, `Loop`) | Never imported |
| `renderer.compute(computeDeformNode)` per edit | No `compute()` call anywhere |
| Neutral/blendshape/corrective weights in GPU storage buffers | Weights live in JS heap (`Float32Array`/`Uint16Array`); only the composed `position` attribute is uploaded |
| Zero-copy compute→vertex handoff via `StorageBufferAttribute` as `position` | `position` is a regular `BufferAttribute`, rewritten from the CPU each edit |
| GPU-side normals | `computeVertexNormals()` on the main thread |
| "< 0.2 ms" GPU eval claim | Client eval is CPU-timed and displayed as such |

The perf card's "Client TypedArray (WebGPU renderer)" row is therefore precise
if read carefully: TypedArrays on the client, WebGPU only as the rasterizer.
The "96% faster / 10× FPS" badges and V1-vs-V2 table remain static copy, but
the pipeline breakdown bar itself is now live: `rig.js`/`scene.js` time each
stage (`recordStageTimes` in `perf.js`) and the bar widths, legend values, and
tooltips — including the previously missing normals/bounds/transfer stage —
update from real measurements.

## 7. Notes, costs, and dead flags

- **Static import cost.** `scene.js` imports `three/webgpu` statically, so
  browsers download the WebGPU backend bundle even when they end up on WebGL.
  A dynamic `import()` inside the `navigator.gpu` branch would make the
  fallback path lighter.
- **Shadow flags are dead.** `castShadow`/`receiveShadow = true` is set on
  the main mesh and agents — but `renderer.shadowMap` is never enabled, so
  no shadow maps render under either backend. (The "Agent shadows" toggle
  was deleted as a placebo; making shadows real needs `shadowMap.enabled`,
  a shadow camera on the key light, and a shadow-receiving ground plane —
  none exists, only the grid helper.) The flags cost nothing, but they also
  do nothing.
- **`preserveDrawingBuffer: true` on both renderers** exists solely for the
  PNG screenshot path (`renderNow()` + `toDataURL`). It forfeits the
  buffer-swap optimization on every frame for a feature used on click.
- **No GPU telemetry exists.** Draw calls, triangle counts, and the VRAM/RAM
  estimates all derive from `renderer.info` + CPU-side array sizes. Adapter
  info, timestamp queries, and real memory accounting are unused.
- **Fallback is silent-ish.** A failed `WebGPURenderer.init()` logs a console
  warning and continues on WebGL; the only user-visible difference is the
  ready-status string. There is no UI indicator of the active backend beyond
  that one line (and the console log).

## 8. If you wanted real WebGPU compute

Measured per-edit profile on this machine (vendored three r180,
Node 24/V8, real face indices from `data/lod_topology.npz`, 100 iterations,
medians) — `computeVertexNormals` + bounds per displayed LOD:

| LOD | verts | faces | normals | bbox | bsphere | all three |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 73,639 | 146,938 | 5.41 ms | 0.34 ms | 0.71 ms | 6.90 ms |
| 1 | 18,439 | 36,776 | 1.48 ms | 0.09 ms | 0.15 ms | 1.73 ms |
| 2 | 10,661 | 21,250 | 0.84 ms | 0.04 ms | 0.07 ms | 0.96 ms |
| 3 | 4,899 | 9,750 | 0.39 ms | 0.06 ms | 0.03 ms | 0.48 ms |
| 4 | 2,461 | 4,878 | 0.19 ms | 0.03 ms | 0.02 ms | 0.24 ms |
| 5–6 | ≤971 | ≤1,906 | ~0.08 ms | ~0.01 ms | ~0.01 ms | ~0.09 ms |

Same-process equivalents of the other stages at LOD1 (identical op counts /
loop shapes, synthetic data): FK ~0.11 ms, blendshapes ~0.08 ms (1 active
weight) / ~1.9 ms (10) / ~4.3 ms (117), CSR correctives ~3.9 ms, LOD1→LOD0
transfer ~0.41 ms. Key structural fact: correctives and blendshapes always
run at LOD-1 resolution, while `computeVertexNormals` runs at the *displayed*
LOD — so at LOD0 it is the single largest stage (~7 ms of a 16.6 ms frame
budget on every scrub event), second-largest at LOD1 (~1.7 ms), negligible
at LOD ≥ 3. Cost scales ~linearly with face count (~37 ns/face).

A compute port would follow `v2/v2.md` but must be re-derived for the
*verified* model: the released correctives are a single global
SparseLinear(750→3000) → ReLU → CSR Linear(55317×3000) over 6D features of
joints 2..126 — there are no per-joint MLPs or `P_j`/mask tables, so the
doc's §§1–2 sketches don't apply as written. The highest-value first step
would be moving `computeVertexNormals()` off the main thread (compute shader
or worker): at LOD0–1 it is the largest or second-largest per-edit cost while
contributing nothing to rig fidelity.

## 9. GPU crowd compute (LOD pools)

The one place TSL compute actually runs: crowd agents (`static/js/gpu/`,
managed in `sim/AgentSimulation.js`), one pool per LOD (1–6; LOD 0
excluded system-wide as a detail-free subdivision).

- **Selection.** A spawn becomes `GpuAgent` iff WebGPU renderer + model
  loaded + a ready pool on the overflow chain (band pool, then coarser)
  has a free slot; otherwise it is the original CPU `Agent`. The anchor is
  always CPU. Each pool loads its own GLB and feeds the skin kernel its
  real LOD table (identity only on LOD1); pools are anchor-independent,
  so GPU agents survive anchor LOD switches.
- **Buffers (per pool).** Static rig/body/LOD tables plus per-tick theta
  (J×7 floats), shape weights, and a shaped flag per agent slot, uploaded
  once per tick. Bone matrices and position/normal outputs live in shared
  mega buffers (one slice per slot); each agent's mesh renders its slice
  via a cached biased index copy, with uv from the pool's own GLB. No
  per-agent GPU buffers or uniforms remain.
- **Kernels** (`kernels.js`, Stage C batched): `fkAll` (S×J threads) walks
  each joint's parent chain — ZYX Euler → quaternion × pre-rotation →
  scaled matrix → × bone inverse — and `skinAll` (S×V threads) gathers
  source verts, adds per-slot blendshapes (gated by the shape buffer), and
  runs LBS with analytic normals. Slot comes from the thread id
  (`slot = t/N`, guarded by `uTotal` uniforms); math conventions mirror
  `rig.js`/`Actor` (row-major bones = chain-world × inverse).
- **Dispatch.** Per pool with agents, `provider.update(pairs)` uploads
  every slot, then issues exactly TWO `renderer.compute()` calls per tick —
  one batched FK, one batched skin — with explicit `[x, y, 1]` sizes (skin
  spills to 2D past 65,535 workgroups) and strict fk-before-skin submit
  ordering. Per-frame CPU is O(1) in agent count: this replaced the
  2N-per-agent dispatch loop (~64 ms queue CPU at 256 agents). A throw
  freezes that pool only (`rec.failed`) instead of killing the sim loop,
  and a handful of dispatches can't overflow the timestamp pool.
- **Quaternion fix (Nov 2026).** The rotation matrix once used the
  euler-only `ew` instead of the premultiplied `qw`, collapsing every agent
  into the same crouch. Pinned by
  `test_fk_kernel_uses_full_premultiplied_quaternion` (static pin — TSL
  bodies execute only on a GPU; behavior is covered by the browser probe).
- **Observability.** The Crowd Stats panel (`#crowd-stats-*`) shows engine,
  auto/manual + pending migrations, per-pool state (used/slots/verts),
  last-tick CPU split, multi-pool dispatch count with upload volume, and a
  per-agent table (mode/slot/LOD/behavior/speed/phase/theta-sum/pos).

