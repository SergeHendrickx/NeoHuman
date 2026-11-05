# MHR Web Viewer — Architecture

Describes the app as built today: a local Flask + Three.js viewer for the
Momentum Human Rig (MHR). Rendering and interaction run almost entirely in the
browser; the Python backend owns the TorchScript model, serves binary rig data,
and keeps an authoritative snapshot for mesh export.

## 1. Big picture

```text
Browser (no build step; ES modules via import map)
  static/index.html + static/styles.css
  static/app.js                    <- boot + all UI wiring
  static/js/*.js                   <- scene, rig, params, gizmo, crowd, sim, perf
  static/js/sim/*.js               <- autonomous-agent simulation
                 |  JSON + packed little-endian binary over HTTP
                 v
Flask server (server.py)           <- localhost only, single global character
  MHREngine                        <- TorchScript inference + snapshots
  exporters.py                     <- OBJ / GLB / USD writers
                 |  torch.jit.load (CPU)
                 v
MHR assets (parent checkout; read-only)
  assets/mhr_model.pt              <- TorchScript model (LOD 1 native)
  assets/lod{0..6}.fbx             <- control-point topology sources
  tools/mhr_LOD_conversion/        <- official barycentric LOD mappings
data/lod_topology.npz (committed)  <- triangulated faces per LOD
```

Key facts:

- **No frontend build.** The page loads `three@0.180.0` from `node_modules`
  through a `/vendor/three` route and an import map. `npm install` is the only
  frontend setup.
- **Single-character server.** `MHREngine` holds one global pose/identity/
  expression state behind an `RLock`. Concurrent browsers would share (and
  fight over) that state.
- **Multi-actor client.** The browser unifies the main character (anchor
  Actor #0) and spawned agents under one `Actor` model with per-character
  parameter vectors — but only the anchor syncs to the backend (§4.10).
- **Dual evaluation.** Both sides can compute the posed character (see §5).
  The viewport shows the *client-side* result; the server snapshot exists for
  export, ground snapping, and LOD asset generation.
- **Units.** Model space is centimeters; the backend converts to meters before
  anything crosses the wire (`/100.0`). Up axis is Y.

## 2. Repository layout

| Path | Role |
| --- | --- |
| `server.py` | Flask app + `MHREngine` (inference, state, API) |
| `exporters.py` | OBJ / GLB / USD mesh writers (also reused for live GLB serving) |
| `static/index.html` | Single-page markup: controls, viewport, perf + crowd cards |
| `static/styles.css` | All styling (~29 KB, one file) |
| `static/app.js` | Boot sequence + every event listener (~990 lines) |
| `static/js/` | Feature modules (table in §4) |
| `static/js/sim/` | Agent simulation (`AgentSimulation`, `Agent`, `GaitController`, `Steering`) |
| `tests/test_server.py` | Backend API/inference/export tests (needs real `mhr_model.pt`) |
| `tests/test_client_parity.py` | Client-math vs PyTorch ground-truth parity tests |
| `tests/test_frontend_wiring.py` | Static markup ↔ `state.js` ID contract + crowd options |
| `tools/build_lod_topology.py` | Generates `data/lod_topology.npz` from FBX control points |
| `tools/remove_sealed_polys.py` | Opens sealed eye/mouth strips across LODs |
| `tools/inspect_blender_rig.py` | Blender-side GLB/USD rig inspection helper |
| `v2/v2.md` | V1 analysis + client-side-evaluation proposal (design doc; partly historical) |
| `v2/mhr_eyes_oral_cavity_module/` | Standalone eyes/oral-cavity prototype (not wired into the app) |

## 3. Backend

### 3.1 `MHREngine` (`server.py`)

Owns all model state:

- `pose_values` (204), `identity_values` (45), `expression_values` (72),
  plus limits, joint names/parents, faces, and per-LOD mappings/skinning.
- `_evaluate_model()` — one TorchScript call
  `model(identity, pose, expression, apply_correctives)` → LOD-1 vertices (cm)
  + skeleton state `(J, 8)` = translation + quaternion + uniform scale.
- `_render_lod()` — maps LOD-1 vertices to the active LOD with the official
  barycentric conversion (`triangle_ids` + `baryc_coords`).
- `_infer()` — evaluate → LOD transfer → optional **snap-to-ground**
  (measures lowest rendered Y, folds a correction into `root_ty`, re-evaluates)
  → packs a `GeometrySnapshot` (meters) with revision, inference ms, `root_ty`.
- `update()` — applies one `set` / `reset_current` / `reset_group` /
  `reset_all` action, then re-infers. `root_ty` is rejected while snapping is
  on (it is server-managed).
- `configure()` — changes LOD / `applyCorrectives` / `snapToGround`, re-infers
  on change. Disabling snap restores the pre-snap `root_ty`.
- Caches: rig-mapping bytes, blendshape weights (raw + gzip), pose-corrective
  weights (raw + gzip), per-LOD neutral GLBs, per-LOD skinning and mappings.
- Metadata enrichments computed at startup: pose→joint influence map (from
  `get_parameter_transform`), identity-component anatomical zones (from LBS
  weights), FACS expression names/zones (loaded from `mhr/face_expression.py`
  by path, with fallback).

### 3.2 HTTP API

Used by the client:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Viewer page |
| `GET` | `/static/<…>` | App assets |
| `GET` | `/vendor/three/<…>` | Local three.js distribution |
| `GET` | `/api/metadata` | Parameters, limits, categories, joints, model config (`no-store`) |
| `GET` | `/api/rig-mapping` | `T_p` matrix + joint offsets + pre-rotations (cached 24 h) |
| `GET` | `/api/weights/blendshapes` | Neutral template + identity/expression deltas (gzip) |
| `GET` | `/api/weights/pose-correctives` | Sparse corrective MLP weights, CSR layer 2 (gzip) |
| `GET` | `/api/model/rigged-glb?lod=&v=` | Neutral rigged GLB for an LOD (`v=` = topology version cache-buster) |
| `GET` | `/api/lod-mapping?lod=` | Barycentric LOD-1 → LOD-N mapping |
| `POST` | `/api/deform?nogeometry=1` | Background param sync; `204` + headers, no body |
| `POST` | `/api/configure` | LOD / correctives / snap changes; returns fresh metadata |
| `GET` | `/api/export/{obj,glb,usd}` | File download built from the server snapshot |
| `GET` | `/api/mixamo/clips` | Mixamo folder listing: name/bytes/mtime per `.fbx`/`.glb` (flat, sorted) |
| `GET` | `/api/mixamo/file/<name>` | Clip bytes; flat namespace, traversal + extension guarded |
| `GET` | `/health` | Status + active config |

Served and tested but **not called by the current frontend** (V1 remnants from
when the server streamed every posed frame):

| Method | Endpoint | Notes |
| --- | --- | --- |
| `GET` | `/api/geometry` | Full vertex + joint buffer (has `?unposed=` variant) |
| `GET` | `/api/topology` | Face indices for the active LOD |
| `POST` | `/api/deform` (without `nogeometry`) | Full geometry response |

Response headers carry the out-of-band channel: `X-MHR-Revision`,
`X-MHR-Inference-Ms` (server TorchScript time), `X-MHR-Root-TY` (post-snap
`root_ty`). Error bodies are generic (`{"error": …}`); exception details are
deliberately not exposed (covered by tests).

### 3.3 Binary wire formats (all little-endian, parsed with `DataView`)

- **rig-mapping**: `u32 rows, u32 cols, u32 joints`, then `T_p` (`rows×cols`
  f32, 889×204), joint offsets (`J×3` f32, cm), pre-rotations (`J×4` f32
  quaternions).
- **blendshapes**: `u32 vCount, idCount, feCount`, neutral `vec4` per vertex,
  then `(idCount + feCount)` `vec4` deltas per vertex (meters).
- **pose-correctives**: magic `0x4350484D` + dims, sparse Layer-0 COO
  (`u16` rows/cols + f32 weights, 750→3000), Layer-2 CSR (`u32` indptr,
  `u16` indices, f32 data; 55317×3000, ~97% sparse).
- **lod-mapping**: `u32 targetCount` (0 = identity for LOD 1), then `u32`
  source triplets + f32 barycentric triplets.

### 3.4 Exporters (`exporters.py`)

Pure functions `export_mesh(vertices, faces, format, rig)`. OBJ is a static
mesh via trimesh; GLB is hand-built (`_GlbBuilder`: JSON chunk + bin chunk,
accessors, skins, joints) and USD via `usd-core` — both include the armature
and 4 normalized LBS influences per vertex. The same GLB writer serves the
live neutral rigged model (`neutral_rigged_glb`, neutral pose, per-LOD cache).

### 3.5 LOD pipeline

Native model topology is LOD 1. Other LODs come from two committed/generated
inputs: `data/lod_topology.npz` (faces per LOD, built by
`tools/build_lod_topology.py`, which reads FBX *control points* directly so
indices still align with the official mappings — renderers that split UV seams
would break that) and the official `lod1_to_lodN_mapping.npz` barycentric
files. Skinning for non-1 LODs is derived by splatting LOD-1 influences
through the barycentric map and keeping the top 4, renormalized.

## 4. Frontend

### 4.1 Module map

| Module | Owns |
| --- | --- |
| `app.js` | Boot (`initialize`), all DOM wiring, reset animation, sim panel glue |
| `js/state.js` | Central `elements` (every DOM query) + `state` (metadata, selection, rig buffers, mesh/skeleton handles) |
| `js/actor.js` | `Actor` (per-character param vectors, skeleton handle, steering + gait tick) + `actorManager` registry / active-target singleton |
| `js/scene.js` | Renderer (WebGPU→WebGL fallback), lights, cameras + `OrbitControls`, helpers, skeleton overlay, on-demand render loop |
| `js/rig.js` | Binary parsers + the full client-side evaluator (pose FK, blendshapes, correctives, LOD transfer) + GLB loading |
| `js/params.js` | Parameter list (search/category/region grouping), slider + readout sync |
| `js/sync.js` | Background `POST /api/deform?nogeometry=1` syncs (pose / blendshape, abort-on-new) |
| `js/gizmo.js` | Clickable joint spheres + single-axis `TransformControls` driving pose params through `T_p` |
| `js/exprVectors.js` | Expression overview lines on the face (min-to-max travel of each shape's strongest vertex): WebGPU fat lines, single click pins a persistent translation tooltip (vector + travel + mirror; Esc/click-miss dismiss), drag-to-scrub values along the line (captures OrbitControls while scrubbing), FACS mirror pairs (highlight-only — shapes move independently), 3D ↔ list selection both ways |
| `js/exprVectorMath.js` | Pure mirror-pairing + strongest-vertex helpers, import-free for node tests |
| `js/crowd.js` | Agent crowd façade over `./sim` (spawn/select/behaviors/LOD/stats) + mesh-replace listener |
| `js/sim/AgentSimulation.js` | Agent lifecycle, play/pause/step/time-scale, picking, selection ring |
| `js/sim/Agent.js` | `Agent extends Actor`: skeleton-clone construction only; steering, gait, and posing are inherited |
| `js/sim/GaitController.js` | Procedural walk (sparse `T_p` accumulation over 11 gait params → bones) |
| `js/sim/Steering.js` | Reynolds behaviors (wander/seek/flee/separation/alignment/cohesion/contain/circle) |
| `js/perf.js` | Telemetry card: ring buffers, frametime graph, VRAM/RAM accounting, tab switching |
| `js/cache.js` | `fetchBinaryAsset`: IndexedDB (`MHR_CACHE_V3`) → HTTP `force-cache` → store |
| `js/ui.js` | Status/loading/input-disabling/model-stats helpers |
| `js/utils.js` | Formatting, `fetchChecked` (JSON error extraction), download filenames |
| `js/mixamo.js` | Mixamo clips: list/load, 54-bone → joint map, global-match ZYX retarget (local-triple fallback), per-actor `ClipPlayer` (own mixer, independent phase), hips-Y bob, spine boost, Global toggle |
| `js/eyesOral.js` | Main-only eyes + oral cavity (always on, no toggle): TSL eyeballs, blendshape blink/gaze/lids, dental arches, skinned tongue, jawDrop-following lower assembly |
| `js/crowdEyes.js` | Instanced eyeballs for all agents (1 draw): CPU head bones / GPU head-chain FK matrices, gaze wander, GPU-only blendshape blinks, per-instance pupils (`aPupil`) |
| `js/crowdMouth.js` | Inner mouths for LOD1 GPU agents: Main's arch/tongue/cavity builders re-anchored per tick under the shared head matrices, jawDrop-following lower assembly, ~12 draws per mouth |
| `js/clothing.js` | Prototype procedural garments (Main only): shirt + shorts from body verts offset along normals, body's own skin weights, refit-on-identity-change |
| `js/skin.js` | `createSkinMaterial(tint)`: standard-PBR node body material with object-space mottling/grain; tint stays in `.color` so clones keep independent tints |
| `js/vram.js` | DOM-free VRAM accounting: sums attribute `byteLength`s deduped by array (pool shells share mega attributes — geometry-level dedupe caused the 77GB bug) |
| `js/bench.js` | Cross-device benchmark: clean-room spawn → warmup → per-frame sample (fps, tick parts, calls/tris, gpuExec) → restore; localStorage history + JSON export/import |
| `js/actions.js` | Agent action catalog: 10 OpenAI-format tool schemas, validators, DI dispatcher (never throws), subscribed audit log |
| `js/agent.js` | Llama tool loop: system prompt, state summary, chat-completions client, multi-round turn runner with approval gates |
| `js/agentPanel.js` | Floating agent chat console: transcript, settings (localStorage key), inline approvals, audit-trail rendering |
| `js/dock.js` | Phase-2 dockview shell: 16 section tabs (viewport, 4 world, 9 actor, telemetry, crowd stats) moved from staging into dock panels, `onPanelActivated` param sync, persisted layout, reset button |

`state.js` is the hub: every module imports `state`/`elements` from it, and
`tests/test_frontend_wiring.py` statically asserts each `querySelector` target
exists in `index.html`, so markup/JS ID drift fails fast.

### 4.2 Boot sequence (`app.js#initialize`)

1. `GET /api/metadata` → parameter definitions, joint table, model config.
2. Parallel `fetchBinaryAsset` of rig-mapping, blendshape weights,
   pose-corrective weights (each IndexedDB-cached).
3. `setRiggedMesh(lod)`: loads the neutral rigged GLB via `GLTFLoader`,
   fetches the LOD mapping if LOD ≠ 1, maps metadata joint order → skeleton
   bones **by name**, binds `bodyMaterial`, registers the main character as
   anchor Actor #0 (seeded from metadata pose values; preserved across LOD
   switches), then evaluates shape + pose and fits the camera. Body +
   armature render under the anchor's group (identity at build), so the
   Scale slider works on Main like on agents; removal is parent-agnostic
   and scale survives re-rigs.
4. `eyesOral.attach()`: anchors the procedural eyes/teeth/tongue to the Main
   head/jaw bones (fails soft with a console warning if joints are missing).
5. Builds categories, selects `head_lean`, shows the pose tab, hides loading.

### 4.3 Client-side evaluation pipeline (`rig.js`)

This is what the viewport shows; it never touches the network:

1. **Pose FK** — `computeThetaJ()`: `θⱼ = T_p · θ_p` (204 params → 127×7
   joint channels). `evaluatePoseClientSide()` composes each bone as
   `pos = (offset + t)/100`, `quat = preRot × eulerZYX(rx,ry,rz)`,
   `scale = 2^s`, then `skeleton.update()`. Verified conventions: Euler order
   `ZYX` (`R = Rz·Ry·Rx`) and exponential scale (see `v2/v2.md` header).
2. **Correctives** — 6D rotation features for joints 2..126 →
   sparse Layer-0 → ReLU → CSR Layer-2 SpMV → per-vertex deltas (cm→m).
   Skipped entirely when the toggle is off.
3. **Blendshapes** — `evaluateBlendshapesCPU` sums only weights with
   `|w| > 1e-6` over neutral + deltas into `unposedBase`.
4. **Compose + LOD** — `updateMeshVertices()`: `base + correctives` at LOD-1
   resolution, then barycentric transfer for other LODs; writes the `position`
   attribute in place, recomputes normals/bounds on the CPU.
5. **Skinning** — free on the GPU: the persistent `SkinnedMesh` + `Skeleton`
   mean posing never rebuilds geometry or materials.

Decoupled follow-ups run through listener sets to avoid import cycles:
`addPoseUpdateListener` (gizmo sphere sync) and `addMeshReplaceListener`
(crowd teardown before LOD mesh swap). Pose evaluation also mirrors the fresh
joint positions into anchor Actor #0 (`mainActor.jointPositions.set(…)`) so the
gizmo can follow the active actor (§4.10).

### 4.4 Rendering (`scene.js`)

- Renderer: tries `WebGPURenderer`, falls back to `WebGLRenderer`. Either way
  it is used as a plain rasterizer — there are no compute shaders; the
  WebGPU/TSL compute design in `v2/v2.md` was never implemented.
- **On-demand rendering**: `requestRender()` coalesces to one rAF; an idle
  timer flips the FPS/perf badges to "Idle" after 250 ms without frames.
  Orbit damping re-requests frames while settling.
- Overlays: preallocated skeleton line/point buffers (sized once per joint
  count), hover tooltip by projecting joints to screen space, grid/axes
  helpers, perspective/orthographic cameras with cardinal + face-fit presets.
  The view/fit/projection buttons live in the viewport corner toolbar
  overlay (the Camera card was deleted); the World card (ex-Appearance)
  holds scene globals + the pose-correctives switch. Click-select raycasts
  agent capsules / CPU meshes / Main (a missing `THREE` import in `app.js`
  once broke every click). View/fit buttons frame the active actor via
  `fitCamera(view, focus)` (`scene.js` keeps the last focus across
  projection toggles); the selection ring starts on Main, matching the
  dropdown/cards. Genuine selection changes jump the camera (skeleton sync
  runs first so face view measures the new actor); spawn never steals the
  selection. Face view frames the posed head midpoint and approaches along
  the posed face-forward from the eye joints (`faceDirection()`), so it
  follows neck/head rotation on Main and agents instead of world front. The skeleton overlay + tooltip + joint spheres
  track the active actor via `syncDisplayedSkeleton()` (`actor.js`: rig
  eval, selection change, pose writes, sim tick; GPU reads the probe);
  spheres hold ~10px via per-joint pixel math, rescaled on camera moves.

### 4.5 Parameter UI + reset (`params.js`, `app.js`)

Pose tab offers search + anatomical category filter; identity/expression tabs
group components under anatomical/face-zone region headers. Slider `input`
evaluates locally at full rate; `change` (release) additionally fires one
background server sync. The numeric readout commits on Enter/blur. Resets
animate values to neutral over 250 ms (ease-out cubic), evaluating locally
each frame, then POST the reset action to the server.

All three write paths (slider, readout, reset) are actor-aware: the value is
written to the global metadata param *and* to the active actor's own vector,
then evaluation branches — anchor runs the full rig.js path (+ server sync),
a selected agent runs `actor.evaluatePose()` with no sync (§4.10). The panel
header shows a "Target Actor" banner with the active actor badge, and
`showParameter` displays the *active actor's* stored value (writing it back
into the global param).

### 4.6 Joint gizmo (`gizmo.js`)

Pose tab only, and it force-enables the skeleton overlay. The gizmo operates
on the **active actor**: its joint spheres track the actor's `jointPositions`,
its proxy syncs from the actor's skeleton, and it re-syncs on every
`actorManager` change. Selecting a sphere attaches a `TransformControls` proxy
in move/rotate/scale mode. The core invariant is **one movement, one
parameter**:
per joint channel it resolves the strongest-driving *unlocked* pose parameter
via the `T_p` column weights, hides every multi-axis handle, and maps the
drag delta back through the anchor parameter's weight
(`Δparam = Δchannel / weight`, clamped to limits). Euler branch cuts are
unwrapped against drag start. Drag deltas write through to the actor's pose
vector (`setParamValue`, which re-poses immediately); anchor drags then run
the full `evaluatePoseClientSide()` path while agent drags run the lighter
`actor.evaluatePose()`. Release batch-syncs touched params to the server for
the anchor only — agents never sync. Selection runs both ways: clicking
a sphere jumps the panel to that joint's strongest driver, and clicking a
parameter reveals its hardest-driven joint. The opening tool follows the
(param, joint) pair's dominant channel, so scale-driven params get the
scalar uniform-scale tool instead of rotate/move. Scale mode shows only
the uniform center box (no per-axis length handles — the drag math was
already scalar); move mode points X through the current param's affected
joints (exact for two, PCA for more), falling back to the bone frame.

### 4.7 Crowd system (`crowd.js`, `sim/`)

Agents are the only crowd system (the old clone/instance stress tiers were
removed): full `AgentSimulation`, where each agent is an `Actor` (see
§4.10): deep-cloned bone hierarchy (independent skeleton, geometry shared
with the anchor), Reynolds steering (wander/flock/circle/scatter + obstacle
avoidance + containment), and procedural per-tick posing through
`GaitController`'s sparse `T_p` path (11 gait params, sinusoidal
phase/cadence). The navbar holds the global time transport
(play/pause/step/skip/time-scale); the loop runs from boot even with zero
agents, and emptying the scene no longer stops it. The Crowd panel keeps
spawn/clear/behavior/LOD plus click-to-select raycasting and camera focus;
per-actor state lives in the
Actor card (pose/identity/expression/eyes/mouth/materials/clothing/
animation/ai tabs, retargeted to the selection, unsupported controls
disabled with reasons). Selecting an agent also makes it the active actor,
so the actors list, gizmo, and Actor card retarget to it; the anchor
additionally offers a `manual` behavior (no tick movement). The card
absorbed the standalone parameter panel (one shared component, reparented
into the pose/identity/expression mounts), the Mixamo card (as the
Animation tab), and the Eyes card (eyeball detail in Eyes, teeth/tongue
in Mouth); the dead target banner went with the panel.

### 4.8 Perf card (`perf.js`, markup in `index.html`)

Live-updating: frametime/eval-latency ring buffers + canvas waveform,
draw calls/triangles from `renderer.info`, mesh/crowd census by scene
traversal, VRAM estimate (geometry attribute bytes + weight buffers),
heap-or-weights RAM figure, cumulative "bandwidth saved" counter, and a live
pipeline breakdown bar (FK / blendshapes / correctives / normals+bounds+LOD /
render widths driven by `recordStageTimes` measurements, with a
normals-vs-bounds-vs-transfer tooltip). The V1-vs-V2 table and "96% faster"
badges remain static marketing copy — see analysis §7.

### 4.9 Caching (`cache.js`)

Binary assets resolve IndexedDB → HTTP → store-back. GLB URLs carry
`?v=<topologyVersion>`, so topology edits bust the cache; the weight/rig URLs
are unversioned (see §7 risk).

### 4.10 Actor system (`actor.js`)

Every character — the anchor main character (Actor #0, `isAnchor`,
`behavior: "manual"`) and each spawned `Agent` — is an `Actor`: own
`poseValues`/`identityValues`/`expressionValues` vectors, own `thetaJ` scratch,
own skeleton handle + `boneIndexMap`, own world `jointPositions`, own
kinematic state (position/velocity/scale/speed/cadence/behavior/color), and two
evaluation entry points: `evaluatePose()` (full `T_p` matvec with a zero-skip,
then the same offset/pre-rotation/scale bone composition as `rig.js`, but no
correctives, blendshapes, or LOD work) and `update()` (steering → integrate →
gait → bones; `manual` short-circuits to a joint refresh). `Agent` is now a
thin subclass that only builds the cloned skeleton + shared-geometry mesh and
rests the bones; everything else is inherited.

`actorManager` is the registry + active-target singleton: `registerActor` /
`unregisterActor` (anchor is unregisterable), `setActiveActor(id)`, and a
change-listener fan-out consumed by the gizmo (re-sync spheres/proxy),
`app.js` (`onActiveActorChanged`: header name, actors-list sync, Actor card refresh),
and the sim panel. Selection flows through `AgentSimulation.selectAgent`,
which sets *both* `selectedAgentId` (ring, card, focus) and the manager's
`activeActorId`.

Targeting rules, applied uniformly by the slider, readout, reset, gizmo, and
appearance handlers:

- Every write lands on the global metadata param *and* the active actor's
  vector; reads for display come from the active actor (and are written back
  into the global param by `showParameter`).
- Anchor edits run the full `rig.js` path (FK + correctives + blendshapes +
  LOD + normals) and sync to the server; agent edits run `actor.evaluatePose()`
  and never touch the network. Appearance edits retarget the same way (anchor
  → shared `bodyMaterial` + crowd; agent → its own material).
- Because agents share the anchor's geometry, per-agent identity/expression
  values are stored but have no visible effect — only pose (bones) differs per
  character. Related sharp edges are collected in §7.

GPU crowd agents (one pool per LOD 1–6, `static/js/gpu/`) are the exception
to the shared-geometry rule: `GpuAgent` extends `Actor` but renders a slot
slice of pool mega buffers instead of a skinned mesh, posed each tick by
batched TSL FK + skin compute from the agent's own `thetaJ` + shape weights
(see `gpu.md` §9), with auto-LOD migration between pools by camera distance
(see `gpu-crowd.md` §5.4). Per-agent identity/expression therefore *do*
render for GPU agents; CPU agents keep the limitation above. The anchor is
always CPU. Steering, waypoints, and gait phase stay on the CPU for both
modes — only FK + skinning moved to the GPU. The Crowd Stats panel exposes
per-pool state, per-agent mode/slot/LOD/timings, and the fallback reason.

### 4.11 Eyes + oral cavity (`eyesOral.js`, Main only)

Procedural attachments for the anchor, adapted from
`v2/mhr_eyes_oral_cavity_module` (dental arches, skinned tongue,
throat/palate/floor, eyeball proportions, iris shading). Always on: the
Eyes & mouth master toggle was removed (crowd eyes/mouths are always on
too). Key deviations from that module, all verified against this rig:

- Lids are the face's own blendshapes (`eyesClosed_L/R`, `lidTightener_*`,
  `upperLidRaiser_*`) — no prosthetic caps, no double geometry.
- Expression channels resolve **by name** against
  `metadata.groups.expression` (module's `FACS_IDX` table does not match our
  72 named actions); gaze also rotates the eyeball meshes.
- Lower assembly (lower teeth, tongue, floor) is head-parented and re-posed
  on every pose/blendshape evaluation from live jaw-bone motion plus a
  `jawDrop`-driven drop/retrusion — never from `jawBone.rotation.x`, which is
  meaningless under our pre-rotated ZYX composition.
- Eye iris/pupil/sclera shading is a branchless TSL port
  (`MeshBasicNodeMaterial` + `colorNode`) so it compiles on WebGPU and the
  WebGL fallback; GLSL `ShaderMaterial` would only run on the latter.
- Module-space content is anchored at runtime: a basis from the live
  `l_eye`/`r_eye`/`c_head_null` joint positions, module eye midpoint pinned
  to ours. No demo-proxy absolute offsets survive.

`rig.js` exposes `addBlendshapeUpdateListener` (mirroring the pose one) so
the lower assembly tracks `jawDrop` no matter which UI wrote it. All writes
go through one `_writeExpressions` batch → `evaluateBlendshapesClientSide`,
and the single-slider readout refreshes when it shows a touched channel.
Crowd agents get instanced eyeballs from `crowdEyes.js` (own iris
material with a per-instance `aPupil` attribute driven by `Actor.pupil`,
matrices from CPU head bones or a GPU head-chain FK over the agent's own
`thetaJ`, gaze wander for all, real `eyesClosed` blinks for GPU agents via
the every-tick weight upload — free). CPU agents can't blink (shared
anchor geometry). LOD1 GPU agents additionally get inner mouths from
`crowdMouth.js` (same arch/tongue/cavity builders re-anchored under the
agent head matrix each tick, lower assembly following the agent's
`jawDrop`; ~12 draws per mouth, shared geometries, neutral teeth).

### 4.12 Agent console (`actions.js`, `agent.js`, `agentPanel.js`)

A floating chat console drives the app through a validated action catalog
(10 tools: select/setParam/spawn/clear/remove/camera/gaze/behavior/
lodColors/benchmark), dispatched onto the same functions the UI uses, so the
agent can never do what the buttons could not. The loop (`runAgentTurn`)
talks to Meta's Llama API over its OpenAI-compatible endpoint
(`.../compat/v1`, user key in localStorage, browser-direct, no backend
secrets), feeds tool results back multi-round, and gates destructive actions
behind inline Approve/Deny. Every dispatch — chat, console, or future
remotes — lands in a subscribed audit log (`createActionLog`, source-tagged,
rendered in the console's Action log section). Pure layers (schemas,
validation, loop, state summary) are node-tested with stubs; `window.mhr`
exposes `{ dispatch, snapshot, actions, log }` as a console plug for
external agents.

## 5. Client–server contract: the dual-evaluation model

Rendering needs no backend after boot. The post-init requests exist because the
server keeps a second, authoritative copy of the *anchor* character for
export. Agent (non-anchor) edits never touch the network at all:

| Flow | Client does | Server does |
| --- | --- | --- |
| Scrub (`input`) | Full local eval, zero network | Nothing |
| Release (`change`) | Re-evaluates (cheap) | `POST /api/deform?nogeometry=1`: applies the param, runs full TorchScript inference, returns `204` + `X-MHR-Root-TY` / `X-MHR-Inference-Ms` |
| Non-anchor edit | `actor.evaluatePose()` on the agent's own vector, zero network — no sync on release either | Nothing (agent state exists only in the browser) |
| Snap-to-ground | Reads `X-MHR-Root-TY`, patches local `root_ty`, re-evaluates | Computes the correction in `_infer()` (lowest-Y → `root_ty`, re-infer) |
| Correctives toggle | Applies locally immediately | Fire-and-forget `POST /api/configure` (response ignored) |

The Export section and global LOD dropdown were deleted from the UI: Main
stays at the server default (LOD 1) and nothing reads the server snapshot
back. `/api/export/*` and full `/api/configure` remain implemented and
server-tested, but uncalled by the client.

Consequences: with no export flow left, the server snapshot is write-only;
`nogeometry` syncs still cost a full server-side inference each (geometry is
computed, then discarded); and anything the server computes that the client
also needs (today: only the ground-snap correction) must round-trip.

## 6. End-to-end walkthroughs

- **Slider scrub**: `input` → `parameter.value=` + active-actor vector →
  anchor: `evaluatePoseClientSide()` (or blendshapes); agent:
  `actor.evaluatePose()` via `setParamValue` → `requestRender()` → coalesced
  rAF → `renderer.render` + `updateRuntimeStats` + `drawPerfGraph`. Release
  adds one `sync.js` POST for the anchor, none for agents.
- **Actor switch** (actors list, 3D pick, or spawn): `selectAgent` →
  `selectedAgentId` + `activeActorId` → manager listeners refresh the header,
  gizmo spheres/proxy, Actor card, and panel values from the new actor's
  vectors → `requestRender()`.
- **Gizmo drag**: pointer events → `proxyToChannels()` (world → parent-local →
  strip pre-rotation → ZYX Euler, unwrapped) → anchor param update →
  `evaluatePoseClientSide()` per move event → `syncPoseParams(changed)` on
  mouse-up.
- **Agent tick**: rAF loop → per agent: steering forces → integrate →
  heading slerp → gait phase advance → `GaitController.evaluateAndApply`
  (sparse `T_p` → bones → `skeleton.update()`) → `requestRender()`.

## 7. Analysis

### Strengths

- **True client-side rig.** FK, blendshapes, sparse-MLP correctives, and LOD
  transfer all run locally against parity-tested math
  (`tests/test_client_parity.py` checks each stage against TorchScript); the UI
  never blocks on inference while scrubbing.
- **Disciplined buffer management.** Persistent `SkinnedMesh`, in-place
  position writes, preallocated skeleton overlays, coalesced on-demand renders,
  scratch-vector reuse in hot paths (`rig.js`, `Steering.js`).
- **Good seams.** Listener sets instead of import cycles (`rig.js` →
  gizmo/crowd), bone mapping by name (not exporter order), ID contract test
  between markup and `state.js`, generic server error envelopes.
- **One crowd system.** The old clone/instance stress tiers were deleted;
  autonomous agents (CPU + GPU pools) are the only crowd left, with
  per-agent cost honestly labeled in the stats panel.
- **Thoughtful sync design.** Abort-on-new coalescing, no per-frame sync;
  the anchor syncs on edit-stop (dirty-drain).
- **Unified actor model.** One `Actor` class serves the anchor, the gizmo
  target, the sim agents, and the Actor card; `Agent` shrank to a constructor.
  Targeting rules (writes fan out, anchor syncs, agents stay local) are
  applied consistently across slider, readout, reset, gizmo, and appearance
  paths.

### Weaknesses and risks

1. **Single global server character.** Two tabs/browsers share one pose state;
   every sync POST mutates it for everyone. Fine for a local single-user tool,
   but it rules out any shared deployment without session state.
2. **`nogeometry` syncs still run full inference.** `engine.update()` always
   calls `_infer()` (TorchScript + snap + LOD transfer) and then throws the
   geometry away. A separate param-only path would make syncs ~free.
3. **Reset POST downloads a body it discards.** `animateReset` POSTs to
   `/api/deform` *without* `?nogeometry=1`, so the server returns the full
   vertex+joints buffer and the client ignores it. One query param fixes it.
4. **`renderTime` has two writers.** Client eval writes local ms; each sync
   response overwrites it with server inference ms (`updateInferenceTime`).
   The readout flickers between two different quantities — last response wins.
5. **Hardcoded telemetry marketing.** The V1-vs-V2 table, "96% faster / 10×
   FPS" badges, and the `226611`-bytes-per-input bandwidth constant are
   static, not measured. They will silently drift from reality. (The pipeline
   bar itself is live since the `recordStageTimes` change.)
6. **Unversioned weight cache.** GLB URLs carry the topology version, but
   rig-mapping/blendshape/corrective URLs don't. Swapping `--model` (or any
   weight change) serves stale IndexedDB bytes until the user clears storage.
   A model-hash query param would fix it.
7. **Brittle gait coupling.** `GaitController.GAIT_PARAMS` hardcodes pose
   parameter *indices* (e.g. `r_knee_bend = 53`). Any model reorder breaks
   locomotion silently — resolve by name from metadata instead.
8. **Dead surface area.** `/api/geometry`, `/api/topology`, and full-body
   `/api/deform` responses are served, tested, and never called by the client
   (V1 leftovers). Same for the unwired `v2/` eyes/oral prototype. Deleting or
   documenting them would shrink the maintenance surface.
9. **Fire-and-forget correctives toggle.** The client applies the toggle
   locally and ignores the configure response; a rejected POST leaves client
   and export snapshot diverged with no signal.
10. **Abort can split batch syncs.** `syncPoseParams` posts indices serially
    under one `AbortController`; a newer sync aborts mid-batch, so the server
    may apply a prefix of a gizmo drag. Harmless today (only export reads it,
    and exports flush), but worth knowing.
11. **CPU normal recompute per edit.** `updateMeshVertices()` runs
    `computeVertexNormals()` + bounds over the full mesh on every pose/shape
    change — the exact main-thread cost `v2/v2.md` item D wanted on the GPU.
    The WebGPU path is renderer-only; no compute shaders were ever added.
12. **Fragile scratch sharing in `Steering.js`.** All behaviors reuse the same
    module-level `_steer/_desired/_v1/_v2`; today every call site copies out
    immediately (`_force.copy(…)`), but one direct hold across calls would
    corrupt forces. Fine, but it deserves its "don't hold" comment.
13. **Dev-server deployment.** `app.run()` (single process, threaded Flask
    dev server) bound to loopback — correct for a local tool, not a service.
14. **Global param vector is aliased "last displayed" state.** Every actor
    write also overwrites the shared metadata param, and `showParameter`
    copies the active actor's value back into it — so the metadata no longer
    means "anchor state", it means "whatever was displayed last". Crowd
    randomize bypasses actor vectors entirely (writes metadata, evaluates,
    then `showParameter` restores the stale actor value to the panel), leaving
    panel and mesh briefly disagreeing until the next edit.
15. **Non-anchor shape values are stored but inapplicable.** Agents share the
    anchor's geometry, so per-agent identity/expression vectors — and the
    identity/expression tabs while an agent is targeted — change numbers that
    nothing renders. Only pose differs per character; the UI doesn't say so.
16. **FK exists twice.** `rig.js` (`computeThetaJ` + bone loop) and
    `Actor.evaluatePose` (same math plus a zero-skip) must be kept in step by
    hand — including the ZYX order and `2^s` conventions. A shared
    thetaJ-to-bones helper would remove the drift risk.
17. **Dual selection state.** `selectedAgentId` (sim: ring, card, focus)
    and `activeActorId` (manager: gizmo, panel, card) are two fields
    kept in sync by convention inside `selectAgent`. Any future selection path
    that updates one and not the other splits the UI.

### Simplification path (export UI gone; backend retained)

The Export section and global LOD dropdown are deleted from the UI, but the
syncs still run and the backend still serves inference: the full
simplification from here is prebaking the per-LOD GLB/mapping/weight blobs
to static files, reimplementing snap-to-ground client-side (lowest local
vertex Y → `root_ty`, a few lines in `rig.js`), deleting `sync.js` + the
deform/configure/export endpoints, and serving the directory with any
static server. The parity tests would remain valuable as the spec for the
client math.

## 8. Key invariants (for future edits)

- Model units are cm; wire + client units are m. `/100.0` on the way out.
- Joint rotations compose `R = Rz·Ry·Rx` → `THREE.Euler(..., "ZYX")`.
- Joint scale channel is exponential: `s = 2^d`.
- `T_p` maps 204 pose params → 127 joints × 7 channels
  `[tx,ty,tz,rx,ry,rz,s]`; it is also the gizmo's param-resolution oracle.
- Corrective MLP input is 6D features of joints 2..126 (first two rows of
  `Rx·Ry·Rz`, identity-subtracted), *not* raw `θⱼ`.
- LOD 1 is native; every other LOD is faces (committed `.npz`) + barycentric
  transfer, for both server snapshots and client meshes.
- Bone identity is the joint *name*; never array order.
- Server snapshot is export truth; client state is viewport truth; the
  `pendingSyncs()` flush is what keeps a scrub-then-export correct.
- Actor #0 is the anchor: main mesh, `manual` behavior, unregisterable, and
  the only actor that syncs to the server. Agents are pose-only variations —
  independent skeletons over geometry shared with the anchor.
- Writes fan out (global param + active-actor vector); display reads come from
  the active actor. Any new write path must do both or the panel and the mesh
  will disagree.

## 9. Parked threads

- **Mixamo retarget verification (awaiting user retest).** Loop-wrap snaps
  were the confirmed twitch cause (open clips hit 111° end→start gaps);
  fixed with per-clip closure detection (`isLooping`, 10° threshold) that
  auto-defaults the Loop checkbox (seamless→loop, open→once+hold) plus a
  Loop metadata row. Then playback froze (mixers advanced, `thetaSum` 0 on
  all actors): ActorCore files nest a second, static rig inside the
  animated one (129 bones, 65 unique names), and our name→bone map kept
  the LAST traversal match while the mixer binds the FIRST — we sampled
  the dead rig. Fixed with first-wins maps in `resolveChannels` +
  `ClipPlayer` (regression: duplicate-name synthetic test). The rest
  reference also moved from bind pose to animation first frame
  (`captureFirstFrameRest`), so play/stop never pops. Verified against
  the real `mixamo.fbx` in node (thetaSum 0.0 → 7–13 through the dance).
  Suites: node 72/72, wiring 37/37. Floating feet / stiff hips traced to
  missing root translation (26cm of crouch doing nothing); fixed with
  first-frame-relative hips-Y bob onto the root slot (thetaJ cm on both
  paths, cm/m auto-detect, X/Z stay steering-owned), plus a live spine
  exaggeration factor (0-4x on c_spine0/1/2, UI slider, faithful at 1x)
  for performances with a stiff mid-back. Frozen knees traced to rest-frame
  twist (our X-down-bone vs Mixamo): knee energy landed on twist (rx 9.9)
  instead of the Z hinge (rz 3.6); fixed with global-match retarget
  (ourGlobal = group x neutral x srcDelta, parent-first, group heading in;
  knee hinge energy 3.6 -> 10.3, world bend 0 -> 36.5deg), UI toggle +
  legacy fallback, verified against real rig data in node. Suites now
  75/75 node, 38/38 wiring. Possible follow-ups: crossfade looping for
  open clips,
  play-once-then-resume-gait for agents, `c_spine3` distribution
  (currently holds rest), per-clip loop-override persistence.
- **Mouths for crowds (shipped, LOD1-only).** `crowdMouth.js` re-anchors
  the Main mouth builders under LOD1 GPU-agent heads each tick (upper
  rigid, lower follows the agent's `jawDrop`, ~12 draws per mouth, shared
  geometries, neutral teeth). Not yet done: merging the per-arch meshes
  down from 12 draws, per-agent teeth morphs (shared style for now).

