# GNM-V Model Flow Visualizer (`pivot/chat-wrapper/flow-viewer`)

A transparent node-canvas frontend for the multi-stage ML pipeline in `pivot/chat-wrapper`: drop a photo, run Meta SAM 3.1, then wire downstream Sapiens / 3D-solver models off the resulting person crops. Vanilla HTML/CSS/ES-modules, no build step, no dependencies.

---

## Start of pipeline: Source Photo → SAM 3.1

New graphs start with the source photo alone — no default action card; wire it into SAM (the usual first step) or any other card from the sidebar. The entry, once wired, works like this:

1. **Asset: Source Photo** — from a run preset (`GET /api/runs/<id>/artifact/source`) or a user drag-and-drop (blob URL). Real pixel dimensions are probed on load and shown on the card badge and in the inspector (`[H, W, 3] uint8`); before the image resolves the card shows `? × ?` rather than guessing.
2. **Action: Meta SAM 3.1** (`sam-3.1`) — segments the photo on a noun-phrase concept (default `people`, editable at input level next to the Concept Prompt pin, with `people / person / face / clothes` chips). Runs `POST /api/debug/detect` with `{ chat_id, concept, image_b64?, image_url?, filename }`. Dropped photos are sent as base64; preset photos are sent as their `/api/runs/…/artifact/…` URL, which the backend resolves from disk. Request timeout is 200 s (just over the backend's 180 s per-call cap, so slow live calls never flip to demo).
3. **Action: Meta Describe** — whole-scene read in one shot (scene, subjects, appearance, composition) via `POST /api/debug/describe` with `{ chat_id, image_b64?, image_url?, filename }`. No boxes, no masks; the description lands as `out-text` with the input echoed in Preview. Same four terminal states as SAM; timeout 200 s.
4. **Action: Apply Mask** — crops one SAM mask server-side via `POST /api/debug/apply-mask` with `{ chat_id, debug_id, mask_id }`, where the ref comes from a Mask item pin exploded from a live SAM run (demo pins carry no ref and fail fast with the remedy). Runs locally (OpenCV, ~1 s, no Meta call) and yields the region as a plain `out-image` — wire it into Describe, Spark or Sapiens to work on one mask's region. Wiring a mask straight into Describe fails with a pointer to this card. The Local Tools group also holds **Action: Image Difference**: a fully local A−B amplified pixel diff (B resampled to A when sizes differ, both capped at 1024px) with mean/peak Δ stats — wire an overlay against its photo to quantify a fit.
5. **Outputs are four typed lists.** On success the backend returns `{ masks[], combined_url, source_url, image_width, image_height, concept, api_trace.meta.latency_s }`, normalized into `boxes`, `masks`, `confidences`, `labels` (instance labels are derived as `"<concept> N"`). Pin badges show the counts, and a Result section appears on the card under the outputs: the combined overlay with toggleable masks/boxes/labels/confidence.
6. **Explode in place (UE-style).** Right-click any filled list pin → Explode splits it into item pins with tiny previews (`box`/`mask` thumbnails, `number`/`text` values) that you can wire downstream individually; double-clicking a filled list pin splits it too. Item thumbnails always keep their original aspect ratio and are never cropped. Collapse removes the item pins and their wires; re-running the card collapses splits first so stale items never linger. Leaving a list whole for batch downstream use is future work.
7. **Transparency: full POST + full return on every API call.** Each run captures hop 1 (viewer → backend: full request/response, images redacted like the backend does) and hop 2 (backend → Meta Model API: full upstream POST, full return, model/request/latency/usage meta). Read it in the card's Result → API tab or the inspector's API tab; Copy exports the whole trace. The Logs tab keeps the card's timestamped run history.
8. **Four honest terminal states.** `Passed` (lists filled), `Passed` with `0 items` (live empty lane — normal, try a plainer noun), `Failed` red (backend answered with an error — no mock data), `Demo (Offline)` amber (backend unreachable — mock data, flagged). The inspector states the cause in each case, and image fallbacks are offline SVG placeholders (no external stock faces). The header pill reads `Backend: Offline (Demo)` when the backend is down.

Pipeline order is fixed: SAM 3.1 first, then the chosen person/subject goes to the LLM for detailed description.

9. **Ask Spark about anything.** **Action: Meta Spark** (`muse-spark-1.3-contributor`) is a generic LLM call: prompt text plus optional images in, reply text out — mirroring `POST /v1/responses` (one `input_text` block, one `input_image` block per wired image). The prompt comes from the Prompt pin, else the typed prompt at input level; with images but no prompt it defaults to describing them. So the SAM flow stays one wire: explode → wire a `Box`/`Mask` item into Images → Run. The reply lands in Result → Preview with the full POST + full return in the API tab. Same four terminal states as SAM; running it with neither prompt nor images fails fast with a remedy.
10. **Generate or edit images.** **Action: Meta Muse Image** (`muse-image-1.0`) is generic too: prompt alone generates (`POST /v1/images/generations`), prompt plus 1–10 images edits (`POST /v1/images/edits`). The render lands in Result → Preview and on the output pin.

### Backend contract for the generic cards

The viewer never calls Meta directly — the API key lives server-side, the backend builds overlay/artifact files, and it captures hop 2 of every trace. The canonical server is `../chat_server.py`; the `chat_server.py` copy in this folder is a staging spare, kept byte-identical. Both resolve shared state (`runs/`, `chats/`, `usage.json`), vendor code and viewer statics to `chat-wrapper/` (the `DATA`/`FLOW` roots), so either copy runs correctly with no split-brain state. Endpoint shapes:

- `POST /api/debug/spark` — request `{ chat_id, prompt, images?: [{ image_b64? | image_url? }] }` (each item reuses `_debug_image` field semantics: raw/data-URI base64, relative run/chat artifact URLs, remote URLs). Backend builds `POST /v1/responses { model: "muse-spark-1.3-contributor", input: [{ role: "user", content: [{ type: "input_text", text: prompt }, { type: "input_image", image_url: <data URL> }...] }], store: false }` and returns `{ chat_id, assistant_text, api_trace: { post, response, meta } }`.
- `POST /api/debug/muse` — same request shape. No images → `POST /v1/images/generations { model: "muse-image-1.0", prompt, output_format: "png", reasoning_strength: "low" }`; images → `POST /v1/images/edits { model, prompt, images: [{ image_url: <data URL> }], output_format: "png", reasoning_strength: "low" }`. Decodes `data[0].b64_json`, saves it under the chat debug dir, returns `{ chat_id, debug_id, image_url, output_format, image_width?, image_height?, api_trace }` with the base64 redacted from the echoed trace. Live-verified 2026-11-04 against `api.meta.ai`: Spark text-only (`orange` exact), Spark + image (`Red` one-word), Muse generation (1.9 MB PNG, 1600×1600, decodes), Muse one-ref edit (decodes) — all `HTTP 200` with full POST + full return in the trace and base64 redacted on both hops (~$0.03 spend). Against a backend without these endpoints both cards fail honestly (`HTTP 404`) and demo data stays clearly flagged.
- `POST /api/debug/describe` — request `{ chat_id, image_b64? | image_url?, filename? }` (same `_debug_image` semantics). One Spark vision read via `POST /v1/responses`; returns `{ chat_id, debug_id, assistant_text, api_trace }` with the source file echoed. Full two-hop trace.
- `POST /api/debug/apply-mask` — request `{ chat_id, debug_id, mask_id }` (a Mask item pin's backend ref, not pixels), no Meta call: crops the mask region server-side (dimmed-background crop) and returns `{ chat_id, debug_id, assistant_text, crop, width, height, box }`. Hop-1-only trace, local CPU.
- `POST /api/debug/fit-overlay` — request `{ chat_id, debug_id, image_b64? | image_url? }` (a live body-fit's debug dir + the fitted photo). Re-renders the saved mesh in the photo frame with the same matched-frame painter body-fit uses (cam_t baked in the OBJ, focal + frame from `fit_stats.json`), blends it 50/50 over the photo on mesh pixels only, and returns `{ chat_id, debug_id, assistant_text, files, photo_url, render_url, overlay_url, width, height, render_backend, stats }` — three separate frames, no stitched strip. Hop-1-only trace, local CPU, no GPU slot; photos resized to the fit frame are noted, never silent.
- `POST /api/debug/body-fit` — request `{ chat_id, image_b64? | image_url?, filename? }` (same `_debug_image` semantics). Runs the local pose→maps→3DB→refine→joint chain plus the `render_sam3body` turntable/still and returns `{ chat_id, debug_id, assistant_text, files, stats, mesh_url, sheet }`; `stats` carries verts/faces/beta/focal/cam_t/refine_used/timings plus `turntable`/`mesh_render`/`render_backend` (or `turntable_error`, which never fails the fit). Hop-1-only trace, `409` when the GPU is held.
- `POST /api/debug/export` — request `{ chat_id, mesh_b64? | mesh_url?, filename? }`. Packages the OBJ into `avatar_character.glb` plus `export_stats.json` (CPU, trimesh), collecting sibling spec files; returns `{ chat_id, debug_id, assistant_text, files, stats, glb_url }`. Hop-1-only trace.
- `POST /api/debug/mesh-render` — request `{ chat_id, mesh_b64? | mesh_url?, filename?, mode, count?, yaw_deg?, size? }` (`_debug_mesh` semantics: OBJ or GLB). Renders an orbit still (`shaded`/`depth`/`normal`, 3/4 yaw 35°) or a labelled `multiview` contact sheet (`count` 2/4/6/8 at even azimuths) with the open3d offscreen painter, numpy fallback when open3d is missing; returns `{ chat_id, debug_id, assistant_text, files, image_url, width, height, render_backend, stats }`. Hop-1-only trace, CPU, no GPU slot.

### Pin types (strict, ComfyUI-style)

Every wire type is declared once in `PIN_TYPES` (`js/util.js`) with its canonical color and the exact list of types that may plug into it. A mask is not an image; conversions happen only through explicit compatibilities:

- `crop` accepts `box` (a box item carries its crop pixels),
- `pointcloud` ↔ `geometry` are interchangeable,
- `image` accepts `normalmap` and `depthmap` (both render as plain RGB views),
- everything else accepts only itself.

Drops onto incompatible pins show a red ring and are rejected with a run-log entry on the source card — never silently re-routed. The model picker likewise only offers cards that accept the dragged type. One deliberate exception: viewer pins declare a pin-level `accepts` list (`image`, `box`, `mask`, `crop`) so crops and masks can be inspected visually; every other pin stays registry-strict.

### Card anatomy

Action cards show INPUTS, OUTPUTS, then the Result section (Preview / API / Logs tabs, present on every action once it has run). All three sections collapse: pin columns fold to a slim strip of live port dots (wires stay attached and draggable), and the Result section folds to its tab bar. States persist per card across re-renders. Sections are visually distinct (green INPUTS, amber OUTPUTS, blue Result). All three Meta families (Meta AI, Sapiens, Avatar Solver) share one universal color — fb-blue `#0866FF` — on headers, badges, icons, menu entries, and their exclusive pin types (`box`/`boxes`, `normalmap`); there is no purple left on anything Meta. Other cards take their type color as an accent. An action's header band reads its run outcome at a glance: green verified pass, red failure, amber demo/unverified, striped amber while running; untouched cards keep the default blue.

Re-running an action deletes its generated downstream assets (transitive cascade) before producing fresh outputs; user-placed cards (viewers, the source photo) survive, and viewers wired from collapsed splits fall back to their own image.

### Photo Viewer cards

`Asset: Photo Viewer` (right-click canvas → Photo Viewer, or drag any image/crop/box/mask pin onto empty canvas and pick it) drops a live view anywhere mid-flow: wire a pin in and the card shows those pixels at their true aspect ratio, with a `← <pin>` badge and a `Viewing` status. Its output passes the pixels through, so SAM stays transparent when a viewer sits between photo and model. The source photo card carries the same optional `View` input, so it can double as a viewer too. Running SAM with no real input image (e.g. from an unwired viewer) fails fast with a remedy instead of calling the backend.

### Sapiens stages (live local GPU)

The four Sapiens2 cards run the real 0.4B models on local CUDA via `POST /api/debug/sapiens-<pose|seg|normal|depth` (180 s timeout, one model resident at a time):

- **Pose**: 308 keypoints + overlay render; output badges 308/238/70 pts. The card draws a custom SVG skeleton (65 links from the true Sapiens2 308 topology) over score-faded dots — green body, magenta face — with hover tooltips naming each point and its score.
- **Seg**: 28-class part overlay + per-class pixel counts; parts badge shows live class count. Below the overlay, legend chips carry each present class's color, pixel count and frame share; clicking a chip isolates that class on a canvas re-render, and hovering the overlay names the class under the cursor. A live seg run also spawns one asset card per detected foreground class (background dropped, biggest first, capped at 12 with the remainder noted) — each shows its isolated-mask thumbnail with pixel count and frame share, and wires onward from its image pin; re-runs clear prior spawns through the usual cascade.
- **Normal**: person-masked normal map render.
- **Depth**: person-masked metric depth render; depth badge shows the live metre range.

Each card takes one image (photo, crop, box or mask — the old crop-only pins are gone), shows the backend's overlay render and summary in Result → Preview, and carries hop-1-only traces labelled `Local GPU compute — no upstream API call`. If a body-scan run holds the GPU the backend answers `409` and the card fails with the remedy instead of queueing. The `308 Keypoints` pins into Normal/Depth document the intended flow but the backend currently re-derives what it needs from the image alone. Offline runs reuse the mock asset producer and are flagged `Demo (Offline)`.

### Body fit + exporter (live local GPU/CPU)

**SAM 3D Body & Joint Solver** takes one image and runs the full local chain from `sam3dbody-ingest` — Sapiens2 308 auto box, 0.4b seg/normal/depth, 3DB fit, identity refine, 300-iteration joint solve, mesh sheets — via `POST /api/debug/body-fit` (30 min timeout, one model resident at a time). The 2×2 mesh sheet lands in Result → Preview, the `out-mesh`/`out-beta` badges carry the real vert/coeff counts, and the fitted OBJ ref rides the mesh pin downstream. A live fit also spawns an `Asset: Parametric 3D Avatar` card showing the server-side turntable (`turntable.jpg` + `mesh_render.png`, Open3D smooth when importable else the numpy painter, via `sam3dbody-ingest/render_sam3body.py`); the mesh ref passes through the asset, so the exporter wires from either card. **Character Exporter** packages a wired mesh into `avatar_character.glb` (trimesh, CPU) via `POST /api/debug/export`, collecting sibling `identity_joint.json` / `fit_stats.json` when the mesh came from a body fit. Every mesh result renders in an interactive 3D stage (vendored three.js r186, orbit/zoom, render-on-demand): the export card shows its GLB, the body-fit card its fitted OBJ below the sheet, and the avatar asset card the OBJ in place of the turntable poster. The poster always covers loading and failure states, so a stage never shows a black box. Every stage carries a ⬇ download pill (top-right) that saves the underlying GLB/OBJ file. Four server render cards sit after the exporter in the menu — **Mesh Renderer** (shaded 512px still), **Mesh Depth** (near-white depth map), **Mesh Normals** (view-space normal map), **Multi-View Renderer** (2/4/6/8 shaded views at even azimuths composed into one labelled contact sheet, count picked on-card) — all rendered by `POST /api/debug/mesh-render` from any wired mesh (body-fit, avatar, exporter) via the open3d offscreen painter (numpy fallback when open3d is missing, named in the response), CPU call with no GPU slot; failures fail loud instead of demoing. A render card with no mesh says exactly why: it names the upstream card and distinguishes demo data (run it live), an unrun card (run it first), and a pass without a mesh ref (stale server — restart and re-run upstream). **Action: Fit Overlay** (same menu group) takes the photo plus the body-fit (or avatar) card — which now carries the fit ref through — and returns three separate frames on `out-overlay` / `out-photo` / `out-render` (blend, fit-frame photo, matched render), all wireable onward (e.g. into Image Difference). Both carry hop-1-only traces; concurrent GPU debug calls get `409` (one call at a time — no OOM), and offline runs fall back to flagged demo data like every other card. Nothing on the canvas fakes a pass anymore.

### Live progress on long jobs

Sapiens and body-fit runs stream real stage events while they work — no more bare `Running...` for 80 seconds. The backend appends `{t, cid, stage, msg, frac}` lines to a chat-level `progress.jsonl` (dual-written to the call's debug dir as the permanent record); the viewer polls it every 2 s, drives a determinate bar plus the current stage label on the card, and folds each stage into the run log. Concurrent runs multiplex by client id; after the call resolves, the viewer backfills the per-call record so the history is complete even when live polls missed. Single-call cards (SAM, Spark, Muse, describe) stay quiet mid-flight — one POST has no honest intermediate state. Progress is best-effort throughout: a missed poll never breaks a run.

Traversal controls: `⏭ Step Forward` (run next ready action, upstream-first), `▶ Auto Traverse` (run every ready action sequentially upstream-first, stopping at the first failure — Sapiens stages never collide on the GPU), `⏮ Reset`, `🪄 Tidy Layout` (topological columns), right-click canvas for the model palette with search. Wires delete with Alt+click (immediate) or right-click on the wire (menu names the route, then deletes); viewers left unwired fall back to their own image. The header's 🧩 Nodes button toggles the docked library sidebar instead (open by default): every placeable card grouped by family (Meta AI, Sapiens, Avatar Solver, Mesh Render, Utilities, Assets) with counts and a search box — including the Source Photo card, which the right-click menu never offered — followed by the full wire/asset type reference (all 18 pin types with canonical colors, accepts rules, and which cards make and take each). Clicking a tile drops the card at the viewport center. Apply Mask lives outside the Meta menu group (Local Tools): it is a local crop, not a Meta call. Scroll zooms toward the cursor except over scrollable card content (logs, legends, traces), where the wheel scrolls the pane instead — pinch-zoom always zooms; the `+`/`−` buttons zoom toward the viewport center. Wires show run state: idle gray while untouched, amber animated while their source card executes, solid green on a verified pass, solid orange for anything holding unverified data (demo fills, cached or viewed artifacts), solid red when it fails. Card dots follow the same scale. A failed card shows its error on the card; when the backend answered health at boot, a dropped request fails loud instead of demoing (standalone/never-reached backends keep the quiet demo path). One GPU guest at a time: debug calls and body-scan runs share a single lock, so the second starter gets a clean 409 with a remedy instead of a second model load.

### Saved flows & asset chronology

`🗂 Flows` (header) saves the current graph — nodes, edges, positions, run state — to browser localStorage under a name; reopen any saved flow later from the same panel. Photos embedded as `blob:` URLs are inlined as data-URLs on save (blob URLs die with the session; native base64, images downscaled to 2048px so phone-photo saves take ms); backend artifact URLs persist as-is, and card aspect ratios restore from stored dimensions with no re-probing. Quota failures surface in the panel instead of silently dropping. The inspector's fourth tab, **Assets**, lists every asset card in the flow oldest-first with thumbnails (never cropped), badges and timestamps; clicking a row selects the card on canvas.

Asset cards are photo-dominant: the image fills the card at its true aspect ratio, single-wire assets expose one unlabeled edge port (drag off it to wire the next model, or drop the wire on empty canvas for the model picker), and full metadata lives in the step inspector. Assets with several outputs keep labeled pins so the outputs stay distinguishable.

Dragging an image off a card onto the canvas spawns a new Image asset card carrying that image (additive — the existing graph is untouched); drops outside the canvas no longer navigate the tab away. Leaving the page with unsaved work triggers the browser's leave-confirmation (the untouched preset is exempt — it reloads identically).

---

## ReUI components

`components/reui/*.tsx` holds React/TypeScript reference implementations of the ReUI primitives (`Frame`, `Stepper`, `Timeline`, `Badge`, `IconTile`, `CodeBlock`) for reuse in React/Vite hosts. The canvas app itself (`index.html`, `reui.css`, `js/`) is vanilla and uses the ReUI tokens/badges/icon-tiles directly — it does not mount the TSX components.

---

## How to run

### Method 1: served by `chat_server.py` (recommended)

From the **repo root**:

```powershell
. .\scripts\env.ps1
.\.venv\Scripts\python.exe pivot\chat-wrapper\chat_server.py --port 8150
```

Open `http://127.0.0.1:8150/flow/` (redirects from `/flow`). Without the backend, the viewer loads with demo data and `Backend: Offline (Demo)` in the header.

### Method 2: standalone static server

ES-module scripts are blocked on `file://`, so serve the folder instead of double-clicking:

```powershell
cd pivot\chat-wrapper\flow-viewer
python -m http.server 8100
```

Open `http://127.0.0.1:8100/index.html`. All backend calls fail over to labelled demo data.

### Tests

```powershell
cd pivot\chat-wrapper\flow-viewer
node tests/util.test.mjs; node tests/execution.test.mjs; node tests/trace-view.test.mjs; node tests/node-renderer.test.mjs; node tests/canvas.test.mjs; node tests/persist.test.mjs; node tests/context-menu.test.mjs; node tests/traverse.test.mjs; node tests/popover.test.mjs; node tests/bodyfit.test.mjs; node tests/app.test.mjs; node tests/wires.test.mjs; node tests/describe.test.mjs; node tests/progress.test.mjs; node tests/overlays.test.mjs; node tests/meshview.test.mjs; node tests/gallery.test.mjs
```

Covers the shared helpers (`js/util.js`: types, wire checks, viewer resolution), the pure execution logic (`js/execution.js`: list derivation, split pins, refs, re-run cascade, mock-producer honesty, run completion), the transparency builders (`js/trace-view.js`: trace/log rendering and escaping), pure renderer helpers (`js/node-renderer.js`: status dots, viewer badges, overlay labels), zoom anchoring (`js/canvas.js`), flow persistence (`js/persist.js`: snapshots, blob inlining, downscale cap, chronology), node duplication (`js/context-menu.js`: deep clone, run-state reset), traversal order (`orderReadyActions`), the picker filter (`compatibleModels`), body-fit/export fills (`fillBodyFitResultView`, `fillExportResultView`, `upstreamMeshUrl`), the pristine-preset check behind run switching (`isPristinePresetFlow`), wire status classes and deletion (`edgeStatusClass`, `deleteEdge`, `renderWireMenuHtml`), describe fills, mask-ref reads plus dispatch routing (`fillDescribeResultView`, `upstreamMaskRef`, `runCard`), live progress parsing/folding (`parseProgressEvents`, `filterProgressEvents`, `applyProgressEvents`), Sapiens displays (`js/overlays.js`: pose SVG jobs, seg legend/isolate math, spawn rows), the 3D stage view-model (`js/mesh-view.js`: mount rules, download names, depth stretch), scroll-vs-zoom hit-testing (`wheelHitScrollable`), header outcome bands (`headerStatusClass`), seg class-card spawning (`buildSegClassNode`, `spawnSegClassCards`), local mesh renders (`upstreamMeshRef`, `fillMeshRenderResultView`, `executeMeshRender`, `meshInputEdge`, `meshInputDiagnosis`, `executeMeshMultiview`, `viewDirForAzimuth`, `multiviewAzimuths`, `multiviewGrid`, `composeMultiviewSheet`), image diffing (`fitDims`, `absDiffStats`, `fillImageDiffResultView`, `executeImageDiff`), fit overlays (`upstreamFitCamera`, `upstreamFitDiagnosis`, `fillFitOverlayResultView`, `executeFitOverlay`), and the node sidebar (`groupCatalogEntries`, `filterGalleryGroups`, `renderGalleryHtml`, `gallerySpawnCenter`, `assetTypeIndex`, `filterAssetTypes`, `renderAssetTypesHtml`). No framework — stdlib `node:test` only. Backend smoke scripts live next to the server (`../smoke_*.py`, run against a live instance).

---

## Directory structure

```
pivot/chat-wrapper/flow-viewer/
├── index.html            # Canvas viewport, header bar, inspector drawer
├── reui.css              # ReUI tokens + flow-2 graph styles
├── js/
│   ├── app.js            # App shell: presets, health/runs, photo-dim probing
│   ├── util.js           # escapeHtml, uid, svgPlaceholder, fetchJson, probeImageSize
│   ├── catalog.js        # Model registry: pins, types, latency, residency
│   ├── canvas.js         # Pan/zoom, drag-and-drop ingest, tidy layout
│   ├── node-renderer.js  # Node cards, blueprint pins, detection overlays
│   ├── wires.js          # Wire drag, bezier routing, edge updates
│   ├── execution.js      # Run engine, live SAM detect, output synthesis
│   ├── popover.js        # Downstream model picker
│   ├── context-menu.js   # Right-click palette (models + card actions)
│   ├── inspector.js      # Step inspector tabs (visuals / assets / hardware / API)
│   ├── trace-view.js     # Transparency HTML builders (API trace, run log)
│   ├── persist.js        # Flow snapshots, blob inlining, asset chronology
│   ├── overlays.js       # Pose skeleton SVG, seg legend/isolate/hover (pure + bind)
│   └── mesh-view.js      # Interactive 3D stage (lazy three.js, render-on-demand)
├── tests/
│   ├── util.test.mjs          # Regression tests for js/util.js (node:test)
│   ├── execution.test.mjs     # Regression tests for list/split/cascade/mock/completion logic
│   ├── trace-view.test.mjs    # Regression tests for trace/log builders
│   ├── node-renderer.test.mjs # Regression tests for dots/badges/overlay labels
│   ├── canvas.test.mjs        # Regression tests for zoom anchoring
│   ├── persist.test.mjs       # Regression tests for flow save/load helpers
│   ├── context-menu.test.mjs  # Regression tests for node duplication
│   ├── traverse.test.mjs      # Regression tests for traversal order
│   ├── popover.test.mjs       # Regression tests for the picker filter
│   ├── bodyfit.test.mjs       # Regression tests for body-fit/export fills
│   ├── app.test.mjs           # Regression tests for the pristine-preset check
│   ├── wires.test.mjs         # Regression tests for wire status classes
│   ├── describe.test.mjs      # Regression tests for describe fills + routing
│   ├── progress.test.mjs      # Regression tests for live progress helpers
│   ├── overlays.test.mjs      # Regression tests for pose/seg displays + fills
│   └── meshview.test.mjs      # Regression tests for the 3D stage + vendor contract
├── data/
│   └── pose_topology.json  # Sapiens2 308 names/links/groups (derived, attributed)
├── vendor/
│   └── three/              # Pinned three.js r186 (owner-approved exception)
├── README.md
└── components/
    └── reui/             # React/TSX ReUI reference components (optional)
        ├── badge.tsx
        ├── code-block.tsx
        ├── frame.tsx
        ├── icon-tile.tsx
        ├── stepper.tsx
        └── timeline.tsx
```
