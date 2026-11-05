# SAM 3D Body ingest + SAM 3D Body × Sapiens2 notebook

**Location:** `A:\dev\GNM-V\pivot\sam3dbody-ingest\` — everything this ingest produced lives in this
folder (the session workspace). `pivot/` is gitignored, so nothing here enters the repo.

**Relocation note.** The first pass wrote its result into the repo root (`kb/sam-3d-body.md`, the
`kb/README.md` index row, a `STATUS.md` section, two `LOG.md` lines, and
`trackA-core/out/sam3dbody-ingest/`). The user's rule is that outputs stay in the workspace only, so
those outside writes were reverted/removed and the content was brought here — the knowledge entry is
[`kb-sam-3d-body.md`](kb-sam-3d-body.md), the evidence is the reports below, and the repo-root files
are back at their pre-session state (`kb/sam-3d-body.md` restored from this session's recorded copy of
its 2026-10-31 text, `trackA-core/out/sam3dbody-ingest/` deleted).

## What was done

The user asked to **ingest `pivot/sam-3d-body`** (2026-11-03, after the real code repo and its weights
appeared on disk), then to **provide an `.ipynb` that runs on an environment where SAM 3D Body and
Sapiens2 both work**. Three parallel workstreams read the code/weights read-only and measured the
model in the shared `.venv` (Python 3.11.14, torch 2.11.0+cu128, RTX 4090); one notebook was built and
executed headlessly.

## Ten-line answer

SAM 3D Body's code repo (`pivot/sam-3d-body`, HEAD `b5c765a`) plus its dinov3 weights (2,109,129,346 B)
run in `.venv`: load 8.2–8.6 s, warm inference 0.17–0.22 s (`body`) / 0.72–0.78 s (`full`), 3.5–3.7 GB
GPU. It returns an 18,439-vert MHR body in metres, 45 identity coefficients, 204 MHR params, 70
keypoints and a camera; those parameters reproduce the MHR TorchScript to RMS 1e-4 mm. The model's head
**predicts 72 facial-expression coefficients and multiplies them by zero** (`mhr_head.py:316`), and also
zeroes body params 130–132 (`:307`) — so a 3DB mesh always has a neutral jaw. The released 70 keypoints
have no mouth/jaw point (only nose/2 eyes/2 ears), but the checkpoint's `keypoint_mapping` rows 70–307
are **238 dense face landmarks** that the code computes and discards — free to recover with one matmul.
The model dirs moved into the code repo, so three pipeline files that hard-code the old paths are
currently broken. `sam-3d-body-vith` is still LFS pointers; the annotation dataset (12.04 GiB, no
`.git`) is training-only; `sam-3d-objects` has no weights; the licence is the SAM License (no
non-commercial clause).

## Files

| File | What it is |
| :--- | :--- |
| [`kb-sam-3d-body.md`](kb-sam-3d-body.md) | The knowledge entry: conclusions, on-disk state, how to run, the output contract, the face/landmark facts, licence, stale-claims table, open questions. The reason the ingest exists. |
| [`code-contract.md`](code-contract.md) | Forward flow, 519/204 layout, units/frame/camera, `inference_type` semantics, hard-coded CUDA sites, capability limits, the two face routes, top-5 traps — every claim file:line cited, `[measured]` vs `[inference]` marked. |
| [`runtime.md`](runtime.md) | Runtime matrix: timings, determinism, output schema, frame/units check, bbox sensitivity, multi-person, keypoint convention, failure modes, CPU experiment, caveats. |
| [`metadata-objects.md`](metadata-objects.md) | `mhr70` enumeration, decoding of `keypoint_mapping (308, 18566)`, `sam-3d-objects` + dataset state, licence hashes, dependency audit, stale-kb delta. |
| [`landmarks-308.md`](landmarks-308.md) | Full enumerations: the 70 `mhr70` names+indices and all 238 face rows (mouth corners 188/189, 204/205; lips 190–219; chin 77; iris 272–289; pupils 290–307). |
| [`render-3db.md`](render-3db.md) | **How to render a fit**: the two headless backends, the verified camera/frame conventions, measured timings, and the gotchas. |
| [`render_sam3body.py`](render_sam3body.py) | CLI renderer (Open3D offscreen or numpy painter): overlay on the photo, mesh on neutral background, 6-view turntable, JSON provenance. |
| [`sam3d_sapiens2.ipynb`](sam3d_sapiens2.ipynb) | **The long notebook** (source). Both Meta models on one image + the 238 recovered face landmarks vs Sapiens2's 238. |
| [`sam3d_sapiens2_mini.ipynb`](sam3d_sapiens2_mini.ipynb) | **The small notebook** (4 cells, ~90 lines): image → Sapiens2 keypoints → 3DB fit → mesh overlay. Change only `IMAGE`; the person box is derived from the keypoints (no box to edit). Defaults to the uploaded `input.png`. |
| [`outputs/notebook/exec/`](outputs/notebook/exec) | Executed copies with outputs (`sam3d_sapiens2.ipynb`, `sam3d_sapiens2_mini.ipynb`), 0 errors. |
| [`run_notebook.py`](run_notebook.py) | Headless executor (`run_notebook.py [notebook]`): writes `outputs/notebook/exec/` and `outputs/notebook/artifacts/` only. |
| [`joint_solve.py`](joint_solve.py) | Joint identity+pose+scale solver against Sapiens2 keypoints (`solve_joint`, `solve_identity_only`, CLI) |
| [`eval_joint.py`](eval_joint.py) | Independent evaluator for any refine-params JSON (own projection/metrics, acceptance criteria) |
| [`run_all.py`](run_all.py) | Batch runner: Sapiens2 (pose/seg/normal/depth) + 3DB per image, then re-solves the **45 MHR identity** against Sapiens2's face/ear keypoints; writes `outputs/<image>/` |
| [`outputs/joint-test/`](outputs/joint-test) | Solver runs per image: params JSON, metrics, report |
| [`outputs/joint-eval/`](outputs/joint-eval) | Independent evaluation: baseline vs joint, A1–A9 acceptance, comparison renders |
| [`outputs/sap-roundtrip/`](outputs/sap-roundtrip) | Render the solved mesh → run Sapiens2 on the render → compare with Sapiens2 on the photo (independent loop-closure check) |
| [`outputs/geom-roundtrip/`](outputs/geom-roundtrip) | Mesh z-buffer depth/normals vs Sapiens2 pointmap depth and normal map (no texture; per-image scale/offset reported) |
| [`outputs/`](outputs) | Per-image results: `01_original`, `02_sapiens_keypoints`, `03_sapiens_depth`, `04_sapiens_seg`, `05_sapiens_normal`, `06_3db_keypoints`, `07_3db_mesh_keypoints` (2×2 render sheet), `08_head_calibration` (raw vs joint before/after), `09_expression` (mouth close-up: photo | neutral | expression), `identity_joint.json` (identity 45 + model_params 204 + cam_t + expression 72), `stats.json` |
| [`input-images/`](input-images) | Photos the batch runs on — top level plus the nested `1024/` set (13 more portraits); the scan is recursive |
| [`outputs/notebook/artifacts/`](outputs/notebook/artifacts) | Generated notebook JPEGs (`01_…`, `02_…`, `03_…`, `mini_sapiens2.jpg`, `mini_3d_overlay.jpg`, `stats.json`), renderer output, render JSONs. |
| [`outputs/measurements/runtime/`](outputs/measurements/runtime) | Measured JSON (`results.json`, CPU runs) and the scripts that produced them. |
| [`scripts/`](scripts) | Checkpoint / TorchScript / 308-mapping introspection scripts and dumped rows (the `[measured]` evidence). |
| `restore/` | The reverted-session bookkeeping: the pre-session `kb/sam-3d-body.md` text and the script that reverted the outside writes. Kept for provenance; safe to delete. |

## Key numbers (all measured here, conditions in the reports)

| Measurement | Value |
| :--- | ---: |
| 3DB load / warm inference | 8.18–8.64 s / 0.166–0.222 s (`body`), 0.722–0.778 s (`full`) |
| 3DB GPU peak | 3,538 MB after load; 3,611 / 3,640 MB running |
| `pred_vertices` vs MHR TorchScript of (45, 204) | RMS 1e-4 mm, max 2e-4 mm |
| run-to-run reproducibility | ≤1e-6 m vertices (not bitwise) |
| Sapiens2 load / inference | 3.7 s / 0.33 s for 308 keypoints |
| 3DB projected 308 rows 0–69 vs its own `pred_keypoints_2d` | max 0.0002 px (self-check) |
| Agreement on the full-body photo (median) | body 13.8 px · hands 13.0 · face 19.0 · mouth corners 21.7 · chin 24.2 |
| Agreement on the portrait (median) | head 16.2 px · face 18.1 · mouth corners 13.7 · chin 5.3 (body/hands excluded: 3DB body is a prior, 9 % of its 70 in frame) |
| CPU run after neutralising 2 `.cuda()` sites | 13.38 s, \|β\| Δ 0.06 % vs GPU (not impossible, just slow) |
| 238 face landmarks | all within median 6.0 cm of the nose on the neutral mesh |

## Rendering a fit — the short version

The repo's own visualisation (`sam_3d_body/visualization/renderer.py` → `tools/vis_utils.py`) needs
**`pyrender` + a GL context**, neither of which is available here. Rendering does not need it: the fit
is one MHR mesh, and two installed routes do it headlessly.

- **Open3D 0.20 OffscreenRenderer** (recommended): Vulkan on the 4090, works with no display, smooth
  shading, **≈0.2 s/frame** at 2250×1500 (0.98 s first frame incl. engine init).
- **Pure numpy/OpenCV painter**: no GL context at all, flat shading, **0.26 s** at 1024²; works even in
  a locked-down sandbox.

Both draw `pred_vertices + pred_cam_t` (camera-frame metres) + `model.head_pose.faces` through 3DB's
own pinhole camera (`focal_length`, principal point = image centre) — the same projection the notebook
validated to 0.0002 px. **Verified convention:** geometry in 3DB camera coordinates with an Open3D
extrinsic of **identity** lines up with the photo (a naive "flip y/z" extrinsic puts the mesh out of
view). Upstream expects `pred_vertices` *without* `pred_cam_t` because its renderer adds the
translation itself.

```powershell
.\.venv\Scripts\python.exe pivot\sam3dbody-ingest\render_sam3body.py --case dancing
.\.venv\Scripts\python.exe pivot\sam3dbody-ingest\render_sam3body.py --case portrait --backend numpy
```

Writes `artifacts/render_<case>_{overlay.png,mesh.png,turntable.jpg,json}`: overlay on the photo, mesh
on neutral background, 6 yaw views, and provenance (`|identity|`, `max|expr|`, camera, timings).
Full details, including the upstream `pyrender` route and the browser/three.js route, are in
[`render-3db.md`](render-3db.md).

## Batch runs per image (`run_all.py` → `outputs/`)

[`run_all.py`](run_all.py) runs **all four Sapiens2 heads** (pose, body-part seg, surface normals,
pointmap→depth) **and SAM 3D Body** on every image found **recursively** under [`input-images/`](input-images), writing one folder per
image under [`outputs/`](outputs) (the single folder all artefacts live in):

| file | what |
| :--- | :--- |
| `01_original.*` | the original image, untouched |
| `02_sapiens_keypoints.jpg` | original + Sapiens2 308 keypoints: **green = 70 body/foot/hand**, **magenta = 238 face** points (legend is drawn on the image) |
| `03_sapiens_depth.jpg` | Sapiens2 pointmap depth (metric, turbo colour map, person-masked) |
| `04_sapiens_seg.jpg` | Sapiens2 body-part segmentation overlay (29 classes) |
| `05_sapiens_normal.jpg` | Sapiens2 surface normals (RGB) |
| `06_3db_keypoints.jpg` | original + 3DB's 70 keypoints (red) matched 1:1 to Sapiens2's 70 (green) |
| `07_3db_mesh_keypoints.jpg` | **2x2 sheet**: photo-textured 3DB mesh + keypoints over the photo / the render on its own / a textured 3/4 view / a clay 3/4 view |
| `08_head_calibration.jpg` | before/after close-up: raw 3DB fit vs the joint identity+pose+scale solve |
| `09_expression.jpg` | face close-up: photo + Sapiens2 face keypoints / the same solve with neutral jaw / with the fitted expression |
| `identity_joint.json` | joint solve output: original + refined identity(45), `mhr_model_params`(204), `cam_t`, expression(72), metrics |
| `stats.json` | size, person box, pose score, seg classes, depth range, identity norm, focal, timings, joint/expression metrics |

```powershell
.\.venv\Scripts\python.exe pivot\sam3dbody-ingest\run_all.py
```

**VRAM.** Models are loaded **one stage at a time** and freed before the next; the process is capped at
95 % of the 24 GB card and asserts a 24 GB budget after every stage. Measured peaks on the 19 images:
pose **2.1 GB**, seg 2.2 GB, normal 2.7 GB, pointmap/depth **4.0 GB**, 3DB 3.4 GB — i.e. well under
the card, because no two models are ever resident. `stats.json` in this folder lists each image's
status (all 19 came out `ok`), depth range and identity norm; per-image wall time is ~16–22 s
including all heads, the joint solve and the render sheets (~6.5 min for 19 images; the 3DB stage
dominates).

The 3DB person box is derived from Sapiens2's in-frame keypoints, so no manual box is needed.

## Joint identity + pose + scale solve from Sapiens2 keypoints (the matching improvement)

The first pass re-solved only the 45 identity coefficients. It improved the face a lot (error −25…−60 %),
but on several images the identity hit its `|Δβ| = 5` cap: small **pose/scale** errors were being forced
into body shape. So [`joint_solve.py`](joint_solve.py) now re-solves, against Sapiens2's keypoints:

| free variable | size | cap / regularisation |
| :--- | :--- | :--- |
| identity | 45 | \|Δβ\| ≤ 5, coefficients within MHR ±10 |
| body pose euler deltas | 130 | \|Δ\| ≤ 0.2 rad per component, L2 reg |
| 6-D global rotation | 6 | small, L2 reg |
| scale latent | 28 | \|Δ\| ≤ 2, L2 reg |
| camera translation | 3 | \|Δt\| ≤ 10 cm |

Loss: robust Huber on pixel error to Sapiens2 keypoints — skin face rows 70–219 weight 1, **52 ear points**
220–271 weight 3 (the width anchor), in-frame body/foot/hand rows 5–69 weight 0.2. Pose/scale/camera stay
close to 3DB's; identity-only result is the initialisation. 300 iterations, **~6–7 s/image**, runs inside
the 3DB stage (peak VRAM 3.45 GB). Each folder gets `identity_joint.json` (original + refined identity 45,
`mhr_model_params` 204, `cam_t`, metrics); `06`/`07`/`08` use the joint result.

**Independently verified** by a second harness ([`eval_joint.py`](eval_joint.py), no solver code reused,
own landmark/projection implementation): it reproduces the solver's table to ≤0.05 px and its own numbers
are below (masked 2D error vs Sapiens2; n = points actually in frame):

| image | face px (raw → identity-only → **joint**) | ears px (raw → id-only → **joint**) | ear-span error vs Sapiens2 (id-only → **joint**) | body px, n |
| :--- | ---: | ---: | ---: | ---: |
| download | 11.27 → 11.74 → **7.10** | 35.24 → 14.92 → **5.50** | −3.55 % → **−0.68 %** | 28.9 → **19.3** (2) |
| Emiru | 13.14 → 9.02 → **6.67** | 44.04 → 14.57 → **7.39** | −4.00 % → **−1.12 %** | 84.7 → **46.6** (25) |
| file | 11.70 → 12.12 → **7.31** | 38.17 → 15.85 → **5.52** | −3.32 % → **−0.48 %** | 30.1 → **20.4** (2) |
| input | 7.80 → 6.83 → **4.16** | 15.14 → 8.65 → **5.49** | −1.53 % → **−0.49 %** | 21.4 → **16.9** (5) |
| remy | 12.58 → 12.92 → **8.07** | 28.19 → 11.84 → **9.38** | −1.53 % → **−2.00 %** | 37.5 → **21.7** (3) |
| teenage-woman | 12.46 → 12.86 → **8.82** | 51.22 → 11.07 → **7.39** | −0.13 % → **+1.02 %** | 9.1 → — (0) |
| **mean** | **11.49 → 10.92 → 7.02** | **35.33 → 12.82 → 6.78** | **−2.34 % → −0.63 %** | — |
| dancing (full body) | 19.28 → — → **3.06** | 21.16 → — → **3.62** | profile (meaningless) | 14.45 → **10.84** (65) |

Joint beats identity-only on every image: face −36 % aggregate, ears −47 %, ear span now within **2.0 %**
everywhere (identity-only was up to 4.0 %). The capped cases are fixed (Emiru face 9.02→6.67,
teenage-woman 12.86→8.82). On the full-body photo the win is a **0.13 rad neck/head pose correction**
(global rotation moves 0.002 rad), face 19.28→3.06 px and 65 in-frame body landmarks 14.45→10.84 px.
Moves stay conservative: max joint rotation 0.178 rad, scale 0.034, translation 2.0 cm, hands unchanged,
height −0.83 %, all finite — the independent evaluator's criteria A1–A9 all pass.

**Rendering** was improved at the same time: `07` is a 2×2 sheet — photo-textured mesh over the photo
(with 3DB keypoints), the same render on its own, a textured 3/4 view, and a clay 3/4 geometry view.
The photo-texture now uses z-buffer visibility (no bleed from occluders), graph-diffused colours for
back-facing vertices, linear-light two-light shading, and supersampled depth-mask compositing over a
gradient (clean antialiased edges, no background fringe).

**Limits.** Identity+pose+scale is still a 45-dim identity + skeleton fit: it cannot invent face *shape*
outside the MHR basis, expression remains neutral (3DB zeroes it), and the Sapiens2 face rows have no
jaw/chin contour, so jaw shape is not solved. On head-and-shoulders crops only 0–5 body keypoints are in
frame, so body numbers there are noise (`n` is reported); use a full-body photo for body evaluation.

**Expression is fitted as well.** SAM 3D Body's head predicts 72 face-expression coefficients but the
released inference code zeroes them (`mhr_head.py:316`; the jaw eulers are zeroed at `:307` and the pose
is truncated to 130 dims at `:198`, so jaw rotation never reaches the rig); the README scopes 3DB to
body/feet/hands, and there is no expression supervision in the release — so a neutral face is a code
decision, not a model limit. The 72 coefficients are just another input to the differentiable MHR forward,
so the solve frees them too (`|expr|` L2 ≤ 3, L2 reg 0.02 toward neutral, mouth rows 188–219 weighted 3
so the ~100 eye-contour rows do not out-vote the lips). Independently verified: **lips 10.77 → 3.31 px
(−69 %)**, inner-lip aperture error **−8.5 → +0.5 px** mean (max 2.9 px), face 7.02 → **5.22 px**; ears,
body, pose, identity and translation unchanged; `|expr|` L2 0.41–0.86 (cap never binds), max coefficient
0.45, all finite, mesh height change ≤0.43 %. `09_expression.jpg` shows photo + Sapiens2 face keypoints /
the same solve with a neutral jaw / the fitted expression. Caveat: the MHR LOD1 mesh here has no teeth,
tongue or eyeballs, so an open mouth is dark inside — the aperture error ≤2.9 px says it does not gape
beyond Sapiens2's own value, but it will never show teeth.

**Depth and normal check (no texture).** The same solved mesh rendered as a z-buffer (metric depth +
camera normals per pixel) is compared with Sapiens2's pointmap depth and normal map (`outputs/geom-roundtrip/`).
Absolute depth is not comparable — the two monocular conventions differ by a per-image scale/offset
(fitted `sap ≈ a·mesh + b`, a = 0.88–1.27) — but after that similarity, and with a **skin-only mask**
(Sapiens2 seg Face_Neck minus hair/clothing/background/lips/teeth, eroded), the depth residual is
**0.54 cm median / 1.48 cm P90** with a **0.18 cm/100 px** residual tilt term, and skin normals agree at
**9.9° median / 13.2° mean** (frame map `(x, −y, −z)`, no frame rotation to correct: best-fit 2.4° global,
13.57° → 13.68° after applying it). The earlier bbox-based numbers (1.6 cm / 21°) were contaminated by hair
and clothing. Residuals concentrate on structures the LOD1 mesh lacks — eyeballs, nostrils, lips, and the
chin-to-neck crease — not on cheeks/forehead/chin pad, where the mesh agrees.
**Skin-normal loss in the solve.** The joint solve accepts Sapiens2's normal map and adds a Huber
cosine term on visible skin vertices (skin mask minus hair/clothing/lips/teeth and minus eye, nostril and
mouth discs; visibility/correspondences fixed from the initialised fit; weight 0.5). Measured on 4 images:
the supervised angle drops **4–5°** (e.g. Emiru 23.0° → 17.2°, download 22.0° → 17.8°) and the independent
pixel-level error 0.2–0.7° (lower face 0.4–0.7°), with no 2D regression (≤0.05 px face, ≤0.06 px ears) and
expression +2–3 %. The 45-d identity cap (`|Δβ|=5`, binding on 13/19 images) is what limits the gain;
relaxing it to 7 helps one image and hurts another. A jaw **silhouette** term was evaluated and rejected
for this data: only **12.3 %** of the lower-face contour borders background (38.7 % borders hair,
48.7 % neck/torso — an internal boundary the mesh cannot fit). Details and tables:
[`outputs/geom-roundtrip/`](outputs/geom-roundtrip/).

**Round-trip validation.** To check the fit is not just matching the keypoints it was fitted to, the
solved mesh is rendered and Sapiens2 is run on that render (`outputs/sap-roundtrip/`): re-detected face
landmarks land **2.4 px** (photo-textured render) / 7.1 px (clay, no photo colour) from Sapiens2's
landmarks on the photo, ears 6.5 / 7.9 px, and on the full-body photo all 65 in-frame body keypoints
re-detect to **9.0 px**. So the visible surface — and the body pose — is where the photo is, not just where
the loss pushed it; the ~6–8 px ear residual is the remaining head-width/shape disagreement.

Evidence: [`outputs/joint-test/`](outputs/joint-test) (solver runs + JSONs) and [`outputs/joint-eval/`](outputs/joint-eval)
(independent acceptance check + comparison renders).

## Run the notebook## Run the notebook

JupyterLab (kernel **`gnmvenv`**, launch from the repo root so the file is reachable):

```powershell
. .\scripts\env.ps1
$env:JUPYTER_PATH = "A:\dev\GNM-V\.cache\jupyter\share\jupyter"
$env:JUPYTER_RUNTIME_DIR = "$PWD\.cache\jupyter\runtime"
$env:JUPYTER_DATA_DIR = "$PWD\.cache\jupyter\data"
$env:JUPYTER_CONFIG_DIR = "$PWD\.cache\jupyter\config"
$env:IPYTHONDIR = "$PWD\.cache\ipython"
.\.venv\Scripts\jupyter-lab.exe --notebook-dir=. --port=8889 --ip=127.0.0.1 --no-browser
```

Headless (writes only beside the notebook — verified clean; pass a different notebook as the argument):

```powershell
. .\scripts\env.ps1
$env:JUPYTER_PATH = "A:\dev\GNM-V\.cache\jupyter\share\jupyter"   # kernel "gnmvenv"
.\.venv\Scripts\python.exe pivot\sam3dbody-ingest\run_notebook.py                                                    # long notebook
.\.venv\Scripts\python.exe pivot\sam3dbody-ingest\run_notebook.py pivot\sam3dbody-ingest\sam3d_sapiens2_mini.ipynb  # small one
```

No installs, no downloads, no second venv: both models are read in place. The long notebook writes JPEGs +
`stats.json` to `outputs/notebook/artifacts/` and its executed copy to `outputs/notebook/exec/`; the small notebook does the same with
`mini_sapiens2.jpg` / `mini_3d_overlay.jpg`. In the running JupyterLab, open
`sam3dbody-ingest/sam3d_sapiens2_mini.ipynb` and run all cells (~20 s): it displays the input image, the
Sapiens2 308 keypoints, and the SAM 3D Body mesh over the photo. To use your own photo, change `IMAGE`
and `BBOX` in the config cell (both are documented there).

## What did not work / traps (details in the reports)

- `inference_type="hand"` **crashes** through `process_one_image` (`{"mhr": None}` then subscripted).
- Upstream `demo.py` is not runnable here (ViTDet + SAM2 + MoGe2 + `pyrender`, all absent) and has no
  path defaults; the manual-bbox path is the working one.
- 3DB never validates inputs: a zero-area bbox gives **silent NaN**; a background box gives plausible
  garbage. Check `x2>x1`, minimum size, `np.isfinite`, |β| ≤ ~10.
- `full` means body + hands, not more face detail; it costs 3 backbone passes and zeroes `pred_pose_raw`.
- The face cannot be recovered from 3DB: 72 expression coefficients and body params 130–132 are zeroed
  in the head. MHR itself accepts a non-zero 72-vector, so expression must be driven by us.
- The 238 face landmarks are MHR-topology only; they do not transfer to GNM's 17,821-vertex head
  without a correspondence.

## Licence (what a README must say)

SAM License, last updated 2025-11-19, **no non-commercial clause**; all six copies on disk are
byte-identical (8,256 B; the `checkpoints/` copy differs only by CRLF/LF). Redistribution only under
the same agreement with a copy, publication acknowledgment, no reverse-engineering/"discovering
underlying components", trade-control compliance, HF-gated access, Meta may terminate → delete.
Reference upstream by URL, keep weights out of the repo, acknowledge SAM 3D; dataset annotations are
SAM-licensed while the images they point at keep their eight upstream licences.
