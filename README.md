# NeoHuman — one photo, a parametric character you can also hear

> The project formerly known as **GNM-V**. This repository contains only the post-pivot work: the
> single-photo avatar pipeline and the node-graph demo that composes it from Meta's models.

Turn one photograph into a metric, animation-ready character:

- **a parametric MHR head/body** fitted with 45 identity + 72 expression coefficients,
- **a canonical A-pose mesh**, a 6-view turnaround, and a **textured GLB** (single 1024² UV atlas),
- **a node-graph demo** where every model invocation is its own card, labelled `API` or `LOCAL`,
- **photoreal, directed character states** generated with Meta Image (neutral, sad, hat-off, chained sad→bald),
- **measured multi-view validation** instead of adjectives.

This repository contains only the post-pivot work: `pivot/sam3dbody-ingest` (the per-image pipeline) and
`pivot/chat-wrapper` (local server + flow viewer). Third-party model repositories and weights are **not**
committed; you fetch them once and point the pipeline at them (see *Models & weights*).

---

## Quickstart

```powershell
# 1. Get this repo
git clone https://github.com/SergeHendrickx/NeoHuman.git NeoHuman
cd NeoHuman

# 2. Fetch the models (one-time; see table below) into e.g. A:\models\
#    and download their weights.

# 3. Configure paths
$env:SAM3DBODY_ROOT   = "A:\models\sam-3d-body"
$env:SAPIENS2_ROOT    = "A:\models\sapiens2"
$env:SAPIENS2_WEIGHTS = "A:\models\weights\Sapiens2"
$env:VGGT_OMEGA_ROOT  = "A:\models\vggt-omega"

# 3b. One-time: fetch the pinned three.js for the 3D viewers (third-party JS is not committed)
python pivot\chat-wrapper\flow-viewer\fetch_vendor.py

# 4. (Optional) Meta Image / Spark key: plain text file with your API key
#    .cache\meta-api-key.txt

# 5. Run the flow viewer
python pivot\chat-wrapper\chat_server.py --port 8150 --host 0.0.0.0
# open http://127.0.0.1:8150/flow
```

On `/flow` the **prepared graph** loads automatically. Drop your photo on *Asset: Source Photo*
(or use the demo photo), then press **Auto Traverse**. The graph runs Sapiens2 → SAM 3D Body → HyDen
depth → texture bake → A-pose → Meta Image turnaround/edits → VGGT, and writes every artifact under
`pivot/chat-wrapper/chats/flow-session/debug/<id>/`.

If you want to stay fully local/offline, delete the four Meta Image cards (or don't press them) — every
other card runs on your GPU.

---

## Requirements

| | |
| :-- | :-- |
| OS | Windows 10/11 or Linux (developed on Windows 11) |
| Python | **3.11** |
| GPU | NVIDIA, **≥ 12 GB VRAM** recommended (we run a 24 GB RTX 4090 with ~4–5 GB peak per stage) |
| CUDA | CUDA 12.x build of PyTorch (we use `torch 2.11.0+cu128`) |
| Disk | model weights ~10–15 GB, plus per-run outputs |

Python packages (tested versions): `torch`, `torchvision`, `numpy 2.x`, `opencv-python`, `trimesh 5.x`,
`scipy`, `open3d`, `pillow`, `pyyaml`. Install the upstream repos' own requirements as well —
they are declared in their repositories, not vendored here.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# install torch matching your CUDA first, then:
.\.venv\Scripts\python.exe -m pip install numpy opencv-python trimesh scipy open3d pillow pyyaml
```

---

## Models & weights (one-time; not committed)

| Model | Role | Where to get it | Env var |
| :-- | :-- | :-- | :-- |
| **SAM 3D Body** + **MHR** | parametric identity/pose/expression solver | Meta `facebookresearch/sam-3d-body` repo and its gated checkpoints (`model.ckpt`, `mhr_model.pt`) | `SAM3DBODY_ROOT` |
| **Sapiens2** (0.4b / 1b) | 308-keypoint supervision, segmentation, normals, metric pointmap, matting | Meta `facebookresearch/sapiens2` code + `facebook/sapiens2` weights | `SAPIENS2_ROOT`, `SAPIENS2_WEIGHTS` |
| **VGGT-Ω** | multi-view cameras/depth, fused point cloud, consistency check | Meta `facebook/vggt-omega` | `VGGT_OMEGA_ROOT` (checkpoint defaults to `<root>/checkpoints/vggt_omega_1b_512.pt`) |
| **HyDen-MoGeV2** | metric point maps for depth refinement (optional refine step) | Meta `facebook/hyden-mogev2-metric-point` (FAIR noncommercial research licence) | see `pivot/sam3dbody-ingest/scripts/hyden_refine.py` |

Expected layout (the defaults, if you skip the env vars):

```
pivot/sam-3d-body/sam-3d-body-dinov3/checkpoints/model.ckpt     + assets/mhr_model.pt
pivot/sapiens2/sapiens/...                                      (Sapiens2 code)
<weights>/pose|seg|normal|pointmap|matting/...                  (Sapiens2 safetensors)
pivot/vggt-omega/checkpoints/vggt_omega_1b_512.pt               (VGGT-Ω code + weights)
```

**Meta Image / Muse Spark:** put your Meta AI API key in `.cache/meta-api-key.txt` (one line, no quotes).
The graph uses `muse-image-1.0`. Calls are made by the server, one per card run; the prepared graph
makes five per full traverse.

> All model licences belong to their upstream projects (several are FAIR non-commercial research
> licences and/or gated). This repository contains **our** code only; no third-party code or weights are
> committed.

---

## Interactive avatar studio (MHR web viewer)

`pivot/mhr-web-viewer/` is the browser studio that gave the project its name: identity/expression/pose
sliders on the Momentum Human Rig, eye and mouth modules, Mixamo/BVH clip playback, and GPU crowd
benchmarks. It is a Flask + three.js app (source only; `node_modules` and demo media are not committed).

```powershell
cd pivot\mhr-web-viewer
npm install
pip install -r requirements.txt
python server.py                 # opens http://127.0.0.1:8891
python server.py --model C:\path\to\mhr_model.pt --port 8892 --no-browser
```

It needs the MHR release assets (`assets/mhr_model.pt`, `assets/lod0-6.fbx`) from the upstream MHR
checkout; pass `--model` if they live elsewhere. Optional content folders (`data/mixamo`, `data/360hdri`,
`data/makehuman`) are not committed.

## Running the pipeline without the UI

```powershell
.\.venv\Scripts\python.exe pivot\sam3dbody-ingest\run_all.py --input input-images --out outputs
```

Per image this writes, under `outputs/<image-stem>/`:

```
input/                original photo
sapiens/              pose keypoints, seg, normals, depth/matting (+ raw arrays)
threedb/              fit mesh, identity + joint solve, A-pose, texture bake, identity_*.json
stats.json, log.txt   per-stage timings, scores, parameters, file manifest
```

The flow viewer writes the same content per card run to `pivot/chat-wrapper/chats/flow-session/debug/<id>/`.

---

## The prepared graph, card by card

| Stage | Cards | What to look for |
| :-- | :-- | :-- |
| 1 Input | Source Photo | one private-person photo; drop yours here |
| 2 Supervision | SapiensPose / Seg / Normal / Depth | `LOCAL` blue+yellow cards, 308 keypoints, metric depth |
| 3 Solver ladder | HMR Baseline → +Sapiens2 (neutral) → +Sapiens2+Expression | three `LOCAL` solver cards, increasing supervision |
| 4 Final asset | Texture Bake (4× card) | textured GLB + UV texture + 6-view sheet |
| 5 Validate/export | Canonical A-Pose → Meta Image 6-view → VGGT → Exporter | A-pose sheet, photoreal turnaround, consistency mm, GLB |
| 6 Variants | hat-off / neutral / sad / sad→bald | `API` blue cards, one Meta Image call each |

Colour semantics: **blue** = Meta cloud API, **blue edge + yellow inner** = local Meta open model,
**violet** = local render/utility (not a model).

### Reading the numbers

- The solver cards print identity/expression magnitudes, focal length, pose score, depth range, and
  vertex counts; the baseline card prints the raw HMR fit for comparison.
- VGGT prints a **cross-view consistency median/best/worst in mm** and exports a point cloud
  (`mhr_match`/`vggt_cloud` PLY) plus report JSON.
- Card **Links** underneath each preview open full-size images, PLY, OBJ and JSON. Double-click a
  preview for a full-screen image.

---

## Troubleshooting

| Symptom | Cause / fix |
| :-- | :-- |
| `Run failed — unknown debug action` | old server process on the same port; kill it and start one fresh |
| Card stuck `Running…` on the old page | hard refresh (Ctrl+Shift+R); the page marks lost responses failed after 3 min |
| `missing, unresolvable or oversize image` | old backend or a relative image URL; use the server's own artifact URLs |
| CUDA out of memory | only one model is resident per stage; close other GPU apps, restart the server |
| Muse `content_policy_violation` | Meta refuses public figures / identity-transfer prompts; use a private-person photo and character-creation wording |
| Face/waist framing shows legs | server too old; the fixed build crops from the body render |
| Blank/refused connection | check the port (`netstat -ano | findstr :8150`) and the server log |

---

## Repository layout

```
pivot/
  sam3dbody-ingest/          # the per-image pipeline
    run_all.py               # Sapiens2 + SAM 3D Body stages, per-image outputs
    joint_solve.py           # identity/pose/scale/expression solver
    eval_joint.py            # independent evaluator
    scripts/                 # analysis + bake scripts (texture, A-pose, VGGT, HyDen)
  chat-wrapper/              # local server + node-graph demo
    chat_server.py           # HTTP server: model actions, runs, artifact serving
    avatar_extra.py          # VGGT multi-view check, A-pose export
    texture_bake.py          # photo → MHR-UV texture + textured GLB + turnaround
    mhr_multiview_match.py   # per-panel MHR fits aligned by head landmarks
    flow-viewer/             # the browser node editor (vanilla JS, no build step)
      data/prepared_flow.json# the prepared demo graph
```

Generated media, point clouds, GLB/PLY exports, per-run outputs, API keys and third-party clones are
git-ignored; the scripts regenerate them.

---

## Attribution & framing

This is a **character-creation** tool, built for the Meta AI Developer Hackathon. It is not a medical,
diagnostic, demographic or "personality analysis" product, and it is not intended to infer anything
about a real person from a photo. Use consented photos of private individuals (or synthetic characters).

The pipeline composes Meta's models — Sapiens2, SAM 3D Body/MHR, HyDen-MoGeV2, VGGT-Ω, Muse Image —
with our own solver, exporters, texture baker and flow viewer. See `pivot/chat-wrapper/README.md` and
`pivot/sam3dbody-ingest/README.md` for the deeper engineering notes; the model-level contract and
measurements live there too.
