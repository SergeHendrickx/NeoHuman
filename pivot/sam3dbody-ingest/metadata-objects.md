# SAM 3D Body (3DB) — metadata, sibling model, data, licence, kb delta

*Measured 2026-11-03 ~01:30 Brussels by teammate `metadata-delta`, read-only, from
`A:\dev\GNM-V\pivot\sam-3d-body` (HEAD `b5c765a`, 2026-02-19) with
`A:\dev\GNM-V\.venv\Scripts\python.exe` (torch 2.11.0 CPU). Commands in §9. Inferences marked **[inf]**, everything
else is a read number or a directly-executed probe. Report is the deliverable; the small scripts + JSON dumps
beside it are the reproducible scan code.*

## Summary — conclusions first

1. **The code repo is on disk and has been executed**: `pivot/sam-3d-body` is the git repo (HEAD `b5c765a`);
   `sam-3d-body-dinov3/`, `-vith/`, `-dataset/`, `sam-3d-objects/` are **real directories inside it**, not links.
   The old `pivot/sam-3d-body-*` paths no longer exist — that is the stale-path delta for `pipeline/avatar/`.
2. **mhr70 has 3 face points and no jaw/mouth/eye-detail at all**: `0 nose, 1 left-eye, 2 right-eye, 3/4 ears`
   then body/hands/feet/neck. **0 jaw, 0 mouth, 0 lip, 0 iris.** 3DB therefore *cannot* supply jaw/mouth landmarks.
3. **The checkpoint's `keypoint_mapping` is (308, 18566) = 308 keypoints as fixed row-stochastic weights over
   18,439 verts + 127 joints**; every row sums to 1. 48 rows are one-hot joints (all in rows 0–69); rows 70–307
   put **zero** weight on joints and are pure vertex blends (5–19 verts, mean 8.7).
4. **Rows 0–69 are exactly `mhr70.py`; rows 70–307 are the 238 non-teeth dense face keypoints** of the Sapiens/
   Sociopticon 308 format (`pivot/sapiens2/sapiens/pose/configs/_base_/keypoints308.py`), in that order. Verified
   structurally (name lists identical for 0–69; 344−36 teeth = 308; face set = 238) and geometrically (§3.3).
5. **Usable for GNM-V anyway**: the 238 face rows give mouth corners 188/189/204/205, 27 lip points 190–219, chin
   77, labiomental groove 76, eyelids/lash lines 96–177, iris 272–289, pupils 290–307. Because the map is a fixed
   linear operator, they can be computed on **any** 18,439-vertex MHR mesh, not only 3DB output.
6. **There is no jaw keypoint in the whole 308 set** (names audited: no `jaw`, no jawline; the only removed points
   are `teeth_1..teeth_36`). Best jaw proxies are chin 77, groove 76, and lower-lip rows 194/200–203/210/216–219.
7. **Independent second reason the jaw is unavailable: 3DB zeroes it.** `mhr_head.py:307` `pred_pose_euler[:, -3:] = 0`
   (comment `### Zero-out jaw`) and `:316` `pred_face = ... * 0`. A 3DB-produced MHR mesh is a neutral-jaw mesh.
8. **`sam-3d-objects` = full clone, `f91db41` (2026-06-02), 0 weights** (`checkpoints/` holds only a `.gitignore`).
   Objects/scene reconstruction with texture; composes with 3DB via `notebook/mesh_alignment.py` (MoGe point-cloud
   scale + pytorch3d). **Verdict: irrelevant to GNM-V** — needs two models not installed (pytorch3d, moge) and
   reconstructs props, not a head.
9. **Dataset is unchanged training data**: 1,546 parquet / 12,932,765,150 B (12.04 GiB), **no `.git`** (not
   resumable), `sa1b_train` 374/1,850 files = 693,919 rows (still incomplete). Images never shipped; `data/`
   scripts expect 8 re-downloaded source datasets under their own licences. Not a runtime dependency.
10. **Licence verdict unchanged**: SAM License, *Last Updated November 19, 2025*, **no non-commercial clause**, all
    six copies on disk identical (root/dataset/dinov3/vith/objects same SHA-256; `checkpoints/LICENSE` differs only
    by CRLF). Conditions that bind us: same-agreement redistribution + licence copy, publication acknowledgment,
    **no reverse-engineering / "discovering underlying components"**, trade-control + no ITAR/military use, Meta may
    terminate → delete, HF gate. Our README must reference upstream by URL, never vendor weights, acknowledge SAM 3D.

## 1. True paths on disk (the layout delta)

| Path (now) | Kind | Verified |
| :--- | :--- | :--- |
| `pivot/sam-3d-body/` | **git repo, HEAD `b5c765a0d89d…`, 2026-02-19 "add arXiv"** | `git rev-parse HEAD` |
| `pivot/sam-3d-body/sam-3d-body-dinov3/` | real dir (HF clone inside code repo), untracked | `Get-Item .LinkType` → none (not a junction/symlink) |
| `…/sam-3d-body-dinov3/checkpoints/model.ckpt` | **2,109,129,346 B** (real) | `Get-ChildItem -Recurse` |
| `…/sam-3d-body-dinov3/checkpoints/assets/mhr_model.pt` | **696,110,248 B** (real) | idem |
| `…/sam-3d-body-vith/` | real dir, **LFS pointers only** (model.ckpt 138 B, assets/mhr_model.pt 137 B, no `checkpoints/`) | idem |
| `…/sam-3d-body-dataset/` | real dir, **no `.git`**; `data/` + LICENSE + README + .gitattributes | `Test-Path .git` → False |
| `…/sam-3d-objects/` | **full clone, HEAD `f91db41`, 2026-06-02**, `.git` present (268.1 MiB) | `git -C … rev-parse HEAD` |
| `pivot/sam-3d-body-dinov3`, `…-vith`, `…-dataset` (old paths) | **ABSENT** (moved, not linked) | `Test-Path` → False |

`git status` in the code repo shows exactly the four nested dirs as `??` — i.e. weights/dataset/object repo are
untracked inside the (gitignored) clone; the repo's own `.gitignore` covers `/checkpoints/` and `*.ckpt`.

**Stale integration paths (must be fixed, not edited by me).** All three point at the moved-away layout:
`pipeline/avatar/from_photo.py:44-45` (`THREEDB_CKPT`, `MHR_MODEL`), `pipeline/avatar/panel_landmarks.py:31`,
`pipeline/avatar/photo_front_head.py:104-105`. Correct values today:

```
pivot/sam-3d-body/sam-3d-body-dinov3/checkpoints/model.ckpt
pivot/sam-3d-body/sam-3d-body-dinov3/checkpoints/assets/mhr_model.pt
```

`from_photo.py:43 THREEDB_REPO = pivot/sam-3d-body` is still correct (that is why the import works).

**Upstream `demo.py` does *not* default to the new layout**: `--image_folder` and `--checkpoint_path` are both
**required** (`demo.py:117-134`), so there is no default to fix — its own epilog example is the relative
`./checkpoints/model.ckpt`. It also builds a ViTDet detector / SAM2 segmentor / MoGe2 FOV head by default
(`--detector_name vitdet`, `--segmentor_name sam2`, `--fov_name moge2`) and needs `--mhr_path`. So the documented
upstream command is only equal to the new layout if you pass
`--checkpoint_path sam-3d-body-dinov3/checkpoints/model.ckpt --mhr_path sam-3d-body-dinov3/checkpoints/assets`.

## 2. `mhr70` — the 70 keypoints (`sam_3d_body/metadata/mhr70.py`)

**All 70 names + indices: [landmarks-308.md §1](landmarks-308.md).** Order: `0 nose, 1 left-eye, 2 right-eye,
3 left-ear, 4 right-ear` → shoulders/elbows/hips/knees/ankles/toes/heels (to 20) → right hand 21–41 →
left hand 42–62 → `63 left-olecranon, 64 right-olecranon, 65 left-cubital-fossa, 66 right-cubital-fossa,
67 left-acromion, 68 right-acromion, 69 neck`.

**Face/jaw/eye/mouth audit of the 70: exactly 5 head-adjacent points — `0 nose`, `1 left_eye`, `2 right_eye`,
`3 left_ear`, `4 right_ear`. Jaw = 0, mouth/lip = 0, eyelid/iris/pupil = 0, teeth/tongue = 0.** Grouped as
declared in `pose_info`: 17 `body_keypoint_names` + 6 `foot_*` + 20 `left_hand_*` + 20 `right_hand_*` +
7 `extra_*` (`neck`, 2 olecranon, 2 cubital fossa, 2 acromion) = 70. Docstring is explicit:
*"The first 70 of 308 MHR keypoints, ignoring the rest for face keypoints"*.

**Name aliases (three vocabularies, same index):** `mhr_names` uses hyphens (`right-thumb-tip`);
`original_keypoint_info` uses underscores (`right_thumb_tip`); `keypoint_info` uses a *different* convention for
hands (`right_thumb4`, `right_forefinger4`, …) with `swap=` partners, and a stale comment at id 45
(`## doesnt match with wholebody`). Indices 15–20 are `left_big_toe` in `keypoint_info` but `left_big_toe_tip` in
`original_keypoint_info`. Use indices, never names.

**Rest of `metadata/__init__.py`** (2.0 KB, no 308 tables): `OPENPOSE_TO_COCO`, `J19_TO_J14`, `LSP_TO_COCO`,
`OPENPOSE_PERMUTATION`, `J19_PERMUTATION`, `COCO_PERMUTATION`, `MHR70_TO_OPENPOSE` (24 entries, 1 missing —
whole-body has an extra nose/neck pair), `MHR70_PERMUTATION` (70-length), `MHR70_TO_LSP`. **The 70→308 relation is
not in metadata/ at all** — it lives only in the checkpoint tensor + `mhr_head.py`.

## 3. `keypoint_mapping` (308, 18439+127) — semantics, and rows 70–307 named

### 3.1 What the tensor is (measured, `_kpscan.py`)

| Property | Value |
| :--- | :--- |
| keys in checkpoint | `head_pose.keypoint_mapping` and `head_pose_hand.keypoint_mapping` (2 copies, same shape) |
| shape / dtype | `(308, 18566)` = `(308, 18439 verts + 127 joints)`, float32 |
| row sum | **all 308 rows sum to 1.000000** (one distinct value) |
| row sparsity | 48 rows one-hot; 260 rows sparse blends (nnz 5–43; face rows nnz 5–19, mean 8.7) |
| weight range | min ≈ −6.8e−14 (float noise), max 1.0 — **no real negative weights**, no extrapolation |
| joint columns touched | 49 rows: the 48 one-hots (rows 5,6,9–12,21–62 …) **+ row 69 = 0.5·joint110 + 0.5·joint111** (neck, blended between the two neck joints) — **all inside rows 0–69** |
| rows 70–307 on joint columns | **0 rows** → pure surface-vertex combinations |
| application (code) | `mhr_head.py:241-251`: `M @ [verts; joint_xyz]` batched over x,y,z, i.e. `kp = M @ vj` with `vj` `(18566,3)` |
| which rows are used at inference | `mhr_head.py:337` and `sam3d_body.py:1614`: `j3d = j3d[:, :70]  # 308 --> 70 keypoints` → **the 238 face rows are computed then discarded** |

### 3.2 Row → name mapping

- **Rows 0–69 = `mhr70`** (identical order, §2).
- **Rows 70–307 = the 238 non-teeth face keypoints of the Sociopticon 308 format**, in format order:
  row = `70 + i`, where *i* indexes `dataset_info["face_keypoint_names"]` **with the 36 `teeth_1..teeth_36`
  entries (format ids 220–255) removed** (344 − 36 = 308; face ids 70–343 → 238).

### 3.3 Evidence that rows 70–307 really are that set

1. **Structural, exact**: the format's first 70 names after teeth removal are *character-identical* in order to
   `mhr70.py` (`['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder',…,'left_acromion',
   'right_acromion','neck']`), and no other 70-keypoint split of the 344/308 set reproduces it.
2. **Geometric (executed)**: neutral MHR TorchScript forward → `M @ [verts;joint_xyz]` gives, in native cm,
   `row0 (nose) = (-0.01, 158.9, 15.5)`, `row1 (left_eye) = (+3.16, 162.3, 12.0)`, `row2 (right_eye) = (−3.14,…)`,
   `row3/4 (ears) = ±7.7`, `row69 (neck) = (0, 146.6, 1.4)`. All 238 rows 70–307 lie within **median 6.0 cm,
   max 16.8 cm** of the nose (i.e. on the head); rows 0–69 span up to 159.8 cm (feet).
   The `l_`/`r_` prefix of each face name predicts the sign of x **176/176 = 100 %** (left = +x; my first test
   used the opposite convention and scored 0/176, which is the same result inverted). 62 of the 238 names carry
   no `l_`/`r_` prefix (midline/contour points).
   *Not proven:* that the correspondence is upstream-intended for every single one of the 238 (a 100 % left/right
   sign match plus a monotone head-local cluster makes an accidental permutation implausible, but I did not
   cross-check against Sapiens2's own 308 predictions on an image).

### 3.4 The 238 face rows — category → rows

| category | n | rows |
| :--- | ---: | :--- |
| nose (bridge/root/tip/nostrils/wings) | 9 (+1 tip) | 71–75, 178–187 |
| eyebrows | 18 | 78–95 |
| eyes: lash/lid/crease lines | 32 | 96–177 (left 96–119, 144–160; right 120–143, 161–177) |
| mouth + lips (both outer and inner) | 31 | 188–219 |
| chin / labiomental groove | 2 | 76, 77 |
| ears (helix/tragus/lobe/concha…) | 52 | 220–271 |
| iris | 18 | 272–289 |
| pupil | 18 | 290–307 |

**The load-bearing subset for a jaw/mouth rig** (mouth corners, lip mids, chin):

```
jaw proxy (best available):  77 tip_of_chin · 76 center_of_labiomental_groove
mouth corners:              188 r_outer_corner_of_mouth · 189 l_outer_corner_of_mouth
                            204 r_inner_corner_of_mouth · 205 l_inner_corner_of_mouth · 190 center_of_cupid_bow
upper outer lip:            192,193,196,197,198,199 midpoints · 206/207 center inner lips
lower outer lip:            191 center · 194,195,200,201,202,203 midpoints
upper inner lip:            208,209,212,213,214,215 · lower inner lip: 210,211,216,217,218,219
```

Full 238-row enumeration: **[landmarks-308.md §2](landmarks-308.md)** (machine-readable in
`_sapiens308.json` / `_face238.txt`). Load-bearing subset repeated here:

### 3.5 What this buys GNM-V, and what it does not

- **Buys**: 238 stable, named face landmarks on *any* MHR mesh at zero cost —
  `kp238 = M[70:] @ MHR_verts` (one matmul, no weights, no network). This is exactly why the texture pass already
  used "308 keypoints, 238 face points" (LOG 2026-10-31 21:45). For the jaw↔vowel seam it supplies mouth corners,
  both lip loops, chin and the groove — i.e. mouth **aperture/shape** measurement, not a jaw hinge.
- **Does not buy**: a jaw landmark (none exists in the 308), or a *predicted* dense face — 3DB discards the 238
  rows, so on a photo the only face evidence is the 3 70-set points.
- **Inference**: mapping to MHR vertices is fixed, so the 238 landmarks are only as good as the MHR mesh they are
  applied to; on GNM's 17,821-vertex head topology the map is **not** applicable (different vertex count/order).

## 4. `sam-3d-objects` (sibling model, nested clone)

| Item | State |
| :--- | :--- |
| clone | full git clone, HEAD `f91db41` "SA-3DAO Release - README Update (#190)", 2026-06-02; `.git` 268.1 MiB |
| weights | **none**: `checkpoints/` contains only `.gitignore` (14 B); no `*.ckpt/pt/pth/safetensors/bin` anywhere under it |
| size on disk | total ~481 MiB, of which `.git` 268 + `notebook/` 213 (image assets) + code 0.7 MiB |
| what it is | "SAM 3D Objects": image + mask → 3D object **shape, texture, layout** (props/scenes), separate model from 3DB; same SAM License |
| link to 3DB | `notebook/demo_3db_mesh_alignment.ipynb` + `notebook/mesh_alignment.py` (18.6 KB): "SAM 3D Body (3DB) Mesh Alignment Utilities — Handles alignment of 3DB meshes to SAM 3D Object, same as MoGe point cloud scale". Imports `pytorch3d` (`Meshes`, rasterizer) and `moge.model.v1.MoGeModel`; notebook calls `load_3db_mesh`, needs both a completed object reconstruction and a 3DB mesh |
| runnable here? | **no** — `pytorch3d` and `moge` are both absent from `.venv` (find_spec False), and no object weights are on disk |
| GNM-V verdict | **irrelevant**: it reconstructs object props for scene composition; no head/face capability, no weights, needs 2 missing deps. Ignore for the demo; only cite as context if the pitch wants "Meta SAM 3D family". |

## 5. `data/` scripts and what is actually on disk

**`data/README.md`** requires a *separate* conda env (py 3.9, torch 2.4, hdbscan, pyntcloud, pykalman,
torchgeometry, pycolmap, projectaria-tools, webdataset, pycocotools…), **HF-gated** annotation access to
`facebook/sam-3d-body-dataset` (rejected in sanctioned jurisdictions), `SAM3D_BODY_ANN_DIR` /
`SAM3D_BODY_WDS_DIR` env vars, and image trees prepared per source dataset under
`data/scripts/{3dpw,aic,coco,egoexo4d,egohumans,harmony4d,mpii,sa1b}/README.md`. `scripts/download.py` = HF
`hf_hub_download` per split (overwrites local dir, no resume); `scripts/create_webdataset.py` joins annotations
to images on disk. Eight source datasets, each under its own licence → **the annotations cannot be used
standalone**, and nothing here is a runtime dependency of inference.

**On disk now** (`Get-ChildItem -Filter *.parquet`, sizes summed):

| | measured |
| :--- | :--- |
| parquet files / bytes | **1,546 files / 12,932,765,150 B = 12.04 GiB** |
| `.git` present | **no** (deleted 2026-10-31 18:34) → not resumable |
| complete splits (11) | unchanged: 3dpw_train 18/22,405 rows · aic_train 173/287,391 · coco_train 24/**39,665** · egoexo4d_physical_train 400/736,850 · egoexo4d_procedure_train 200/348,359 · egoexo4d_*_test 20/36.7 MiB · egohumans_train 125/271,944 · harmony4d_train 200/496,036 · harmony4d_test 6/11,190 · **mpii_train 6/7,135** |
| incomplete split | **`sa1b_train` 374 of 1,850 files (20 %), 2,740 MiB, 693,919 rows** (kb's 319,916 rows @171 files is stale) |
| row-count check | footers read for coco (39,665), mpii (7,135), 3dpw (22,405) — all match the kb exactly |

Not needed for the demo; keep as-is (12 GB of A:), do not resume.

## 6. Licences — what our README must say

All six copies are the **same file**: root, dataset, dinov3, vith and sam-3d-objects `LICENSE` are
8,256 B with SHA-256 `315341214AA8…`; `checkpoints/LICENSE` is 8,204 B / `B3A5A0E2D973…` and is
**identical after CRLFâ†’LF normalisation** (confirmed).

Header: *"Last Updated: November 19, 2025"*. It is a **custom Meta SAM License**, not OSI/CC. **No
non-commercial clause anywhere** (grep for `non-commercial`/`commercial` finds no restriction) — the
"non-commercial-style" shorthand in `kb/pivot-stack.md` stays wrong. Binding conditions (quoted line numbers in
`sam-3d-body/LICENSE`): redistribution only under the same agreement **with a copy** (§4a); publication of
results must **acknowledge the use of SAM Materials** (§31); compliance with **Trade Controls**, no ITAR /
military / nuclear / espionage / weapons use (§33–35); **"will not involve or encourage others to reverse
engineer, decompile or discover the underlying components"** (§34); patent-litigation termination (§25, §46);
Meta may terminate on breach and you must **delete and cease use** (§48); access is **HF-gated** (account, DOB,
country, affiliation) and rejected in sanctioned jurisdictions.

**README obligation (one line):** "SAM 3D Body / SAM 3D Objects / the SAM 3D Body dataset are used under Meta's
SAM License (Last Updated November 19, 2025) — reference upstream by URL, weights and annotations stay out of this
repo; any publication acknowledges SAM 3D. Dataset annotations are SAM-licensed; the images they point at remain
under each source dataset's licence."

## 7. Environment actuality for the manual-bbox path

`importlib.util.find_spec` in `A:\dev\GNM-V\.venv`:

| present | absent |
| :--- | :--- |
| torch, torchvision, numpy, cv2, timm, einops, **yacs**, **roma**, pytorch_lightning, PIL, trimesh, open3d, scipy, pandas, pyarrow, **omegaconf**, **braceexpand**, **pyrootutils**, optree | **detectron2, pyrender, hydra, webdataset, scikit-image, pycocotools, xtcocotools, fvcore, jsonlines, loguru, chump, moge, pytorch3d, smplx, hdbscan** |

The manual-bbox path (`from_photo.py`) needs none of the absent ones — that matches STATUS/LOG (load 9.3–9.7 s,
body 1.3–1.6 s). Absent deps matter only for: the upstream `demo.py` detector/segmentor/FOV
(detectron2/sam2/moge2), dataset prep (webdataset/pycocotools), and `sam-3d-objects` alignment (pytorch3d/moge).
Note the kb's "missing yacs/roma" is now stale — they were installed 2026-10-31 20:15 (§8).

## 8. Delta against `kb/sam-3d-body.md` (and STATUS / pivot-stack)

| old kb statement | today's fact | evidence |
| :--- | :--- | :--- |
| "the **code repo is not on disk**, so nothing can run yet" (§Conclusion, §1, §5) | **wrong now** — `pivot/sam-3d-body` is the code repo, HEAD `b5c765a`; it has been run (inference timings, `pipeline/avatar/`), `__pycache__/*.pyc` dated 10/31 19:56–20:03 | `git rev-parse`, LOG 20:15, pyc timestamps |
| `pivot/sam-3d-body-dinov3/` etc. as top-level paths (§1 table) | **moved** one level deeper: `pivot/sam-3d-body/sam-3d-body-{dinov3,vith,dataset}`; old paths absent; real dirs, not links | `Test-Path`, `Get-Item .LinkType` |
| "checkpoints/ is untracked, not ignored (`?? checkpoints/`)" (§1) | inside the code repo the whole nested clone is untracked (`?? sam-3d-body-dinov3/`); the repo's `.gitignore` covers `/checkpoints/` and `*.ckpt` | `git status --short`, `.gitignore` |
| "the `.venv` **missing** … yacs, roma …" (§5) | **present now** (installed 2026-10-31 20:15 with omegaconf, braceexpand, pyrootutils) | find_spec; LOG 20:15 |
| open unknown #1 "**nothing has been executed**" | **resolved**: executed; 519 = 6+260+45+28+108+72 verified by shape, pred_vertices = MHR forward at 0.01 scale | STATUS 293–299, kb §2 |
| open unknown #2 "`keypoint_mapping` → 70: which 308 / which 70" | **resolved here** (§3): 308 = Sociopticon format; rows 0–69 = mhr70, rows 70–307 = 238 non-teeth face points; inference keeps only `[:, :70]` | this report |
| open unknown #3 "is 3DB's MHR face dense enough to retarget GNM?" | **partly resolved**: 238 named face points on the mesh surface (mouth 31, eyes 32, iris/pupil 36, ears 52) but **no jaw point and no expression** — landmark density yes, articulation no | §3 |
| open unknown #4 "`sa1b_train` final size/rows" | resolved as final: **374/1,850 files, 693,919 rows, incomplete, non-resumable** | footers, §5 |
| open unknown #5 "Windows + detectron2 build untested" | **moot** for our path (manual bbox, detectron2 never installed); still untested and unnecessary | §7 |
| open unknown #6 "vith vs dinov3 accuracy" | still open — vith weights absent (pointers only) | §1 |
| "`pred_face` hard-zeroed" | **still true, and stronger**: jaw rotation is zeroed too (`pred_pose_euler[:, -3:] = 0`) | `mhr_head.py:307,316` |
| dataset `.git` deletion/"≈24 GB" projection (§4 note) | dataset is **12.04 GiB parquet, no `.git`, 1,546 files**; 374 `sa1b` files; cost = the parquet only | §5 |
| `kb/pivot-stack.md:18` "the SAM 3D Body **code** repo (facebookresearch/sam-3d-body) is missing" | **stale** — repo present and running | as above |
| `STATUS.md:179,184-185,201` (code repo not local; old `pivot/sam-3d-body-*` paths) | **stale**, and internally inconsistent with `STATUS.md:783+`/`286-299` which already say it is local and running | Select-String, this report |
| `pipeline/avatar/from_photo.py:44-45`, `panel_landmarks.py:31`, `photo_front_head.py:104-105` | **stale paths** — will raise `missing 3DB checkpoint`; fix to `pivot/sam-3d-body/sam-3d-body-dinov3/checkpoints/...` | §1 |
| kb §6 licence reading | **still correct** (2025-11-19, no non-commercial) — re-verified byte-level | §6 |

## 9. Commands used (all read-only)

```powershell
git -C A:\dev\GNM-V\pivot\sam-3d-body rev-parse HEAD ; ... status --short
Get-ChildItem <dir> -Recurse -File | Measure-Object Length -Sum      # sizes, parquet counts
Get-Item <path> -Force | Select LinkType,Target                      # links vs real dirs
Get-FileHash <LICENSE copies> -Algorithm SHA256                      # 6 licence copies
& .venv\Scripts\python.exe _kpscan.py     # torch.load(mmap=True): keypoint_mapping stats
& .venv\Scripts\python.exe _parse308.py   # ast-parse of keypoints308.py (no import of 3rd-party code)
& .venv\Scripts\python.exe _geocheck.py   # MHR jit forward + M @ [verts;joints]: 3D keypoint check
& .venv\Scripts\python.exe _cats.py       # category -> row ranges ; pyarrow footers via ParquetFile(...).metadata.num_rows
& .venv\Scripts\python.exe -c "importlib.util.find_spec(...)"        # §7 table
```

Artifacts beside this file: `_kpscan.py`, `_parse308.py`, `_geocheck.py`, `_cats.py`,
`_sapiens308.json` (all 344 names + 238 face list), `_kp308rows.json`, `_face238.txt`.
Nothing under `pivot/sam-3d-body/`, `kb/`, `LOG.md`, `STATUS.md`, `pipeline/` was modified.
