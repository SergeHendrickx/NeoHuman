# SAM 3D Body (3DB) — code + weights on disk, contract, and the two places the face is switched off

**Conclusion (read this far).** `pivot/sam-3d-body` is now the *full code repo* (HEAD `b5c765a`,
2026-02-19) with the dinov3 weights beside it. It runs in `.venv` on the 4090: load **8.2–8.6 s**,
one photo → MHR fit **≈0.2 s warm** (`inference_type="body"`) or **≈0.75 s** (`"full"`), ~3.6 GB GPU.
It returns an **18,439-vert MHR body in metres** (root-centred, camera axes), 70 keypoints, 45 identity
coefficients, 204 MHR model params and a camera translation; `pred_vertices` reproduce the MHR
TorchScript from those parameters to **RMS 1e-4 mm** — the identity link is exact.

**The face is disabled twice, inside 3DB's head, not by MHR.** The head predicts 72 facial-expression
coefficients and multiplies them by zero (`mhr_head.py:316`), and it zeroes body params 130–132
(`mhr_head.py:307`); the MHR TorchScript itself accepts a non-zero 72-vector. So **a 3DB mesh always
has a neutral jaw** — jaw/expression must be driven from outside 3DB (MHR's 72 `blend_45..116` or GNM).

**The 70 released keypoints have no mouth/jaw landmark** (only nose, 2 eyes, 2 ears — idx 0–4), but the
checkpoint's `keypoint_mapping` is a `(308, 18566)` linear map whose rows **70–307 are 238 dense face
landmarks** that the code computes and throws away (`j3d = j3d[:, :70]`). They are free: one matmul on
the returned vertices (§4). Mouth corners are rows 188/189 (outer) and 204/205 (inner).

**Operational delta:** the model dirs were moved **inside** the code repo
(`pivot/sam-3d-body/sam-3d-body-dinov3/…`), so the three pipeline files that hard-code the old
`pivot/sam-3d-body-dinov3/…` paths are **broken until their defaults are updated** (§7).

*Provenance: measured 2026-11-03 01:25–01:40 Brussels by the lead + three teammates, `.venv`
Python 3.11.14 / torch 2.11.0+cu128 / RTX 4090. Full evidence: [code contract](code-contract.md),
[runtime matrix](runtime.md),
[metadata + licence + delta](metadata-objects.md),
[308 landmark tables](landmarks-308.md). The old entry's claims that
this one supersedes are listed in §8.*

## 1. What is on disk now

| Path | State | Usable? |
| :--- | :--- | :--- |
| `pivot/sam-3d-body/` | code repo, HEAD `b5c765a`, gitignored clone | **yes** |
| `pivot/sam-3d-body/sam-3d-body-dinov3/` | HF clone; root `model.ckpt` = 138 B LFS pointer; **`checkpoints/` holds the real weights** | **yes** |
| `pivot/sam-3d-body/sam-3d-body-dinov3/checkpoints/` | `model.ckpt` **2,109,129,346 B**, `assets/mhr_model.pt` **696,110,248 B**, `model_config.yaml`, `LICENSE`, HF download metadata pinned to `11aaa346…` | **yes** |
| `pivot/sam-3d-body/sam-3d-body-vith/` | clone, **LFS pointers only** (138 B ckpt), no `checkpoints/` | **no** |
| `pivot/sam-3d-body/sam-3d-body-dataset/` | 1,546 parquet / **12.04 GiB**, no `.git` (not resumable), `sa1b_train` 374/1,850 files = 693,919 rows | training-only |
| `pivot/sam-3d-body/sam-3d-objects/` | full clone `f91db41` (268 MiB `.git`), **no weights**, needs pytorch3d + MoGe (both absent) | **irrelevant** |

The old `pivot/sam-3d-body-dinov3/…` layout is **gone** (`Test-Path` False) — see §7. `sam-3d-objects`
reconstructs object geometry/texture/layout, composes with 3DB only through a notebook that needs
pytorch3d + MoGe; it has no head model and no weights, so it adds nothing to GNM-V.
The dataset is the supervision set (annotations only, images never shipped, eight upstream licences);
inference needs the checkpoint alone.

## 2. Running it here

Manual-bbox path, no detector/segmentor/FOV model (all `None`), which is what `pipeline/avatar/` does:

```python
sys.path.insert(0, r"A:\dev\GNM-V\pivot\sam-3d-body")
os.environ["TORCH_HOME"] = r"A:\dev\GNM-V\.cache\torch"     # DINOv3 *code* comes from torch.hub
from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
model, cfg = load_sam_3d_body(ckpt, device="cuda", mhr_path=mhr)   # ckpt path must be …/checkpoints/model.ckpt
est = SAM3DBodyEstimator(model, cfg, human_detector=None, human_segmentor=None, fov_estimator=None)
out = est.process_one_image(rgb_uint8, bboxes=np.array([[x1,y1,x2,y2]], np.float32),
                            inference_type="body")              # list of one dict per box
```

- **Load 8.2–8.6 s** (four processes); **warm inference 0.17–0.22 s** (`body`), **0.72–0.78 s**
  (`full`), first call in a process ≈1.15–1.22 s. **GPU peak 3.5–3.7 GB** allocator.
- `full` = body + two extra 512² hand crops + hand decoder + a 4-point prompt re-run (3 backbones);
  it only *adds* `lhand_bbox`/`rhand_bbox` and **zeroes `pred_pose_raw`**. Use `body` unless hands.
- `inference_type="hand"` **crashes** through `process_one_image` (`mhr: None` then subscripted).
- **CPU is not impossible — it is blocked by two hard-coded `.cuda()` calls.** Unpatched CPU run dies
  at `sam_3d_body_estimator.py:160` (`recursive_to(batch,"cuda")`); after neutralising that only, at
  `sam3d_body.py:1037` via `.cuda()` in `get_ray_condition` (`:1034`). Neutralise **both** at runtime
  (no file edit) and CPU inference **succeeds**: 13.38 s, |β| 3.5567 vs GPU 3.5588 (Δ 0.06 %) — ~67×
  slower. *Corrects STATUS.md's "CPU inference is impossible upstream".*
- Dependencies: manual-bbox path needs only torch/torchvision/cv2/roma/einops/omegaconf|yacs/
  pytorch_lightning + the torch.hub DINOv3 code. **Absent and not needed** on this path: detectron2,
  sam2, MoGe, pyrender, hydra, webdataset, skimage, pycocotools, xtcocotools, fvcore, pytorch3d.
  Upstream `demo.py` is *not* runnable here (defaults to ViTDet detector + SAM2 + MoGe2 + pyrender
  visualisation) and it has no path defaults; pass `--checkpoint_path sam-3d-body-dinov3/checkpoints/model.ckpt`.
- The load warning "model and loaded state dict do not match exactly" is **benign**: 113 missing keys,
  0 unexpected — 102 are TorchScript MHR/rig buffers that are loaded by `torch.jit` separately, plus
  `backbone.encoder.mask_token` and `hand_pose_comps_ori`; 10 spot-checked weights are bit-identical.
- Deterministic to float32 round-off, **not bitwise**: repeated runs differ by ≤1e-6 m on vertices
  (1 µm), 2.4e-4 px on keypoints.

## 3. The contract

**Input**: person box (xyxy, original pixels) → `GetBBoxCenterScale(padding=1.25)` → aspect 0.75 then
1.0 → affine warp to **512×512** (config `MODEL.IMAGE_SIZE`) → ImageNet mean/std. The warped, *expanded*
`bbox_scale` is written back and used by the camera maths. No intrinsics supplied → synthetic
`cam_int` with `f = sqrt(w²+h²)` (2704.16 px on 2250×1500, 1448.15 on 1024²) — depth inherits that guess.

**Output dict (one per person, `body`):** `pred_vertices (18439,3)` m · `pred_keypoints_3d (70,3)` m ·
`pred_keypoints_2d (70,2)` px in the original image, **no confidence channel** · `pred_joint_coords
(127,3)` · `pred_global_rots (127,3,3)` · `pred_pose_raw (266,)` (6-D global rot + 260 continuous;
zeroed on `full`) · `global_rot (3,)` · `body_pose_params (133,)` · `hand_pose_params (108,)` ·
`scale_params (28,)` · `shape_params (45,)` · `expr_params (72,)` **all zeros** · `mhr_model_params
(204,)` · `focal_length` scalar · `pred_cam_t (3,)` m · `bbox` echo.

**Geometry.** `mhr_model_params` = 136 pose (3 root translation×10 dm + 3 global euler + 130 body) +
68 scale. `519 = 6 + 260 + 45 + 28 + 2×54 + 72` from the head. MHR TorchScript is **centimetres, y-up,
origin at the feet**; 3DB does `/100` then negates y,z → **metres, camera axes, root at origin, not
translated** — add `pred_cam_t` for camera coordinates (`tz = 2·f/(bbox_scale·s)`; behaves as metric
depth). `apply_correctives` is on (defaulted 4th TorchScript arg) and is a learned pose-dependent
vertex delta; don't compare meshes across that flag.

**Batch/people.** Batch is `[1, num_person, 3, 512, 512]`. Two boxes in one call return two dicts
(0.68 s) but are **not** equal to two separate calls (max |Δβ| 0.24; max vertex Δ 0.11 m; even two
identical crops differ) — batching perturbs the estimate.

**Validation: none.** A zero-area box gives **silently NaN** vertices (only a numpy `RuntimeWarning` at
`data/transforms/common.py:270`); 1-px, 30-px, background-only and inverted boxes give finite,
plausible-looking garbage (|β| 3.7–4.3, 1.35–1.76 m tall). The caller must check `x2>x1`, minimum
size, `np.isfinite`, and |β| ≤ ~10. Identity is **not box-invariant**: +10 %/20 % padding moves the
mesh 6.6/13.4 mm RMS; 180° image rotation gives cos(β)=0.978 and 25.9 mm RMS (a rotated prior is
recovered, not a different person).

## 4. The face: what is off, and the 238 landmarks that are free

1. **Expression is predicted then killed.** `mhr_head.py:316`: `pred_face = pred[:,…72] * 0`.
   `expr_params` in the output is zeros — bbox/mask/cam_int/keypoint-prompt cannot restore it
   (measured for all four).
2. **Body params 130–132 are zeroed too** (`mhr_head.py:307`, "Zero-out jaw" in the code comment).
   *What joint that block is, the code never says; calling it "the jaw" is an inference* — MHR's
   parameter transform has no jaw row (see [mhr.md](../../kb/mhr.md): jaw is the `jawDrop` blendshape,
   `c_jaw` is not parameter-driven). Treat it as "an articulation block 3DB refuses to predict".
3. **MHR itself does expression.** TorchScript schema (measured): `forward(identity (N,45),
   model_params (N,204), face_expr (N,72), apply_correctives=True) -> (verts, skel_state)`. So a
   non-zero 72-vector works; it just has to be supplied by us, together with whatever 130–132 should
   carry. `pipeline/avatar/from_photo.py` already calls this for verification.
4. **The 70 keypoints cannot give jaw/mouth landmarks.** `metadata/mhr70.py:3` — "the first 70 of 308
   MHR keypoints, ignoring the rest for face keypoints". Indices 0–4 = nose, 2 eyes, 2 ears; 5–20 body
   (incl. feet); 21–62 hands; 63–69 olecranon/cubital/acromion/neck. **No mouth, jaw, lip, eyelid,
   iris, teeth or tongue point exists in the 70.**
5. **But all 308 rows are in the checkpoint and 238 of them are face points.** `keypoint_mapping`
   `(308, 18566)` is a fixed row-stochastic map over 18,439 verts + 127 joints (every row sums to 1;
   48 one-hot joint rows, all in 0–69; rows 70–307 are pure vertex blends, 5–19 verts, mean 8.7).
   Rows 0–69 = the 70 released keypoints exactly; rows 70–307 = the **238 non-teeth dense face points
   of the Sapiens/Sociopticon 308 format** (the 36 `teeth_*` points are removed; 344−36 = 308).
   The code computes them in the skinning matmul and discards them (`mhr_head.py:337`,
   `sam3d_body.py:1614`). Recover with `mapping[70:] @ concat(verts, joints)` on the **pre-flip**
   vertex array (`mhr_head.py:241-251` runs before the y/z flip at `:340`).
   Useful rows: mouth corners **188/189** (outer), **204/205** (inner); lips **190–219** (lower-lip
   centre 191, inner centres 206/207); chin tip **77**, labiomental groove **76**; eyes/lids 96–177;
   ears 220–271; iris 272–289; pupils 290–307. **No "jaw" point exists in the 308 set either.**
   This is already exploited by `pipeline/avatar/panel_landmarks.py` (texture camera fitting), not by
   the identity path.
6. **Face is a small part of the mesh**: nose at (0, 158.9, 15.5) cm; vertices within 5/7/9 cm of the
   nose = **1,169 / 2,332 / 3,120 of 18,439** (6.3 / 12.6 / 16.9 %). The 70 keypoints are *regressions*
   (e.g. the nose point is a weighted sum of 8 vertices), not vertex picks.

## 5. What it means for GNM-V

3DB contributes **identity, body pose, hands, 70 keypoints, 238 face landmarks and a camera** — and
nothing that articulates. The voice/jaw coupling cannot come from 3DB: a photo → 3DB → MHR avatar has
a neutral jaw and a closed mouth, and the model has no tongue/teeth/eyeball geometry at all (one
watertight MHR surface). The talk track is MHR's 72 `blend_45..116` expressions driven by our own
layer; the 3DB `shape` (45) + `mhr_model_params` (204) are exactly the inputs MHR needs, and our
baked asset already applies identity that way.

## 6. Licence (what the README must say)

SAM License, "Last Updated: November 19, 2025", **no non-commercial clause** (correcting the earlier
"non-commercial" shorthand). Six copies on disk are byte-identical — code repo, dataset, dinov3, vith,
sam-3d-objects: 8,256 B SHA `315341214AA8…`; `checkpoints/LICENSE` is 8,204 B and identical after
CRLF→LF. Conditions: redistribution only under the same agreement with a copy; acknowledgment in
publications; **no reverse-engineering/decompiling/"discovering underlying components"**; trade-control
compliance, no ITAR/military use; access is HF-gated and Meta may terminate → delete. Practically:
reference upstream by URL, keep weights out of the repo, acknowledge SAM 3D; dataset annotations are
SAM-licensed while the images they point at stay under eight upstream dataset licences.

## 7. Repair needed: the pipeline's hard-coded 3DB paths

The move into the code repo broke every default that still says `pivot/sam-3d-body-dinov3/…`
(current true paths are `pivot/sam-3d-body/sam-3d-body-dinov3/checkpoints/{model.ckpt, assets/mhr_model.pt}`):

- `pipeline/avatar/from_photo.py:44-45` (`THREEDB_CKPT`, `MHR_MODEL`) — `:43 THREEDB_REPO` is already right
- `pipeline/avatar/panel_landmarks.py:31` (`THREEDB_CKPT`)
- `pipeline/avatar/photo_front_head.py:104-105`

Not fixed by this ingest (outside its write scope, and it is tracked code): pass `--ckpt/--mhr` today,
or update the three defaults. `demo.py` has no path defaults; with the new layout pass
`--checkpoint_path sam-3d-body-dinov3/checkpoints/model.ckpt --mhr_path …/checkpoints/assets` and a
detector-free `--detector_name ""`.

## 8. Superseded claims (old entry, 2026-10-31 — kept so the record is honest)

| Old claim | Now |
| :--- | :--- |
| "the code repo is not on disk, so nothing can run yet" | code present (HEAD `b5c765a`) and it runs; this ingest executed the checkpoint |
| `pred_keypoints_2d (70,3)` | **(70,2)**, pixels, no confidence channel |
| "`pred_face` is hard-zeroed upstream" | true (`mhr_head.py:316`) **and incomplete** — params 130–132 are zeroed too (`:307`), and the 72 coefficients are *predicted then* zeroed |
| "MHR TorchScript self-contained; 308×18439+127; 519 split; cm→m; 4-arg correctives" | all **confirmed** by measurement |
| "which 70 is mhr70?" (open q. #2) | `metadata/mhr70.py`: first 70 of 308; rows 70–307 are the 238 face landmarks (recovered, §4) |
| dataset "1,344 files / 11.4 GB; sa1b 171/1,850, 319,916 rows" | 1,546 files / **12.04 GiB**; sa1b **374/1,850 = 693,919 rows**; `.git` gone, not resumable |
| "deps missing: yacs, roma, …" / "detectron2 needed" | yacs, roma, omegaconf, braceexpand, pyrootutils installed; **no detectron2 needed on the manual-bbox path** |
| "CPU is impossible upstream" (STATUS) | two hard-coded `.cuda()` calls; CPU works after neutralising both (13.4 s, |β| Δ 0.06 %) |

## 9. Rendering the fit (headless — no pyrender needed)

Upstream's visualisation (`sam_3d_body/visualization/renderer.py` → `tools/vis_utils.py`,
`notebook/utils.py`) is **`pyrender` + a GL context**; `pyrender` is not installed and Windows needs a
desktop/pyglet window, so it is not runnable here as shipped. Rendering needs none of that — the fit is
one MHR mesh, and two installed routes work headlessly:

- **Open3D 0.20 `OffscreenRenderer`** (Vulkan on the 4090, no display): ≈0.2 s per 2250×1500 frame
  (0.98 s including engine init); smooth per-vertex shading. `setup_camera(PinholeCameraIntrinsic(W,H,f,f,W/2,H/2), np.eye(4))`.
- **Pure numpy/OpenCV painter** (project → sort far→near → `cv2.fillConvexPoly` per triangle): no GL at
  all, 0.26 s at 1024², flat shading.

Geometry for either: `pred_vertices + pred_cam_t` (metres, 3DB camera frame) and `model.head_pose.faces`
(`(36874,3)`; the estimator output dict has no `"faces"` key), projected by 3DB's `focal_length` with
principal point `(W/2, H/2)`. **Convention verified:** with geometry in 3DB camera coordinates, an
Open3D extrinsic of **identity** matches the photo (naive y/z-flip extrinsics lose the mesh entirely) —
i.e. `setup_camera(intrinsic, extrinsic)` takes a camera-to-world transform. The upstream `Renderer`
instead wants `pred_vertices` *without* `pred_cam_t` and applies the translation itself.

Caveats: 3DB carries **no UVs/materials** (clay render only; textures come from the separate bake
pipeline), the jaw is neutral (eyes closed, photo expression not reproduced), and a head-crop fit has an
extrapolated body. Working example + sample renders: [`render_sam3body.py`](render_sam3body.py),
[`render-3db.md`](render-3db.md).

## 10. Matching improvement: joint identity + pose + scale solve (Sapiens2 keypoints)

3DB's fitted face is systematically too narrow and Sapiens2's face/ear keypoints locate it better. Two
refinements were built and measured (code in `joint_solve.py`; independent verification in `eval_joint.py`):

1. **Identity-only** — re-solve the 45 MHR identity coefficients against Sapiens2's skin-face rows
   (70–219), 52 ear points (220–271, weight 3) and in-frame body rows. Face error dropped 25–60 %, ear
   span from 7–20 % short to 0–5 %, but `|Δβ|` hit the 5 cap on 4/7 images: pose/scale errors were being
   absorbed by identity.
2. **Joint** (the one used) — additionally free body-pose 130 euler deltas, 6-D global rotation, 28-d
   scale latent and camera translation (300 iters, ~6–7 s/image, peak VRAM 3.45 GB). Identity-only result
   is the init. Caps: `|Δβ|≤5` in ±10, pose ≤0.2 rad/comp, scale ≤2, translation ≤10 cm.

Independent harness result (masked 2D error vs Sapiens2, px; raw → identity-only → **joint**): face mean
11.49 → 10.92 → **7.02**, ears 35.33 → 12.82 → **6.78**, ear-span error −2.34 % → −0.63 % (≤2.0 %
everywhere), all 6 portraits improved; full-body photo (dancing) face 19.28 → **3.06**, 65 in-frame body
landmarks 14.45 → **10.84**. Capped cases fixed (Emiru face 9.02→6.67; teenage-woman 12.86→8.82). Moves
stay conservative (max joint 0.178 rad, scale 0.034, translation 2.0 cm, hands unchanged, height −0.83 %,
all finite; A1–A9 pass). The main body-page gain is a **neck/head pose correction** (dancing: 0.13 rad at
the neck, global rotation 0.002 rad).

**Expression (why 3DB looks neutral, and how we un-neutral it).** The pose head emits a 519-d vector
that includes 72 face-expression coefficients (`mhr_head.py:53,63`), but inference multiplies them by zero
(`:316`) and zeroes the jaw eulers (`:307`), while `mhr_forward` truncates the pose to 130 dims (`:198`)
so jaw rotation never reaches the rig; the estimator surfaces that zero as `expr_params`
(`sam_3d_body_estimator.py:210`). README scopes 3DB to body/feet/hands, the release has no expression
supervision, and the mesh is one skin-only 18,439-vertex LOD1 with no teeth/tongue/eyeballs. It is a code
decision, not a model limit: the 72 coefficients are an input to the differentiable MHR forward, so the
joint solve frees them (L2 cap 3, reg 0.02 to neutral, mouth rows 188–219 weight 3). Independent result:
lips 10.77 → 3.31 px, inner-lip aperture error −8.5 → +0.5 px (max 2.9), face 7.02 → 5.22 px; ears/body/
pose/identity unchanged; |expr| 0.41–0.86, all finite. Caveat: no teeth/tongue geometry, so an open mouth
is dark inside.

**Loop-closure check.** Rendering the solved mesh and re-running Sapiens2 pose on it (`outputs/sap-roundtrip/`)
recovers Sapiens2's photo landmarks to 2.4 px on the face (photo-textured render) / 7.1 px (clay render,
no texture), 6.5 / 7.9 px on the ears, and 9.0 px on 65 in-frame body keypoints of the full-body photo. That
is independent evidence the fit is geometry, not texture or landmark overfit.

**Depth/normal loop-closure (no texture), skin-masked.** The solved mesh's z-buffer depth agrees with
Sapiens2's pointmap at **0.54 cm median / 1.48 cm P90** after a per-image scale/offset fitted on the
skin mask (`sap ≈ a·mesh + b`, a = 0.88–1.27; the two conventions are not in the same absolute metres),
with a residual **0.18 cm/100 px** vertical tilt term; skin normals agree at **9.9° median / 13.2° mean**
with the frame map `(x, −y, −z)` and no rotation correction to make (best-fit 2.4° global, residual
unchanged after applying it). Unmasked bbox numbers (1.6 cm / 21°) were contaminated by hair/clothing.
Residuals concentrate on structures the LOD1 mesh lacks (eyeballs, nostrils, lips, chin-to-neck crease).

**Skin-normal dense term (implemented).** The joint solve now also minimises a Huber cosine loss
between mesh normals (recomputed from vertices each iteration) and Sapiens2's normal map, sampled at
visible skin vertices (Face_Neck minus hair/cloth/lips/teeth and minus eye/nostril/mouth discs; weight
0.5). Ablation: supervised angle −4–5° (Emiru 23.0→17.2°), independent pixel-level error −0.2–0.7°,
2D metrics unchanged, expression +2–3 %; the `|Δβ|=5` cap (binding on 13/19) limits absorption. A
skin↔background jaw-silhouette term was measured as applicable to only 12.3 % of the lower-face contour
(38.7 % borders hair, 48.7 % neck/torso = internal boundary), so it was not implemented; normals are the
only dense signal with broad support here.

**Limits.** Still a 45-dim identity + skeleton fit: it cannot invent face shape outside the MHR basis;
expression stays neutral (3DB zeroes it); no jaw/chin-contour landmarks exist in the Sapiens2 308 set, so
jaw shape is not solved. Head-and-shoulders crops have only 0–5 in-frame body keypoints, so body metrics
there are noise (n is reported).

**Geometry form:** the solve keeps 3DB's camera convention (`verts + cam_t`); a delta in
`mhr_model_params[:3]` would be double-counted, so translation is optimised on `cam_t` instead.

## 11. Open questions / next tests

1. **What is body-param block 130–132?** The code comment says jaw; MHR's parameter transform does
   not list a jaw parameter. Resolve by driving those 3 params on the TorchScript and measuring which
   vertices move (one script, ~2 min) — matters only if we want MHR jaw articulation rather than the
   `jawDrop` blendshape.
2. **Does a non-zero 72-vector produce usable visemes?** Feed MHR `blend_45..116` from our own driver
   and compare against parselmouth formants — that is the A-track experiment, not an ingest question.
3. **vith weights** never fetched (1.69 GB); only a blocker if the vith backbone is chosen.
4. **Face landmarks on GNM**: the 238-point map is MHR-topology only (18,439 verts); it does not
   transfer to GNM's 17,821-vertex mesh without a correspondence.
