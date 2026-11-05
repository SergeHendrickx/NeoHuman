# SAM 3D Body — code contract (verified against code on disk + checkpoint + TorchScript)

Scope: `pivot/sam-3d-body` (HEAD `b5c765a0d89d…`, 2026-02-19). All paths below are relative to
`pivot/sam-3d-body/`. Anything marked **[measured]** was executed here with
`A:\dev\GNM-V\.venv\Scripts\python.exe` (torch 2.11.0+cu128) on the local checkpoint
`sam-3d-body-dinov3/checkpoints/model.ckpt` and `checkpoints/assets/mhr_model.pt`; anything marked
**[inference]** is my reading, not a measurement. `[measured]` — not a single number here is copied
from `kb/sam-3d-body.md`.

## Read this far — the 15 load-bearing facts

1. **One skinned mesh, no parts.** Output `pred_vertices` = MHR mesh, **18,439 verts / 36,874 tris**
   `[measured: head_pose.faces (36874,3), keypoint_mapping (308,18566)]`. No teeth/tongue/eyeball
   submeshes exist anywhere in the checkpoint (no `teeth|tongue|eye|iris` tensor names `[measured]`).
2. **Facial expression is hard-zeroed, and so is the jaw.** `mhr_head.py:316`
   `pred_face = pred[:, count:count+72] * 0` and `mhr_head.py:307` `pred_pose_euler[:, -3:] = 0`.
   The 72 face coefficients *are predicted by the head and then multiplied by zero*.
3. **But the MHR model itself does support expression.** TorchScript schema `[measured]`:
   `forward(identity (N,45), model_params (N,204), face_expr (N,72), apply_correctives=True) ->
   (verts (N,18439,3), skel_state (N,127,8))`. Zeroing is 3DB's head, not MHR's capability.
4. **238 dense facial keypoints exist and are thrown away.** `keypoint_mapping` has 308 rows; code
   slices `j3d[:, :70]` (`mhr_head.py:337`, `sam3d_body.py:1614`). **[measured]** rows 70–307 all
   land above y=146 cm in a 17×14×14 cm box around the face centroid (0, 160.8, 10.3) cm on the
   neutral mesh — i.e. they are face landmarks, computable with one matmul from the vertex array the
   model already returns. `metadata/mhr70.py:3` says so in words: "The first 70 of 308 … ignoring the
   rest for face keypoints".
5. **Units: the TorchScript is centimetres, y-up, origin at the feet** `[measured: neutral extent
   130.6 × 172.7 × 39.5 cm, y ∈ [0, 172.6]]`. 3DB divides by 100 and negates y,z
   (`mhr_head.py:233-234, 340-343`) → `pred_vertices` is **metres, y-up→3DB camera convention, root
   at the origin, NOT translated**. Add `pred_cam_t` to get camera coordinates.
6. **`pred_cam_t` is a real camera translation in metres**, `tz = 2·f_px / (bbox_scale·s)`
   (`camera_head.py:84-96`); the mesh is in metres so the projection is consistent. `focal_length`
   is `cam_int[0,0]`, and with no intrinsics supplied `cam_int` is a **synthetic diagonal-focal**
   matrix `f = sqrt(w²+h²)` (`data/utils/prepare_batch.py:65-75`) — a guess, not a measurement.
7. **519 = 6 + 260 + 45 + 28 + 2×54 + 72** (`mhr_head.py:56-64`); the head emits 519 (checkpoint
   `head_pose.proj.layers.1.weight (519,1024)`, `init_pose.weight (1,519)` `[measured]`).
8. **`mhr_model_params` = 204 = 136 pose + 68 scale** (`mhr_head.py:213-221`, `return_model_params`
   at `:261-262`). 136 = 3 root translation×10 + 3 global euler + 130 body
   (`mhr_head.py:198, 212-215`); scales = `scale_mean + scale_params @ scale_comps`
   (`mhr_head.py:207`, buffers `(68,)` and `(28,68)` `[measured]`).
9. **Keypoint prompts are decoder conditioning, not a face route.** In `body` mode the only prompt is
   a dummy `[0,0,-2]` (`sam3d_body.py:1118-1119`); in `full` mode 4 prompts = 2 wrists + 2 elbows
   (`sam3d_body.py:1421-1431`). `mhr70` has **no face landmarks beyond nose/eye/ear** (`metadata/
   mhr70.py:5-76`). Prompts re-condition the same 519-dim head; they cannot add face resolution.
10. **`inference_type="hand"` is broken through `process_one_image`** — it returns a dict with
    `mhr: None`, then `sam_3d_body_estimator.py:191` does `out = pose_output["mhr"]` and `:199`
    subscripts `None`. `full` works and only *adds* `lhand_bbox`/`rhand_bbox`.
11. **CUDA is hard-coded** at `sam_3d_body_estimator.py:160` (`recursive_to(batch,"cuda")`),
    `sam3d_body.py:1034` (`.cuda()` in `get_ray_condition`, always called at `:1077`), and the
    `full`-mode hand passes (`:1246, :1282, :1289, :1292, :1508, :1512`). A CPU run breaks at `:160`.
12. **The backbone code is fetched by `torch.hub` at import/construct time** —
    `models/backbones/dinov3.py:15-21` `torch.hub.load("facebookresearch/dinov3", …, source="github",
    pretrained=False)`; weights come from the checkpoint, the *code* comes from `TORCH_HOME`.
    Every other optional module (detector/segmentor/FOV) can be `None` with `bboxes=` supplied.
13. **`pred_pose_raw` is zeroed in `full` mode** (`sam3d_body.py:1621-1623`) and valid only in `body`
    mode; do not carry it across inference types.
14. **Batch shape is `[1, num_person, 3, 512, 512]`** (`prepare_batch.py:45-58`); `_initialize_batch`
    flattens to `batch_size*num_person` (`base_model.py:69-87`) and the output loop indexes the
    person axis (`sam_3d_body_estimator.py:195`). Multiple boxes = one joint forward, per-person rays.
15. **Deterministic in `eval()`**: `@torch.no_grad()` at `sam_3d_body_estimator.py:63`, sampler
    randomness gated on `is_train` (`decoders/keypoint_prompt_sampler.py:99-140`) and unused at
    inference. No RNG in the inference path.

## 1. Forward flow of `process_one_image` → `run_inference`

`SAM3DBodyEstimator.__init__` builds two pipelines (`sam_3d_body_estimator.py:48-61`):
`GetBBoxCenterScale(padding=1.25)` → `TopdownAffine(input_size=512, use_udp=False)` →
`ToTensor()`; the hand variant uses `padding=0.9`.

1. **Boxes.** supplied `bboxes` win (`:107-109`); else detector (`:110-123`); else the whole image
   with `is_crop=False` (`:124-126`). Path inputs are read BGR→RGB (`:99-101, :133-134`); an `ndarray`
   is assumed RGB (`:103`).
2. **Crop.** `GetBBoxCenterScale(padding=1.25)` (`common.py:110-140`) then `TopdownAffine(512,
   use_udp=False)`, which expands the box to aspect 0.75 then 1.0 and **writes the expanded scale back
   into `bbox_scale`** (`common.py:259-268`) — the value the camera maths later uses. Warp,
   `affine_trans`, `ori_img_size` at `common.py:288-327`.
3. **Batch.** `prepare_batch` stacks per box into `[1,N,3,512,512]`, injects `mask_score=0` when no
   mask, and a default `cam_int` (`prepare_batch.py:31-75`). `data_preprocess` applies ImageNet
   mean/std (`base_model.py:35-67`); the 32-px side crop is `vit_hmr*`-only (`:51-65`), dinov3 uses
   the full 512². Then `recursive_to(batch,"cuda")` + `_initialize_batch`
   (`sam_3d_body_estimator.py:160-161`).
4. **Rays + backbone.** `get_ray_condition` maps pixels through the inverse affine, subtracts the
   principal point, divides by focal (`sam3d_body.py:1027-1055`); the backbone is called with
   `extra_embed=None` (`:1097-1099`, `dinov3.py:33`) and the ray map is added **inside each decoder
   stream** by a 1×1 conv over `[B,C+99,H,W]` (`:373` body, `:600` hand;
   `models/modules/camera_embed.py:17-46`).
5. **Token + decoder.** `init_to_token_mhr([condition_info, init_pose(519), init_cam(3)])`; the CLIFF
   condition is `((cx−cx₀)/f, (cy−cy₀)/f, b/f)` (`:248-281`, `USE_INTRIN_CENTER: true` → intrinsic
   principal point). Then `prev_embeddings` + `prompt_embeddings` (`:377-392`) + 70 keypoint and 70
   keypoint3d tokens (`:422-461`) enter the 6-layer SAM-style decoder, which calls the head after
   **every layer but the last**, then once more (`promptable_decoder.py:128-177`); `run_inference`
   keeps the last entry (`:1132`).
6. **Heads + skinning.** `head_pose` → 519 (`mhr_head.py:271-367`), `head_camera` → 3
   (`camera_head.py:45-59`), then `camera_project` (`:917-970`); skinning lives in
   `mhr_head.mhr_forward` → `self.mhr(shape, model_params, face)` (§3). Per-person dict at
   `sam_3d_body_estimator.py:194-216`.
7. **One environment trap here**: the dinov3 *code* is fetched by `torch.hub` at construct time
   (`backbones/dinov3.py:15-21`); the weights come from the checkpoint, the source from `TORCH_HOME`.

**What "308" means.** `keypoint_mapping` is a fixed sparse linear map `(308, 18439+127)` from
`[skinned verts; joint coords]` to 308 MHR keypoints (`mhr_head.py:241-251`, buffer `(308,18566)`
`[measured]`). Rows are sparse (1–43 non-zeros per row `[measured]`). Only rows 0–69 are kept
(`mhr_head.py:337`) and named in `metadata/mhr70.py`. **Rows 70–307 are face/dense keypoints that cost
nothing to evaluate** (see §7).

## 2. The 519 decomposition, exactly as the code splits it

`mhr_head.py:50-64`: `npose = 6 + 260 + 45 + 28 + 54·2 + 72 = 519`. Read out in this order at
`mhr_head.py:291-317`:

| idx | width | code symbol | meaning as used |
|---|---:|---|---|
| 0–5 | 6 | `global_rot_6d` → `rot6d_to_rotmat` → `roma.rotmat_to_euler("ZYX")` | root rotation, **not** translation (`:292-296`) |
| 6–265 | 260 | `pred_pose_cont` → `compact_cont_to_model_params_body` | 133-dim body model params re-encoded: 3-DoF joints as 6-D, 1-DoF as sin/cos, + 6 trans (`modules/mhr_utils.py:321-353`) |
| 266–310 | 45 | `pred_shape` | MHR identity coefficients |
| 311–338 | 28 | `pred_scale` | scale-state components, 68-dim state = `scale_mean + s @ scale_comps` |
| 339–446 | 108 | `pred_hand` (54+54) | left/right hand PCA coefficients |
| 447–518 | 72 | `pred_face` | face expression coefficients → **multiplied by 0** |

`pred_pose_raw` is `cat([global_rot_6d, pred_pose_cont])` = **266 dims**, not 519
(`mhr_head.py:347-349`). The old kb's "519 = 6+260+45+28+2×54+72" is correct; its claim that
`pred_keypoints_2d` is `(70,3)` is **wrong** — it is `(70,2)` (`camera_head.py:106`,
`sam3d_body.py:1645`), which `pipeline/avatar/from_photo.py:280` already assumes.

## 3. `mhr_head.py` plumbing — how (45, 204, 72) become vertices

`mhr_forward` (`mhr_head.py:163-269`), all `[measured]` against the TorchScript:

- **scale** `scales = scale_mean[None] + scale_params @ scale_comps` → `(N,68)` (`:207`).
- **pose** `full_pose_params = cat([global_trans*10, global_rot(euler), body_pose[:, :130]])` →
  `(N,136)`; `*10` only because MHR root translation is in decimetres (`:212-215`). **3DB always
  passes `global_trans = 0`** (`:296`), so the mesh is root-centred and the camera translation lives
  entirely in `pred_cam_t`.
- **hands**: `replace_hands_in_pose` decodes the 54+54 hand PCA into the 27+27 hand joint eulers and
  writes them at `hand_joint_idxs_left/right` (`:136-161`). Immediately before, `MHRHead.forward`
  zeroes the hands inside the body pose (`:305`, mask `mhr_param_hand_mask`,
  `modules/mhr_utils.py:388-391`) so the two paths cannot double-count.
- **model_params** = `cat([full_pose_params (136), scales (68)])` = **204** (`:221`).
- **call** `self.mhr(shape_params, model_params, expr_params)` (`:227-229`) — **3 positional args**,
  so `apply_correctives` takes its default `True` (`[measured]` schema default).
  `MOMENTUM_ENABLED` is false here (`:21-33`) → pure `torch.jit.load` (`:114-117`), no pymomentum.
- **`apply_correctives`** selects the TorchScript's `pose_correctives_model`: a learned, pose-dependent
  vertex delta added **on top of** `blend_shape(identity) + face_expressions(face)`, after which
  `skin_points` runs `[measured]` from the scripted code. Turning it off changes the mesh; nothing in
  3DB ever turns it off.
- **units/convention**: `verts/100`, `joint_coords/100` (`:233-234`), then `verts[...,[1,2]] *= -1`
  and the same for `j3d`, `jcoords` (`:340-343`, repeated in the `full` re-forward at
  `sam3d_body.py:1615-1617`) = "Camera system difference". Result: **metres, camera-frame axes,
  root at origin, no translation**.
- **return dict** (`:346-367`): `pred_pose_raw, pred_pose_rotmat(None), global_rot, body_pose, shape,
  scale, hand, face, pred_keypoints_3d(70,3), pred_vertices(18439,3), pred_joint_coords(127,3),
  faces, joint_global_rots(127,3,3), mhr_model_params(204)`.

## 4. Camera: `pred_cam_t`, `focal_length`, ray condition

`PerspectiveHead.perspective_projection` (`camera_head.py:61-110`) is the whole camera contract:

```
pred_cam[..., [0,2]] *= -1                      # camera-system sign flip   (:79)
s, tx, ty = pred_cam[:,0], pred_cam[:,1], pred_cam[:,2]
bs  = bbox_scale[0] * s * default_scale_factor  # (:85; =1 for body, 10 for hand)
tz  = 2 * focal / bs                            # focal = cam_int[0,0]      (:86-87)
cx  = 2*(bbox_center[0] - cx0)/bs               # cx0 = intrinsic principal point (USE_INTRIN_CENTER) or w/2
cy  = 2*(bbox_center[1] - cy0)/bs               # (:89-94)
pred_cam_t = [tx+cx, ty+cy, tz]                 # metres                    (:96)
j3d_cam = points_3d + pred_cam_t;  j2d = K @ (j3d_cam/z)                     (:99-103)
```
Returned: `pred_keypoints_2d (N,70,2)` in **original image pixels**, `pred_cam_t (N,3)`,
`focal_length (N,)`, `pred_keypoints_2d_depth (N,70)` (`camera_head.py:105-110`). The same projection
is applied to the vertices to give `pred_keypoints_2d_verts` (`sam3d_body.py:947-966`).
Sanity of the algebra: a 1.7 m mesh at the returned `tz` projects to `1.7·f/tz` pixels, which
reproduces the crop box height — i.e. `tz` behaves as metric depth **[inference, from the formula]**.

Ray condition (`sam3d_body.py:1027-1055`): builds a per-pixel `(x,y)` grid, maps it through the
inverse affine into full-image pixels, subtracts the principal point and divides by focal → 2-channel
ray map, `B×N×2×512×512`; then Fourier-encoded to 99 channels and concatenated with the image
features (`camera_embed.py:17-46`). It is what makes the camera a *conditioning* input rather than a
post-hoc fit; it is also the `.cuda()` line.

## 5. `inference_type` — what actually differs

`run_inference` (`sam3d_body.py:1197-1647`), estimator `:179-189`:

- `body`: one pass, body decoder only (`forward_step(..., "body")` → `body_batch_idx` = all,
  `hand_batch_idx = []`, `:1183-1195`). Output keys as §1.9. Output 2D/3D/vertices are the last
  intermediate head prediction, already camera-projected.
- `full`: body pass → `_get_hand_box` → **two extra 512² hand crops** (left one is a *flipped* image,
  `:1238-1247`) → hand decoder → 4 validity criteria (wrist-angle < `thresh_wrist_angle=1.4` rad,
  hand box > 64 px, all hand 2D inside the box, 2D wrist distance < 0.25 of box) (`:1286-1382`) →
  a **4-point keypoint prompt** (2 wrists from the hand decoders + 2 elbows from the body) re-runs
  the body decoder (`:1421-1466`) → hand pose/scale/shape are fused and an analytic wrist xzy is
  written back if valid (`:1470-1593`) → a final `mhr_forward` + manual reprojection
  (`:1598-1645`). Returns the 5-tuple `(pose_output, batch_lhand, batch_rhand, lhand_output,
  rhand_output)`; the estimator adds `lhand_bbox`/`rhand_bbox` (`:218-258`).
- `hand`: hand decoder only, `batch["ray_cond_hand"]`; returns `{"mhr": None, "mhr_hand": …}` — and
  `process_one_image` then crashes (§0.10). Use `model.run_inference(...)` directly if you need the
  hand stream alone.

Cost of `full`: 3 backbone passes + 3 decoder passes vs 1+1 for `body`; body output is the same
parameterisation either way. `pipeline/avatar/from_photo.py:314-315` defaults to `body`.

## 6. Capability limits (what the returned mesh is not)

- **No facial expression**: `mhr_head.py:316` (`* 0`) — the only facial-expression path the returned
  parameters expose is zeros. `expr_params` is still in the output dict, all zeros.
- **No jaw articulation**: `mhr_head.py:307` zeroes the last 3 of the 133 body model params
  (indices 130–132, the last 3-DoF joint in `compact_cont_to_model_params_body`'s index table,
  `mhr_utils.py:323`). The code never names that joint; that it is MHR's jaw is **[inference]**.
- **No teeth/tongue/eyes geometry**: one vertex array, one skinning call `[measured]` (TorchScript
  returns `verts (1,18439,3)` and `skel_state (1,127,8)` only); no part tensors in the checkpoint.
  Eyes/teeth are UV texture, not geometry, in this output.
- **Face resolution**: **[measured]** on the neutral mesh, nose = (0, 158.9, 15.5) cm, eyes
  (±3.2, 162.4, 12.0), ears (±7.7, 159.7, 2.5); vertices within 5/7/9 cm of the nose =
  **1,169 / 2,332 / 3,120 of 18,439 (6.3 % / 12.6 % / 16.9 %)**. So ~1–2 k vertices cover the face.
- **Keypoints are sparse regressions, not vertex picks**: the nose keypoint is a weighted sum of
  **8** vertices (`[measured]`), so `pred_keypoints_2d` cannot be used as a dense face landmark set.
- **`mhr70` has 5 head keypoints**: nose, 2 eyes, 2 ears (`metadata/mhr70.py:5-11, 75`) — no mouth,
  no jaw, no brows. Nothing in 3DB tracks articulation.

## 7. The two routes to more face (both real, both code-verified, neither is "supported")

1. **238 face keypoints, free.** `head_pose.keypoint_mapping[70:308] @ [verts; joint_coords]`
   reproduces MHR's dense face landmarks from the *same* `pred_vertices` the estimator already
   returns (`mhr_head.py:241-251` is the matmul; the truncation to 70 is `:337`). **[measured]** on
   the neutral mesh all 238 land above y=146 cm inside a 17×14×14 cm box at the face centroid.
   Consequence: identity-driven face shape *is* observable from 3DB output without touching the
   head. Requires the vertex array **before** the `[1,2] *= -1` flip or a matching flip of the
   mapping output (the mapping is applied pre-flip, `mhr_head.py:241-251` vs `:340`).
2. **Nonzero expression by calling the TorchScript yourself.** `head_pose.mhr` is loaded and kept on
   the model (`mhr_head.py:114-117`), and `mhr_model_params` (204) + `shape_params` (45) are returned
   (`sam3d_body_estimator.py:209, 214`). `mhr(shape, model_params, face72)` with a non-zero 72-vector
   works — `[measured]` on zero inputs, and `pipeline/avatar/from_photo.py:176-228` already does
   exactly this call for verification. To articulate the jaw you must also un-zero params 130–132
   inside the 136-pose block, which 3DB never predicts (`mhr_head.py:307`), so those values must be
   supplied from outside (**[inference]**: that is the only remaining unknown in this route).

## 8. Device / batch / environment traps

- Hard CUDA: `sam_3d_body_estimator.py:160`; `sam3d_body.py:1034`; `:1246, :1282` (`full` hand
  batches); `:1289, :1292, :1508, :1512` (`torch.LongTensor(...).cuda()`); default
  `load_sam_3d_body(device="cuda")` (`build_models.py:10`). `mhr_head.py:110-116` picks cuda-if-
  available regardless of the requested device, so `device="cpu"` on a CUDA box gives a mixed model.
- Required imports on the manual-bbox path only: `cv2, numpy, torch, torchvision.transforms,
  roma, einops, omegaconf/yacs, pytorch_lightning (checkpoint.py), torch.hub dinov3 code`.
  **Not** needed when `bboxes=` is passed and `use_mask=False`: detectron2, sam2, moge2
  (`tools/build_detector.py:73`, `tools/build_sam.py:25-30`, `tools/build_fov_estimator.py:24`).
- `demo.py:51` builds the SAM2 segmentor whenever `segmentor_name != "sam2"` **or**
  `segmentor_path` is non-empty, and `notebook/utils.py:65-71` only when `segmentor_path` is set.
- Batch/people: `_flatten_person` divides the person axis by `_batch_size` (`base_model.py:80-87`);
  `_get_hand_box` indexes `batch["affine_trans"][0, :, …]` and `np.max(..., axis=1)`, i.e. it assumes
  **`batch_size == 1` and num_person ≥ 1** (`sam3d_body.py:1683-1757`). Passing several boxes keeps
  one forward but does not fix that assumption.
- `demo.py:38` uses `torch.device("cuda") if available`; the estimator's `transform` uses
  `cfg.MODEL.IMAGE_SIZE` = `[512,512]` (`sam-3d-body-dinov3/model_config.yaml:5-7`).

## 9. Top 5 traps (things that look supported but are not)

1. **`expr_params` in the output is zeros** — and it is in *every* output dict, so "we got 72
   expression coefficients" is meaningless unless you check they are not all zero.
2. **`face`/`jaw` look like they are predicted** — the head does emit 72 face coeffs (`:316`) and the
   133-dim pose has 3 jaw-ish params, and both are zeroed after prediction. Reading the head's output
   dimension (519) and assuming expression works is the exact error the old kb warned about, in the
   opposite direction: the *capacity* exists, the *path* is disabled.
3. **`inference_type="full"` reads as "full detail"** — it means "body + hands", costs 3 backbones,
   and returns a *different* `pred_pose_raw` (zeroed, `sam3d_body.py:1621-1623`).
4. **`focal_length`/`cam_int` look measured** — with no `--intrinsics`, `cam_int` is the synthetic
   diagonal-focal default (`prepare_batch.py:65-75`), so depth (`tz = 2f/bs`) inherits that guess and
   `pred_cam_t` is not metric truth.
5. **Keypoint prompts look like a face-detail control** — they are 1 keypoint per click mapped to a
   *decoder* token (`prompt_encoder.py:108-129, 152-188`); `mhr70` offers no mouth/jaw point to
   click. The lever for face shape is the 45 identity coefficients and (outside 3DB) the 72 face
   coefficients + jaw params.

## 10. Relation to the existing integration and the old kb

`pipeline/avatar/from_photo.py` is consistent with everything above: manual bbox, all optional
modules `None` (`:265-268`), `inference_type="body"` default, `pred_vertices` saved as metres
(`:385`), the 3DB-vs-TorchScript equality check with the `except TypeError` 4-arg fallback (`:196-202`)
— that fallback exists because the local script's `apply_correctives` is a *defaulted* 4th arg
**[measured]**, so the 3-arg path is the one taken.

Contradictions with `kb/sam-3d-body.md` (written when only the weights repos were on disk):

| kb claim | code / measurement |
|---|---|
| "the code repo is not on disk, so nothing can run yet" (§1, §5) | code present, HEAD `b5c765a`; it runs — this report executed the checkpoint and TorchScript |
| `pred_keypoints_2d (70,3)` (§2) | `(70,2)`; the estimator's keypoints carry no confidence channel (`camera_head.py:106`) |
| "`pred_face` is hard-zeroed upstream" (§3) | correct (`mhr_head.py:316`), but incomplete: the **jaw pose is zeroed too** (`:307`) and the 72 coeffs are *predicted then* zeroed |
| 519 decomposition, 204 params, `(36874,3)` faces, `keypoint_mapping (308, 18439+127)`, cm→m, z/y flip, `apply_correctives` 4th arg (§2) | all **confirmed** `[measured]` |
| "nothing has been executed" / open question "which 70 is mhr70" (§7) | `metadata/mhr70.py` answers it: first 70 of 308, and rows 70–307 are the discarded face landmarks (§7 above) |

Reproduction of the measurements (transient scripts, kept beside this file):
`introspect_ckpt.py` (state-dict shapes, tensor-name search), `introspect_mhr.py` (TorchScript code +
schema + units probe), `introspect_face.py` (nose/eye/ear landmark vertices, face-region radii),
`introspect_308.py` (all 308 keypoints evaluated on the neutral mesh). All from
`A:\dev\GNM-V\.venv\Scripts\python.exe`, no installs, `pivot/` untouched.
