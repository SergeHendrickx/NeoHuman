# SAM 3D Body (3DB) runtime verification matrix

**Scope.** Question per section: what does the *installed* SAM 3D Body actually do on this machine?
Every number below was measured in-process with the shared `.venv`, torch 2.11.0+cu128, RTX 4090
(25.76 GB), on 2026-11-03. Nothing under `pivot/` was modified; the model code is imported in place.

**Fixed context for every measurement**

| | |
| :--- | :--- |
| interpreter | `A:\dev\GNM-V\.venv\Scripts\python.exe` — Python 3.11.14, numpy 2.4.6, cv2 5.0.0, torch 2.11.0+cu128 (CUDA 12.8), torch import 1.18–1.35 s |
| code | `A:\dev\GNM-V\pivot\sam-3d-body` (on `sys.path`) |
| checkpoint | `pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\model.ckpt` — 2,109,129,346 B |
| MHR model | `...\checkpoints\assets\mhr_model.pt` — 696,110,248 B (TorchScript) |
| `TORCH_HOME` | `A:\dev\GNM-V\.cache\torch` (DINOv3 hub code found offline) |
| config | `checkpoints/model_config.yaml`; `MODEL.IMAGE_SIZE = [512, 512]` (crops are 512², not 896²) |
| estimator | `SAM3DBodyEstimator(model, cfg, human_detector=None, human_segmentor=None, fov_estimator=None)` |
| inputs | dancing.jpg 2250×1500, manual person bbox `[660,210,1510,1430]`; portrait 1024×1024, manual bbox `[200,60,850,1000]`; RGB arrays |
| driver | `measure_gpu.py` (one process: load once, then all runs), `measure_cpu.py`, `probe_cpu_patched.py`, `probe_cuda_env.py` |
| output stats | `torch.cuda.max_memory_allocated()` (allocator, not nvidia-smi); peak reset before every relevant run |

---

## 1. How long does model load and inference take, and is it deterministic?

**Measurement.** `load_sam_3d_body(...)` timed with `perf_counter` around the call; `process_one_image`
timed per call (3 repeats each, same process, after the single load). Every call includes the
`torch.cuda.empty_cache()` that upstream does at the top of `process_one_image`, plus bbox crop /
affine preprocessing.

| run | first (cold) call | warm calls (2nd–3rd) | GPU peak alloc |
| :--- | ---: | ---: | ---: |
| model load | **8.18–8.64 s** (four loads: 8.18/8.21/8.45/8.64) | — | **3,538 MB** after load (3,731 MB reserved) |
| dancing, `inference_type="body"` | 1.156 s | **0.194 / 0.222 s** (median 0.208) | 3,611 MB |
| dancing, `inference_type="full"` | 1.217 s | **0.722 / 0.778 s** (median 0.750) | 3,640 MB |
| portrait, `inference_type="body"` | 0.193 s | **0.166 / 0.187 s** (median 0.176) | 3,612 MB |
| two people in one call, body | — | 0.68–0.71 s | 3,672 MB |

So the Lead's smoke numbers (load 9.73 s, body 2.08 s, full 1.17 s) reproduce as **cold numbers**: the
first call in a process costs ≈1.15 s while warm calls cost ≈0.20 s. Warm **body ≈ 0.2 s**, warm
**full ≈ 0.75 s** per image (full ≈ 3.6× body, because the hand decoder adds a second MHR forward).
Peak GPU memory is constant at 3.6 GB regardless of `body`/`full`; a second person adds ~60 MB.

**Determinism.** 3 identical calls, same input: **not bitwise identical**, differences at float32
round-off: `pred_vertices` max |Δ| = 1e-6 m (1 µm), `pred_keypoints_2d` 2.4e-4 px,
`shape_params` 1e-6, `mhr_model_params` 2e-6, `pred_cam_t` 1e-6. Repeat 3 vs repeat 1 identical in
magnitude. Practical consequence: not reproducible bit-for-bit, reproducible to ~1 µm on mesh vertices.

## 2. What exactly is in one person's output dict?

**Measurement.** `process_one_image(portrait, bboxes=PBOX, inference_type="body")`, one person; every
key inspected for shape/dtype. All 17 expected keys are present, **no extra keys**, no missing keys.

| key | shape | dtype | range | note |
| :--- | :--- | :--- | :--- | :--- |
| `pred_vertices` | (18439, 3) | float32 | −1.770…0.331 | posed mesh, **metres**, camera frame |
| `pred_keypoints_3d` | (70, 3) | float32 | −1.671…0.322 | metres, camera frame |
| `pred_keypoints_2d` | (70, 2) | float32 | −350…5681 | pixels in the **original** image; no confidence channel |
| `pred_joint_coords` | (127, 3) | float32 | −1.763…0.322 | metres |
| `pred_global_rots` | (127, 3, 3) | float32 | −1…1 | rotation matrices, one per joint |
| `pred_pose_raw` | (266,) | float32 | −1.867…12.631 | non-zero on the **body** path; zeroed only on `full` (`sam3d_body.py:1621`) |
| `global_rot` | (3,) | float32 | −0.061…0.063 | Euler radians |
| `body_pose_params` | (133,) | float32 | −0.591…0.608 | 76 non-zero |
| `hand_pose_params` | (108,) | float32 | −0.167…0.221 | all 108 non-zero even on `body` (hand decoder not run, values still emitted) |
| `scale_params` | (28,) | float32 | −5.27…3.07 | all non-zero |
| `shape_params` | (45,) | float32 | −1.128…2.026 | MHR identity vector, all non-zero |
| `expr_params` | (72,) | float32 | **0.0** | all exactly zero — see §3 |
| `mhr_model_params` | (204,) | float32 | −0.591…0.645 | 198 non-zero: root t·10 + Euler + 130 body + 68 scale-state |
| `focal_length` | scalar | float32 | 1448.15 (portrait) / 2704.16 (dancing) | **np scalar, not array**; = default diagonal √(W²+H²) because no intrinsics were supplied |
| `pred_cam_t` | (3,) | float32 | 0.004…1.644 | camera-frame translation (dancing: 0.139 / 0.872 / 3.847) |
| `bbox` | (4,) | float32 | 60…1000 | **echo of the input bbox** in original-image pixels |
| `mask` | None | — | — | `None` unless masks are supplied |

`full` adds exactly two keys: `lhand_bbox`, `rhand_bbox` (§8). The estimator's batch convention:
1 person → `batch["img"]` = `(1, 1, 3, 512, 512)`; 2 people → `(1, 2, 3, 512, 512)`. `dim0` is an outer
batch of size 1, `dim1` is the number of people; the loop `for idx in range(batch["img"].shape[1])`
therefore returns exactly one dict per person (§7).

## 3. Is `expr_params` really all zeros, and can any input restore it?

**Measurement.** `expr_params` inspected after: body inference, full inference, inference with a
supplied mask (bbox-rectangle mask), and inference with explicit `cam_int`.

| input | `expr_params` count | all zero | absmax |
| :--- | ---: | :--- | ---: |
| `inference_type="body"` | 72 | yes | 0.0 |
| `inference_type="full"` | 72 | yes | 0.0 |
| body + supplied mask | 72 | yes | 0.0 (mask does change identity a little: max |Δβ| = 0.044) |
| body + explicit `cam_int` | 72 | yes | 0.0 |

**Answer: no input can restore it.** The zeroing is inside the model:
`pivot/sam-3d-body/sam_3d_body/models/heads/mhr_head.py:316` —
`pred_face = pred[:, count:count+self.num_face_comps] * 0`. Every forward (body, full, and the
`run_keypoint_prompt` path at `sam3d_body.py:1649`, which is not exposed by the estimator anyway) goes
through that multiply. A non-zero face needs an edit of `pivot/` or a re-implementation of the head,
not a different prompt.

## 4. Are `pred_vertices` the MHR TorchScript output, in metres, in the camera frame?

**Measurement.** Re-ran the MHR TorchScript the 3DB head already holds
(`est.model.head_pose.mhr(beta(45), mhr_model_params(204), expr(72))`) and compared with the emitted
`pred_vertices` after the documented transform (`/100`, y,z negated).

| check | value |
| :--- | ---: |
| `pred_vertices` vs TorchScript, RMS | **0.0001 mm** |
| `pred_vertices` vs TorchScript, max | **0.0002 mm** |
| `pred_vertices` vs zero-pose TorchScript, max | 1,533.7 mm |
| `pred_vertices` vs zero-pose TorchScript, RMS | 677.6 mm |
| mesh extent (x, y, z) | **1.125 × 1.858 × 0.397 m** |
| vertex max |v| | 1.760 m |
| TorchScript raw extent | 112.5 × 185.8 × 39.7 **cm** |

So: numerically identical to the MHR TorchScript (residual is float32 print noise), **metres**, and
**not a neutral mesh** — the pose accounts for up to 1.53 m of vertex displacement vs zero pose.
Camera frame (x right, y down, z forward) is confirmed by the bbox-shift test: shifting the bbox
+40 px in x changes `pred_cam_t` x from 0.13915 → 0.14023 and moves the mesh (raw vertex displacement
mean 4.4 mm, max 10.4 mm; rigid-fit residual 3.8 mm RMS / 8.8 mm max). The output is expressed relative
to a camera tied to the image and bbox, not in a canonical world frame.

## 5. How sensitive is identity/mesh to the box? (dancing.jpg, 3 repeats each)

**Measurement.** Same image, same person, bbox variants; compare `shape_params` (identity) and the
posed mesh. `rigid` = per-vertex residual after the best rigid fit to the base mesh, i.e. shape change
with pose removed.

| variant | \|β\| | cos(β, β_base) | \|Δβ\| | rigid RMS | rigid max | raw vertex disp mean |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| base `[660,210,1510,1430]` | 3.559 | 1.0000 | 0 | 0 mm | 0 mm | 0 mm |
| pad 10 % | 3.624 | 0.99883 | 0.186 | 6.63 mm | 13.26 mm | 9.06 mm |
| pad 20 % | 3.684 | 0.99441 | 0.403 | 13.39 mm | 23.27 mm | 14.37 mm |
| shift x +40 px | 3.565 | 0.99963 | 0.097 | 3.76 mm | 8.80 mm | 4.39 mm |
| shift y +40 px | 3.517 | 0.99963 | 0.105 | 4.77 mm | 14.91 mm | 8.88 mm |
| image rotated 180° (bbox rotated with it) | 3.107 | 0.97816 | — | 25.88 mm | 69.16 mm | 1,277 mm (pose flip) |

Identity stays plausible in every variant: **0 of 45 coefficients outside the ±10 MHR identity limits**
(absmax 1.42). But it is not box-invariant: ±20 % padding moves the identity vector by |Δβ| = 0.40 and
deforms the mesh by up to 23 mm after pose removal, and a 180° image rotation changes β by cos 0.978
(25.9 mm RMS shape residual). The box is a real input to the identity estimate.

## 6. Multiple people in one call

**Measurement.** Synthetic two-person image (dancing.jpg concatenated with itself → 4500×1500), one call
with 2 bboxes, then two separate 1-bbox calls on the same crops.

| | one call, 2 people | two separate calls |
| :--- | :--- | :--- |
| outputs | 2 dicts, `batch["img"]` = `(1,2,3,512,512)` | 1 dict each |
| per-person \|β\| | 3.629 and 3.549 | 3.559 and 3.559 |
| time | 0.68–0.71 s | 0.18–0.20 s + 0.18–0.20 s |

The two crops are the **same person**, yet the one-call result differs **between the two people** and
from the separate calls: max |Δβ| = **0.244** (person 0) / 0.157 (person 1), max vertex difference
**0.111 m**. So batching is not output-invariant — batching two people perturbs each estimate
materially. For a 1-person call the indexing is correct (returns 1 dict, not 0 or 2):
`for idx in range(batch["img"].shape[1])` with `out[...][idx]` and `batch["bbox"][0, idx]`.

## 7. The 70 keypoints

**Measurement.** `pred_keypoints_2d` on dancing.jpg; names from
`sam_3d_body/metadata/mhr70.py`; re-derived the projection from `pred_keypoints_3d + pred_cam_t`,
`focal_length`.

- count **70**, shape (70,2) float32, **no confidence channel**; 100 % of points land inside the image.
- **5 of the 70 are face keypoints** (by name: `nose`, `left-eye`, `right-eye`, `left-ear`, `right-ear`,
  indices 0–4). The rest are body/joint/hand points (20 per hand: tip + 3 joints × 5 fingers + wrist);
  this model has **no dense face landmarks** — 5 points total for the face.
- coordinate convention **verified**, not assumed: `x = (x3d + cam_t_x)·f/(z3d + cam_t_z) + W/2` and
  the same for y (f = 2704.16 px, W/H = 2250/1500) reproduces the emitted 2D points with
  **max error 9.2e-05 px**. They are **pixels in the original input image** (origin top-left, x right,
  y down).
- the 5 face points on dancing.jpg: nose (914.0, 270.4), left-eye (900.6, 290.0), right-eye (882.6,
  296.0), left-ear (931.8, 351.6), right-ear (891.0, 368.1). (The "head band above the neck" count is
  32, but that is inflated by the raised arms in this pose; the name-based 5/70 is the meaningful count.)

## 8. `full` vs `body` — hands

**Measurement.** dancing.jpg with `inference_type="full"` (the Lead's portrait smoke also showed this).

- `full` **adds only `lhand_bbox` and `rhand_bbox`** — every other key is unchanged in name. No separate
  hand mesh or joint list is returned; hand pose lives inside `hand_pose_params` (108 = 2 × 54).
- both hand boxes are always present (they are derived from joints, not detected): `lhand_bbox`
  `[707.8, 85.1, 859.5, 236.7]` (the raised hand near the head), `rhand_bbox`
  `[782.0, 812.7, 970.8, 1001.6]` (hand at the hip/leg).
- `hand_pose_params`: (108,), **all 108 non-zero**, norm 3.863, absmax 0.875. There is no
  "no hand visible" flag — a hand box always exists.
- cost: 0.72–0.78 s warm vs 0.19–0.22 s for `body` (≈3.6×), +29 MB GPU peak.

## 9. Failure modes: what does it do with a bad box?

**Measurement.** same image, deliberately bad boxes; exceptions caught and traced.

| case | result |
| :--- | :--- |
| zero-area bbox `[700,700,700,700]` | **no exception**, returns 1 person with `shape_params` and `pred_vertices` **all NaN** (silent garbage); only a numpy `RuntimeWarning: invalid value encountered in scalar divide` at `data/transforms/common.py:270` |
| 1-pixel bbox | no exception, finite output, `\|β\|` = 4.138, mesh extent 0.31 × 1.38 × 0.37 m — plausible-looking garbage |
| 30-px bbox | no exception, finite, `\|β\|` = 4.262, extent 0.28 × 1.35 × 0.59 m |
| bbox on empty road (person absent) | no exception, finite, `\|β\|` = 4.011, extent 0.92 × 1.40 × 0.49 m — it hallucinates a person, as a top-down model must |
| inverted bbox (x2 < x1) | no exception, finite, `\|β\|` = 3.708, extent 1.09 × 1.76 × 0.33 m — no validation |

**Answer:** the library validates nothing about the box. Only the exact zero-area case produces NaN; all
other garbage returns plausible-looking numbers. Any caller must enforce `x2 > x1`, `y2 > y1`, a
minimum box size, and check `np.isfinite(pred_vertices)` and `|β|` plausibility itself.

## 10. CPU inference — the stale `STATUS.md:785` claim

`STATUS.md` says: *"CPU inference is impossible upstream (hardcoded CUDA)"*. Reproduced, then refuted
as a model-level limitation. No file under `pivot/` was edited — the CUDA calls were neutralised at
runtime in the loaded module only.

1. **The documented failure.** Load with `device="cpu"`, run `process_one_image`: the batch is still
   moved to CUDA by the hardcoded `batch = recursive_to(batch, "cuda")`
   (`sam_3d_body_estimator.py:160`), and the CPU model then meets a CUDA tensor:
   `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0
   and cpu!` at `pivot/sam-3d-body/sam_3d_body/models/meta_arch/base_model.py:49` (`data_preprocess`).
2. **Second hardcode.** Neutralising only call (1), the next failure is inside
   `get_ray_condition`: `meshgrid_xy` is built with a hardcoded `.cuda()`
   (`sam3d_body.py:1034`), and `sam3d_body.py:1037` then fails with the same device-mismatch
   `RuntimeError`. These are the only two CUDA hardcodes on the path (grep: 2 `.cuda()`-style sites +
   `recursive_to(batch,"cuda")`).
3. **CPU inference works.** Neutralising **both** at runtime (`recursive_to` → identity,
   `torch.Tensor.cuda` → identity, model on CPU): one body inference **succeeds in 13.38 s** (CPU load
   7.97 s), returns `pred_vertices` `(18439,3)` and `|β|` = **3.5567** vs the GPU run's **3.5588**
   (Δ = 0.0021, 0.06 %) — numerically equivalent up to CPU/GPU kernel differences.

**Verdict:** the claim is true only of the *unmodified* code path. It is **two hardcoded `.cuda()`
calls, not a compute requirement**; a caller-side runtime patch (or a fork) gives working CPU inference
at ~13.4 s/image vs 0.2 s on the 4090 this box has (67× slower, measured once). Caveat: the
`CUDA_VISIBLE_DEVICES=""` trick does **not** cleanly hide the GPU on this torch build — it gives
`is_available() True`, `device_count() 0`, yet `tensor.cuda()` still returns `cuda:0`
(`results_cpu.json`, `probe_cuda_env.py`), so "absolutely no GPU installed" was not simulated; the CPU
run above did all computation with model parameters on CPU.

## 11. The load-time "state dict does not match exactly" warning

`load_sam_3d_body` prints a long missing-keys list via the repo's own loader
(`utils/checkpoint.py:113`, `strict=False`). Counts and classification:

| | count | classification |
| :--- | ---: | :--- |
| missing keys | **113** | 102 are buffers inside the TorchScript MHR rigs (`head_pose.mhr.character_torch.*` 51, `head_pose_hand.mhr.character_torch.*` 51) + `*.pose_correctives_model.*` 3+3 + `*.face_expressions_model.shape_vectors` 1+1 + `*.hand_pose_comps_ori` 1+1; plus `backbone.encoder.mask_token` 1 (MAE mask token, unused at inference) |
| unexpected keys | **0** | nothing present in the checkpoint was ignored |
| spot-check of loaded weights (10 layers, decoder + prompt encoder) | max diff **0.0** | loaded values are bit-identical to the checkpoint |

**Answer: benign.** The missing entries are non-persistent TorchScript buffers (rig geometry, blend
shapes, pose correctives, mask token), not learned parameters; zero unexpected keys and exact weight
matches mean the checkpoint is fully applied. Do not treat this warning as a load failure.

---

## What failed / did not work

- **Zero-area bbox silently returns NaN** instead of raising (§9) — the only failure mode that is not
  "plausible garbage".
- **Batching two people is not equivalent to two calls** (§6): 0.24 |Δβ|, 0.11 m vertex difference.
- **`CUDA_VISIBLE_DEVICES=""` does not hide the GPU** on torch 2.11.0+cu128 here (§10 caveat).
- **My own script bugs** (not model failures): the first two driver runs crashed on
  `cfg.MODEL.IMAGE_SIZE` being a list and on unpacking `run_once`'s return; the bbox-variant contact
  sheet preview was never written because of a stacking bug in the preview helper — the variant
  numbers are in `results.json` (`bbox_sensitivity`), the picture is missing.
- **Not measured:** multi-GPU, `--inference-type hand` alone, keypoint-prompt inference, and GPU power
  draw. The `masks` path with a real detector/segmentor was not exercised (only a synthetic rectangle
  mask); `mask`-conditioned inference changes identity only slightly (Δβ 0.044) and does not touch
  expression.

## Artifact table

| file | what it is |
| :--- | :--- |
| `measure_gpu.py` | driver: loads 3DB once (captures the load warning), runs every GPU measurement matrix, writes `results.json` + the two keypoint previews |
| `measure_cpu.py` | CPU reproduction: `device='cpu'` path with unmodified `recursive_to` (failure A) and with it neutralised (failure B); writes `results_cpu.json` |
| `probe_cpu_patched.py` | neutralises both CUDA hardcodes at runtime and runs one CPU inference; writes `results_cpu_probec.json` |
| `probe_cuda_env.py` | one-shot diagnostic of `CUDA_VISIBLE_DEVICES=""` behaviour under torch 2.11 |
| `extract_summary.py` | helper that prints the compact per-section summary from `results.json` (no new data) |
| `results.json` | **all** measurements: load/timing/determinism, output schema, expr, MHR-TorchScript recompute, bbox sensitivity, multi-person, keypoints, full/hand, failure modes, load-warning analysis |
| `results_cpu.json` | exact CPU tracebacks: `base_model.py:49` (first hardcode) then `sam3d_body.py:1037` (second); `probe_cuda_env` context |
| `results_cpu_probec.json` | CPU inference result after neutralising both hardcodes: 13.38 s, `|β|` 3.5567 vs GPU 3.5588 |
| `preview_dancing_kp70.jpg` | 1100×733 overlay: manual bbox and all 70 `pred_keypoints_2d` on dancing.jpg (red = the 5 face points) |
| `preview_portrait_kp70.jpg` | same overlay on the 1024×1024 portrait with the head bbox |

No `.npy` vertex dumps and no media were written (disk budget); the recompute residuals in
`results.json` are the evidence, not a saved mesh.

## Reproduction

```powershell
$env:PYTHONIOENCODING = "utf-8"
& A:\dev\GNM-V\.venv\Scripts\python.exe A:\dev\GNM-V\pivot\sam3dbody-ingest\runtime\measure_gpu.py
& A:\dev\GNM-V\.venv\Scripts\python.exe A:\dev\GNM-V\pivot\sam3dbody-ingest\runtime\measure_cpu.py
& A:\dev\GNM-V\.venv\Scripts\python.exe A:\dev\GNM-V\pivot\sam3dbody-ingest\runtime\probe_cpu_patched.py
```

(`TORCH_HOME` is set inside each script; one load per process, ~9 s, so the whole matrix runs in about
a minute on this box.)
