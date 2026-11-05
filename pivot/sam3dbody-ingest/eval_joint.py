"""Independent evaluator for SAM 3D Body refinements (identity-only vs joint identity+pose/scale).

Recomputes 3DB's 308 landmarks from *parameters only* -- ``(identity 45, mhr_model_params 204)``
through the MHR TorchScript forward, then projects them through 3DB's pinhole camera and compares
against Sapiens2's 2D keypoints. It deliberately does NOT import ``run_all.py`` or ``joint_solve.py``:
every metric (projection, grouping, ear span, deltas, plausibility) is re-implemented here, so a bug
or a tuned objective in a solver cannot hide in the numbers.

What it evaluates, per image and per params JSON:
  * reprojection error (px) to Sapiens2 308 keypoints, grouped: head rows 0-4, body rows 5-69,
    skin face rows 70-219, ears rows 220-271, overall 0-271 (iris/pupil 272-307 excluded: the MHR
    LOD1 mesh has no eyeball geometry). Only points that are in frame in both are used; n reported.
  * 2D ear span |row 241 - row 267| against Sapiens2's span.
  * identity norm / |delta beta|, pose delta (global rot + body euler, hands split out),
    scale-state delta, translation delta (cam_t and in-params global_trans).
  * plausibility: finite vertices/landmarks, |beta| <= 10, face landmarks inside the person bbox,
    mesh extent sane, per-group landmark displacement.
  * before/after renders: photo | raw 3DB | identity-only | joint, as full-body overlays and face
    crops (photo-textured) plus clay head crops; saved under outputs/joint-eval/<stem>/.
  * outputs/joint-eval/summary.json + summary.csv + a markdown table on stdout.

Params JSON accepted (keys discovered flexibly, both formats work):
  * run_all.py identity-only:  outputs/<stem>/identity_refined.json
        original_shape_params (45), refined_shape_params (45)
    cam_t / focal are NOT in that file -> read from outputs/<stem>/stats.json.
  * joint_solve.py:            outputs/joint-test/<stem>.json
        any of refined_shape_params / shape_params / identity / beta
        any of refined_model_params / mhr_model_params / model_params / params   (204)
        optional cam_t / pred_cam_t / focal_length / focal / global_scale.

Usage (from A:\\dev\\GNM-V):
    .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\eval_joint.py --prep        # build cache
    .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\eval_joint.py --run baseline
    .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\eval_joint.py --run joint
    .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\eval_joint.py --run all --stems Emiru
Flags: --refresh-prep (re-run Sapiens2 + 3DB for the cache), --no-render, --stems ...

VRAM: the process is capped at 40 % of the card because a teammate shares the GPU; peak is printed.
The Sapiens2 pose model and the 3DB model are loaded one at a time and freed between stages.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import cv2

REPO = Path(__file__).resolve().parents[2]          # A:\dev\GNM-V
HERE = Path(__file__).resolve().parent              # pivot\sam3dbody-ingest
PIVOT = REPO / "pivot"
OUT = HERE / "outputs" / "joint-eval"
CACHE = OUT / "cache"
PREP_JSON = CACHE / "prep.json"

INPUT_DIR = HERE / "input-images"
OUTPUT_DIR = HERE / "outputs"
JOINT_DIR = HERE / "outputs" / "joint-test"

THREEDB = PIVOT / "sam-3d-body"
THREEDB_CKPT = THREEDB / "sam-3d-body-dinov3" / "checkpoints" / "model.ckpt"
THREEDB_MHR = THREEDB / "sam-3d-body-dinov3" / "checkpoints" / "assets" / "mhr_model.pt"
SAPH = PIVOT / "sapiens2"
SAPIENS_W = Path(r"D:\AI_Models\3_Vision_Perception\Segmentation\facebook\Sapiens2")

os.environ.setdefault("TORCH_HOME", str(REPO / ".cache" / "torch"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

VRAM_FRACTION = 0.40
torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)

# --------------------------------------------------------------------------- landmark row groups
HEAD_ROWS = np.arange(0, 5)          # nose, left eye, right eye, left ear, right ear
BODY_ROWS = np.arange(5, 70)         # shoulders/elbows/hips/knees/ankles/feet/hands/neck
FACE_SKIN_ROWS = np.arange(70, 220)  # brows, lids, lash, crease, nose, lips
EAR_ROWS = np.arange(220, 272)       # 52 ear points per side; 241/267 are the ear-lobe tips
LIP_ROWS = np.arange(188, 220)       # mouth corners, cupid bow, upper/lower/inner lips
SKIN_ROWS_LEGACY = np.arange(70, 272)  # what run_all.identity_refine averaged (face + ears)
OVERALL_ROWS = np.arange(0, 272)
EAR_TIP_L, EAR_TIP_R = 241, 267

MESH_COLOUR = (0.60, 0.65, 0.75)
MHR_N_PARAMS = 204
MHR_IDENTITY_DIM = 45


# --------------------------------------------------------------------------- small helpers
def free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def vram(tag: str) -> None:
    peak = torch.cuda.max_memory_allocated() / 2**20
    now = torch.cuda.memory_allocated() / 2**20
    print(f"[vram] {tag:16s} now {now:7.1f} MB | peak {peak:7.1f} MB "
          f"(cap {VRAM_FRACTION:.0%} = {0.4 * 24 * 1024:.0f} MB)")
    torch.cuda.reset_peak_memory_stats()


def wrap(a: np.ndarray) -> np.ndarray:
    """Wrap angles to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------- model loading
def sapiens_imports():
    sys.path.insert(0, str(SAPH))
    if "sapiens" not in sys.modules:
        pkg = types.ModuleType("sapiens")
        pkg.__path__ = [str(SAPH / "sapiens")]
        pkg.__version__ = "2.0.0"
        sys.modules["sapiens"] = pkg
    import sapiens.backbones  # noqa: F401  (registers the Sapiens2 backbone)


def load_pose():
    """Sapiens2 0.4b pose model, exactly the recipe run_all.py uses."""
    sapiens_imports()
    from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
    from sapiens.pose.models import init_model

    model = init_model(str(SAPH / "sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/"
                               "sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py"),
                       str(SAPIENS_W / "pose/sapiens2_0.4b_pose.safetensors"), device="cuda")
    model.pose_metainfo = parse_pose_metainfo(
        dict(from_file=str(SAPH / "sapiens/pose/configs/_base_/keypoints308.py")))
    c = dict(model.cfg.codec)
    c.pop("type")
    model.codec = UDPHeatmap(**c)
    return model


def load_3db():
    sys.path.insert(0, str(THREEDB))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

    model, cfg = load_sam_3d_body(str(THREEDB_CKPT), device="cuda", mhr_path=str(THREEDB_MHR))
    est = SAM3DBodyEstimator(model, cfg, human_detector=None, human_segmentor=None, fov_estimator=None)
    faces = np.asarray(model.head_pose.faces.detach().cpu().numpy(), np.int32)
    return model, est, faces


# --------------------------------------------------------------------------- Sapiens2 / 3DB passes
def pose_run(model, bgr, bbox=None):
    H, W = bgr.shape[:2]
    if bbox is None:
        bbox = [0, 0, W - 1, H - 1]
    info = dict(img=bgr, bbox=np.asarray([bbox], np.float32), bbox_score=np.ones(1, np.float32))
    data = model.data_preprocessor(model.pipeline(info))
    with torch.no_grad():
        pred = model(data["inputs"])
    kp, score = model.codec.decode(pred[0].cpu().numpy())
    kp = kp[0] if np.ndim(kp) == 3 else kp
    score = np.asarray(score[0] if np.ndim(score) == 2 else score).reshape(-1)
    m = data["data_samples"]["meta"]
    kp = (kp / np.array(m["input_size"]) * np.array(m["bbox_scale"]) + np.array(m["bbox_center"])
          - 0.5 * np.array(m["bbox_scale"]))
    return kp, score


def person_box(kp, W, H, pad=0.10):
    inside = (np.isfinite(kp).all(1) & (kp[:, 0] >= 0) & (kp[:, 0] <= W - 1)
              & (kp[:, 1] >= 0) & (kp[:, 1] <= H - 1))
    pts = kp[inside]
    if len(pts) < 10:
        return [0, 0, W - 1, H - 1]
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    w, h = x2 - x1, y2 - y1
    return [max(0.0, x1 - pad * w), max(0.0, y1 - pad * h),
            min(W - 1.0, x2 + pad * w), min(H - 1.0, y2 + pad * h)]


def fit_3db(est, bgr, bbox):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = est.process_one_image(rgb, bboxes=np.asarray([bbox], np.float32), inference_type="body")[0]
    return out


# --------------------------------------------------------------------------- params -> landmarks
def mesh_and_landmarks(model, beta, params, expr=None):
    """Mesh vertices (metres, 3DB camera axes, root-centred) and 308 landmarks for params.

    Independent of the solvers: only calls the frozen MHR TorchScript through the released model
    object. ``params`` is the 204-d mhr_model_params as 3DB defines it:
    [global_trans*10 (3), global_rot euler (3), body_pose euler (130), scale_state (68)].
    """
    hp = model.head_pose
    beta_t = torch.tensor(np.asarray(beta, np.float32), device="cuda")[None]
    params_t = torch.tensor(np.asarray(params, np.float32), device="cuda")[None]
    expr_t = (torch.zeros(1, 72, device="cuda") if expr is None
              else torch.tensor(np.asarray(expr, np.float32), device="cuda")[None])
    with torch.no_grad():
        vcm, sk = hp.mhr(beta_t, params_t, expr_t)
        lm = (hp.keypoint_mapping.detach().float()
              @ torch.cat([vcm[0], sk[0, :, :3]], 0)).cpu().numpy() / 100.0
    verts = np.asarray(vcm[0].cpu().numpy(), np.float64) / 100.0 * np.array([1.0, -1.0, -1.0])
    lm = lm * np.array([1.0, -1.0, -1.0])
    return verts, lm


def project(lm, f, cam_t, W, H):
    """3DB's pinhole: x = f*X/Z + W/2, y = f*Y/Z + H/2 after adding cam_t."""
    q = lm + np.asarray(cam_t, np.float64)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = f * q[:, 0] / q[:, 2] + W / 2.0
        v = f * q[:, 1] / q[:, 2] + H / 2.0
    return np.stack([u, v], 1)


# --------------------------------------------------------------------------- metrics
def group_error(uv, kp, score, rows, W, H):
    """Mean/median/p95 px error on rows where both the reference and reprojection are in frame."""
    ref = kp[rows]
    pred = uv[rows]
    in_frame = ((ref[:, 0] >= 0) & (ref[:, 0] < W) & (ref[:, 1] >= 0) & (ref[:, 1] < H)
                & np.isfinite(ref).all(1) & (score[rows] > 0.2)
                & (pred[:, 0] >= 0) & (pred[:, 0] < W) & (pred[:, 1] >= 0) & (pred[:, 1] < H)
                & np.isfinite(pred).all(1))
    n = int(in_frame.sum())
    n_pred_out = int(((score[rows] > 0.2) & np.isfinite(ref).all(1)
                      & ~((pred[:, 0] >= 0) & (pred[:, 0] < W) & (pred[:, 1] >= 0)
                          & (pred[:, 1] < H))).sum())
    n_ref_in = int(((ref[:, 0] >= 0) & (ref[:, 0] < W) & (ref[:, 1] >= 0) & (ref[:, 1] < H)
                    & np.isfinite(ref).all(1) & (score[rows] > 0.2)).sum())
    if n == 0:
        return dict(n=0, n_ref_in=n_ref_in, n_pred_out=n_pred_out, mean_px=None, median_px=None, p95_px=None)
    e = np.linalg.norm(pred[in_frame] - ref[in_frame], axis=1)
    return dict(n=n, n_ref_in=n_ref_in, n_pred_out=n_pred_out,
                mean_px=round(float(e.mean()), 3),
                median_px=round(float(np.median(e)), 3),
                p95_px=round(float(np.percentile(e, 95)), 3))


def mean_px(uv, kp, rows):
    e = np.linalg.norm(uv[rows] - kp[rows], axis=1)
    return round(float(np.mean(e)), 3) if np.isfinite(e).all() else None


def evaluate_case(model, case, ref):
    """Full metric set for one params set. ``case``/``ref`` carry beta, params, cam_t, f, kp, score."""
    W, H = case["size_wh"]
    beta = np.asarray(case["beta"], np.float64)
    params = np.asarray(case["params"], np.float64)
    cam_t = np.asarray(case["cam_t"], np.float64)
    f = float(case["focal"])
    scale = float(case.get("global_scale", 1.0) or 1.0)
    kp = np.asarray(ref["sapiens_kp"], np.float64)
    score = np.asarray(ref["sapiens_score"], np.float64)

    verts, lm = mesh_and_landmarks(model, beta, params, case.get("expr"))
    if scale != 1.0:
        verts = verts * scale
        lm = lm * scale
    uv = project(lm, f, cam_t, W, H)

    m: dict = {}
    for name, rows in (("head", HEAD_ROWS), ("body", BODY_ROWS), ("face_skin", FACE_SKIN_ROWS),
                       ("ear", EAR_ROWS), ("lips", LIP_ROWS), ("overall", OVERALL_ROWS)):
        g = group_error(uv, kp, score, rows, W, H)
        m[f"err_{name}"] = g["mean_px"]
        m[f"n_{name}"] = g["n"]
        m[f"p95_{name}"] = g["p95_px"]
        if name in ("face_skin", "ear", "overall", "body", "head", "lips"):
            m[f"n_ref_in_{name}"] = g["n_ref_in"]
            m[f"n_pred_out_{name}"] = g["n_pred_out"]

    # --- legacy (unmasked) metrics so we can compare against run_all's own reported numbers
    m["legacy_skin_err_px"] = mean_px(uv, kp, SKIN_ROWS_LEGACY)
    m["legacy_ear_err_px"] = mean_px(uv, kp, EAR_ROWS)

    # --- ear span
    ear_pred = float(np.linalg.norm(uv[EAR_TIP_L] - uv[EAR_TIP_R]))
    ear_sap = float(np.linalg.norm(kp[EAR_TIP_L] - kp[EAR_TIP_R]))
    m["ear_span_2d_px"] = round(ear_pred, 2)
    m["ear_span_sapiens_px"] = round(ear_sap, 2)
    m["ear_span_err_pct"] = (round(100.0 * (ear_pred - ear_sap) / ear_sap, 2) if ear_sap > 1e-6 else None)

    # --- mouth opening / width: catches the "expression opened the jaw too far" failure mode
    def pair(a, b):
        if not (np.isfinite(uv[[a, b]]).all() and np.isfinite(kp[[a, b]]).all()):
            return None, None, None
        p = float(np.linalg.norm(uv[a] - uv[b]))
        s = float(np.linalg.norm(kp[a] - kp[b]))
        return round(p, 2), round(s, 2), round(p - s, 2)
    m["lip_aperture_px"], m["lip_aperture_sapiens_px"], m["lip_aperture_err_px"] = pair(206, 207)
    m["lip_width_px"], m["lip_width_sapiens_px"], m["lip_width_err_px"] = pair(188, 189)

    # --- identity
    rbeta = np.asarray(ref["beta"], np.float64)
    m["identity_l2"] = round(float(np.linalg.norm(beta)), 3)
    m["identity_l2_raw"] = round(float(np.linalg.norm(rbeta)), 3)
    m["delta_beta_l2"] = round(float(np.linalg.norm(beta - rbeta)), 3)
    m["beta_max_abs"] = round(float(np.abs(beta).max()), 3)
    m["beta_over_10"] = int((np.abs(beta) > 10).sum())

    # --- pose / scale / translation deltas against the raw 3DB fit's model params
    rparams = np.asarray(ref["params"], np.float64)
    d = params - rparams
    m["delta_trans_inparams_m"] = round(float(np.linalg.norm(d[0:3] / 10.0)), 4)
    dglob = wrap(d[3:6])
    m["delta_global_rot_rad"] = round(float(np.linalg.norm(dglob)), 4)
    m["delta_global_rot_deg"] = round(float(np.rad2deg(np.linalg.norm(dglob))), 2)
    dbody = wrap(d[6:136])
    HANDS = np.arange(68, 122)          # model_params slots of the 54 hand-euler params
    body_mask = np.ones(130, bool)
    body_mask[HANDS - 6] = False
    m["delta_body_pose_l2_rad"] = round(float(np.linalg.norm(dbody)), 4)
    m["delta_body_pose_max_rad"] = round(float(np.abs(dbody).max()), 4)
    m["delta_hand_pose_l2_rad"] = round(float(np.linalg.norm(dbody[HANDS - 6])), 4)
    m["delta_body_no_hands_l2_rad"] = round(float(np.linalg.norm(dbody[body_mask])), 4)
    m["delta_body_no_hands_max_rad"] = round(float(np.abs(dbody[body_mask]).max()), 4)
    top = np.argsort(-np.abs(dbody[body_mask]))[:3]
    m["top_body_delta_slots"] = [int(i) for i in (np.arange(130)[body_mask][top])]
    m["top_body_delta_rad"] = [round(float(dbody[body_mask][i]), 3) for i in top]
    m["delta_scale_l2"] = round(float(np.linalg.norm(d[136:204])), 4)
    m["delta_scale_max"] = round(float(np.abs(d[136:204]).max()), 4)
    rc_t = np.asarray(ref["cam_t"], np.float64)
    m["delta_cam_t_m"] = round(float(np.linalg.norm(cam_t - rc_t)), 4)

    # --- face expression coefficients (72): the head predicts them, 3DB zeroes them at inference
    e = case.get("expr")
    if e is None:
        m["expr_l2"] = 0.0
        m["expr_max_abs"] = 0.0
        m["expr_over_3"] = 0
        m["expr_at_cap"] = False
        m["expr_finite"] = True
    else:
        e = np.asarray(e, np.float64).ravel()
        m["expr_l2"] = round(float(np.linalg.norm(e)), 4)
        m["expr_max_abs"] = round(float(np.abs(e).max()), 4) if e.size else 0.0
        m["expr_over_3"] = int((np.abs(e) > 3.0).sum())
        m["expr_at_cap"] = bool(np.linalg.norm(e) > 0.99 * 3.0)
        m["expr_finite"] = bool(np.isfinite(e).all())
        m["expr_n"] = int(e.size)

    # --- plausibility / numerics
    m["finite_verts"] = bool(np.isfinite(verts).all())
    m["finite_landmarks"] = bool(np.isfinite(lm).all())
    m["finite_uv"] = bool(np.isfinite(uv).all())
    m["vert_abs_max_m"] = round(float(np.abs(verts).max()), 3) if np.isfinite(verts).all() else None
    h = np.ptp(verts[:, 1])
    m["mesh_height_m"] = round(float(h), 3)
    if "_verts" in ref:
        vraw = ref["_verts"]
    else:
        vraw, _ = mesh_and_landmarks(model, rbeta, rparams)
    m["mesh_height_raw_m"] = round(float(np.ptp(vraw[:, 1])), 3)
    m["mesh_height_delta_pct"] = (round(100.0 * (h - np.ptp(vraw[:, 1])) / np.ptp(vraw[:, 1]), 2)
                                  if np.ptp(vraw[:, 1]) > 1e-9 else None)
    bbox = np.asarray(case["bbox_used"], np.float64)
    face_uv = uv[FACE_SKIN_ROWS]
    inside_bbox = ((face_uv[:, 0] >= bbox[0]) & (face_uv[:, 0] <= bbox[2])
                   & (face_uv[:, 1] >= bbox[1]) & (face_uv[:, 1] <= bbox[3])
                   & np.isfinite(face_uv).all(1))
    m["face_in_bbox_frac"] = round(float(inside_bbox.mean()), 3)
    body_uv = uv[BODY_ROWS]
    inside_img = ((body_uv[:, 0] >= 0) & (body_uv[:, 0] < W) & (body_uv[:, 1] >= 0)
                  & (body_uv[:, 1] < H) & np.isfinite(body_uv).all(1))
    m["body_in_frame_frac"] = round(float(inside_img.mean()), 3)

    # --- what moved, in px and in metres
    for name, rows in (("face_skin", FACE_SKIN_ROWS), ("ear", EAR_ROWS), ("lips", LIP_ROWS),
                       ("head", HEAD_ROWS), ("body", BODY_ROWS)):
        du = np.linalg.norm(uv[rows] - ref["_uv"][rows], axis=1)
        m[f"disp_px_{name}"] = round(float(np.nanmean(du)), 2) if np.isfinite(du).any() else None
        dd = np.linalg.norm(lm[rows] - ref["_lm"][rows], axis=1)
        m[f"disp_m_{name}"] = round(float(np.nanmean(dd)), 4) if np.isfinite(dd).any() else None
    m["_uv"] = uv
    m["_lm"] = lm
    m["_verts"] = verts
    m["_expr"] = e
    m["_cam_t"] = cam_t
    m["_focal"] = f
    return m


# --------------------------------------------------------------------------- rendering
def texture_vertices(verts, faces, bgr, f, W, H):
    """Per-vertex photo colours (front-facing vertices projected into the photo) + baked lambert."""
    from scipy.spatial import cKDTree

    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)

    z = verts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([f * verts[:, 0] / np.maximum(z, 1e-6) + W / 2,
                       f * verts[:, 1] / np.maximum(z, 1e-6) + H / 2], 1)
    front = (vn * (-verts)).sum(1) > 0
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0)
    vis = np.where(front & inside)[0]
    colours = np.full((len(verts), 3), 0.45)
    if len(vis) == 0:
        return colours
    photo = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    xi = np.clip(uv[vis, 0].astype(int), 0, W - 1)
    yi = np.clip(uv[vis, 1].astype(int), 0, H - 1)
    colours[vis] = photo[yi, xi]
    missing = np.setdiff1d(np.arange(len(verts)), vis)
    if len(missing):
        _, nn = cKDTree(verts[vis]).query(verts[missing], k=1)
        colours[missing] = colours[vis[nn]]
    light = np.array([-0.35, -0.55, 0.75])
    light /= np.linalg.norm(light)
    lam = np.clip((vn * light).sum(1), 0, 1)
    return np.clip(colours * (0.45 + 0.55 * lam[:, None]), 0, 1)


def render_numpy(verts, faces, f, W, H, vcol=None):
    """Deterministic numpy/OpenCV painter fallback (same camera); used when Filament returns black."""
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64)
    z = verts[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([f * verts[:, 0] / z + W / 2, f * verts[:, 1] / z + H / 2], 1)
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    light = np.array([-0.35, -0.55, 0.75])
    light /= np.linalg.norm(light)
    shade = 0.30 + 0.70 * np.abs((fn * light).sum(1))
    depth = z[faces].mean(1)
    order = np.argsort(-depth)
    img = np.full((H, W, 3), 26, np.uint8)
    base = np.asarray(MESH_COLOUR)[::-1] * 255.0            # BGR
    for i in order:
        tri = faces[i]
        zt = z[tri]
        if (zt <= 0.05).any() or not np.isfinite(zt).all():
            continue
        pts = uv[tri]
        if not np.isfinite(pts).all():
            continue
        if (pts[:, 0].max() < 0 or pts[:, 0].min() >= W or pts[:, 1].max() < 0
                or pts[:, 1].min() >= H):
            continue
        col = base * shade[i] if vcol is None else np.clip(vcol[tri].mean(0), 0, 1) * 255.0
        cv2.fillConvexPoly(img, np.round(pts).astype(np.int32),
                           (float(col[2]), float(col[1]), float(col[0])), lineType=cv2.LINE_AA)
    return img


class Renderer:
    """Reuses one Open3D OffscreenRenderer per image size (engine init is ~0.7 s)."""

    def __init__(self, W, H):
        import open3d as o3d
        import open3d.visualization.rendering as rendering

        self.o3d = o3d
        self.rendering = rendering
        self.W, self.H = W, H
        self.r = rendering.OffscreenRenderer(W, H)
        self.r.scene.set_background([0.10, 0.11, 0.13, 1.0])

    def render(self, verts, faces, f, vcol=None):
        # A transient all-black frame was seen with Filament on this box (random, ~1 in 5 frames);
        # retrying the same renderer fixes it. Destroying a live Filament renderer aborts, so the
        # last resort is the deterministic numpy painter, not a new renderer.
        img = None
        for attempt in range(4):
            img = self._render_once(verts, faces, f, vcol)
            if img.max() > 0:
                break
            print(f"[render] all-black frame, retry {attempt + 1}/4 (vcol={vcol is not None})",
                  flush=True)
        if img.max() == 0:
            print("[render] Filament kept returning black, using numpy painter", flush=True)
            img = render_numpy(verts, faces, f, self.W, self.H, vcol)
        bg = img[2, 2].astype(int)
        mask = np.abs(img.astype(int) - bg[None, None, :]).max(2) > 8
        return img, mask

    def _render_once(self, verts, faces, f, vcol=None):
        o3d, rendering = self.o3d, self.rendering
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, np.float64))
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, np.int32))
        mesh.compute_vertex_normals()
        mat = rendering.MaterialRecord()
        if vcol is None:
            mesh.paint_uniform_color(list(MESH_COLOUR))
            mat.shader = "defaultLit"
            mat.base_color = [*MESH_COLOUR, 1.0]
            mat.base_roughness = 0.55
        else:
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(vcol, 0, 1))
            mat.shader = "defaultUnlit"
            mat.base_color = [1.0, 1.0, 1.0, 1.0]
        self.r.scene.clear_geometry()
        self.r.scene.add_geometry("body", mesh, mat)
        if vcol is None:
            self.r.scene.scene.set_sun_light([-0.35, -0.55, 0.75], [1, 1, 1], 28000)
        intr = self.o3d.camera.PinholeCameraIntrinsic(self.W, self.H, float(f), float(f),
                                                      self.W / 2, self.H / 2)
        self.r.setup_camera(intr, np.eye(4))
        return np.asarray(self.r.render_to_image()).copy()

    def close(self):
        self.r = None


def compose_panels(panels, labels, gap=8, title_h=30):
    H = max(p.shape[0] for p in panels)
    out = []
    for p, lab in zip(panels, labels):
        h, w = p.shape[:2]
        p2 = cv2.resize(p, (max(1, int(round(w * H / h))), H),
                        interpolation=cv2.INTER_AREA) if h != H else p.copy()
        if p2.shape[0] < title_h + 4:
            p2 = np.vstack([np.zeros((title_h, p2.shape[1], 3), p2.dtype), p2])
        out.append(p2)
    panels2, wmax = [], max(p.shape[1] for p in out)
    H2 = max(p.shape[0] for p in out)
    for p, lab in zip(out, labels):
        canvas = np.full((H2, wmax, 3), 24, np.uint8)
        canvas[H2 - p.shape[0]:, :p.shape[1]] = p
        cv2.putText(canvas, lab, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.45, min(0.8, wmax / 900.0)), (240, 240, 240), 2, cv2.LINE_AA)
        panels2.append(canvas)
    stitched = panels2[0]
    for p in panels2[1:]:
        stitched = np.hstack([stitched, np.full((H2, gap, 3), 24, np.uint8), p])
    return stitched


def head_crop_box(uv_face, W, H, pad=0.35):
    pts = uv_face[np.isfinite(uv_face).all(1)]
    if len(pts) < 10:
        return 0, 0, W, H
    x1, y1 = np.floor(pts.min(0)).astype(int)
    x2, y2 = np.ceil(pts.max(0)).astype(int)
    pad = int(pad * max(x2 - x1, y2 - y1))
    return (max(x1 - pad, 0), max(y1 - pad, 0), min(x2 + pad, W), min(y2 + pad, H))


def draw_kps(bgr, kp, rows=None):
    vis = bgr.copy()
    rows = np.arange(len(kp)) if rows is None else rows
    for i in rows:
        x, y = kp[i]
        if np.isfinite([x, y]).all() and 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (int(x), int(y)), 1, (200, 70, 255), -1)
    return vis


# --------------------------------------------------------------------------- prep (cache)
def case_specs():
    """The six ingest images plus the full-body dancing photo."""
    specs = []
    exts = (".png", ".jpg", ".jpeg", ".webp")
    for img in sorted(p for p in INPUT_DIR.iterdir() if p.suffix.lower() in exts):
        stats = OUTPUT_DIR / img.stem / "stats.json"
        if not stats.exists():
            print(f"[skip] {img.stem}: no stats.json")
            continue
        specs.append(dict(stem=img.stem, image=str(img), kind="ingest",
                          stats=str(stats), bbox=None))
    d = THREEDB / "notebook" / "images" / "dancing.jpg"
    if d.exists():
        specs.append(dict(stem="dancing", image=str(d), kind="body",
                          stats=None, bbox=[660.0, 210.0, 1510.0, 1430.0]))
    return specs


def build_prep(specs, refresh=False):
    if PREP_JSON.exists() and not refresh:
        print(f"[cache] {PREP_JSON}")
        return json.loads(PREP_JSON.read_text(encoding="utf-8"))
    CACHE.mkdir(parents=True, exist_ok=True)
    prep: dict[str, dict] = {}

    # ---- stage 1: Sapiens2 308 keypoints (2D reference)
    t = time.time()
    pose = load_pose()
    for c in specs:
        bgr = cv2.imread(c["image"], cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"unreadable image {c['image']}")
        H, W = bgr.shape[:2]
        if c["bbox"] is None:
            kp_any, _ = pose_run(pose, bgr)
            bbox = [float(v) for v in person_box(kp_any, W, H)]
        else:
            bbox = [float(v) for v in c["bbox"]]
        kp, score = pose_run(pose, bgr, bbox)
        d: dict = dict(stem=c["stem"], image=c["image"], kind=c["kind"], size_wh=[W, H],
                       bbox_used=bbox, sapiens_kp=np.asarray(kp, np.float64).tolist(),
                       sapiens_score=np.asarray(score, np.float64).tolist())
        if c["stats"]:
            st = json.loads(Path(c["stats"]).read_text(encoding="utf-8"))
            d["stats_json"] = st
            d["person_box_stats"] = st.get("person_box_xyxy")
        prep[c["stem"]] = d
        print(f"[prep/pose] {c['stem']:48s} {W}x{H} bbox={[round(v, 1) for v in bbox]}")
    del pose
    free()
    vram("after sapiens2")
    print(f"[stage] sapiens2 pose {time.time() - t:.1f} s")

    # ---- stage 2: raw 3DB fit (params + camera) for every image
    t = time.time()
    model, est, faces = load_3db()
    np.save(CACHE / "faces.npy", faces)
    for c in specs:
        bgr = cv2.imread(c["image"], cv2.IMREAD_COLOR)
        d = prep[c["stem"]]
        bbox = d["bbox_used"]
        out = fit_3db(est, bgr, bbox)
        d["raw"] = dict(shape_params=np.asarray(out["shape_params"], np.float64).tolist(),
                        mhr_model_params=np.asarray(out["mhr_model_params"], np.float64).tolist(),
                        cam_t=np.asarray(out["pred_cam_t"], np.float64).tolist(),
                        focal=float(out["focal_length"]),
                        pred_keypoints_2d=np.asarray(out["pred_keypoints_2d"], np.float64).tolist())
        print(f"[prep/3db ] {c['stem']:48s} |beta|={np.linalg.norm(out['shape_params']):.2f} "
              f"focal={float(out['focal_length']):.1f} cam_t={np.round(out['pred_cam_t'], 2).tolist()}")
        del out
    del est, model
    free()
    vram("after 3db")
    print(f"[stage] 3db fit {time.time() - t:.1f} s")

    PREP_JSON.write_text(json.dumps(prep, indent=1), encoding="utf-8")
    print(f"[cache] wrote {PREP_JSON} ({PREP_JSON.stat().st_size / 1e6:.1f} MB)")
    return prep


# --------------------------------------------------------------------------- params JSON loading
def _pick(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return np.asarray(d[k], np.float64)
    return None


def _pick_nested(d, keys, subkeys=("joint", "refined", "refine")):
    """Look for a key inside a nested result block first (joint_solve.py nests under ``joint``)."""
    for sk in subkeys:
        v = d.get(sk)
        if isinstance(v, dict):
            got = _pick(v, keys)
            if got is not None:
                return got
    return _pick(d, keys)


IDENTITY_KEYS = ["refined_shape_params", "shape_params", "refined_identity", "identity",
                 "beta", "refined_beta"]
PARAM_KEYS = ["refined_model_params", "mhr_model_params", "model_params", "refined_params",
              "params", "refined_mhr_model_params"]
CAMT_KEYS = ["cam_t", "pred_cam_t", "refined_cam_t", "camera_translation", "translation"]
FOCAL_KEYS = ["focal_length", "focal_px", "focal"]
EXPR_KEYS = ["refined_expr_params", "expr_params", "expression_params", "face_params", "expr"]


def load_params_json(path: Path, prep: dict) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    beta = _pick_nested(d, IDENTITY_KEYS)
    params = _pick_nested(d, PARAM_KEYS)
    if beta is None:
        raise SystemExit(f"{path}: no identity coefficients found (keys: {list(d)[:12]})")
    if params is None:
        print(f"[warn] {path}: no 204-d model params -> pose/scale stay at the raw 3DB fit")
    cam_t = _pick_nested(d, CAMT_KEYS)
    focal = _pick_nested(d, FOCAL_KEYS)
    expr = _pick_nested(d, EXPR_KEYS)
    if expr is not None and expr.size != 72:
        print(f"[warn] {path}: expression vector has {expr.size} entries, expected 72 -> ignored")
        expr = None
    sub = d.get("joint") if isinstance(d.get("joint"), dict) else {}
    gs = sub.get("global_scale", d.get("global_scale", d.get("scale_global")))
    solver = {}
    for sk in ("joint", "identity_only", "before_3db_metrics", "original"):
        if isinstance(d.get(sk), dict):
            solver[sk] = d[sk]
    return dict(beta=beta, params=params, expr=expr,
                cam_t=(None if cam_t is None else cam_t),
                focal=(None if focal is None else float(focal) if np.ndim(focal) == 0 else None),
                global_scale=(1.0 if gs is None else float(gs)),
                solver_json=solver, raw_json=d)


def make_case(name, prep_entry, beta, params=None, cam_t=None, focal=None, global_scale=1.0,
              source="", expr=None):
    raw = prep_entry["raw"]
    return dict(name=name, stem=prep_entry["stem"], source=source,
                beta=np.asarray(beta, np.float64),
                params=(np.asarray(raw["mhr_model_params"], np.float64) if params is None
                        else np.asarray(params, np.float64)),
                cam_t=(np.asarray(raw["cam_t"], np.float64) if cam_t is None
                       else np.asarray(cam_t, np.float64)),
                focal=(float(raw["focal"]) if focal is None else float(focal)),
                global_scale=global_scale,
                expr=(None if expr is None else np.asarray(expr, np.float64)),
                size_wh=prep_entry["size_wh"], bbox_used=prep_entry["bbox_used"],
                sapiens_kp=np.asarray(prep_entry["sapiens_kp"], np.float64),
                sapiens_score=np.asarray(prep_entry["sapiens_score"], np.float64),
                image=prep_entry["image"])


def reference_case(prep_entry):
    raw = prep_entry["raw"]
    return dict(beta=np.asarray(raw["shape_params"], np.float64),
                params=np.asarray(raw["mhr_model_params"], np.float64),
                cam_t=np.asarray(raw["cam_t"], np.float64),
                focal=float(raw["focal"]),
                sapiens_kp=np.asarray(prep_entry["sapiens_kp"], np.float64),
                sapiens_score=np.asarray(prep_entry["sapiens_score"], np.float64),
                bbox_used=prep_entry["bbox_used"],
                size_wh=prep_entry["size_wh"])


def raw_metrics(model, prep_entry):
    ref = reference_case(prep_entry)
    m = evaluate_case(model, dict(beta=ref["beta"], params=ref["params"], cam_t=ref["cam_t"],
                                  focal=ref["focal"], size_wh=ref["size_wh"],
                                  bbox_used=ref["bbox_used"], global_scale=1.0), ref)
    return m


# --------------------------------------------------------------------------- run
def collect_cases(prep, run, stems):
    """Params sets to evaluate per stem for this run."""
    cases: list[dict] = []
    warnings: list[str] = []
    for stem, entry in prep.items():
        if stems and stem not in stems:
            continue
        raw = entry["raw"]
        # raw 3DB fit is always shown as the reference
        cases.append(make_case("raw_3db", entry, raw["shape_params"], source="3db fit"))
        # identity-only JSON from run_all.py
        idj = OUTPUT_DIR / stem / "threedb" / "identity_refined.json"
        if idj.exists() and (run in ("baseline", "all", "joint", "expr")):
            j = json.loads(idj.read_text(encoding="utf-8"))
            b = j.get("refined_shape_params")
            if b is not None:
                cases.append(make_case("identity_only", entry, b, source=str(idj)))
        # joint JSON from the teammate (identity+pose+scale, NO expression: expr is zeroed)
        jj = JOINT_DIR / f"{stem}.json"
        if run in ("joint", "expr", "all") and jj.exists():
            try:
                lp = load_params_json(jj, entry)
                c = make_case("joint", entry, lp["beta"], lp["params"],
                              lp["cam_t"], lp["focal"], lp["global_scale"],
                              source=str(jj))
                c["solver"] = lp["solver_json"]
                cases.append(c)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{stem}: could not read {jj}: {type(exc).__name__}: {exc}")
        elif run in ("joint", "all"):
            warnings.append(f"{stem}: no joint params at {jj}")
        # joint identity+pose+scale+expression: run_all.py writes it for input-images;
        # for the full-body dancing photo it is a standalone file (outputs/joint-test/dancing_expr.json)
        expr_candidates = [OUTPUT_DIR / stem / "threedb" / "identity_joint.json",
                           JOINT_DIR / f"{stem}_expr.json",
                           JOINT_DIR / f"{stem}.json"]
        ej = next((p for p in expr_candidates
                   if p.exists() and "refined_expr_params" in p.read_text(encoding="utf-8")), None)
        if run in ("expr", "all") and ej is not None:
            try:
                lp = load_params_json(ej, entry)
                if lp["expr"] is None:
                    warnings.append(f"{stem}: {ej} has no expression vector")
                # same identity/pose/scale/camera, expression ON
                cases.append(make_case("joint_expr", entry, lp["beta"], lp["params"],
                                       lp["cam_t"], lp["focal"], lp["global_scale"],
                                       source=str(ej), expr=lp["expr"]))
                # same params, expression OFF: isolates what the 72 coefficients do
                cases.append(make_case("joint_expr_off", entry, lp["beta"], lp["params"],
                                       lp["cam_t"], lp["focal"], lp["global_scale"],
                                       source=str(ej), expr=None))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{stem}: could not read {ej}: {type(exc).__name__}: {exc}")
        elif run in ("expr", "all"):
            warnings.append(f"{stem}: no expression params in "
                            f"{[str(p) for p in expr_candidates]}")
    return cases, warnings


def run_eval(args) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = case_specs()
    if args.stems:
        specs = [s for s in specs if s["stem"] in args.stems]
    prep = build_prep(specs, refresh=args.refresh_prep)
    if args.stems:
        prep = {k: v for k, v in prep.items() if k in args.stems}
    missing = [s["stem"] for s in specs if s["stem"] not in prep]
    if missing:
        raise SystemExit(f"prep cache lacks {missing}; rerun with --refresh-prep")

    cases, warnings = collect_cases(prep, args.run, args.stems)
    if not cases:
        raise SystemExit("nothing to evaluate")
    for w in warnings:
        print(f"[warn] {w}")

    t = time.time()
    model, est, faces = load_3db()
    if (CACHE / "faces.npy").exists():
        faces = np.load(CACHE / "faces.npy")
    vram("after 3db load")

    rows: list[dict] = []
    by_stem: dict[str, list[dict]] = {}
    for stem, entry in prep.items():
        stem_cases = [c for c in cases if c["stem"] == stem]
        if not stem_cases:
            continue
        ref = reference_case(entry)
        bgr = cv2.imread(entry["image"], cv2.IMREAD_COLOR)
        W, H = entry["size_wh"]
        ref_verts, ref_lm = mesh_and_landmarks(model, ref["beta"], ref["params"])
        ref["_verts"] = ref_verts
        ref["_lm"] = ref_lm
        ref["_uv"] = project(ref_lm, ref["focal"], ref["cam_t"], W, H)
        results = {}
        for c in stem_cases:
            m = evaluate_case(model, c, ref)
            results[c["name"]] = m
            row = dict(stem=stem, case=c["name"], kind=entry["kind"], source=c["source"],
                       W=W, H=H, run=args.run)
            row.update({k: v for k, v in m.items() if not k.startswith("_")})
            if "solver" in c:
                sj = c["solver"]
                row["solver_joint"] = sj.get("joint", {})
                row["solver_identity_only"] = sj.get("identity_only", {})
                row["solver_before_3db"] = sj.get("before_3db_metrics", {})
            rows.append(row)
            print(f"[eval] {stem:42s} {c['name']:14s} "
                  f"face={m['err_face_skin']} px (n={m['n_face_skin']}) "
                  f"ear={m['err_ear']} px  span={m['ear_span_2d_px']}/{m['ear_span_sapiens_px']} px "
                  f"({m['ear_span_err_pct']}%)  body={m['err_body']} px (n={m['n_body']})")

        if not args.no_render:
            stem_dir = OUT / stem
            stem_dir.mkdir(parents=True, exist_ok=True)
            render_compare(stem_dir, entry, ref, results, bgr, faces, args.run)
            if "joint_expr" in results:
                render_expr_compare(stem_dir, entry, results, bgr, faces, args.run)
        by_stem[stem] = results
    del est, model
    free()
    vram("after evaluation")

    # ---- persist
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary = dict(generated=stamp, run=args.run, prep_json=str(PREP_JSON),
                   vram_fraction=VRAM_FRACTION, n_cases=len(rows), rows=rows,
                   warnings=warnings)
    (OUT / f"summary_{args.run}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    # also per-stem metrics json next to renders
    for stem, res in by_stem.items():
        p = OUT / stem / f"metrics_{args.run}.json"
        p.write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                 for k, v in res.items()}, indent=1), encoding="utf-8")

    # csv
    keys = ["stem", "case", "kind", "run", "W", "H",
            "err_head", "n_head", "err_body", "n_body", "err_face_skin", "n_face_skin",
            "err_ear", "n_ear", "err_lips", "n_lips", "err_overall", "n_overall",
            "ear_span_2d_px", "ear_span_sapiens_px", "ear_span_err_pct",
            "lip_aperture_px", "lip_aperture_sapiens_px", "lip_aperture_err_px",
            "lip_width_px", "lip_width_sapiens_px", "lip_width_err_px",
            "expr_l2", "expr_max_abs", "expr_over_3", "expr_at_cap",
            "legacy_skin_err_px", "legacy_ear_err_px",
            "identity_l2", "identity_l2_raw", "delta_beta_l2", "beta_max_abs", "beta_over_10",
            "delta_global_rot_rad", "delta_body_pose_l2_rad", "delta_body_pose_max_rad",
            "delta_hand_pose_l2_rad", "delta_body_no_hands_l2_rad", "delta_body_no_hands_max_rad",
            "delta_scale_l2", "delta_scale_max", "delta_trans_inparams_m", "delta_cam_t_m",
            "mesh_height_m", "mesh_height_raw_m", "mesh_height_delta_pct",
            "face_in_bbox_frac", "body_in_frame_frac",
            "finite_verts", "finite_landmarks", "finite_uv",
            "disp_px_face_skin", "disp_px_ear", "disp_px_lips", "disp_px_body",
            "disp_m_face_skin", "disp_m_body"]
    csv_path = OUT / f"summary_{args.run}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[out] {OUT / f'summary_{args.run}.json'}  +  {csv_path}")
    print(f"[out] renders under {OUT / '<stem>'}/")
    print(f"[time] {time.time() - t:.1f} s")
    return 0


def fit_height(img, h):
    """Resize a panel to a fixed height (upscales tiny head crops so they are readable)."""
    return cv2.resize(img, (max(1, int(round(img.shape[1] * h / img.shape[0]))), h),
                      interpolation=cv2.INTER_CUBIC)


def render_compare(stem_dir, entry, ref, results, bgr, faces, run):
    """Overlay + face + clay comparison panels: photo | raw | identity | joint."""
    W, H = entry["size_wh"]
    kp = np.asarray(entry["sapiens_kp"], np.float64)
    crop = head_crop_box(kp[SKIN_ROWS_LEGACY], W, H, pad=0.25)
    names = [n for n in ("raw_3db", "identity_only", "joint") if n in results]
    labels = {"raw_3db": "raw 3DB", "identity_only": "identity-only", "joint": "joint"}
    label_extra = {"raw_3db": "", "identity_only": "", "joint": ""}
    if "identity_only" in results and "joint" in results:
        label_extra["identity_only"] = f" |dβ|={results['identity_only']['delta_beta_l2']:.2f}"
        label_extra["joint"] = (f" |dβ|={results['joint']['delta_beta_l2']:.2f}"
                                f" dpose={results['joint']['delta_global_rot_rad']:.2f}rad")
    renderer = Renderer(W, H)
    FACE_H, FULL_H = 480, 760
    try:
        face_panels = []
        face_labels = []
        full_panels = []
        full_labels = []
        clay_panels = []
        clay_labels = []
        photo_kps = draw_kps(bgr, kp, FACE_SKIN_ROWS)
        photo_crop = photo_kps[crop[1]:crop[3], crop[0]:crop[2]].copy()
        for n in names:
            m = results[n]
            cam_t = m["_cam_t"]
            cam_verts = m["_verts"] + np.asarray(cam_t, np.float64)[None]
            vcol = texture_vertices(cam_verts, faces, bgr, m["_focal"], W, H)
            timg, _tmask = renderer.render(cam_verts, faces, m["_focal"], vcol=None)
            timg_t, tmask_t = renderer.render(cam_verts, faces, m["_focal"], vcol=vcol)
            # textured overlay on the photo
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            a = cv2.GaussianBlur(tmask_t.astype(np.float32), (0, 0), 1.0)[..., None] * 0.55
            over = np.clip(rgb * (1 - a) + timg_t.astype(np.float32) * a, 0, 255).astype(np.uint8)
            full = cv2.cvtColor(over, cv2.COLOR_RGB2BGR)
            full_panels.append(fit_height(full, FULL_H))
            full_labels.append(f"{labels[n]}{label_extra[n]}")
            face_panels.append(fit_height(full[crop[1]:crop[3], crop[0]:crop[2]], FACE_H))
            face_labels.append(f"{labels[n]}{label_extra[n]}")
            # clay render, head crop
            clay = cv2.cvtColor(timg, cv2.COLOR_RGB2BGR)
            clay_panels.append(fit_height(clay[crop[1]:crop[3], crop[0]:crop[2]], FACE_H))
            clay_labels.append(f"clay {labels[n]}{label_extra[n]}")
        face_panels = [fit_height(photo_crop, FACE_H)] + face_panels
        face_labels = ["photo + Sapiens2 face kps"] + face_labels
        full_panels = [fit_height(bgr, FULL_H)] + full_panels
        full_labels = ["photo"] + full_labels
        clay_panels = [fit_height(photo_crop, FACE_H)] + clay_panels
        clay_labels = ["photo"] + clay_labels
        cv2.imwrite(str(stem_dir / f"face_compare_{run}.jpg"),
                    compose_panels(face_panels, face_labels))
        cv2.imwrite(str(stem_dir / f"overlay_compare_{run}.jpg"),
                    compose_panels(full_panels, full_labels))
        cv2.imwrite(str(stem_dir / f"clay_compare_{run}.jpg"),
                    compose_panels(clay_panels, clay_labels))
    finally:
        renderer.close()


def render_expr_compare(stem_dir, entry, results, bgr, faces, run):
    """Close-up face panels for judging the 72 expression coefficients: photo | identity-only |
    joint with expr=0 | joint with the solved expression. Textured mesh over the photo."""
    W, H = entry["size_wh"]
    kp = np.asarray(entry["sapiens_kp"], np.float64)
    crop = head_crop_box(kp[SKIN_ROWS_LEGACY], W, H, pad=0.18)
    labels = {"identity_only": "identity-only", "joint": "joint (no expr)",
              "joint_expr_off": "joint, expr=0", "joint_expr": "joint + expr"}
    names = [n for n in ("identity_only", "joint", "joint_expr_off", "joint_expr") if n in results]
    renderer = Renderer(W, H)
    PANEL_H = 560
    try:
        photo_kps = draw_kps(bgr, kp, FACE_SKIN_ROWS)
        panels = [fit_height(photo_kps[crop[1]:crop[3], crop[0]:crop[2]], PANEL_H)]
        titles = ["photo + Sapiens2 mouth kps"]
        for n in names:
            m = results[n]
            cam_verts = m["_verts"] + np.asarray(m["_cam_t"], np.float64)[None]
            vcol = texture_vertices(cam_verts, faces, bgr, m["_focal"], W, H)
            timg_t, tmask_t = renderer.render(cam_verts, faces, m["_focal"], vcol=vcol)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            a = cv2.GaussianBlur(tmask_t.astype(np.float32), (0, 0), 1.0)[..., None] * 0.55
            over = np.clip(rgb * (1 - a) + timg_t.astype(np.float32) * a, 0, 255).astype(np.uint8)
            full = cv2.cvtColor(over, cv2.COLOR_RGB2BGR)
            txt = (f"{labels[n]} lips={m['err_lips']:.1f}px"
                   if m.get("err_lips") is not None else labels[n])
            if n in ("joint_expr", "joint_expr_off") and m.get("expr_l2"):
                txt += f" |e|={m['expr_l2']:.2f}"
            panels.append(fit_height(full[crop[1]:crop[3], crop[0]:crop[2]], PANEL_H))
            titles.append(txt)
        cv2.imwrite(str(stem_dir / f"face_compare_expr.jpg"), compose_panels(panels, titles))
        # a tighter mouth-only strip makes the lip shape readable
        lip = kp[LIP_ROWS]
        lx1, ly1 = np.floor(lip.min(0)).astype(int)
        lx2, ly2 = np.ceil(lip.max(0)).astype(int)
        pad = int(0.55 * max(lx2 - lx1, ly2 - ly1))
        lx1, ly1 = max(lx1 - pad, 0), max(ly1 - pad, 0)
        lx2, ly2 = min(lx2 + pad, W), min(ly2 + pad, H)
        m_panels = [fit_height(photo_kps[ly1:ly2, lx1:lx2], 360)]
        m_titles = ["photo"]
        for n in names:
            m = results[n]
            cam_verts = m["_verts"] + np.asarray(m["_cam_t"], np.float64)[None]
            vcol = texture_vertices(cam_verts, faces, bgr, m["_focal"], W, H)
            timg_t, tmask_t = renderer.render(cam_verts, faces, m["_focal"], vcol=vcol)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            a = cv2.GaussianBlur(tmask_t.astype(np.float32), (0, 0), 1.0)[..., None] * 0.55
            over = np.clip(rgb * (1 - a) + timg_t.astype(np.float32) * a, 0, 255).astype(np.uint8)
            full = cv2.cvtColor(over, cv2.COLOR_RGB2BGR)
            m_panels.append(fit_height(full[ly1:ly2, lx1:lx2], 360))
            m_titles.append(labels[n])
        cv2.imwrite(str(stem_dir / f"mouth_compare_expr.jpg"), compose_panels(m_panels, m_titles))
    finally:
        renderer.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["baseline", "joint", "expr", "all"], default="all")
    ap.add_argument("--prep", action="store_true", help="only build the prep cache")
    ap.add_argument("--refresh-prep", action="store_true")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--stems", nargs="*", default=None)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    specs = case_specs()
    if a.stems:
        specs = [s for s in specs if s["stem"] in a.stems]
    if a.prep:
        build_prep(specs, refresh=True)
        print(f"[vram] peak {torch.cuda.max_memory_allocated() / 2**20:.1f} MB")
        return 0
    return run_eval(a)


if __name__ == "__main__":
    sys.exit(main())
