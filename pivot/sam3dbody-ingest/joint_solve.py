"""Joint identity + pose + scale refinement of SAM 3D Body against Sapiens2 308 keypoints.

The existing identity-only refinement (``run_all.py::identity_refine``) re-solves the 45 MHR
identity coefficients with pose, scale and camera frozen at 3DB's values. It drops the face error a
lot but it has to explain *everything* with body shape, so it hits its |delta beta| = 5 cap on two
images and it cannot fix a mis-posed jaw/neck or a slightly wrong body scale.

This file solves all of it together, against the same Sapiens2 308 keypoints:

    free parameters                        dim   init / cap
    identity  beta                         45    identity-only result (or 3DB's); |d|_2 <= 5 from 3DB, |b|<=10
    global rotation (euler delta)           3    model_params[3:6];            |d| <= 0.2 rad / comp
    body pose (euler delta)                70    model_params[6:136] free idx;  |d| <= 0.2 rad / comp
    scale state (latent delta)             28    out["scale_params"];          |d|_2 <= 2
    camera translation (metres)             3    cam_t;                        |d|_2 <= 0.10 m

The 130 body-pose entries in ``model_params[6:136]`` are 124 euler angles + 6 joint translations.
The 54 hand euler values (indices 62..115) and the 6 translation entries (124..129) are held at
3DB's value: hands are not in this loss, and letting them move only invents bend. That leaves 70
free body-euler deltas.

**Scale.** ``model_params[136:204]`` is the 68-d *scale state*, produced by
``scale_mean + scale_params @ scale_comps`` from the 28-d latent that 3DB predicts. We optimize the
latent (28) and rebuild the state each step, so the result stays on the model's scale manifold. If
the checkpoint's own latent does not reproduce the state in ``mhr_model_params`` (checked at
runtime), the code falls back to optimizing the 68 state directly.

**Translation.** ``model_params[:3]`` are ``global_trans * 10`` and 3DB calls ``mhr_forward`` with
``global_trans = 0``, keeping the translation in ``pred_cam_t`` instead - the established
projection convention is ``verts + cam_t``. Putting a translation delta inside ``model_params[:3]``
would therefore be counted twice by every downstream consumer of ``mesh_and_landmarks`` +
``project_cam``. So the translation is optimized as a 3-vector added to ``cam_t`` in metres
(the same physical quantity) and returned as the refined ``cam_t``.

**Loss.** Huber (delta = 6 px) on the Euclidean pixel error to Sapiens2's 308 keypoints, weighted
per row group and normalised by the total weight:

    rows   0-  4  head/vertex blends      weight 3
    rows   5- 69  body / feet / hands     weight 0.2 (only where Sapiens2 and the 3DB projection are
                                          both in frame and the projected z > 0)
    rows  70-219  face skin               weight 1
    rows 220-271  ears (the width anchor) weight 3
    rows 272-307  iris / pupil            EXCLUDED - MHR has no eyeball geometry

plus small L2 regularisers toward the 3DB / identity-only solution: identity 0.02, pose 0.05,
scale 0.02, global rotation 0.1, translation 2.0 (all on the same scale as the
``/100``-normalised data term, i.e. ~one pixel^2 per 100).

The forward is ``model.head_pose.mhr(identity[1,45], model_params[1,204], expr[1,72])`` (a
TorchScript module, differentiable w.r.t. both identity and model_params), then
``keypoint_mapping @ cat(verts, joints) / 100``, the camera-frame sign flip ``[1,-1,-1]``, ``+cam_t``
and a pinhole projection with ``f`` and ``(W/2, H/2)`` - exactly the convention of
``run_all.py::mesh_and_landmarks`` / ``project_cam``.

Usage (from A:\\dev\\GNM-V):
    .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\joint_solve.py --case dancing
    .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\joint_solve.py --all
    .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\joint_solve.py --all --body-weight 1.0

Writes ``outputs/joint-test/<stem>.json`` (params + before/identity-only/joint metrics) and
``outputs/joint-test/report.md`` (the comparison table). ``solve_joint`` is the function to call from
``run_all.py``:

    from joint_solve import solve_joint
    res = solve_joint(model, out, kp, score, W, H, f, cam_t,
                      beta_init=identity_only_beta, iters=500)
    if res["applied"]:
        beta, params, cam_t = res["shape_params"], res["mhr_model_params"], res["cam_t"]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(REPO / ".cache" / "torch"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

THREEDB = REPO / "pivot" / "sam-3d-body"
THREEDB_CKPT = THREEDB / "sam-3d-body-dinov3" / "checkpoints" / "model.ckpt"
THREEDB_MHR = THREEDB / "sam-3d-body-dinov3" / "checkpoints" / "assets" / "mhr_model.pt"
SAPH = REPO / "pivot" / "sapiens2"
SAPIENS_W = Path(r"D:\AI_Models\3_Vision_Perception\Segmentation\facebook\Sapiens2")

DEFAULT_INPUT = HERE / "input-images"
DEFAULT_OUTPUT = HERE / "outputs"
DEFAULT_TEST = HERE / "outputs" / "joint-test"
DANCING = THREEDB / "notebook" / "images" / "dancing.jpg"
DANCING_BBOX = [660, 210, 1510, 1430]

EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

import torch  # noqa: E402

VRAM_FRACTION = 0.40          # <= ~9.6 GB: the card is shared with another process

# --------------------------------------------------------------------------- keypoint row groups
HEAD_ROWS = np.arange(0, 5)          # vertex blends, usable
BODY_ROWS = np.arange(5, 70)         # 5-20 body, 21-69 hands/feet
FACE_ROWS = np.arange(70, 220)       # brows / lids / nose / lips
MOUTH_ROWS = np.arange(188, 220)     # mouth corners + outer/inner lips (what "expression" mostly moves)
EAR_ROWS = np.arange(220, 272)       # 52 ear points - the width anchor
IRIS_ROWS = np.arange(272, 308)      # EXCLUDED: no eyeball geometry in MHR

BASE_WEIGHT = np.zeros(308, np.float64)
BASE_WEIGHT[HEAD_ROWS] = 3.0
BASE_WEIGHT[FACE_ROWS] = 1.0
BASE_WEIGHT[MOUTH_ROWS] = 3.0        # smile/lip shape should not be out-voted by the ~100 eye-contour rows
BASE_WEIGHT[EAR_ROWS] = 3.0
BASE_WEIGHT[BODY_ROWS] = 0.2

# free body-euler indices inside model_params[6:136]: 124 eulers, minus the 54 hand joints,
# never the 6 joint-translation entries (124..129 are translations, not angles).
_HANDS = set(range(62, 116))
FREE_BODY_IDX = np.array([i for i in range(124) if i not in _HANDS], np.int64)

HUBER_DELTA = 6.0
IDENTITY_CAP = 5.0
IDENTITY_ABS = 10.0
ROT_CAP = 0.2
BODY_CAP = 0.2
SCALE_CAP = 2.0
TRANS_CAP_M = 0.10
EXPR_CAP = 3.0                # L2 norm of the 72 MHR expression coefficients
NORMAL_WEIGHT = 0.5          # weight of the skin-normal cosine term (1 - n·n) vs the keypoint data term
EXPR_REG = 0.02               # L2 regularisation toward the neutral expression


# --------------------------------------------------------------------------- models / inference
def sapiens_imports() -> None:
    sys.path.insert(0, str(SAPH))
    if "sapiens" not in sys.modules:
        pkg = types.ModuleType("sapiens")
        pkg.__path__ = [str(SAPH / "sapiens")]
        pkg.__version__ = "2.0.0"
        sys.modules["sapiens"] = pkg
    import sapiens.backbones  # noqa: F401


def load_pose():
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
    return model, est


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
    import cv2

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = est.process_one_image(rgb, bboxes=np.asarray([bbox], np.float32), inference_type="body")[0]
    return out, float(out["focal_length"])


# --------------------------------------------------------------------------- projection / metrics
def _project_np(hp, beta, params, cam_t, f, W, H, expr=None):
    """308 landmarks in pixels for one (identity, model_params, cam_t, expression) tuple, no grad."""
    dev = params.device
    if expr is None:
        face = torch.zeros(1, 72, device=dev)
    else:
        face = expr if expr.dim() == 2 else expr[None]
    with torch.no_grad():
        v, sk = hp.mhr(beta[None], params[None], face)
        p3 = (hp.keypoint_mapping.detach().float() @ torch.cat([v[0], sk[0, :, :3]], 0)).cpu().numpy()
    p3 = p3 / 100.0 * np.array([1.0, -1.0, -1.0]) + np.asarray(cam_t, np.float64)[None]
    z = p3[:, 2]
    uv = np.stack([f * p3[:, 0] / z + W / 2, f * p3[:, 1] / z + H / 2], 1)
    return p3, uv


def _eval_masks(kp, score, uv0, p3_0, W, H, score_thr=0.2):
    """In-frame/visible selections, fixed once from the 3DB base projection (same for before/after)."""
    inf = (np.isfinite(kp).all(1) & (kp[:, 0] >= 0) & (kp[:, 0] < W)
           & (kp[:, 1] >= 0) & (kp[:, 1] < H) & (score > score_thr))
    uvinf = (np.isfinite(uv0).all(1) & (uv0[:, 0] >= 0) & (uv0[:, 0] < W)
             & (uv0[:, 1] >= 0) & (uv0[:, 1] < H))
    return {
        "head": HEAD_ROWS[inf[HEAD_ROWS]],
        "body": BODY_ROWS[inf[BODY_ROWS] & uvinf[BODY_ROWS] & (p3_0[BODY_ROWS, 2] > 0)],
        "face": FACE_ROWS[inf[FACE_ROWS]],
        "ear": EAR_ROWS[inf[EAR_ROWS]],
    }


def _metrics(uv, kp, masks):
    d = np.linalg.norm(uv - kp, axis=1)
    out = {}
    for name, sel in masks.items():
        out[f"{name}_err_px"] = round(float(np.mean(d[sel])), 2) if len(sel) else None
        out[f"n_{name}"] = int(len(sel))
    out["ear_span_px"] = round(float(np.linalg.norm(uv[241] - uv[267])), 1)
    out["ear_span_px_sapiens"] = round(float(np.linalg.norm(kp[241] - kp[267])), 1)
    out["skin_err_px"] = out["face_err_px"]
    return out


# --------------------------------------------------------------------------- identity-only (baseline)
def solve_identity_only(model, out, kp, score, W, H, f, cam_t, iters=700, ear_weight=3.0,
                        delta_cap=IDENTITY_CAP):
    """The existing identity-only refinement (mirror of run_all.py::identity_refine), plus metrics.

    Reproduced here rather than imported so the baseline is computed by the same metric code as
    the joint solve (and so importing this module never runs run_all's 95 % VRAM cap).
    """
    hp = model.head_pose
    dev = torch.device("cuda")
    MAP = hp.keypoint_mapping.detach().float()
    params = torch.as_tensor(np.asarray(out["mhr_model_params"], np.float32), device=dev)
    beta0 = torch.as_tensor(np.asarray(out["shape_params"], np.float32), device=dev)
    cam_t_t = torch.as_tensor(np.asarray(cam_t, np.float32), device=dev)
    centre = torch.tensor([W / 2, H / 2], dtype=torch.float32, device=dev)

    def inb(idx):
        return ((kp[idx, 0] >= 0) & (kp[idx, 0] < W) & (kp[idx, 1] >= 0) & (kp[idx, 1] < H)
                & (score[idx] > 0.2))

    sel_face, sel_ear, sel_body = FACE_ROWS[inb(FACE_ROWS)], EAR_ROWS[inb(EAR_ROWS)], BODY_ROWS[inb(BODY_ROWS)]
    sel = np.concatenate([sel_face, sel_ear, sel_body])
    if len(sel_ear) < 10 or len(sel) < 60:
        return {"applied": False, "reason": "too few in-frame face/ear keypoints"}
    w = np.concatenate([np.ones(len(sel_face)), np.full(len(sel_ear), ear_weight),
                        np.full(len(sel_body), 0.2)]).astype(np.float32)
    tgt = torch.as_tensor(kp[sel], dtype=torch.float32, device=dev)
    wt = torch.as_tensor(w, dtype=torch.float32, device=dev)

    def landmarks(beta):
        verts, sk = hp.mhr(beta[None], params[None], torch.zeros(1, 72, device=dev))
        p3 = (MAP @ torch.cat([verts[0], sk[0, :, :3]], 0)) / 100.0
        p3 = torch.stack([p3[:, 0], -p3[:, 1], -p3[:, 2]], 1) + cam_t_t[None]
        return p3, p3[:, :2] / p3[:, 2:3] * f + centre

    def huber(e, d=HUBER_DELTA):
        a = e.abs()
        return torch.where(a < d, 0.5 * e * e, d * (a - 0.5 * d))

    beta = beta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([beta], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=0.002)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        _, uv = landmarks(beta)
        e = torch.linalg.norm(uv[sel] - tgt, dim=1)
        loss = (huber(e) * wt).sum() / wt.sum() / 100.0 + 0.01 * ((beta - beta0) ** 2).sum()
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            d = beta - beta0
            n = float(d.norm())
            if n > delta_cap:
                beta.data.copy_(beta0 + d * (delta_cap / n))

    beta_np = beta.detach().cpu().numpy().astype(np.float64)
    p3, uv = _project_np(hp, beta, params, cam_t, f, W, H)
    masks = _eval_masks(kp, score, uv, p3, W, H)
    return {"applied": True, "beta": beta_np,
            "delta_l2": round(float(np.linalg.norm(beta_np - np.asarray(out["shape_params"]))), 3),
            "metrics": _metrics(uv, kp, masks), "masks": masks}


# --------------------------------------------------------------------------- the joint solve
def solve_joint(model, out, kp, score, W, H, f, cam_t, *, beta_init=None, iters=500,
                body_weight=0.2, verbose=False, identity_only=None, identity_iters=700,
                fit_expr=True, mouth_weight=None, normal_target=None, normal_mask=None,
                normal_weight=NORMAL_WEIGHT, faces=None, normal_max_dim=700):
    """Jointly refine identity + global rotation + body pose + scale state + translation [+ expression].

    ``fit_expr=True`` also frees the 72 MHR face-expression coefficients (3DB's head predicts them
    but zeroes them before the rig, so the neutral face is a code decision, not a model limit).

    Args:
        model: the SAM 3D Body model (``model.head_pose`` carries ``mhr``, ``keypoint_mapping``,
            ``scale_mean``, ``scale_comps``).
        out: one 3DB output dict (needs ``shape_params``, ``mhr_model_params``, ``scale_params``).
        kp, score: Sapiens2 308 keypoints (pixels) and confidences.
        W, H, f: image size and 3DB focal length in pixels.
        cam_t: 3DB ``pred_cam_t`` (metres).
        beta_init: 45-d identity to start from (identity-only result); defaults to 3DB's.

    ``normal_target``/``normal_mask``: optional Sapiens2 normal map (in 3DB's camera frame) and a bool
    pixel mask (skin, no hair/cloth/eyes/mouth). When given, a Huber cosine loss on the normals of
    visible skin vertices is added, which constrains the jaw/cheek/forehead curvature that keypoints miss.

    Returns dict with ``applied``, ``shape_params`` (45), ``mhr_model_params`` (204),
    ``cam_t`` (3), ``expr_params`` (72), ``metrics`` {before, identity_only, joint},
    ``caps_bound``, ``seconds``.
    """
    t_start = time.time()
    hp = model.head_pose
    dev = torch.device("cuda")
    MAP = hp.keypoint_mapping.detach().float()
    p0 = torch.as_tensor(np.asarray(out["mhr_model_params"], np.float32), device=dev)
    beta_3db = torch.as_tensor(np.asarray(out["shape_params"], np.float32), device=dev)
    cam_t_t = torch.as_tensor(np.asarray(cam_t, np.float32), device=dev)
    centre = torch.tensor([W / 2, H / 2], dtype=torch.float32, device=dev)
    expr0 = torch.zeros(1, 72, device=dev)
    expr = torch.zeros(72, device=dev, requires_grad=bool(fit_expr))

    def expr_arg():
        return expr[None] if fit_expr else expr0

    idx_t = torch.as_tensor(FREE_BODY_IDX, dtype=torch.long, device=dev)

    beta_anchor = (torch.as_tensor(np.asarray(beta_init, np.float32), device=dev)
                   if beta_init is not None else beta_3db.clone())

    # --- scale: optimize the 28-d latent that generates model_params[136:204]
    scale_mean = hp.scale_mean.detach().float()
    scale_comps = hp.scale_comps.detach().float()
    s0 = np.asarray(out.get("scale_params", np.zeros(28, np.float32)), np.float32).reshape(-1)
    s0_t = torch.as_tensor(s0, device=dev)
    scale_latent_ok = True
    if s0.size == 28:
        with torch.no_grad():
            recon = scale_mean + s0_t @ scale_comps
            if float((recon - p0[136:204]).abs().max()) > 2e-3:
                scale_latent_ok = False
    if scale_latent_ok:
        d_scale = s0.size
        base_state = None
    else:                                     # fall back to the 68-d state directly
        d_scale = 68
        base_state = p0[136:204].clone()

    # --- fixed selections (in-frame, both projections visible, z > 0)
    with torch.no_grad():
        p3_b, uv_b, _ = _project_forward(hp, MAP, p0, beta_3db, cam_t_t, expr_arg(), scale_mean,
                                      scale_comps, s0_t, scale_latent_ok, base_state,
                                      torch.zeros(3, device=dev), torch.zeros(len(FREE_BODY_IDX), device=dev),
                                      torch.zeros(d_scale, device=dev), torch.zeros(3, device=dev), f, centre)
    p3_b_np, uv_b_np = p3_b.cpu().numpy(), uv_b.cpu().numpy()
    masks = _eval_masks(kp, score, uv_b_np, p3_b_np, W, H)

    sel = np.concatenate([masks["head"], masks["face"], masks["ear"], masks["body"]])
    if len(sel) < 60:
        return {"applied": False, "reason": f"too few in-frame keypoints ({len(sel)})",
                "seconds": round(time.time() - t_start, 2)}
    wts = BASE_WEIGHT.copy()
    wts[BODY_ROWS] = body_weight
    if mouth_weight is not None:
        wts[MOUTH_ROWS] = mouth_weight
    w_np = wts[sel]
    tgt = torch.as_tensor(kp[sel], dtype=torch.float32, device=dev)
    wt = torch.as_tensor(w_np, dtype=torch.float32, device=dev)

    # --- optional skin-normal supervision (fixed visible vertices, fixed correspondences)
    faces_np = (np.asarray(faces, np.int32) if faces is not None
                else np.asarray(hp.faces.detach().cpu().numpy(), np.int32))
    faces_t = torch.as_tensor(faces_np, dtype=torch.long, device=dev)
    nsup = _setup_normal_supervision(hp, MAP, p0, beta_anchor, cam_t_t,
                                     expr_arg().detach() if fit_expr else expr0,
                                     f, W, H, faces_np, normal_target, normal_mask,
                                     max_dim=normal_max_dim)
    n_idx = (torch.as_tensor(nsup["idx"], dtype=torch.long, device=dev) if nsup else None)
    n_tgt = (torch.as_tensor(nsup["tgt"], dtype=torch.float32, device=dev) if nsup else None)

    beta = beta_anchor.clone().requires_grad_(True)
    drot = torch.zeros(3, device=dev, requires_grad=True)
    dbody = torch.zeros(len(FREE_BODY_IDX), device=dev, requires_grad=True)
    ds = torch.zeros(d_scale, device=dev, requires_grad=True)
    dcam = torch.zeros(3, device=dev, requires_grad=True)

    opt = torch.optim.Adam([
        {"params": [beta], "lr": 0.02},
        {"params": [drot], "lr": 0.003},
        {"params": [dbody], "lr": 0.01},
        {"params": [ds], "lr": 0.01},
        {"params": [dcam], "lr": 0.002},
        {"params": [expr], "lr": 0.02},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1), eta_min=0.001)

    def huber(e, d=HUBER_DELTA):
        a = e.abs()
        return torch.where(a < d, 0.5 * e * e, d * (a - 0.5 * d))

    def run(beta, drot, dbody, ds, dcam):
        return _project_forward(hp, MAP, p0, beta, cam_t_t, expr_arg(), scale_mean, scale_comps, s0_t,
                                scale_latent_ok, base_state, drot, dbody, ds, dcam, f, centre)

    last = None
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        _, uv, verts = run(beta, drot, dbody, ds, dcam)
        e = torch.sqrt(((uv[sel] - tgt) ** 2).sum(1) + 1e-12)
        data = (huber(e) * wt).sum() / wt.sum() / 100.0
        if n_idx is not None:
            vn = _vertex_normals_torch(verts, faces_t)
            cos_n = (vn[n_idx] * n_tgt).sum(1).clamp(-1.0, 1.0)
            data = data + normal_weight * (1.0 - cos_n).mean()
        reg = (0.02 * ((beta - beta_anchor) ** 2).sum()
               + 0.05 * (dbody ** 2).sum()
               + 0.02 * (ds ** 2).sum()
               + 0.10 * (drot ** 2).sum()
               + 2.00 * (dcam ** 2).sum()
               + (EXPR_REG * (expr ** 2).sum() if fit_expr else 0.0))
        (data + reg).backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            d = beta.data - beta_3db
            n = float(d.norm())
            if n > IDENTITY_CAP:
                beta.data.copy_(beta_3db + d * (IDENTITY_CAP / n))
            beta.data.clamp_(-IDENTITY_ABS, IDENTITY_ABS)
            dbody.data.clamp_(-BODY_CAP, BODY_CAP)
            drot.data.clamp_(-ROT_CAP, ROT_CAP)
            n = float(ds.data.norm())
            if n > SCALE_CAP:
                ds.data.mul_(SCALE_CAP / n)
            n = float(dcam.data.norm())
            if n > TRANS_CAP_M:
                dcam.data.mul_(TRANS_CAP_M / n)
            if fit_expr:
                n = float(expr.data.norm())
                if n > EXPR_CAP:
                    expr.data.mul_(EXPR_CAP / n)
        if verbose and (it % 100 == 0 or it == iters - 1):
            with torch.no_grad():
                last = (float(data), float(reg), float(e.mean()))
            print(f"    it {it:4d}  data {last[0]:.4f}  reg {last[1]:.4f}  mean px {last[2]:.2f}")

    # --- final params / projection
    with torch.no_grad():
        p3, uv, verts_f = run(beta, drot, dbody, ds, dcam)
        params = _build_params(p0, scale_mean, scale_comps, s0_t, scale_latent_ok, base_state,
                               drot, dbody, ds)
        cam_t_out = cam_t_t + dcam
        p3_j, uv_j = _project_np(hp, beta, params, cam_t_out.cpu().numpy(), f, W, H,
                                 expr=expr.detach())
    uv_j_np = uv_j
    m_joint = _metrics(uv_j_np, kp, masks)
    normal_final = None
    if n_idx is not None:
        with torch.no_grad():
            vn = _vertex_normals_torch(verts_f, faces_t)
            ang = torch.rad2deg(torch.arccos((vn[n_idx] * n_tgt).sum(1).clamp(-1.0, 1.0)))
        normal_final = {"n_verts": int(len(n_idx)), "angle_init_deg": round(nsup["angle0"], 2),
                        "angle_final_deg": round(float(ang.mean()), 2),
                        "angle_median_deg": round(float(ang.median()), 2),
                        "weight": float(normal_weight)}

    # before (3DB) and identity-only, measured with the same masks
    m_before = _metrics(uv_b_np, kp, masks)
    id_only = (identity_only if identity_only is not None
               else solve_identity_only(model, out, kp, score, W, H, f, cam_t, iters=identity_iters))
    m_id = None
    if id_only.get("applied"):
        _, uv_id = _project_np(hp, torch.as_tensor(id_only["beta"].astype(np.float32), device=dev),
                               p0, cam_t, f, W, H)
        m_id = _metrics(uv_id, kp, masks)

    d_beta_3db = (beta.detach() - beta_3db).cpu().numpy()
    d_beta_anchor = (beta.detach() - beta_anchor).cpu().numpy()
    body_delta = dbody.detach().cpu().numpy()
    scale_delta = ds.detach().cpu().numpy()
    rot_delta = drot.detach().cpu().numpy()
    trans_delta = dcam.detach().cpu().numpy()

    caps = {
        "identity_at_cap": bool(np.linalg.norm(d_beta_3db) > 0.995 * IDENTITY_CAP),
        "body_at_cap": int(np.sum(np.abs(body_delta) > 0.99 * BODY_CAP)),
        "rotation_at_cap": int(np.sum(np.abs(rot_delta) > 0.99 * ROT_CAP)),
        "scale_at_cap": bool(np.linalg.norm(scale_delta) > 0.99 * SCALE_CAP),
        "translation_at_cap": bool(np.linalg.norm(trans_delta) > 0.99 * TRANS_CAP_M),
        "expr_at_cap": bool(fit_expr and float(expr.detach().norm()) > 0.99 * EXPR_CAP),
    }

    secs = round(time.time() - t_start, 2)
    good = bool(np.isfinite(uv_j_np).all()
                and m_joint["face_err_px"] is not None
                and m_joint["face_err_px"] <= (m_before["face_err_px"] or 1e9) * 1.05)
    return {
        "applied": good,
        "seconds": secs,
        "shape_params": beta.detach().cpu().numpy().astype(np.float64),
        "mhr_model_params": params.detach().cpu().numpy().astype(np.float64),
        "cam_t": cam_t_out.detach().cpu().numpy().astype(np.float64),
        "expr_params": expr.detach().cpu().numpy().astype(np.float64) if fit_expr
        else np.zeros(72, np.float64),
        "beta_init_from": "identity_only" if beta_init is not None else "3db",
        "scale_mode": "latent28" if scale_latent_ok else "state68",
        "pose_delta_130": _body_delta_130(body_delta),
        "normal_supervision": normal_final,
        "metrics": {
            "before_3db": m_before,
            "identity_only": m_id,
            "joint": m_joint,
            "delta_beta_from_3db": round(float(np.linalg.norm(d_beta_3db)), 3),
            "delta_beta_from_init": round(float(np.linalg.norm(d_beta_anchor)), 3),
            "delta_beta_max_abs": round(float(np.abs(d_beta_3db).max()), 3),
            "identity_l2_3db": round(float(np.linalg.norm(np.asarray(out["shape_params"]))), 3),
            "identity_l2_joint": round(float(np.linalg.norm(beta.detach().cpu().numpy())), 3),
            "delta_pose_norm": round(float(np.linalg.norm(body_delta)), 3),
            "delta_pose_max_abs": round(float(np.abs(body_delta).max()), 4),
            "delta_rot_norm": round(float(np.linalg.norm(rot_delta)), 4),
            "delta_scale_norm": round(float(np.linalg.norm(scale_delta)), 3),
            "delta_trans_m": round(float(np.linalg.norm(trans_delta)), 4),
            "expr_l2": round(float(expr.detach().norm()), 3) if fit_expr else 0.0,
            "expr_max_abs": round(float(expr.detach().abs().max()), 3) if fit_expr else 0.0,
        },
        "caps_bound": caps,
        "iters": iters,
        "body_weight": body_weight,
        "fit_expr": bool(fit_expr),
    }


def _body_delta_130(body_delta):
    out = np.zeros(130, np.float64)
    out[FREE_BODY_IDX] = body_delta
    return out


def _build_params(p0, scale_mean, scale_comps, s0_t, scale_latent_ok, base_state, drot, dbody, ds):
    dev = p0.device
    dbody_full = torch.zeros(130, device=dev).scatter(0, torch.as_tensor(FREE_BODY_IDX, device=dev), dbody)
    if scale_latent_ok:
        scales = scale_mean + (s0_t + ds) @ scale_comps
    else:
        scales = base_state + ds
    return torch.cat([p0[:3], p0[3:6] + drot, p0[6:136] + dbody_full, scales], 0)


def _vertex_normals_torch(verts, faces_t):
    """Area-weighted vertex normals, differentiable w.r.t. the vertices."""
    a = verts[faces_t[:, 0]]
    b = verts[faces_t[:, 1]]
    c = verts[faces_t[:, 2]]
    fn = torch.cross(b - a, c - a, dim=1)
    fn = fn / fn.norm(dim=1, keepdim=True).clamp(min=1e-12)
    vn = torch.zeros_like(verts)
    for k in range(3):
        vn.index_add_(0, faces_t[:, k], fn)
    return vn / vn.norm(dim=1, keepdim=True).clamp(min=1e-12)


def _vertex_normals_np(verts, faces):
    a = verts[faces[:, 0]]
    b = verts[faces[:, 1]]
    c = verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def _zbuf_np(verts, faces, f, W, H, max_dim=700):
    """Coarse z-buffer for the visibility test of normal supervision (numpy, one-off)."""
    scale = min(1.0, max_dim / max(W, H))
    w, h = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    z = verts[:, 2]
    uv = np.stack([f * verts[:, 0] / np.maximum(z, 1e-6) + W / 2,
                   f * verts[:, 1] / np.maximum(z, 1e-6) + H / 2], 1) * scale
    tri = faces[z[faces].min(1) > 1e-3]
    zb = np.full((h, w), np.inf, np.float32)
    for k in range(len(tri)):
        t3 = uv[tri[k]]
        x0 = max(0, int(np.floor(t3[:, 0].min()))); x1 = min(w - 1, int(np.ceil(t3[:, 0].max())))
        y0 = max(0, int(np.floor(t3[:, 1].min()))); y1 = min(h - 1, int(np.ceil(t3[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        ax, ay = t3[0]; bx, by = t3[1]; cx, cy = t3[2]
        det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(det) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w1 = ((gx - ax) * (cy - ay) - (cx - ax) * (gy - ay)) / det
        w2 = ((bx - ax) * (gy - ay) - (gx - ax) * (by - ay)) / det
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        zz = (w0 * z[tri[k, 0]] + w1 * z[tri[k, 1]] + w2 * z[tri[k, 2]])[inside]
        rows, cols = np.nonzero(inside)
        np.minimum.at(zb, (rows + y0, cols + x0), zz.astype(np.float32))
    return zb, scale


def _setup_normal_supervision(hp, MAP, p0, beta, cam_t, expr, f, W, H, faces, target, mask,
                              max_dim=700):
    """Fix visible skin vertices and bilinearly sample their Sapiens2 target normals (once)."""
    if target is None or mask is None or faces is None:
        return None
    with torch.no_grad():
        face = (torch.zeros(1, 72, device=beta.device) if expr is None
                else (expr if expr.dim() == 2 else expr[None]))
        v, sk = hp.mhr(beta[None], p0[None], face)
        cam_np = (cam_t.detach().cpu().numpy() if torch.is_tensor(cam_t)
                  else np.asarray(cam_t, np.float64))
        verts = (v[0].cpu().numpy() / 100.0) * np.array([1.0, -1.0, -1.0]) + cam_np[None]
    vn = _vertex_normals_np(verts, faces)
    zb, zs = _zbuf_np(verts, faces, f, W, H, max_dim=max_dim)
    z = verts[:, 2]
    u = f * verts[:, 0] / np.maximum(z, 1e-6) + W / 2
    vv = f * verts[:, 1] / np.maximum(z, 1e-6) + H / 2
    inside = (u >= 0) & (u < W) & (vv >= 0) & (vv < H) & (z > 0)
    bh, bw = zb.shape
    ys = np.clip((vv * zs).astype(int), 0, bh - 1)
    xs = np.clip((u * zs).astype(int), 0, bw - 1)
    front = (vn * -verts).sum(1) > 0
    occl = z <= zb[ys, xs] + 0.02
    ok = inside & front & occl
    tnorm = np.linalg.norm(target, axis=2)
    xi = np.clip(u.astype(int), 0, W - 1)
    yi = np.clip(vv.astype(int), 0, H - 1)
    ok &= mask[yi, xi] & (tnorm[yi, xi] > 0.5)
    idx = np.nonzero(ok)[0]
    if len(idx) < 50:
        return None
    # bilinear sample of the target at the chosen vertices
    xi = np.clip(u[idx], 0, W - 1.001)
    yi = np.clip(vv[idx], 0, H - 1.001)
    x0 = np.floor(xi).astype(np.int32); y0 = np.floor(yi).astype(np.int32)
    fx = (xi - x0)[:, None]; fy = (yi - y0)[:, None]
    c00 = target[y0, x0]; c10 = target[y0, np.minimum(x0 + 1, W - 1)]
    c01 = target[np.minimum(y0 + 1, H - 1), x0]
    c11 = target[np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)]
    tgt = c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy) + c01 * (1 - fx) * fy + c11 * fx * fy
    tgt = tgt / np.maximum(np.linalg.norm(tgt, axis=1, keepdims=True), 1e-8)
    cos0 = np.clip((vn[idx] * tgt).sum(1), -1, 1)
    return {"idx": idx, "tgt": tgt.astype(np.float32),
            "angle0": float(np.degrees(np.arccos(np.mean(cos0))))}


def _project_forward(hp, MAP, p0, beta, cam_t_t, expr0, scale_mean, scale_comps, s0_t,
                     scale_latent_ok, base_state, drot, dbody, ds, dcam, f, centre):
    params = _build_params(p0, scale_mean, scale_comps, s0_t, scale_latent_ok, base_state,
                           drot, dbody, ds)
    v, sk = hp.mhr(beta[None], params[None], expr0)
    verts = (v[0] / 100.0) * torch.tensor([1.0, -1.0, -1.0], device=v.device)
    p3 = (MAP @ torch.cat([v[0], sk[0, :, :3]], 0)) / 100.0
    p3 = torch.stack([p3[:, 0], -p3[:, 1], -p3[:, 2]], 1) + cam_t_t[None] + dcam[None]
    z = p3[:, 2].clamp(min=0.01)
    uv = torch.stack([f * p3[:, 0] / z, f * p3[:, 1] / z], 1) + centre[None]
    return p3, uv, verts


# --------------------------------------------------------------------------- visuals
def _vis_three(bgr, kp, uvs, labels):
    """Cropped face/body comparison: Sapiens2 (green) vs projected 3DB landmarks (coloured dots)."""
    import cv2

    H, W = bgr.shape[:2]
    face = kp[FACE_ROWS]
    face = face[np.isfinite(face).all(1)]
    x1, y1 = np.floor(face.min(0)).astype(int)
    x2, y2 = np.ceil(face.max(0)).astype(int)
    pad = int(0.35 * max(x2 - x1, y2 - y1))
    x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
    x2, y2 = min(x2 + pad, W), min(y2 + pad, H)
    cw, ch = max(x2 - x1, 8), max(y2 - y1, 8)
    panels = []
    for uv, lab in zip([None] + list(uvs), labels):
        vis = bgr[y1:y2, x1:x2].copy()
        pts = kp if uv is None else uv
        for i, (x, y) in enumerate(pts):
            if i >= 272:
                continue
            x, y = x - x1, y - y1
            if not np.isfinite([x, y]).all() or not (0 <= x < cw and 0 <= y < ch):
                continue
            col = (60, 60, 255) if i >= 220 else ((200, 80, 255) if i >= 70 else (0, 200, 0))
            cv2.circle(vis, (int(x), int(y)), 2, col, -1)
        cv2.putText(vis, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(vis)
    h = max(p.shape[0] for p in panels)
    sep = np.full((h, 6, 3), 24, np.uint8)
    out = []
    for p in panels:
        if p.shape[0] != h:
            p = cv2.resize(p, (int(round(p.shape[1] * h / p.shape[0])), h))
        out.extend([p, sep])
    return np.hstack(out[:-1])


# --------------------------------------------------------------------------- CLI
def build_cases() -> list[dict]:
    cases = []
    for p in sorted(DEFAULT_INPUT.iterdir()) if DEFAULT_INPUT.exists() else []:
        if p.suffix.lower() not in EXTS:
            continue
        st = DEFAULT_OUTPUT / p.stem / "stats.json"
        bbox = None
        if st.exists():
            bbox = json.loads(st.read_text(encoding="utf-8")).get("person_box_xyxy")
        cases.append(dict(stem=p.stem, image=p, bbox=bbox, kind="crop"))
    if DANCING.exists():
        cases.append(dict(stem="dancing", image=DANCING, bbox=DANCING_BBOX, kind="full-body"))
    return cases


def _row(name, m_b, m_i, m_j, extra=""):
    def f(d, k):
        return "-" if d is None or d.get(k) is None else f"{d[k]:.2f}"
    return (f"| {name} | {f(m_b, 'skin_err_px')} / {f(m_i, 'skin_err_px')} / {f(m_j, 'skin_err_px')} "
            f"| {f(m_b, 'ear_err_px')} / {f(m_i, 'ear_err_px')} / {f(m_j, 'ear_err_px')} "
            f"| {f(m_b, 'ear_span_px')} / {f(m_i, 'ear_span_px')} / {f(m_j, 'ear_span_px')} "
            f"| {f(m_j, 'ear_span_px_sapiens')} "
            f"| {f(m_b, 'body_err_px')} / {f(m_i, 'body_err_px')} / {f(m_j, 'body_err_px')} {extra} |")


def main(argv=None) -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", nargs="*", default=None, help="image stems, or 'dancing'")
    ap.add_argument("--all", action="store_true", help="all input-images + dancing")
    ap.add_argument("--out", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--body-weight", type=float, default=0.2)
    ap.add_argument("--identity-iters", type=int, default=700)
    ap.add_argument("--no-vis", action="store_true")
    a = ap.parse_args(argv)

    torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION)   # hard cap: shared GPU
    torch.cuda.reset_peak_memory_stats()

    cases = build_cases()
    if not a.all and a.case:
        want = set(a.case)
        cases = [c for c in cases if c["stem"] in want]
    elif not a.all and not a.case:
        print("cases:", ", ".join(c["stem"] for c in cases))
        print("use --case <stem...> or --all")
        return 0
    if not cases:
        raise SystemExit("no matching cases")
    a.out.mkdir(parents=True, exist_ok=True)

    imgs, keypoints = {}, {}
    t0 = time.time()
    pmodel = load_pose()
    for c in cases:
        bgr = cv2.imread(str(c["image"]), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[{c['stem']}] unreadable: {c['image']}")
            continue
        imgs[c["stem"]] = bgr
        H, W = bgr.shape[:2]
        if not c["bbox"]:
            kp_any, _ = pose_run(pmodel, bgr)
            c["bbox"] = person_box(kp_any, W, H)
        kp, score = pose_run(pmodel, bgr, c["bbox"])
        keypoints[c["stem"]] = (kp, score, W, H)
        print(f"[{c['stem']}] {W}x{H} box {[round(float(v), 1) for v in c['bbox']]}  sapiens ok")
    del pmodel
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[stage] sapiens pose {time.time()-t0:.1f} s  peak {torch.cuda.max_memory_allocated()/2**20:.0f} MB")

    model, est = load_3db()
    results = {}
    for c in cases:
        stem = c["stem"]
        if stem not in keypoints:
            continue
        t = time.time()
        kp, score, W, H = keypoints[stem]
        bgr = imgs[stem]
        try:
            out, f = fit_3db(est, bgr, c["bbox"])
            cam_t0 = np.asarray(out["pred_cam_t"], np.float64)
            id_only = solve_identity_only(model, out, kp, score, W, H, f, cam_t0,
                                          iters=a.identity_iters)
            res = solve_joint(model, out, kp, score, W, H, f, cam_t0,
                              beta_init=(id_only["beta"] if id_only.get("applied") else None),
                              identity_only=id_only, iters=a.iters, body_weight=a.body_weight,
                              verbose=False)
            if not res.get("applied"):
                print(f"[{stem}] joint solve not applied: {res.get('reason')}")
            base = np.asarray(out["shape_params"], np.float64)
            p3_b, uv_b = _project_np(model.head_pose,
                                     torch.as_tensor(base, dtype=torch.float32, device="cuda"),
                                     torch.as_tensor(np.asarray(out["mhr_model_params"], np.float32), device="cuda"),
                                     cam_t0, f, W, H)
            payload = {
                "format": "mhr-joint-refine/1",
                "image": c["image"].name,
                "kind": c["kind"],
                "model": "SAM 3D Body dinov3 -> joint identity+pose+scale solve on Sapiens2 308 keypoints",
                "size_wh": [W, H],
                "bbox_xyxy": [round(float(v), 1) for v in c["bbox"]],
                "focal_px": round(f, 2),
                "config": {"iters": a.iters, "identity_only_iters": a.identity_iters,
                           "body_weight": a.body_weight,
                           "weights": {"head": 3.0, "face": 1.0, "ear": 3.0, "body": a.body_weight},
                           "caps": {"identity_delta_l2": IDENTITY_CAP, "identity_abs": IDENTITY_ABS,
                                    "rotation_per_comp": ROT_CAP, "body_euler_per_comp": BODY_CAP,
                                    "scale_delta_l2": SCALE_CAP, "translation_m": TRANS_CAP_M},
                           "huber_px": HUBER_DELTA},
                "original": {
                    "shape_params": [round(float(v), 6) for v in out["shape_params"]],
                    "mhr_model_params": [round(float(v), 6) for v in np.asarray(out["mhr_model_params"])],
                    "cam_t": [round(float(v), 6) for v in cam_t0],
                },
                "identity_only": None if not id_only.get("applied") else {
                    "shape_params": [round(float(v), 6) for v in id_only["beta"]],
                    "delta_l2": id_only["delta_l2"],
                    "metrics": res["metrics"]["identity_only"],
                },
                "joint": {
                    "applied": bool(res.get("applied")),
                    "shape_params": [round(float(v), 6) for v in res["shape_params"]],
                    "mhr_model_params": [round(float(v), 6) for v in res["mhr_model_params"]],
                    "cam_t": [round(float(v), 6) for v in res["cam_t"]],
                    "pose_delta_130": [round(float(v), 6) for v in res["pose_delta_130"]],
                    "scale_mode": res["scale_mode"],
                    "beta_init_from": res["beta_init_from"],
                    "metrics": res["metrics"]["joint"],
                    "deltas": {k: res["metrics"][k] for k in res["metrics"] if k.startswith("delta")
                               or k.startswith("identity_l2")},
                    "caps_bound": res["caps_bound"],
                    "seconds": res["seconds"],
                },
                "before_3db_metrics": res["metrics"]["before_3db"],
                "seconds": {"fit_3db": round(time.time() - t, 2)},
            }
            (a.out / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            results[stem] = payload
            print(f"[{stem}] joint: skin {payload['before_3db_metrics']['skin_err_px']:.2f} -> "
                  f"{res['metrics']['joint']['skin_err_px']:.2f} px, "
                  f"body {payload['before_3db_metrics']['body_err_px']} -> "
                  f"{res['metrics']['joint']['body_err_px']}, "
                  f"|dbeta| {res['metrics']['delta_beta_from_3db']:.2f}, "
                  f"pose {res['metrics']['delta_pose_norm']:.2f}, "
                  f"scale {res['metrics']['delta_scale_norm']:.2f}, {res['seconds']:.1f} s")
            if not a.no_vis:
                try:
                    p3_, uv_j = _project_np(model.head_pose,
                                            torch.as_tensor(res["shape_params"].astype(np.float32), device="cuda"),
                                            torch.as_tensor(res["mhr_model_params"].astype(np.float32), device="cuda"),
                                            res["cam_t"], f, W, H)
                    p3_i, uv_i = _project_np(model.head_pose,
                                             torch.as_tensor(out["shape_params"].astype(np.float32), device="cuda"),
                                             torch.as_tensor(np.asarray(out["mhr_model_params"], np.float32), device="cuda"),
                                             np.asarray(out["pred_cam_t"], np.float64), f, W, H)
                    vis = _vis_three(bgr, kp, [uv_b, uv_i, uv_j],
                                    ["Sapiens2", "3DB base", "identity-only", "joint"])
                    cv2.imwrite(str(a.out / f"{stem}_kps.jpg"),
                                cv2.resize(vis, (min(1800, vis.shape[1]), int(vis.shape[0] * min(1800, vis.shape[1]) / vis.shape[1]))))
                except Exception as exc:  # noqa: BLE001
                    print(f"[{stem}] vis failed: {type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[{stem}] FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    del est, model
    gc.collect()
    torch.cuda.empty_cache()

    # --- table + report
    hdr = ("| case | skin err px (3db / id-only / joint) | ear err px (3db / id-only / joint) "
           "| ear span px (3db / id-only / joint) | span sapiens | body err px (3db / id-only / joint) | "
           "|dβ|3db | Δβ init | pose Δ | rot Δ rad | scale Δ | trans Δ m | s | caps bound |\n"
           "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    lines, summary = [], []
    for stem, pl in results.items():
        j = pl["joint"]
        b = pl["before_3db_metrics"]
        i = (pl["identity_only"] or {}).get("metrics")
        m = j["metrics"]
        d = j["deltas"]
        cb = j["caps_bound"]
        caps = ",".join(k.replace("_at_cap", "") for k, v in cb.items() if v)
        lines.append(
            f"| {stem} | {_fmt(b,'skin_err_px')} / {_fmt(i,'skin_err_px')} / {_fmt(m,'skin_err_px')} "
            f"| {_fmt(b,'ear_err_px')} / {_fmt(i,'ear_err_px')} / {_fmt(m,'ear_err_px')} "
            f"| {_fmt(b,'ear_span_px')} / {_fmt(i,'ear_span_px')} / {_fmt(m,'ear_span_px')} "
            f"| {_fmt(m,'ear_span_px_sapiens')} "
            f"| {_fmt(b,'body_err_px')} / {_fmt(i,'body_err_px')} / {_fmt(m,'body_err_px')} "
            f"(n={b['n_body']}) "
            f"| {d['delta_beta_from_3db']:.2f} | {d['delta_beta_from_init']:.2f} "
            f"| {d['delta_pose_norm']:.3f} | {d['delta_rot_norm']:.3f} | {d['delta_scale_norm']:.3f} "
            f"| {d['delta_trans_m']:.4f} | {j['seconds']:.1f} | {caps or '-'} |")
        summary.append((stem, pl))
        print(lines[-1])

    report = ["# Joint identity + pose + scale solve against Sapiens2 keypoints", "",
              "`pivot/sam3dbody-ingest/joint_solve.py` · generated "
              f"{time.strftime('%Y-%m-%d %H:%M')}", "",
              "Free parameters: identity 45 · global rotation 3 · body euler 70 · scale latent 28 · "
              "translation 3 m. Huber(delta=6 px) on 308 Sapiens2 keypoints, weights head 3 / face 1 / "
              f"ear 3 / body {a.body_weight} (in-frame only). {a.iters} iters (identity-only baseline: "
              f"{a.identity_iters}).", "", "## Table", "", hdr, *lines, "",
              "## Per-case JSON", ""]
    report += [f"- [`{s}.json`]({s}.json) — original / identity-only / joint params + metrics"
               for s, _ in summary]
    (a.out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"\npeak allocated {peak:.0f} MB (cap {VRAM_FRACTION*100:.0f}% of "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GB)")
    print(f"wrote {a.out}")
    return 0


def _fmt(d, k):
    if d is None or d.get(k) is None:
        return "-"
    return f"{d[k]:.2f}"


if __name__ == "__main__":
    sys.exit(main())
