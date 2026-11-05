"""Run Sapiens2 (pose / depth / seg / normal) and SAM 3D Body on every image in input-images/.

For each image, writes a subfolder under outputs/<image-stem>/ with:
  01_original.<ext>            the original image, untouched
  02_sapiens_keypoints.jpg     original + Sapiens2 308 keypoints
  03_sapiens_depth.jpg         Sapiens2 pointmap depth (metric, turbo, person-masked)
  04_sapiens_seg.jpg           Sapiens2 body-part segmentation overlay (29 classes)
  05_sapiens_normal.jpg        Sapiens2 surface normals (RGB)
  06_3db_keypoints.jpg         original + SAM 3D Body 70 keypoints matched to Sapiens2's 70
  07_3db_mesh_keypoints.jpg    3DB mesh rendered over the photo + 3DB keypoints
  stats.json                   sizes, bbox, identity, focal, timings, depth range

**VRAM budget.** Models are loaded one stage at a time and freed before the next: at most ONE
Sapiens2 model or the 3DB model is resident, so peak is a few GB, not the sum of four Sapiens2
heads + 3DB. The process is hard-capped at 95 % of the 24 GB card and peak allocation is printed and
asserted after every stage.

Usage (from A:\\dev\\GNM-V):
    .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\run_all.py
    .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\run_all.py --input <dir> --out <dir> [--images a.png b.png]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import cv2

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
os.environ.setdefault("TORCH_HOME", str(REPO / ".cache" / "torch"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

THREEDB = Path(os.environ.get("SAM3DBODY_ROOT", str(REPO / "pivot" / "sam-3d-body")))
THREEDB_CKPT = THREEDB / "sam-3d-body-dinov3" / "checkpoints" / "model.ckpt"
THREEDB_MHR = THREEDB / "sam-3d-body-dinov3" / "checkpoints" / "assets" / "mhr_model.pt"
SAPH = Path(os.environ.get("SAPIENS2_ROOT", str(REPO / "pivot" / "sapiens2")))
SAPIENS_W = Path(os.environ.get("SAPIENS2_WEIGHTS",
                               r"D:\AI_Models\3_Vision_Perception\Segmentation\facebook\Sapiens2"))
SAPIENS_1B_POINTMAP = r"D:\AI_Models\3_Vision_Perception\Segmentation\facebook\Sapiens2\pointmap\sapiens2_1b_pointmap.safetensors"
SAPIENS_1B_MATTING = r"D:\AI_Models\3_Vision_Perception\Segmentation\facebook\Sapiens2\matting\sapiens2_1b_matting.safetensors"

sys.path.insert(0, str(HERE))
try:
    from joint_solve import solve_joint as _solve_joint
except Exception:  # noqa: BLE001 - optional module
    _solve_joint = None

DEFAULT_INPUT = HERE / "input-images"
DEFAULT_OUT = HERE / "outputs"

MESH_COLOUR = (0.60, 0.65, 0.75)
VRAM_BUDGET_MB = 24_000

import torch  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.95)          # hard cap below the 24 GB card


# --------------------------------------------------------------------------- helpers
def free(objest: dict) -> None:
    for v in objest.values():
        del v
    gc.collect()
    torch.cuda.empty_cache()


def vram(tag: str) -> None:
    peak = torch.cuda.max_memory_allocated() / 2**20
    now = torch.cuda.memory_allocated() / 2**20
    print(f"[vram] {tag:18s} now {now:7.1f} MB | peak {peak:7.1f} MB (budget {VRAM_BUDGET_MB} MB)")
    assert peak < VRAM_BUDGET_MB, f"VRAM peak {peak:.0f} MB over budget"
    torch.cuda.reset_peak_memory_stats()


def sapiens_imports():
    sys.path.insert(0, str(SAPH))
    if "sapiens" not in sys.modules:
        pkg = types.ModuleType("sapiens")
        pkg.__path__ = [str(SAPH / "sapiens")]
        pkg.__version__ = "2.0.0"
        sys.modules["sapiens"] = pkg
    import sapiens.backbones  # noqa: F401  (registers the Sapiens2 backbone)


def _load_log(msg: str):
    import datetime as _dt
    import threading as _th
    try:
        mem = f" cuda={torch.cuda.memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else ""
    except Exception:
        mem = ""
    print("[{t}][load][t{th}] {m}{mem}".format(
        t=_dt.datetime.now().strftime("%H:%M:%S"), th=_th.get_native_id(), m=msg, mem=mem), flush=True)


def load_pose():
    sapiens_imports()
    from sapiens.pose.datasets import UDPHeatmap, parse_pose_metainfo
    from sapiens.pose.models import init_model

    _load_log("pose: loading weights")
    _t0 = time.time()
    model = init_model(str(SAPH / "sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/"
                               "sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py"),
                       str(SAPIENS_W / "pose/sapiens2_0.4b_pose.safetensors"), device="cuda")
    _load_log("pose: ready in %.1fs" % (time.time() - _t0))
    model.pose_metainfo = parse_pose_metainfo(
        dict(from_file=str(SAPH / "sapiens/pose/configs/_base_/keypoints308.py")))
    c = dict(model.cfg.codec); c.pop("type")
    model.codec = UDPHeatmap(**c)
    return model


def load_dense(task: str, config: str, weights: str):
    sapiens_imports()
    from sapiens.dense.models import init_model as dense_init

    _load_log("%s: loading weights" % task)
    _t0 = time.time()
    m = dense_init(str(SAPH / "sapiens/dense/configs" / config), str(SAPIENS_W / weights), device="cuda")
    _load_log("%s: ready in %.1fs" % (task, time.time() - _t0))
    return m


def load_3db():
    sys.path.insert(0, str(THREEDB))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    _load_log("3db: loading weights")
    _t0 = time.time()

    model, cfg = load_sam_3d_body(str(THREEDB_CKPT), device="cuda", mhr_path=str(THREEDB_MHR))
    _load_log("3db: ready in %.1fs" % (time.time() - _t0))
    est = SAM3DBodyEstimator(model, cfg, human_detector=None, human_segmentor=None, fov_estimator=None)
    faces = np.asarray(model.head_pose.faces.detach().cpu().numpy(), np.int32)
    return model, est, faces


# --------------------------------------------------------------------------- sapiens tasks
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
    kp = kp / np.array(m["input_size"]) * np.array(m["bbox_scale"]) + np.array(m["bbox_center"]) - 0.5 * np.array(m["bbox_scale"])
    return kp, score


def person_box(kp, W, H, pad=0.10):
    inside = (np.isfinite(kp).all(1) & (kp[:, 0] >= 0) & (kp[:, 0] <= W - 1)
              & (kp[:, 1] >= 0) & (kp[:, 1] <= H - 1))
    pts = kp[inside]
    if len(pts) < 10:
        return [0, 0, W - 1, H - 1]
    x1, y1 = pts.min(0); x2, y2 = pts.max(0)
    w, h = x2 - x1, y2 - y1
    return [max(0.0, x1 - pad * w), max(0.0, y1 - pad * h),
            min(W - 1.0, x2 + pad * w), min(H - 1.0, y2 + pad * h)]


SEG_FACE_NECK = 3
SEG_EXCLUDE = (0, 1, 2, 4, 24, 25, 26, 27, 28)   # bg, apparel, eyeglass, hair, lips, teeth, tongue


def skin_mask_from_labels(labels, dilate=2, erode=2):
    """Sapiens2 seg Face_Neck minus hair/clothing/background/lips/teeth, boundary-safe."""
    import cv2

    base = labels == SEG_FACE_NECK
    excl = np.isin(labels, SEG_EXCLUDE).astype(np.uint8)
    k = np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)
    excl = cv2.dilate(excl, k).astype(bool)
    m = (base & ~excl).astype(np.uint8)
    if erode:
        m = cv2.erode(m, np.ones((2 * erode + 1, 2 * erode + 1), np.uint8))
    return m.astype(bool)


def feature_exclusion_mask(kp, W, H):
    """Pixels where the LOD1 mesh cannot match (eyeballs, nostrils, lips) - excluded from normal loss."""
    import cv2

    m = np.zeros((H, W), np.uint8)
    if not np.isfinite(kp[70:272]).all():
        return m.astype(bool)
    face = kp[70:272]
    fw = max(20.0, float(np.ptp(face[:, 0])))
    groups = [(np.array([1, 2]), 0.055 * fw),          # eye centres
              (np.arange(178, 188), 0.020 * fw),       # nostril wings / nose base
              (np.arange(188, 220), 0.018 * fw)]       # mouth corners + lips
    for rows, r in groups:
        for i in rows:
            x, y = kp[i]
            if np.isfinite([x, y]).all() and 0 <= x < W and 0 <= y < H:
                cv2.circle(m, (int(round(x)), int(round(y))), max(3, int(round(r))), 255, -1)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return m.astype(bool)


def seg_run(model, seg_vis, bgr):
    import torch.nn.functional as F

    data = model.data_preprocessor(model.pipeline(dict(img=bgr)))
    with torch.no_grad():
        logits = model(data["inputs"])
    logits = F.interpolate(logits, size=bgr.shape[:2], mode="bilinear")
    labels = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return labels, seg_vis._visualize_segmentation(bgr, labels)


def normal_run(model, bgr, seg_mask):
    import torch.nn.functional as F

    data = model.data_preprocessor(model.pipeline(dict(img=bgr)))
    with torch.no_grad():
        n = model(data["inputs"])
        n = n / torch.norm(n, dim=1, keepdim=True).clamp(min=1e-8)
    pl, pr, pt, pb = data["data_samples"]["meta"]["padding_size"]
    n = n[:, :, pt:data["inputs"].shape[2] - pb, pl:data["inputs"].shape[3] - pr]
    n = F.interpolate(n, size=bgr.shape[:2], mode="bilinear", align_corners=False)
    n = n[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    n[seg_mask == 0] = -1.0
    return n, ((n + 1) / 2 * 255).astype(np.uint8)[:, :, ::-1]


def depth_run(model, bgr, seg_mask):
    import torch.nn.functional as F

    data = model.data_preprocessor(model.pipeline(dict(img=bgr)))
    with torch.no_grad():
        pointmap, scale = model(data["inputs"])
        pointmap = pointmap / scale
    pl, pr, pt, pb = data["data_samples"]["meta"]["padding_size"]
    pointmap = pointmap[:, :, pt:data["inputs"].shape[2] - pb, pl:data["inputs"].shape[3] - pr]
    pointmap = F.interpolate(pointmap, size=bgr.shape[:2], mode="bilinear", align_corners=False)
    depth = pointmap[0, 2].cpu().numpy().astype(np.float32)

    vis = np.full((*depth.shape, 3), 100, np.uint8)
    fg = (seg_mask > 0) & np.isfinite(depth) & (depth > 0)
    if fg.any():
        lo, hi = np.percentile(depth[fg], [1, 99])
        lo, hi = max(float(lo), 1e-3), max(float(hi), lo + 1e-3)
        inv = 1.0 / np.clip(depth[fg], 1e-6, None)
        max_inv = min(1.0 / lo, 1.0 / 0.1)
        min_inv = max(1.0 / 250.0, 1.0 / hi)
        t = np.clip((inv - min_inv) / (max_inv - min_inv), 0, 1)
        t_full = np.zeros(depth.shape, np.uint8)
        t_full[fg] = (t * 255).astype(np.uint8)
        vis[fg] = cv2.applyColorMap(t_full, cv2.COLORMAP_TURBO)[fg]
    return depth, vis


# --------------------------------------------------------------------------- 3db + drawing
def matting_run(model, bgr):
    import torch.nn.functional as F

    data = model.data_preprocessor(model.pipeline(dict(img=bgr)))
    with torch.no_grad():
        a = model(data["inputs"])
    try:
        pl, pr, pt, pb = data["data_samples"]["meta"]["padding_size"]
        a = a[:, :, pt:data["inputs"].shape[2] - pb, pl:data["inputs"].shape[3] - pr]
    except Exception:  # noqa: BLE001 - matting test pipeline may not expose padding meta
        pass
    a = F.interpolate(a, size=bgr.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    return a.cpu().numpy().astype(np.float32)


def fit_3db(est, bgr, bbox):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = est.process_one_image(rgb, bboxes=np.asarray([bbox], np.float32), inference_type="body")[0]
    verts = np.asarray(out["pred_vertices"], np.float64) + np.asarray(out["pred_cam_t"], np.float64)[None, :]
    return out, verts, float(out["focal_length"])


# --------------------------------------------------------------------------- identity refinement
FACE_ROWS = np.arange(70, 308)
SKIN_ROWS = np.arange(70, 272)             # brows/lids/nose/lips + ears (iris/pupil excluded: no eyeball geometry)
EAR_ROWS = np.arange(220, 272)             # 52 ear points - the width anchor
FACE_SKIN = np.arange(70, 220)             # lower-face landmarks (brows, lids, nose, lips)
BODY_ROWS = np.arange(5, 70)


def mesh_and_landmarks(model, out, beta, model_params=None, expr=None):
    """Mesh vertices and all 308 landmarks for one identity/pose (+ expression) in 3DB's camera frame."""
    hp = model.head_pose
    shape = torch.tensor(np.asarray(beta, np.float32), device="cuda")
    p = np.asarray(model_params if model_params is not None else out["mhr_model_params"], np.float32)
    params = torch.tensor(p, device="cuda")
    face = (torch.zeros(1, 72, device="cuda") if expr is None
            else torch.tensor(np.asarray(expr, np.float32).reshape(1, 72), device="cuda"))
    with torch.no_grad():
        vcm, sk = hp.mhr(shape[None], params[None], face)
        lm = (hp.keypoint_mapping.detach().float()
              @ torch.cat([vcm[0], sk[0, :, :3]], 0)).cpu().numpy() / 100.0
    verts = np.asarray(vcm[0].cpu().numpy(), np.float64) / 100.0 * np.array([1.0, -1.0, -1.0])
    return verts, lm * np.array([1.0, -1.0, -1.0])


def project_cam(lm, f, cam_t_f, W, H):
    q = lm + np.asarray(cam_t_f, np.float64)[None]
    return np.stack([f * q[:, 0] / q[:, 2] + W / 2, f * q[:, 1] / q[:, 2] + H / 2], 1)


def identity_refine(model, out, kp, score, W, H, f, cam_t_f,
                    ear_weight=3.0, iters=700, delta_cap=5.0):
    """Re-solve the 45 MHR identity coefficients against Sapiens2's face/ear landmarks.

    The ears carry the width (weight ``ear_weight``); brows/lids/nose/lips weight 1; in-frame body
    keypoints weight 0.2 to keep global proportions sane. Pose, scale and camera stay as 3DB gave
    them, so the output is a valid 45-d MHR identity — the thing we actually want corrected.
    """
    hp = model.head_pose
    MAP = hp.keypoint_mapping.detach().float()
    params = torch.tensor(np.asarray(out["mhr_model_params"], np.float32), device="cuda")
    beta0 = torch.tensor(np.asarray(out["shape_params"], np.float32), device="cuda")
    cam_t_t = torch.tensor(np.asarray(cam_t_f, np.float32), device="cuda")
    centre = torch.tensor([W / 2, H / 2], dtype=torch.float32, device="cuda")

    def inb(idx):
        return ((kp[idx, 0] >= 0) & (kp[idx, 0] < W) & (kp[idx, 1] >= 0) & (kp[idx, 1] < H)
                & (score[idx] > 0.2))

    sel_face, sel_ear, sel_body = FACE_SKIN[inb(FACE_SKIN)], EAR_ROWS[inb(EAR_ROWS)], BODY_ROWS[inb(BODY_ROWS)]
    sel = np.concatenate([sel_face, sel_ear, sel_body])
    if len(sel_ear) < 10 or len(sel) < 60:
        return {"applied": False, "reason": "too few in-frame face/ear keypoints"}
    w = np.concatenate([np.ones(len(sel_face)), np.full(len(sel_ear), ear_weight),
                        np.full(len(sel_body), 0.2)]).astype(np.float32)
    tgt = torch.tensor(kp[sel], dtype=torch.float32, device="cuda")
    wt = torch.tensor(w, dtype=torch.float32, device="cuda")

    def landmarks(beta):
        verts, sk = hp.mhr(beta[None], params[None], torch.zeros(1, 72, device="cuda"))
        p3 = (MAP @ torch.cat([verts[0], sk[0, :, :3]], 0)) / 100.0
        p3 = torch.stack([p3[:, 0], -p3[:, 1], -p3[:, 2]], 1) + cam_t_t[None]
        return p3, p3[:, :2] / p3[:, 2:3] * f + centre

    def huber(e, d=6.0):
        a = e.abs()
        return torch.where(a < d, 0.5 * e * e, d * (a - 0.5 * d))

    def metrics(uv_np):
        skin = np.linalg.norm(uv_np[SKIN_ROWS] - kp[SKIN_ROWS], axis=1).mean()
        ear = np.linalg.norm(uv_np[EAR_ROWS] - kp[EAR_ROWS], axis=1).mean()
        ear2d = float(np.linalg.norm(uv_np[241] - uv_np[267]))
        sap_ear2d = float(np.linalg.norm(kp[241] - kp[267]))
        return float(skin), float(ear), ear2d, sap_ear2d

    beta = beta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([beta], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=0.002)
    with torch.no_grad():
        _, uv0 = landmarks(beta)
        skin0, ear0, ear2d0, sap_ear = metrics(uv0.cpu().numpy())
    for _ in range(iters):
        opt.zero_grad()
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
    with torch.no_grad():
        _, uv1 = landmarks(beta)
        skin1, ear1, ear2d1, _ = metrics(uv1.cpu().numpy())
        d = beta - beta0
        delta = float(d.norm())
    return {"applied": bool(np.isfinite(delta) and skin1 <= skin0 * 1.05),
            "beta": beta.detach().cpu().numpy().astype(np.float64),
            "delta_l2": round(delta, 3),
            "identity_l2_before": round(float(beta0.norm()), 3),
            "identity_l2_after": round(float(beta.norm()), 3),
            "skin_err_before_px": round(skin0, 2), "skin_err_after_px": round(skin1, 2),
            "ear_err_before_px": round(ear0, 2), "ear_err_after_px": round(ear1, 2),
            "ear_span_px_before": round(ear2d0, 1), "ear_span_px_after": round(ear2d1, 1),
            "ear_span_px_sapiens": round(sap_ear, 1)}


def render_mesh(verts, faces, f, W, H):
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(list(MESH_COLOUR))
    r = rendering.OffscreenRenderer(W, H)
    r.scene.set_background([0.10, 0.11, 0.13, 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [*MESH_COLOUR, 1.0]
    mat.base_roughness = 0.55
    r.scene.add_geometry("body", mesh, mat)
    r.scene.scene.set_sun_light([-0.35, -0.55, 0.75], [1, 1, 1], 28000)
    r.setup_camera(o3d.camera.PinholeCameraIntrinsic(W, H, f, f, W / 2, H / 2), np.eye(4))
    img = np.asarray(r.render_to_image()).copy()
    bg = img[2, 2].astype(int)
    return img, np.abs(img.astype(int) - bg[None, None, :]).max(2) > 8


def draw_pose(bgr, kp):
    H, W = bgr.shape[:2]
    vis = bgr.copy()
    for i, (x, y) in enumerate(kp):
        if not np.isfinite([x, y]).all():
            continue
        if i >= 70:
            cv2.circle(vis, (int(x), int(y)), 1, (200, 70, 255), -1)      # 238 face points: magenta
        else:
            cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)         # 70 body/foot/hand: green
    def text(s, y):
        cv2.putText(vis, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.42, min(0.65, W / 1200.0)), (255, 255, 255), 2, cv2.LINE_AA)
    text("Sapiens2 308 keypoints", 26)
    text("green = 70 body/foot/hand   magenta = 238 face", 50)
    return vis


def draw_3db_match(bgr, kp3, kp_sap):
    H, W = bgr.shape[:2]
    vis = bgr.copy()
    for i in range(min(70, len(kp3))):
        a, b = kp3[i], kp_sap[i]
        if (np.isfinite(a).all() and np.isfinite(b).all()
                and 0 <= a[0] < W and 0 <= a[1] < H and 0 <= b[0] < W and 0 <= b[1] < H):
            cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (150, 150, 150), 1, cv2.LINE_AA)
    for x, y in kp3[:70]:
        if np.isfinite([x, y]).all() and 0 <= x < W and 0 <= y < H:
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), 2)
    for x, y in kp_sap[:70]:
        if np.isfinite([x, y]).all() and 0 <= x < W and 0 <= y < H:
            cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
    cv2.putText(vis, "3DB kps (red) vs Sapiens2 (green), 1:1", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.45, min(0.8, W / 1600.0)), (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def vertex_normals(verts, faces):
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(b - a, c - a)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def depth_buffer(verts, faces, f, W, H, max_dim=900):
    """Z-buffer of the projected mesh at reduced resolution -> (zbuf, scale)."""
    scale = min(1.0, max_dim / max(W, H))
    w, h = max(1, int(round(W * scale))), max(1, int(round(H * scale)))
    z = verts[:, 2]
    uv = np.stack([f * verts[:, 0] / np.maximum(z, 1e-6) + W / 2,
                   f * verts[:, 1] / np.maximum(z, 1e-6) + H / 2], 1) * scale
    tri = faces[z[faces].min(1) > 1e-3]
    zbuf = np.full((h, w), np.inf, np.float32)
    for k in range(len(tri)):
        t = uv[tri[k]]
        x0 = max(0, int(np.floor(t[:, 0].min())))
        x1 = min(w - 1, int(np.ceil(t[:, 0].max())))
        y0 = max(0, int(np.floor(t[:, 1].min())))
        y1 = min(h - 1, int(np.ceil(t[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        ax, ay = t[0]; bx, by = t[1]; cx, cy = t[2]
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
        np.minimum.at(zbuf, (rows + y0, cols + x0), zz.astype(np.float32))
    return zbuf, scale


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_obj(path: Path, verts, faces, comment: str = ""):
    with open(path, "w", encoding="utf-8") as fh:
        if comment:
            fh.write(f"# {comment}\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


def _json_safe(obj):
    return json.loads(json.dumps(_jsonable(obj)))


def _jsonable(v):
    """Recursively convert numpy types so json.dumps never sees them."""
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def texture_mesh(verts, faces, bgr, f, W, H):
    """Photo colours per vertex (z-buffer visibility, linear-light shading) for an unlit render."""
    from scipy.spatial import cKDTree

    vn = vertex_normals(verts, faces)
    z = verts[:, 2]
    uv = np.stack([f * verts[:, 0] / np.maximum(z, 1e-6) + W / 2,
                   f * verts[:, 1] / np.maximum(z, 1e-6) + H / 2], 1)
    front = (vn * (-verts)).sum(1) > 0
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (z > 0)

    zbuf, zb_scale = depth_buffer(verts, faces, f, W, H)
    bh, bw = zbuf.shape
    ys = np.clip((uv[:, 1] * zb_scale).astype(int), 0, bh - 1)
    xs = np.clip((uv[:, 0] * zb_scale).astype(int), 0, bw - 1)
    visible = front & inside & (z <= zbuf[ys, xs] + 0.02)      # 2 cm depth tolerance
    vis = np.where(visible)[0]

    photo = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    colours = np.full((len(verts), 3), 0.35, np.float64)
    if len(vis):
        photo_lin = _srgb_to_linear(photo)
        # bilinear sample at the visible vertices (cv2.remap needs a 2-D map, so do it by hand)
        xi = np.clip(uv[vis, 0], 0, W - 1.001)
        yi = np.clip(uv[vis, 1], 0, H - 1.001)
        x0 = np.floor(xi).astype(np.int32); y0 = np.floor(yi).astype(np.int32)
        fx = (xi - x0)[:, None]; fy = (yi - y0)[:, None]
        c00 = photo_lin[y0, x0]; c10 = photo_lin[y0, np.minimum(x0 + 1, W - 1)]
        c01 = photo_lin[np.minimum(y0 + 1, H - 1), x0]
        c11 = photo_lin[np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)]
        colours[vis] = (c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy)
                        + c01 * (1 - fx) * fy + c11 * fx * fy)
        # seed non-visible vertices from the nearest visible one, then diffuse across the mesh graph
        missing_mask = ~visible
        if missing_mask.any():
            from scipy.spatial import cKDTree
            _, nn = cKDTree(verts[vis]).query(verts[missing_mask], k=1)
            colours[missing_mask] = colours[vis[nn]]
            edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
            edges = np.vstack([edges, edges[:, ::-1]])
            for _ in range(120):
                acc = np.zeros_like(colours)
                cnt = np.zeros(len(verts))
                np.add.at(acc, edges[:, 0], colours[edges[:, 1]])
                np.add.at(cnt, edges[:, 0], 1.0)
                upd = missing_mask & (cnt > 0)
                if not upd.any():
                    break
                colours[upd] = acc[upd] / cnt[upd, None]
    # two-light linear-light shading, then back to sRGB for the unlit renderer
    key = np.array([-0.35, -0.55, 0.75]); key /= np.linalg.norm(key)
    fill = np.array([0.55, -0.25, -0.55]); fill /= np.linalg.norm(fill)
    lam = 0.28 + 0.60 * np.clip(vn @ key, 0, 1) + 0.12 * np.clip(vn @ fill, 0, 1)
    return _linear_to_srgb(colours * lam[:, None]).astype(np.float64)


def _bg_gradient(w, h):
    top = np.array([22.0, 24.0, 28.0]); bottom = np.array([40.0, 43.0, 49.0])
    rows = np.linspace(0, 1, h)[:, None, None]
    grad = top[None, None, :] * (1 - rows) + bottom[None, None, :] * rows
    return np.repeat(grad, w, axis=1).astype(np.uint8)


def _render_once(rw, rh, rf, mesh, mat, want_depth=False):
    """One supersampled render; returns (rgb, object mask from the depth buffer),
    plus the raw depth buffer when want_depth is set."""
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    r = rendering.OffscreenRenderer(rw, rh)
    r.scene.set_background([0.0, 0.0, 0.0, 1.0])
    r.scene.add_geometry("body", mesh, mat)
    r.scene.scene.set_sun_light([-0.35, -0.55, 0.75], [1, 1, 1], 28000)
    r.setup_camera(o3d.camera.PinholeCameraIntrinsic(rw, rh, rf, rf, rw / 2, rh / 2), np.eye(4))
    img = np.asarray(r.render_to_image()).copy()
    depth = np.asarray(r.render_to_depth_image()).copy()
    # object pixels: near depth OR any rendered colour above black (fills NaN-depth triangles)
    mask = (depth < 0.995) | (img.max(axis=2) > 10)
    if want_depth:
        return img, mask, depth
    return img, mask


def _finish(img_hi, mask_hi, W, H):
    """Replace the background with the gradient at supersampled res, then downsample (clean AA edges)."""
    gh, gw = img_hi.shape[:2]
    bg_hi = cv2.resize(_bg_gradient(gw, gh), (gw, gh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    clean = np.where(mask_hi[..., None], img_hi.astype(np.float32), bg_hi)
    img = cv2.resize(clean, (W, H), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(mask_hi.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8), alpha


def _make_mesh(verts, faces, vcol=None, colour=None):
    import open3d as o3d
    import open3d.visualization.rendering as rendering

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mat = rendering.MaterialRecord()
    if vcol is not None:
        mesh.vertex_colors = o3d.utility.Vector3dVector(vcol)
        mat.shader = "defaultUnlit"
        mat.base_color = [1.0, 1.0, 1.0, 1.0]
    else:
        mesh.paint_uniform_color(list(colour or MESH_COLOUR))
        mat.shader = "defaultLit"
        mat.base_color = [*(colour or MESH_COLOUR), 1.0]
        mat.base_roughness = 0.55
    return mesh, mat


def render_rgb(verts, faces, f, W, H, vcol=None, colour=None, ss=None):
    """Render the mesh (supersampled) over a soft gradient background."""
    if ss is None:
        ss = 2.0 if W * H * 4 <= 12_000_000 else 1.4
    rw, rh = max(2, int(round(W * ss))), max(2, int(round(H * ss)))
    mesh, mat = _make_mesh(verts, faces, vcol=vcol, colour=colour)
    img_hi, mask_hi = _render_once(rw, rh, f * (rw / W), mesh, mat)
    img, alpha = _finish(img_hi, mask_hi, W, H)
    return img, alpha > 0.5


def render_clay_view(verts, faces, f, H, yaw=35.0, vcol=None):
    """A 3/4 view, framed to a square tile: clay-shaded or photo-textured (vcol) mesh."""
    p = verts - verts.mean(axis=0)
    th = np.deg2rad(yaw)
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    p = p @ rot.T
    height_m = float(np.ptp(p[:, 1]))
    f2 = 1.1 * H
    dist = f2 * height_m / (0.85 * H)
    world = p + np.array([0.0, 0.0, dist])
    mesh, mat = _make_mesh(world, faces, vcol=vcol)
    img_hi, mask_hi = _render_once(H, H, f2, mesh, mat)
    img, _ = _finish(img_hi, mask_hi, H, H)
    return img


def orbit_world(verts, H, yaw_deg=35.0, center_frac=0.0, window_frac=1.0, y_down=False):
    """Center a mesh, yaw it around Y, and frame it at a fit distance.

    The orbit framing render_clay_view uses, exposed so every orbit painter
    (open3d and the numpy fallbacks) shares it. center_frac/window_frac select a
    height window for face/waist/body framings. Returns (world_verts, focal).
    """
    v = np.asarray(verts, np.float64)
    p = v - v.mean(axis=0)
    th = np.deg2rad(yaw_deg)
    c, s = np.cos(th), np.sin(th)
    p = p @ np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]).T
    height_m = float(np.ptp(p[:, 1])) or 1e-6
    window = max(height_m * window_frac, 1e-6)
    f2 = 1.1 * H
    dist = f2 * window / (0.85 * H)
    # Face/waist framings assume +Y is the head. Camera-frame meshes exported
    # by SAM 3D Body are Y-down, so the window has to move the other way or
    # "face" lands on the legs.
    y_sign = -1.0 if y_down else 1.0
    return p - np.array([0.0, y_sign * center_frac * height_m, 0.0]) + np.array([0.0, 0.0, dist]), f2


ORBIT_FRAMINGS = {"body": (0.0, 1.0), "waist": (0.24, 0.54), "face": (0.42, 0.24)}
"""Subject windows as (center_frac, window_frac) of mesh height, tuned for upright bodies."""


def render_orbit(verts, faces, size, yaw_deg=35.0, mode="shaded", colour=None, ss=1.5, framing="body",
                 y_down=False):
    """Orbit still via the open3d offscreen painter: shaded, depth or normal.

    Same framing as render_clay_view; depth is near-bright over the true
    depth buffer, normal maps camera-space normals to RGB (unlit shader).
    framing selects a subject window (body, waist, face). Returns (rgb uint8,
    mask bool). Raises ImportError without open3d.
    """
    try:
        center_frac, window_frac = ORBIT_FRAMINGS[framing]
    except KeyError:
        raise ValueError("unknown framing '{f}' (body, waist, face)".format(f=framing))
    world, f2 = orbit_world(verts, size, yaw_deg, center_frac, window_frac, y_down=y_down)
    faces = np.asarray(faces, np.int32)
    rw = rh = max(2, int(round(size * ss)))
    rf = f2 * (rw / size)
    if mode == "normal":
        import open3d as o3d
        tmp = o3d.geometry.TriangleMesh()
        tmp.vertices = o3d.utility.Vector3dVector(world)
        tmp.triangles = o3d.utility.Vector3iVector(faces)
        tmp.compute_vertex_normals()
        vcol = np.asarray(tmp.vertex_normals) * 0.5 + 0.5
        mesh, mat = _make_mesh(world, faces, vcol=vcol)
    else:
        mesh, mat = _make_mesh(world, faces, colour=colour)
    img_hi, mask_hi, depth_hi = _render_once(rw, rh, rf, mesh, mat, want_depth=True)
    if not np.any(mask_hi):
        raise ValueError("mesh projects to no pixels at yaw {0} deg".format(yaw_deg))
    if mode == "depth":
        d = depth_hi.astype(np.float64)
        d[~mask_hi] = np.nan
        lo, hi = float(np.nanmin(d)), float(np.nanmax(d))
        g = np.zeros(d.shape, np.float64)
        if hi - lo > 1e-9:
            g[mask_hi] = (1.0 - (d[mask_hi] - lo) / (hi - lo)) * 255.0
        else:
            g[mask_hi] = 255.0
        gray3 = np.stack([g, g, g], axis=-1).astype(np.uint8)
        img, alpha = _finish(gray3, mask_hi, size, size)
    else:
        img, alpha = _finish(img_hi, mask_hi, size, size)
    return img, alpha > 0.5


def _orbit_project(world, faces, f, size):
    """Shared projection for the numpy orbit painters. Returns (uv, tri, z)."""
    world = np.asarray(world, np.float64)
    faces = np.asarray(faces, np.int32)
    z = world[:, 2]
    ok = z > 1e-3
    uv = np.stack([f * world[:, 0] / np.where(ok, z, 1) + size / 2.0,
                   f * world[:, 1] / np.where(ok, z, 1) + size / 2.0], axis=1)
    return uv, faces[ok[faces].all(axis=1)], z


def render_depth_np(world, faces, f, size):
    """Z-buffer-ish depth map (near=bright) on the orbit framing. Pure numpy/OpenCV."""
    uv, tri, z = _orbit_project(world, faces, f, size)
    if len(tri) == 0:
        raise ValueError("mesh projects to no pixels")
    depth = z[tri].mean(axis=1)
    lo, hi = float(depth.min()), float(depth.max())
    span = hi - lo if hi - lo > 1e-9 else 1.0
    gray = ((1.0 - (depth - lo) / span) * 255.0).astype(np.int32)
    order = np.argsort(-depth)
    img = np.zeros((size, size, 3), np.uint8)
    mask = np.zeros((size, size), np.uint8)
    for k in order:
        pts = uv[tri[k]]
        if not np.isfinite(pts).all():
            continue
        if pts[:, 0].max() < 0 or pts[:, 0].min() > size - 1 or pts[:, 1].max() < 0 or pts[:, 1].min() > size - 1:
            continue
        poly = np.round(pts).astype(np.int32)
        g = int(gray[k])
        cv2.fillConvexPoly(img, poly, (g, g, g), lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, poly, 255, lineType=cv2.LINE_AA)
    return img, mask > 127


def render_normal_np(world, faces, f, size):
    """Faceted normal map (camera-space normals to RGB) on the orbit framing. Pure numpy/OpenCV."""
    uv, tri, _ = _orbit_project(world, faces, f, size)
    world = np.asarray(world, np.float64)
    if len(tri) == 0:
        raise ValueError("mesh projects to no pixels")
    xyz = world[tri]
    n = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    colours = ((n * 0.5 + 0.5) * 255.0).astype(np.int32)
    z = world[:, 2]
    order = np.argsort(-z[tri].mean(axis=1))
    img = np.zeros((size, size, 3), np.uint8)
    mask = np.zeros((size, size), np.uint8)
    for k in order:
        pts = uv[tri[k]]
        if not np.isfinite(pts).all():
            continue
        if pts[:, 0].max() < 0 or pts[:, 0].min() > size - 1 or pts[:, 1].max() < 0 or pts[:, 1].min() > size - 1:
            continue
        poly = np.round(pts).astype(np.int32)
        colour = tuple(int(c) for c in colours[k])
        cv2.fillConvexPoly(img, poly, colour, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, poly, 255, lineType=cv2.LINE_AA)
    return img, mask > 127


def compose_panels(panels, labels, gap=8):
    """Side-by-side panels resized to a common height, each with a title."""
    H = max(p.shape[0] for p in panels)
    out = []
    for p, lab in zip(panels, labels):
        h, w = p.shape[:2]
        p2 = cv2.resize(p, (int(round(w * H / h)), H), interpolation=cv2.INTER_AREA) if h != H else p.copy()
        cv2.putText(p2, lab, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, min(0.85, p2.shape[1] / 900.0)), (240, 240, 240), 2, cv2.LINE_AA)
        out.append(p2)
    sep = np.full((H, gap, 3), 24, np.uint8)
    stitched = out[0]
    for p in out[1:]:
        stitched = np.hstack([stitched, sep, p])
    return stitched


def compose_grid(panels, labels, cols=2, gap=8):
    """Grid of panels (letterboxed to a common cell), each with a title."""
    cell_h = max(p.shape[0] for p in panels)
    cell_w = max(p.shape[1] for p in panels)
    rows = (len(panels) + cols - 1) // cols
    grid = np.full((rows * cell_h + (rows - 1) * gap, cols * cell_w + (cols - 1) * gap, 3), 24, np.uint8)
    for i, (p, lab) in enumerate(zip(panels, labels)):
        h, w = p.shape[:2]
        sc = min(cell_w / w, cell_h / h)
        p2 = cv2.resize(p, (max(1, int(round(w * sc))), max(1, int(round(h * sc)))), interpolation=cv2.INTER_AREA)
        cv2.putText(p2, lab, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, min(0.85, p2.shape[1] / 700.0)), (240, 240, 240), 2, cv2.LINE_AA)
        r, c = divmod(i, cols)
        y0 = r * (cell_h + gap) + (cell_h - p2.shape[0]) // 2
        x0 = c * (cell_w + gap) + (cell_w - p2.shape[1]) // 2
        grid[y0:y0 + p2.shape[0], x0:x0 + p2.shape[1]] = p2
    return grid


def overlay_photo(bgr, render_rgb, mask, kp3, alpha=0.85):
    H, W = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.0)[..., None] * alpha
    out = np.clip(rgb * (1 - a) + render_rgb.astype(np.float32) * a, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    for x, y in kp3[:70]:
        if np.isfinite([x, y]).all() and 0 <= x < W and 0 <= y < H:
            cv2.circle(vis, (int(x), int(y)), 5, (0, 0, 255), 2)
    return vis


def draw_mesh_kps(bgr, img, mask, kp3):  # kept for the fallback path
    return overlay_photo(bgr, img, mask, kp3, alpha=0.5)


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--images", nargs="*", default=None)
    a = ap.parse_args(argv)

    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
    images = ([a.input / n for n in a.images] if a.images
              else sorted(p for p in a.input.rglob("*") if p.is_file() and p.suffix.lower() in exts))
    if not images:
        raise SystemExit(f"no images in {a.input}")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "_global").mkdir(exist_ok=True)

    # unique folder key per image: bare stem, or <parent>__<stem> if two inputs share a stem
    keys: dict[Path, str] = {}
    for img in images:
        k = img.stem
        if k in keys.values():
            k = f"{img.parent.name}__{k}"
        keys[img] = k
    images = list(keys)

    data: dict[str, dict] = {}
    for p in images:
        key = keys[p]
        job = a.out / key
        (job).mkdir(parents=True, exist_ok=True)
        for sub in ("input", "sapiens", "threedb", "sapiens/raw", "threedb/raw"):
            (job / sub).mkdir(parents=True, exist_ok=True)
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        data[key] = dict(path=p, bgr=bgr, status="ok", timings={})
        if bgr is not None:
            shutil.copy2(p, a.out / key / "input" / f"01_original{p.suffix.lower()}")

    print(f"images: {len(images)} -> {a.out}")

    # ---- 1. pose: keypoints + person box for 3DB
    t = time.time()
    model = load_pose()
    for p in images:
        d = data[keys[p]]
        if d["bgr"] is None:
            continue
        t0 = time.time()
        bgr = d["bgr"]; H, W = bgr.shape[:2]
        kp_any, _ = pose_run(model, bgr)
        d["bbox"] = person_box(kp_any, W, H)
        d["kp"], d["score"] = pose_run(model, bgr, d["bbox"])
        cv2.imwrite(str(a.out / keys[p] / "sapiens" / "02_sapiens_keypoints.jpg"), draw_pose(bgr, d["kp"]))
        np.savez_compressed(a.out / keys[p] / "sapiens" / "raw" / "sapiens_pose.npz",
                            kp=d["kp"].astype(np.float32), score=d["score"].astype(np.float32),
                            bbox=np.asarray(d["bbox"], np.float32))
        (a.out / keys[p] / "sapiens" / "raw" / "person_box.json").write_text(
            json.dumps({"bbox_xyxy": [round(float(v), 3) for v in d["bbox"]],
                        "pose_score_mean": round(float(np.mean(d["score"])), 4),
                        "pose_score_median": round(float(np.median(d["score"])), 4)}, indent=2),
            encoding="utf-8")
        d["timings"]["pose"] = round(time.time() - t0, 2)
    del model; free({}); vram("after pose")
    print(f"[stage] sapiens pose {time.time()-t:.1f} s")

    # ---- 2. seg
    t = time.time()
    model = load_dense("seg", "seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
                       "seg/sapiens2_0.4b_seg.safetensors")
    from sapiens.dense.visualizers import SegVisualizer
    seg_vis = SegVisualizer(class_palette_type="dome29", with_labels=False)
    for p in images:
        d = data[keys[p]]
        if d["bgr"] is None:
            continue
        t0 = time.time()
        labels, overlay = seg_run(model, seg_vis, d["bgr"])
        d["labels"] = labels
        d["skin"] = skin_mask_from_labels(labels)
        cv2.imwrite(str(a.out / keys[p] / "sapiens" / "raw" / "sapiens_seg_labels.png"), labels)
        cv2.imwrite(str(a.out / keys[p] / "sapiens" / "raw" / "skin_mask.png"),
                    (d["skin"].astype(np.uint8) * 255))
        np.savez_compressed(a.out / keys[p] / "sapiens" / "raw" / "seg_class_counts.npz",
                            **{f"class_{int(c)}": np.array(int((labels == c).sum()))
                               for c in np.unique(labels)})
        cv2.imwrite(str(a.out / keys[p] / "sapiens" / "04_sapiens_seg.jpg"), overlay)
        d["timings"]["seg"] = round(time.time() - t0, 2)
    del seg_vis, model; free({}); vram("after seg")
    print(f"[stage] sapiens seg {time.time()-t:.1f} s")

    # ---- 3. normal
    t = time.time()
    model = load_dense("normal", "normal/metasim_render_people/"
                                 "sapiens2_0.4b_normal_metasim_render_people-1024x768.py",
                       "normal/sapiens2_0.4b_normal.safetensors")
    for p in images:
        d = data[keys[p]]
        if d["bgr"] is None:
            continue
        t0 = time.time()
        sn, vis = normal_run(model, d["bgr"], d["labels"])
        d["sn"] = sn.astype(np.float16)          # kept for the normal loss in the 3DB stage
        np.save(a.out / keys[p] / "sapiens" / "raw" / "sapiens_normals.npy", d["sn"])
        cv2.imwrite(str(a.out / keys[p] / "sapiens" / "05_sapiens_normal.jpg"), vis)
        d["timings"]["normal"] = round(time.time() - t0, 2)
    del model; free({}); vram("after normal")
    print(f"[stage] sapiens normal {time.time()-t:.1f} s")

    # ---- 4. depth (pointmap)
    t = time.time()
    model = load_dense("pointmap", "pointmap/render_people/"
                                   "sapiens2_0.4b_pointmap_render_people-1024x768.py",
                       "pointmap/sapiens2_0.4b_pointmap.safetensors")
    for p in images:
        d = data[keys[p]]
        if d["bgr"] is None:
            continue
        t0 = time.time()
        depth, vis = depth_run(model, d["bgr"], d["labels"])
        np.save(a.out / keys[p] / "sapiens" / "raw" / "sapiens_depth_m.npy", depth.astype(np.float32))
        cv2.imwrite(str(a.out / keys[p] / "sapiens" / "03_sapiens_depth.jpg"), vis)
        person = d["labels"] > 0
        d["depth_m"] = ([round(float(np.percentile(depth[person], 1)), 3),
                         round(float(np.percentile(depth[person], 99)), 3)] if person.any() else None)
        d["timings"]["depth"] = round(time.time() - t0, 2)
    del model; free({}); vram("after depth")
    print(f"[stage] sapiens depth {time.time()-t:.1f} s")

    # ---- 4b. sapiens 1b pointmap + 1b matting
    t = time.time()
    model = load_dense("pointmap", "pointmap/render_people/sapiens2_1b_pointmap_render_people-1024x768.py",
                       SAPIENS_1B_POINTMAP)
    for p in images:
        d = data[keys[p]]
        if d["bgr"] is None:
            continue
        t0 = time.time()
        depth1b, vis1b = depth_run(model, d["bgr"], d["labels"])
        np.save(a.out / keys[p] / "sapiens" / "raw" / "sapiens_pointmap_1b_depth_m.npy", depth1b.astype(np.float32))
        cv2.imwrite(str(a.out / keys[p] / "sapiens" / "10_pointmap1b_depth.jpg"), vis1b)
        person = d["labels"] > 0
        d["depth1b_m"] = ([round(float(np.percentile(depth1b[person], 1)), 3),
                           round(float(np.percentile(depth1b[person], 99)), 3)] if person.any() else None)
        d["timings"]["pointmap_1b"] = round(time.time() - t0, 2)
    del model; free({}); vram("after pointmap 1b")
    print(f"[stage] sapiens pointmap 1b {time.time()-t:.1f} s")

    t = time.time()
    model = load_dense("matting", "matting/gss_p3m_metasim/sapiens2_1b_matting_gss_p3m_metasim-1024x768.py",
                       SAPIENS_1B_MATTING)
    for p in images:
        d = data[keys[p]]
        if d["bgr"] is None:
            continue
        t0 = time.time()
        alpha = matting_run(model, d["bgr"])
        np.save(a.out / keys[p] / "sapiens" / "raw" / "sapiens_matting_1b_alpha.npy", alpha)
        cv2.imwrite(str(a.out / keys[p] / "sapiens" / "11_matting1b_alpha.png"),
                    (np.clip(alpha, 0, 1) * 255).astype(np.uint8))
        person = d["labels"] > 0
        d["alpha_stats"] = {
            "range": [round(float(alpha.min()), 4), round(float(alpha.max()), 4)],
            "mean_on_person": round(float(alpha[person].mean()), 4) if person.any() else None,
            "frac_above_0_5_on_person": round(float((alpha[person] > 0.5).mean()), 4) if person.any() else None,
        }
        d["timings"]["matting_1b"] = round(time.time() - t0, 2)
    del model; free({}); vram("after matting 1b")
    print(f"[stage] sapiens matting 1b {time.time()-t:.1f} s")

    # ---- 5. 3DB fit + renders
    t = time.time()
    model, est, faces = load_3db()
    for p in images:
        d = data[keys[p]]
        if d["bgr"] is None:
            continue
        t0 = time.time()
        bgr = d["bgr"]; H, W = bgr.shape[:2]
        try:
            out, verts, f = fit_3db(est, bgr, d["bbox"])
            raw = a.out / keys[p] / "threedb" / "raw"
            np.savez_compressed(
                raw / "3db_outputs.npz",
                **{k: np.asarray(v) for k, v in out.items()
                   if isinstance(v, (np.ndarray, list)) and k != "faces"},
                faces=np.asarray(faces, np.int32))
            np.save(raw / "3db_vertices_fit.npy", np.asarray(verts, np.float64))
            _write_obj(a.out / keys[p] / "threedb" / "mesh_fit.obj", np.asarray(verts, np.float64), faces,
                       "SAM 3D Body fit (camera frame, metres)")
            (raw / "3db_keypoints_2d.json").write_text(
                json.dumps({"pred_keypoints_2d": np.asarray(out["pred_keypoints_2d"], np.float64).round(4).tolist(),
                            "focal_px": round(float(f), 3),
                            "pred_cam_t": [round(float(x), 6) for x in np.asarray(out["pred_cam_t"]).reshape(-1)]},
                           indent=2), encoding="utf-8")
            d["identity_l2"] = float(np.linalg.norm(out["shape_params"]))
            d["focal"] = f
            d["cam_t"] = [round(float(v), 3) for v in out["pred_cam_t"]]
            kp3_before = np.asarray(out["pred_keypoints_2d"], np.float64)
            kp3 = kp3_before
            cam_t = np.asarray(out["pred_cam_t"], np.float64)
            verts0 = verts

            # first the identity-only solve (fast init + fallback), then the joint identity+pose/scale solve
            n_target = n_mask = None
            if "sn" in d and "skin" in d:
                sn = d["sn"].astype(np.float32) * np.array([1.0, -1.0, -1.0], np.float32)  # -> 3DB frame
                n_target = sn / np.maximum(np.linalg.norm(sn, axis=2, keepdims=True), 1e-8)
                n_mask = (d["skin"] & (np.linalg.norm(d["sn"].astype(np.float32), axis=2) > 0.5)
                          & ~feature_exclusion_mask(d["kp"], W, H))

            ref = identity_refine(model, out, d["kp"], d["score"], W, H, f, cam_t)
            d["identity_refine"] = {k: v for k, v in ref.items() if k != "beta"}
            verts_r = verts0
            kp3 = kp3_before
            beta_ref = np.asarray(ref.get("beta", out["shape_params"]), np.float64)
            params_ref = np.asarray(out["mhr_model_params"], np.float64)
            used = "raw"
            joint = None
            verts_neutral = verts0
            if _solve_joint is not None:
                try:
                    joint = _solve_joint(model, out, d["kp"], d["score"], W, H, f, cam_t,
                                         beta_init=beta_ref, identity_only=ref, iters=300,
                                         normal_target=n_target, normal_mask=n_mask,
                                         faces=faces, normal_weight=0.5)
                    d["joint_solve"] = {k: v for k, v in joint.items()
                                        if k not in ("shape_params", "mhr_model_params", "cam_t")}
                except Exception as exc:  # noqa: BLE001
                    d["joint_solve"] = {"applied": False, "error": f"{type(exc).__name__}: {exc}"}
                    joint = None
            if joint is not None and joint.get("applied"):
                beta_ref = np.asarray(joint["shape_params"], np.float64)
                params_ref = np.asarray(joint["mhr_model_params"], np.float64)
                cam_t = np.asarray(joint["cam_t"], np.float64)
                expr_ref = np.asarray(joint.get("expr_params", np.zeros(72)), np.float64)
                verts_root, lm_r = mesh_and_landmarks(model, out, beta_ref, model_params=params_ref,
                                                      expr=expr_ref)
                verts_r = verts_root + cam_t[None]          # camera-frame metres for rendering
                verts_neutral_root, _ = mesh_and_landmarks(model, out, beta_ref, model_params=params_ref)
                verts_neutral = verts_neutral_root + cam_t[None]
                kp3 = project_cam(lm_r[:70], f, cam_t, W, H)
                d["identity_l2"] = float(np.linalg.norm(beta_ref))
                d["cam_t"] = [round(float(v), 3) for v in cam_t]
                used = "joint"
                _write_obj(a.out / keys[p] / "threedb" / "mesh_refined.obj", np.asarray(verts_r, np.float64), faces,
                           "refined mesh (camera frame, metres)")
                (a.out / keys[p] / "threedb" / "identity_joint.json").write_text(json.dumps({
                    "format": "mhr-identity-joint/1",
                    "image": p.name,
                    "model": "SAM 3D Body dinov3 -> identity(45)+pose+scale re-solved on Sapiens2 keypoints",
                    "original_shape_params": [round(float(v), 6) for v in out["shape_params"]],
                    "original_mhr_model_params": [round(float(v), 6) for v in out["mhr_model_params"]],
                    "original_cam_t": [round(float(v), 6) for v in out["pred_cam_t"]],
                    "refined_shape_params": [round(float(v), 6) for v in beta_ref],
                    "refined_mhr_model_params": [round(float(v), 6) for v in params_ref],
                    "refined_cam_t": [round(float(v), 6) for v in cam_t],
                    "refined_expr_params": [round(float(v), 6) for v in expr_ref],
                    "metrics": _jsonable(d["joint_solve"]),
                }, indent=2), encoding="utf-8")
            elif ref.get("applied"):
                verts_root, lm_r = mesh_and_landmarks(model, out, beta_ref)
                cam_t = np.asarray(out["pred_cam_t"], np.float64)
                verts_r = verts_root + cam_t[None]
                kp3 = project_cam(lm_r[:70], f, cam_t, W, H)
                d["identity_l2"] = float(np.linalg.norm(beta_ref))
                used = "identity"
                _write_obj(a.out / keys[p] / "threedb" / "mesh_refined.obj", np.asarray(verts_r, np.float64), faces,
                           "refined mesh (camera frame, metres)")
                (a.out / keys[p] / "threedb" / "identity_refined.json").write_text(json.dumps({
                    "format": "mhr-identity-refined/1",
                    "image": p.name,
                    "model": "SAM 3D Body dinov3 -> 45 MHR identity, re-solved on Sapiens2 face+ear keypoints",
                    "original_shape_params": [round(float(v), 6) for v in out["shape_params"]],
                    "refined_shape_params": [round(float(v), 6) for v in beta_ref],
                    "refine": d["identity_refine"],
                }, indent=2), encoding="utf-8")
            d["refine_used"] = used

            cv2.imwrite(str(a.out / keys[p] / "threedb" / "06_3db_keypoints.jpg"),
                        draw_3db_match(bgr, kp3, d["kp"]))
            try:
                # 2x2 sheet: overlay with kps | textured front | textured 3/4 | clay 3/4
                vcol = texture_mesh(verts_r, faces, bgr, f, W, H)
                timg, tmask = render_rgb(verts_r, faces, f, W, H, vcol=vcol)
                overlay = overlay_photo(bgr, timg, tmask, kp3, alpha=0.62)
                front = cv2.cvtColor(timg, cv2.COLOR_RGB2BGR)
                tile = max(W, H)
                side_t = cv2.cvtColor(render_clay_view(verts_r, faces, f, tile, yaw=35.0, vcol=vcol),
                                      cv2.COLOR_RGB2BGR)
                side_c = cv2.cvtColor(render_clay_view(verts_r, faces, f, tile, yaw=35.0),
                                      cv2.COLOR_RGB2BGR)
                panel = compose_grid([overlay, front, side_t, side_c],
                                     ["3DB mesh + kps over photo", "3DB render (photo colours)",
                                      "same mesh, 3/4 view", "geometry only (clay)"], cols=2)
                cv2.imwrite(str(a.out / keys[p] / "threedb" / "07_3db_mesh_keypoints.jpg"), panel)
                # 08: identity refinement before/after close-up (clay, supersampled)
                face = d["kp"][FACE_ROWS]
                x1, y1 = np.floor(face.min(0)).astype(int)
                x2, y2 = np.ceil(face.max(0)).astype(int)
                pad = int(0.2 * max(x2 - x1, y2 - y1))
                x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
                x2, y2 = min(x2 + pad, W), min(y2 + pad, H)
                crops = []
                for tag, vv, kk in (("before", verts0, kp3_before), ("after", verts_r, kp3)):
                    img_b, mask_b = render_rgb(vv, faces, f, W, H, colour=MESH_COLOUR, ss=1.5)
                    ov = overlay_photo(bgr, img_b, mask_b, kk, alpha=0.55)
                    crops.append(ov[y1:y2, x1:x2].copy())
                cal_panel = compose_panels(
                    crops, ["before: 3DB identity",
                            f"after: {used} solve on Sapiens2 keypoints"])
                cv2.imwrite(str(a.out / keys[p] / "threedb" / "08_head_calibration.jpg"), cal_panel)
                # 09: expression close-up - photo + Sapiens2 face kps | neutral jaw | fitted expression
                photo = bgr.copy()
                for r in FACE_ROWS:
                    x, y = d["kp"][r]
                    if np.isfinite([x, y]).all() and 0 <= x < W and 0 <= y < H:
                        cv2.circle(photo, (int(x), int(y)), 2, (0, 255, 0), -1)
                img_n, _ = render_rgb(verts_neutral, faces, f, W, H, colour=MESH_COLOUR, ss=1.6)
                img_e, _ = render_rgb(verts_r, faces, f, W, H, colour=MESH_COLOUR, ss=1.6)

                def _crop_scale(im):
                    c = im[y1:y2, x1:x2].copy()
                    s = max(1.0, min(3.0, 820.0 / max(1, c.shape[0])))
                    return cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

                expr_l2 = d["joint_solve"].get("metrics", {}).get("expr_l2", 0) if d["joint_solve"] else 0
                panel9 = compose_panels(
                    [_crop_scale(photo), _crop_scale(img_n), _crop_scale(img_e)],
                    ["photo + Sapiens2 face kps", "joint solve, neutral jaw",
                     f"joint + expression fit (|expr|={expr_l2:.2f})"])
                cv2.imwrite(str(a.out / keys[p] / "threedb" / "09_expression.jpg"), panel9)
            except Exception as exc:  # noqa: BLE001
                d["status"] = f"mesh_render_failed:{type(exc).__name__}"
                cv2.imwrite(str(a.out / keys[p] / "threedb" / "07_3db_mesh_keypoints.jpg"),
                            draw_3db_match(bgr, kp3, d["kp"]))
            if not np.isfinite(verts).all():
                d["status"] = "3db_nonfinite"
        except Exception as exc:  # noqa: BLE001
            d["status"] = f"3db_failed:{type(exc).__name__}"
        d["timings"]["3db"] = round(time.time() - t0, 2)
    del est, model; free({}); vram("after 3db")
    print(f"[stage] 3db {time.time()-t:.1f} s")

    # ---- 6. stats + README
    summary = []
    for p in images:
        d = data[keys[p]]
        if d["bgr"] is None:
            summary.append(dict(image=p.name, status="unreadable", size="-", seconds="-"))
            continue
        H, W = d["bgr"].shape[:2]
        d["timings"]["total"] = round(sum(d["timings"].values()), 2)
        folder = a.out / keys[p]
        raw = folder / "raw"
        person = d["labels"] > 0

        def _pct(arr, q):
            return [round(float(v), 4) for v in np.percentile(np.asarray(arr, np.float64), q)]

        stats = {
            "image": p.name,
            "input_file": str(p),
            "input_size_bytes": int(p.stat().st_size),
            "input_sha256": _sha256(p),
            "size_wh": [W, H],
            "status": d["status"],
            "person_box_xyxy": [round(float(v), 1) for v in d["bbox"]],
            "pose_score": {"mean": round(float(d["score"].mean()), 4),
                           "median": round(float(np.median(d["score"])), 4),
                           "p05": round(float(np.percentile(d["score"], 5)), 4),
                           "p95": round(float(np.percentile(d["score"], 95)), 4)},
            "seg_classes": sorted(int(c) for c in np.unique(d["labels"])),
            "seg_class_pixels": {int(c): int((d["labels"] == c).sum()) for c in np.unique(d["labels"])},
            "skin_pixels": int(d["skin"].sum()),
            "depth_m": d.get("depth_m"),
            "depth1b_m": d.get("depth1b_m"),
            "matting_1b_alpha": d.get("alpha_stats"),
            "depth_on_person_m": (_pct(np.load(folder / "sapiens" / "raw" / "sapiens_depth_m.npy")[person],
                                      [1, 5, 50, 95, 99]) if "depth_m" in d else None),
            "identity_l2": round(d["identity_l2"], 3) if "identity_l2" in d else None,
            "focal_px": round(d["focal"], 1) if "focal" in d else None,
            "cam_t_m": d.get("cam_t"),
            "identity_refine": d.get("identity_refine"),
            "joint_solve": d.get("joint_solve"),
            "normal_supervision": (d.get("joint_solve") or {}).get("normal_supervision"),
            "refine_used": d.get("refine_used"),
            "timings_s": d["timings"],
            "raw_files": {"sapiens": sorted(q.name for q in (folder / "sapiens" / "raw").iterdir())
                          if (folder / "sapiens" / "raw").exists() else [],
                          "threedb": sorted(q.name for q in (folder / "threedb" / "raw").iterdir())
                          if (folder / "threedb" / "raw").exists() else []},
            "files": sorted(q.relative_to(folder).as_posix() for q in folder.rglob("*") if q.is_file()),
            "model_paths": {"sam3dbody": str(THREEDB_CKPT), "mhr": str(THREEDB_MHR),
                            "sapiens2": str(SAPIENS_W)},
        }
        (folder / "stats.json").write_text(json.dumps(_jsonable(stats), indent=2), encoding="utf-8")

        # per-image human-readable log
        lines = [f"image: {p.name}", f"input: {p} ({p.stat().st_size} B sha256 {_sha256(p)[:16]}...)",
                 f"size: {W}x{H}", f"person box: {[round(float(v), 1) for v in d['bbox']]}",
                 f"pose score: mean {stats['pose_score']['mean']} median {stats['pose_score']['median']}",
                 f"seg classes: {stats['seg_classes']} | skin px {stats['skin_pixels']}",
                 f"depth m on person: {stats['depth_on_person_m']}",
                 f"focal px: {stats['focal_px']}", f"cam_t: {stats['cam_t_m']}",
                 f"refine used: {stats['refine_used']} | identity |b| {stats['identity_l2']}"]
        if d.get("joint_solve"):
            js = d["joint_solve"]
            lines.append(f"joint: applied={js.get('applied')} seconds={js.get('seconds')} fit_expr={js.get('fit_expr')}")
            if js.get("metrics"):
                mm = js["metrics"]
                lines.append("joint metrics: " + json.dumps({
                    k: (mm.get(k) if not isinstance(mm.get(k), dict) else {kk: mm[k].get(kk) for kk in
                                                                           ("face_err_px", "ear_err_px", "ear_span_px")})
                    for k in ("before_3db", "identity_only", "joint", "delta_beta_from_3db",
                              "delta_pose_norm", "delta_scale_norm", "delta_trans_m", "expr_l2")}))
        lines.append("timings: " + json.dumps(d["timings"]))
        lines.append("files: " + ", ".join(sorted(q.relative_to(folder).as_posix() for q in folder.rglob("*") if q.is_file())))
        (folder / "log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary.append(dict(image=p.name, status=d["status"], size=f"{W}x{H}",
                            seconds=d["timings"]["total"]))
        print(f"[{keys[p]}] {d['status']:24s} {W}x{H}  {d['timings']['total']:.1f} s")

    (a.out / "_global" / "_last_run.md").write_text(
        "# Sapiens2 + SAM 3D Body — all outputs\n\n"
        "**Every generated artefact lives in this folder.** Per-image results are one subfolder per input; "
        "the validation/render sets are in the subfolders below.\n\n"
        f"Input: `{a.input}` ({len(images)} images, scanned recursively).\n\n"
        "| subfolder | what |\n| :--- | :--- |\n"
        "| `<image-stem>/` | per-image results (table below) |\n"
        "| `joint-test/` | joint-solver runs: params JSON, metrics, comparison report |\n"
        "| `joint-eval/` | independent evaluation (acceptance checks, comparison renders) |\n"
        "| `sap-roundtrip/` | solved mesh rendered, Sapiens2 re-detected, compared with the photo |\n"
        "| `geom-roundtrip/` | mesh z-buffer depth/normals vs Sapiens2 pointmap/normal maps |\n"
        "| `notebook/` | headless notebook execution outputs (`exec/`, `artifacts/`) |\n"
        "| `measurements/` | ingest measurement JSON and the scripts that produced them |\n\n"
        "| file (inside each image folder) | what |\n| :--- | :--- |\n"
        "| `01_original.*` | the original image |\n"
        "| `02_sapiens_keypoints.jpg` | original + Sapiens2 308 keypoints (green body/hand, magenta face) |\n"
        "| `03_sapiens_depth.jpg` | Sapiens2 pointmap depth (metric, turbo, person-masked) |\n"
        "| `04_sapiens_seg.jpg` | Sapiens2 body-part segmentation (29 classes) |\n"
        "| `05_sapiens_normal.jpg` | Sapiens2 surface normals (RGB) |\n"
        "| `06_3db_keypoints.jpg` | original + SAM 3D Body 70 keypoints (red) matched 1:1 to Sapiens2's 70 (green) |\n"
        "| `07_3db_mesh_keypoints.jpg` | 2x2 sheet: textured 3DB mesh + keypoints over the photo / the render on its own / a textured 3/4 view / a clay 3/4 view |\n"
        "| `08_head_calibration.jpg` | before/after close-up: raw 3DB vs joint identity+pose+scale solve |\n"
        "| `09_expression.jpg` | face close-up: photo + Sapiens2 face keypoints / neutral jaw / fitted expression |\n"
        "| `identity_joint.json` | joint solve: original + refined identity(45), mhr_model_params(204), cam_t, expression(72), metrics |\n"
        "| `stats.json` | size, person box, pose score, seg classes, depth range, identity norm, focal, timings |\n\n"
        "Generated by [`run_all.py`](../run_all.py). Models are loaded one stage at a time and freed; "
        "the process is capped at 95 % of the 24 GB card. The 3DB person box is derived from Sapiens2's "
        "in-frame keypoints.\n\n"
        "| image | status | size | s |\n| :--- | :--- | :--- | ---: |\n"
        + "\n".join(f"| {s['image']} | {s['status']} | {s['size']} | {s['seconds']} |" for s in summary) + "\n",
        encoding="utf-8")

    print(f"\nDone: {len(images)} image(s) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
