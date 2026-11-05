"""Align the 4-view VGGT point cloud to our metric MHR A-pose mesh (body-landmark Umeyama), export metric PLY."""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(REPO / "pivot" / "vggt-omega"))

import cv2
import numpy as np
import torch
import trimesh

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
import run_all as R

JOB = BASE / "outputs" / "Emiru"
OUT = JOB / "threedb"
PANELS = [("+45", OUT / "muse_fullbody_p45.png"), ("-45", OUT / "muse_fullbody_m45.png"),
          ("0", OUT / "muse_fullbody_0.png"), ("180", OUT / "muse_fullbody_180.png")]
VGGT_CKPT = REPO / "pivot" / "vggt-omega" / "checkpoints" / "vggt_omega_1b_512.pt"
KP_ROWS = np.array([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# ---- VGGT on the 4 panels
imgs = load_and_preprocess_images([str(p) for _, p in PANELS], image_resolution=512)
vm = VGGTOmega().eval()
vm.load_state_dict(torch.load(VGGT_CKPT, map_location="cpu"))
vm = vm.cuda()
with torch.no_grad():
    preds = vm(imgs.cuda())
depth = preds["depth"][0, ..., 0].cpu().numpy().astype(np.float32)
conf = preds["depth_conf"][0].cpu().numpy().astype(np.float32)
ext, K = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
ext = ext[0].cpu().numpy().astype(np.float32); K = K[0].cpu().numpy().astype(np.float32)
del vm; torch.cuda.empty_cache()

pose_model = R.load_pose()
model3d, est, faces = R.load_3db()

# ---- our metric A-pose mesh + keypoints
j = json.loads((OUT / "identity_joint.json").read_text(encoding="utf-8"))
beta = np.asarray(j["refined_shape_params"])
params = np.asarray(j["refined_mhr_model_params"]).copy()
params[:6] = 0.0
params[6:136] = 0.0
vroot, lm_ap = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params, expr=np.zeros(72))
center = vroot.mean(0)
verts_ap = vroot - center
lm_ap = lm_ap - center
mesh_kp = lm_ap[KP_ROWS]


def umeyama(src, dst):
    """similarity transform mapping src -> dst (both Nx3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, Sig, Vt = np.linalg.svd(C)
    S_fix = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_fix[2, 2] = -1
    Rm = U @ S_fix @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = float(np.trace(np.diag(Sig) @ S_fix) / var_s)
    t = mu_d - s * (Rm @ mu_s)
    return s, Rm, t


def world_kp(i, bgr):
    kp, _ = R.pose_run(pose_model, bgr)
    Hp, Wp = bgr.shape[:2]
    Hv, Wv = depth[i].shape
    sel = np.isfinite(kp[KP_ROWS]).all(1)
    pts2 = kp[KP_ROWS][sel].copy()
    pts2[:, 0] *= Wv / Wp
    pts2[:, 1] *= Hv / Hp
    z = np.zeros(len(pts2), np.float32)
    for k, (x, y) in enumerate(pts2):
        xi, yi = int(np.clip(x, 0, Wv - 1)), int(np.clip(y, 0, Hv - 1))
        z[k] = depth[i, yi, xi]
    Xc = np.stack([(pts2[:, 0] - K[i, 0, 2]) / K[i, 0, 0] * z,
                   (pts2[:, 1] - K[i, 1, 2]) / K[i, 1, 1] * z, z], 1)
    Rm, tt = ext[i][:3, :3], ext[i][:3, 3]
    Xw = (Rm.T @ (Xc - tt[None]).T).T
    return Xw, sel


report = {}
transforms = {}
for i, (name, path) in enumerate(PANELS):
    bgr = cv2.imread(str(path))
    Xw, sel = world_kp(i, bgr)
    dst = mesh_kp[sel]
    ok = np.isfinite(Xw).all(1) & (Xw[:, 2] > 0)
    s, Rm, t = umeyama(Xw[ok], dst[ok])
    pred = s * (Xw[ok] @ Rm.T) + t
    res = np.linalg.norm(pred - dst[ok], axis=1) * 1000
    report[name] = {"n_points": int(ok.sum()), "scale_to_m": round(s, 5),
                    "residual_median_mm": round(float(np.median(res)), 1),
                    "residual_p90_mm": round(float(np.percentile(res, 90)), 1)}
    transforms[name] = (s, Rm, t)
    print(name, json.dumps(report[name]))

# ---- transform the fused VGGT cloud with the best-scoring view transform
best = min(report, key=lambda k: report[k]["residual_median_mm"])
s, Rm, t = transforms[best]
print("best view:", best, "scale", round(s, 5))

cloud = trimesh.load(str(OUT / "vggt_fullbody_cloud.ply"), process=False)
pts = np.asarray(cloud.vertices)
cols = np.asarray(cloud.colors) if hasattr(cloud, "colors") and cloud.colors is not None and len(cloud.colors) else np.full((len(pts), 3), 200, np.uint8)
pts_m = s * (pts @ Rm.T) + t
trimesh.PointCloud(pts_m, colors=cols).export(OUT / "vggt_fullbody_cloud_metric.ply")
print("wrote", OUT / "vggt_fullbody_cloud_metric.ply")

# ---- report + visual: cloud vs mesh rendered in the front ortho view
from render_ortho import raster_ortho
from render_ortho import SIZE
import importlib.util
spec = importlib.util.spec_from_file_location("ro", BASE / "outputs" / "_global" / "meta-image-test" / "ortho-neutral" / "render_ortho.py")
ro = importlib.util.module_from_spec(spec); spec.loader.exec_module(ro)
yy = pts_m[:, 1]
lo, hi = np.percentile(yy, [1, 99])
side = 1.15 * (hi - lo)
cx, cy = pts_m[:, 0].mean(), (lo + hi) / 2
sc = (ro.SIZE * ro.SS) / side
img_cloud = ro.raster_ortho(pts_m, np.zeros((0, 3), np.int64), np.tile(np.array([0.85, 0.15, 0.15]), (len(pts_m), 1)),
                            sc, cx - side / 2, cy - side / 2) if False else None
# simpler: 2D scatter of cloud + projected mesh points
import cv2
S = ro.SIZE
canvas = np.full((S, S, 3), 240, np.uint8)
u = (pts_m[:, 0] - (cx - side / 2)) * sc
v = (pts_m[:, 1] - (cy - side / 2)) * sc
ui = np.clip(u, 0, S - 1).astype(int); vi = np.clip(v, 0, S - 1).astype(int)
canvas[vi, ui] = (60, 60, 200)
mu = (verts_ap[:, 0] + 0 - (cx - side / 2)) * sc
mv = (verts_ap[:, 1] - (cy - side / 2)) * sc
mui = np.clip(mu, 0, S - 1).astype(int); mvi = np.clip(mv, 0, S - 1).astype(int)
canvas[mvi, mui] = (60, 180, 60)
cv2.imwrite(str(OUT / "cloud_vs_mesh_metric.jpg"), canvas)
report["alignment"] = {"best_view": best, "scale_to_metres": round(float(s), 5),
                       "translation_m": [round(float(x), 4) for x in t]}
(OUT / "cloud_to_mhr_alignment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("wrote", OUT / "cloud_vs_mesh_metric.jpg")
