"""6-view head cloud with KNOWN camera rotations (0,-45,45,-90,90,180): VGGT depths + Sapiens2 landmarks,
solve per-view scale+translation against view 0, fuse, export PLY + consistency numbers."""
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

import run_all as R
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

STEM = sys.argv[1] if len(sys.argv) > 1 else "teenage-woman-long-wavy-white-gray__01971-2947889296"
JOB = BASE / "outputs" / STEM
OUT = JOB / "threedb"
PANELS = ["muse_face_0.png", "muse_face_m45.png", "muse_face_45.png",
          "muse_face_m90.png", "muse_face_90.png", "muse_face_180.png"]
YAWS = [0.0, -45.0, 45.0, -90.0, 90.0, 180.0]
VGGT_CKPT = REPO / "pivot" / "vggt-omega" / "checkpoints" / "vggt_omega_1b_512.pt"
paths = [OUT / p for p in PANELS]

# VGGT depths
imgs = load_and_preprocess_images([str(p) for p in paths], image_resolution=512)
vm = VGGTOmega().eval()
vm.load_state_dict(torch.load(VGGT_CKPT, map_location="cpu"))
vm = vm.cuda()
with torch.no_grad():
    preds = vm(imgs.cuda())
depth = preds["depth"][0, ..., 0].cpu().numpy().astype(np.float32)
conf = preds["depth_conf"][0].cpu().numpy().astype(np.float32)
_, K = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
K = K[0].cpu().numpy().astype(np.float32)
del vm; torch.cuda.empty_cache()

# Sapiens2 seg (mask) + pose (landmarks)
R.sapiens_imports()
from sapiens.dense.visualizers import SegVisualizer
seg_model = R.load_dense("seg", "seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
                         "seg/sapiens2_0.4b_seg.safetensors")
seg_vis = SegVisualizer(class_palette_type="dome29", with_labels=False)
pose_model = R.load_pose()

ref_rows = np.arange(70, 272)
clouds = {}
ref_kp3d = None
report = {"subject": STEM, "yaws": YAWS, "frames": {}}
for i, (p, yaw) in enumerate(zip(paths, YAWS)):
    bgr = cv2.imread(str(p))
    Hp, Wp = bgr.shape[:2]
    Hv, Wv = depth[i].shape
    labels, _ = R.seg_run(seg_model, seg_vis, bgr)
    person = cv2.resize((labels > 0).astype(np.uint8), (Wv, Hv), interpolation=cv2.INTER_NEAREST) > 0
    kp, _ = R.pose_run(pose_model, bgr)
    uv = kp[ref_rows].copy()
    uv[:, 0] *= Wv / Wp; uv[:, 1] *= Hv / Hp
    z = np.zeros(len(uv), np.float32)
    for k, (x, y) in enumerate(uv):
        xi, yi = int(np.clip(x, 0, Wv - 1)), int(np.clip(y, 0, Hv - 1))
        z[k] = depth[i, yi, xi]
    ok = np.isfinite(z) & (z > 0.05)
    Xc = np.stack([(uv[:, 0] - K[i, 0, 2]) / K[i, 0, 0] * z,
                   (uv[:, 1] - K[i, 1, 2]) / K[i, 1, 1] * z, z], 1)
    th = np.deg2rad(yaw)
    c, s = np.cos(th), np.sin(th)
    Ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    Xr = (Ry.T @ Xc.T).T                        # known-rotation init: to view-0 frame
    if i == 0:
        ref_kp3d = Xr
        frame_s, frame_t = 1.0, np.zeros(3)
    else:
        m = ok & np.isfinite(ref_kp3d).all(1)
        A = np.zeros((3 * m.sum(), 4)); b = np.zeros(3 * m.sum())
        A[0::3, 0] = Xr[m, 0]; A[0::3, 1] = 1; b[0::3] = ref_kp3d[m, 0]
        A[1::3, 0] = Xr[m, 1]; A[1::3, 2] = 1; b[1::3] = ref_kp3d[m, 1]
        A[2::3, 0] = Xr[m, 2]; A[2::3, 3] = 1; b[2::3] = ref_kp3d[m, 2]
        frame_s, t1, t2, t3 = np.linalg.lstsq(A, b, rcond=None)[0]
        frame_t = np.array([t1, t2, t3])
    # full cloud in the view-0 frame
    ys, xs = np.nonzero(person & (conf[i] > np.percentile(conf[i], 25)) & np.isfinite(depth[i]) & (depth[i] > 0.05))
    d = depth[i][ys, xs]
    Xc = np.stack([(xs - K[i, 0, 2]) / K[i, 0, 0] * d, (ys - K[i, 1, 2]) / K[i, 1, 1] * d, d], 1)
    Xr = (Ry.T @ Xc.T).T
    clouds[p] = frame_s * Xr + frame_t[None]
    report["frames"][p.name] = {"yaw": yaw, "scale": round(float(frame_s), 4),
                           "translation": [round(float(v), 4) for v in frame_t],
                           "n_points": int(len(clouds[p]))}
    print(p.name, json.dumps({k: report["frames"][p.name][k] for k in ("yaw", "scale", "n_points")}), flush=True)
del seg_model, seg_vis, pose_model; R.free({})

# consistency between views (head cloud only)
def chamfer(A, B):
    A = A[:: max(1, len(A) // 4000)]; B = B[:: max(1, len(B) // 4000)]
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.median(d) * 1000), float(np.percentile(d, 90) * 1000)

names = list(clouds)
pairs = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        med, p90 = chamfer(clouds[names[i]], clouds[names[j]])
        pairs[f"{Path(names[i]).stem}__{Path(names[j]).stem}"] = {"median_mm": round(med, 1), "p90_mm": round(p90, 1)}
vals = [v["median_mm"] for v in pairs.values()]
report["pairs"] = pairs
report["summary"] = {"n_pairs": len(vals), "median_mm": round(float(np.median(vals)), 1),
                     "best_mm": round(float(np.min(vals)), 1), "worst_mm": round(float(np.max(vals)), 1)}
print(json.dumps(report["summary"], indent=1))

# fused PLY with per-view colours
palette = [(230, 50, 50), (245, 150, 30), (200, 220, 40), (60, 200, 90), (50, 120, 240), (150, 60, 230)]
pts = np.concatenate([clouds[n] for n in names])
cols = np.concatenate([np.tile(np.array(palette[k], np.uint8), (len(clouds[n]), 1)) for k, n in enumerate(names)])
trimesh.PointCloud(pts, colors=cols).export(OUT / "head6_vggt_cloud_known_cams.ply")
(OUT / "head6_vggt_known_cams.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("wrote", OUT / "head6_vggt_cloud_known_cams.ply")

# visual: cloud in view-0 frame
S = 900
lo, hi = np.percentile(pts[:, 1], [1, 99]); cx = pts[:, 0].mean(); side = 1.15 * (hi - lo)
sc = S / side; x0, y0 = cx - side / 2, (lo + hi) / 2 - side / 2
canvas = np.full((S, S, 3), 240, np.uint8)
for k in np.argsort(-pts[:, 2]):
    x = int((pts[k, 0] - x0) * sc); y = int((pts[k, 1] - y0) * sc)
    if 0 <= x < S and 0 <= y < S:
        canvas[y, x] = cols[k][::-1]
cv2.imwrite(str(OUT / "head6_vggt_known_cams.png"), canvas)
print("wrote", OUT / "head6_vggt_known_cams.png")
