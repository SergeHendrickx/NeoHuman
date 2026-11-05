"""VGGT point-cloud consistency of the Muse 2x2 full-body grid (+45, -45, 0, 180)."""
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

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
import run_all as R2

JOB = BASE / "outputs" / "Emiru"
OUT = JOB / "threedb"
PANELS = [("+45", OUT / "muse_fullbody_p45.png"), ("-45", OUT / "muse_fullbody_m45.png"),
          ("0", OUT / "muse_fullbody_0.png"), ("180", OUT / "muse_fullbody_180.png")]
VGGT_CKPT = REPO / "pivot" / "vggt-omega" / "checkpoints" / "vggt_omega_1b_512.pt"
paths = [p for _, p in PANELS]

imgs = load_and_preprocess_images([str(p) for p in paths], image_resolution=512)
model = VGGTOmega().eval()
model.load_state_dict(torch.load(VGGT_CKPT, map_location="cpu"))
model = model.cuda()
with torch.no_grad():
    preds = model(imgs.cuda())
depth = preds["depth"][0, ..., 0].cpu().numpy().astype(np.float32)
conf = preds["depth_conf"][0].cpu().numpy().astype(np.float32)
ext, K = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
ext = ext[0].cpu().numpy().astype(np.float32); K = K[0].cpu().numpy().astype(np.float32)
del model; torch.cuda.empty_cache()

seg_model = R2.load_dense("seg", "seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
                          "seg/sapiens2_0.4b_seg.safetensors")
from sapiens.dense.visualizers import SegVisualizer
seg_vis = SegVisualizer(class_palette_type="dome29", with_labels=False)
pose_model = R2.load_pose()

worlds, heads = {}, {}
for i, (name, path) in enumerate(PANELS):
    bgr = cv2.imread(str(path))
    Hp, Wp = bgr.shape[:2]
    labels, _ = R2.seg_run(seg_model, seg_vis, bgr)
    person = cv2.resize((labels > 0).astype(np.uint8), (depth[i].shape[1], depth[i].shape[0]),
                        interpolation=cv2.INTER_NEAREST) > 0
    kp, _ = R2.pose_run(pose_model, bgr)
    face = kp[70:272][np.isfinite(kp[70:272]).all(1)]
    fc = face.mean(0)
    fr = 1.3 * max(np.ptp(face[:, 0]), np.ptp(face[:, 1]))
    ys_i, xs_i = np.mgrid[0:depth[i].shape[0], 0:depth[i].shape[1]]
    xs_p = xs_i * (Wp / depth[i].shape[1]); ys_p = ys_i * (Hp / depth[i].shape[0])
    head = person & (((xs_p - fc[0]) ** 2 + (ys_p - fc[1]) ** 2) < fr ** 2)
    m = person & (conf[i] > np.percentile(conf[i], 25)) & np.isfinite(depth[i]) & (depth[i] > 0.05)
    ys, xs = np.nonzero(m)
    d = depth[i][ys, xs]
    Xc = np.stack([(xs - K[i, 0, 2]) / K[i, 0, 0] * d, (ys - K[i, 1, 2]) / K[i, 1, 1] * d, d], 1)
    Rm, tt = ext[i][:3, :3], ext[i][:3, 3]
    worlds[name] = (Rm.T @ (Xc - tt[None]).T).T
    heads[name] = head[m]
    print(name, "points", len(worlds[name]), "head", int(head[m].sum()))
del seg_model, seg_vis, pose_model; R2.free({})


def chamfer(A, B):
    A = A[:: max(1, len(A) // 3000)]
    B = B[:: max(1, len(B) // 3000)]
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.median(d) * 1000), float(np.percentile(d, 90) * 1000)


report = {"n_points": {n: int(len(worlds[n])) for n, _ in PANELS},
          "extrinsics": {n: {"R": ext[i][:3, :3].round(3).tolist(), "t": ext[i][:3, 3].round(3).tolist()}
                         for i, (n, _) in enumerate(PANELS)}}
names = [n for n, _ in PANELS]
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        med, p90 = chamfer(worlds[a], worlds[b])
        hm, hp = chamfer(worlds[a][heads[a]], worlds[b][heads[b]])
        report[f"{a}__{b}"] = {"all_median_mm": round(med, 1), "all_p90_mm": round(p90, 1),
                               "head_median_mm": round(hm, 1), "head_p90_mm": round(hp, 1)}
        print(a, b, json.dumps(report[f"{a}__{b}"]))

# visual: all clouds in the 0-deg (front) camera frame, colour-coded
cols = {"+45": (1, 0, 0), "-45": (1, 1, 0), "0": (0, 1, 0), "180": (0, 0, 1)}
pts = np.concatenate([worlds[n] for n in names])
cc = np.concatenate([np.tile(cols[n], (len(worlds[n]), 1)) for n in names])
# front camera = view "0"
i0 = names.index("0")
R0, t0m, K0 = ext[i0][:3, :3], ext[i0][:3, 3], K[i0]
Xc = (R0 @ pts.T).T + t0m[None]
z = np.maximum(Xc[:, 2], 1e-6)
u = K0[0, 0] * Xc[:, 0] / z + K0[0, 2]
v = K0[1, 1] * Xc[:, 1] / z + K0[1, 2]
S = 900
u = (u - u.min()) / max(1e-6, (u.max() - u.min())) * (S - 1)
v = (v - v.min()) / max(1e-6, (v.max() - v.min())) * (S - 1)
canvas = np.full((S, S, 3), 240, np.uint8)
order = np.argsort(-z)
for k in order:
    x, y = int(u[k]), int(v[k])
    if 0 <= x < S and 0 <= y < S:
        c = cc[k]
        canvas[y, x] = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
cv2.imwrite(str(OUT / "vggt_fullbody_grid_rgb.png"), canvas)
(OUT / "vggt_fullbody_grid_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

# ---- PLY exports: fused multi-view point cloud (coloured by view) + per-view clouds
import trimesh
rgb = np.clip(cc * 255, 0, 255).astype(np.uint8)
trimesh.PointCloud(pts, colors=rgb).export(OUT / "vggt_fullbody_cloud.ply")
for n in names:
    c = np.array(cols[n])[None, :]
    trimesh.PointCloud(worlds[n], colors=np.clip(np.tile(c, (len(worlds[n]), 1)) * 255, 0, 255).astype(np.uint8)) \
        .export(OUT / f"vggt_cloud_{n.replace('+', 'p').replace('-', 'm')}.ply")
print("wrote", OUT / "vggt_fullbody_cloud.ply")
