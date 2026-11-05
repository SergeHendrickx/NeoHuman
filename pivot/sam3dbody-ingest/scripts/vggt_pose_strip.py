"""VGGT point-cloud consistency of the 3 generated pose-strip panels (front / side / back)."""
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

JOB = BASE / "outputs" / "Emiru"
OUT = JOB / "threedb" / "pose_strip_test"
PANELS = ["front", "left_side", "back"]
paths = [OUT / f"panel_{n}.png" for n in PANELS]
VGGT_CKPT = REPO / "pivot" / "vggt-omega" / "checkpoints" / "vggt_omega_1b_512.pt"

imgs = load_and_preprocess_images([str(p) for p in paths], image_resolution=512)
model = VGGTOmega().eval()
model.load_state_dict(torch.load(VGGT_CKPT, map_location="cpu"))
model = model.cuda()
with torch.no_grad():
    preds = model(imgs.cuda())
depth = preds["depth"][0, ..., 0].cpu().numpy().astype(np.float32)
conf = preds["depth_conf"][0].cpu().numpy().astype(np.float32)
ext, K = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
ext = ext[0].cpu().numpy().astype(np.float32)
K = K[0].cpu().numpy().astype(np.float32)
del model; 
torch.cuda.empty_cache()
print("views", depth.shape, "relative rotations:")
for i in range(3):
    print(" ", PANELS[i], "R", np.round(ext[i][:3, :3], 3).tolist(), "t", np.round(ext[i][:3, 3], 3).tolist())

V = len(PANELS)
import run_all as R2

seg_model = R2.load_dense("seg", "seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
                          "seg/sapiens2_0.4b_seg.safetensors")
from sapiens.dense.visualizers import SegVisualizer
seg_vis = SegVisualizer(class_palette_type="dome29", with_labels=False)
pose_model = R2.load_pose()

worlds, masks, heads = {}, {}, {}
for i, n in enumerate(PANELS):
    bgr = cv2.imread(str(paths[i]))
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
    Xc = np.stack([(xs - K[i, 0, 2]) / K[i, 0, 0] * d,
                   (ys - K[i, 1, 2]) / K[i, 1, 1] * d, d], 1)
    Rm, tt = ext[i][:3, :3], ext[i][:3, 3]
    worlds[n] = (Rm.T @ (Xc - tt[None]).T).T
    masks[n] = m
    heads[n] = head[m]
    print(n, "person points", len(worlds[n]), "head points", int(heads[n].sum()))
del seg_model, seg_vis, pose_model; R2.free({})

# normalise scale: put all clouds in the front camera frame (world = front camera already up to scale)
# chamfer pairwise, all points and head subset (top 18% of the body in front view)
def chamfer(A, B):
    A = A[:: max(1, len(A) // 3000)]
    B = B[:: max(1, len(B) // 3000)]
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.median(d) * 1000), float(np.percentile(d, 90) * 1000)

head_sel = heads

report = {"n_points": {n: int(len(worlds[n])) for n in PANELS},
          "extrinsics": {n: {"R": ext[i][:3, :3].round(3).tolist(), "t": ext[i][:3, 3].round(3).tolist()}
                         for i, n in enumerate(PANELS)}}
for i in range(V):
    for j in range(i + 1, V):
        a, b = PANELS[i], PANELS[j]
        med, p90 = chamfer(worlds[a], worlds[b])
        hm, hp = chamfer(worlds[a][head_sel[a]], worlds[b][head_sel[b]])
        report[f"{a}__{b}"] = {"all_median_mm": round(med, 1), "all_p90_mm": round(p90, 1),
                               "head_median_mm": round(hm, 1), "head_p90_mm": round(hp, 1)}
        print(a, b, json.dumps(report[f"{a}__{b}"]))

# visual: project all clouds into the front view, colour-coded
H0, W0 = 900, 900
K0 = K[0]
pts = np.concatenate([worlds[n] for n in PANELS])
cols = np.concatenate([np.tile(c, (len(worlds[n]), 1)) for n, c in zip(PANELS, [(1, 0, 0), (0, 1, 0), (0, 0, 1)])])
y = pts[:, 1]
zz = pts[:, 2]
u = (K0[0, 0] * pts[:, 0] / np.maximum(zz, 1e-6) + K0[0, 2])
v = (K0[1, 1] * pts[:, 1] / np.maximum(zz, 1e-6) + K0[1, 2])
# fit into the canvas
u = (u - u.min()) / max(1e-6, (u.max() - u.min())) * (W0 - 1)
v = (v - v.min()) / max(1e-6, (v.max() - v.min())) * (H0 - 1)
canvas = np.full((H0, W0, 3), 240, np.uint8)
for x, yy, c in zip(u.astype(int), v.astype(int), cols):
    if 0 <= x < W0 and 0 <= yy < H0:
        canvas[yy, x] = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
cv2.imwrite(str(OUT / "vggt_points_rgb.png"), canvas)
(OUT / "vggt_pointcloud_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("wrote", OUT / "vggt_points_rgb.png")
