"""Cross-model depth/geometry matching on one view: HyDen-metric vs Sapiens2 pointmap (0.4b/1b) vs our mesh.
Each map is aligned to the mesh camera-frame z by a robust 1-D scale+shift on face pixels, then compared."""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import cv2
import numpy as np
import torch

import joint_solve as J
import run_all as R

STEM = sys.argv[1] if len(sys.argv) > 1 else "tex"
JOB = BASE / "outputs" / STEM
photo = next((BASE / "input-images").rglob(f"{STEM}.*"))
bgr = cv2.imread(str(photo)); H, W = bgr.shape[:2]
stats = json.loads((JOB / "stats.json").read_text(encoding="utf-8"))
j = json.loads((JOB / "threedb" / "identity_joint.json").read_text(encoding="utf-8"))
f = float(stats["focal_px"])
labels = cv2.imread(str(JOB / "sapiens" / "raw" / "sapiens_seg_labels.png"), cv2.IMREAD_GRAYSCALE)
skin = labels == 3

model3d, est, faces = R.load_3db()
beta = np.asarray(j["refined_shape_params"]); params = np.asarray(j["refined_mhr_model_params"])
cam = np.asarray(j["refined_cam_t"]); expr = np.asarray(j["metrics"]["expr_params"])
vroot, lm = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params, expr=expr)
verts = vroot + cam[None]
del model3d; R.free({})

# face-surface vertices (from the keypoint mapping) as sampling points
MAP = None  # not needed; use vertices whose projection lands on skin and that are front-facing
z = verts[:, 2]
uv = np.stack([f * verts[:, 0] / z + W / 2, f * verts[:, 1] / z + H / 2], 1)
inside = (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H) & (z > 0.2) & (z < 3.0)
xi = np.clip(uv[:, 0], 0, W - 1).astype(int); yi = np.clip(uv[:, 1], 0, H - 1).astype(int)
sel = inside & skin[yi, xi]
# keep only vertices in the head area (upper quarter of the frame)
sel &= (uv[:, 1] < 0.45 * H)
print("sample vertices:", int(sel.sum()))
pix = (xi[sel], yi[sel])
zm = z[sel]

maps = {}
hy = np.load(JOB / "sapiens" / "raw" / "hyden_metric_points.npy")  # 518x518x3
maps["HyDen-metric"] = cv2.resize(hy[..., 2], (W, H), interpolation=cv2.INTER_LINEAR)
d04 = J.load_pointmap_depth(STEM) if hasattr(J, "load_pointmap_depth") else None
maps["Sapiens2 pointmap 0.4b"] = np.load(JOB / "sapiens" / "raw" / "sapiens_depth_m.npy")
maps["Sapiens2 pointmap 1b"] = np.load(JOB / "sapiens" / "raw" / "sapiens_pointmap_1b_depth_m.npy")

report = {}
ref = None
for name, m in maps.items():
    v = m[pix]
    ok = np.isfinite(v) & (v > 0.05)
    if ok.sum() < 50:
        report[name] = {"ok": int(ok.sum()), "error": "too few valid samples"}
        continue
    A = np.stack([v[ok], np.ones(int(ok.sum()))], 1)
    s, t = np.linalg.lstsq(A, zm[ok], rcond=None)[0]
    pred = s * v + t
    r = (zm[ok] - pred) * 1000.0
    corr = float(np.corrcoef(zm[ok], v[ok])[0, 1])
    report[name] = {"n": int(ok.sum()), "scale": round(float(s), 4), "shift_m": round(float(t), 4),
                    "corr": round(corr, 4),
                    "residual_median_mm": round(float(np.median(np.abs(r))), 2),
                    "residual_p90_mm": round(float(np.percentile(np.abs(r), 90)), 2)}
    print(name, json.dumps(report[name]))
    if name == "HyDen-metric":
        ref = pred

# pairwise agreement between models in mesh units (HyDen is the reference frame)
if ref is not None:
    for name, m in maps.items():
        if name == "HyDen-metric":
            continue
        v = m[pix]
        ok = np.isfinite(v) & (v > 0.05)
        A = np.stack([v[ok], np.ones(int(ok.sum()))], 1)
        s, t = np.linalg.lstsq(A, zm[ok], rcond=None)[0]
        pred = s * v + t
        common = ok.copy()
        d = (pred[common] - ref[common]) * 1000.0
        report.setdefault("pairwise_vs_HyDen", {})[name] = {
            "median_abs_mm": round(float(np.median(np.abs(d))), 2),
            "p90_mm": round(float(np.percentile(np.abs(d), 90)), 2)}

(JOB / "sapiens" / "cross_model_depth.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=1))
