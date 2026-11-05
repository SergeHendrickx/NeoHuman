"""Test Meta HyDen-MoGeV2 metric point model on a job photo and compare its metric geometry to our mesh."""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
MD = REPO / "pivot" / "metadepth"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(MD))
sys.path.insert(0, str(MD.parent))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import run_all as R
from metadepth.mogev2 import HyDenMoGe, MODEL_CONFIGS

STEM = sys.argv[1] if len(sys.argv) > 1 else "tex"
JOB = BASE / "outputs" / STEM
PHOTO = next((BASE / "input-images").rglob(f"{STEM}.*"))
CKPT = REPO / "pivot" / "hyden-mogev2-metric-point" / "hyden_mogev2_metric_point_vitl_fp32_f1066593896.pth"
OUT = JOB / "sapiens"
OUT.mkdir(parents=True, exist_ok=True)

cfg = dict(MODEL_CONFIGS["vitl_dinov2"])
cfg.pop("normal_head", None)          # this checkpoint is the metric-point variant (no normal head)
model = HyDenMoGe(**cfg)
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
res = model.load_state_dict(sd, strict=False)
print("missing keys:", len(res.missing_keys), "unexpected keys:", len(res.unexpected_keys))
if res.missing_keys:
    print("  missing sample:", res.missing_keys[:5])
if res.unexpected_keys:
    print("  unexpected sample:", res.unexpected_keys[:5])
del sd
model = model.to("cuda").eval()

raw = Image.open(PHOTO).convert("RGB")
W0, H0 = raw.size
tf = transforms.Compose([transforms.Resize((518, 518)), transforms.ToTensor()])
inp = tf(raw).unsqueeze(0).to("cuda")
with torch.no_grad():
    out = model(inp)
print("output keys:", sorted(out.keys()))
for k, v in out.items():
    if torch.is_tensor(v):
        print(" ", k, tuple(v.shape), v.dtype, float(v.min()), float(v.max()))
points = out["points"][0].cpu().numpy().astype(np.float32)     # H,W,3 (518x518)
valid = out.get("valid_mask", out.get("mask"))
valid = valid[0].cpu().numpy().astype(bool) if valid is not None else np.ones(points.shape[:2], bool)
print("valid fraction:", round(float(valid.mean()), 3))

# scale point map back to the original photo resolution for comparison
pts_full = cv2.resize(points, (W0, H0), interpolation=cv2.INTER_LINEAR)
mask_full = cv2.resize(valid.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST) > 0
np.savez_compressed(JOB / "sapiens" / "raw" / "hyden_mogev2_metric_point.npz",
                    points_518=points, valid_518=valid, points_full=pts_full, valid_full=mask_full)

# compare against our fitted mesh depth in the same pixel grid (project mesh with our camera)
model3d, est, faces = R.load_3db()
j = json.loads((JOB / "threedb" / "identity_joint.json").read_text(encoding="utf-8"))
stats = json.loads((JOB / "stats.json").read_text(encoding="utf-8"))
f = float(stats["focal_px"])
beta = np.asarray(j["refined_shape_params"]); params = np.asarray(j["refined_mhr_model_params"])
cam_t = np.asarray(j["refined_cam_t"]); expr = np.asarray(j.get("metrics", {}).get("expr_params", np.zeros(72)))
vroot, lm = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta,
                                 model_params=params, expr=expr)
verts = vroot + cam_t[None]
lm = lm + cam_t[None]
del model3d; R.free({})

# project a few key landmarks and compare metric depth
kp = np.load(JOB / "sapiens" / "raw" / "sapiens_pose.npz")["kp"][:, :2]
rows = [1, 2, 3, 4, 77, 178]
uv = kp[rows]
d_mesh = lm[rows, 2]
xi = np.clip(uv[:, 0].astype(int), 0, W0 - 1); yi = np.clip(uv[:, 1].astype(int), 0, H0 - 1)
p_moge = pts_full[yi, xi]
ok = mask_full[yi, xi] & (np.abs(p_moge[:, 2]) > 1e-6) & np.isfinite(p_moge).all(1)
print("landmark compare (mesh z vs MoGe z):")
for r, zm, pm, o in zip(rows, d_mesh, p_moge, ok):
    print(f"  row {r:3d} mesh_z {zm:6.3f} m  moge_z {pm[2] if o else float('nan'):6.3f}  ratio {zm/pm[2] if o and pm[2] else float('nan'):.3f}")

s = float(np.median(d_mesh[ok] / p_moge[ok, 2]))
print("median mesh_z / moge_z scale:", round(s, 4))

# ear width in metric 3D (MoGe) vs mesh, plus face height ratio
P = p_moge
ok2 = ok.copy()
if ok2.all():
    ratio_ear = np.linalg.norm(P[2] - P[3]) / np.linalg.norm(P[4] - 0.5 * (P[0] + P[1]))
    ratio_mesh = np.linalg.norm(lm[3] - lm[4]) / np.linalg.norm(lm[77] - 0.5 * (lm[1] + lm[2]))
    print("ear_w/face_h: moge %.3f vs mesh %.3f" % (ratio_ear, ratio_mesh))
    print("ear width m: moge %.4f vs mesh %.4f" % (np.linalg.norm(P[2] - P[3]), np.linalg.norm(lm[3] - lm[4])))

report = {"photo": str(PHOTO), "keys": sorted(out.keys()), "valid_fraction": round(float(valid.mean()), 3),
          "mesh_to_moge_scale": round(s, 4),
          "moge_point_range_m": [round(float(np.percentile(points[..., 2][valid], 1)), 3),
                                 round(float(np.percentile(points[..., 2][valid], 99)), 3)]}
(JOB / "sapiens" / "hyden_mogev2_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=1))
