"""Results sheet: baseline vs HyDen-depth-refined mesh over the photo, plus the metrics."""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import cv2
import numpy as np

import run_all as R

STEM = "tex"
JOB = BASE / "outputs" / STEM
photo = next((BASE / "input-images").rglob(f"{STEM}.*"))
bgr = cv2.imread(str(photo))
H, W = bgr.shape[:2]
stats = json.loads((JOB / "stats.json").read_text(encoding="utf-8"))
f = float(stats["focal_px"])
base = json.loads((JOB / "threedb" / "identity_joint.json").read_text(encoding="utf-8"))
ref = json.loads((JOB / "threedb" / "identity_joint_hyden.json").read_text(encoding="utf-8"))
report = json.loads((JOB / "threedb" / "hyden_refine_report.json").read_text(encoding="utf-8"))
kps = np.load(JOB / "sapiens" / "raw" / "sapiens_pose.npz")["kp"][:, :2]

model3d, est, faces = R.load_3db()


def mesh_overlay(j):
    beta = np.asarray(j["refined_shape_params"], np.float64)
    params = np.asarray(j["refined_mhr_model_params"], np.float64)
    cam = np.asarray(j["refined_cam_t"], np.float64)
    expr = np.asarray(j.get("expr_params", j.get("metrics", {}).get("expr_params", np.zeros(72))), np.float64)
    vroot, _ = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta,
                                    model_params=params, expr=expr)
    verts = vroot + cam[None]
    img, mask = R.render_rgb(verts, faces, f, W, H, colour=(0.55, 0.58, 0.62), ss=1.0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    m = mask.astype(bool)
    out = bgr.copy()
    out[m] = (0.55 * bgr[m] + 0.45 * img[m]).astype(np.uint8)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (0, 255, 0), 3)
    for pt in kps[70:272].astype(int):
        cv2.circle(out, tuple(pt), 2, (0, 255, 255), -1)
    return out


ov_b = mesh_overlay(base)
ov_r = mesh_overlay(ref)
del model3d; R.free({})

# crop to the head: bbox of face keypoints with margin, square
face = kps[70:272]
x1, y1 = face.min(0).astype(int); x2, y2 = face.max(0).astype(int)
pad = int(0.25 * max(x2 - x1, y2 - y1))
x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
side = max(x2 - x1, y2 - y1)
x2 = min(W, x1 + side); y2 = min(H, y1 + side)
x1 = max(0, x2 - side); y1 = max(0, y2 - side)
crops = []
for name, im in (("original", bgr), ("baseline fit", ov_b), ("HyDen-depth refined", ov_r)):
    c = im[y1:y2, x1:x2]
    c = cv2.resize(c, (700, 700), interpolation=cv2.INTER_AREA)
    cv2.putText(c, name, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 4, cv2.LINE_AA)
    cv2.putText(c, name, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (250, 250, 250), 1, cv2.LINE_AA)
    crops.append(c)

# metrics panel
pan = np.full((700, 700, 3), 245, np.uint8)
kb, ka = report["keypoints_before"], report["keypoints_after"]
db, da = report["depth_residual_mm"], report["depth_residual_mm"]
rows = [
    ("tex.webp - metric-depth refine", ""),
    ("", ""),
    ("keypoint face err", f"{kb['face_err_px']:.2f} -> {ka['face_err_px']:.2f} px"),
    ("keypoint ear err", f"{kb['ear_err_px']:.2f} -> {ka['ear_err_px']:.2f} px"),
    ("ear span", f"{kb['ear_span_px']:.1f} -> {ka['ear_span_px']:.1f} px"),
    ("dense depth median", f"{db['before_median']:.1f} -> {da['after_median']:.1f} mm"),
    ("dense depth p90", f"{db['before_p90']:.1f} -> {da['after_p90']:.1f} mm"),
    ("identity delta", f"{report['delta_beta_norm']:.2f} (cap 5)"),
    ("supervised face verts", f"{report['n_supervised_vertices']}"),
    ("HyDen metric scale", f"{report['depth_fit']['s']:.3f} x + {report['depth_fit']['t_m']*1000:.0f} mm"),
]
y = 60
for a, b in rows:
    if not a and not b:
        y += 20; continue
    cv2.putText(pan, a, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, cv2.LINE_AA)
    if b:
        cv2.putText(pan, b, (24, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (20, 90, 160), 2, cv2.LINE_AA)
    y += 78

sheet = np.full((720, 4 * 710 + 10, 3), 30, np.uint8)
for i, c in enumerate(crops + [pan]):
    sheet[10:710, 10 + i * 710:10 + i * 710 + 700] = c
cv2.imwrite(str(JOB / "threedb" / "results_hyden_refine.jpg"), sheet)
print("wrote", JOB / "threedb" / "results_hyden_refine.jpg")
