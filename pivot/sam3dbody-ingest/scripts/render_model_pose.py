"""Render the refined MHR mesh in the fitted camera pose (same pose as the photo): clay + photo-textured."""
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
bgr = cv2.imread(str(photo)); H, W = bgr.shape[:2]
stats = json.loads((JOB / "stats.json").read_text(encoding="utf-8"))
f = float(stats["focal_px"])
j = json.loads((JOB / "threedb" / "identity_joint_hyden.json").read_text(encoding="utf-8"))
ref = json.loads((JOB / "threedb" / "identity_joint.json").read_text(encoding="utf-8"))
kps = np.load(JOB / "sapiens" / "raw" / "sapiens_pose.npz")["kp"][:, :2]

model3d, est, faces = R.load_3db()
beta = np.asarray(j["refined_shape_params"]); params = np.asarray(j["refined_mhr_model_params"])
cam = np.asarray(j["refined_cam_t"]); expr = np.asarray(j["expr_params"])
vroot, _ = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params, expr=expr)
verts = vroot + cam[None]
vcol = R.texture_mesh(verts, faces, bgr, f, W, H)
clay_img, clay_mask = R.render_rgb(verts, faces, f, W, H, colour=(0.62, 0.65, 0.70), ss=1.0)
tex_img, tex_mask = R.render_rgb(verts, faces, f, W, H, vcol=vcol, ss=1.0)
del model3d; R.free({})

# crop around the face/head keypoints
face = kps[70:272]
x1, y1 = face.min(0).astype(int); x2, y2 = face.max(0).astype(int)
pad = int(0.35 * max(x2 - x1, y2 - y1))
x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
side = max(x2 - x1, y2 - y1)
x2 = min(W, x1 + side); y2 = min(H, y1 + side)
x1 = max(0, x2 - side); y1 = max(0, y2 - side)
print("crop", x1, y1, x2, y2, "side", side)

panels = []
for name, img in (("original photo", bgr),
                  ("MHR mesh - clay", cv2.cvtColor(clay_img, cv2.COLOR_RGB2BGR)),
                  ("MHR mesh - photo texture", cv2.cvtColor(tex_img, cv2.COLOR_RGB2BGR))):
    c = img[y1:y2, x1:x2].copy()
    c = cv2.resize(c, (800, 800), interpolation=cv2.INTER_AREA)
    cv2.putText(c, name, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 5, cv2.LINE_AA)
    cv2.putText(c, name, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (250, 250, 250), 2, cv2.LINE_AA)
    panels.append(c)
sep = np.full((800, 8, 3), 30, np.uint8)
sheet = np.hstack([panels[0], sep, panels[1], sep, panels[2]])
cv2.imwrite(str(JOB / "threedb" / "results_model_pose.jpg"), sheet)
print("wrote", JOB / "threedb" / "results_model_pose.jpg")
