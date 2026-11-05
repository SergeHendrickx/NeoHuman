"""Canonical A-pose turnaround for every job: keep identity + scale, zero the pose, render 0/±45/90/180."""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import cv2
import numpy as np

import run_all as R

OUT_ROOT = BASE / "outputs"
TILE = 520
YAWS = [(-90.0, "-90"), (-45.0, "-45"), (0.0, "0"), (45.0, "+45"), (90.0, "+90"), (180.0, "180")]

model3d, est, faces = R.load_3db()

panels_by_job = {}
jobs = []
for d in sorted(OUT_ROOT.iterdir()):
    jp = d / "threedb" / "identity_joint.json"
    if not d.is_dir() or not jp.exists():
        continue
    j = json.loads(jp.read_text(encoding="utf-8"))
    beta = np.asarray(j["refined_shape_params"], np.float64)
    params = np.asarray(j["refined_mhr_model_params"], np.float64).copy()
    params[:6] = 0.0
    params[6:136] = 0.0                      # canonical pose: zero body euler == MHR's relaxed A-pose
    vroot, _ = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params,
                                    expr=np.zeros(72))
    verts = vroot - vroot.mean(0)
    center = verts.mean(0)
    panels = []
    for yaw, lab in YAWS:
        th = np.deg2rad(yaw)
        c, s = np.cos(th), np.sin(th)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        p = (verts - center[None]) @ rot.T
        height = float(np.ptp(p[:, 1]))
        f2 = 1.1 * TILE
        dist = f2 * height / (0.86 * TILE)
        world = p + np.array([0.0, 0.0, dist])
        mesh, mat = R._make_mesh(world, faces, colour=R.MESH_COLOUR)
        img_hi, mask_hi = R._render_once(TILE, TILE, f2, mesh, mat)
        img, _ = R._finish(img_hi, mask_hi, TILE, TILE)
        im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(im, lab, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(im, lab, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 1, cv2.LINE_AA)
        panels.append(im)
    sheet = np.hstack(panels)
    cv2.imwrite(str(d / "threedb" / "turnaround_apose.jpg"), sheet)
    (d / "threedb" / "pose_canonical.json").write_text(json.dumps({
        "pose": "A-pose (body euler zeroed; MHR rest pose)", "yaws_deg": [y for y, _ in YAWS],
        "identity_shape_params": beta.round(5).tolist(),
        "scale_state": params[136:204].round(5).tolist(),
        "expression": "zero (neutral)",
    }, indent=2), encoding="utf-8")
    panels_by_job[d.name] = panels
    jobs.append(d.name)
del model3d; R.free({})
print("A-pose turnarounds:", len(jobs))

# contact sheet with a few subjects
pick = [n for n in ("Emiru", "input", "batel-studio-ttSRjiYG_WM-unsplash_1", "joseph-gonzalez-iFgRcqHznqg-unsplash_1",
                    "michael-dam-mEZ3PoFGs_k-unsplash_1", "tex") if n in panels_by_job]
rows = []
for n in pick:
    row = np.hstack(panels_by_job[n])
    label = np.full((70, row.shape[1], 3), 24, np.uint8)
    cv2.putText(label, n, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
    rows.append(np.vstack([label, row]))
W = max(r.shape[1] for r in rows)
rows = [np.hstack([r, np.full((r.shape[0], W - r.shape[1], 3), 24, np.uint8)]) for r in rows]
contact = np.vstack(rows)
cv2.imwrite(str(OUT_ROOT / "_global" / "apose_turnarounds_contact.jpg"), contact)
print("wrote", OUT_ROOT / "_global" / "apose_turnarounds_contact.jpg")
