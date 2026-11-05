"""Export the MHR-fit 6-view head point cloud: fit each panel, align by head landmarks, fuse -> PLY."""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import cv2
import numpy as np
import trimesh

import run_all as R

STEM = sys.argv[1] if len(sys.argv) > 1 else "teenage-woman-long-wavy-white-gray__01971-2947889296"
JOB = BASE / "outputs" / STEM
OUT = JOB / "threedb"
PANELS = ["muse_face_0.png", "muse_face_m45.png", "muse_face_45.png",
          "muse_face_m90.png", "muse_face_90.png", "muse_face_180.png"]
YAWS = [0.0, -45.0, 45.0, -90.0, 90.0, 180.0]

pose_model = R.load_pose()
model3d, est, faces = R.load_3db()


def procrustes(A, B):
    Ac, Bc = A.mean(0), B.mean(0)
    A0, B0 = A - Ac, B - Bc
    U, S, Vt = np.linalg.svd(B0.T @ A0)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    s = S.sum() / (A0 ** 2).sum()
    return s, Rm, Bc - s * (Rm @ Ac)


def head_pts(verts, lm):
    fp = lm[70:272]
    c = fp.mean(0)
    r = float(np.percentile(np.linalg.norm(fp - c[None], axis=1), 95))
    keep = np.linalg.norm(verts - c, axis=1) < 2.0 * r
    return verts[keep], fp


fits = {}
ref_v = ref_lm = None
for name, yaw in zip(PANELS, YAWS):
    bgr = cv2.imread(str(OUT / name))
    h, w = bgr.shape[:2]
    kp_any, _ = R.pose_run(pose_model, bgr)
    bbox = R.person_box(kp_any, w, h)
    kp, score = R.pose_run(pose_model, bgr, bbox)
    out, verts, f = R.fit_3db(est, bgr, bbox)
    cam = np.asarray(out["pred_cam_t"], np.float64)
    ref = R.identity_refine(model3d, out, kp, score, w, h, f, cam)
    beta_ref = np.asarray(ref.get("beta", out["shape_params"]), np.float64)
    joint = R._solve_joint(model3d, out, kp, score, w, h, f, cam, beta_init=beta_ref,
                           identity_only=ref, iters=250, faces=faces, normal_weight=0.0)
    beta = np.asarray(joint.get("shape_params", beta_ref))
    params = np.asarray(joint.get("mhr_model_params", out["mhr_model_params"]))
    expr = np.asarray(joint.get("expr_params", np.zeros(72)))
    camr = np.asarray(joint.get("cam_t", cam))
    vroot, lm = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params, expr=expr)
    verts_cam = vroot + camr[None]
    lm_cam = lm + camr[None]
    V, L = head_pts(verts_cam, lm_cam)
    if ref_v is None:
        ref_v, ref_lm = V, L
        s, Rm, t = 1.0, np.eye(3), np.zeros(3)
    else:
        s, Rm, t = procrustes(L, ref_lm)
    fits[name] = s * (V @ Rm.T) + t
    print(name, "head verts", len(fits[name]), "scale", round(float(s), 4), flush=True)
del pose_model, model3d; R.free({})

palette = [(230, 50, 50), (245, 150, 30), (200, 220, 40), (60, 200, 90), (50, 120, 240), (150, 60, 230)]
pts = np.concatenate([fits[n] for n in PANELS])
cols = np.concatenate([np.tile(np.array(palette[k], np.uint8), (len(fits[n]), 1)) for k, n in enumerate(PANELS)])
trimesh.PointCloud(pts, colors=cols).export(OUT / "head6_mhr_fused_cloud.ply")
print("wrote", OUT / "head6_mhr_fused_cloud.ply", len(pts), "points")

# fixed visual: centre on the x/y range of the cloud
S = 900
x_lo, x_hi = np.percentile(pts[:, 0], [1, 99])
y_lo, y_hi = np.percentile(pts[:, 1], [1, 99])
span = 1.1 * max(x_hi - x_lo, y_hi - y_lo)
cx, cy = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
sc = S / span
x0, y0 = cx - span / 2, cy - span / 2
canvas = np.full((S, S, 3), 240, np.uint8)
for k in np.argsort(-pts[:, 2]):
    x = int((pts[k, 0] - x0) * sc); y = int((pts[k, 1] - y0) * sc)
    if 0 <= x < S and 0 <= y < S:
        canvas[y, x] = cols[k][::-1]
for leg_i, n in enumerate(PANELS):
    pass
cv2.imwrite(str(OUT / "head6_mhr_fused_cloud.png"), canvas)
print("wrote", OUT / "head6_mhr_fused_cloud.png")
