"""6-view head matching: fit MHR to each of the 6 Muse head panels, align by head landmarks, chamfer."""
import json
import sys
import time
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import cv2
import numpy as np

import joint_solve as J
import run_all as R

STEM = sys.argv[1] if len(sys.argv) > 1 else "teenage-woman-long-wavy-white-gray__01971-2947889296"
JOB = BASE / "outputs" / STEM
OUT = JOB / "threedb"
PANELS = ["muse_face_0.png", "muse_face_m45.png", "muse_face_45.png",
          "muse_face_m90.png", "muse_face_90.png", "muse_face_180.png"]
YAWS = [0.0, -45.0, 45.0, -90.0, 90.0, 180.0]

pose_model = R.load_pose()
model3d, est, faces = R.load_3db()

fits = {}
for name, yaw in zip(PANELS, YAWS):
    pth = OUT / name
    bgr = cv2.imread(str(pth))
    h, w = bgr.shape[:2]
    t0 = time.time()
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
    vroot, lm = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta,
                                     model_params=params, expr=expr)
    verts_cam = vroot + camr[None]
    lm_cam = lm + camr[None]
    fits[name] = dict(verts=verts_cam, lm=lm_cam, beta=beta)
    m = joint.get("metrics") or {}
    print(f"{name:22s} yaw {yaw:+6.1f} beta {np.linalg.norm(beta):.2f} "
          f"face {(m.get('joint') or {}).get('face_err_px')} ear {(m.get('joint') or {}).get('ear_err_px')} "
          f"({time.time()-t0:.1f}s)", flush=True)
del pose_model, model3d; R.free({})


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


ref_v, ref_lm = head_pts(fits[PANELS[0]]["verts"], fits[PANELS[0]]["lm"])
aligned = {}
report = {"subject": STEM, "pairs": {}}
for name in PANELS:
    V, L = head_pts(fits[name]["verts"], fits[name]["lm"])
    s, Rm, t = procrustes(L, ref_lm)
    aligned[name] = s * (V @ Rm.T) + t

for i in range(len(PANELS)):
    for j in range(i + 1, len(PANELS)):
        a, b = PANELS[i], PANELS[j]
        A, B = aligned[a], aligned[b]
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
        report["pairs"][f"{a}__{b}"] = {"median_mm": round(float(np.median(d) * 1000), 1),
                                        "p90_mm": round(float(np.percentile(d, 90) * 1000), 1)}
vals = [v["median_mm"] for v in report["pairs"].values()]
report["summary"] = {"n_pairs": len(vals), "median_over_pairs_mm": round(float(np.median(vals)), 1),
                     "best_mm": round(float(np.min(vals)), 1), "worst_mm": round(float(np.max(vals)), 1)}
(OUT / "head6_fit_matching.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report["summary"], indent=1))

# visual: 6 aligned heads in one ortho view, colours per yaw
from importlib import import_module
sys.path.insert(0, str(BASE / "outputs" / "_global" / "meta-image-test" / "ortho-neutral"))
import render_ortho as ro
S = ro.SIZE
allp = np.vstack(list(aligned.values()))
lo, hi = np.percentile(allp[:, 1], [1, 99])
cx = allp[:, 0].mean()
side = 1.15 * (hi - lo)
sc = (S * ro.SS) / side
x0, y0 = cx - side / 2, (lo + hi) / 2 - side / 2
canvas = np.full((S, S, 3), 238, np.uint8)
palette = [(230, 60, 60), (240, 150, 30), (200, 220, 40), (60, 200, 80), (50, 120, 240), (150, 60, 230)]
for k, name in enumerate(PANELS):
    P = aligned[name]
    u = ((P[:, 0] - x0) * sc).astype(int); v = ((P[:, 1] - y0) * sc).astype(int)
    ok = (u >= 0) & (u < S) & (v >= 0) & (v < S)
    canvas[v[ok], u[ok]] = palette[k]
cv2.imwrite(str(OUT / "head6_fit_points.png"), canvas)
print("wrote", OUT / "head6_fit_matching.json", OUT / "head6_fit_points.png")
