"""Test the Muse 1x3 full-body pose strip: split, fit each panel, compare identities and aligned head geometry."""
import json
import sys
import time
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import cv2
import numpy as np
import torch

import joint_solve as J
import run_all as R

STEM = "Emiru"
JOB = BASE / "outputs" / STEM
STRIP = JOB / "threedb" / "muse_body_strip_1x3.png"
OUT = JOB / "threedb" / "pose_strip_test"
OUT.mkdir(parents=True, exist_ok=True)
PANELS = ["front", "left_side", "back"]

img = cv2.imread(str(STRIP))
H, W = img.shape[:2]
q = W // 3
crops = {n: img[:, i * q:(i + 1) * q] for i, n in enumerate(PANELS)}
for n, c in crops.items():
    cv2.imwrite(str(OUT / f"panel_{n}.png"), c)

models = {}
def get(k):
    if k not in models:
        if k == "pose":
            models[k] = R.load_pose()
        elif k == "seg":
            R.sapiens_imports()
            from sapiens.dense.visualizers import SegVisualizer
            models["seg_vis"] = SegVisualizer(class_palette_type="dome29", with_labels=False)
            models[k] = R.load_dense("seg", "seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
                                     "seg/sapiens2_0.4b_seg.safetensors")
        elif k == "normal":
            models[k] = R.load_dense("normal", "normal/metasim_render_people/sapiens2_0.4b_normal_metasim_render_people-1024x768.py",
                                     "normal/sapiens2_0.4b_normal.safetensors")
        elif k == "depth":
            models[k] = R.load_dense("pointmap", "pointmap/render_people/sapiens2_0.4b_pointmap_render_people-1024x768.py",
                                     "pointmap/sapiens2_0.4b_pointmap.safetensors")
        elif k == "3db":
            models["3db"] = R.load_3db()
    return models[k]

res = {}
for name in PANELS:
    bgr = crops[name]
    h, w = bgr.shape[:2]
    t = {}
    t0 = time.time()
    kp_any, _ = R.pose_run(get("pose"), bgr)
    bbox = R.person_box(kp_any, w, h)
    kp, score = R.pose_run(get("pose"), bgr, bbox)
    t["pose"] = round(time.time() - t0, 2)
    t0 = time.time()
    labels, _ = R.seg_run(get("seg"), models["seg_vis"], bgr)
    skin = R.skin_mask_from_labels(labels)
    t["seg"] = round(time.time() - t0, 2)
    t0 = time.time()
    sn, _ = R.normal_run(get("normal"), bgr, labels)
    t["normal"] = round(time.time() - t0, 2)
    t0 = time.time()
    depth, _ = R.depth_run(get("depth"), bgr, labels)
    t["depth"] = round(time.time() - t0, 2)
    t0 = time.time()
    model3d, est, faces = get("3db")
    out, verts, f = R.fit_3db(est, bgr, bbox)
    cam = np.asarray(out["pred_cam_t"], np.float64)
    ref = R.identity_refine(model3d, out, kp, score, w, h, f, cam)
    beta_ref = np.asarray(ref.get("beta", out["shape_params"]), np.float64)
    joint = R._solve_joint(model3d, out, kp, score, w, h, f, cam, beta_init=beta_ref,
                           identity_only=ref, iters=300, faces=faces, normal_weight=0.0)
    t["3db_joint"] = round(time.time() - t0, 2)
    beta = np.asarray(joint.get("shape_params", beta_ref))
    params = np.asarray(joint.get("mhr_model_params", out["mhr_model_params"]))
    expr = np.asarray(joint.get("expr_params", np.zeros(72)))
    vroot, lm = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params, expr=expr)
    camr = np.asarray(joint.get("cam_t", cam))
    verts_cam = vroot + camr[None]
    lm_cam = lm + camr[None]
    m = (joint.get("metrics") or {})
    res[name] = {
        "status": "ok", "timings": t,
        "identity_l2": round(float(np.linalg.norm(beta)), 3),
        "expr_l2": round(float(np.linalg.norm(expr)), 3),
        "metrics": {k: (m.get(k) or {}) for k in ("before_3db", "identity_only", "joint")},
        "beta": beta.tolist(), "cam_t": camr.tolist(),
    }
    np.savez_compressed(OUT / f"fit_{name}.npz", verts=verts_cam, lm=lm_cam, beta=beta, params=params,
                        expr=expr, cam_t=camr)
    print(name, json.dumps({k: res[name][k] for k in ("identity_l2", "expr_l2")}),
          {k: res[name]["metrics"]["joint"].get(k) for k in ("face_err_px", "ear_err_px")}, flush=True)

# ---- compare identities and aligned head geometry
def procrustes(A, B):
    """similarity transform mapping A -> B (rows)."""
    Ac, Bc = A.mean(0), B.mean(0)
    A0, B0 = A - Ac, B - Bc
    U, S, Vt = np.linalg.svd(B0.T @ A0)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    s = S.sum() / (A0 ** 2).sum()
    t = Bc - s * (Rm @ Ac)
    return s, Rm, t


def head_pts(verts, lm):
    fpts = lm[70:272]
    c = fpts.mean(0)
    r = np.percentile(np.linalg.norm(fpts - c, axis=1), 95)
    keep = np.linalg.norm(verts - c, axis=1) < 2.2 * r
    return verts[keep], lm[70:272]


report = {"panels": res, "comparisons": {}}
fits = {n: np.load(OUT / f"fit_{n}.npz") for n in PANELS}
for i in range(len(PANELS)):
    for j in range(i + 1, len(PANELS)):
        a, b = PANELS[i], PANELS[j]
        Va, La = head_pts(fits[a]["verts"], fits[a]["lm"])
        Vb, Lb = head_pts(fits[b]["verts"], fits[b]["lm"])
        s, Rm, t = procrustes(La, Lb)
        Va2 = (s * (Va @ Rm.T)) + t
        d = np.sqrt(((Va2[:, None, :] - Vb[None, :, :]) ** 2).sum(-1)).min(1)
        db = np.linalg.norm(fits[a]["beta"] - fits[b]["beta"])
        report["comparisons"][f"{a}_{b}"] = {
            "identity_beta_L2_diff": round(float(db), 3),
            "head_chamfer_median_mm": round(float(np.median(d) * 1000), 1),
            "head_chamfer_p90_mm": round(float(np.percentile(d, 90) * 1000), 1),
            "fit_scale": round(float(s), 4),
        }
        print(a, b, json.dumps(report["comparisons"][f"{a}_{b}"]), flush=True)

(OUT / "pose_strip_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("wrote", OUT / "pose_strip_report.json")
