"""MHR multi-view match: fit the MHR body independently to every panel of a view sheet,
align the fits by head landmarks (Procrustes), and report the residual geometry.

This is the strong consistency check for generated/view sheets: per-panel MHR fits align at
a few millimetres on the same subject, roughly 5-10x better than VGGT's monocular estimates.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
INGEST = REPO / "pivot" / "sam3dbody-ingest"
if str(INGEST) not in sys.path:
    sys.path.insert(0, str(INGEST))


def _jsonable(o):
    """Recursively convert numpy values/arrays so reports and stats always serialise."""
    import numpy as _np
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, _np.generic):
        return o.item()
    return o


def _head_pts(verts, lm):
    fp = lm[70:272]
    c = fp.mean(0)
    r = float(np.percentile(np.linalg.norm(fp - c[None], axis=1), 95))
    keep = np.linalg.norm(verts - c[None], axis=1) < 2.0 * r
    return verts[keep], fp


def _procrustes(A, B):
    Ac, Bc = A.mean(0), B.mean(0)
    A0, B0 = A - Ac, B - Bc
    U, S, Vt = np.linalg.svd(B0.T @ A0)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    s = S.sum() / (A0 ** 2).sum()
    return s, Rm, Bc - s * (Rm @ Ac)


def _chamfer(A, B):
    A = A[:: max(1, len(A) // 3000)]
    B = B[:: max(1, len(B) // 3000)]
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.median(d) * 1000), float(np.percentile(d, 90) * 1000)


def mhr_multiview_match(bgr, out: Path, rows: int | None = None, cols: int | None = None,
                        progress=None) -> tuple[str, list[str], dict]:
    """Fit MHR to every panel, align head-to-head, report mm residuals + fused head cloud."""
    import run_all as R
    from avatar_extra import _grid_for, split_sheet

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda *a, **k: None)

    rows, cols = _grid_for(bgr.shape[1], bgr.shape[0], rows, cols)
    if rows * cols < 2:
        raise ValueError("need at least 2 panels for a multi-view match")
    panels = split_sheet(bgr, rows, cols)
    pdir = out / "panels"
    pdir.mkdir(exist_ok=True)
    paths = []
    for i, p in enumerate(panels):
        fp = pdir / f"panel_{i:02d}.png"
        cv2.imwrite(str(fp), p)
        paths.append(fp)
    cv2.imwrite(str(out / "source_sheet.png"), bgr)

    progress("start", f"mhr-match: {rows}x{cols} grid, loading models", 0.03)
    pose_model = R.load_pose()
    model3d, est, faces = R.load_3db()
    fits = []
    try:
        for i, pth in enumerate(paths):
            bgr_p = cv2.imread(str(pth))
            h, w = bgr_p.shape[:2]
            progress("fit", f"mhr-match: fitting panel {i + 1}/{len(paths)}",
                     0.05 + 0.85 * i / len(paths))
            kp_any, _ = R.pose_run(pose_model, bgr_p)
            bbox = R.person_box(kp_any, w, h)
            kp, score = R.pose_run(pose_model, bgr_p, bbox)
            out3d, verts, f = R.fit_3db(est, bgr_p, bbox)
            cam = np.asarray(out3d["pred_cam_t"], np.float64)
            beta0 = np.asarray(out3d["shape_params"], np.float64)
            joint = R._solve_joint(model3d, out3d, kp, score, w, h, f, cam,
                                   beta_init=beta0, iters=150, faces=faces, normal_weight=0.0)
            beta = np.asarray(joint.get("shape_params", beta0))
            params = np.asarray(joint.get("mhr_model_params", out3d["mhr_model_params"]))
            expr = np.asarray(joint.get("expr_params", np.zeros(72)))
            camr = np.asarray(joint.get("cam_t", cam))
            vroot, lm = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta,
                                             model_params=params, expr=expr)
            vcam = vroot + camr[None]
            lcam = lm + camr[None]
            m = (joint.get("metrics") or {}).get("joint") or (joint.get("metrics") or {})
            fits.append({"index": i, "verts": vcam, "lm": lcam, "beta": beta,
                         "focal": float(f), "size_wh": [int(w), int(h)],
                         "face_err_px": m.get("face_err_px"), "ear_err_px": m.get("ear_err_px")})
    finally:
        del pose_model
        try:
            del model3d
        except Exception:
            pass
        R.free({})

    progress("align", "mhr-match: aligning fits by head landmarks", 0.9)
    ref_v, ref_lm = _head_pts(fits[0]["verts"], fits[0]["lm"])
    aligned, per_panel = [], []
    for fit in fits:
        V, L = _head_pts(fit["verts"], fit["lm"])
        s, Rm, t = _procrustes(L, ref_lm)
        A = s * (V @ Rm.T) + t
        aligned.append(A)
        rms = float(np.sqrt(np.mean(np.sum((s * (L @ Rm.T) + t - ref_lm) ** 2, axis=1))) * 1000)
        ang = float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))
        per_panel.append({"panel": fit["index"], "beta_l2": round(float(np.linalg.norm(fit["beta"])), 3),
                          "focal_px": round(fit["focal"], 1),
                          "face_err_px": fit["face_err_px"], "ear_err_px": fit["ear_err_px"],
                          "align_scale": round(float(s), 4), "align_rot_deg": round(float(ang), 1),
                          "align_rms_mm": round(rms, 2)})

    def _num(v):
        try:
            return None if v is None else float(v)
        except Exception:
            return None

    for pp in per_panel:
        pp['face_err_px'] = _num(pp.get('face_err_px'))
        pp['ear_err_px'] = _num(pp.get('ear_err_px'))

    pairs = {}
    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            med, p90 = _chamfer(aligned[i], aligned[j])
            pairs[f"panel{i:02d}__panel{j:02d}"] = {"median_mm": round(med, 1), "p90_mm": round(p90, 1)}
    vals = [v["median_mm"] for v in pairs.values()]
    summary = {"n_panels": len(aligned), "n_pairs": len(vals),
               "median_mm": round(float(np.median(vals)), 1) if vals else None,
               "best_mm": round(float(np.min(vals)), 1) if vals else None,
               "worst_mm": round(float(np.max(vals)), 1) if vals else None,
               "mean_align_rms_mm": round(float(np.mean([p["align_rms_mm"] for p in per_panel])), 2)}

    # fused head cloud, colour per panel
    import trimesh
    palette = [(230, 50, 50), (245, 150, 30), (200, 220, 40), (60, 200, 90),
               (50, 120, 240), (150, 60, 230), (30, 200, 200), (240, 60, 200)]
    pts = np.concatenate(aligned)
    vcolors = np.concatenate([np.tile(np.array(palette[i % len(palette)], np.uint8), (len(A), 1))
                              for i, A in enumerate(aligned)])
    trimesh.PointCloud(pts, colors=vcolors).export(out / "mhr_match_cloud.ply")

    # quick ortho-ish preview: front projection, painter's sort by z
    S = 900
    xy = pts[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    span = max(float((hi - lo).max()), 1e-6)
    uv = ((xy - lo) / span * (S - 1)).astype(int)
    canvas = np.full((S, S, 3), 240, np.uint8)
    for k in np.argsort(-pts[:, 2]):
        x, y = uv[k]
        y = S - 1 - y
        if 0 <= x < S and 0 <= y < S:
            canvas[y, x] = vcolors[k][::-1]
    cv2.imwrite(str(out / "mhr_match_cloud.png"), canvas)

    report = _jsonable({"grid": [rows, cols], "per_panel": per_panel, "pairs": pairs,
                        "summary": summary, "seconds": round(time.time() - t0, 1)})
    (out / "mhr_match_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    files = ["source_sheet.png", "mhr_match_report.json", "mhr_match_cloud.ply", "mhr_match_cloud.png"] + \
            [f"panels/{p.name}" for p in paths]
    summary_text = ("MHR multi-view match: {r}x{c} grid, {n} independent MHR fits aligned by head "
                    "landmarks - cross-view median {m} mm (best {b}, worst {w}), mean landmark RMS {e} mm.").format(
        r=rows, c=cols, n=len(aligned), m=summary["median_mm"], b=summary["best_mm"],
        w=summary["worst_mm"], e=summary["mean_align_rms_mm"])
    stats = _jsonable({"grid": [rows, cols], "n_panels": len(aligned), "summary": summary,
                       "seconds": report["seconds"], "report_file": "mhr_match_report.json",
                       "cloud_file": "mhr_match_cloud.ply", "preview_file": "mhr_match_cloud.png",
                       "per_panel": per_panel})
    return summary_text, files, stats
