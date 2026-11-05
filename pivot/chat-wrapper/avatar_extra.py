"""Extra debug actions for the chat-wrapper flow viewer: VGGT multi-view check + canonical A-pose export.

These are imported by chat_server.py's debug router. Each function follows the same contract as
``body_fit_debug``: (summary: str, files: list[str], stats: dict) and writes artifacts into ``out``.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
INGEST = REPO / "pivot" / "sam3dbody-ingest"
VGGT_DIR = Path(os.environ.get("VGGT_OMEGA_ROOT", str(REPO / "pivot" / "vggt-omega")))
VGGT_CKPT = Path(os.environ.get("VGGT_OMEGA_CKPT",
                                str(VGGT_DIR / "checkpoints" / "vggt_omega_1b_512.pt")))
if str(INGEST) not in sys.path:
    sys.path.insert(0, str(INGEST))
if str(VGGT_DIR) not in sys.path:
    sys.path.insert(0, str(VGGT_DIR))


def _grid_for(w: int, h: int, rows: int | None, cols: int | None):
    if rows and cols:
        return int(rows), int(cols)
    ar = w / float(h)
    # Order matters for ambiguous aspect ratios: a 2:1 sheet is more likely a
    # 2x4 view grid than two side-by-side images, and VGGT needs >=3 views to
    # say anything. Prefer the higher-view interpretation on a tie.
    candidates = [(2, 4, 2.0), (4, 2, 0.5), (1, 6, 6.0), (1, 5, 5.0), (1, 4, 4.0),
                  (1, 3, 3.0), (2, 3, 1.5), (3, 2, 1.5), (2, 2, 1.0),
                  (1, 2, 2.0), (2, 1, 0.5), (6, 1, 1.0 / 6), (4, 1, 0.25), (5, 1, 0.2)]
    best = min(candidates, key=lambda c: abs(ar - c[2]) + (0.3 if c[2] == 1.5 and ar < 1 else 0.0))
    return best[0], best[1]


def split_sheet(bgr, rows: int, cols: int):
    h, w = bgr.shape[:2]
    ch, cw = h // rows, w // cols
    panels = []
    for r in range(rows):
        for c in range(cols):
            panels.append(bgr[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw].copy())
    return panels


def _chamfer(a, b, max_pts=4000):
    a = a[:: max(1, len(a) // max_pts)]
    b = b[:: max(1, len(b) // max_pts)]
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.median(d) * 1000), float(np.percentile(d, 90) * 1000)


def vggt_check(bgr, out: Path, rows: int | None = None, cols: int | None = None,
               progress=None) -> tuple[str, list[str], dict]:
    """Run VGGT-Ω over a multi-view sheet and report cross-view consistency (mm)."""
    import run_all as R

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    rows, cols = _grid_for(bgr.shape[1], bgr.shape[0], rows, cols)
    if rows * cols < 2:
        raise ValueError("need at least 2 panels for a multi-view check")
    grid_rows, grid_cols = int(rows), int(cols)
    panels = split_sheet(bgr, rows, cols)
    pdir = out / "panels"
    pdir.mkdir(exist_ok=True)
    paths = []
    for i, p in enumerate(panels):
        fp = pdir / f"panel_{i:02d}.png"
        cv2.imwrite(str(fp), p)
        paths.append(fp)
    if progress:
        progress("start", f"vggt: {rows}x{cols} grid, {len(paths)} views", 0.05)
    cv2.imwrite(str(out / "source_sheet.png"), bgr)

    # --- VGGT
    import torch
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    imgs = load_and_preprocess_images([str(p) for p in paths], image_resolution=512)
    vm = VGGTOmega().eval()
    vm.load_state_dict(torch.load(VGGT_CKPT, map_location="cpu"))
    vm = vm.cuda()
    with torch.no_grad():
        preds = vm(imgs.cuda())
    depth = preds["depth"][0, ..., 0].cpu().numpy().astype(np.float32)
    conf = preds["depth_conf"][0].cpu().numpy().astype(np.float32)
    ext, K = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
    ext = ext[0].cpu().numpy().astype(np.float32)
    K = K[0].cpu().numpy().astype(np.float32)
    del vm, imgs, preds
    R.free({})
    if progress:
        progress("mid", "vggt: cameras + depth done, masking persons", 0.45)

    # --- Sapiens2 person masks (one model resident at a time)
    seg_model = R.load_dense("seg", "seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
                             "seg/sapiens2_0.4b_seg.safetensors")
    from sapiens.dense.visualizers import SegVisualizer
    seg_vis = SegVisualizer(class_palette_type="dome29", with_labels=False)
    clouds = {}
    for i, pth in enumerate(paths):
        bgr_p = cv2.imread(str(pth))
        Hp, Wp = bgr_p.shape[:2]
        labels, _ = R.seg_run(seg_model, seg_vis, bgr_p)
        person = cv2.resize((labels > 0).astype(np.uint8), (depth[i].shape[1], depth[i].shape[0]),
                            interpolation=cv2.INTER_NEAREST) > 0
        m = person & (conf[i] > np.percentile(conf[i], 25)) & np.isfinite(depth[i]) & (depth[i] > 0.05)
        ys, xs = np.nonzero(m)
        d = depth[i][ys, xs]
        Xc = np.stack([(xs - K[i, 0, 2]) / K[i, 0, 0] * d,
                       (ys - K[i, 1, 2]) / K[i, 1, 1] * d, d], 1)
        Rm, tt = ext[i][:3, :3], ext[i][:3, 3]
        clouds[i] = (Rm.T @ (Xc - tt[None]).T).T
    del seg_model, seg_vis
    R.free({})
    if progress:
        progress("mid", "vggt: consistency", 0.75)

    names = [f"view{i:02d}" for i in range(len(paths))]
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            med, p90 = _chamfer(clouds[i], clouds[j])
            pairs[f"{names[i]}__{names[j]}"] = {"median_mm": round(med, 1), "p90_mm": round(p90, 1)}
    vals = [v["median_mm"] for v in pairs.values()]
    summary_stats = {"n_pairs": len(vals),
                     "median_mm": round(float(np.median(vals)), 1) if vals else None,
                     "best_mm": round(float(np.min(vals)), 1) if vals else None,
                     "worst_mm": round(float(np.max(vals)), 1) if vals else None}

    # --- point cloud PLY + preview (colours by view)
    import trimesh
    palette = [(230, 50, 50), (245, 150, 30), (200, 220, 40), (60, 200, 90),
               (50, 120, 240), (150, 60, 230), (30, 200, 200), (240, 60, 200)]
    pts = np.concatenate([clouds[i] for i in range(len(names))])
    cols = np.concatenate([np.tile(np.array(palette[i % len(palette)], np.uint8), (len(clouds[i]), 1))
                           for i in range(len(names))])
    trimesh.PointCloud(pts, colors=cols).export(out / "vggt_cloud.ply")
    i0 = 0
    R0, t0m, K0 = ext[i0][:3, :3], ext[i0][:3, 3], K[i0]
    Xc = (R0 @ pts.T).T + t0m[None]
    z = np.maximum(Xc[:, 2], 1e-6)
    u = K0[0, 0] * Xc[:, 0] / z + K0[0, 2]
    v = K0[1, 1] * Xc[:, 1] / z + K0[1, 2]
    S = 900
    u = (u - u.min()) / max(1e-6, u.max() - u.min()) * (S - 1)
    v = (v - v.min()) / max(1e-6, v.max() - v.min()) * (S - 1)
    canvas = np.full((S, S, 3), 240, np.uint8)
    for k in np.argsort(-z):
        x, y = int(u[k]), int(v[k])
        if 0 <= x < S and 0 <= y < S:
            canvas[y, x] = cols[k][::-1]
    cv2.imwrite(str(out / "vggt_cloud.png"), canvas)

    report = {"grid": [grid_rows, grid_cols], "n_views": len(names),
              "extrinsics": {names[i]: {"t": ext[i][:3, 3].round(3).tolist()} for i in range(len(names))},
              "pairs": pairs, "summary": summary_stats,
              "seconds": round(time.time() - t0, 1)}
    (out / "vggt_report.json").write_text(
        json.dumps(report, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)),
        encoding="utf-8")

    files = ["source_sheet.png", "vggt_report.json", "vggt_cloud.ply", "vggt_cloud.png"] + \
            [f"panels/{p.name}" for p in paths]
    s = summary_stats
    summary = ("VGGT-Omega multi-view check: {r}x{c} grid, {n} views — cross-view median {m} mm "
               "(best {b}, worst {w}), cameras in one frame, point cloud exported.").format(
        r=grid_rows, c=grid_cols, n=len(names), m=s["median_mm"], b=s["best_mm"], w=s["worst_mm"])
    stats = {"grid": [grid_rows, grid_cols], "n_views": len(names), "summary": s,
             "pairs": pairs, "seconds": report["seconds"], "report_file": "vggt_report.json",
             "cloud_file": "vggt_cloud.ply", "preview_file": "vggt_cloud.png"}
    return summary, files, stats


def apose_export(out_dir: Path, fit_dir: Path, out: Path, progress=None) -> tuple[str, list[str], dict]:
    """Canonical A-pose mesh + 6-view turnaround sheet from a body-fit debug dir."""
    import run_all as R

    out.mkdir(parents=True, exist_ok=True)
    jpath = fit_dir / "identity_joint.json"
    if not jpath.is_file():
        raise FileNotFoundError("identity_joint.json missing in the fit dir")
    if progress:
        progress("start", "apose: rebuilding mesh", 0.05)
    j = json.loads(jpath.read_text(encoding="utf-8"))
    model3d, est, faces = R.load_3db()
    beta = np.asarray(j["refined_shape_params"], np.float64)
    params = np.asarray(j["refined_mhr_model_params"], np.float64).copy()
    params[:6] = 0.0
    params[6:136] = 0.0
    vroot, _ = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params,
                                    expr=np.zeros(72))
    verts = vroot - vroot.mean(0)
    del model3d
    R.free({})

    import trimesh
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(out / "mesh_apose.ply")
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(out / "mesh_apose.obj")

    TILE = 420
    yaws = [(-90.0, "-90"), (-45.0, "-45"), (0.0, "0"), (45.0, "+45"), (90.0, "+90"), (180.0, "180")]
    center = verts.mean(0)
    height = float(np.ptp(verts[:, 1]))
    panels = []
    for yaw, lab in yaws:
        th = np.deg2rad(yaw)
        c, s = np.cos(th), np.sin(th)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        p = (verts - center[None]) @ rot.T
        f2 = 1.1 * TILE
        dist = f2 * height / (0.86 * TILE)
        world = p + np.array([0.0, 0.0, dist])
        mesh, mat = R._make_mesh(world, faces, colour=R.MESH_COLOUR)
        img_hi, mask_hi = R._render_once(TILE, TILE, f2, mesh, mat)
        img, _ = R._finish(img_hi, mask_hi, TILE, TILE)
        im = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(im, lab, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(im, lab, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
        panels.append(im)
    sheet = np.hstack(panels)
    cv2.imwrite(str(out / "turnaround_apose.jpg"), sheet)
    grid23 = np.vstack([np.hstack(panels[0:3]), np.hstack(panels[3:6])])
    cv2.imwrite(str(out / "turnaround_apose_2x3.jpg"), grid23)
    (out / "pose_canonical.json").write_text(json.dumps({
        "pose": "A-pose (body euler zeroed)", "yaws_deg": [y for y, _ in yaws],
        "identity_shape_params": beta.round(5).tolist(),
        "scale_state": params[136:204].round(5).tolist(), "expression": "zero (neutral)"}, indent=2),
        encoding="utf-8")
    if progress:
        progress("mid", "apose: done", 0.9)
    files = ["mesh_apose.ply", "mesh_apose.obj", "turnaround_apose.jpg",
             "turnaround_apose_2x3.jpg", "pose_canonical.json"]
    return ("Canonical A-pose export: 18,439-vert MHR mesh (identity + scale, pose zeroed, neutral "
            "expression), 6-view turnaround sheet."), files, {
        "verts": int(len(verts)), "sheet": "turnaround_apose_2x3.jpg",
        "sheet_wide": "turnaround_apose.jpg",
        "mesh_file": "mesh_apose.ply", "obj_file": "mesh_apose.obj"}
