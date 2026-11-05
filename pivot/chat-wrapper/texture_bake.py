"""Texture bake for the flow viewer: project the source photo onto the fitted MHR mesh,
fill occluded/interior vertices, bake an MHR-UV texture, export a textured GLB + preview.
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


def _zbuffer(verts, faces, f, W, H):
    z = verts[:, 2]
    ok = z > 1e-6
    u = f * verts[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = f * verts[:, 1] / np.maximum(z, 1e-6) + H / 2
    zb = np.full((H, W), np.inf, np.float32)
    for tri in faces:
        if not ok[tri].all():
            continue
        ax, ay, az = u[tri[0]], v[tri[0]], z[tri[0]]
        bx, by, bz = u[tri[1]], v[tri[1]], z[tri[1]]
        cx, cy, cz = u[tri[2]], v[tri[2]], z[tri[2]]
        x1 = max(0, int(np.floor(min(ax, bx, cx)))); x2 = min(W - 1, int(np.ceil(max(ax, bx, cx))))
        y1 = max(0, int(np.floor(min(ay, by, cy)))); y2 = min(H - 1, int(np.ceil(max(ay, by, cy))))
        if x2 < x1 or y2 < y1:
            continue
        det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(det) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x1, x2 + 1) + 0.5, np.arange(y1, y2 + 1) + 0.5)
        w1 = ((gx - ax) * (cy - ay) - (cx - ax) * (gy - ay)) / det
        w2 = ((bx - ax) * (gy - ay) - (gx - ax) * (by - ay)) / det
        w0 = 1.0 - w1 - w2
        ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not ins.any():
            continue
        rr, cc = np.nonzero(ins)
        zz = (w0[ins] * az + w1[ins] * bz + w2[ins] * cz).astype(np.float32)
        cur = zb[rr + y1, cc + x1]
        upd = zz < cur
        zb[rr[upd] + y1, cc[upd] + x1] = zz[upd]
    return zb


def texture_bake(mesh_path: Path, bgr, out: Path, progress=None) -> tuple[str, list[str], dict]:
    """Project bgr through the fit camera onto the fitted MHR mesh and bake textures."""
    import trimesh
    import run_all as R

    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    fit_dir = mesh_path.parent
    j = json.loads((fit_dir / "identity_joint.json").read_text(encoding="utf-8"))
    fs = json.loads((fit_dir / "fit_stats.json").read_text(encoding="utf-8"))
    W, H = (int(v) for v in fs["size_wh"])
    f = float(fs["focal_px"])
    beta = np.asarray(j["refined_shape_params"], np.float64)
    params = np.asarray(j["refined_mhr_model_params"], np.float64)
    expr = np.asarray(j.get("refined_expr_params", np.zeros(72)), np.float64).reshape(-1)
    cam = np.asarray(j["refined_cam_t"], np.float64).reshape(-1)
    if progress:
        progress("start", "texture: rebuilding fitted mesh", 0.06)

    model3d, est, faces = R.load_3db()
    mesh_uv = model3d.head_pose.mhr.character_torch.mesh
    texcoords = mesh_uv.texcoords.detach().cpu().numpy().astype(np.float32)
    tfaces = mesh_uv.texcoord_faces.detach().cpu().numpy().astype(np.int32)
    vroot, _ = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta,
                                    model_params=params, expr=expr)
    del model3d
    R.free({})
    verts = vroot + cam[None]
    V = len(verts)

    if progress:
        progress("mid", "texture: projecting photo ({v} verts)".format(v=V), 0.35)
    z = verts[:, 2]
    u = f * verts[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = f * verts[:, 1] / np.maximum(z, 1e-6) + H / 2
    zb = _zbuffer(verts, faces, f, W, H)
    ui = np.clip(u.astype(int), 0, W - 1)
    vi = np.clip(v.astype(int), 0, H - 1)
    inside = (u > 1) & (u < W - 2) & (v > 1) & (v < H - 2)
    visible = inside & (np.abs(z - zb[vi, ui]) < 0.01)

    # normals: front-facing only
    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    normals = tri.vertex_normals
    view_dir = -verts / np.maximum(np.linalg.norm(verts, axis=1, keepdims=True), 1e-9)
    facing = np.einsum("ij,ij->i", normals, view_dir) > 0.05
    sel = np.where(visible & facing)[0]

    img = bgr.astype(np.float32)
    xs, ys = u[sel], v[sel]
    x0 = np.clip(np.floor(xs).astype(int), 0, W - 1); x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    wx = (xs - x0)[:, None]; wy = (ys - y0)[:, None]
    cols = (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x1] * wx * (1 - wy)
            + img[y1, x0] * (1 - wx) * wy + img[y1, x1] * wx * wy)
    cols = cols[:, ::-1]                      # BGR (cv2) -> RGB for vertex/texture colours
    vcol = np.zeros((V, 3), np.float32)
    vcol[sel] = cols

    # fill unseen vertices with the nearest visible colour (no black holes)
    if len(sel) < V:
        from scipy.spatial import cKDTree
        tree = cKDTree(verts[sel])
        missing = np.setdiff1d(np.arange(V), sel)
        dist, idx = tree.query(verts[missing], k=1)
        vcol[missing] = vcol[sel][idx]

    if progress:
        progress("mid", "texture: baking UV texture", 0.65)
    # UV bake from per-vertex colours
    TEX = 1024
    tex = np.zeros((TEX, TEX, 3), np.float32)
    uvp = np.clip(texcoords * (TEX - 1), 0, TEX - 1)
    for fi in range(len(tfaces)):
        tri_uv = uvp[tfaces[fi]]
        x1 = max(0, int(np.floor(tri_uv[:, 0].min()))); x2 = min(TEX - 1, int(np.ceil(tri_uv[:, 0].max())))
        y1 = max(0, int(np.floor(tri_uv[:, 1].min()))); y2 = min(TEX - 1, int(np.ceil(tri_uv[:, 1].max())))
        if x2 < x1 or y2 < y1:
            continue
        det = ((tri_uv[1, 0] - tri_uv[0, 0]) * (tri_uv[2, 1] - tri_uv[0, 1])
               - (tri_uv[2, 0] - tri_uv[0, 0]) * (tri_uv[1, 1] - tri_uv[0, 1]))
        if abs(det) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x1, x2 + 1) + 0.5, np.arange(y1, y2 + 1) + 0.5)
        w1 = ((gx - tri_uv[0, 0]) * (tri_uv[2, 1] - tri_uv[0, 1]) - (tri_uv[2, 0] - tri_uv[0, 0]) * (gy - tri_uv[0, 1])) / det
        w2 = ((tri_uv[1, 0] - tri_uv[0, 0]) * (gy - tri_uv[0, 1]) - (gx - tri_uv[0, 0]) * (tri_uv[1, 1] - tri_uv[0, 1])) / det
        w0 = 1 - w1 - w2
        ins = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not ins.any():
            continue
        fv = faces[fi]
        c = w0[ins, None] * vcol[fv[0]][None] + w1[ins, None] * vcol[fv[1]][None] + w2[ins, None] * vcol[fv[2]][None]
        rr, cc = np.nonzero(ins)
        tex[rr + y1, cc + x1] = c
    tex_img = np.clip(tex[::-1], 0, 255).astype(np.uint8)
    cv2.imwrite(str(out / "texture_uv_1024.png"), cv2.cvtColor(tex_img, cv2.COLOR_RGB2BGR))

    if progress:
        progress("mid", "texture: exporting GLB + preview", 0.85)
    # textured GLB (vertex colours; GLTFLoader renders COLOR_0)
    rgba = np.concatenate([np.clip(vcol, 0, 255).astype(np.uint8),
                           np.full((V, 1), 255, np.uint8)], axis=1)
    tm = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba, process=False)
    tm.export(out / "mesh_textured.glb")

    # textured 6-view turnaround - what VGGT gets to match (texture beats clay)
    if progress:
        progress("mid", "texture: rendering textured turnaround", 0.8)
    TILE = 340
    yaws = [(-90.0, "-90"), (-45.0, "-45"), (0.0, "0"), (45.0, "+45"), (90.0, "+90"), (180.0, "180")]
    center = verts.mean(0)
    height = float(np.ptp(verts[:, 1]))
    vcol_norm = np.clip(vcol, 0, 255).astype(np.float32) / 255.0
    panels = []
    for yaw, lab in yaws:
        th = np.deg2rad(yaw)
        c, s = np.cos(th), np.sin(th)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        p3 = (verts - center[None]) @ rot.T
        f2 = 1.1 * TILE
        dist = f2 * height / (0.86 * TILE)
        world = p3 + np.array([0.0, 0.0, dist])
        img_r, _ = R.render_rgb(world, faces, f2, TILE, TILE, vcol=vcol_norm, ss=1.0)
        im = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)
        cv2.putText(im, lab, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 15, 15), 3, cv2.LINE_AA)
        cv2.putText(im, lab, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
        panels.append(im)
    sheet = np.hstack(panels)
    cv2.imwrite(str(out / "textured_turnaround.jpg"), sheet)
    # 2x3 grid (3:2) - the layout Muse edits and VGGT auto-detects best
    rows = [np.hstack(panels[0:3]), np.hstack(panels[3:6])]
    grid23 = np.vstack(rows)
    cv2.imwrite(str(out / "textured_turnaround_2x3.jpg"), grid23)

    # preview render with the baked colours
    try:
        img_r, _ = R.render_rgb(verts, faces, 1100.0, 640, 640,
                                vcol=np.clip(vcol, 0, 255).astype(np.float32) / 255.0, ss=1.0)
        cv2.imwrite(str(out / "textured_preview.jpg"), cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
        preview = "textured_preview.jpg"
    except Exception:
        preview = None
    files = ["mesh_textured.glb", "texture_uv_1024.png", "textured_turnaround.jpg",
             "textured_turnaround_2x3.jpg"] + ([preview] if preview else [])
    coverage = float(len(sel)) / V
    stats = {"verts": V, "focal_px": round(f, 1), "size_wh": [W, H],
             "projected_coverage": round(coverage, 3), "seconds": round(time.time() - t0, 1),
             "glb": "mesh_textured.glb", "texture": "texture_uv_1024.png",
             "preview": preview, "sheet": "textured_turnaround_2x3.jpg",
             "sheet_wide": "textured_turnaround.jpg"}
    summary = ("Texture bake: photo projected through the fitted camera onto {v:,} MHR verts "
               "({p:.0%} directly visible, rest filled), UV texture 1024 baked, textured GLB exported.").format(
        v=V, p=coverage)
    return summary, files, stats
