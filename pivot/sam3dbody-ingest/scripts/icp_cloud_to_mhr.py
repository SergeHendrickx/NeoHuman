"""ICP (with scale) refinement of the VGGT point clouds onto the metric MHR A-pose mesh."""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import numpy as np
import trimesh
from scipy.spatial import cKDTree

JOB = BASE / "outputs" / "Emiru"
OUT = JOB / "threedb"
ALIGN = json.loads((OUT / "cloud_to_mhr_alignment.json").read_text(encoding="utf-8"))
PANELS = ["+45", "-45", "0", "180"]
FILES = {"+45": "vggt_cloud_p45.ply", "-45": "vggt_cloud_m45.ply",
         "0": "vggt_cloud_0.ply", "180": "vggt_cloud_180.ply"}

mesh = trimesh.load(str(OUT / "mesh_apose.ply"), process=False)
sample, _ = trimesh.sample.sample_surface(mesh, 60000)
tree = cKDTree(sample)


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, Sig, Vt = np.linalg.svd(C)
    S_fix = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_fix[2, 2] = -1
    Rm = U @ S_fix @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = float(np.trace(np.diag(Sig) @ S_fix) / var_s)
    t = mu_d - s * (Rm @ mu_s)
    return s, Rm, t


# initial global transform (keypoint Umeyama from the best view)
s0 = float(ALIGN["alignment"]["scale_to_metres"])
t0 = np.asarray(ALIGN["alignment"]["translation_m"], np.float64)
# rotation: reconstruct from the stored per-view? not stored -> align with identity R (front view R~I)
R0 = np.eye(3)


def residual(pts):
    d, _ = tree.query(pts)
    return float(np.median(d) * 1000), float(np.percentile(d, 90) * 1000)


report = {}
aligned_all = {}
for name in PANELS:
    cloud = trimesh.load(str(OUT / FILES[name]), process=False)
    pts = np.asarray(cloud.vertices)
    cols = np.asarray(cloud.colors) if cloud.colors is not None else None
    s, Rm, t = s0, R0, t0
    P = pts.copy()
    for it in range(30):
        P = s * (pts @ Rm.T) + t
        d, idx = tree.query(P[:: max(1, len(P) // 30000)])
        sub_pts = (P[:: max(1, len(P) // 30000)])
        sub_tgt = sample[idx]
        keep = d < np.percentile(d, 75)
        ds, dR, dt = umeyama(sub_pts[keep], sub_tgt[keep])
        # compose incremental similarity
        s = ds * s
        Rm = dR @ Rm
        t = ds * (dR @ t) + dt
    P = s * (pts @ Rm.T) + t
    med, p90 = residual(P)
    report[name] = {"scale_to_m": round(float(s), 5), "residual_median_mm": round(med, 1),
                    "residual_p90_mm": round(p90, 1), "rotation": Rm.round(4).tolist(),
                    "translation_m": t.round(4).tolist()}
    print(name, json.dumps({k: report[name][k] for k in ("scale_to_m", "residual_median_mm", "residual_p90_mm")}))
    trimesh.PointCloud(P, colors=cols).export(OUT / f"vggt_cloud_{name.replace('+', 'p').replace('-', 'm')}_icp.ply")
    aligned_all[name] = (P, cols)

# fused cloud: merge ICPed views
Pall = np.vstack([aligned_all[n][0] for n in PANELS])
Call = np.vstack([aligned_all[n][1] if aligned_all[n][1] is not None else np.zeros((0, 3), np.uint8) for n in PANELS])
trimesh.PointCloud(Pall, colors=Call).export(OUT / "vggt_fullbody_cloud_icp.ply")
report["fused"] = {"n_points": int(len(Pall))}
(OUT / "cloud_icp_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("fused points", len(Pall), "->", OUT / "vggt_fullbody_cloud_icp.ply")

# visual: fused ICP cloud vs mesh points
import cv2
S = 900
allp = np.vstack([Pall, sample])
lo, hi = np.percentile(allp[:, 1], [1, 99])
cx = allp[:, 0].mean()
side = 1.18 * (hi - lo)
sc = S / side
x0, y0 = cx - side / 2, (lo + hi) / 2 - side / 2
canvas = np.full((S, S, 3), 240, np.uint8)
for i, p in enumerate(Pall):
    u = int((p[0] - x0) * sc); v = int((p[1] - y0) * sc)
    if 0 <= u < S and 0 <= v < S:
        canvas[v, u] = (180, 60, 60)
for p in sample[::3]:
    u = int((p[0] - x0) * sc); v = int((p[1] - y0) * sc)
    if 0 <= u < S and 0 <= v < S:
        canvas[v, u] = (60, 170, 60)
cv2.imwrite(str(OUT / "cloud_icp_vs_mesh.jpg"), canvas)
print("wrote", OUT / "cloud_icp_vs_mesh.jpg")
