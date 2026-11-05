"""ICP (similarity) refinement of the already-metric VGGT cloud onto the MHR A-pose mesh."""
import json
import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from pathlib import Path

OUT = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\outputs\Emiru\threedb")
mesh = trimesh.load(str(OUT / "mesh_apose.ply"), process=False)
sample, _ = trimesh.sample.sample_surface(mesh, 80000)
tree = cKDTree(sample)

cloud = trimesh.load(str(OUT / "vggt_fullbody_cloud_metric.ply"), process=False)
pts = np.asarray(cloud.vertices)
cols = np.asarray(cloud.colors) if cloud.colors is not None else None
print("cloud", len(pts), "mesh samples", len(sample))


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, Sig, Vt = np.linalg.svd(C)
    fix = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        fix[2, 2] = -1
    Rm = U @ fix @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = float(np.trace(np.diag(Sig) @ fix) / var_s)
    t = mu_d - s * (Rm @ mu_s)
    return s, Rm, t


def medres(P):
    d, _ = tree.query(P[:: max(1, len(P) // 40000)])
    return float(np.median(d) * 1000), float(np.percentile(d, 75) * 1000)


print("init residual mm", [round(x, 2) for x in medres(pts)])
s, Rm, t = 1.0, np.eye(3), np.zeros(3)
for it in range(40):
    P = s * (pts @ Rm.T) + t
    step = max(1, len(P) // 40000)
    Pp = P[::step]
    d, idx = tree.query(Pp)
    keep = d < np.percentile(d, 70)
    ds, dR, dt = umeyama(Pp[keep], sample[idx][keep])
    s = ds * s
    Rm = dR @ Rm
    t = ds * (dR @ t) + dt
    if it % 10 == 0:
        print(f"it {it:2d} scale {s:.4f} residual mm {[round(x, 2) for x in medres(s * (pts @ Rm.T) + t)]}")
P = s * (pts @ Rm.T) + t
med, p75 = medres(P)
print("final", {"scale": round(s, 4), "median_mm": round(med, 2), "p75_mm": round(p75, 2)})

trimesh.PointCloud(P, colors=cols).export(OUT / "vggt_fullbody_cloud_icp.ply")
report = {"n_points": int(len(P)), "scale_correction": round(float(s), 5),
          "residual_median_mm": round(med, 2), "residual_p75_mm": round(p75, 2),
          "rotation": Rm.round(5).tolist(), "translation_m": t.round(5).tolist()}
(OUT / "icp_cloud_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

# visual: ICP cloud (red) vs mesh surface samples (green)
S = 900
allp = np.vstack([P, sample])
lo, hi = np.percentile(allp[:, 1], [1, 99]); cx = allp[:, 0].mean()
side = 1.18 * (hi - lo); sc = S / side
x0, y0 = cx - side / 2, (lo + hi) / 2 - side / 2
canvas = np.full((S, S, 3), 240, np.uint8)
u = ((P[:, 0] - x0) * sc).astype(int); v = ((P[:, 1] - y0) * sc).astype(int)
ok = (u >= 0) & (u < S) & (v >= 0) & (v < S)
canvas[v[ok], u[ok]] = (170, 70, 70)
u = ((sample[:, 0] - x0) * sc).astype(int); v = ((sample[:, 1] - y0) * sc).astype(int)
ok = (u >= 0) & (u < S) & (v >= 0) & (v < S)
canvas[v[ok], u[ok]] = (70, 170, 70)
cv2.imwrite(str(OUT / "cloud_icp_vs_mesh.jpg"), canvas)
print("wrote", OUT / "vggt_fullbody_cloud_icp.ply", OUT / "cloud_icp_vs_mesh.jpg")
