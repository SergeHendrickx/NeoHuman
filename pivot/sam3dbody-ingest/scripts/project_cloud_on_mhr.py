"""Project the metric VGGT cloud directly onto the MHR A-pose mesh (no ICP) - KDTree approximation."""
import json
import numpy as np
import trimesh
import cv2
from scipy.spatial import cKDTree
from pathlib import Path

OUT = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\outputs\Emiru\threedb")
mesh = trimesh.load(str(OUT / "mesh_apose.ply"), process=False)
cloud = trimesh.load(str(OUT / "vggt_fullbody_cloud_metric.ply"), process=False)
pts = np.asarray(cloud.vertices)
cols = np.asarray(cloud.colors)[:, :3] if cloud.colors is not None else None

N_S = 400000
sample, face_idx = trimesh.sample.sample_surface(mesh, N_S)
tree = cKDTree(sample)
d, idx = tree.query(pts)
closest = sample[idx]
fn = mesh.face_normals[face_idx[idx]]
signed = (np.einsum("ij,ij->i", pts - closest, fn)) * 1000.0

mv = np.asarray(mesh.vertices)
y = mv[:, 1]
lo, hi = np.percentile(y, [0.5, 99.5])
head_c = mv[y < lo + 0.16 * (hi - lo)].mean(0)
head = np.linalg.norm(pts - head_c, axis=1) < 0.13

report = {
    "n_points": int(len(pts)),
    "abs_distance_mm": {"median": round(float(np.median(d) * 1000), 2),
                        "p75": round(float(np.percentile(d * 1000, 75)), 2),
                        "p90": round(float(np.percentile(d * 1000, 90)), 2)},
    "signed_mm": {"median": round(float(np.median(signed)), 2),
                  "p10": round(float(np.percentile(signed, 10)), 2),
                  "p90": round(float(np.percentile(signed, 90)), 2)},
    "head_region": {"n": int(head.sum()),
                    "abs_median_mm": round(float(np.median(d[head] * 1000)), 2),
                    "signed_median_mm": round(float(np.median(signed[head])), 2)},
    "body_region": {"n": int((~head).sum()),
                    "abs_median_mm": round(float(np.median(d[~head] * 1000)), 2),
                    "signed_median_mm": round(float(np.median(signed[~head])), 2)},
}
print(json.dumps(report, indent=2))

trimesh.PointCloud(closest, colors=cols).export(OUT / "vggt_cloud_projected_on_mhr.ply")
np.savez_compressed(OUT / "cloud_projection.npz", cloud=pts, closest=closest, dist_mm=d * 1000,
                    signed_mm=signed)

S = 900
allp = np.vstack([pts, mv])
lo_y, hi_y = np.percentile(allp[:, 1], [1, 99])
cx = allp[:, 0].mean()
side = 1.18 * (hi_y - lo_y)
sc = S / side
x0, y0 = cx - side / 2, (lo_y + hi_y) / 2 - side / 2
canvas = np.full((S, S, 3), 240, np.uint8)
u = ((mv[:, 0] - x0) * sc).astype(int); v = ((mv[:, 1] - y0) * sc).astype(int)
ok = (u >= 0) & (u < S) & (v >= 0) & (v < S)
canvas[v[ok], u[ok]] = (215, 215, 215)
vmax = 25.0
cmap = cv2.applyColorMap(np.clip((signed / vmax * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
u = ((pts[:, 0] - x0) * sc).astype(int); v = ((pts[:, 1] - y0) * sc).astype(int)
ok = (u >= 0) & (u < S) & (v >= 0) & (v < S)
canvas[v[ok], u[ok]] = cmap[ok]
cv2.putText(canvas, "signed offset (mm): blue=-25  red=+25", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)
cv2.imwrite(str(OUT / "cloud_projection_offsets.jpg"), canvas)
(OUT / "cloud_projection_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("wrote", OUT / "vggt_cloud_projected_on_mhr.ply", OUT / "cloud_projection_offsets.jpg")
