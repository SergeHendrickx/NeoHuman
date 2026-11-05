"""Visual: metric VGGT cloud vs MHR A-pose mesh (front ortho), from the exported PLYs."""
import numpy as np
import cv2
import trimesh
from pathlib import Path

OUT = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\outputs\Emiru\threedb")
cloud = trimesh.load(str(OUT / "vggt_fullbody_cloud_metric.ply"), process=False)
pts = np.asarray(cloud.vertices)
cols = np.asarray(cloud.colors) if cloud.colors is not None and len(cloud.colors) else np.full((len(pts), 3), 200, np.uint8)
mesh = trimesh.load(str(OUT / "mesh_apose.ply"), process=False)
fv = np.asarray(mesh.vertices); ff = np.asarray(mesh.faces)
sample, _ = trimesh.sample.sample_surface(mesh, 20000)

S = 900
allp = np.vstack([pts, sample])
lo, hi = np.percentile(allp[:, 1], [1, 99])
cx = allp[:, 0].mean()
side = 1.18 * (hi - lo)
sc = S / side
x0, y0 = cx - side / 2, (lo + hi) / 2 - side / 2
canvas = np.full((S, S, 3), 240, np.uint8)


def draw(P, colour, rad=1):
    u = ((P[:, 0] - x0) * sc).astype(int)
    v = ((P[:, 1] - y0) * sc).astype(int)
    ok = (u >= 0) & (u < S) & (v >= 0) & (v < S)
    canvas[v[ok], u[ok]] = colour


draw(pts, (170, 90, 90))
draw(sample, (90, 160, 90))
# colour-code cloud by its own RGB at higher density
u = ((pts[:, 0] - x0) * sc).astype(int); v = ((pts[:, 1] - y0) * sc).astype(int)
ok = (u >= 0) & (u < S) & (v >= 0) & (v < S)
canvas[v[ok], u[ok]] = cols[ok][:, :3][:, ::-1]
cv2.imwrite(str(OUT / "cloud_vs_mesh_metric.jpg"), canvas)
print("wrote", OUT / "cloud_vs_mesh_metric.jpg", "cloud pts", len(pts))
