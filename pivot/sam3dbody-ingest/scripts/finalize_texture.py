"""Finalise the baked face texture (correct v-orientation + colour order) and render a preview."""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import run_all as R

STEM = "teenage-woman-long-wavy-white-gray__01971-2947889296"
OUT = BASE / "outputs" / STEM / "threedb"
d = np.load(OUT / "texture_bake_data.npz")
vcol = d["vcol"]; texcoords = d["texcoords"]; tfaces = d["texcoord_faces"]; faces = d["faces"]; canon = d["canon"]
V = len(vcol)
TEX = 2048

uv_sum = np.zeros((V, 2)); uv_cnt = np.zeros(V)
for tri in range(len(tfaces)):
    for k in range(3):
        uv_sum[faces[tri, k]] += texcoords[tfaces[tri, k]]
        uv_cnt[faces[tri, k]] += 1
uvv = uv_sum / np.maximum(uv_cnt[:, None], 1)

tex = np.zeros((TEX, TEX, 3), np.float32)
uvp = np.clip(texcoords * (TEX - 1), 0, TEX - 1)
for fi in range(len(tfaces)):
    tri = tfaces[fi]
    vt = uvp[tri]
    x1 = max(0, int(np.floor(vt[:, 0].min()))); x2 = min(TEX - 1, int(np.ceil(vt[:, 0].max())))
    y1 = max(0, int(np.floor(vt[:, 1].min()))); y2 = min(TEX - 1, int(np.ceil(vt[:, 1].max())))
    if x2 < x1 or y2 < y1:
        continue
    det = (vt[1, 0] - vt[0, 0]) * (vt[2, 1] - vt[0, 1]) - (vt[2, 0] - vt[0, 0]) * (vt[1, 1] - vt[0, 1])
    if abs(det) < 1e-9:
        continue
    gx, gy = np.meshgrid(np.arange(x1, x2 + 1) + 0.5, np.arange(y1, y2 + 1) + 0.5)
    w1 = ((gx - vt[0, 0]) * (vt[2, 1] - vt[0, 1]) - (vt[2, 0] - vt[0, 0]) * (gy - vt[0, 1])) / det
    w2 = ((vt[1, 0] - vt[0, 0]) * (gy - vt[0, 1]) - (gx - vt[0, 0]) * (vt[1, 1] - vt[0, 1])) / det
    w0 = 1 - w1 - w2
    ins = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
    if not ins.any():
        continue
    fv = faces[fi]
    cols = w0[ins, None] * vcol[fv[0]][None] + w1[ins, None] * vcol[fv[1]][None] + w2[ins, None] * vcol[fv[2]][None]
    rr, cc = np.nonzero(ins)
    tex[rr + y1, cc + x1] = cols

tex = np.clip(tex[::-1], 0, 255).astype(np.uint8)          # v-up orientation
cv2.imwrite(str(OUT / "mhr_face_texture_2048.png"), tex)
print("wrote", OUT / "mhr_face_texture_2048.png")

# preview: texture sampled with standard OBJ convention (row 0 = v=1)
def sample(uv):
    x = np.clip((uv[:, 0] * (TEX - 1)).astype(int), 0, TEX - 1)
    y = np.clip(((1 - uv[:, 1]) * (TEX - 1)).astype(int), 0, TEX - 1)
    return tex[y, x]

model3d, est, facesm = R.load_3db()
del model3d; R.free({})
vc = sample(uvv).astype(np.float32)[:, ::-1] / 255.0
img_r, _ = R.render_rgb(canon, facesm, 1100.0, 640, 640, vcol=vc, ss=1.0)
cv2.imwrite(str(OUT / "mhr_face_textured_preview.jpg"), cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR))
print("wrote", OUT / "mhr_face_textured_preview.jpg")
