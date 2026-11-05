"""Verify the baked texture: build variants (v-flip / channel order) and render the mesh from the texture."""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "outputs" / "_global" / "meta-image-test" / "ortho-neutral"))

import render_ortho as RO
import run_all as R

STEM = "teenage-woman-long-wavy-white-gray__01971-2947889296"
OUT = BASE / "outputs" / STEM / "threedb"
d = np.load(OUT / "texture_bake_data.npz")
vcol = d["vcol"]; texcoords = d["texcoords"]; tfaces = d["texcoord_faces"]; faces = d["faces"]; canon = d["canon"]
V = len(vcol)

# per-vertex UV (average of corners); texcoords indexing != vertex indexing
uv_sum = np.zeros((V, 2)); uv_cnt = np.zeros(V)
for tri in range(len(tfaces)):
    for k in range(3):
        uv_sum[faces[tri, k]] += texcoords[tfaces[tri, k]]
        uv_cnt[faces[tri, k]] += 1
uvv = uv_sum / np.maximum(uv_cnt[:, None], 1)

TEX = 2048
tex_rgb = np.clip(vcol, 0, 255).astype(np.uint8)  # placeholder replaced by rasterised texture variants


def raster_uv(vcol_bgrish, flip):
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
    if flip:
        tex = tex[::-1]
    return np.clip(tex, 0, 255).astype(np.uint8)


def sample(tex, uv):
    x = np.clip((uv[:, 0] * (TEX - 1)).astype(int), 0, TEX - 1)
    y = np.clip(((1 - uv[:, 1]) if True else uv[:, 1]) * (TEX - 1), 0, TEX - 1).astype(int)
    return tex[y, x]


model3d, est, facesm = R.load_3db()
del model3d; R.free({})

variants = {}
for flip in (False, True):
    tex = raster_uv(vcol, flip)
    img = cv2.cvtColor(tex, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT / f"tex_variant_flip{int(flip)}.png"), img)
    col = sample(tex, uvv).astype(np.float32)
    for ch, colx in (("bgr", col), ("rgb", col[:, ::-1])):
        vc = colx / 255.0
        img_r, mask = R.render_rgb(canon, facesm, 1100.0, 512, 512, vcol=vc, ss=1.0)
        out = cv2.cvtColor(img_r, cv2.COLOR_RGB2BGR)
        cv2.putText(out, f"flip={int(flip)} {ch}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
        variants[(flip, ch)] = out

sep = np.full((512, 8, 3), 30, np.uint8)
grid = np.hstack([variants[(False, "bgr")], sep, variants[(False, "rgb")], sep,
                  variants[(True, "bgr")], sep, variants[(True, "rgb")]])
cv2.imwrite(str(OUT / "texture_variants_render.jpg"), grid)
print("wrote", OUT / "texture_variants_render.jpg")
