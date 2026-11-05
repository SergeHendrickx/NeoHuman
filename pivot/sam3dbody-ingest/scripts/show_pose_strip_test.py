"""Visual for the Emiru pose-strip test: the 3 generated panels, the 3 fitted heads, and their RGB alignment."""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "outputs" / "_global" / "meta-image-test" / "ortho-neutral"))

import cv2
import numpy as np

import render_ortho as RO

JOB = BASE / "outputs" / "Emiru"
OUT = JOB / "threedb" / "pose_strip_test"
SIZE = RO.SIZE
PANELS = ["front", "left_side", "back"]
COLS = {"front": (1.0, 0.05, 0.05), "left_side": (0.05, 1.0, 0.05), "back": (0.15, 0.25, 1.0)}


def procrustes(A, B):
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


fits = {n: np.load(OUT / f"fit_{n}.npz") for n in PANELS}
# faces come from the MHR model; reuse faces saved inside any fit? not saved -> load 3db faces
sys.path.insert(0, str(BASE))
import run_all as R
model3d, est, faces = R.load_3db()
del model3d; R.free({})

ref_lm = fits["front"]["lm"][70:272]
ref_center = ref_lm.mean(0)
ref_radius = float(np.percentile(np.linalg.norm(ref_lm - ref_center[None], axis=1), 95))
side = 2.6 * ref_radius
scale = (SIZE * RO.SS) / side
x0 = ref_center[0] - side / 2
y0 = ref_center[1] - side / 2

aligned = {}
for n in PANELS:
    lm = fits[n]["lm"][70:272]
    verts = fits[n]["verts"]
    if n == "front":
        s, Rm, t = 1.0, np.eye(3), np.zeros(3)
    else:
        s, Rm, t = procrustes(lm, ref_lm)
    aligned[n] = s * (verts @ Rm.T) + t

renders = {}
for n in PANELS:
    v = aligned[n]
    clay = np.tile(np.array(COLS[n], np.float32), (len(v), 1))
    rgb = RO.raster_ortho(v, faces, clay, scale, x0, y0)
    renders[n] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT / f"aligned_{n}.png"), renders[n])

bg = np.array([238, 238, 238], np.uint8)
comp = np.tile(bg, (SIZE, SIZE, 1)).astype(np.float32)
order = ["front", "left_side", "back"]
for n in order:
    im = renders[n].astype(np.float32)
    mask = (np.abs(im - 235).max(2) > 8)
    comp[mask] = 0.5 * comp[mask] + 0.5 * im[mask]
comp = np.clip(comp, 0, 255).astype(np.uint8)
cv2.imwrite(str(OUT / "rgb_alignment.png"), comp)


def label(im, text):
    im = im.copy()
    cv2.putText(im, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(im, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 1, cv2.LINE_AA)
    return im


# top row: generated panels
th = 420
tops = []
for n in PANELS:
    p = cv2.imread(str(OUT / f"panel_{n}.png"))
    w = int(p.shape[1] * th / p.shape[0])
    tops.append(label(cv2.resize(p, (w, th), interpolation=cv2.INTER_AREA), n))
Wtop = sum(t.shape[1] for t in tops)
top = np.hstack(tops)

# bottom row: 3 aligned heads + RGB composite
cell = SIZE
bottom = []
for n in PANELS + ["rgb"]:
    im = cv2.imread(str(OUT / ("rgb_alignment.png" if n == "rgb" else f"aligned_{n}.png")))
    bottom.append(label(cv2.resize(im, (cell, cell), interpolation=cv2.INTER_AREA),
                        "aligned " + n if n != "rgb" else "RGB: front=red side=green back=blue"))
bottom = np.hstack(bottom)

W = max(top.shape[1], bottom.shape[1])
rows = []
for r in (top, bottom):
    pad = np.full((r.shape[0], W, 3), 30, np.uint8)
    pad[:, :r.shape[1]] = r
    rows.append(pad)
sheet = np.vstack([rows[0], np.full((10, W, 3), 30, np.uint8), rows[1]])
cv2.imwrite(str(OUT / "show_pose_strip_test.jpg"), sheet)
print("wrote", OUT / "show_pose_strip_test.jpg")
