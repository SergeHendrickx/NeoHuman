"""Whole-body version of the pose-strip test: align the 3 fitted bodies, chamfer per region, RGB composite."""
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
import run_all as R

JOB = BASE / "outputs" / "Emiru"
OUT = JOB / "threedb" / "pose_strip_test"
SIZE = RO.SIZE
PANELS = ["front", "left_side", "back"]
COLS = {"front": (1.0, 0.05, 0.05), "left_side": (0.05, 1.0, 0.05), "back": (0.15, 0.25, 1.0)}
BODY_ROWS = np.arange(5, 70)


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


model3d, est, faces = R.load_3db()
del model3d; R.free({})
fits = {n: np.load(OUT / f"fit_{n}.npz") for n in PANELS}
ref_lm = fits["front"]["lm"][70:272]

aligned = {}
report = {}
for n in PANELS:
    verts = fits[n]["verts"]
    lm = fits[n]["lm"][70:272]
    if n == "front":
        s, Rm, t = 1.0, np.eye(3), np.zeros(3)
    else:
        s, Rm, t = procrustes(lm, ref_lm)
    aligned[n] = s * (verts @ Rm.T) + t
    report[n] = {"scale_to_front": round(float(s), 4)}

# per-region chamfer (front as reference cloud)
V0 = aligned["front"]
h = float(np.ptp(V0[:, 1]))
y = V0[:, 1] - V0[:, 1].min()
regions = {"head": y < 0.22 * h, "torso": (y >= 0.22 * h) & (y < 0.55 * h),
           "arms+legs": y >= 0.55 * h}

def chamfer(A, B):
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.median(d) * 1000), float(np.percentile(d, 90) * 1000)

report["chamfer_mm"] = {}
for n in ("left_side", "back"):
    Va = aligned[n]
    rep = {}
    for rname, sel in regions.items():
        A = V0[sel][::max(1, int(sel.sum()) // 2500)]
        B = Va[sel][::max(1, int(sel.sum()) // 2500)]
        med, p90 = chamfer(A, B)
        rep[rname] = {"median": round(med, 1), "p90": round(p90, 1)}
    # overall
    A = V0[::20]; B = Va[::20]
    med, p90 = chamfer(A, B)
    rep["all"] = {"median": round(med, 1), "p90": round(p90, 1)}
    report["chamfer_mm"][n] = rep
    print(n, json.dumps(rep), flush=True)

# full-body RGB composite in A-pose-ish front view
center = V0.mean(0)
height = float(np.ptp(V0[:, 1]))
side = 1.25 * height
scale = (SIZE * RO.SS) / side
x0 = center[0] - side / 2
y0 = center[1] - side / 2
renders = {}
for n in PANELS:
    clay = np.tile(np.array(COLS[n], np.float32), (len(aligned[n]), 1))
    rgb = RO.raster_ortho(aligned[n], faces, clay, scale, x0, y0)
    renders[n] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT / f"body_aligned_{n}.png"), renders[n])
comp = np.tile(np.array([238, 238, 238], np.uint8), (SIZE, SIZE, 1)).astype(np.float32)
for n in PANELS:
    im = renders[n].astype(np.float32)
    mask = (np.abs(im - 235).max(2) > 8)
    comp[mask] = 0.5 * comp[mask] + 0.5 * im[mask]
comp = np.clip(comp, 0, 255).astype(np.uint8)
cv2.imwrite(str(OUT / "body_rgb_alignment.png"), comp)

def label(im, text):
    im = im.copy()
    cv2.putText(im, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(im, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 1, cv2.LINE_AA)
    return im

th = 520
tops = []
for n in PANELS:
    p = cv2.imread(str(OUT / f"panel_{n}.png"))
    w = int(p.shape[1] * th / p.shape[0])
    tops.append(label(cv2.resize(p, (w, th), interpolation=cv2.INTER_AREA), n))
top = np.hstack(tops)
bottom = np.hstack([label(cv2.resize(renders[n], (SIZE, SIZE), interpolation=cv2.INTER_AREA), "aligned " + n)
                    for n in PANELS] +
                   [label(cv2.resize(cv2.imread(str(OUT / "body_rgb_alignment.png")), (SIZE, SIZE),
                                     interpolation=cv2.INTER_AREA), "RGB front=red side=green back=blue")])
W = max(top.shape[1], bottom.shape[1])
rows = []
for r in (top, bottom):
    pad = np.full((r.shape[0], W, 3), 30, np.uint8)
    pad[:, :r.shape[1]] = r
    rows.append(pad)
sheet = np.vstack([rows[0], np.full((10, W, 3), 30, np.uint8), rows[1]])
cv2.imwrite(str(OUT / "show_body_test.jpg"), sheet)
(OUT / "body_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("wrote", OUT / "show_body_test.jpg")
