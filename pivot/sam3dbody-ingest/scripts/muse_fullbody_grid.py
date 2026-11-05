"""Emiru: 2x2 grid of square full-body A-pose renders (+45, -45, 0, 180) -> Muse photoreal turnaround."""
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import run_all as R

STEM = sys.argv[1] if len(sys.argv) > 1 else "Emiru"
JOB = BASE / "outputs" / STEM
PHOTO = next((BASE / "input-images").rglob(f"{STEM}.*"))
OUT = JOB / "threedb"
TILE = 600
YAWS = [45.0, -45.0, 0.0, 180.0]      # row-major 2x2

spec = importlib.util.spec_from_file_location("meta_image", REPO / "pipeline" / "avatar" / "meta_image.py")
mi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mi)

PROMPT = (
    "Image 1 is a flat-shaded 2x2 GRID of four FULL-BODY views of a person standing in A-pose "
    "(arms slightly away from the sides), in this row-major order: top-left = +45 degrees, "
    "top-right = -45 degrees, bottom-left = 0 (front), bottom-right = 180 (back). Image 2 is the "
    "reference photo of the person. Turn image 1 into photorealistic views of the SAME PERSON as "
    "image 2: same identity and face, same body shape and proportions, same skin tone, same hairstyle "
    "and hair colour, wearing the same clothing and shoes as image 2. Keep the SAME 2x2 grid, the same "
    "panel order and boundaries, the same A-pose, the same full-length framing (top of head to soles of "
    "the feet inside every panel), the same scale and camera angles in each panel. FLAT, EVEN, SHADOWLESS "
    "LIGHTING, plain light-grey seamless background, sharp focus, photographic, no text, no labels, "
    "no numbers, no borders, no watermark."
)
mi.PROMPTS["head-render"] = PROMPT
mi.REFERENCE_LABELS[0] = ("Image 1 (flat-shaded 2x2 full-body A-pose grid: +45, -45, 0, 180; preserve its "
                          "layout, panel order, pose, framing and scale):")
mi.REFERENCE_LABELS[1] = "Image 2 (the real person: identity, hair, clothing and body shape reference):"

# render the 2x2 guide from the canonical A-pose mesh
model3d, est, faces = R.load_3db()
j = json.loads((OUT / "identity_joint.json").read_text(encoding="utf-8"))
beta = np.asarray(j["refined_shape_params"])
params = np.asarray(j["refined_mhr_model_params"]).copy()
params[:6] = 0.0
params[6:136] = 0.0
vroot, _ = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params, expr=np.zeros(72))
verts = vroot - vroot.mean(0)
del model3d; R.free({})
center = verts.mean(0)
height = float(np.ptp(verts[:, 1]))
panels = []
for yaw in YAWS:
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
    panels.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
grid = np.full((2 * TILE, 2 * TILE, 3), 235, np.uint8)
for i, p in enumerate(panels):
    r, c = divmod(i, 2)
    grid[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = p
guide = OUT / "guide_fullbody_grid_2x2.png"
cv2.imwrite(str(guide), grid)
print("guide", grid.shape)

t0 = time.time()
rc = mi.main(["--mode", "head-render", "--reference", str(guide), "--reference2", str(PHOTO),
              "--out", str(OUT / "raw_fullbody_grid"), "--size", "1024x1024", "--timeout", "300"])
sheet = OUT / "raw_fullbody_grid" / "sheet.png"
if rc == 0 and sheet.exists():
    dst = OUT / "muse_fullbody_grid_2x2.png"
    shutil.copy2(sheet, dst)
    im = cv2.imread(str(dst))
    h, w = im.shape[:2]
    quads = {"+45": im[:h // 2, :w // 2], "-45": im[:h // 2, w // 2:],
             "0": im[h // 2:, :w // 2], "180": im[h // 2:, w // 2:]}
    for k, v in quads.items():
        cv2.imwrite(str(OUT / f"muse_fullbody_{k.replace('+', 'p').replace('-', 'm')}.png"), v)
    (OUT / "muse_fullbody_grid.json").write_text(json.dumps({
        "image": STEM, "guide": str(guide), "photo": str(PHOTO), "size": [w, h],
        "seconds": round(time.time() - t0, 1), "yaws_rowmajor": YAWS, "prompt": PROMPT}, indent=2), encoding="utf-8")
    print("saved", dst, "in", round(time.time() - t0, 1), "s")
else:
    raise SystemExit(f"Muse failed rc={rc}")
