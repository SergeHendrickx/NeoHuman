"""Emiru: two guide layouts sent to Meta Image - 2x3 face grid and 1x3 full-body pose strip."""
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
TILE_F = 512
FACE_YAWS = [0.0, -45.0, 45.0, -90.0, 90.0, 180.0]     # row-major 2 rows x 3 cols
BODY_YAWS = [0.0, -90.0, 180.0]                        # 1 row x 3 cols

spec = importlib.util.spec_from_file_location("meta_image", REPO / "pipeline" / "avatar" / "meta_image.py")
mi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mi)

FACE_PROMPT = (
    "Image 1 is a flat-shaded 2x3 GRID of six head-and-shoulders views of a BALD mannequin head, in this "
    "order: row 1 = front, 3/4 left (-45), 3/4 right (+45); row 2 = left profile (-90), right profile (+90), "
    "back (180). Image 2 is the real person. Turn image 1 into photorealistic views of the SAME PERSON as "
    "image 2: same identity and face, same skin tone and skin texture, same eyebrows, same hair and hair "
    "colour as image 2 in every panel. Keep the SAME 2x3 grid, the same panel order, boundaries, head pose "
    "per panel, head scale and framing (head and upper neck, centred in each panel). FLAT, EVEN, SHADOWLESS "
    "LIGHTING, plain light-grey background, sharp focus, photographic, no text, no labels, no numbers, "
    "no borders, no watermark."
)
BODY_PROMPT = (
    "Image 1 is a flat-shaded 1x3 ROW of three FULL-BODY views of a mannequin in A-pose, in this order: "
    "front, left profile, back. Image 2 is the real person. Turn image 1 into photorealistic views of the "
    "SAME PERSON as image 2: same identity and face, same body shape and proportions, same skin tone, same "
    "hairstyle and hair colour, same clothing as image 2. Keep the SAME 1x3 row, the same panel order and "
    "boundaries, the same A-pose (arms slightly away from the body), the same full-body framing (head to "
    "feet inside every panel), the same scale and camera angles. FLAT, EVEN, SHADOWLESS LIGHTING, plain "
    "light-grey background, sharp focus, photographic, no text, no labels, no numbers, no borders, "
    "no watermark."
)

model3d, est, faces = R.load_3db()
j = json.loads((OUT / "identity_joint.json").read_text(encoding="utf-8"))
beta = np.asarray(j["refined_shape_params"])
params = np.asarray(j["refined_mhr_model_params"]).copy()
params[:6] = 0.0
params[6:136] = 0.0
vroot, lm = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params, expr=np.zeros(72))
verts = vroot - vroot.mean(0)
face_lm = lm[70:272]
face_center = (face_lm.mean(0) - vroot.mean(0))
face_radius = float(np.percentile(np.linalg.norm(face_lm - face_lm.mean(0), axis=1), 95))
body_center = verts.mean(0)
body_h = float(np.ptp(verts[:, 1]))


def render_view(yaw, tile_w, tile_h, head: bool):
    th = np.deg2rad(yaw)
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    p = (verts - (face_center if head else body_center)[None]) @ rot.T
    f2 = 1.2 * tile_w
    if head:
        dist = f2 * (2.0 * face_radius) / (0.80 * tile_w)
    else:
        dist = f2 * tile_h / (0.92 * tile_h) * (2.0 * body_h) / (0.86 * tile_h) * (tile_h / tile_h)
        # fit the full body height into the tile
        f2 = 1.1 * tile_h
        dist = f2 * body_h / (0.90 * tile_h)
    world = p + np.array([0.0, 0.0, dist])
    mesh, mat = R._make_mesh(world, faces, colour=R.MESH_COLOUR)
    img_hi, mask_hi = R._render_once(tile_w, tile_h, f2, mesh, mat)
    img, _ = R._finish(img_hi, mask_hi, tile_w, tile_h)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# 2x3 face grid (3 cols x 2 rows), panels square
face_panels = [render_view(y, TILE_F, TILE_F, True) for y in FACE_YAWS]
grid = np.full((2 * TILE_F, 3 * TILE_F, 3), 235, np.uint8)
for i, p in enumerate(face_panels):
    r, c = divmod(i, 3)
    grid[r * TILE_F:(r + 1) * TILE_F, c * TILE_F:(c + 1) * TILE_F] = p
cv2.imwrite(str(OUT / "guide_face_grid_2x3.png"), grid)

# 1x3 body strip: panels 512 x 1024 -> 1536 x 1024
BW, BH = 512, 1024
body_panels = [render_view(y, BW, BH, False) for y in BODY_YAWS]
strip = np.full((BH, 3 * BW, 3), 235, np.uint8)
for i, p in enumerate(body_panels):
    strip[:, i * BW:(i + 1) * BW] = p
cv2.imwrite(str(OUT / "guide_body_strip_1x3.png"), strip)
del model3d; R.free({})
print("guides written", grid.shape, strip.shape)

rows = []
for tag, guide, prompt, outname in (
        ("face_grid_2x3", OUT / "guide_face_grid_2x3.png", FACE_PROMPT, "muse_face_grid_2x3.png"),
        ("body_strip_1x3", OUT / "guide_body_strip_1x3.png", BODY_PROMPT, "muse_body_strip_1x3.png")):
    mi.PROMPTS["head-render"] = prompt
    mi.REFERENCE_LABELS[0] = ("Image 1 (flat-shaded guide; preserve its layout, panel order, pose and "
                              "framing exactly):")
    mi.REFERENCE_LABELS[1] = "Image 2 (the real person: identity, hair, clothing reference):"
    t0 = time.time()
    rc = mi.main(["--mode", "head-render", "--reference", str(guide), "--reference2", str(PHOTO),
                  "--out", str(OUT / f"raw_{tag}"), "--size", "1536x1024", "--timeout", "300"])
    sheet = OUT / f"raw_{tag}" / "sheet.png"
    ok = rc == 0 and sheet.exists()
    if ok:
        shutil.copy2(sheet, OUT / outname)
    rows.append({"layout": tag, "status": "ok" if ok else f"rc={rc}", "seconds": round(time.time() - t0, 1)})
    print(json.dumps(rows[-1]), flush=True)
(OUT / "muse_layout_test.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
