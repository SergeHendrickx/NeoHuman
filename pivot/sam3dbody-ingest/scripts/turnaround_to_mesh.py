"""Full-body turnaround -> mesh test: Muse 4-panel turnaround + our fitted MHR body rendered from the same views."""
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import cv2
import numpy as np

import run_all as R

STEM = sys.argv[1] if len(sys.argv) > 1 else "Emiru"
JOB = BASE / "outputs" / STEM
photo = next((BASE / "input-images").rglob(f"{STEM}.*"))
OUT = JOB / "threedb"
TILE = 700
YAWS = [-90.0, 0.0, 90.0, 180.0]

PROMPT = (
    "Generate ONE image containing exactly FOUR equal panels side by side, separated by thin light-grey "
    "gutters, showing the SAME FULL-BODY PERSON from four views in this order: FRONT view, LEFT SIDE view, "
    "BACK view, RIGHT SIDE view. The person stands straight with arms slightly away from the body, the "
    "ENTIRE body from top of head to soles of the feet inside every panel. Same identity, face, skin tone, "
    "hairstyle and hair colour, and the same clothing in all four panels. Plain light-grey seamless "
    "background, FLAT EVEN SHADOWLESS LIGHTING, sharp focus, photographic, orthographic-like (no wide-angle "
    "distortion), no text, no labels, no borders, exactly four panels."
)

# --- 1. Muse turnaround (cached)
mp = OUT / "muse_turnaround.png"
if not mp.exists():
    spec = importlib.util.spec_from_file_location("meta_image", REPO / "pipeline" / "avatar" / "meta_image.py")
    mi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mi)
    mi.PROMPTS["full-body"] = PROMPT
    mi.REFERENCE_LABELS[0] = "Reference photo of the person (same identity; make the 4-panel full-body turnaround):"
    rc = mi.main(["--mode", "full-body", "--reference", str(photo), "--out", str(OUT / "muse_turnaround_raw"),
                  "--size", "1536x1024", "--timeout", "300"])
    sheet = OUT / "muse_turnaround_raw" / "sheet.png"
    if rc != 0 or not sheet.exists():
        raise SystemExit("Muse turnaround failed")
    cv2.imwrite(str(mp), cv2.imread(str(sheet)))
muse = cv2.imread(str(mp))
Hm, Wm = muse.shape[:2]
q = Wm // 4
muse_panels = [muse[:, i * q:(i + 1) * q] for i in range(4)]

# --- 2. our fitted MHR body rendered from the same four views
model3d, est, faces = R.load_3db()
j = json.loads((OUT / "identity_joint.json").read_text(encoding="utf-8"))
beta = np.asarray(j["refined_shape_params"]); params = np.asarray(j["refined_mhr_model_params"])
cam = np.asarray(j["refined_cam_t"]); expr = np.asarray(j.get("metrics", {}).get("expr_params", np.zeros(72)))
vroot, _ = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params, expr=expr)
verts = vroot + cam[None]
del model3d; R.free({})
center = verts.mean(0)


def render_full(yaw):
    th = np.deg2rad(yaw)
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    p = (verts - center[None]) @ rot.T
    height = float(np.ptp(p[:, 1]))
    f2 = 1.1 * TILE
    dist = f2 * height / (0.84 * TILE)
    world = p + np.array([0.0, 0.0, dist])
    mesh, mat = R._make_mesh(world, faces, colour=R.MESH_COLOUR)
    img_hi, mask_hi = R._render_once(TILE, TILE, f2, mesh, mat)
    img, _ = R._finish(img_hi, mask_hi, TILE, TILE)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


mesh_panels = []
for yaw in YAWS:
    im = render_full(yaw)
    label = {0.0: "0 front", 90.0: "+90 side", -90.0: "-90 side", 180.0: "180 back"}[yaw]
    cv2.putText(im, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.putText(im, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 1, cv2.LINE_AA)
    mesh_panels.append(im)

sep = np.full((TILE, 8, 3), 30, np.uint8)
row_m = np.hstack([p for pair in zip(mesh_panels, [sep] * 4) for p in pair][:-1])
# resize muse panels to the same tile height, keep aspect
mp_res = []
for p in muse_panels:
    h = TILE
    w = int(p.shape[1] * h / p.shape[0])
    mp_res.append(cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA))
row_u = np.hstack([p for pair in zip(mp_res, [sep] * 4) for p in pair][:-1])
W_common = max(row_m.shape[1], row_u.shape[1])


def pad_row(r):
    out = np.full((r.shape[0], W_common, 3), 30, np.uint8)
    out[:, :r.shape[1]] = r
    return out


sheet = np.vstack([pad_row(row_u), np.full((8, W_common, 3), 30, np.uint8), pad_row(row_m)])
cv2.imwrite(str(OUT / "turnaround_to_mesh.jpg"), sheet)
print("wrote", OUT / "turnaround_to_mesh.jpg")
