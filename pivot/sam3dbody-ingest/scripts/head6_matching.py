"""6-view HEAD matching for one subject: 2x3 flat-shaded face grid -> Muse photoreal grid -> VGGT consistency."""
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(REPO / "pivot" / "vggt-omega"))

import cv2
import numpy as np
import torch

import run_all as R
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

STEM = sys.argv[1] if len(sys.argv) > 1 else "teenage-woman-long-wavy-white-gray__01971-2947889296"
JOB = BASE / "outputs" / STEM
PHOTO = next((BASE / "input-images").rglob(f"{STEM}.*"))
OUT = JOB / "threedb"
TILE = 512
FACE_YAWS = [0.0, -45.0, 45.0, -90.0, 90.0, 180.0]
VGGT_CKPT = REPO / "pivot" / "vggt-omega" / "checkpoints" / "vggt_omega_1b_512.pt"

spec = importlib.util.spec_from_file_location("meta_image", REPO / "pipeline" / "avatar" / "meta_image.py")
mi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mi)

PROMPT = (
    "Image 1 is a flat-shaded 2x3 GRID of six head-and-shoulders views of a BALD mannequin head, in this "
    "order: row 1 = front, 3/4 left (-45), 3/4 right (+45); row 2 = left profile (-90), right profile (+90), "
    "back (180). Image 2 is the real person. Turn image 1 into photorealistic views of the SAME PERSON as "
    "image 2: same identity and face, same skin tone and skin texture, same eyebrows, same hair and hair "
    "colour as image 2 in every panel. Keep the SAME 2x3 grid, the same panel order, boundaries, head pose "
    "per panel, head scale and framing (head and upper neck, centred in each panel). FLAT, EVEN, SHADOWLESS "
    "LIGHTING, plain light-grey background, sharp focus, photographic, no text, no labels, no numbers, "
    "no borders, no watermark."
)

# ---- 1. guide: 2x3 face grid from the canonical A-pose/neutral mesh
model3d, est, faces = R.load_3db()
j = json.loads((OUT / "identity_joint.json").read_text(encoding="utf-8"))
beta = np.asarray(j["refined_shape_params"])
params = np.asarray(j["refined_mhr_model_params"]).copy()
params[:6] = 0.0
params[6:136] = 0.0
vroot, lm = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params, expr=np.zeros(72))
verts = vroot - vroot.mean(0)
face_lm = lm[70:272] - vroot.mean(0)
fc = face_lm.mean(0)
fr = float(np.percentile(np.linalg.norm(face_lm - fc[None], axis=1), 95))
del model3d; R.free({})

panels = []
for yaw in FACE_YAWS:
    th = np.deg2rad(yaw)
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    p = (verts - fc[None]) @ rot.T
    f2 = 1.2 * TILE
    dist = f2 * (2.0 * fr) / (0.80 * TILE)
    world = p + np.array([0.0, 0.0, dist])
    mesh, mat = R._make_mesh(world, faces, colour=R.MESH_COLOUR)
    img_hi, mask_hi = R._render_once(TILE, TILE, f2, mesh, mat)
    img, _ = R._finish(img_hi, mask_hi, TILE, TILE)
    panels.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
grid = np.full((2 * TILE, 3 * TILE, 3), 235, np.uint8)
for i, p in enumerate(panels):
    r, c = divmod(i, 3)
    grid[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE] = p
guide = OUT / "guide_face_grid_2x3.png"
cv2.imwrite(str(guide), grid)
print("guide", grid.shape)

# ---- 2. Muse
mi.PROMPTS["head-render"] = PROMPT
mi.REFERENCE_LABELS[0] = "Image 1 (flat-shaded 2x3 face-view grid; preserve layout, panel order, pose and framing):"
mi.REFERENCE_LABELS[1] = "Image 2 (the real person: identity, hair and skin reference):"
t0 = time.time()
rc = mi.main(["--mode", "head-render", "--reference", str(guide), "--reference2", str(PHOTO),
              "--out", str(OUT / "raw_face_grid"), "--size", "1536x1024", "--timeout", "300"])
sheet = OUT / "raw_face_grid" / "sheet.png"
if rc != 0 or not sheet.exists():
    raise SystemExit(f"Muse failed rc={rc}")
dst = OUT / "muse_face_grid_2x3.png"
cv2.imwrite(str(dst), cv2.imread(str(sheet)))
muse = cv2.imread(str(dst))
H, W = muse.shape[:2]
q_w, q_h = W // 3, H // 2
panel_files = []
for i, yaw in enumerate(FACE_YAWS):
    r, c = divmod(i, 3)
    p = muse[r * q_h:(r + 1) * q_h, c * q_w:(c + 1) * q_w]
    f = OUT / f"muse_face_{yaw:+.0f}.png".replace("+", "p").replace("-.", "m")
    f = OUT / f"muse_face_{str(int(yaw)).replace('-', 'm')}.png"
    cv2.imwrite(str(f), p)
    panel_files.append(f)
print("muse grid", muse.shape, "->", [f.name for f in panel_files])

# ---- 3. VGGT on the 6 panels -> head consistency (person mask by Sapiens2 seg)
imgs = load_and_preprocess_images([str(f) for f in panel_files], image_resolution=512)
vm = VGGTOmega().eval()
vm.load_state_dict(torch.load(VGGT_CKPT, map_location="cpu"))
vm = vm.cuda()
with torch.no_grad():
    preds = vm(imgs.cuda())
depth = preds["depth"][0, ..., 0].cpu().numpy().astype(np.float32)
conf = preds["depth_conf"][0].cpu().numpy().astype(np.float32)
ext, K = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
ext = ext[0].cpu().numpy().astype(np.float32); K = K[0].cpu().numpy().astype(np.float32)
del vm; torch.cuda.empty_cache()
print("VGGT extrinsics:")
for i, f in enumerate(panel_files):
    print(" ", f.name, "yaw", FACE_YAWS[i], "t", np.round(ext[i][:3, 3], 3).tolist())

seg_model = R.load_dense("seg", "seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
                         "seg/sapiens2_0.4b_seg.safetensors")
from sapiens.dense.visualizers import SegVisualizer
seg_vis = SegVisualizer(class_palette_type="dome29", with_labels=False)
worlds = {}
for i, f in enumerate(panel_files):
    bgr = cv2.imread(str(f))
    Hp, Wp = bgr.shape[:2]
    labels, _ = R.seg_run(seg_model, seg_vis, bgr)
    person = cv2.resize((labels > 0).astype(np.uint8), (depth[i].shape[1], depth[i].shape[0]),
                        interpolation=cv2.INTER_NEAREST) > 0
    m = person & (conf[i] > np.percentile(conf[i], 25)) & np.isfinite(depth[i]) & (depth[i] > 0.05)
    ys, xs = np.nonzero(m)
    d = depth[i][ys, xs]
    Xc = np.stack([(xs - K[i, 0, 2]) / K[i, 0, 0] * d, (ys - K[i, 1, 2]) / K[i, 1, 1] * d, d], 1)
    Rm, tt = ext[i][:3, :3], ext[i][:3, 3]
    worlds[f.name] = (Rm.T @ (Xc - tt[None]).T).T
del seg_model, seg_vis; R.free({})


def chamfer(A, B):
    A = A[:: max(1, len(A) // 4000)]
    B = B[:: max(1, len(B) // 4000)]
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.median(d) * 1000), float(np.percentile(d, 90) * 1000)


names = list(worlds)
report = {"subject": STEM, "yaws": FACE_YAWS}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        med, p90 = chamfer(worlds[names[i]], worlds[names[j]])
        report[f"{names[i]}__{names[j]}"] = {"median_mm": round(med, 1), "p90_mm": round(p90, 1)}
vals = [v["median_mm"] for k, v in report.items() if isinstance(v, dict) and "median_mm" in v]
report["summary"] = {"n_pairs": len(vals), "median_over_pairs_mm": round(float(np.median(vals)), 1),
                     "worst_pair_median_mm": round(float(np.max(vals)), 1)}
(OUT / "head6_matching.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report["summary"], indent=1))

# colours by view, project into the 0-deg camera frame
palette = [(1, .1, .1), (1, .6, 0), (.6, 1, 0), (0, .9, .3), (0, .5, 1), (.6, .1, 1)]
pts = np.concatenate([worlds[n] for n in names])
cc = np.concatenate([np.tile(palette[i % len(palette)], (len(worlds[n]), 1)) for i, n in enumerate(names)])
i0 = names.index([n for n, y in zip(names, FACE_YAWS) if y == 0.0][0])
R0, t0m, K0 = ext[i0][:3, :3], ext[i0][:3, 3], K[i0]
Xc = (R0 @ pts.T).T + t0m[None]
z = np.maximum(Xc[:, 2], 1e-6)
u = K0[0, 0] * Xc[:, 0] / z + K0[0, 2]
v = K0[1, 1] * Xc[:, 1] / z + K0[1, 2]
S = 900
u = (u - u.min()) / max(1e-6, (u.max() - u.min())) * (S - 1)
v = (v - v.min()) / max(1e-6, (v.max() - v.min())) * (S - 1)
canvas = np.full((S, S, 3), 240, np.uint8)
for k in np.argsort(-z):
    x, y = int(u[k]), int(v[k])
    if 0 <= x < S and 0 <= y < S:
        canvas[y, x] = (int(cc[k][2] * 255), int(cc[k][1] * 255), int(cc[k][0] * 255))
cv2.imwrite(str(OUT / "head6_points_rgb.png"), canvas)
print("wrote", OUT / "head6_matching.json", OUT / "head6_points_rgb.png")
