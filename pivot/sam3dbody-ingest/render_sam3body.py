"""Render a SAM 3D Body (MHR) mesh headlessly — no pyrender, no display needed.

Two backends:
  * ``open3d`` (default): Open3D 0.20 OffscreenRenderer, Vulkan on the GPU. ~0.2 s for a 2250x1500 frame.
  * ``numpy``: pure numpy + OpenCV painter's-algorithm rasteriser (no Open3D, no GL). ~1-2 s.

Both draw the *same geometry*: 3DB's ``pred_vertices`` plus ``pred_cam_t`` (camera-frame metres), its
own pinhole camera (``focal_length``, principal point at the image centre) and the MHR faces.

Outputs (default ``pivot/sam3dbody-ingest/artifacts/``):
  render_<case>_overlay.png     mesh composited on the original photo (exact 3DB camera)
  render_<case>_mesh.png        mesh alone on a neutral background
  render_<case>_turntable.jpg   six yaw views of the fitted mesh
  render_<case>.json            provenance and numbers

Usage (from A:\\dev\\GNM-V):
  .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\render_sam3body.py --case dancing
  .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\render_sam3body.py --case portrait --backend numpy
  .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\render_sam3body.py --image <path> --bbox x1 y1 x2 y2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]                 # A:\dev\GNM-V
HERE = Path(__file__).resolve().parent
os.environ.setdefault("GNM_REPO", str(REPO))
os.environ.setdefault("TORCH_HOME", str(REPO / ".cache" / "torch"))

THREEDB = REPO / "pivot" / "sam-3d-body"
THREEDB_CKPT = THREEDB / "sam-3d-body-dinov3" / "checkpoints" / "model.ckpt"
THREEDB_MHR = THREEDB / "sam-3d-body-dinov3" / "checkpoints" / "assets" / "mhr_model.pt"
OUT_DEFAULT = HERE / "artifacts"

CASES = {
    "dancing": ("pivot/sam-3d-body/notebook/images/dancing.jpg", [660, 210, 1510, 1430]),
    "portrait": ("pivot/teenage-woman-long-wavy-white-gray__01971-2947889296.png", [200, 60, 850, 1000]),
}

BASE_COLOUR = (0.58, 0.64, 0.74)
LIGHT_DIR = np.array([-0.35, -0.55, 0.75], np.float64)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)


# ------------------------------------------------------------------------------------- 3DB fit
def load_estimator():
    import torch  # noqa: PLC0415

    sys.path.insert(0, str(THREEDB))
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # noqa: PLC0415

    t0 = time.time()
    model, cfg = load_sam_3d_body(str(THREEDB_CKPT), device="cuda", mhr_path=str(THREEDB_MHR))
    est = SAM3DBodyEstimator(model, cfg, human_detector=None, human_segmentor=None, fov_estimator=None)
    faces = np.asarray(model.head_pose.faces.detach().cpu().numpy(), np.int32)
    return est, faces, model, time.time() - t0


def fit(est, bgr, bbox):
    import cv2  # noqa: PLC0415

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t0 = time.time()
    out = est.process_one_image(rgb, bboxes=np.asarray([bbox], np.float32), inference_type="body")[0]
    return out, time.time() - t0


def camera_geometry(out):
    """World/camera points (metres) + pinhole parameters that reproduce 3DB's projection."""
    verts = np.asarray(out["pred_vertices"], np.float64) + np.asarray(out["pred_cam_t"], np.float64)[None, :]
    return verts, float(out["focal_length"])


# ------------------------------------------------------------------------------------- open3d
def _o3d_ready():
    try:
        import open3d  # noqa: F401,PLC0415

        return True
    except Exception:  # noqa: BLE001
        return False


def render_open3d(verts, faces, f, width, height, bg=(0.10, 0.11, 0.13)):
    """Rendered RGB (uint8) + mask via Open3D's headless Filament/Vulkan renderer.

    Convention (verified): geometry is passed in 3DB's camera frame (x right, y down, z forward) and
    the camera extrinsic is the identity — Open3D's ``setup_camera(intrinsic, extrinsic)`` takes the
    camera-to-world transform, so identity = camera at the origin looking +z with y down, which is
    exactly 3DB's frame. The pinhole intrinsic is (f, f, W/2, H/2).
    """
    import open3d as o3d  # noqa: PLC0415
    import open3d.visualization.rendering as rendering  # noqa: PLC0415

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(list(BASE_COLOUR))

    r = rendering.OffscreenRenderer(width, height)
    scene = r.scene
    scene.set_background([bg[0], bg[1], bg[2], 1.0])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [*BASE_COLOUR, 1.0]
    mat.base_roughness = 0.55
    mat.base_reflectance = 0.12
    scene.add_geometry("body", mesh, mat)
    scene.scene.set_sun_light([-0.35, -0.55, 0.75], [1.0, 1.0, 1.0], 28000)
    try:                                                     # scene ambient differs across versions
        scene.scene.set_indirect_light(0.35)
    except Exception:  # noqa: BLE001 - lighting is cosmetic, keep going
        pass
    intr = o3d.camera.PinholeCameraIntrinsic(width, height, f, f, width / 2.0, height / 2.0)
    r.setup_camera(intr, np.eye(4))
    img = np.asarray(r.render_to_image()).copy()
    bg_px = img[min(height - 1, 2), min(width - 1, 2)].astype(np.int32)
    mask = np.abs(img.astype(np.int32) - bg_px[None, None, :]).max(axis=2) > 8
    return img, mask


# ------------------------------------------------------------------------------------- numpy
def render_numpy(verts, faces, f, width, height, bg=(0.10, 0.11, 0.13)):
    """Rendered RGB (uint8) + mask from a pure numpy/OpenCV painter's-algorithm rasteriser."""
    import cv2  # noqa: PLC0415

    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int32)
    z = verts[:, 2]
    ok = z > 1e-3
    uv = np.stack([f * verts[:, 0] / np.where(ok, z, 1) + width / 2.0,
                   f * verts[:, 1] / np.where(ok, z, 1) + height / 2.0], axis=1)

    tri_ok = ok[faces].all(axis=1)
    tri = faces[tri_ok]
    xyz = verts[tri]                                     # (F,3,3)
    n = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    lam = np.clip(np.abs(n @ LIGHT_DIR), 0.0, 1.0)       # two-sided: mesh has no culled backfaces
    shade = (0.30 + 0.70 * lam)[:, None]
    colours = (np.asarray(BASE_COLOUR)[None, :] * shade * 255.0).astype(np.int32)

    order = np.argsort(-z[tri].mean(axis=1))              # far first -> painter's algorithm
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :] = (np.asarray(bg) * 255).astype(np.uint8)
    mask = np.zeros((height, width), np.uint8)
    for k in order:
        pts = uv[tri[k]]
        if not np.isfinite(pts).all():
            continue
        if pts[:, 0].max() < 0 or pts[:, 0].min() > width - 1 or pts[:, 1].max() < 0 or pts[:, 1].min() > height - 1:
            continue
        poly = np.round(pts).astype(np.int32)
        colour = tuple(int(c) for c in colours[k])
        cv2.fillConvexPoly(img, poly, colour, lineType=cv2.LINE_AA)
        cv2.fillConvexPoly(mask, poly, 255, lineType=cv2.LINE_AA)
    return img, mask > 127


# ------------------------------------------------------------------------------------- views
def _raster(backend, verts, faces, f, width, height, bg):
    if backend == "open3d":
        return render_open3d(verts, faces, f, width, height, bg)
    return render_numpy(verts, faces, f, width, height, bg)


def overlay_photo(photo_bgr, img, mask, alpha=0.5):
    """Composite the render (rgb, mask) onto the photo (BGR) with a soft alpha."""
    import cv2  # noqa: PLC0415

    photo = cv2.cvtColor(photo_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    render = img.astype(np.float32)
    a = mask.astype(np.float32) * alpha
    a = cv2.GaussianBlur(a, (0, 0), 1.2)[..., None]
    out = photo * (1.0 - a) + render * a
    return np.clip(out, 0, 255).astype(np.uint8)


def turntable(backend, verts, faces, f, yaws_deg, size=900, bg=(0.16, 0.17, 0.19), margin=0.78):
    """Six yaw views of the fitted mesh, centred and framed from a fixed distance."""
    import cv2  # noqa: PLC0415

    centre = verts.mean(axis=0)
    p = verts - centre
    height_m = float(p[:, 1].max() - p[:, 1].min())
    f_tt = 1.15 * size
    dist = f_tt * height_m / (margin * size)
    frames = []
    for yaw in yaws_deg:
        c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        world = p @ rot.T
        world = world + np.array([0.0, 0.0, dist])
        img, mask = _raster(backend, world, faces, f_tt, size, size, bg)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img, f"{int(yaw):+d} deg", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (235, 235, 235), 2, cv2.LINE_AA)
        frames.append(img)
    return frames


def contact_sheet(frames, cols=6):
    import cv2  # noqa: PLC0415

    rows = (len(frames) + cols - 1) // cols
    h, w = frames[0].shape[:2]
    sheet = np.full((rows * h, cols * w, 3), 24, np.uint8)
    for i, fr in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[r * h:(r + 1) * h, c * w:(c + 1) * w] = fr
    return sheet


# ------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render a SAM 3D Body fit headlessly (Open3D or numpy backend).")
    ap.add_argument("--case", choices=sorted(CASES), default="dancing", help="built-in image/bbox pair")
    ap.add_argument("--image", type=Path, help="custom image (overrides --case)")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="person box in original pixels (required with --image)")
    ap.add_argument("--backend", choices=["open3d", "numpy", "auto"], default="auto")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--views", type=int, default=6, help="turntable views (0 disables)")
    ap.add_argument("--size", type=int, default=900, help="turntable tile size (px)")
    ap.add_argument("--alpha", type=float, default=0.5, help="overlay opacity")
    a = ap.parse_args(argv)

    import cv2  # noqa: PLC0415

    case = "custom" if a.image else a.case
    if a.image:
        if not a.bbox:
            ap.error("--image needs --bbox x1 y1 x2 y2")
        image_path, bbox = Path(a.image), list(map(float, a.bbox))
    else:
        rel, bbox = CASES[a.case]
        image_path = REPO / rel
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise SystemExit(f"cannot read {image_path}")
    H, W = bgr.shape[:2]

    backend = a.backend
    if backend == "auto":
        backend = "open3d" if _o3d_ready() else "numpy"

    t_load0 = time.time()
    est, faces, model, load_s = load_estimator()
    out, fit_s = fit(est, bgr, bbox)
    verts, f = camera_geometry(out)

    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    img, mask = _raster(backend, verts, faces, f, W, H, bg=(0.10, 0.11, 0.13))
    render_s = time.time() - t0
    overlay = overlay_photo(bgr, img, mask, a.alpha)
    files = {
        "overlay": a.out / f"render_{case}_overlay.png",
        "mesh": a.out / f"render_{case}_mesh.png",
    }
    cv2.imwrite(str(files["overlay"]), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(files["mesh"]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    tt_list = []
    if a.views:
        yaws = np.linspace(-180, 180, a.views, endpoint=False) + (180 / a.views)
        t0 = time.time()
        frames = turntable(backend, verts, faces, f, yaws, size=a.size)
        sheet = contact_sheet(frames, cols=min(a.views, 6))
        files["turntable"] = a.out / f"render_{case}_turntable.jpg"
        cv2.imwrite(str(files["turntable"]), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tt_list = [round(float(y), 1) for y in yaws]
        tt_s = time.time() - t0
    else:
        tt_s = 0.0

    meta = {
        "tool": "render_sam3body.py",
        "backend": backend,
        "case": case,
        "image": str(image_path),
        "image_size": [W, H],
        "bbox_xyxy": [float(v) for v in bbox],
        "model": "SAM 3D Body dinov3 (pivot/sam-3d-body) -> MHR LOD1 mesh",
        "geometry": {
            "vertices": int(verts.shape[0]), "faces": int(faces.shape[0]),
            "frame": "3DB camera frame, metres: pred_vertices + pred_cam_t; x right, y down, z forward",
            "focal_px": f, "principal_point": [W / 2.0, H / 2.0],
        },
        "fit": {"identity_l2": float(np.linalg.norm(out["shape_params"])),
                "expr_max_abs": float(np.abs(out["expr_params"]).max()),
                "seconds": fit_s},
        "timing_s": {"model_load": load_s, "fit": fit_s, "render": render_s, "turntable": tt_s},
        "turntable_yaws_deg": tt_list,
        "files": {k: str(v) for k, v in files.items()},
    }
    (a.out / f"render_{case}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"backend={backend}  {case}  {W}x{H}  mesh {verts.shape[0]} verts / {faces.shape[0]} faces")
    print(f"load {load_s:.1f}s  fit {fit_s:.2f}s  render {render_s:.2f}s  turntable {tt_s:.2f}s"
          f"  |identity|={np.linalg.norm(out['shape_params']):.2f}")
    for k, v in files.items():
        print(f"  {k:10s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
