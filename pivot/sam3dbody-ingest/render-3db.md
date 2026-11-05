# How to render a SAM 3D Body fit — headless, on this box

**Short answer.** The repo's own renderer needs `pyrender`, which is **not installed** and needs a GL
context on Windows, so upstream visualization does not run here as-is. Rendering a fit is a one-mesh
problem, and two things on this box do it headlessly with no extra installs: **Open3D 0.20's
OffscreenRenderer** (Vulkan on the 4090, smooth shading) and a **pure numpy/OpenCV painter rasteriser**
(no GPU context at all). [`render_sam3body.py`](render_sam3body.py) wraps both, renders the fitted mesh
through 3DB's **own camera**, composites it on the photo, and makes a 6-view turntable.

## What you actually get from the model

| piece | where | shape / value |
| :--- | :--- | :--- |
| vertices | `out["pred_vertices"]` | `(18439, 3)` float32, **metres**, root-centred, 3DB camera axes, **not translated** |
| translation | `out["pred_cam_t"]` | `(3,)` metres — add it to vertices to get camera coordinates |
| faces | `est.faces` / `model.head_pose.faces` | `(36874, 3)` int — the MHR LOD1 topology (also in the checkpoint) |
| camera | `out["focal_length"]` | scalar px; principal point is `(W/2, H/2)` for the supplied-bbox path |
| projection | — | `x = f·X/Z + W/2`, `y = f·Y/Z + H/2` with `X,Y,Z = pred_vertices + pred_cam_t` |

Three traps, all confirmed in the ingest reports: (1) the vertices are **already** in 3DB's camera
convention (y/z flipped, metres) — do not flip them again; (2) the mesh has **no UVs or materials**
(the MHR TorchScript carries none), so it renders as untextured clay; (3) the fit is **neutral-jaw**
(expression and jaw params are zeroed in 3DB's head), so eyes are closed/neutral and the mouth is what
the identity predicts, not what the photo shows.

The notebook ([`sam3d_sapiens2.ipynb`](sam3d_sapiens2.ipynb)) verifies this projection independently:
its 308-keypoint projection reproduces 3DB's own `pred_keypoints_2d` to **max 0.0002 px**. The renders
below use exactly the same formula.

## Route 1 (recommended): Open3D OffscreenRenderer

Installed: `open3d 0.20.0`. `rendering.OffscreenRenderer` works **headless** here — it resolves the
Vulkan backend and picks the RTX 4090 (`FEngine resolved backend: Vulkan`); no display, no EGL, no
pyrender. Measured: **0.98 s** for the first 2250×1500 frame (≈0.7 s of that is engine init), then
**≈0.2 s** per frame; a 6-view turntable takes **2.5 s**.

```python
import numpy as np, open3d as o3d, open3d.visualization.rendering as rendering

verts = np.asarray(out["pred_vertices"], np.float64) + np.asarray(out["pred_cam_t"], np.float64)  # camera coords
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(np.asarray(model.head_pose.faces.detach().cpu()))
mesh.compute_vertex_normals()                      # smooth shading
mesh.paint_uniform_color([0.58, 0.64, 0.74])

r = rendering.OffscreenRenderer(W, H)
mat = rendering.MaterialRecord(); mat.shader = "defaultLit"; mat.base_color = [0.58, 0.64, 0.74, 1.0]
r.scene.set_background([0.10, 0.11, 0.13, 1.0])
r.scene.add_geometry("body", mesh, mat)
r.scene.scene.set_sun_light([-0.35, -0.55, 0.75], [1, 1, 1], 28000)
intr = o3d.camera.PinholeCameraIntrinsic(W, H, out["focal_length"], out["focal_length"], W / 2, H / 2)
r.setup_camera(intr, np.eye(4))                    # see convention note below
img = np.asarray(r.render_to_image())
```

**Camera convention, verified empirically.** Geometry goes in 3DB's camera frame and the extrinsic is
the **identity**. Tested alternatives: `diag(1,-1,-1)` (the naive "flip to OpenGL" matrix) puts the
whole mesh out of view; `identity`, on the other hand, produces a render whose silhouette bounding box
matches the projected vertices (x 720–1522 vs 720–1523 px on the dancing photo) and whose pose matches
the dancer. So Open3D's `setup_camera(intrinsic, extrinsic)` consumes a **camera-to-world** transform,
and identity means "camera at the origin, +z forward, y down" — exactly 3DB's frame. Keep that in mind
if you ever compose it with a different world transform.

**Mask for compositing:** use a background-colour difference, not the depth image — Open3D's
`render_to_depth_image` is normalised and saturates near the far plane, so it under-covers the feet.
Background-pixel distance > 8 (of 255) works on all frames tested.

## Route 2: pure numpy/OpenCV painter (no GL context needed)

`render_numpy()` in the script: project with the same pinhole formula, backface-agnostic two-sided
Lambert on face normals, sort triangles far→near, `cv2.fillConvexPoly` each one. **0.26 s** for a
1024² frame, 2.4 s for the 6-view turntable — as fast as Open3D on this geometry because the triangles
are small. Shading is **flat/faceted** (per-face) rather than smooth, which is visible on the portrait
cheek. It needs only numpy + OpenCV, so it also works in environments where no GL/Vulkan context exists
(e.g. a locked-down sandbox). Painter's algorithm is only approximate for intersecting geometry; a
closed body mesh renders correctly in every view tested.

## Route 3: upstream `pyrender` (what the repo actually ships)

`sam_3d_body/visualization/renderer.py` builds a `pyrender.OffscreenRenderer` scene with five ray
lights and `trimesh`; `tools/vis_utils.py` adds the 70-keypoint skeleton on top;
`notebook/utils.py` uses it in the demo. This route needs **`pyrender` installed** (absent here:
`pyrender`, `PyOpenGL`, `pyglet`, `moderngl` all missing) **and** a usable GL context — on
Windows that means a desktop session/pyglet window, on Linux EGL/OSMesa. Nothing installed it because
it is not needed: Route 1 runs on the 4090 today, and Routes 1–2 do not change the repo. If you do want
it: `uv pip install pyrender pyglet` (plus `pyopengl-accelerate` optionally), then

```python
from sam_3d_body.visualization.renderer import Renderer
from tools.vis_utils import visualize_sample_together
rend = Renderer(focal_length=out["focal_length"], faces=est.faces)
rgb = rend(out["pred_vertices"], out["pred_cam_t"], bgr_copy)   # renderer adds cam_t itself
```

Note it expects `pred_vertices` **without** `pred_cam_t` added (the renderer applies the translation),
whereas Routes 1–2 here take camera-space vertices directly.

## Route 4: the existing browser viewers

For interaction rather than stills, the repo already has three.js viewers (`app/`, the product
`parts_viewer.html`, the MHR LOD0 viewer). They consume a GLB; export one with
`pipeline/mhr/export_glb.py` / `product/export_glb.py` (the identity+bake pipeline). That is the right
route for a demo page, not for a headless still.

## Usage

```powershell
# default: Open3D backend, both built-in cases would need two runs
.\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\render_sam3body.py --case dancing
.\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\render_sam3body.py --case portrait --backend numpy

# any photo + manual person box (the only input 3DB needs on this path)
.\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\render_sam3body.py ^
    --image C:\\path\\to\\photo.jpg --bbox 660 210 1510 1430 --out pivot\\sam3dbody-ingest\\artifacts
```

Options: `--backend open3d|numpy|auto` (auto = open3d if importable), `--views N` (0 disables the
turntable), `--size` (turntable tile, default 900), `--alpha` (overlay opacity, default 0.5), `--out`.

Outputs per case, into `artifacts/`:

| file | what |
| :--- | :--- |
| `render_<case>_overlay.png` | mesh rendered through 3DB's exact camera, composited on the photo (feathered 1.2 px alpha mask) |
| `render_<case>_mesh.png` | the same render on a dark neutral background |
| `render_<case>_turntable.jpg` | 6 yaw views (−150…+150°) of the fitted mesh, centred and framed from a fixed distance |
| `render_<case>.json` | backend, image/bbox, focal, camera frame, `\|identity\|`, `max\|expr\|`, timings, yaws, file list |

Measured end-to-end (RTX 4090, `.venv`, 2026-11-03): model load **8.7–9.0 s**; 3DB fit **1.2 s**;
render **0.26 s** (numpy, 1024²) / **0.98 s first + 0.2 s** (Open3D, 2250×1500); turntable **2.4–2.5 s**
for 6 views. The fit is reused across all views — one forward pass, no re-fitting per view.

## Gotchas that will bite a renderer

- **A bad bbox is silently accepted** — 3DB has no input validation; a zero-area box gives NaN vertices
  and a background box gives plausible garbage. Check `np.isfinite(verts).all()` and `x2 > x1` before
  rendering (see [`runtime.md`](runtime.md) §failure modes).
- **Faces are topology, not attributes**: use `model.head_pose.faces` / `est.faces`; there is no
  `out["faces"]` in the estimator's output dict.
- **No UVs / textures**: `pred_vertices` + `faces` only. For a textured avatar you have to go through
  the repo's bake + texture pipeline, not a direct 3DB render.
- **The turntable rotates about y** — in 3DB's camera frame y points **down** (x right, z forward), so
  "up" is −y; rotate about y and do not also flip the mesh.
- **On a head-and-shoulders photo the body is a prior**: the portrait overlay shows a correctly aligned
  head but an extrapolated body. Render the head close-up, not the full frame, if you use that case.
- **Overlay alpha**: a hard mask leaves jaggies; the script feathers with a 1.2 px Gaussian.
