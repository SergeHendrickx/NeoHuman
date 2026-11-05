# NeoHuman

A local, interactive avatar studio for the [Momentum Human Rig (MHR)](../README.md):
shape your own NeoHuman (identity, expression, pose, eyes, mouth, clothing),
play Mixamo clips and BVH poses on it, then spawn GPU crowds and benchmark them
— all live in the browser, with a small Flask backend for model inference.

Land on your avatar: the first screen is a close-up of your character with the
Identity sliders. Crowd simulation, telemetry, the AI console, and other
technical tabs hide behind **Developer tools** (the Dev checkbox in the
viewport toolbar) — enable it to reveal them.

## Quickstart

Requirements: the repo's Python environment (3.11+), Node.js + npm, and an MHR
checkout with release assets (`assets/mhr_model.pt`, `assets/lod0.fbx`–`lod6.fbx`).

```powershell
npm install
python -m pip install -r requirements.txt
python server.py
```

Then open <http://127.0.0.1:8891>. Useful flags:

```powershell
python server.py --port 8892 --no-browser        # probe/CI instance
python server.py --host 0.0.0.0 --port 8893      # reach over LAN (use a secure context for WebGPU)
python server.py --model "C:\path\to\mhr_model.pt"
```

Content folders (all optional, overridden via `--mixamo`, `--hdri`, `--poses`):

- `data/mixamo/` — Mixamo `.fbx`/`.glb` clips, played retargeted on any actor.
- `data/360hdri/` — 360° `.exr`/`.hdr` environments with thumbnails.
- `data/makehuman/` — BVH pose presets plus teeth/tongue/eye-decoration assets.
- `data/logo/` — NeoHuman artwork.

## Features

- **Identity / Expression / Pose** — every parameter as sliders, with visual
  expression-vector editing in the 3D view, skeleton gizmo posing, and a pose
  preset library with thumbnails.
- **Eyes & mouth** — procedural eye shader with iris presets, fitted or
  procedural teeth (size/count/crookedness/gap/overbite/fangs), tongue shapes,
  modelled eyebrows/eyelashes, per-actor gaze and pupils.
- **Clothing** — shirt/shorts fitted to the body, with color and thickness.
- **Motion clips** — Mixamo retargeting (direction-rule transfer, spine boost,
  loop/hold/stop), Next/Prev transport, per-actor playback.
- **Shareable avatars** — Identity Copy/Import produces a compact versioned
  code (`?avatar=` links autoload it).
- **AI console** (dev) — in-app agent over an OpenAI-compatible API
  (Meta AI default), with a command menu and an action log.
- **Crowds** (dev) — autonomous GPU/CPU agents with wandering, auto-LOD,
  frustum culling, crowd eyes/blink, LOD heat colors, and per-agent stats.
- **Benchmark** (dev, Performance tab) — fixed-load or ramp-to-fps-floor runs
  with crowd framing, live progress, and saved history for cross-device
  comparison (4090 vs iGPU vs Quest).

## URL parameters

- `?benchmark` (alias `?bench`) — autostart a benchmark: `?benchmark&mode=ramp&step=10&interval=2&minfps=20`
  (`agents`, `seconds`, `lod`, `device` also supported; ramp is the default).
- `?avatar=<code>` — load a shared avatar on boot.
- `?noui` — hide all overlays for clean video capture.

## Testing

Backend + wiring tests (need the real MHR assets):

```powershell
python -m pytest tests -q
```

Frontend module tests:

```powershell
npm test
```

## Architecture

```text
Browser (Three.js, WebGPU Preferred)
  static/index.html + styles.css + app.js + js/ (~40 modules)
                 | HTTP and packed binary arrays
                 v
Flask server (server.py)
                 | TorchScript inference (CPU)
                 v
MHR model + official LOD mappings + FBX topology
```

`server.py` owns one thread-safe character state and evaluates the native
TorchScript LOD 1 model; other LODs apply MHR's official barycentric mappings.
The client keeps persistent meshes, evaluates pose/blendshapes/correctives
locally, and lets the GPU do skinning — the backend is only on the hook for
inference, assets, and exports (OBJ/GLB/USD).

Deeper docs: [arch.md](arch.md) (system architecture + analysis),
[gpu.md](gpu.md) (WebGPU usage), [gpu-crowd.md](gpu-crowd.md) (GPU crowd design).

## HTTP API

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/` | Viewer page |
| `GET` | `/api/metadata` | Parameter, model, and skeleton metadata |
| `GET` | `/api/rig-mapping`, `/api/lod-mapping`, `/api/topology`, `/api/geometry` | Packed rig/topology/geometry arrays |
| `GET` | `/api/weights/blendshapes`, `/api/weights/pose-correctives` | Packed weight arrays |
| `POST` | `/api/deform` | `set` / `set_group` / `reset_group` parameter values |
| `POST` | `/api/configure` | LOD, pose correctives, ground snapping |
| `GET` | `/api/mixamo/clips`, `/api/mixamo/file/<name>` | Clip library |
| `GET` | `/api/hdri/list`, `/api/hdri/file/<name>`, `/api/hdri/thumb/<name>` | Environments |
| `GET` | `/api/pose/list`, `/api/pose/file/<name>`, `/api/pose/thumb/<name>` | Pose presets |
| `GET` | `/api/mhasset/list`, `/api/mhasset/file/<path>` | Teeth/tongue/decoration assets |
| `GET` | `/api/model/rigged-glb` | Rigged base mesh |
| `GET` | `/api/export/<obj\|glb\|usd>` | Download the evaluated mesh |
| `GET` | `/health` | Server status and configuration |

`set_group` example (whole-vector write; lengths must match the group):

```json
{
  "action": "set_group",
  "group": "identity",
  "index": 0,
  "values": [0.0, 0.5, 0.0]
}
```

## Project layout

```text
MHR/
  assets/                  # Released model weights and FBX assets
  web-viewer/
    static/                # Browser UI: app.js + js/ modules, index.html, styles.css
    tests/                 # pytest suites + node module tests (agent/avatar/gpu/mixamo/...)
    tools/                 # Topology + eye-corner utilities
    data/                  # Committed topology archive, logo; local-only content folders
    exporters.py           # OBJ, GLB, and USD writers
    server.py              # Flask API and MHR inference engine
    requirements.txt       # Python dependencies
    package.json           # Three.js / dockview dependencies
```
