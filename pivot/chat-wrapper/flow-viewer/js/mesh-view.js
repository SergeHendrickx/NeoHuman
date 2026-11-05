/**
 * Interactive 3D stage for mesh cards (three.js r186, vendored).
 *
 * Mounts an orbit/zoom viewer for GLB export results and OBJ fitted-mesh
 * refs. Render-on-demand only (controls change / resize / first frame) —
 * no always-on repaint. A poster image (turntable/sheet) covers loading and
 * every failure mode, so the stage never shows a black box.
 *
 * three.js is imported lazily inside mountMeshViewer so this module stays
 * import-safe under node (tests cover the pure view-model only).
 */
import { escapeHtml } from "./util.js";

export const THREE_VERSION = "0.186.1";
export const MESH_ACTION_KEYS = ["action-export", "action-body-fit", "action-texture",
                               "action-apose", "action-vggt-check", "action-mhr-match"];

/** Pure: the viewable mesh ref on a node, or null. GLB wins over OBJ. */
export function meshViewerKind(node) {
  const url = node.glbUrl || node.meshObjUrl || node.plyUrl || node.cloudUrl || null;
  if (!url) return null;
  const ext = String(url).split("?")[0].split(".").pop().toLowerCase();
  const format = (ext === "glb" || ext === "gltf") ? "glb" : (ext === "ply" ? "ply" : "obj");
  return { url, format };
}

/** Pure: whether a 3D stage belongs on this card. */
export function shouldMountMeshViewer(node) {
  if (!meshViewerKind(node)) return false;
  if (node.nodeType === "asset") return node.assetKind === "mesh";
  return MESH_ACTION_KEYS.includes(node.templateKey);
}

/**
 * Load a model for the viewer: GLB/OBJ as meshes, PLY as a coloured point cloud
 * (VGGT clouds are point sets with vertex colours, not surfaces).
 */
export async function loadViewerModel(THREE, loaders, kind) {
  const { GLTFLoader, OBJLoader, PLYLoader } = loaders;
  if (kind.format === "glb") return (await new GLTFLoader().loadAsync(kind.url)).scene;
  if (kind.format === "obj") return await new OBJLoader().loadAsync(kind.url);
  const geo = await new PLYLoader().loadAsync(kind.url);
  const pos = geo.getAttribute("position");
  const box = new THREE.Box3().setFromBufferAttribute(pos);
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z, 1e-3);
  const mat = new THREE.PointsMaterial({
    size: maxDim * 0.0035,
    sizeAttenuation: true,
    vertexColors: !!geo.getAttribute("color"),
    color: geo.getAttribute("color") ? 0xffffff : 0x9aa4b2,
  });
  return new THREE.Points(geo, mat);
}

/** Pure: download filename from the model URL tail, or a format fallback. */
export function meshDownloadName(url, format) {
  const tail = String(url || "").split("?", 1)[0].split("/").pop() || "";
  if (/^[A-Za-z0-9][A-Za-z0-9._-]*\.(glb|gltf|obj|ply)$/i.test(tail)) return tail;
  return `model.${format === "obj" ? "obj" : "glb"}`;
}

/**
 * Pure: stage markup. The poster URL (turntable/sheet) backs loading and
 * failure states; the status line is replaced by the canvas on success. The
 * download pill links the same-origin model file, so the browser saves it.
 */
export function renderMeshStageHtml(node, posterUrl) {
  const kind = meshViewerKind(node);
  if (!kind) return "";
  const poster = posterUrl ? ` data-mesh-poster="${escapeHtml(posterUrl)}"` : "";
  return `<div class="flow-mesh-stage" data-mesh-stage="1" data-mesh-format="${kind.format}"${poster}>`
    + `<div class="flow-mesh-status">Loading 3D ${kind.format.toUpperCase()}…</div>`
    + `<a class="flow-mesh-download" href="${escapeHtml(kind.url)}" download="${escapeHtml(meshDownloadName(kind.url, kind.format))}" title="Download the ${kind.format.toUpperCase()} file">⬇ ${kind.format.toUpperCase()}</a>`
    + `</div>`;
}

/** Pure: human fallback reason for a mount failure. */
export function meshFailureReason(err) {
  const msg = String((err && err.message) || err || "");
  if (/WebGL/i.test(msg) || /context/i.test(msg)) return "WebGL unavailable in this browser";
  if (/404/.test(msg)) return "model file not found (404)";
  if (/Failed to fetch|NetworkError|load/i.test(msg)) return "model fetch failed";
  if (/import|module|three/i.test(msg)) return "3D library failed to load";
  return msg ? msg.slice(0, 120) : "unknown 3D error";
}

export const MESH_OFFSCREEN_SIZE = 512;
export const MESH_RENDER_MODES = ["shaded", "depth", "normal"];
export const MULTIVIEW_COUNTS = [2, 4, 6, 8];
export const MULTIVIEW_CELL = 384;
/** Stage elevation: the 3/4-view dir sits ~14.7° above the horizon. */
export const MULTIVIEW_ELEVATION_DEG = 14.7;

/**
 * Pure: unit view direction for an azimuth (0° = front/+Z) at the stage
 * elevation. Multiview sheets step the azimuth and hold this elevation.
 */
export function viewDirForAzimuth(azDeg, elDeg = MULTIVIEW_ELEVATION_DEG) {
  const az = Number(azDeg) * Math.PI / 180;
  const el = Number(elDeg) * Math.PI / 180;
  return [Math.sin(az) * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el)];
}

/** Pure: evenly spaced front-starting azimuths for an N-view sheet. */
export function multiviewAzimuths(n) {
  const count = Math.floor(Number(n));
  if (!Number.isFinite(count) || count <= 0) return [];
  return Array.from({ length: count }, (_, i) => (i * 360) / count);
}

/** Pure: contact-sheet grid (cols, rows) per supported view count. */
export function multiviewGrid(n) {
  const grids = { 2: [2, 1], 4: [2, 2], 6: [3, 2], 8: [4, 2] };
  return grids[Number(n)] || [2, 2];
}

/**
 * Pure: contrast-stretch an RGBA readback's red channel (white-near depth)
 * over its full range into a new grayscale RGBA array. Background (0)
 * stays black; an all-black readback stays black.
 */
export function stretchDepthPixels(src) {
  let max = 0;
  for (let i = 0; i < src.length; i += 4) {
    if (src[i] > max) max = src[i];
  }
  const out = new Uint8ClampedArray(src.length);
  if (max <= 0) return out;
  for (let i = 0; i < src.length; i += 4) {
    const v = Math.round((src[i] / max) * 255);
    out[i] = v; out[i + 1] = v; out[i + 2] = v; out[i + 3] = 255;
  }
  return out;
}

/**
 * Browser-only: offscreen still of a mesh URL — shaded PBR, white-near
 * depth, or view-space normals — at MESH_OFFSCREEN_SIZE square. Same
 * upright + fit + 3/4-view convention as the interactive stage, so the
 * still matches what the stage shows. Resolves { dataUrl, width, height };
 * throws loud, human errors (no browser, no WebGL, fetch failed, bad
 * model). The GL context is created and destroyed per call — no resident
 * viewers, no context leak.
 */
export async function renderMeshStill(meshUrl, format, mode, opts = {}) {
  if (typeof window === "undefined" || typeof document === "undefined") {
    throw new Error("3D render needs a browser page");
  }
  if (!MESH_RENDER_MODES.includes(mode)) throw new Error(`unknown render mode "${mode}"`);
  if (!window.WebGLRenderingContext) throw new Error("WebGL unavailable in this browser");
  const { THREE, GLTFLoader, OBJLoader, PLYLoader, RoomEnvironment } = await loadThree();
  const size = Number(opts.size) > 0 ? Math.floor(Number(opts.size)) : MESH_OFFSCREEN_SIZE;
  const renderer = new THREE.WebGLRenderer({ antialias: mode === "shaded", alpha: mode === "shaded" });
  try {
    renderer.setSize(size, size, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    if (mode === "shaded") {
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 1.0;
    }
    const scene = new THREE.Scene();
    if (mode !== "shaded") {
      scene.overrideMaterial = mode === "depth" ? new THREE.MeshDepthMaterial() : new THREE.MeshNormalMaterial();
    }
    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 1000);
    let pmrem = null, envTex = null;
    if (mode === "shaded") {
      pmrem = new THREE.PMREMGenerator(renderer);
      envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
      scene.environment = envTex;
      const key = new THREE.DirectionalLight(0xffffff, 1.2);
      key.position.set(2, 3, 4);
      scene.add(key);
      scene.add(new THREE.AmbientLight(0xffffff, 0.25));
    }
    const object = await loadViewerModel(THREE, { GLTFLoader, OBJLoader, PLYLoader },
      { url: meshUrl, format });
    try {
      if (mode === "shaded") {
        object.traverse(o => {
          if (o.isMesh && (!o.material || (Array.isArray(o.material) && o.material.length === 0))) {
            o.material = new THREE.MeshStandardMaterial({ color: 0x9aa4b2, roughness: 0.65, metalness: 0.05 });
          }
        });
      }
      if (mode === "normal") {
        // Normal maps need the normal attribute; meshes without one (some
        // exports) render black, so derive smooth normals — never overriding
        // author normals when present.
        object.traverse(o => {
          if (o.isMesh && o.geometry && !o.geometry.attributes.normal) {
            o.geometry.computeVertexNormals();
          }
        });
      }
      // Same convention as the live stage: only camera-frame OBJ is Y-down.
      object.rotation.x = (format === "obj") ? Math.PI : 0;
      const box = new THREE.Box3().setFromObject(object);
      const center = box.getCenter(new THREE.Vector3());
      const dims = box.getSize(new THREE.Vector3());
      object.position.sub(center);
      scene.add(object);
      const maxDim = Math.max(dims.x, dims.y, dims.z, 1e-3);
      const dist = (maxDim / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.25;
      const dir = (opts.azimuthDeg === null || opts.azimuthDeg === undefined)
        ? new THREE.Vector3(0.9, 0.55, 1.9).normalize()
        : new THREE.Vector3(...viewDirForAzimuth(opts.azimuthDeg));
      camera.position.copy(dir.multiplyScalar(dist));
      if (mode === "depth") {
        // Tight planes squeeze the full 8-bit range onto the body.
        camera.near = Math.max(dist - maxDim * 0.75, 0.01);
        camera.far = dist + maxDim * 0.75;
      } else {
        camera.near = Math.max(dist / 100, 0.01);
        camera.far = dist * 100;
      }
      camera.updateProjectionMatrix();
      camera.lookAt(0, 0, 0);
      renderer.setClearColor(0x000000, mode === "shaded" ? 0 : 1);
      renderer.render(scene, camera);
      // Read back through a 2D canvas in the same task.
      const flat = document.createElement("canvas");
      flat.width = size; flat.height = size;
      const fctx = flat.getContext("2d");
      fctx.drawImage(renderer.domElement, 0, 0);
      if (mode === "depth") {
        const img = fctx.getImageData(0, 0, size, size);
        fctx.putImageData(new ImageData(stretchDepthPixels(img.data), size, size), 0, 0);
      }
      return { dataUrl: flat.toDataURL("image/png"), width: size, height: size };
    } finally {
      scene.remove(object);
      disposeObject(object);
      if (pmrem) pmrem.dispose();
      if (envTex) envTex.dispose();
    }
  } finally {
    renderer.dispose();
    try { renderer.forceContextLoss(); } catch { /* already gone */ }
  }
}

/**
 * Browser-only: contact sheet from rendered stills — grid per
 * multiviewGrid, azimuth label chip on each cell, stage-dark gutters.
 * Resolves { dataUrl, width, height }.
 */
export async function composeMultiviewSheet(stills, cell = MULTIVIEW_CELL) {
  if (typeof document === "undefined") throw new Error("Sheet compose needs a browser page");
  if (!Array.isArray(stills) || stills.length === 0) throw new Error("no stills to compose");
  const [cols, rows] = multiviewGrid(stills.length);
  const W = cols * cell, H = rows * cell;
  const sheet = document.createElement("canvas");
  sheet.width = W; sheet.height = H;
  const ctx = sheet.getContext("2d");
  ctx.fillStyle = "#0b0e13";
  ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < stills.length; i++) {
    const img = await new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error(`view ${i + 1} image failed to decode`));
      el.src = stills[i].dataUrl;
    });
    const x = (i % cols) * cell, y = Math.floor(i / cols) * cell;
    ctx.drawImage(img, x, y, cell, cell);
    const tag = String(stills[i].label || "");
    if (tag) {
      ctx.font = "600 20px ui-monospace, monospace";
      const tw = ctx.measureText(tag).width;
      ctx.fillStyle = "rgba(9, 12, 16, 0.85)";
      ctx.fillRect(x + 8, y + 8, tw + 16, 28);
      ctx.fillStyle = "#e6edf3";
      ctx.fillText(tag, x + 16, y + 29);
    }
  }
  return { dataUrl: sheet.toDataURL("image/png"), width: W, height: H };
}

let threePromise = null;

function loadThree() {
  if (!threePromise) {
    threePromise = Promise.all([
      import("three"),
      import("three/addons/controls/OrbitControls.js"),
      import("three/addons/loaders/GLTFLoader.js"),
      import("three/addons/loaders/OBJLoader.js"),
      import("three/addons/loaders/PLYLoader.js"),
      import("three/addons/environments/RoomEnvironment.js"),
    ]).then(([THREE, { OrbitControls }, { GLTFLoader }, { OBJLoader }, { PLYLoader }, { RoomEnvironment }]) => (
      { THREE, OrbitControls, GLTFLoader, OBJLoader, PLYLoader, RoomEnvironment }
    ));
  }
  return threePromise;
}

function showMeshFallback(stage, err) {
  const reason = meshFailureReason(err);
  const poster = stage.dataset.meshPoster;
  stage.querySelectorAll("canvas").forEach(c => c.remove());
  let status = stage.querySelector(".flow-mesh-status");
  if (!status) {
    status = document.createElement("div");
    status.className = "flow-mesh-status";
    stage.prepend(status);
  }
  status.textContent = `3D unavailable: ${reason}`;
  if (poster && !stage.querySelector("img.flow-mesh-poster")) {
    const img = document.createElement("img");
    img.className = "flow-node-img flow-mesh-poster";
    img.src = poster;
    img.alt = "3D preview unavailable — poster render";
    img.loading = "lazy";
    img.onerror = () => { img.style.display = "none"; };
    stage.appendChild(img);
  }
}

function disposeObject(root) {
  root.traverse(o => {
    if (o.geometry) o.geometry.dispose();
    const mats = Array.isArray(o.material) ? o.material : (o.material ? [o.material] : []);
    mats.forEach(m => {
      Object.values(m).forEach(v => {
        if (v && v.isTexture) v.dispose();
      });
      m.dispose();
    });
  });
}

/** Drop one card's viewer: invalidates in-flight mounts, frees the GL context. */
export function disposeMeshViewer(app, nodeId) {
  app._meshGen = app._meshGen || {};
  app._meshGen[nodeId] = (app._meshGen[nodeId] || 0) + 1;
  const rec = app._meshViewers ? app._meshViewers.get(nodeId) : null;
  if (!rec) return;
  try { rec.dispose(); } catch { /* already gone */ }
  app._meshViewers.delete(nodeId);
}

async function mountMeshViewer(app, node, stage) {
  const kind = meshViewerKind(node);
  if (!kind || typeof window === "undefined") return;
  app._meshGen = app._meshGen || {};
  const gen = (app._meshGen[node.id] || 0) + 1;
  app._meshGen[node.id] = gen;
  const live = () => app._meshGen[node.id] === gen && stage.isConnected;

  if (!window.WebGLRenderingContext) {
    throw new Error("WebGL unavailable in this browser");
  }
  const { THREE, OrbitControls, GLTFLoader, OBJLoader, PLYLoader, RoomEnvironment } = await loadThree();
  if (!live()) return;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  if (!live()) { renderer.dispose(); return; }
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 4 / 3, 0.01, 1000);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environment = envTex;
  const key = new THREE.DirectionalLight(0xffffff, 1.2);
  key.position.set(2, 3, 4);
  scene.add(key);
  scene.add(new THREE.AmbientLight(0xffffff, 0.25));

  const object = await loadViewerModel(THREE, { GLTFLoader, OBJLoader, PLYLoader }, kind);
  if (!live()) { disposeObject(object); renderer.dispose(); pmrem.dispose(); envTex.dispose(); return; }
  // OBJ refs ship without materials: give bare meshes a neutral PBR skin
  // so the fit reads as a body, not whiteout.
  object.traverse(o => {
    if (o.isMesh && (!o.material || (Array.isArray(o.material) && o.material.length === 0))) {
      o.material = new THREE.MeshStandardMaterial({ color: 0x9aa4b2, roughness: 0.65, metalness: 0.05 });
    }
  });
  // 3DB OBJ exports are Y-down camera coordinates, so they need a half-turn
  // about X. glTF is Y-up by spec and our PLY point clouds already are too -
  // flipping them made the model look like "face where the feet should be".
  object.rotation.x = (kind.format === "obj") ? Math.PI : 0;
  const box = new THREE.Box3().setFromObject(object);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  object.position.sub(center);
  scene.add(object);
  // Fit the tallest dimension inside the vertical fov with margin (a tall
  // body cropped head and feet when fitted by bounding-sphere radius).
  const maxDim = Math.max(size.x, size.y, size.z, 1e-3);
  const dist = (maxDim / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.25;
  const dir = new THREE.Vector3(0.9, 0.55, 1.9).normalize();
  camera.position.copy(dir.multiplyScalar(dist));
  camera.near = Math.max(dist / 100, 0.01);
  camera.far = dist * 100;
  camera.updateProjectionMatrix();

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = dist * 0.35;
  controls.maxDistance = dist * 4;
  controls.update();

  // Render-on-demand with a live loop only while the pointer is down:
  // damping animates during the drag, everything sleeps after. (Calling
  // controls.update() inside a change listener recurses — update dispatches
  // change synchronously — so the loop owns updates and resize owns the rest.)
  const render = () => {
    if (app._meshGen[node.id] !== gen) return;
    renderer.render(scene, camera);
  };
  let rafId = 0;
  const loop = () => {
    if (app._meshGen[node.id] !== gen) { rafId = 0; return; }
    controls.update();
    renderer.render(scene, camera);
    rafId = requestAnimationFrame(loop);
  };
  controls.addEventListener("start", () => {
    if (!rafId) rafId = requestAnimationFrame(loop);
  });
  controls.addEventListener("end", () => {
    if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
    render();
  });
  // The card drags on mousedown bubbling from anywhere in the card: keep
  // canvas gestures on the canvas (same pattern as the param editor).
  renderer.domElement.addEventListener("mousedown", (e) => e.stopPropagation());

  const fit = () => {
    const w = Math.max(stage.clientWidth, 16);
    const h = Math.max(stage.clientHeight, 16);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    render();
  };
  const ro = (typeof ResizeObserver !== "undefined") ? new ResizeObserver(fit) : null;
  if (ro) ro.observe(stage);

  if (!live()) {
    if (ro) ro.disconnect();
    controls.dispose(); disposeObject(object);
    renderer.dispose(); pmrem.dispose(); envTex.dispose();
    return;
  }
  stage.querySelectorAll(".flow-mesh-status").forEach(s => s.remove());
  renderer.domElement.className = "flow-mesh-canvas";
  stage.prepend(renderer.domElement);
  const hint = document.createElement("div");
  hint.className = "flow-mesh-hint";
  hint.textContent = "drag to orbit · scroll to zoom";
  stage.appendChild(hint);
  fit();

  app._meshViewers = app._meshViewers || new Map();
  app._meshViewers.set(node.id, {
    stage,
    dispose: () => {
      if (rafId) cancelAnimationFrame(rafId);
      if (ro) ro.disconnect();
      controls.dispose();
      disposeObject(object);
      pmrem.dispose();
      envTex.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === stage) stage.removeChild(renderer.domElement);
    },
  });
}

/**
 * Per-card bind (mirrors bindDisplayOpts): disposes the previous viewer for
 * this card, then mounts when the markup carries a mesh stage.
 */
export function bindMeshViewer(app, node, el) {
  disposeMeshViewer(app, node.id);
  if (!el || typeof el.querySelector !== "function") return;
  const stage = el.querySelector("[data-mesh-stage]");
  if (!stage || !shouldMountMeshViewer(node)) return;
  if (typeof window === "undefined") return;
  mountMeshViewer(app, node, stage).catch(err => {
    if (stage.isConnected) showMeshFallback(stage, err);
  });
}
