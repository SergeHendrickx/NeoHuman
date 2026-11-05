import * as THREE from "three";
import { WebGPURenderer } from "three/webgpu";
import { PostProcessing } from "three/webgpu";
import { pass } from "three/tsl";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import {
  JOINT_HOVER_RADIUS_PX,
  JOINT_OVERLAP_TOLERANCE_PX,
  VIEW_DIRECTIONS,
  elements,
  state,
} from "./state.js";
import { RGBELoader } from "three/addons/loaders/RGBELoader.js";
import { EXRLoader } from "three/addons/loaders/EXRLoader.js";
import { dof } from "three/addons/tsl/display/DepthOfFieldNode.js";
import { bloom } from "three/addons/tsl/display/BloomNode.js";
import { createDotTexture } from "./utils.js";
import { drawPerfGraph, recordFrameSample, recordStageTimes, setPerfIdle, updateRuntimeStats } from "./perf.js";
import { createSkinMaterial } from "./skin.js";

export let renderer = null;

export async function initRenderer() {
  let created = null;
  let webgpu = false;

  if (typeof navigator !== "undefined" && navigator.gpu) {
    try {
      const gpuRenderer = new WebGPURenderer({
        canvas: elements.canvas,
        antialias: true,
        alpha: false,
        preserveDrawingBuffer: true,
        powerPreference: "high-performance",
        // Zero-cost when the adapter lacks timestamp-query: the backend
        // gates tracking internally. Feeds the Crowd Stats GPU exec row.
        trackTimestamp: true,
      });
      await gpuRenderer.init();
      created = gpuRenderer;
      webgpu = true;
      console.log("NeoHuman: WebGPU renderer active.");
    } catch (err) {
      console.warn("WebGPU initialization fallback to WebGL:", err);
    }
  }

  if (!created) {
    created = new THREE.WebGLRenderer({
      canvas: elements.canvas,
      antialias: true,
      alpha: false,
      preserveDrawingBuffer: true,
      powerPreference: "high-performance",
    });
    webgpu = false;
    console.log("NeoHuman: WebGL renderer active.");
  }

  renderer = created;
  state.isWebGPU = webgpu;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.setClearColor(elements.backgroundColor.value, 1);
}

export const scene = new THREE.Scene();
export const perspectiveCamera = new THREE.PerspectiveCamera(32, 1, 0.01, 100);
export const orthographicCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
export let activeCamera = perspectiveCamera;
export let controls = new OrbitControls(activeCamera, elements.canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.screenSpacePanning = true;
controls.addEventListener("change", () => requestRender());

// ---- Camera follow: translate camera + target by the actor delta ----
let followOn = false;
let followProvider = null;
const followPrev = new THREE.Vector3();
let followInit = false;
const followDelta = new THREE.Vector3();

/** fn: () => THREE.Vector3-like world position, or null when N/A. */
export function setFollowProvider(fn) {
  followProvider = typeof fn === "function" ? fn : null;
  followInit = false;
}

export function setFollow(on) {
  followOn = Boolean(on);
  followInit = false;
  requestRender();
}

export function isFollowing() {
  return followOn;
}

function applyFollow() {
  if (!followOn || !followProvider) return;
  const p = followProvider();
  if (!p) return;
  if (!followInit) {
    followPrev.copy(p);
    followInit = true;
    return;
  }
  followDelta.copy(p).sub(followPrev);
  const jumped = followDelta.lengthSq() > 0.25;
  if (jumped) {
    // Selection switch or teleport: snap the anchor, don't fly there.
    followPrev.copy(p);
    return;
  }
  if (followDelta.lengthSq() > 0) {
    activeCamera.position.add(followDelta);
    controls.target.add(followDelta);
    followPrev.copy(p);
  }
}

// ---- Post chain (WebGPU only): DoF + bloom, direct render otherwise ----
export const fxState = {
  dof: false, focus: 3, range: 1.5, bokeh: 1,
  bloom: false, strength: 0.5, threshold: 0.85,
};
let post = null;
let postKey = "";
let dofNode = null;
let bloomNode = null;

export function effectsActive() {
  return state.isWebGPU && (fxState.dof || fxState.bloom);
}

export function getEffect(name) {
  return fxState[name];
}

export function setEffect(name, value) {
  if (!(name in fxState)) return;
  if (name === "dof" && value && !fxState.dof) {
    // Sensible default: focus where the camera already looks.
    fxState.focus = Number(activeCamera.position.distanceTo(controls.target).toFixed(1));
  }
  fxState[name] = value;
  // Live uniforms: no chain rebuild while sliding.
  if (post) {
    if (name === "focus" && dofNode) dofNode.focusDistanceNode.value = value;
    if (name === "range" && dofNode) dofNode.focalLengthNode.value = value;
    if (name === "bokeh" && dofNode) dofNode.bokehScaleNode.value = value;
    if (name === "strength" && bloomNode) bloomNode.strength.value = value;
    if (name === "threshold" && bloomNode) bloomNode.threshold.value = value;
  }
  requestRender();
}

export function currentFocusDistance() {
  return Number(activeCamera.position.distanceTo(controls.target).toFixed(1));
}

function renderScene() {
  renderer.info?.reset?.();
  if (effectsActive()) {
    ensurePost();
    post.render();
  } else {
    if (post) disposePost();
    renderer.render(scene, activeCamera);
  }
}

function disposePost() {
  try { dofNode?.dispose?.(); } catch { /* already gone */ }
  try { bloomNode?.dispose?.(); } catch { /* already gone */ }
  try { post?.dispose?.(); } catch { /* already gone */ }
  post = null;
  postKey = "";
  dofNode = null;
  bloomNode = null;
}

function ensurePost() {
  const key = `${state.projection}:${fxState.dof}:${fxState.bloom}`;
  if (post && key === postKey) return;
  disposePost();
  post = new PostProcessing(renderer);
  const scenePass = pass(scene, activeCamera);
  let out = scenePass.getTextureNode();
  if (fxState.dof) {
    dofNode = dof(out, scenePass.getViewZNode(), fxState.focus, fxState.range, fxState.bokeh);
    out = dofNode;
  }
  if (fxState.bloom) {
    bloomNode = bloom(out, fxState.strength, 0.4, fxState.threshold);
    out = out.add(bloomNode);
  }
  post.outputNode = out;
  postKey = key;
}

const hemisphereLight = new THREE.HemisphereLight(0xdde8ff, 0x283040, 2.15);
scene.add(hemisphereLight);
const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
keyLight.position.set(3, 4, 5);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x7798ff, 1.4);
rimLight.position.set(-4, 2, -3);
scene.add(rimLight);

export const bodyMaterial = createSkinMaterial(elements.bodyColor.value);

let backgroundTexture = null;

export function backgroundPanoramaActive() {
  return backgroundTexture !== null;
}

// Equirectangular 360 panorama (.hdr via RGBELoader, .exr via EXRLoader,
// anything else via the image loader) as both the visible background and
// the IBL environment. Works on WebGPU (node background/env sampling) and
// the WebGL fallback (classic equirect background + auto-PMREM
// environment) with no PMREMGenerator involved — it cannot run on
// WebGPURenderer. HDR sources stay linear; only LDR images get sRGB.
export async function loadBackgroundPanorama(url, name) {
  const isHdr = /\.hdr$/i.test(name || url);
  const isExr = /\.exr$/i.test(name || url);
  const loader = isExr ? new EXRLoader() : isHdr ? new RGBELoader() : new THREE.TextureLoader();
  const tex = await loader.loadAsync(url);
  if (!isHdr && !isExr) tex.colorSpace = THREE.SRGBColorSpace;
  tex.mapping = THREE.EquirectangularReflectionMapping;
  clearBackgroundPanorama(false);
  backgroundTexture = tex;
  scene.background = tex;
  scene.environment = tex;
  requestRender();
  return tex;
}

export function clearBackgroundPanorama(render = true) {
  if (backgroundTexture) {
    if (scene.background === backgroundTexture) scene.background = null;
    if (scene.environment === backgroundTexture) scene.environment = null;
    backgroundTexture.dispose();
    backgroundTexture = null;
  } else {
    scene.background = null;
    scene.environment = null;
  }
  if (renderer) renderer.setClearColor(elements.backgroundColor.value, 1);
  if (render) requestRender();
}

export function setEnvironmentIntensity(v) {
  scene.environmentIntensity = v;
  requestRender();
}

const skeletonMaterial = new THREE.LineBasicMaterial({
  color: 0xffc45c,
  transparent: true,
  opacity: 0.95,
  depthTest: false,
});
const skeletonLinesGeometry = new THREE.BufferGeometry();
const skeletonLines = new THREE.LineSegments(
  skeletonLinesGeometry,
  skeletonMaterial,
);
skeletonLines.renderOrder = 3;
skeletonLines.frustumCulled = false;

const skeletonPointsGeometry = new THREE.BufferGeometry();
export const skeletonPoints = new THREE.Points(
  skeletonPointsGeometry,
  new THREE.PointsMaterial({
    color: 0xfff1bd,
    size: 0.045,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.95,
    map: createDotTexture(),
    alphaTest: 0.5,
    depthTest: false,
  }),
);
skeletonPoints.renderOrder = 4;
skeletonPoints.frustumCulled = false;
export const skeletonOverlay = new THREE.Group();
skeletonOverlay.add(skeletonLines, skeletonPoints);
skeletonOverlay.visible = false;
scene.add(skeletonOverlay);
const skeletonProjection = new THREE.Vector3();

export const gridHelper = new THREE.GridHelper(4, 20, 0x526077, 0x2b3442);
gridHelper.material.transparent = true;
gridHelper.material.opacity = 0.52;
scene.add(gridHelper);

export const axesHelper = new THREE.AxesHelper(0.45);
axesHelper.visible = false;
scene.add(axesHelper);

// Buffers are sized from the loaded model's joint count on first use and only
// reallocated if it ever changes, so posing never churns GPU buffers.
function ensureSkeletonCapacity(jointCount) {
  const lineFloats = (jointCount - 1) * 2 * 3;
  const pointFloats = jointCount * 3;
  const lineAttr = skeletonLinesGeometry.getAttribute("position");
  if (!lineAttr || lineAttr.array.length !== lineFloats) {
    skeletonLinesGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(lineFloats), 3),
    );
  }
  const pointAttr = skeletonPointsGeometry.getAttribute("position");
  if (!pointAttr || pointAttr.array.length !== pointFloats) {
    skeletonPointsGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(pointFloats), 3),
    );
  }
}

export function updateSkeleton(joints) {
  ensureSkeletonCapacity(state.metadata.model.jointCount);
  state.jointPositions = joints;
  if (!state.projectedJoints || state.projectedJoints.length !== joints.length) {
    state.projectedJoints = new Float32Array(joints.length);
  }
  const linePos = skeletonLines.geometry.attributes.position.array;
  let lineIdx = 0;
  state.metadata.jointParents.forEach((parent, child) => {
    if (parent < 0 || parent === child) return;
    linePos[lineIdx++] = joints[child * 3];
    linePos[lineIdx++] = joints[child * 3 + 1];
    linePos[lineIdx++] = joints[child * 3 + 2];
    linePos[lineIdx++] = joints[parent * 3];
    linePos[lineIdx++] = joints[parent * 3 + 1];
    linePos[lineIdx++] = joints[parent * 3 + 2];
  });
  skeletonLines.geometry.attributes.position.needsUpdate = true;
  skeletonPoints.geometry.attributes.position.array.set(joints);
  skeletonPoints.geometry.attributes.position.needsUpdate = true;
}

export function hideBoneTooltip() {
  elements.boneTooltip.hidden = true;
}

// Pinned tooltips (a single-click expression readout) survive hover: hover
// paths check the pin, dismiss paths clear it.
let canvasTooltipPinned = false;

export function pinCanvasTooltip() {
  canvasTooltipPinned = true;
}

export function dismissCanvasTooltip() {
  canvasTooltipPinned = false;
  elements.boneTooltip.hidden = true;
}

function showBoneTooltip(event, names) {
  showCanvasTooltip(event, names);
}

export function showCanvasTooltip(event, rows) {
  elements.boneTooltip.replaceChildren(
    ...rows.map((text) => {
      const row = document.createElement("div");
      row.textContent = text;
      return row;
    }),
  );
  elements.boneTooltip.hidden = false;
  const viewportRect = elements.viewport.getBoundingClientRect();
  const margin = 8;
  const offset = 14;
  const left = Math.min(
    event.clientX - viewportRect.left + offset,
    elements.viewport.clientWidth - elements.boneTooltip.offsetWidth - margin,
  );
  const top = Math.min(
    event.clientY - viewportRect.top + offset,
    elements.viewport.clientHeight - elements.boneTooltip.offsetHeight - margin,
  );
  elements.boneTooltip.style.left = `${Math.max(margin, left)}px`;
  elements.boneTooltip.style.top = `${Math.max(margin, top)}px`;
}

export function updateBoneTooltip(event) {
  if (canvasTooltipPinned) return;
  if (
    !elements.skeleton.checked
    || event.buttons !== 0
    || !state.jointPositions
  ) {
    hideBoneTooltip();
    return;
  }

  const canvasRect = elements.canvas.getBoundingClientRect();
  const pointerX = event.clientX - canvasRect.left;
  const pointerY = event.clientY - canvasRect.top;
  const projected = state.projectedJoints;
  for (let joint = 0; joint < state.metadata.model.jointCount; joint += 1) {
    skeletonProjection
      .fromArray(state.jointPositions, joint * 3)
      .project(activeCamera);
    projected[joint * 3] =
      (skeletonProjection.x * 0.5 + 0.5) * canvasRect.width;
    projected[joint * 3 + 1] =
      (-skeletonProjection.y * 0.5 + 0.5) * canvasRect.height;
    projected[joint * 3 + 2] = skeletonProjection.z;
  }

  let closestJoint = -1;
  let closestDistance = JOINT_HOVER_RADIUS_PX ** 2;
  for (let joint = 0; joint < state.metadata.model.jointCount; joint += 1) {
    const offset = joint * 3;
    if (Math.abs(projected[offset + 2]) > 1) continue;
    const distance =
      (pointerX - projected[offset]) ** 2
      + (pointerY - projected[offset + 1]) ** 2;
    if (distance < closestDistance) {
      closestDistance = distance;
      closestJoint = joint;
    }
  }

  if (closestJoint < 0) {
    hideBoneTooltip();
    return;
  }

  const closestOffset = closestJoint * 3;
  const overlapDistance = JOINT_OVERLAP_TOLERANCE_PX ** 2;
  const names = [];
  for (let joint = 0; joint < state.metadata.model.jointCount; joint += 1) {
    const offset = joint * 3;
    if (Math.abs(projected[offset + 2]) > 1) continue;
    const distance =
      (projected[offset] - projected[closestOffset]) ** 2
      + (projected[offset + 1] - projected[closestOffset + 1]) ** 2;
    if (distance <= overlapDistance) {
      names.push(state.metadata.jointNames[joint]);
    }
  }
  showBoneTooltip(event, names);
}

function cameraMetrics() {
  if (state.skinnedMesh && state.skinnedMesh.geometry.boundingBox) {
    const box = state.skinnedMesh.geometry.boundingBox;
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDimension = Math.max(size.x, size.y, size.z);
    return { size, center, maxDimension };
  }
  if (!state.bounds) return null;
  const size = state.bounds.getSize(new THREE.Vector3());
  const center = state.bounds.getCenter(new THREE.Vector3());
  const maxDimension = Math.max(size.x, size.y, size.z);
  return { size, center, maxDimension };
}

function viewDirection(view) {
  return VIEW_DIRECTIONS[view] ?? VIEW_DIRECTIONS.front;
}

function faceMetrics() {
  if (!state.jointPositions || !state.metadata) return null;
  const names = state.metadata.jointNames;
  const head = names.indexOf("c_head");
  const crown = names.indexOf("c_head_null");
  if (head < 0 || crown < 0) return null;
  const headPos = new THREE.Vector3().fromArray(state.jointPositions, head * 3);
  const crownPos = new THREE.Vector3().fromArray(state.jointPositions, crown * 3);
  const center = headPos.clone().add(crownPos).multiplyScalar(0.5);
  const size = Math.max(headPos.distanceTo(crownPos) * 1.7, 0.2);
  return {
    size: new THREE.Vector3(size, size, size),
    center,
    maxDimension: size,
    direction: faceDirection(headPos),
  };
}

// Posed face-forward from the displayed joints (same head-frame recipe as
// eyesOral.js, in world space): up = crown - head, lateral = l_eye - r_eye,
// forward = lateral x up, sign-checked against the eye midpoint. Null when
// the eye joints are missing or degenerate, so the caller falls back to
// world front. state.jointPositions mirrors the active actor, so this
// follows neck/head rotation on actor #0 and on any selected agent.
function faceDirection(headPos) {
  const names = state.metadata.jointNames;
  const lEye = names.indexOf("l_eye");
  const rEye = names.indexOf("r_eye");
  const crown = names.indexOf("c_head_null");
  if (lEye < 0 || rEye < 0 || crown < 0) return null;
  const lPos = new THREE.Vector3().fromArray(state.jointPositions, lEye * 3);
  const rPos = new THREE.Vector3().fromArray(state.jointPositions, rEye * 3);
  const crownPos = new THREE.Vector3().fromArray(state.jointPositions, crown * 3);
  const eyeMid = lPos.clone().add(rPos).multiplyScalar(0.5);
  const up = crownPos.sub(headPos);
  const lat = lPos.sub(rPos);
  if (up.lengthSq() < 1e-10 || lat.lengthSq() < 1e-10) return null;
  up.normalize();
  lat.normalize();
  const fwd = new THREE.Vector3().crossVectors(lat, up);
  if (fwd.lengthSq() < 1e-6) return null;
  fwd.normalize();
  if (eyeMid.sub(headPos).dot(fwd) < 0) fwd.negate();
  return fwd;
}

// Last explicit framing target (agent box from the UI, or null for actor #0).
// Persists across projection toggles, which refit without a new target.
let lastFocus = null;

/**
 * Manual near/far/fov. Near/far apply to both cameras; fov is perspective
 * only (the input disables itself under orthographic). Presets recompute
 * near/far and write them back via syncCameraInputs so the UI never lies.
 */
export function setCameraParams({ near, far, fov } = {}) {
  if (near !== undefined && Number.isFinite(near)) {
    const n = Math.min(10, Math.max(0.001, near));
    perspectiveCamera.near = n;
    orthographicCamera.near = n;
  }
  if (far !== undefined && Number.isFinite(far)) {
    const f = Math.min(2000, Math.max(1, far));
    perspectiveCamera.far = f;
    orthographicCamera.far = f;
  }
  if (perspectiveCamera.near >= perspectiveCamera.far) {
    perspectiveCamera.near = perspectiveCamera.far / 1000;
    orthographicCamera.near = perspectiveCamera.near;
  }
  if (fov !== undefined && Number.isFinite(fov)) {
    perspectiveCamera.fov = Math.min(120, Math.max(10, fov));
  }
  perspectiveCamera.updateProjectionMatrix();
  orthographicCamera.updateProjectionMatrix();
  syncCameraInputs();
  requestRender();
}

/** Reflect live camera state in the Camera tab inputs (called after presets). */
export function syncCameraInputs() {
  if (elements.cameraNear) elements.cameraNear.value = perspectiveCamera.near.toFixed(4);
  if (elements.cameraFar) elements.cameraFar.value = perspectiveCamera.far.toFixed(1);
  if (elements.cameraFov) {
    elements.cameraFov.value = perspectiveCamera.fov.toFixed(1);
    const ortho = state.projection !== "perspective";
    elements.cameraFov.disabled = ortho;
    elements.cameraFov.title = ortho
      ? "Perspective field of view — disabled under orthographic projection"
      : "Perspective field of view, degrees";
  }
}

export function fitCamera(view = state.lastView, focus) {
  if (focus !== undefined) lastFocus = focus;
  const face = view === "face";
  const metrics = lastFocus ?? (face ? faceMetrics() ?? cameraMetrics() : cameraMetrics());
  if (!metrics) return;
  state.lastView = view;
  const direction = face
    ? (metrics.direction ?? VIEW_DIRECTIONS.front)
    : viewDirection(view);
  const distance =
    (metrics.maxDimension / (2 * Math.tan(THREE.MathUtils.degToRad(perspectiveCamera.fov / 2)))) * 1.18;

  perspectiveCamera.position
    .copy(metrics.center)
    .addScaledVector(direction, distance);
  perspectiveCamera.near = Math.max(distance / 1000, 0.001);
  perspectiveCamera.far = distance * 20;
  perspectiveCamera.updateProjectionMatrix();

  const half = metrics.maxDimension * 0.62;
  orthographicCamera.left = -half;
  orthographicCamera.right = half;
  orthographicCamera.top = half;
  orthographicCamera.bottom = -half;
  orthographicCamera.position
    .copy(metrics.center)
    .addScaledVector(direction, distance);
  orthographicCamera.near = Math.max(distance / 1000, 0.001);
  orthographicCamera.far = distance * 20;
  orthographicCamera.updateProjectionMatrix();

  controls.target.copy(metrics.center);
  activeCamera.lookAt(metrics.center);
  controls.update();
  syncCameraInputs();
}

function replaceControls(camera) {
  const target = controls.target.clone();
  controls.dispose();
  activeCamera = camera;
  controls = new OrbitControls(activeCamera, elements.canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.screenSpacePanning = true;
  controls.target.copy(target);
  controls.update();
  controls.addEventListener("change", () => requestRender());
}

export function setProjection(mode) {
  if (mode !== "orthographic" && mode !== "perspective") return;
  if (state.projection === mode) {
    syncProjectionUI();
    return;
  }
  state.projection = mode;
  replaceControls(mode === "orthographic" ? orthographicCamera : perspectiveCamera);
  syncProjectionUI();
  fitCamera(state.lastView);
  requestRender();
}

export function toggleProjection() {
  setProjection(state.projection === "perspective" ? "orthographic" : "perspective");
}

/** Toolbar button + Camera-tab segmented control show one state. */
export function syncProjectionUI() {
  const ortho = state.projection !== "perspective";
  if (elements.projection) elements.projection.textContent = ortho ? "Perspective" : "Orthographic";
  if (elements.cameraProjPersp) elements.cameraProjPersp.setAttribute("aria-pressed", String(!ortho));
  if (elements.cameraProjOrtho) elements.cameraProjOrtho.setAttribute("aria-pressed", String(ortho));
}

export function resizeRenderer() {
  const width = Math.max(1, Math.round(elements.viewport.clientWidth));
  const height = Math.max(1, Math.round(elements.viewport.clientHeight));
  renderer.setSize(width, height, false);
  const aspect = width / height;
  perspectiveCamera.aspect = aspect;
  perspectiveCamera.updateProjectionMatrix();

  const verticalHalf = (orthographicCamera.top - orthographicCamera.bottom) / 2;
  orthographicCamera.left = -verticalHalf * aspect;
  orthographicCamera.right = verticalHalf * aspect;
  orthographicCamera.updateProjectionMatrix();
}

let renderRequested = false;
let frameCount = 0;
let lastFpsUpdateTime = 0;
let lastFrameTimestamp = 0;
let idleTimer = null;

export function requestRender() {
  if (!renderRequested) {
    renderRequested = true;
    requestAnimationFrame(renderFrame);
  }
}

export function renderNow() {
  const tRender0 = performance.now();
  renderScene();
  recordStageTimes({ render: performance.now() - tRender0 });
  updateRuntimeStats(renderer, scene);
  maybeResolveRenderTime();
}

let renderExecCount = 0;
let renderExecMs = null;
let renderExecResolveCount = 0;
let renderResolvePending = false;

export function readRenderExecMs() {
  return renderExecMs;
}

/**
 * Drains the render timestamp pool (true whole-scene GPU ms/render).
 * Without this the pool overflows ~1024 renders after enabling
 * trackTimestamp. One resolve in flight max; no-op without support.
 */
function maybeResolveRenderTime() {
  renderExecCount++;
  if (renderResolvePending || renderExecCount % 30 !== 0) return;
  const backend = renderer?.backend;
  if (!backend || backend.trackTimestamp !== true) return;
  renderResolvePending = true;
  const from = renderExecResolveCount;
  renderer.resolveTimestampsAsync("render").then(
    (ms) => {
      renderResolvePending = false;
      if (typeof ms !== "number" || !Number.isFinite(ms)) return;
      const renders = Math.max(1, renderExecCount - from);
      renderExecMs = ms / renders;
      renderExecResolveCount = renderExecCount;
    },
    () => {
      renderResolvePending = false;
    },
  );
}

export function markRenderIdle() {
  setPerfIdle(true);
  if (elements.fpsCounter) {
    elements.fpsCounter.textContent = "Idle";
    elements.fpsCounter.title = "On-demand rendering: 0% idle GPU spin";
  }
  if (elements.fpsUnit) {
    elements.fpsUnit.textContent = "(0% GPU)";
  }
  if (elements.perfFpsVal) {
    elements.perfFpsVal.textContent = "Idle";
  }
  if (elements.perfFpsBadge) {
    elements.perfFpsBadge.className = "perf-badge is-muted";
    elements.perfFpsBadge.textContent = "On-Demand";
  }
  if (elements.perfFpsSub) {
    elements.perfFpsSub.textContent = "0% idle GPU spin";
  }
  drawPerfGraph();
}

function renderFrame(timestamp) {
  renderRequested = false;
  const now = timestamp || performance.now();
  applyFollow();
  const dampingActive = controls.update();
  const tRender0 = performance.now();
  // We drive our own rAF loop instead of setAnimationLoop, so per three's
  // docs we reset info manually each frame: otherwise drawCalls accumulates
  // forever and only the invocation counter (`calls`) is readable.
  renderScene();
  recordStageTimes({ render: performance.now() - tRender0 });
  maybeResolveRenderTime();

  if (idleTimer) {
    clearTimeout(idleTimer);
    idleTimer = null;
  }
  setPerfIdle(false);

  const delta = lastFrameTimestamp ? Math.min(now - lastFrameTimestamp, 66) : 16.6;
  recordFrameSample(delta);

  if (lastFrameTimestamp === 0 || now - lastFrameTimestamp > 300) {
    frameCount = 1;
    lastFpsUpdateTime = now;
  } else {
    frameCount++;
    const elapsed = now - lastFpsUpdateTime;
    if (elapsed >= 300) {
      const fps = Math.round((frameCount * 1000) / elapsed);
      if (elements.fpsCounter) {
        elements.fpsCounter.textContent = `${fps}`;
        elements.fpsCounter.title = `${fps} FPS active render rate`;
      }
      if (elements.fpsUnit) {
        elements.fpsUnit.textContent = "FPS";
      }
      if (elements.perfFpsVal) {
        elements.perfFpsVal.textContent = `${fps} FPS`;
      }
      if (elements.perfFpsBadge) {
        elements.perfFpsBadge.className = fps >= 55 ? "perf-badge is-success" : "perf-badge is-accent";
        elements.perfFpsBadge.textContent = "Active";
      }
      if (elements.perfFpsSub) {
        elements.perfFpsSub.textContent = `Frametime: ${delta.toFixed(1)} ms`;
      }
      frameCount = 0;
      lastFpsUpdateTime = now;
    }
  }
  lastFrameTimestamp = now;

  updateRuntimeStats(renderer, scene);
  drawPerfGraph();

  if (dampingActive) {
    requestRender();
  } else {
    idleTimer = setTimeout(markRenderIdle, 250);
  }
}

markRenderIdle();
