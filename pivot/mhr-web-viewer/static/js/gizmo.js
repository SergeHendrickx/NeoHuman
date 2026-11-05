import * as THREE from "three";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { currentParameter, elements, isLocked, state } from "./state.js";
import { rebuildParameterList, showParameter } from "./params.js";
import { actorManager, registerSphereSync } from "./actor.js";
import {
  addPoseUpdateListener,
  computeThetaJ,
  evaluatePoseClientSide,
} from "./rig.js";
import { syncPoseParams } from "./sync.js";
import { simulation } from "./sim/AgentSimulation.js";
import {
  activeCamera,
  controls,
  dismissCanvasTooltip,
  requestRender,
  scene,
  skeletonOverlay,
  skeletonPoints,
} from "./scene.js";

// Joint gizmo: each joint gets a clickable sphere on any actor tab. Selecting
// one attaches a Blender-style move / rotate / scale gizmo. Dragging it drives
// the underlying pose parameters through the rig mapping, so sliders, server
// sync, correctives, and exports all stay consistent.

const CLICK_PIXEL_TOLERANCE = 6;
const TP_WEIGHT_EPS = 1e-12;
const SPHERE_RADIUS = 0.014;
const SPHERE_SCREEN_PX = 10;
const GIZMO_SIZE = 0.65;
const SPHERE_COLOR = new THREE.Color(0xfff1bd);
const SPHERE_SELECTED_COLOR = new THREE.Color(0xffb02e);
const MODE_CHANNELS = { move: [0, 1, 2], rotate: [3, 4, 5], scale: [6] };
const TC_MODE = { move: "translate", rotate: "rotate", scale: "scale" };
// Multi-axis handles can never satisfy "one movement, one parameter", so they
// stay hidden; the scale-mode center box is the exception, it is uniform.
const MULTI_AXIS_HANDLES = /^(XY|YZ|XZ|XYZ|E|XYZE)$/;

let transformControls = null;
let proxy = null;
let spheres = null;
let sphereCount = 0;
let selectedJoint = -1;
let activeMode = "rotate";
let dragState = null;
let isGizmoDragging = false;
let suppressClick = false;
let downX = 0;
let downY = 0;
let downOnCanvas = false;

const raycaster = new THREE.Raycaster();
const pointerNdc = new THREE.Vector2();
const dummy = new THREE.Object3D();
const _parentInv = new THREE.Matrix4();
const _localM = new THREE.Matrix4();
const _rotM = new THREE.Matrix4();
const _quatM = new THREE.Matrix4();
const _preQ = new THREE.Quaternion();
const _euler = new THREE.Euler(0, 0, 0, "ZYX");
const _v3 = new THREE.Vector3();
const _col0 = new THREE.Vector3();
const _col1 = new THREE.Vector3();
const _col2 = new THREE.Vector3();

// Strongest-driven joint for one pose parameter (the inverse of
// channelPrimary): joint index, or -1 when the column is empty/dead.
export function strongestJointForParam(paramIndex) {
  if (state.currentGroup !== "pose" || !state.rigMapping || !state.metadata) return -1;
  const { TpMatrix, tpCols, jointCount } = state.rigMapping;
  if (!TpMatrix || paramIndex < 0 || paramIndex >= tpCols) return -1;
  let best = -1;
  let bestW = 0;
  for (let j = 0; j < jointCount; j++) {
    for (let c = 0; c < 7; c++) {
      const w = Math.abs(TpMatrix[(j * 7 + c) * tpCols + paramIndex]);
      if (w > bestW) {
        bestW = w;
        best = j;
      }
    }
  }
  return bestW > TP_WEIGHT_EPS ? best : -1;
}

// List→skeleton: selecting a parameter reveals (and selects) its joint,
// completing the skeleton→list jump (jumpToJointParam).
export function revealJointForParam(paramIndex) {
  const best = strongestJointForParam(paramIndex);
  if (best < 0) clearJointSelection();
  else selectJoint(best, paramJointMode(paramIndex, best));
}

// Strongest unlocked pose parameter driving one joint channel, or null when
// the channel has no available control.
function channelPrimary(joint, channel) {
  const { TpMatrix, tpCols } = state.rigMapping;
  const poseParams = state.metadata.groups.pose;
  const rowOffset = (joint * 7 + channel) * tpCols;
  let best = null;
  for (let col = 0; col < tpCols; col++) {
    const weight = TpMatrix[rowOffset + col];
    if (Math.abs(weight) <= TP_WEIGHT_EPS) continue;
    const param = poseParams[col];
    if (!param || isLocked(param)) continue;
    if (!best || Math.abs(weight) > Math.abs(best.weight)) {
      best = { index: col, weight };
    }
  }
  return best;
}

function jointModes(joint) {
  const available = (channel) => channelPrimary(joint, channel) !== null;
  return {
    move: [0, 1, 2].some(available),
    rotate: [3, 4, 5].some(available),
    scale: available(6),
  };
}

// Show exactly the axes the joint can drive; anything else stays invisible.
function updateHandleVisibility() {
  if (!transformControls || selectedJoint < 0) return;
  if (activeMode === "scale") {
    // All flags on: the XYZ center box requires every flag (core gates each
    // letter). The single-axis traverse below then strips everything but the
    // center box, leaving the scalar uniform-scale tool.
    transformControls.showX = true;
    transformControls.showY = true;
    transformControls.showZ = true;
    return;
  }
  const base = activeMode === "move" ? 0 : 3;
  transformControls.showX = channelPrimary(selectedJoint, base) !== null;
  transformControls.showY = channelPrimary(selectedJoint, base + 1) !== null;
  transformControls.showZ = channelPrimary(selectedJoint, base + 2) !== null;
}

function getActiveActor() {
  return actorManager.activeActor;
}

function getActiveSkeleton() {
  const actor = getActiveActor();
  return actor?.skeleton || state.skeleton;
}

function getActiveBoneIndexMap() {
  const actor = getActiveActor();
  return actor?.boneIndexMap || state.boneIndexMap;
}

function selectedBone() {
  const skeleton = getActiveSkeleton();
  const boneIndexMap = getActiveBoneIndexMap();
  if (!skeleton || !boneIndexMap || selectedJoint < 0) return null;
  return skeleton.bones[boneIndexMap[selectedJoint]];
}

// Principal axis through world points (the line connecting affected
// joints): exact for two points, power-iteration PCA for more. Oriented
// from the first point toward the last. Null when degenerate.
export function axisThroughPoints(points) {
  if (!points || points.length < 2) return null;
  const first = points[0];
  const last = points[points.length - 1];
  const dx = last.x - first.x, dy = last.y - first.y, dz = last.z - first.z;
  if (points.length === 2) {
    const len = Math.hypot(dx, dy, dz);
    if (len < 1e-9) return null;
    return new THREE.Vector3(dx / len, dy / len, dz / len);
  }
  let mx = 0, my = 0, mz = 0;
  for (const p of points) { mx += p.x; my += p.y; mz += p.z; }
  mx /= points.length; my /= points.length; mz /= points.length;
  let cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
  for (const p of points) {
    const x = p.x - mx, y = p.y - my, z = p.z - mz;
    cxx += x * x; cxy += x * y; cxz += x * z;
    cyy += y * y; cyz += y * z; czz += z * z;
  }
  let vx = dx, vy = dy, vz = dz;
  let vlen = Math.hypot(vx, vy, vz);
  if (vlen < 1e-9) { vx = 1; vy = 0; vz = 0; vlen = 1; }
  vx /= vlen; vy /= vlen; vz /= vlen;
  for (let i = 0; i < 48; i++) {
    const nx = cxx * vx + cxy * vy + cxz * vz;
    const ny = cxy * vx + cyy * vy + cyz * vz;
    const nz = cxz * vx + cyz * vy + czz * vz;
    const nlen = Math.hypot(nx, ny, nz);
    if (nlen < 1e-12) return null;
    vx = nx / nlen; vy = ny / nlen; vz = nz / nlen;
  }
  if (vx * dx + vy * dy + vz * dz < 0) { vx = -vx; vy = -vy; vz = -vz; }
  return new THREE.Vector3(vx, vy, vz);
}

const _moveY = new THREE.Vector3();
const _moveZ = new THREE.Vector3();
const _moveM = new THREE.Matrix4();
const _worldY = new THREE.Vector3(0, 1, 0);
const _worldZ = new THREE.Vector3(0, 0, 1);

// Move-tool orientation from the current param's affected joints: X runs
// through the joints (l_upleg→r_upleg spans the hips), Y/Z completed
// orthonormally. Null when unresolvable (falls back to the bone).
function affectedJointsMoveQuat() {
  const params = state.metadata?.groups?.pose;
  const param = params?.[state.selectedIndex?.pose];
  const names = param?.joints;
  if (!names || names.length < 2 || !state.jointPositions) return null;
  const jointNames = state.metadata.jointNames;
  const pts = [];
  for (const name of names) {
    const j = jointNames.indexOf(name);
    if (j < 0) continue;
    pts.push(new THREE.Vector3().fromArray(state.jointPositions, j * 3));
  }
  if (pts.length < 2) return null;
  const dir = axisThroughPoints(pts);
  if (!dir) return null;
  const up = Math.abs(dir.y) > 0.9 ? _worldZ : _worldY;
  _moveZ.crossVectors(dir, up).normalize();
  _moveY.crossVectors(_moveZ, dir).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(_moveM.makeBasis(dir, _moveY, _moveZ));
}

function syncProxyFromBone() {
  if (selectedJoint < 0 || !proxy) return;
  const bone = selectedBone();
  if (!bone) return;
  bone.getWorldPosition(proxy.position);
  // Move mode points X through the param's affected joints so the
  // translation axis matches the param's extent, not one bone's twist.
  const moveQuat = activeMode === "move" ? affectedJointsMoveQuat() : null;
  if (moveQuat) proxy.quaternion.copy(moveQuat);
  else bone.getWorldQuaternion(proxy.quaternion);
  proxy.scale.setScalar(activeMode === "scale" ? bone.scale.x : 1);
  proxy.updateMatrix();
}

function ensureSpheres(jointCount) {
  if (spheres && sphereCount === jointCount) return;
  if (spheres) {
    scene.remove(spheres);
    spheres.geometry.dispose();
    spheres.material.dispose();
  }
  const geometry = new THREE.SphereGeometry(SPHERE_RADIUS, 12, 8);
  const material = new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0.9,
    depthTest: false,
  });
  spheres = new THREE.InstancedMesh(geometry, material, jointCount);
  spheres.frustumCulled = false;
  spheres.renderOrder = 5;
  spheres.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  sphereCount = jointCount;
  scene.add(spheres);
  updateSphereColors();
}

export function syncJointSpheres() {
  if (!state.metadata) return;
  const jointCount = state.metadata.model.jointCount;
  ensureSpheres(jointCount);
  const actor = getActiveActor();
  const positions = (actor && actor.jointPositions) ? actor.jointPositions : state.jointPositions;
  if (!positions) return;
  // Constant screen size: world diameter from true pixels at each joint.
  const canvasH = elements.canvas.clientHeight || 1;
  const ortho = activeCamera.isOrthographicCamera;
  const orthoWpp = ortho ? (activeCamera.top - activeCamera.bottom) / canvasH : 0;
  const perspK = ortho
    ? 0
    : (2 * Math.tan(THREE.MathUtils.degToRad(activeCamera.fov / 2))) / canvasH;
  for (let j = 0; j < jointCount; j++) {
    dummy.position.fromArray(positions, j * 3);
    dummy.quaternion.identity();
    const wpp = ortho ? orthoWpp : perspK * dummy.position.distanceTo(activeCamera.position);
    dummy.scale.setScalar((SPHERE_SCREEN_PX * wpp) / (2 * SPHERE_RADIUS));
    dummy.updateMatrix();
    spheres.setMatrixAt(j, dummy.matrix);
  }
  spheres.instanceMatrix.needsUpdate = true;
  updateSphereVisibility();
}

function onCameraChanged() {
  if (spheres) syncJointSpheres();
}

let sphereCamHook = null;
function hookCameraRescale() {
  if (sphereCamHook === controls || !controls) return;
  sphereCamHook = controls;
  controls.addEventListener("change", onCameraChanged);
}

function updateSphereColors() {
  if (!spheres) return;
  for (let j = 0; j < sphereCount; j++) {
    spheres.setColorAt(j, j === selectedJoint ? SPHERE_SELECTED_COLOR : SPHERE_COLOR);
  }
  if (spheres.instanceColor) spheres.instanceColor.needsUpdate = true;
}

export function updateSphereVisibility() {
  const show = skeletonOverlay.visible && sphereCount > 0;
  if (spheres) spheres.visible = show;
  skeletonPoints.visible = skeletonOverlay.visible && !show;
}

function onPoseEvaluated() {
  syncJointSpheres();
  if (selectedJoint >= 0 && !isGizmoDragging) syncProxyFromBone();
}

function updateToolbar() {
  const visible = selectedJoint >= 0;
  elements.jointToolbar.hidden = !visible;
  if (!visible) return;
  elements.jointToolbarName.textContent = state.metadata.jointNames[selectedJoint];
  const modes = jointModes(selectedJoint);
  const buttons = {
    move: elements.jointModeMove,
    rotate: elements.jointModeRotate,
    scale: elements.jointModeScale,
  };
  for (const [mode, button] of Object.entries(buttons)) {
    button.disabled = !modes[mode];
    button.classList.toggle("is-active", mode === activeMode && modes[mode]);
  }
}

export function hasJointSelection() {
  return selectedJoint >= 0;
}

// Dominant channel of one parameter for one joint, as a gizmo mode —
// so scale-driven params open the scalar (uniform-scale) tool instead of
// rotate/move. Null when the pair has no weight.
export function paramJointMode(paramIndex, joint) {
  if (!state.rigMapping) return null;
  const { TpMatrix, tpCols, jointCount } = state.rigMapping;
  if (!TpMatrix || paramIndex < 0 || paramIndex >= tpCols) return null;
  if (joint < 0 || joint >= jointCount) return null;
  let bestC = -1;
  let bestW = 0;
  for (let c = 0; c < 7; c++) {
    const w = Math.abs(TpMatrix[(joint * 7 + c) * tpCols + paramIndex]);
    if (w > bestW) {
      bestW = w;
      bestC = c;
    }
  }
  if (bestC < 0 || bestW <= TP_WEIGHT_EPS) return null;
  return bestC <= 2 ? "move" : bestC <= 5 ? "rotate" : "scale";
}

export function selectJoint(joint, preferredMode = null) {
  if (!state.metadata || !state.skeleton || !transformControls) return;
  selectedJoint = joint;
  const modes = jointModes(joint);
  if (preferredMode && modes[preferredMode]) {
    activeMode = preferredMode;
  } else if (!modes[activeMode]) {
    activeMode = modes.rotate ? "rotate" : modes.move ? "move" : "scale";
  }
  const hasControls = modes.move || modes.rotate || modes.scale;
  syncProxyFromBone();
  if (hasControls) {
    transformControls.setMode(TC_MODE[activeMode]);
    updateHandleVisibility();
    transformControls.attach(proxy);
    transformControls.enabled = true;
    transformControls.getHelper().visible = true;
  } else {
    transformControls.detach();
    transformControls.enabled = false;
    transformControls.getHelper().visible = false;
  }
  updateToolbar();
  updateSphereColors();
  dismissCanvasTooltip();
  requestRender();
}

export function clearJointSelection() {
  if (selectedJoint < 0) return;
  selectedJoint = -1;
  dragState = null;
  if (transformControls) {
    transformControls.detach();
    transformControls.enabled = false;
    transformControls.getHelper().visible = false;
  }
  updateToolbar();
  updateSphereColors();
  requestRender();
}

export function setJointMode(mode) {
  if (selectedJoint < 0 || !TC_MODE[mode]) return;
  if (isGizmoDragging) return;
  if (!jointModes(selectedJoint)[mode]) return;
  activeMode = mode;
  transformControls.setMode(TC_MODE[mode]);
  updateHandleVisibility();
  syncProxyFromBone();
  updateToolbar();
  requestRender();
}

// Re-resolve modes and handles after lock state changes (snap toggle).
export function refreshJointControls() {
  if (selectedJoint < 0) return;
  selectJoint(selectedJoint);
}

// Strongest unlocked driver across the joint's channels: "the" parameter of
// an orb click. The panel jumps to it so the slider follows the joint.
export function jumpToJointParam(joint) {
  if (!state.rigMapping || !state.metadata) return null;
  let best = null;
  for (let channel = 0; channel < 7; channel++) {
    const primary = channelPrimary(joint, channel);
    if (!primary) continue;
    if (!best || Math.abs(primary.weight) > Math.abs(best.weight)) best = primary;
  }
  if (!best) return null;
  state.selectedIndex.pose = best.index;
  elements.search.value = "";
  elements.category.value = "";
  rebuildParameterList();
  return best.index;
}

export function syncGizmoCamera() {
  if (transformControls) transformControls.camera = activeCamera;
  hookCameraRescale();
}

export function handleGroupChanged() {
  // Skeleton stays exactly as the user left it: no auto-enable on the
  // default Pose group (avatar-first landing page).
  if (state.currentGroup !== "pose") {
    clearJointSelection();
  }
  updateSphereVisibility();
}

// Translate the proxy's dragged transform back into the 7 joint channels.
// Only the active mode's channels are filled in; the rest stay undefined.
function proxyToChannels() {
  if (selectedJoint < 0 || !state.rigMapping || !dragState) return null;
  const { jointOffsets, jointPreRotations } = state.rigMapping;
  const joint = dragState.joint;
  const bone = selectedBone();
  if (!bone || !bone.parent) return null;
  _parentInv.copy(bone.parent.matrixWorld).invert();
  proxy.updateMatrix();

  const out = new Array(7);
  if (dragState.mode === "move") {
    _localM.multiplyMatrices(_parentInv, proxy.matrix);
    _v3.setFromMatrixPosition(_localM);
    out[0] = _v3.x * 100 - jointOffsets[joint * 3];
    out[1] = _v3.y * 100 - jointOffsets[joint * 3 + 1];
    out[2] = _v3.z * 100 - jointOffsets[joint * 3 + 2];
  } else if (dragState.mode === "rotate") {
    _localM.multiplyMatrices(_parentInv, proxy.matrix);
    _col0.setFromMatrixColumn(_localM, 0).normalize();
    _col1.setFromMatrixColumn(_localM, 1).normalize();
    _col2.setFromMatrixColumn(_localM, 2).normalize();
    _rotM.makeBasis(_col0, _col1, _col2);
    _preQ.set(
      jointPreRotations[joint * 4],
      jointPreRotations[joint * 4 + 1],
      jointPreRotations[joint * 4 + 2],
      jointPreRotations[joint * 4 + 3],
    );
    _quatM.makeRotationFromQuaternion(_preQ.invert());
    _rotM.premultiply(_quatM);
    _euler.setFromRotationMatrix(_rotM, "ZYX");
    // Unwrap against the drag-start pose so Euler branch cuts can't jump params.
    for (const [index, raw] of [[3, _euler.x], [4, _euler.y], [5, _euler.z]]) {
      const start = dragState.startChannels[index];
      out[index] = raw + Math.round((start - raw) / (2 * Math.PI)) * 2 * Math.PI;
    }
  } else {
    // Joints scale uniformly: collapse the drag to one factor and snap the
    // proxy back to uniform so the next event measures cleanly.
    const factor = Math.max(
      0.01,
      (proxy.scale.x + proxy.scale.y + proxy.scale.z) / 3,
    );
    proxy.scale.setScalar(factor);
    proxy.updateMatrix();
    out[6] = Math.log2(factor);
  }
  return out;
}

function onGizmoMouseDown() {
  if (selectedJoint < 0) return;
  isGizmoDragging = true;
  suppressClick = true;
  controls.enabled = false;
  const actor = getActiveActor();
  const thetaJ = (actor && !actor.isAnchor && actor.thetaJ) ? actor.thetaJ : computeThetaJ();
  const base = selectedJoint * 7;
  const poseParams = state.metadata.groups.pose;
  dragState = {
    actor,
    joint: selectedJoint,
    mode: activeMode,
    startChannels: Array.from(thetaJ.subarray(base, base + 7)),
    startValues: new Map(),
    primaries: new Map(),
    changed: new Set(),
  };
  for (const channel of MODE_CHANNELS[activeMode]) {
    const primary = channelPrimary(selectedJoint, channel);
    if (!primary) continue;
    dragState.primaries.set(channel, primary);
    if (!dragState.startValues.has(primary.index)) {
      const val = actor ? actor.getParamValue("pose", primary.index) : poseParams[primary.index].value;
      dragState.startValues.set(primary.index, val);
    }
  }
  dismissCanvasTooltip();
}

function onGizmoObjectChange() {
  if (!dragState || dragState.joint !== selectedJoint) return;
  const axis = transformControls.axis;
  if (!axis) return;
  // One movement, one parameter: only single-axis drags map to a channel
  // (plus the uniform scale-center box). Anything else is ignored.
  let channel = -1;
  if (dragState.mode === "scale") {
    channel = 6;
  } else if (axis.length === 1) {
    const axisIndex = "XYZ".indexOf(axis);
    if (axisIndex < 0) return;
    channel = (dragState.mode === "move" ? 0 : 3) + axisIndex;
  } else {
    return;
  }
  const primary = dragState.primaries.get(channel);
  if (!primary) return;
  const target = proxyToChannels();
  if (!target) return;
  const delta = target[channel] - dragState.startChannels[channel];
  const poseParams = state.metadata.groups.pose;
  const param = poseParams[primary.index];
  const start = dragState.startValues.get(primary.index);
  const next = Math.min(
    param.max,
    Math.max(param.min, start + delta / primary.weight),
  );
  param.value = next;
  const actor = dragState.actor;
  if (actor) {
    actor.setParamValue("pose", primary.index, next);
  }
  dragState.changed.add(primary.index);
  if (!actor || actor.isAnchor) {
    evaluatePoseClientSide();
  } else {
    actor.evaluatePose();
    simulation.refreshGpuAgents();
    syncJointSpheres();
  }
  const current = currentParameter();
  if (current && state.currentGroup === "pose" && current.index === primary.index) {
    showParameter(current);
  }
  requestRender();
}

function onGizmoMouseUp() {
  isGizmoDragging = false;
  controls.enabled = true;
  if (!dragState) return;
  const changed = [...dragState.changed];
  const actor = dragState.actor;
  dragState = null;
  if (changed.length > 0 && (!actor || actor.isAnchor)) {
    syncPoseParams(changed);
  }
  syncProxyFromBone();
  requestRender();
}

function onCanvasPointerDown(event) {
  downX = event.clientX;
  downY = event.clientY;
  downOnCanvas = true;
}

function onPointerUp(event) {
  const wasSuppressed = suppressClick;
  suppressClick = false;
  const wasDownOnCanvas = downOnCanvas;
  downOnCanvas = false;
  if (wasSuppressed) return;
  if (event.target !== elements.canvas || !wasDownOnCanvas) return;
  if (event.button !== 0) return;
  const dx = event.clientX - downX;
  const dy = event.clientY - downY;
  if (dx * dx + dy * dy > CLICK_PIXEL_TOLERANCE ** 2) return;
  if (!spheres || !spheres.visible) return;
  if (!state.jointPositions) return;
  const rect = elements.canvas.getBoundingClientRect();
  pointerNdc.set(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  );
  raycaster.setFromCamera(pointerNdc, activeCamera);
  const hits = raycaster.intersectObject(spheres);
  if (hits.length > 0 && hits[0].instanceId !== undefined) {
    const joint = hits[0].instanceId;
    const paramIdx = jumpToJointParam(joint);
    selectJoint(joint, paramIdx === null ? null : paramJointMode(paramIdx, joint));
  } else {
    clearJointSelection();
  }
}

// TransformControls can only hide per-axis handles via showX/Y/Z, which still
// leaves plane and free-rotate handles whenever 2+ axes are available. Hide
// those after every helper update so one movement can only ever mean one
// parameter. Hide == ungrabbable: the picker skips invisible handles
// (intersectObjectWithRay filters on visible). Handle names verified against
// three@0.180.0 examples/jsm/controls/TransformControls.js.
function installSingleAxisConstraint() {
  const helper = transformControls.getHelper();
  const baseUpdate = helper.updateMatrixWorld.bind(helper);
  helper.updateMatrixWorld = (force) => {
    baseUpdate(force);
    if (!helper.visible) return;
    // Scale is one scalar (channel 6): uniform center box only — no
    // per-axis length handles, no planes. Every other mode keeps its
    // single axes and drops only multi-axis handles.
    const scalarOnly = transformControls.mode === "scale";
    helper.traverse((obj) => {
      if (!obj.name) return;
      if (scalarOnly) {
        if (obj.name !== "XYZ") obj.visible = false;
        return;
      }
      if (!MULTI_AXIS_HANDLES.test(obj.name)) return;
      obj.visible = false;
    });
  };
}

export function initGizmo() {
  proxy = new THREE.Object3D();
  proxy.visible = false;
  scene.add(proxy);

  transformControls = new TransformControls(activeCamera, elements.canvas);
  transformControls.setSize(GIZMO_SIZE);
  transformControls.setSpace("local");
  transformControls.setMode("rotate");
  transformControls.enabled = false;
  const helper = transformControls.getHelper();
  helper.visible = false;
  scene.add(helper);
  installSingleAxisConstraint();

  transformControls.addEventListener("mouseDown", onGizmoMouseDown);
  transformControls.addEventListener("objectChange", onGizmoObjectChange);
  transformControls.addEventListener("mouseUp", onGizmoMouseUp);

  elements.canvas.addEventListener("pointerdown", onCanvasPointerDown);
  window.addEventListener("pointerup", onPointerUp);

  elements.jointModeMove.addEventListener("click", () => setJointMode("move"));
  elements.jointModeRotate.addEventListener("click", () => setJointMode("rotate"));
  elements.jointModeScale.addEventListener("click", () => setJointMode("scale"));
  elements.jointDeselect.addEventListener("click", () => clearJointSelection());

  registerSphereSync(syncJointSpheres);
  hookCameraRescale();

  addPoseUpdateListener(onPoseEvaluated);

  actorManager.addChangeListener((activeActor) => {
    syncJointSpheres();
    if (selectedJoint >= 0 && !isGizmoDragging) {
      syncProxyFromBone();
    }
    requestRender();
  });
}
