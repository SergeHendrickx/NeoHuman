import * as THREE from "three";
import { currentParameter, currentParameters, elements, isLocked, state } from "./js/state.js";
import { fetchChecked, formatNumber } from "./js/utils.js";
import { createActionLog, createDispatcher } from "./js/actions.js";
import { actorManager, syncDisplayedSkeleton } from "./js/actor.js";
import { initAgentPanel } from "./js/agentPanel.js";
import { initCommandMenu } from "./js/commandMenu.js";
import { initMenubar } from "./js/menubar.js";
import { fetchBinaryAsset } from "./js/cache.js";
import {
  hideLoading,
  setInputsDisabled,
  setStatus,
  showLoading,
  updateModelControls,
  updateModelStatistics,
} from "./js/ui.js";
import {
  buildCategories,
  rebuildParameterList,
  selectGroup,
  showParameter,
} from "./js/params.js";
import {
  activeCamera,
  axesHelper,
  bodyMaterial,
  clearBackgroundPanorama,
  controls,
  fitCamera,
  gridHelper,
  dismissCanvasTooltip,
  pinCanvasTooltip,
  initRenderer,
  loadBackgroundPanorama,
  readRenderExecMs,
  renderNow,
  renderer,
  requestRender,
  resizeRenderer,
  setCameraParams,
  setEffect,
  setEnvironmentIntensity,
  setFollow,
  setFollowProvider,
  setProjection,
  syncCameraInputs,
  syncProjectionUI,
  getEffect,
  showCanvasTooltip,
  skeletonOverlay,
  toggleProjection,
  updateBoneTooltip,
} from "./js/scene.js";
import {
  evaluateBlendshapesClientSide,
  evaluatePoseClientSide,
  parseBlendshapes,
  parsePoseCorrectives,
  parseRigMapping,
  setRiggedMesh,
} from "./js/rig.js";
import {
  syncBlendshapeServer,
  syncGroupParams,
  syncPoseParams,
  syncPoseServer,
} from "./js/sync.js";
import {
  AVATAR_CODE_VERSION,
  packAvatarData,
  unpackAvatarCode,
  validateAvatarData,
} from "./js/avatarCodec.js";
import { drawPerfGraph, recordBandwidthSaved } from "./js/perf.js";
import {
  clearJointSelection,
  handleGroupChanged,
  hasJointSelection,
  initGizmo,
  refreshJointControls,
  revealJointForParam,
  setJointMode,
  syncGizmoCamera,
  updateSphereVisibility,
} from "./js/gizmo.js";
import { fetchMixamoClips, getMixamoTargetRig, loadMixamoClip, setGlobalMatch, setMixamoTargetRig, setSpineBoost } from "./js/mixamo.js";
import { encodePoseId, fetchPoseList, loadPosePreset, retargetPoseToTheta } from "./js/poses.js";
import { CARD_TINT_DEFAULT, DEFAULT_BROW_ID, DEFAULT_LASH_ID, IRIS_PRESETS, STRAND_TINT_DEFAULT, eyesOral } from "./js/eyesOral.js";
import { fetchMhassetList, isCardDecoId } from "./js/mhAssets.js";
import { crowdEyes } from "./js/crowdEyes.js";
import { crowdMouth } from "./js/crowdMouth.js";
import {
  describeExprVector,
  exprVectorWorldSegment,
  initExprVectors,
  pickExprVector,
  setExprVectorHover,
  setExprVectorSelected,
  setExprVectorsVisible,
} from "./js/exprVectors.js";
import { formatDeltaMm } from "./js/exprVectorMath.js";
import { initIdentityGrid, syncIdentityGrid } from "./js/identityGrid.js";
import { shirt, shorts } from "./js/clothing.js";
import {
  addSimulationChangeListener,
  agentCount,
  clearAgents,
  focusSelectedAgent,
  getAgentBehavior,
  getSelectedAgent,
  getSelectedAgentId,
  getShowLodColors,
  getSimulationStats,
  isAgentSimulationActive,
  listAgents,
  pickAgentAtNdc,
  prewarmGpuAgents,
  refreshGpuAgents,
  setSelectionRingVisible,
  removeAgent,
  selectAgent,
  setAgentBehavior,
  setAgentLodPin,
  setAutoLodEnabled,
  getSimulationTimeScale,
  setFrustumCullingEnabled,
  setOneAgentBehavior,
  setShowLodColors,
  setSimulationTimeScale,
  simulation,
  spawnAgents,
  startSimulation,
  stepSimulation,
  toggleAgentSimulation,
} from "./js/crowd.js";
import {
  cancelBenchmark,
  clearRuns,
  detectEngine,
  exportRuns,
  importRuns,
  loadRuns,
  runBenchmark,
  runRampBenchmark,
  saveRun,
} from "./js/bench.js";

// ?noui: hide every UI overlay for clean scene recording (?noui=0 forces off).
const NOUI = (() => {
  try {
    const v = new URLSearchParams(location.search).get("noui");
    return v !== null && !["0", "false", "off", "no"].includes(v.toLowerCase());
  } catch {
    return false;
  }
})();
if (NOUI) document.body.classList.add("noui");

let activeResetAnimationId = null;

function cancelResetAnimation() {
  if (activeResetAnimationId) {
    cancelAnimationFrame(activeResetAnimationId);
    activeResetAnimationId = null;
  }
}

function animateReset(action) {
  cancelResetAnimation();

  let itemsToReset = [];
  const current = currentParameter();
  const actor = actorManager.activeActor;
  // A held pose or playing clip re-applies every tick and would win over the
  // tween below, leaving the character visibly stuck. Pose-covering resets
  // drop the override first (before the neutral early return: held values
  // already read as neutral).
  const coversPose = action === "reset_all"
    || (action === "reset_group" && state.currentGroup === "pose")
    || (action === "reset_current" && !!current && state.metadata.groups.pose.includes(current));
  const droppedOverride = coversPose && !!(actor?.heldPose || actor?.activeClip);
  if (coversPose && actor?.heldPose) releasePosePreset();
  if (coversPose && actor?.activeClip) stopMixamoOnSelected();
  if (action === "reset_current") {
    if (current) itemsToReset = [current];
  } else if (action === "reset_group") {
    itemsToReset = currentParameters();
  } else if (action === "reset_all") {
    itemsToReset = Object.values(state.metadata.groups).flat();
  }

  const targets = itemsToReset
    .filter((item) => !isLocked(item))
    .map((item) => ({
      item,
      start: item.value,
      target: Math.min(item.max, Math.max(item.min, 0)),
    }))
    .filter((entry) => Math.abs(entry.start - entry.target) > 1e-5);

  if (targets.length === 0) {
    if (!droppedOverride) setStatus("Already at neutral rest position.");
    return;
  }

  const hasPose = targets.some((e) => state.metadata.groups.pose.includes(e.item));
  const hasShape = targets.some(
    (e) =>
      state.metadata.groups.identity.includes(e.item) ||
      state.metadata.groups.expression.includes(e.item),
  );

  const duration = 250;
  const startTime = performance.now();

  function easeOutCubic(x) {
    return 1 - Math.pow(1 - x, 3);
  }

  function step(now) {
    const elapsed = now - startTime;
    const progress = Math.min(1.0, elapsed / duration);
    const ease = easeOutCubic(progress);

    for (let i = 0; i < targets.length; i++) {
      const t = targets[i];
      const val = t.start + (t.target - t.start) * ease;
      t.item.value = val;
      if (actor) {
        const group = state.metadata.groups.pose.includes(t.item)
          ? "pose"
          : state.metadata.groups.identity.includes(t.item)
            ? "identity"
            : "expression";
        actor.setParamValue(group, t.item.index, val);
      }
    }

    if (current) {
      showParameter(current);
    }

    if (!actor || actor.isAnchor) {
      if (hasShape) evaluateBlendshapesClientSide();
      if (hasPose) evaluatePoseClientSide();
    } else {
      if (hasPose) actor.evaluatePose();
      refreshGpuAgents();
    }
    requestRender();

    if (progress < 1.0) {
      activeResetAnimationId = requestAnimationFrame(step);
    } else {
      activeResetAnimationId = null;
      for (let i = 0; i < targets.length; i++) {
        const t = targets[i];
        t.item.value = t.target;
        if (actor) {
          const group = state.metadata.groups.pose.includes(t.item)
            ? "pose"
            : state.metadata.groups.identity.includes(t.item)
              ? "identity"
              : "expression";
          actor.setParamValue(group, t.item.index, t.target);
        }
      }
      if (current) {
        showParameter(current);
      }
      if (!actor || actor.isAnchor) {
        if (hasShape) evaluateBlendshapesClientSide();
        if (hasPose) evaluatePoseClientSide();
        fetch("/api/deform", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action,
            group: state.currentGroup,
            index: current ? current.index : 0,
            value: null,
          }),
        }).catch(() => {});
      } else {
        if (hasPose) actor.evaluatePose();
        refreshGpuAgents();
      }
      requestRender();
      setStatus(`Reset completed (${action.replace("_", " ")})`);
    }
  }

  activeResetAnimationId = requestAnimationFrame(step);
}

function commitParameterValue() {
  cancelResetAnimation();
  const value = elements.value.valueAsNumber;
  if (Number.isFinite(value)) {
    const parameter = currentParameter();
    if (!parameter || isLocked(parameter)) return;
    parameter.value = value;
    elements.slider.value = value;
    elements.activeParameter.textContent =
      `${state.currentGroup} / ${parameter.name} = ${formatNumber(value)}`;
    const actor = actorManager.activeActor;
    if (actor) {
      actor.setParamValue(state.currentGroup, parameter.index, value);
    }
    if (!actor || actor.isAnchor) {
      if (state.currentGroup === "pose") {
        evaluatePoseClientSide();
        syncPoseServer(parameter.index, value);
      } else if (state.currentGroup === "identity") {
        evaluateBlendshapesClientSide();
        syncBlendshapeServer("identity", parameter.index, value);
      } else if (state.currentGroup === "expression") {
        evaluateBlendshapesClientSide();
        syncBlendshapeServer("expression", parameter.index, value);
      }
    } else {
      refreshGpuAgents();
      requestRender();
    }
  } else {
    showParameter(currentParameter());
  }
}

function crowdSpawnCount(element, fallback, cap) {
  const parsed = Math.floor(Number(element.value));
  if (!Number.isFinite(parsed) || parsed < 1) return fallback;
  return Math.min(parsed, cap);
}

function actorModeTag(agent) {
  return agent.isGpuAgent ? "GPU" : "CPU";
}

function refreshCrowdTotals() {
  if (elements.crowdSimTotal) {
    const agents = simulation.agents;
    const gpu = agents.filter((a) => a.isGpuAgent).length;
    const mode = agents.length === 0
      ? ""
      : gpu === agents.length
        ? " · GPU"
        : gpu === 0
          ? " · CPU"
          : ` · ${gpu} GPU + ${agents.length - gpu} CPU`;
    elements.crowdSimTotal.textContent = `${agentCount()} active${mode}`;
  }
}

let lastActorsListKey = "";

function renderActorsList() {
  if (!elements.actorsList) return;
  const currentSelected = getSelectedAgentId();
  const key = `${currentSelected}|${simulation.agents.map((a) => `${a.id}:${a.behavior}:${actorModeTag(a)}`).join(",")}`;
  if (key === lastActorsListKey) return;
  lastActorsListKey = key;
  const rows = [
    `<div class="actor-row${currentSelected === 0 ? " is-selected" : ""}" data-id="0" role="option" aria-selected="${currentSelected === 0}">🌟 Actor #0</div>`,
  ];
  for (const agent of simulation.agents) {
    const selected = agent.id === currentSelected;
    rows.push(
      `<div class="actor-row${selected ? " is-selected" : ""}" data-id="${agent.id}" role="option" aria-selected="${selected}">👤 Actor #${agent.id} (${agent.behavior} · ${actorModeTag(agent)})</div>`,
    );
  }
  if (simulation.agents.length === 0) {
    rows.push(
      `<div class="empty-state">
        <p><strong>No crowd yet.</strong> Actor #0 is your hand-editable character; spawn autonomous agents to fill the scene.</p>
        <div class="button-row">
          <button type="button" data-empty="spawn">Spawn 6 agents</button>
          <button type="button" data-empty="crowd">Crowd tab</button>
          <button type="button" data-empty="cmdk" title="Command menu">Commands</button>
        </div>
      </div>`,
    );
  }
  elements.actorsList.innerHTML = rows.join("");
}

// Dock-aware panel activation with a static-strip fallback.
function activatePanel(panelId, wtab) {
  const panel = window.mhr?.dockApi?.getPanel?.(panelId);
  if (panel) {
    panel.api.setActive();
    return;
  }
  document.querySelector(`[data-wtab="${wtab}"]`)?.click();
}

// Framing box for the active actor: null lets fitCamera measure the active
// actor's own joints (actor #0 or agent — face view reads the posed head plus
// the posed face-forward, so neck/head rotation is honoured). Other views
// give agents a position box since there is no posed body metric for them.
function activeActorFocus(view) {
  const actor = actorManager.activeActor;
  if (!actor || actor.isAnchor || !actor.position) return null;
  if (view === "face") {
    syncDisplayedSkeleton();
    return null;
  }
  const s = actor.scale || 1;
  const p = actor.position;
  return { center: new THREE.Vector3(p.x, 0.95 * s, p.z), maxDimension: 1.9 * s };
}

// Landing look: a lopsided smirk with a sideways glance and a tilted head.
// Pushed deliberately hard: it greets playfully and demos the expression +
// pose stack on the first frame. Committed like ordinary edits on the anchor.
const LANDING_SMILE = [
  ["lipCornerPuller_L", 0.95],
  ["lipCornerPuller_R", 0.05],
  ["lipCornerDepressor_R", 0.3],
  ["dimpler_L", 0.5],
  ["cheekRaiser_L", 0.5],
  ["upperLipRaiser_L", 0.3],
  ["outerBrowRaiser_L", 0.7],
  ["lidTightener_L", 0.25],
];
// Head roll toward his right + a touch of turn toward the smirk (bend is
// about +Z, lean about +Y; screen-right is his left, so bend+ rolls right).
const LANDING_POSE = [
  ["head_bend", 0.18],
  ["neck_bend", 0.1],
  ["head_lean", 0.12],
  ["neck_lean", 0.08],
];
function applyLandingLook() {
  const actor = actorManager.activeActor;
  const expr = state.metadata?.groups?.expression ?? [];
  const pose = state.metadata?.groups?.pose ?? [];
  let touchedExpr = false;
  for (const [name, value] of LANDING_SMILE) {
    const item = expr.find((p) => p.name === name);
    if (!item || isLocked(item)) continue;
    item.value = value;
    actor?.setParamValue("expression", item.index, value);
    touchedExpr = true;
  }
  let touchedPose = false;
  for (const [name, value] of LANDING_POSE) {
    const item = pose.find((p) => p.name === name);
    if (!item || isLocked(item)) continue;
    item.value = value;
    actor?.setParamValue("pose", item.index, value);
    touchedPose = true;
  }
  if (!touchedExpr && !touchedPose) return;
  if (!actor || actor.isAnchor) {
    if (touchedExpr) {
      evaluateBlendshapesClientSide();
      syncGroupParams("expression", expr.map((p) => p.value));
    }
    if (touchedPose) {
      evaluatePoseClientSide();
      syncGroupParams("pose", pose.map((p) => p.value));
    }
  } else {
    if (touchedPose) actor.evaluatePose();
    refreshGpuAgents();
  }
}

// "Wipe that smile off my face": neutral expression + straight head +
// centered gaze, so newcomers can start from a clean face. Identity and
// body pose are left alone; the server mirrors via one group reset plus
// the head-joint batch.
const WIPE_HEAD_POSE = ["head_bend", "neck_bend", "head_lean", "neck_lean"];
function wipeLandingLook() {
  const actor = actorManager.activeActor;
  const expr = state.metadata?.groups?.expression ?? [];
  const pose = state.metadata?.groups?.pose ?? [];
  for (const item of expr) {
    if (isLocked(item)) continue;
    const target = Math.min(item.max, Math.max(item.min, 0));
    item.value = target;
    actor?.setParamValue("expression", item.index, target);
  }
  const headIdx = [];
  for (const name of WIPE_HEAD_POSE) {
    const item = pose.find((p) => p.name === name);
    if (!item || isLocked(item)) continue;
    item.value = 0;
    actor?.setParamValue("pose", item.index, 0);
    headIdx.push(item.index);
  }
  if (!actor || actor.isAnchor) {
    evaluateBlendshapesClientSide();
    evaluatePoseClientSide();
    fetch("/api/deform", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "reset_group",
        group: "expression",
        index: 0,
        value: null,
      }),
    }).catch(() => {});
    if (headIdx.length) syncPoseParams(headIdx);
  } else {
    actor.evaluatePose();
    refreshGpuAgents();
  }
  setActiveGaze(0, 0);
  if (elements.gazeYaw) {
    elements.gazeYaw.value = 0;
    elements.gazeYawVal.textContent = "0°";
  }
  if (elements.gazePitch) {
    elements.gazePitch.value = 0;
    elements.gazePitchVal.textContent = "0°";
  }
  showParameter(currentParameter());
  requestRender();
  setStatus("Clean face: expression neutral, head straight, gaze centered.");
}

// ---- Shareable NeoHuman codes ----
// Full params + anchor look, JSON -> deflate -> base64url. Params apply to
// any active actor; the rest of the look is anchor-only like its UI.
function groupCounts() {
  const groups = state.metadata?.groups ?? {};
  return {
    identity: groups.identity?.length ?? 0,
    expression: groups.expression?.length ?? 0,
    pose: groups.pose?.length ?? 0,
  };
}
function collectAvatarState(actor) {
  const groups = state.metadata.groups;
  const r4 = (v) => Math.round(Number(v) * 10000) / 10000;
  const take = (name) => groups[name].map((p) => r4(actor.getParamValue(name, p.index)));
  const num = (el) => (el ? Number(el.value) : 0);
  return {
    v: AVATAR_CODE_VERSION,
    app: "neohuman",
    params: { identity: take("identity"), expression: take("expression"), pose: take("pose") },
    scale: r4(actor.scale || 1),
    look: {
      skin: elements.bodyColor?.value ?? null,
      pupil: r4(num(elements.eyesPupil)),
      irisIn: elements.eyesIrisIn?.value ?? null,
      irisOut: elements.eyesIrisOut?.value ?? null,
      gaze: [num(elements.gazeYaw), num(elements.gazePitch)],
      brow: { on: !!elements.decoBrowOn?.checked, id: eyesOral.deco.browId ?? null, tint: elements.decoBrowTint?.value ?? null },
      lash: { on: !!elements.decoLashOn?.checked, id: eyesOral.deco.lashId ?? null, tint: elements.decoLashTint?.value ?? null },
      mouth: r4(num(elements.mouthOpen)),
      teeth: {
        fitted: !!elements.fittedTeeth?.checked,
        size: r4(num(elements.teethSize)),
        count: Math.round(num(elements.teethCount)),
        crooked: r4(num(elements.teethCrooked)),
        gap: r4(num(elements.teethGap)),
        overbite: r4(num(elements.teethOverbite)),
        fangs: r4(num(elements.teethFangs)),
      },
      tongue: {
        out: r4(num(elements.tongueOut)),
        curl: r4(num(elements.tongueCurl)),
        width: r4(num(elements.tongueWidth)),
        roll: r4(num(elements.tongueRoll)),
      },
      shirt: { on: shirt.enabled, color: elements.clothColor?.value ?? null },
      shorts: { on: shorts.enabled, color: elements.clothShortsColor?.value ?? null },
      thickness: r4(num(elements.clothThickness)),
    },
  };
}
// Push a value through a control exactly like a user edit (both events: some
// bindings listen to input, checkboxes to change).
function setAvatarControl(el, value, checkbox = false) {
  if (!el || value === undefined || value === null) return;
  if (checkbox) {
    el.checked = value !== false && value !== 0 && value !== "false";
    el.dispatchEvent(new Event("change"));
  } else {
    el.value = value;
    el.dispatchEvent(new Event("input"));
    el.dispatchEvent(new Event("change"));
  }
}
async function applyAvatarState(actor, data) {
  const groups = state.metadata.groups;
  for (const name of ["identity", "expression", "pose"]) {
    const items = groups[name];
    const values = data.params[name];
    for (let i = 0; i < items.length; i++) {
      if (isLocked(items[i])) continue;
      items[i].value = values[i];
      actor.setParamValue(name, items[i].index, values[i]);
    }
  }
  actor.setScale(data.scale || 1);
  if (!actor || actor.isAnchor) {
    evaluateBlendshapesClientSide();
    evaluatePoseClientSide();
    await syncGroupParams("identity", groups.identity.map((p) => p.value));
    await syncGroupParams("expression", groups.expression.map((p) => p.value));
    await syncGroupParams("pose", groups.pose.map((p) => p.value));
  } else {
    actor.evaluatePose();
    refreshGpuAgents();
  }
  const look = data.look ?? {};
  setActiveGaze(look.gaze?.[0] ?? 0, look.gaze?.[1] ?? 0);
  if (elements.gazeYaw) {
    elements.gazeYaw.value = look.gaze?.[0] ?? 0;
    elements.gazePitch.value = look.gaze?.[1] ?? 0;
    elements.gazeYawVal.textContent = `${look.gaze?.[0] ?? 0}°`;
    elements.gazePitchVal.textContent = `${look.gaze?.[1] ?? 0}°`;
  }
  setAvatarControl(elements.eyesPupil, look.pupil);
  let lookNote = "";
  if (!actor.isAnchor) {
    lookNote = " (face detail is actor-#0-only)";
  } else {
    setAvatarControl(elements.bodyColor, look.skin);
    setAvatarControl(elements.eyesIrisIn, look.irisIn);
    setAvatarControl(elements.eyesIrisOut, look.irisOut);
    for (const [slot, want, onEl, tintEl, list] of [
      ["brow", look.brow, elements.decoBrowOn, elements.decoBrowTint, decoBrowList],
      ["lash", look.lash, elements.decoLashOn, elements.decoLashTint, decoLashList],
    ]) {
      if (!want) continue;
      if (want.id && list.some((a) => a.id === want.id)) await applyDecoStyle(slot, want.id);
      setAvatarControl(tintEl, want.tint);
      setAvatarControl(onEl, want.on, true);
    }
    setAvatarControl(elements.mouthOpen, look.mouth);
    setAvatarControl(elements.fittedTeeth, look.teeth?.fitted, true);
    setAvatarControl(elements.teethSize, look.teeth?.size);
    setAvatarControl(elements.teethCount, look.teeth?.count);
    setAvatarControl(elements.teethCrooked, look.teeth?.crooked);
    setAvatarControl(elements.teethGap, look.teeth?.gap);
    setAvatarControl(elements.teethOverbite, look.teeth?.overbite);
    setAvatarControl(elements.teethFangs, look.teeth?.fangs);
    setAvatarControl(elements.tongueOut, look.tongue?.out);
    setAvatarControl(elements.tongueCurl, look.tongue?.curl);
    setAvatarControl(elements.tongueWidth, look.tongue?.width);
    setAvatarControl(elements.tongueRoll, look.tongue?.roll);
    if (look.shirt) {
      shirt.setEnabled(look.shirt.on !== false);
      setSwitchButton(elements.clothToggle, elements.clothToggleState, shirt.enabled);
      if (look.shirt.color && elements.clothColor) {
        elements.clothColor.value = look.shirt.color;
        shirt.setColor(look.shirt.color);
      }
    }
    if (look.shorts) {
      shorts.setEnabled(look.shorts.on !== false);
      setSwitchButton(elements.clothShortsToggle, elements.clothShortsToggleState, shorts.enabled);
      if (look.shorts.color && elements.clothShortsColor) {
        elements.clothShortsColor.value = look.shorts.color;
        shorts.setColor(look.shorts.color);
      }
    }
    setAvatarControl(elements.clothThickness, look.thickness);
    if (shirt.enabled) shirt.refit();
    if (shorts.enabled) shorts.refit();
  }
  if (elements.crowdActorScale) {
    elements.crowdActorScale.value = (data.scale || 1).toFixed(2);
    elements.crowdActorScaleVal.textContent = `${(data.scale || 1).toFixed(2)}x`;
  }
  refreshActorCard();
  syncIdentityGrid();
  showParameter(currentParameter());
  requestRender();
  setStatus(`Avatar applied to ${actor.name}${lookNote}.`);
}
function openAvatarDialog(text = "", select = false) {
  const dlg = document.getElementById("dlg-avatar");
  const area = document.getElementById("avatar-code-text");
  const errEl = document.getElementById("avatar-code-error");
  if (!dlg || !area) return;
  area.value = text;
  if (errEl) errEl.textContent = "";
  if (!dlg.open) dlg.showModal();
  if (select) { area.focus(); area.select(); } else { area.focus(); }
}
async function copyAvatarCode() {
  const actor = actorManager.activeActor;
  if (!actor) { setStatus("Select a character first.", "error"); return; }
  try {
    const code = await packAvatarData(collectAvatarState(actor));
    try {
      await navigator.clipboard.writeText(code);
      setStatus(`Avatar code copied (${code.length} chars).`);
    } catch {
      openAvatarDialog(code, true);
      setStatus("Clipboard blocked — copy the code from the dialog.");
    }
  } catch (err) {
    setStatus(`Copy failed: ${err.message}`, "error");
  }
}
async function importAvatarCode() {
  const actor = actorManager.activeActor;
  if (!actor) { setStatus("Select a character first.", "error"); return; }
  openAvatarDialog();
}
async function applyAvatarDialog() {
  const actor = actorManager.activeActor;
  const area = document.getElementById("avatar-code-text");
  const errEl = document.getElementById("avatar-code-error");
  const dlg = document.getElementById("dlg-avatar");
  if (!actor || !area) return;
  try {
    const data = validateAvatarData(await unpackAvatarCode(area.value.trim()), groupCounts());
    await applyAvatarState(actor, data);
    dlg?.close();
  } catch (err) {
    if (errEl) errEl.textContent = err.message;
  }
}
async function applyUrlAvatar() {
  let code = null;
  try { code = new URLSearchParams(location.search).get("avatar"); } catch { return; }
  if (!code) return;
  const actor = actorManager.activeActor;
  if (!actor) return;
  try {
    const data = validateAvatarData(await unpackAvatarCode(code), groupCounts());
    await applyAvatarState(actor, data);
    setStatus("Avatar loaded from link.");
  } catch (err) {
    setStatus(`Avatar link invalid: ${err.message}`, "error");
  }
}

function setDisabledWithReason(el, disabled, reason) {
  if (!el) return;
  if (el.dataset.origTitle === undefined) el.dataset.origTitle = el.title || "";
  el.disabled = Boolean(disabled);
  el.title = disabled ? reason : el.dataset.origTitle;
}

// Syncs the Actor card to the active actor (actor #0 included). Runs on every
// sim notification, so slider writes skip the focused control — otherwise a
// drag would snap back mid-gesture. Controls that cannot apply to the
// current target disable with the reason inline.
function refreshActorCard() {
  const actor = actorManager.activeActor;
  if (!actor) return;
  const isAnchor = actor.isAnchor;
  const isGPU = !isAnchor && actor.isGpuAgent;
  const isCPU = !isAnchor && !actor.isGpuAgent;

  if (isAnchor) {
    elements.crowdActorTitle.textContent = "🌟 Actor #0";
    elements.crowdActorStateBadge.textContent = "Manual";
    elements.crowdActorStateBadge.className = "perf-badge is-muted";
  } else {
    const lod = actor.isGpuAgent ? `LOD ${actor.lod}` : actor.lodName.toUpperCase();
    elements.crowdActorTitle.textContent = `Actor #${actor.id} · ${actorModeTag(actor)} · ${lod}`;
    const isMoving = actor.currentSpeed > 0.08;
    elements.crowdActorStateBadge.textContent =
      actor.behavior === "idle" ? "Idle" : isMoving ? "Walking" : "Stationary";
    elements.crowdActorStateBadge.className = isMoving
      ? "perf-badge is-accent"
      : "perf-badge is-muted";
  }
  setDisabledWithReason(elements.crowdActorRemove, isAnchor, "Actor #0 cannot be removed");

  // Identity (single body-color picker for every target).
  if (document.activeElement !== elements.bodyColor && actor.color) {
    elements.bodyColor.value = `#${actor.color.getHexString()}`;
  }
  if (document.activeElement !== elements.crowdActorScale) {
    elements.crowdActorScale.value = actor.scale.toFixed(2);
    elements.crowdActorScaleVal.textContent = `${actor.scale.toFixed(2)}x`;
  }
  if (elements.crowdActorLod && document.activeElement !== elements.crowdActorLod) {
    elements.crowdActorLod.value = isGPU ? String(actor.lodPin ?? "auto") : "auto";
  }
  setDisabledWithReason(elements.crowdActorLod, !isGPU,
    isAnchor ? "Actor #0 has no LOD pool" : "LOD pinning is GPU-only");

  // Face (retargets: actor #0 via eyesOral, agents via crowd eyes/weights).
  const gaze = isAnchor ? eyesOral.gaze : crowdEyes.getAgentGaze(actor.id) || { yaw: 0, pitch: 0 };
  if (document.activeElement !== elements.gazeYaw) {
    elements.gazeYaw.value = gaze.yaw;
    elements.gazeYawVal.textContent = `${Math.round(gaze.yaw)}°`;
  }
  if (document.activeElement !== elements.gazePitch) {
    elements.gazePitch.value = gaze.pitch;
    elements.gazePitchVal.textContent = `${Math.round(gaze.pitch)}°`;
  }
  elements.actorGazeRow.hidden = isAnchor;
  if (!isAnchor && document.activeElement !== elements.eyesPupil) {
    elements.eyesPupil.value = actor.pupil ?? 0.45;
    elements.eyesPupilVal.textContent = Number(actor.pupil ?? 0.45).toFixed(2);
  }
  setDisabledWithReason(elements.eyesBlink, isCPU, "CPU agents share actor #0's mesh and cannot blink");
  const jawIdx = crowdEyes.jawDropIndex();
  const mouthVal = isAnchor ? eyesOral.jawDrop
    : isGPU && jawIdx >= 0 && actor.expressionValues.length > jawIdx
      ? actor.expressionValues[jawIdx]
      : 0;
  if (document.activeElement !== elements.mouthOpen) {
    elements.mouthOpen.value = mouthVal;
    elements.mouthOpenVal.textContent = Number(mouthVal).toFixed(2);
  }
  setDisabledWithReason(elements.mouthOpen, isCPU, "CPU agents share actor #0's mesh and cannot open the mouth");
  setSwitchButton(elements.eyesAutoblink, elements.eyesAutoblinkState, eyesOral.autoBlink);
  setDisabledWithReason(elements.eyesAutoblink, !isAnchor, "Auto-blink is actor-#0-only (crowd blink lives in World → Agents)");
  elements.actorFaceNote.hidden = !isCPU;
  // Face detail: Actor-#0-only hardware (module-owned eyeballs, teeth, tongue).
  // Pupil is per-actor (agents read it into the instanced eyes).
  for (const el of [
    elements.eyesIrisIn, elements.eyesIrisOut,
    elements.teethCount, elements.teethCrooked, elements.teethGap, elements.teethSize,
    elements.teethOverbite, elements.teethFangs,
    elements.tongueOut, elements.tongueCurl, elements.tongueWidth, elements.tongueRoll,
  ]) {
    setDisabledWithReason(el, !isAnchor, "Actor-#0-only face detail");
  }

  // Body (garments actor-#0-only; wireframe works for every target).
  const garments = [
    elements.clothToggle, elements.clothColor, elements.clothShortsToggle,
    elements.clothShortsColor, elements.clothThickness, elements.clothRefit,
  ];
  for (const el of garments) {
    setDisabledWithReason(el, !isAnchor, "Garments are actor-#0-only for now");
  }
  elements.actorBodyNote.hidden = isAnchor;
  if (actor.material) {
    elements.wireframe.checked = Boolean(actor.material.wireframe);
  }

  // Motion (agents only; the anchor is posed by hand).
  elements.actorAiNote.hidden = !isAnchor;
  const motionReason = "Actor #0 is posed by hand; steering applies to spawned agents";
  setDisabledWithReason(elements.crowdActorBehavior, isAnchor, motionReason);
  setDisabledWithReason(elements.crowdActorSpeed, isAnchor, motionReason);
  setDisabledWithReason(elements.crowdActorCadence, isAnchor, motionReason);
  if (!isAnchor) {
    if (document.activeElement !== elements.crowdActorBehavior) {
      elements.crowdActorBehavior.value = actor.behavior;
    }
    if (document.activeElement !== elements.crowdActorSpeed) {
      elements.crowdActorSpeed.value = actor.maxSpeed.toFixed(1);
      elements.crowdActorSpeedVal.textContent = `${actor.maxSpeed.toFixed(1)} m/s`;
    }
    if (document.activeElement !== elements.crowdActorCadence) {
      elements.crowdActorCadence.value = actor.cadence.toFixed(1);
      elements.crowdActorCadenceVal.textContent = actor.cadence.toFixed(1);
    }
  }
  elements.crowdActorPos.textContent = `Pos: ${actor.position.x.toFixed(2)}, ${actor.position.z.toFixed(2)}`;
  elements.crowdActorCurspeed.textContent = `Speed: ${actor.currentSpeed.toFixed(2)} m/s`;
}

let lastCamFocusId = null;

function onActiveActorChanged(actor) {
  if (!actor) return;
  renderActorsList();
  syncPoseGridSelection();
  syncDisplayedSkeleton();
  syncIdentityGrid(); if (actor.id !== lastCamFocusId) {
    // Genuine selection change (register/unregister notifies keep the id):
    // jump the orbit camera onto the newly selected actor. Skeleton sync
    // runs first so face view measures the NEW actor's posed joints.
    lastCamFocusId = actor.id;
    fitCamera(state.lastView, activeActorFocus(state.lastView));
  }
  refreshActorCard();
  if (state.metadata) {
    const current = currentParameter();
    if (current) {
      showParameter(current);
    }
  }
  requestRender();
}

function refreshSimMetrics() {
  if (!elements.crowdSimFrame || !elements.crowdSimTime) return;
  const stats = getSimulationStats();
  elements.crowdSimFrame.textContent = `Frame: ${stats.frameIndex}`;
  elements.crowdSimTime.textContent = `${stats.simTime.toFixed(2)}s`;
}

let lastStatsTableKey = "";

// Stats panel refresh: off by default (checkbox), throttled to ~4Hz when on.
// The panel (headline + 600-row table + theta sums) is the dominant `ui`
// cost at scale; user actions (boot, LOD switch, pin, Refresh button) force.
let crowdStatsAutorefresh = false;
let lastStatsRefreshFrame = -1000;
const STATS_REFRESH_EVERY = 15;

function refreshCrowdStats(force = false) {
  if (!elements.crowdStatsBody) return;
  if (!force) {
    if (!crowdStatsAutorefresh) return;
    if (simulation.frameIndex - lastStatsRefreshFrame < STATS_REFRESH_EVERY) return;
  }
  lastStatsRefreshFrame = simulation.frameIndex;
  const stats = getSimulationStats();
  const agents = simulation.agents;
  let gpuCount = 0;
  const lodMix = new Map();
  for (const agent of agents) {
    if (!agent.isGpuAgent) continue;
    gpuCount++;
    lodMix.set(agent.lod, (lodMix.get(agent.lod) || 0) + 1);
  }
  const cpuCount = agents.length - gpuCount;
  const tickMs = stats.timings.agentMs + stats.timings.gpuMs + stats.timings.uiMs;
  elements.crowdStatsHeadline.textContent = agents.length === 0
    ? `idle · ${stats.gpu.mode === "gpu" ? "GPU ready" : "CPU"}`
    : `${agents.length} agents · ${gpuCount} GPU + ${cpuCount} CPU · tick ${tickMs.toFixed(2)}ms`;
  const lod = state.metadata?.model?.lod ?? "?";
  elements.crowdStatsEngine.textContent = `${state.isWebGPU ? "WebGPU" : "WebGL"} · LOD ${lod}`;
  const autoTag = stats.gpu.autoLod ? "auto" : "manual";
  const pending = stats.gpu.pendingMigrations || 0;
  elements.crowdStatsProvider.textContent = stats.gpu.mode === "gpu"
    ? `${autoTag} · ready · ${stats.gpu.dims.J} joints · ${stats.gpu.dims.B} shapes` +
      (pending > 0 ? ` · ${pending} moves pending` : "")
    : `CPU fallback · ${stats.gpu.reason}`;
  if (elements.crowdStatsPools && stats.gpu.pools) {
    elements.crowdStatsPools.textContent = stats.gpu.pools.map((p) => {
      if (p.status === "ready") {
        const kv = p.verts >= 1000 ? `${(p.verts / 1000).toFixed(1)}k` : `${p.verts}`;
        return `L${p.lod} ${p.used}/${p.slots}·${kv}v${p.failed ? " FROZEN" : ""}`;
      }
      if (p.status === "loading") return `L${p.lod} …`;
      if (p.status === "failed") return `L${p.lod} FAIL`;
      return `L${p.lod} —`;
    }).join("  ");
    const failedPool = stats.gpu.pools.find((p) => p.status === "failed" && p.error);
    elements.crowdStatsPools.title = failedPool
      ? `L${failedPool.lod}: ${failedPool.error}`
      : "Per-LOD GPU pool: used/slots · verts. Pools warm lazily; migration waits for the target.";
  }
  elements.crowdStatsTick.textContent =
    `${tickMs.toFixed(2)} ms (agents ${stats.timings.agentMs.toFixed(2)} · post ${stats.timings.gpuMs.toFixed(2)} · ui ${stats.timings.uiMs.toFixed(2)})`;
  elements.crowdStatsTick.title =
    `CPU cost only. post = probe ${stats.timings.probeMs.toFixed(2)} + fill ${stats.timings.fillMs.toFixed(2)} + queue ${stats.timings.queueMs.toFixed(2)} ms. ` +
    "GPU execution is async; see GPU exec row.";
  const pairs = stats.timings.pairCount;
  let uploadKb = 0;
  if (pairs > 0 && stats.gpu.mode === "gpu") {
    uploadKb = (pairs * ((stats.gpu.dims.J * 7 + stats.gpu.dims.B) * 4 + 4)) / 1024;
  }
  const activePools = stats.gpu.pools ? stats.gpu.pools.filter((p) => p.used > 0).length : 0;
  const dispatches = Math.max(1, activePools) * 2;
  elements.crowdStatsDispatch.textContent = pairs > 0
    ? `${dispatches} dispatches in ${dispatches} submits (${activePools} pools × FK+skin over ${pairs} agents) · upload ${uploadKb.toFixed(1)} KB`
    : "—";
  if (!state.isWebGPU) {
    elements.crowdStatsGpuexec.textContent = "—";
  } else if (renderer?.backend?.trackTimestamp !== true) {
    elements.crowdStatsGpuexec.textContent = "n/a (no timestamp-query support)";
  } else {
    const c = pairs > 0 ? stats.timings.gpuExecMs : -1;
    const cText = c === -1 ? "—" : c === null ? "…" : c.toFixed(2);
    const r = readRenderExecMs();
    const rText = r === null ? "…" : r.toFixed(2);
    elements.crowdStatsGpuexec.textContent = `compute ${cText} · render ${rText} ms/frame`;
  }
  if (elements.crowdStatsLodmix) {
    const parts = [...lodMix.entries()].sort((a, b) => a[0] - b[0])
      .map(([lod, n]) => `L${lod}:${n}`);
    if (cpuCount > 0) parts.push(`CPU:${cpuCount}`);
    elements.crowdStatsLodmix.textContent = parts.length > 0 ? parts.join("  ") : "—";
  }
  if (elements.crowdStatsLodmoves) {
    const blocked = stats.gpu.blockedMigrations || 0;
    const queued = Math.max(0, stats.gpu.pendingMigrations - blocked);
    elements.crowdStatsLodmoves.textContent =
      `pending ${stats.gpu.pendingMigrations} (${queued} queued · ${blocked} blocked) · ` +
      `+${stats.gpu.migratedThisTick} last tick · ${stats.gpu.movedTotal} total · ${stats.gpu.migrateFails} failed`;
  }
  if (elements.crowdStatsFrustum) {
    const culled = stats.gpu.culled || 0;
    elements.crowdStatsFrustum.textContent = stats.gpu.frustumEnabled
      ? `on · ${agents.length - culled} visible / ${culled} culled`
      : `off · ${agents.length} visible`;
  }
  if (elements.crowdStatsSpawn) {
    const s = stats.gpu.lastSpawn;
    elements.crowdStatsSpawn.textContent = !s
      ? "—"
      : `${s.added} (${s.gpu}G+${s.cpu}C) · ${s.wallMs.toFixed(0)}ms wall / ` +
        `${s.constructMs.toFixed(0)} construct / ${s.notifyMs.toFixed(0)} notify / ` +
        `${s.rafWaitMs.toFixed(0)} raf-wait / ${s.batches} batches`;
    elements.crowdStatsSpawn.title = !s
      ? "No spawn yet this session."
      : `LOD mix: ${s.mix || "—"} · slowest batch ${s.slowestBatchMs.toFixed(1)}ms. raf-wait is inter-batch pacing, not work.`;
  }
  const every = agents.length <= 40 ? 5 : agents.length <= 150 ? 15 : 30;
  const key = `${getSelectedAgentId()}|${agents.length}:${agents.map((agent) => agent.id).join(",")}:${Math.floor(stats.frameIndex / every)}`;
  if (key === lastStatsTableKey) return;
  lastStatsTableKey = key;
  if (agents.length === 0) {
    elements.crowdStatsBody.innerHTML = `<tr><td colspan="9" class="quiet">No agents.</td></tr>`;
    return;
  }
  const selectedId = getSelectedAgentId();
  elements.crowdStatsBody.innerHTML = agents.map((agent) => {
    let thetaSum = 0;
    const theta = agent.thetaJ;
    for (let i = 0; i < theta.length; i++) thetaSum += theta[i];
    const lodCell = agent.isGpuAgent ? `L${agent.lod}${agent.lodPin !== null && agent.lodPin !== undefined ? "📌" : ""}` : "—";
    const sel = agent.id === selectedId ? ` class="is-selected"` : "";
    return (
      `<tr data-id="${agent.id}"${sel}><td>${agent.id}</td><td>${actorModeTag(agent)}</td>` +
      `<td>${agent.isGpuAgent ? agent._record?.slot ?? "—" : "—"}</td>` +
      `<td>${lodCell}</td>` +
      `<td>${agent.behavior}</td><td>${agent.currentSpeed.toFixed(2)}</td>` +
      `<td>${agent.gaitPhase.toFixed(2)}</td><td>${thetaSum.toFixed(3)}</td>` +
      `<td>${agent.position.x.toFixed(1)}, ${agent.position.z.toFixed(1)}</td></tr>`
    );
  }).join("");
}

// ---- Motion clips (Mixamo): any selected character performs a clip ----
let mixamoClipList = [];
let posePresetList = [];
let jointIndexCache = null;
let anchorClipRaf = 0;
let anchorClipLast = 0;
let mixamoInspectTimer = 0;
let lastMixamoInspectName = "";

async function refreshMixamoList() {
  if (!elements.mixamoClip) return;
  try {
    const data = await fetchMixamoClips(true);
    mixamoClipList = data.clips;
    if (elements.mixamoCount) {
      elements.mixamoCount.textContent = data.available
        ? `${data.clips.length} clips`
        : "folder missing";
    }
    populateMixamoSelect();
    updateMixamoStatus();
  } catch (error) {
    elements.mixamoClip.innerHTML = `<option value="">List failed</option>`;
    updateMixamoStatus(`Clip list failed: ${error.message}`);
  }
}

function populateMixamoSelect() {
  const select = elements.mixamoClip;
  if (!select) return;
  const previous = select.value;
  const query = (elements.mixamoFilter?.value || "").toLowerCase();
  const shown = mixamoClipList
    .filter((clip) => clip.name.toLowerCase().includes(query))
    .slice(0, 300);
  select.replaceChildren();
  if (shown.length === 0) {
    const label = mixamoClipList.length === 0 && !query
      ? "No clips — drop .fbx files into data/mixamo"
      : "No matches";
    select.append(new Option(label, ""));
    return;
  }
  for (const clip of shown) {
    select.append(new Option(`${clip.name} (${(clip.bytes / 1024).toFixed(0)} KB)`, clip.name));
  }
  if (shown.some((clip) => clip.name === previous)) select.value = previous;
  scheduleMixamoInspect();
}

function stopActorClip(actor) {
  if (actor?.activeClip) {
    actor.activeClip.dispose();
    actor.activeClip = null;
    actor.restoreBasePose?.();
  }
}

function jointIndexByName() {
  if (!jointIndexCache && state.metadata?.jointNames) {
    jointIndexCache = new Map(state.metadata.jointNames.map((name, index) => [name, index]));
  }
  return jointIndexCache ?? new Map();
}

async function refreshPoseGrid() {
  const grid = elements.poseGrid;
  if (!grid) return;
  let poses = [];
  try {
    poses = await fetchPoseList();
  } catch {
    // Server folder missing: leave a quiet note, sliders still work.
  }
  posePresetList = poses;
  if (elements.posePresetCount) {
    elements.posePresetCount.textContent = poses.length === 0
      ? "No poses — drop .bvh files into data/makehuman"
      : `${poses.length} poses`;
  }
  renderPoseTiles();
}

function poseBadgeText(license) {
  const norm = (license || "").replace(/[^a-z0-9]/gi, "").toLowerCase();
  if (norm === "cc0") return "";
  if (norm.includes("by")) return "BY";
  return "?";
}

function renderPoseTiles() {
  const grid = elements.poseGrid;
  if (!grid) return;
  const query = (elements.poseFilter?.value || "").toLowerCase();
  grid.replaceChildren();
  const shown = posePresetList.filter((pose) =>
    !query || pose.name.toLowerCase().includes(query) || pose.set.toLowerCase().includes(query),
  );
  if (shown.length === 0) {
    const note = document.createElement("span");
    note.className = "quiet";
    note.textContent = posePresetList.length === 0 ? "No poses found." : "No poses match.";
    grid.append(note);
    return;
  }
  for (const pose of shown) {
    const tile = document.createElement("div");
    tile.className = "hdri-thumb pose-thumb";
    tile.dataset.poseId = pose.id;
    tile.title = `${pose.name} — ${pose.license || "license unknown"}\nRight-click for license details`;
    tile.tabIndex = 0;
    tile.setAttribute("role", "option");
    tile.setAttribute("aria-selected", "false");
    const img = document.createElement("img");
    img.src = `/api/pose/thumb/${encodePoseId(pose.id)}`;
    img.alt = "";
    img.loading = "lazy";
    img.addEventListener("error", () => img.remove());
    const caption = document.createElement("span");
    caption.textContent = pose.name;
    tile.append(img, caption);
    const badge = poseBadgeText(pose.license);
    if (badge) {
      const tag = document.createElement("b");
      tag.className = "pose-badge";
      tag.textContent = badge;
      tag.title = pose.license || "Unknown license — check before reuse";
      tile.append(tag);
    }
    tile.addEventListener("click", () => applyPosePreset(pose.id));
    tile.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        applyPosePreset(pose.id);
      }
    });
    tile.addEventListener("contextmenu", (event) => {
      event.preventDefault();
      showPoseLicense(event.clientX, event.clientY, pose);
    });
    grid.append(tile);
  }
  syncPoseGridSelection();
}

function syncPoseGridSelection() {
  const grid = elements.poseGrid;
  if (!grid) return;
  const held = actorManager.activeActor?.heldPose?.id;
  for (const tile of grid.querySelectorAll(".pose-thumb")) {
    const selected = tile.dataset.poseId === held;
    tile.classList.toggle("is-selected", selected);
    tile.setAttribute("aria-selected", String(selected));
  }
}

const DEV_WTABS = ["performance", "crowd", "actors"];
const DEV_ATABS = ["ai"];
let setDevPanelsVisibleFn = null;

function isDevTools() {
  try {
    return localStorage.getItem("mhr.devTools") === "1";
  } catch {
    return false;
  }
}

function setDevTools(on) {
  try {
    localStorage.setItem("mhr.devTools", on ? "1" : "0");
  } catch {
    // Private mode etc: stay ephemeral.
  }
  applyDevTools();
}

// Avatar-first default view: technical world tabs + telemetry hide until
// Developer tools are enabled. Covers both the dock layout (panels) and
// the static fallback (strips + telemetry section).
function applyDevTools() {
  const dev = isDevTools();
  if (elements.devTools) elements.devTools.checked = dev;
  try {
    setDevPanelsVisibleFn?.(dev);
  } catch (err) {
    console.warn("[devtools]", err?.message ?? err);
  }
  if (elements.worldTabs) {
    for (const name of DEV_WTABS) {
      elements.worldTabs.querySelector(`[data-wtab="${name}"]`)
        ?.toggleAttribute("hidden", !dev);
    }
    const active = elements.worldTabs.querySelector("[data-wtab].is-active");
    if (!dev && active && DEV_WTABS.includes(active.dataset.wtab)) {
      elements.worldTabs.querySelector('[data-wtab="world"]')?.click();
    }
  }
  if (elements.actorTabs) {
    for (const name of DEV_ATABS) {
      elements.actorTabs.querySelector(`[data-atab="${name}"]`)
        ?.toggleAttribute("hidden", !dev);
    }
    const active = elements.actorTabs.querySelector("[data-atab].is-active");
    if (!dev && active && DEV_ATABS.includes(active.dataset.atab)) {
      switchActorTab("identity");
    }
  }
  if (elements.telemetryPane) elements.telemetryPane.hidden = !dev;
  // Simulation transport is a dev instrument: hidden until Developer tools.
  for (const el of [elements.crowdSimPlay, elements.crowdSimStep, elements.crowdSimSkip]) {
    if (el) el.hidden = !dev;
  }
  elements.crowdSimSpeed?.closest("label")?.toggleAttribute("hidden", !dev);
}

function showWelcomeIfFirstRun() {
  let seen = true;
  try {
    seen = localStorage.getItem("mhr.welcomed") === "1";
  } catch {
    // Private mode: never nag.
  }
  if (seen || !elements.welcomeOverlay) return;
  elements.welcomeOverlay.hidden = false;
}

function dismissWelcome(enableDev) {
  try {
    localStorage.setItem("mhr.welcomed", "1");
  } catch {
    // Private mode etc: overlay returns next visit.
  }
  if (elements.welcomeOverlay) elements.welcomeOverlay.hidden = true;
  if (enableDev) setDevTools(true);
}

let decoBrowList = [];
let decoLashList = [];

async function refreshDecoGrids() {
  let assets = [];
  try {
    assets = await fetchMhassetList();
  } catch {
    // Server folder missing: grids keep their quiet note.
  }
  const isObj = (a) => a.id.endsWith(".obj");
  decoBrowList = assets.filter((a) => isObj(a) && /brow/i.test(a.id));
  decoLashList = assets.filter((a) => isObj(a) && /lash/i.test(a.id));
  renderDecoTiles("brow");
  renderDecoTiles("lash");
}

function renderDecoTiles(slot) {
  const grid = slot === "brow" ? elements.decoBrowGrid : elements.decoLashGrid;
  if (!grid) return;
  const list = slot === "brow" ? decoBrowList : decoLashList;
  const fallback = slot === "brow" ? DEFAULT_BROW_ID : DEFAULT_LASH_ID;
  const current = (slot === "brow" ? eyesOral.deco.browId : eyesOral.deco.lashId) ?? fallback;
  grid.replaceChildren();
  if (list.length === 0) {
    const note = document.createElement("span");
    note.className = "quiet";
    note.textContent = "No styles found.";
    grid.append(note);
    return;
  }
  for (const asset of list) {
    const tile = document.createElement("div");
    tile.className = "hdri-thumb pose-thumb";
    tile.dataset.decoId = asset.id;
    tile.title = `${asset.name} — CC0 (MakeHuman)`;
    tile.tabIndex = 0;
    tile.setAttribute("role", "option");
    const img = document.createElement("img");
    if (asset.thumb) {
      img.src = `/api/mhasset/file/${asset.thumb}`;
    }
    img.alt = "";
    img.loading = "lazy";
    img.addEventListener("error", () => img.remove());
    const caption = document.createElement("span");
    caption.textContent = asset.name;
    tile.append(img, caption);
    const apply = () => applyDecoStyle(slot, asset.id);
    tile.addEventListener("click", apply);
    tile.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        apply();
      }
    });
    const selected = asset.id === current;
    tile.classList.toggle("is-selected", selected);
    tile.setAttribute("aria-selected", String(selected));
    grid.append(tile);
  }
}

function syncDecoGridSelection(slot) {
  const grid = slot === "brow" ? elements.decoBrowGrid : elements.decoLashGrid;
  if (!grid) return;
  const current = slot === "brow" ? eyesOral.deco.browId : eyesOral.deco.lashId;
  for (const tile of grid.querySelectorAll(".pose-thumb")) {
    const selected = tile.dataset.decoId === current;
    tile.classList.toggle("is-selected", selected);
    tile.setAttribute("aria-selected", String(selected));
  }
}

async function applyDecoStyle(slot, id) {
  const ok = slot === "brow"
    ? await eyesOral.setBrowStyle(id)
    : await eyesOral.setLashStyle(id);
  if (!ok) return;
  // Cards multiply the texture by the tint (default white); strands take
  // the tint as their absolute color (default near-black).
  const tint = isCardDecoId(id) ? CARD_TINT_DEFAULT : STRAND_TINT_DEFAULT;
  const picker = slot === "brow" ? elements.decoBrowTint : elements.decoLashTint;
  if (picker) picker.value = tint;
  eyesOral.setDecoTint(slot, tint);
  syncDecoGridSelection(slot);
}

async function applyPosePreset(id) {
  const actor = actorManager.activeActor;
  if (!actor) {
    setStatus("Select a character first.", "error");
    return;
  }
  if (!state.metadata?.jointNames || !actor.rigMapping) {
    setStatus("Rig not ready yet.", "error");
    return;
  }
  const meta = posePresetList.find((pose) => pose.id === id);
  setStatus(`Loading pose ${meta?.name ?? id}…`);
  try {
    const pose = await loadPosePreset(id);
    const idx = jointIndexByName();
    const { theta, joints } = retargetPoseToTheta(pose, idx, state.metadata.jointNames.length, {
      targetRig: getMixamoTargetRig(),
      // Authored base, not live: a previous hold must not leak into the solve.
      baseTheta: actor._baseThetaJ,
      groupQuat: actor.group.quaternion,
      offsets: state.rigMapping.jointOffsets,
    });
    stopActorClip(actor);
    actor.heldPose = { id, theta, joints, rootIdx: idx.get("root") };
    if (actor.behavior === "manual") {
      actor.applyHeldPose();
      syncDisplayedSkeleton();
    }
    syncPoseGridSelection();
    setStatus(`Pose ${meta?.name ?? id} held on ${actor.name} (${pose.mapped}/${pose.chains} chains).`);
    requestRender();
  } catch (error) {
    setStatus(`Pose failed: ${error?.message || error}`, "error");
  }
}

function releasePosePreset() {
  const actor = actorManager.activeActor;
  if (!actor?.heldPose) {
    setStatus("No held pose on this character.");
    return;
  }
  actor.heldPose = null;
  if (actor.behavior === "manual") {
    actor.computeThetaJFromParams();
    actor.applyThetaJ(actor.rigMapping);
    syncDisplayedSkeleton();
  }
  syncPoseGridSelection();
    setStatus(`Pose released on ${actor.name}.`);
  requestRender();
}

function hidePoseLicense() {
  document.getElementById("pose-license-pop")?.remove();
  document.removeEventListener("scroll", hidePoseLicense, true);
}

function showPoseLicense(x, y, pose) {
  hidePoseLicense();
  const pop = document.createElement("div");
  pop.id = "pose-license-pop";
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#39;" }[c]));
  const lic = pose.license || "Unknown — check before reuse";
  pop.innerHTML =
    `<b>${esc(pose.name)}</b>` +
    (pose.description ? `<span>${esc(pose.description)}</span>` : "") +
    `<span>License: <b>${esc(lic)}</b></span>` +
    (pose.copyright ? `<span>${esc(pose.copyright)}</span>` : "") +
    (pose.set ? `<span class="quiet">${esc(pose.set)}</span>` : "") +
    (pose.license && /by/i.test(pose.license) ? `<span class="quiet">CC-BY: credit the author on reuse.</span>` : "");
  pop.style.left = `${Math.min(x, window.innerWidth - 260)}px`;
  pop.style.top = `${Math.min(y, window.innerHeight - 150)}px`;
  document.body.append(pop);
  document.addEventListener("scroll", hidePoseLicense, true);
  document.addEventListener("click", hidePoseLicense, { once: true });
}

function playNextMixamo() {
  const sel = elements.mixamoClip;
  if (sel && sel.options.length > 0) {
    let idx = sel.selectedIndex;
    for (let i = 0; i < sel.options.length; i++) {
      idx = (idx + 1) % sel.options.length;
      if (sel.options[idx].value) break;
    }
    sel.selectedIndex = idx;
    sel.dispatchEvent(new Event("change"));
  }
  void playMixamoOnSelected();
}

async function playMixamoOnSelected() {
  const actor = actorManager.activeActor;
  const name = elements.mixamoClip?.value;
  if (!actor) {
    updateMixamoStatus("Select a character first.");
    return;
  }
  if (!name) {
    updateMixamoStatus("Pick a clip first.");
    return;
  }
  if (!state.metadata?.jointNames || !actor.rigMapping) {
    updateMixamoStatus("Rig not ready yet.");
    return;
  }
  updateMixamoStatus(`Loading ${name}…`);
  try {
    const clip = await loadMixamoClip(name, state.metadata.jointNames);
    stopActorClip(actor);
    actor.activeClip = clip.createPlayer({ loop: elements.mixamoLoop?.checked !== false });
    actor.heldPose = null;
    syncPoseGridSelection();
    if (actor.isAnchor) ensureAnchorClipPump();
    console.info(
      `[mixamo] playing ${name} on ${actor.name} ` +
      `(${clip.mappedCount}/${clip.boneCount} bones, ${clip.duration.toFixed(2)}s).`,
    );
    updateMixamoStatus();
    requestRender();
  } catch (error) {
    updateMixamoStatus(`Load failed: ${error.message}`);
  }
}

function scheduleMixamoInspect() {
  if (mixamoInspectTimer) clearTimeout(mixamoInspectTimer);
  mixamoInspectTimer = setTimeout(() => {
    mixamoInspectTimer = 0;
    void inspectMixamoSelection();
  }, 250);
}

async function inspectMixamoSelection() {
  const name = elements.mixamoClip?.value;
  if (!name) {
    lastMixamoInspectName = "";
    clearMixamoMeta();
    return;
  }
  try {
    const clip = await loadMixamoClip(name, state.metadata?.jointNames || []);
    if (elements.mixamoClip?.value !== name) return; // stale
    const md = clip.getMetadata();
    renderMixamoMeta(md);
    if (name !== lastMixamoInspectName) {
      lastMixamoInspectName = name;
      if (elements.mixamoLoop) elements.mixamoLoop.checked = clip.isLooping;
    }
    console.info(
      `[mixamo] meta ${name}: take="${md.takeName}" dur=${md.duration.toFixed(2)}s fps=${md.fps ?? "?"} ` +
      `tracks=${md.trackCount}/${md.keyCount}k bones=${md.mappedCount}/${md.boneCount} ` +
      `unmapped=[${md.unmapped.join(",")}] hipsY=${md.restHipsY === null ? "?" : md.restHipsY.toFixed(1)} ` +
      `loop=${md.isLooping ? "seamless" : "open"}(${md.loopGapDeg.toFixed(1)}°)`,
    );
  } catch (error) {
    clearMixamoMeta();
    updateMixamoStatus(`Inspect failed: ${error.message}`);
  }
}

function renderMixamoMeta(md) {
  if (elements.mixamoMetaTake) elements.mixamoMetaTake.textContent = md.takeName || "—";
  if (elements.mixamoMetaDuration) {
    elements.mixamoMetaDuration.textContent = `${md.duration.toFixed(2)}s` + (md.fps ? ` · ~${md.fps}fps` : "");
  }
  if (elements.mixamoMetaTracks) {
    elements.mixamoMetaTracks.textContent = `${md.trackCount} tracks · ${md.keyCount} keys`;
  }
  if (elements.mixamoMetaBones) {
    elements.mixamoMetaBones.textContent = `${md.mappedCount}/${md.boneCount} mapped`;
  }
  if (elements.mixamoMetaUnmapped) {
    elements.mixamoMetaUnmapped.textContent = md.unmapped.length === 0 ? "—" : md.unmapped.join(", ");
    elements.mixamoMetaUnmapped.title = md.unmapped.join(", ");
  }
  if (elements.mixamoMetaHips) {
    elements.mixamoMetaHips.textContent = md.restHipsY === null ? "—" : md.restHipsY.toFixed(1);
  }
  if (elements.mixamoMetaLoop) {
    elements.mixamoMetaLoop.textContent = md.isLooping
      ? `seamless · gap ${md.loopGapDeg.toFixed(1)}°`
      : `open · gap ${md.loopGapDeg.toFixed(1)}°`;
  }
}

function clearMixamoMeta() {
  for (const el of [
    elements.mixamoMetaTake,
    elements.mixamoMetaDuration,
    elements.mixamoMetaTracks,
    elements.mixamoMetaBones,
    elements.mixamoMetaUnmapped,
    elements.mixamoMetaHips,
    elements.mixamoMetaLoop,
  ]) {
    if (el) el.textContent = "—";
  }
}

function stopMixamoOnSelected() {
  stopActorClip(actorManager.activeActor);
  updateMixamoStatus();
  requestRender();
}

function stopAllMixamoClips() {
  for (const actor of actorManager.getAllActors()) stopActorClip(actor);
  updateMixamoStatus();
  requestRender();
}

function syncMixamoLoop() {
  const loop = elements.mixamoLoop?.checked !== false;
  for (const actor of actorManager.getAllActors()) {
    if (actor.activeClip) actor.activeClip.setLoop(loop);
  }
}

function updateMixamoStatus(override) {
  if (!elements.mixamoStatus) return;
  if (override !== undefined) {
    elements.mixamoStatus.textContent = override;
    return;
  }
  const playing = [];
  for (const actor of actorManager.getAllActors()) {
    if (actor.activeClip) {
      const label = actor.isAnchor ? "Actor #0" : `#${actor.id}`;
      playing.push(`${label} ← ${actor.activeClip.clip.name}`);
    }
  }
  elements.mixamoStatus.textContent = playing.length === 0
    ? "No clip playing."
    : playing.join(" · ");
}

// The anchor is not in the sim tick, so its clip needs its own pump while
// active (wall clock, global-speed scaled) plus stepAnchorClip for manual
// stepping. Agents advance through Actor.update (sim time, pause-aware).
// Manual stepping while paused: advance the anchor clip by exact sim time,
// mirroring what simulation.step does for agents.
function stepAnchorClip(dt) {
  const anchor = actorManager.getActor(0);
  if (!anchor?.activeClip) return;
  anchor.applyClipPose(dt);
  syncDisplayedSkeleton();
  updateMixamoStatus();
  requestRender();
}

function ensureAnchorClipPump() {
  if (!anchorClipRaf) {
    anchorClipLast = performance.now();
    anchorClipRaf = requestAnimationFrame(pumpAnchorClip);
  }
}

function pumpAnchorClip(now) {
  anchorClipRaf = 0;
  const anchor = actorManager.getActor(0);
  if (!anchor?.activeClip) return;
  if (!isAgentSimulationActive()) {
    // Paused: freeze the clip without advancing or rendering; the chain
    // stays alive and anchorClipLast tracks now so resume has no dt leap.
    anchorClipLast = now;
    anchorClipRaf = requestAnimationFrame(pumpAnchorClip);
    return;
  }
  const dt = Math.min(0.1, Math.max(0, (now - anchorClipLast) / 1000));
  anchorClipLast = now;
  anchor.applyClipPose(dt * getSimulationTimeScale());
  syncDisplayedSkeleton();
  updateMixamoStatus();
  requestRender();
  anchorClipRaf = requestAnimationFrame(pumpAnchorClip);
}

function syncSimPlayButton() {
  if (!elements.crowdSimPlay) return;
  const active = isAgentSimulationActive();
  elements.crowdSimPlay.classList.toggle("is-on", active);
  elements.crowdSimPlay.setAttribute("aria-pressed", String(active));
  elements.crowdSimPlay.textContent = active ? "Play: On" : "Play: Off";
}

// Single crowd-eyes pass for Face edits while the sim is paused
// (when running, the tick covers it).
function refreshCrowdEyesOnce() {
  if (isAgentSimulationActive()) return;
  crowdEyes.update(simulation.agents, simulation.simTime, performance.now());
  crowdMouth.update(simulation.agents);
  requestRender();
}

// Agent console: action implementations over the real app functions.
// Mirrors the slider/button paths exactly (locks, clamps, anchor sync) so the
// agent can never do what the UI could not.
function createAgentBindings() {
  return {
    selectActor: async ({ actorId }) => {
      selectAgent(actorId);
      const id = getSelectedAgentId();
      return { selected: id, name: id === 0 ? "Actor #0" : `Actor #${id}` };
    },
    setParam: async ({ group, index, value }) => {
      const meta = state.metadata?.groups?.[group]?.[index];
      if (!meta) throw new Error(`unknown param ${group}[${index}]`);
      if (isLocked(meta)) throw new Error(`param locked: ${group}/${meta.name}`);
      const v = Math.min(meta.max, Math.max(meta.min, value));
      meta.value = v;
      const actor = actorManager.activeActor;
      if (actor) actor.setParamValue(group, index, v);
      if (!actor || actor.isAnchor) {
        if (group === "pose") {
          evaluatePoseClientSide();
          syncPoseServer(index, v);
        } else {
          evaluateBlendshapesClientSide();
          syncBlendshapeServer(group, index, v);
        }
      } else {
        refreshGpuAgents();
        requestRender();
      }
      if (state.currentGroup === group && currentParameter()?.index === index) {
        showParameter(meta);
      }
      return { group, index, name: meta.name, value: v, clamped: v !== value };
    },
    spawnAgents: async ({ count }) => {
      const r = await spawnAgents(count);
      return { added: r.added, total: r.total, gpu: r.gpuAdded, cpu: r.cpuAdded };
    },
    clearAgents: async () => {
      clearAgents();
      return { cleared: true, agents: agentCount() };
    },
    removeActor: async ({ actorId }) => {
      if (!listAgents().some((a) => a.id === actorId)) throw new Error(`no agent ${actorId}`);
      removeAgent(actorId);
      return { removed: actorId, agents: agentCount() };
    },
    camera: async ({ view }) => {
      if (view === "focus") focusSelectedAgent();
      else fitCamera(view, activeActorFocus(view));
      return { view };
    },
    gaze: async ({ yaw, pitch }) => {
      elements.gazeYaw.value = yaw;
      elements.gazePitch.value = pitch;
      elements.gazeYawVal.textContent = `${yaw}°`;
      elements.gazePitchVal.textContent = `${pitch}°`;
      return { yaw, pitch, ...setActiveGaze(yaw, pitch) };
    },
    setBehavior: async ({ behavior, actorId }) => {
      if (actorId == null) {
        setAgentBehavior(behavior);
        return { behavior, scope: "all" };
      }
      setOneAgentBehavior(actorId, behavior);
      return { behavior, scope: actorId };
    },
    lodColors: async ({ enabled }) => {
      setShowLodColors(enabled);
      return { lodColors: getShowLodColors() };
    },
    benchmark: async ({ agents, seconds }) => {
      const r = await runBenchmark({ agents, seconds, lod: "auto" }, () => {});
      const round2 = (n) => Math.round(n * 100) / 100;
      return {
        fpsAvg: round2(r.fpsAvg), frameP95: round2(r.frameP95),
        agents: r.gpuAgents + r.cpu, lodMix: r.lodMix,
      };
    },
  };
}

function getAgentSnapshot() {
  const agents = listAgents();
  const selectedId = getSelectedAgentId();
  const sel = agents.find((a) => a.id === selectedId);
  return {
    agents: agents.length,
    gpu: agents.filter((a) => a.gpu).length,
    cpu: agents.filter((a) => !a.gpu).length,
    selectedId,
    selected: sel
      ? { behavior: sel.behavior, x: Math.round(sel.x * 10) / 10, z: Math.round(sel.z * 10) / 10 }
      : null,
    behavior: getAgentBehavior(),
    lodColors: getShowLodColors(),
    simRunning: isAgentSimulationActive(),
    view: state.lastView ?? "front",
  };
}

// Shared by the Eyes sliders and the agent gaze action.
function setActiveGaze(yaw, pitch) {
  const target = actorManager.activeActor;
  if (!target || target.isAnchor) {
    eyesOral.setGaze(yaw, pitch);
    return { target: "main" };
  }
  crowdEyes.setAgentGaze(target.id, yaw, pitch);
  refreshCrowdEyesOnce();
  return { target: target.id };
}

function syncAutolodButton() {
  if (!elements.crowdSimAutolod) return;
  const active = simulation.autoLod;
  elements.crowdSimAutolod.classList.toggle("is-on", active);
  elements.crowdSimAutolod.setAttribute("aria-pressed", String(active));
  elements.crowdSimAutolod.textContent = active ? "LOD: Auto" : "LOD: Manual";
}

function syncLodcolorsButton() {
  if (!elements.crowdSimLodcolors) return;
  const active = simulation.showLodColors;
  elements.crowdSimLodcolors.classList.toggle("is-on", active);
  elements.crowdSimLodcolors.setAttribute("aria-pressed", String(active));
  elements.crowdSimLodcolors.textContent = active ? "LOD colors: On" : "LOD colors: Off";
}

async function initialize() {
  try {
    showLoading("Loading rigged MHR model…");
    const metadataResponse = await fetchChecked(
      "/api/metadata",
      { cache: "no-store" },
      "The viewer backend did not return model metadata.",
    );
    state.metadata = await metadataResponse.json();
    for (const items of Object.values(state.metadata.groups ?? {})) {
      for (const item of items ?? []) {
        if (!Number.isFinite(item.value)) {
          item.value = Math.min(item.max ?? 0, Math.max(item.min ?? 0, 0));
        }
      }
    }

    const [rigMappingBuffer, blendshapesBuffer, correctivesBuffer] = await Promise.all([
      fetchBinaryAsset(
        "/api/rig-mapping",
        "The viewer backend did not return rig mapping.",
      ),
      fetchBinaryAsset(
        "/api/weights/blendshapes",
        "The viewer backend did not return blendshape weights.",
      ),
      fetchBinaryAsset(
        "/api/weights/pose-correctives",
        "The viewer backend did not return pose corrective weights.",
      ),
    ]);

    state.rigMapping = parseRigMapping(rigMappingBuffer);
    setMixamoTargetRig(
      state.rigMapping.jointPreRotations,
      state.metadata.jointParents,
      state.rigMapping.jointOffsets,
    );
    state.blendshapes = parseBlendshapes(blendshapesBuffer);
    state.poseCorrectives = parsePoseCorrectives(correctivesBuffer);

    await setRiggedMesh(state.metadata.model.lod, { fit: true });
    initExprVectors();
    prewarmGpuAgents();
    if (!eyesOral.attach()) {
      console.warn("Eyes & mouth attachments did not find their anchor joints.");
    }

    updateModelStatistics();
    updateModelControls();
    elements.poseCount.textContent = state.metadata.groups.pose.length;
    elements.identityCount.textContent = state.metadata.groups.identity.length;
    elements.expressionCount.textContent = state.metadata.groups.expression.length;
    const dockApi = window.mhr?.dockApi;
    dockApi?.getPanel("pose")?.api.setTitle(`Pose (${state.metadata.groups.pose.length})`);
    dockApi?.getPanel("identity")?.api.setTitle(`Identity (${state.metadata.groups.identity.length})`);
    dockApi?.getPanel("expression")?.api.setTitle(`Expression (${state.metadata.groups.expression.length})`);

    buildCategories();
    const preferred = state.metadata.groups.pose.find(
      (parameter) => parameter.name === "head_lean",
    );
    if (preferred) state.selectedIndex.pose = preferred.index;
    selectGroup("pose"); syncIdentityGrid(); applyLandingLook();
    setActiveGaze(-14, 0); // Sideways glance away from the smirk.
    elements.gazeYaw.value = -14; elements.gazeYawVal.textContent = "-14°";
    handleGroupChanged();
    if (!document.body.classList.contains("is-docked")) {
      switchActorTab(localStorage.getItem("mhr.actorTab") || "identity", { persist: false });
    }
    fitCamera("face", activeActorFocus("face")); // Land on a face close-up like the welcome shot.
    // Time flows from boot: the loop runs even with zero agents.
    startSimulation();
    const agentActionLog = createActionLog();
    const agentDispatcher = createDispatcher(createAgentBindings(), { log: agentActionLog });
    initAgentPanel({ dispatcher: agentDispatcher, getSnapshot: getAgentSnapshot, log: agentActionLog });
    initCommandMenu({
      dispatch: (name, args) => agentDispatcher.dispatch(name, args, { source: "palette" }),
      listActors: () => actorManager.getAllActors().map((a) => ({ id: a.id, name: a.name })),
      isSimRunning: () => isAgentSimulationActive(),
      toggleSim: () => {
        const running = toggleAgentSimulation();
        syncSimPlayButton();
        setStatus(running ? "Agent simulation resumed." : "Agent simulation paused.");
      },
      isHeaderHidden: () => document.body.classList.contains("topbar-hidden"),
      setHeaderHidden,
      resetLayout: () => window.mhr.resetDockLayout?.(),
      setStatus,
    });
    // Minimal console plug: drive the app from DevTools or an external agent.
    window.mhr = {
      ...(window.mhr ?? {}),
      dispatch: (name, args) => agentDispatcher.dispatch(name, args, { source: "console" }),
      log: () => agentActionLog.entries,
      avatarCode: { packAvatarData, unpackAvatarCode, collectAvatarState, applyAvatarState },
      snapshot: getAgentSnapshot,
      actions: agentDispatcher.actionNames(),
      restFrames: () => {
        // Our REST global orientations per joint, composed from pre-rotations
        // down the bone parent chain (pose-independent: ignores live params).
        const names = state.metadata.jointNames;
        const skel = state.skeleton;
        const bones = skel.bones;
        const { jointPreRotations } = state.rigMapping;
        const out = {};
        for (const jn of ["l_upleg", "l_lowleg", "l_foot", "l_uparm", "l_lowarm", "l_wrist"]) {
          const j = names.indexOf(jn);
          if (j < 0) { out[jn] = null; continue; }
          // Chain of joint indices from the root to j (via bone parents).
          const chain = [];
          let b = bones[state.boneIndexMap[j]];
          const seen = new Set();
          while (b && !seen.has(b)) {
            seen.add(b);
            const ji = state.boneIndexMap.indexOf(bones.indexOf(b));
            if (ji >= 0) chain.unshift(ji);
            b = b.parent && b.parent.isBone ? b.parent : null;
          }
          let x = 0, y = 0, z = 0, w = 1;
          for (const ji of chain) {
            const px = jointPreRotations[ji * 4], py = jointPreRotations[ji * 4 + 1];
            const pz = jointPreRotations[ji * 4 + 2], pw = jointPreRotations[ji * 4 + 3];
            const nx = pw * x + px * w + py * z - pz * y;
            const ny = pw * y - px * z + py * w + pz * x;
            const nz = pw * z + px * y - py * x + pz * w;
            const nw = pw * w - px * x - py * y - pz * z;
            x = nx; y = ny; z = nz; w = nw;
          }
          out[jn] = [x, y, z, w].map((v) => +v.toFixed(4));
        }
        return out;
      },
      limbFlex: () => {
        const a = actorManager.getActor(0);
        if (!a) return null;
        const names = state.metadata.jointNames;
        const P = (n) => {
          const i = names.indexOf(n) * 3;
          return [a.jointPositions[i], a.jointPositions[i + 1], a.jointPositions[i + 2]];
        };
        const ang = (A, B, C) => {
          const u = [A[0] - B[0], A[1] - B[1], A[2] - B[2]];
          const v = [C[0] - B[0], C[1] - B[1], C[2] - B[2]];
          const du = Math.hypot(...u);
          const dv = Math.hypot(...v);
          if (!(du > 1e-9 && dv > 1e-9)) return null;
          const c = Math.min(1, Math.max(-1, (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (du * dv)));
          return +((Math.acos(c) * 180) / Math.PI).toFixed(1);
        };
        return {
          kneeL: ang(P("l_upleg"), P("l_lowleg"), P("l_foot")),
          kneeR: ang(P("r_upleg"), P("r_lowleg"), P("r_foot")),
          elbowL: ang(P("l_uparm"), P("l_lowarm"), P("l_wrist")),
          elbowR: ang(P("r_uparm"), P("r_lowarm"), P("r_wrist")),
        };
      },
      actorJoints: (id = 0) => {
        const a = actorManager.getActor(id);
        if (!a) return null;
        return {
          names: [...state.metadata.jointNames],
          pos: [...a.jointPositions],
          groupPos: a.group.position.toArray(),
          groupQuat: [a.group.quaternion.x, a.group.quaternion.y, a.group.quaternion.z, a.group.quaternion.w],
        };
      },
      clipDebug: () => actorManager.getAllActors()
        .filter((a) => a.activeClip)
        .map((a) => {
          let sum = 0;
          for (let i = 0; i < a.thetaJ.length; i += 7) {
            sum += Math.abs(a.thetaJ[i + 3]) + Math.abs(a.thetaJ[i + 4]) + Math.abs(a.thetaJ[i + 5]);
          }
          return {
            actor: a.isAnchor ? "Actor #0" : `#${a.id}`,
            gpu: Boolean(a.isGpuAgent),
            time: +a.activeClip.time.toFixed(2),
            actionTime: +a.activeClip.action.time.toFixed(2),
            channels: a.activeClip.channels.length,
            thetaSum: +sum.toFixed(3),
            globalPath: a.activeClip._useGlobal?.() ?? null,
            src: a.activeClip.sourceFlex?.() ?? null,
            legs: (a.activeClip.channelDebug?.() ?? [])
              .filter((c) => /UpLeg|Leg|Foot|Hips|Spine|Neck|Head|Arm|ForeArm|Hand/.test(c.bone)),
          };
        }),
    };
    hideLoading();
    setStatus(
      state.isWebGPU
        ? "Ready (WebGPU Rendering, Pose Correctives & Hardware Skinning 60+ FPS)"
        : "Ready (Client-side Correctives, Blendshapes & Hardware Skinning 60+ FPS)",
    );
    renderActorsList();
    refreshActorCard();
    refreshSimMetrics();
    refreshCrowdStats(true);
    scheduleMixamoInspect();
    onActiveActorChanged(actorManager.activeActor);
    requestRender();
    triggerUrlBenchmark();
    void applyUrlAvatar();
  } catch (error) {
    elements.loadingText.textContent = error.message;
    setStatus(error.message, "error");
  }
}

const ACTOR_TABS = ["identity", "pose", "expression", "eyes", "mouth", "materials", "clothing", "animation", "ai"];
const PARAM_TABS = ["pose", "expression"];

// One shared parameter component, reparented into the active tab. Used by
// the fallback strip (switchActorTab) and the dock activation callback alike.
function showParamGroup(group) {
  document.getElementById(`param-mount-${group}`).appendChild(elements.paramComponent);
  if (state.currentGroup !== group) {
    selectGroup(group);
    handleGroupChanged();
  }
}

// Tab-coupled overlays: visiting Pose reveals the skeleton, visiting
// Expression reveals the vector field; leaving restores the pre-visit state
// instead of stomping a manual choice. Real user toggles clear auto
// ownership (tracked via the suppression flag around synthetic events).
let suppressOverlayEvent = false;
const tabAutoOverlay = { skeleton: false, exprVectors: false };
function setOverlayToggle(el, on) {
  if (!el || el.checked === on) return;
  suppressOverlayEvent = true;
  try {
    el.checked = on;
    el.dispatchEvent(new Event("change"));
  } finally {
    suppressOverlayEvent = false;
  }
}
function syncTabAutoOverlays(next) {
  if (next === "pose") {
    if (!elements.skeleton.checked) {
      tabAutoOverlay.skeleton = true;
      setOverlayToggle(elements.skeleton, true);
    }
  } else if (tabAutoOverlay.skeleton) {
    tabAutoOverlay.skeleton = false;
    setOverlayToggle(elements.skeleton, false);
  }
  if (next === "expression") {
    if (elements.exprVectors && !elements.exprVectors.checked) {
      tabAutoOverlay.exprVectors = true;
      setOverlayToggle(elements.exprVectors, true);
    }
  } else if (tabAutoOverlay.exprVectors) {
    tabAutoOverlay.exprVectors = false;
    setOverlayToggle(elements.exprVectors, false);
  }
}

function switchActorTab(name, { persist = true } = {}) {
  const requested = ACTOR_TABS.includes(name) ? name : "identity";
  // Restored dev-only tabs (AI) fall back for non-dev sessions.
  const next = !isDevTools() && DEV_ATABS.includes(requested) ? "identity" : requested;
  elements.actorTabs.querySelectorAll("[data-atab]").forEach((tab) => {
    const active = tab.dataset.atab === next;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  for (const panel of ACTOR_TABS) {
    document.getElementById(`atab-${panel}`).hidden = panel !== next;
  }
  if (PARAM_TABS.includes(next)) showParamGroup(next); if (next === "identity") syncIdentityGrid(); syncTabAutoOverlays(next);
  if (persist) localStorage.setItem("mhr.actorTab", next);
}

// Docked: activate the dock tab (param sync runs via onPanelActivated).
// Fallback: drive the static strip. Closed dock tabs are a safe no-op.
function showActorTab(name) {
  if (document.body.classList.contains("is-docked")) {
    window.mhr?.dockApi?.getPanel?.(name)?.api?.setActive?.();
    return;
  }
  switchActorTab(name);
}

// Collapsible header: hiding keeps the viewport tall (mobile); the slim
// restore strip or View menu brings it back. Persisted across reloads.
const TOPBAR_LS = "neohuman.topbarHidden";
function syncTopbar() {
  const hidden = document.body.classList.contains("topbar-hidden");
  if (elements.topbarShow) elements.topbarShow.hidden = !hidden;
}
function setHeaderHidden(hidden) {
  document.body.classList.toggle("topbar-hidden", hidden);
  try {
    localStorage.setItem(TOPBAR_LS, hidden ? "1" : "0");
  } catch {
    // Private mode etc: stay ephemeral.
  }
  syncTopbar();
}

function wireEventListeners() { initIdentityGrid();
  document.getElementById("identity-wipe-smile")?.addEventListener("click", wipeLandingLook);
  document.getElementById("identity-avatar-copy")?.addEventListener("click", () => void copyAvatarCode());
  document.getElementById("identity-avatar-import")?.addEventListener("click", () => void importAvatarCode());
  document.getElementById("avatar-code-apply")?.addEventListener("click", () => void applyAvatarDialog());
  document.getElementById("avatar-code-copy")?.addEventListener("click", async () => {
    const area = document.getElementById("avatar-code-text");
    if (!area) return;
    try {
      await navigator.clipboard.writeText(area.value);
      setStatus("Avatar code copied.");
    } catch {
      area.focus(); area.select();
      setStatus("Clipboard blocked — press Ctrl+C.", "error");
    }
  });
  elements.actorTabs.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-atab]");
    if (tab) switchActorTab(tab.dataset.atab);
  });
  elements.search.addEventListener("input", () => rebuildParameterList());
  elements.category.addEventListener("change", () => rebuildParameterList());
  function selectParameter(index) {
    if (state.requestPending) return;
    state.selectedIndex[state.currentGroup] = index;
    elements.parameter.querySelectorAll(".parameter-item").forEach((candidate) => {
      const selected = Number(candidate.dataset.index) === index;
      candidate.classList.toggle("is-selected", selected);
      candidate.setAttribute("aria-selected", String(selected));
    });
    elements.parameter
      .querySelector(`.parameter-item[data-index="${index}"]`)
      ?.scrollIntoView({ block: "nearest" });
    showParameter(currentParameter());
    revealJointForParam(index);
    requestRender();
  }
  elements.parameter.addEventListener("click", (event) => {
    const item = event.target.closest(".parameter-item");
    if (!item || state.requestPending) return;
    selectParameter(Number(item.dataset.index));
  });
  elements.slider.addEventListener("input", () => {
    cancelResetAnimation();
    recordBandwidthSaved(226611);
    const val = Number(elements.slider.value);
    elements.value.value = formatNumber(val);
    const parameter = currentParameter();
    if (!parameter || isLocked(parameter)) return;
    if (parameter) {
      elements.activeParameter.textContent =
        `${state.currentGroup} / ${parameter.name} = ${formatNumber(val)}`;
      parameter.value = val;
    }
    const actor = actorManager.activeActor;
    if (actor) {
      actor.setParamValue(state.currentGroup, parameter.index, val);
    }
    if (!actor || actor.isAnchor) {
      if (state.currentGroup === "pose") {
        evaluatePoseClientSide();
      } else if (state.currentGroup === "identity" || state.currentGroup === "expression") {
        evaluateBlendshapesClientSide();
      }
    } else {
      refreshGpuAgents();
      requestRender();
    }
  });
  elements.slider.addEventListener("change", () => {
    const val = Number(elements.slider.value);
    const parameter = currentParameter();
    if (!parameter || isLocked(parameter)) return;
    parameter.value = val;
    const actor = actorManager.activeActor;
    if (actor) {
      actor.setParamValue(state.currentGroup, parameter.index, val);
    }
    if (!actor || actor.isAnchor) {
      if (state.currentGroup === "pose") {
        evaluatePoseClientSide();
        syncPoseServer(parameter.index, val);
      } else if (state.currentGroup === "identity") {
        evaluateBlendshapesClientSide();
        syncBlendshapeServer("identity", parameter.index, val);
      } else if (state.currentGroup === "expression") {
        evaluateBlendshapesClientSide();
        syncBlendshapeServer("expression", parameter.index, val);
      }
    } else {
      refreshGpuAgents();
      requestRender();
    }
  });
  elements.value.addEventListener("change", commitParameterValue);
  elements.value.addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    commitParameterValue();
  });
  elements.resetCurrent.addEventListener("click", () => animateReset("reset_current"));
  elements.resetGroup.addEventListener("click", () => animateReset("reset_group"));
  elements.resetAll.addEventListener("click", () => animateReset("reset_all"));
  elements.correctives.addEventListener("click", () => {
    const nextVal = !state.metadata.model.applyCorrectives;
    state.metadata.model.applyCorrectives = nextVal;
    elements.correctives.classList.toggle("is-on", nextVal);
    elements.correctives.setAttribute("aria-checked", String(nextVal));
    elements.correctivesState.textContent = nextVal ? "On" : "Off";
    evaluatePoseClientSide();
    requestRender();
    setStatus(`Pose correctives ${nextVal ? "enabled" : "disabled"}`);

    fetch("/api/configure", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ applyCorrectives: nextVal }),
    }).catch(() => {});
  });
  if (elements.crowdSimSpawn) {
    elements.crowdSimSpawn.addEventListener("click", async () => {
      elements.crowdSimSpawn.disabled = true;
      try {
        const requested = crowdSpawnCount(elements.crowdSimCount, 6, Infinity);
        const { added, total, gpuAdded, cpuAdded, ms } = await spawnAgents(requested);
        refreshCrowdTotals();
        syncSimPlayButton();
        const message = `Spawned ${added} autonomous ${added === 1 ? "agent" : "agents"} — ${total} active (${gpuAdded} GPU + ${cpuAdded} CPU, ${(ms / 1000).toFixed(1)}s).`;
        elements.crowdSimSummary.textContent = message;
        setStatus(message);
      } catch (error) {
        setStatus(error.message, "error");
      } finally {
        elements.crowdSimSpawn.disabled = false;
      }
    });
  }

  if (elements.crowdSimClear) {
    elements.crowdSimClear.addEventListener("click", () => {
      clearAgents();
      refreshCrowdTotals();
      syncSimPlayButton();
      elements.crowdSimSummary.textContent = "Cleared all simulated agents.";
      setStatus("Cleared autonomous agents.");
    });
  }

  if (elements.crowdSimBehavior) {
    elements.crowdSimBehavior.addEventListener("change", () => {
      setAgentBehavior(elements.crowdSimBehavior.value);
      const mode = elements.crowdSimBehavior.value;
      const message = `Switched agent behavior to ${mode}.`;
      elements.crowdSimSummary.textContent = message;
      setStatus(message);
    });
  }

  if (elements.crowdSimPlay) {
    elements.crowdSimPlay.addEventListener("click", () => {
      const isRunning = toggleAgentSimulation();
      syncSimPlayButton();
      const message = isRunning ? "Agent simulation resumed." : "Agent simulation paused.";
      elements.crowdSimSummary.textContent = message;
      setStatus(message);
    });
  }

  if (elements.topbarShow) {
    elements.topbarShow.addEventListener("click", () => setHeaderHidden(false));
  }
  try {
    if (localStorage.getItem(TOPBAR_LS) === "1") document.body.classList.add("topbar-hidden");
  } catch {
    // Private mode etc: stay ephemeral.
  }
  syncTopbar();

  addSimulationChangeListener(() => {
    refreshCrowdTotals();
    syncSimPlayButton();
    eyesOral.setPaused(!isAgentSimulationActive());
    syncAutolodButton();
    syncLodcolorsButton();
    renderActorsList();
    refreshActorCard();
    refreshSimMetrics();
    refreshCrowdStats();
    updateMixamoStatus();
  });

  actorManager.addChangeListener(onActiveActorChanged);

  if (elements.crowdStatsBody) {
    elements.crowdStatsBody.addEventListener("click", (event) => {
      const row = event.target.closest("[data-id]");
      if (!row) return;
      selectAgent(Number(row.dataset.id));
      refreshActorCard();
      refreshCrowdStats(true);
    });
  }
  if (elements.actorsList) {
    elements.actorsList.addEventListener("click", (event) => {
      const action = event.target.closest("[data-empty]");
      if (action) {
        if (action.dataset.empty === "spawn") {
          if (elements.crowdSimCount) elements.crowdSimCount.value = "6";
          elements.crowdSimSpawn?.click();
        } else if (action.dataset.empty === "crowd") {
          activatePanel("crowd", "crowd");
        } else if (action.dataset.empty === "cmdk") {
          document.getElementById("cmdk-open")?.click();
        }
        return;
      }
      const row = event.target.closest("[data-id]");
      if (!row) return;
      selectAgent(Number(row.dataset.id));
      refreshActorCard();
    });
  }

  if (elements.crowdSimAutolod) {
    elements.crowdSimAutolod.addEventListener("click", () => {
      setAutoLodEnabled(!simulation.autoLod);
      syncAutolodButton();
    });
    syncAutolodButton();
  }

  if (elements.crowdSimLodcolors) {
    elements.crowdSimLodcolors.addEventListener("click", () => {
      setShowLodColors(!simulation.showLodColors);
      syncLodcolorsButton();
    });
    syncLodcolorsButton();
  }

  if (elements.crowdActorLod) {
    elements.crowdActorLod.addEventListener("change", () => {
      setAgentLodPin(getSelectedAgentId(), elements.crowdActorLod.value);
      refreshCrowdStats(true);
    });
  }

  if (elements.crowdStatsAutorefresh) {
    elements.crowdStatsAutorefresh.addEventListener("change", () => {
      crowdStatsAutorefresh = elements.crowdStatsAutorefresh.checked;
      if (crowdStatsAutorefresh) refreshCrowdStats(true);
    });
  }

  if (elements.crowdStatsRefresh) {
    elements.crowdStatsRefresh.addEventListener("click", () => {
      refreshCrowdStats(true);
    });
  }

  if (elements.mixamoFilter) {
    elements.mixamoFilter.addEventListener("input", populateMixamoSelect);
  }
  if (elements.mixamoRefresh) {
    elements.mixamoRefresh.addEventListener("click", () => {
      void refreshMixamoList();
    });
  }
  if (elements.mixamoPlay) {
    elements.mixamoPlay.addEventListener("click", () => {
      void playMixamoOnSelected();
    });
  }
  if (elements.mixamoNext) {
    elements.mixamoNext.addEventListener("click", () => {
      playNextMixamo();
    });
  }
  if (elements.mixamoStop) {
    elements.mixamoStop.addEventListener("click", stopMixamoOnSelected);
  }
  if (elements.mixamoStopAll) {
    elements.mixamoStopAll.addEventListener("click", stopAllMixamoClips);
  }
  if (elements.mixamoClip) {
    elements.mixamoClip.addEventListener("change", scheduleMixamoInspect);
  }
  if (elements.mixamoLoop) {
    elements.mixamoLoop.addEventListener("change", syncMixamoLoop);
  }
  if (elements.mixamoGlobal) {
    const syncGlobalMatch = () => {
      setGlobalMatch(elements.mixamoGlobal.checked !== false);
    };
    elements.mixamoGlobal.addEventListener("change", syncGlobalMatch);
    syncGlobalMatch();
  }
  if (elements.mixamoSpine) {
    const syncSpineBoost = () => {
      const v = setSpineBoost(parseFloat(elements.mixamoSpine.value));
      if (elements.mixamoSpineVal) elements.mixamoSpineVal.textContent = `${v}x`;
    };
    elements.mixamoSpine.addEventListener("input", syncSpineBoost);
    syncSpineBoost();
  }
  // Independent of model load: read the server clip folder right away.
  void refreshMixamoList();

  if (elements.crowdSimFocus) {
    elements.crowdSimFocus.addEventListener("click", () => {
      focusSelectedAgent();
    });
  }

  if (elements.crowdActorRemove) {
    elements.crowdActorRemove.addEventListener("click", () => {
      removeAgent(getSelectedAgentId());
      renderActorsList();
      refreshActorCard();
    });
  }

  if (elements.crowdActorSpeed) {
    elements.crowdActorSpeed.addEventListener("input", () => {
      const agent = getSelectedAgent();
      if (!agent) return;
      agent.setMaxSpeed(Number(elements.crowdActorSpeed.value));
      elements.crowdActorSpeedVal.textContent = `${agent.maxSpeed.toFixed(1)} m/s`;
    });
  }

  if (elements.crowdActorScale) {
    elements.crowdActorScale.addEventListener("input", () => {
      const actor = actorManager.activeActor;
      if (!actor) return;
      actor.setScale(Number(elements.crowdActorScale.value));
      elements.crowdActorScaleVal.textContent = `${actor.scale.toFixed(2)}x`;
      requestRender();
    });
  }

  if (elements.crowdActorCadence) {
    elements.crowdActorCadence.addEventListener("input", () => {
      const agent = getSelectedAgent();
      if (!agent) return;
      agent.setCadence(Number(elements.crowdActorCadence.value));
      elements.crowdActorCadenceVal.textContent = agent.cadence.toFixed(1);
    });
  }

  if (elements.crowdActorBehavior) {
    elements.crowdActorBehavior.addEventListener("change", () => {
      const agent = getSelectedAgent();
      if (!agent) return;
      agent.setBehavior(elements.crowdActorBehavior.value);
      refreshActorCard();
      renderActorsList();
    });
  }

  if (elements.actorAiNote) {
    elements.actorAiNote.addEventListener("click", (event) => {
      const action = event.target.closest("[data-ai]");
      if (!action) return;
      if (action.dataset.ai === "agents") {
        activatePanel("actors", "actors");
      } else if (action.dataset.ai === "spawn") {
        if (elements.crowdSimCount) elements.crowdSimCount.value = "6";
        elements.crowdSimSpawn?.click();
      }
    });
  }

  if (elements.crowdSimStep) {
    elements.crowdSimStep.addEventListener("click", () => {
      stepSimulation(1);
      stepAnchorClip(getSimulationTimeScale() / 60);
    });
  }

  if (elements.crowdSimSkip) {
    elements.crowdSimSkip.addEventListener("click", () => {
      stepSimulation(10);
      stepAnchorClip((10 * getSimulationTimeScale()) / 60);
    });
  }

  if (elements.crowdSimSpeed) {
    elements.crowdSimSpeed.addEventListener("change", () => {
      setSimulationTimeScale(Number(elements.crowdSimSpeed.value));
    });
  }

  // 3D Canvas Raycasting Click to Select Actor
  let canvasClickStart = { x: 0, y: 0 };
  function canvasNdc(e) {
    const rect = elements.canvas.getBoundingClientRect();
    return new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    );
  }
  // Expression-line scrub drag: pointerdown on a line captures the gesture
  // from OrbitControls (same precedent as the joint gizmo); moving along
  // the line's projection scrubs min→max, release without moving stays a
  // click (select + tooltip). Locked shapes are click-only.
  let lineDrag = null;
  const dragRay = new THREE.Raycaster();
  const dragRayPoint = new THREE.Vector3();
  const dragSegPoint = new THREE.Vector3();
  function endLineDrag() {
    lineDrag = null;
    controls.enabled = true;
    elements.canvas.style.cursor = "";
  }
  // Live write during a scrub: slider-`input` parity (no server sync yet).
  function writeExprDragValue(i, val) {
    const param = state.metadata.groups.expression[i];
    if (!param || isLocked(param)) return;
    recordBandwidthSaved(226611);
    param.value = val;
    elements.activeParameter.textContent =
      `expression / ${param.name} = ${formatNumber(val)}`;
    const actor = actorManager.activeActor;
    if (actor) actor.setParamValue("expression", i, val);
    if (!actor || actor.isAnchor) {
      evaluateBlendshapesClientSide();
    } else {
      refreshGpuAgents();
      requestRender();
    }
    const current = currentParameter();
    if (current && state.currentGroup === "expression" && current.index === i) {
      showParameter(current);
    }
  }
  // Release commit after a scrub: slider-`change` parity (server sync).
  function commitExprDragValue(i) {
    const param = state.metadata.groups.expression[i];
    if (!param || isLocked(param)) return;
    const actor = actorManager.activeActor;
    if (!actor || actor.isAnchor) {
      evaluateBlendshapesClientSide();
      syncBlendshapeServer("expression", i, param.value);
    } else {
      refreshGpuAgents();
      requestRender();
    }
  }
  function updateLineDrag(e) {
    if (e.pointerId !== lineDrag.pointerId) return;
    const dx = e.clientX - lineDrag.startX;
    const dy = e.clientY - lineDrag.startY;
    if (!lineDrag.active) {
      if (dx * dx + dy * dy < 16) return;
      // Commit to the drag: edit this shape (auto-switch tab), cache the
      // world segment (the head bone is static while scrubbing blendshapes).
      showActorTab("expression");
      selectParameter(lineDrag.expr);
      lineDrag.seg = exprVectorWorldSegment(lineDrag.expr);
      if (!lineDrag.seg) {
        endLineDrag();
        return;
      }
      cancelResetAnimation();
      lineDrag.active = true;
      elements.canvas.style.cursor = "grabbing";
    }
    dragRay.setFromCamera(canvasNdc(e), activeCamera);
    const { a, b } = lineDrag.seg;
    dragRay.ray.distanceSqToSegment(a, b, dragRayPoint, dragSegPoint);
    const len = a.distanceTo(b);
    const t = len < 1e-9 ? 0 : Math.min(1, Math.max(0, dragSegPoint.distanceTo(a) / len));
    const param = state.metadata.groups.expression[lineDrag.expr];
    writeExprDragValue(lineDrag.expr, param.min + t * (param.max - param.min));
    const detail = describeExprVector(lineDrag.expr);
    if (detail) {
      showCanvasTooltip(e, exprTooltipRows(detail));
      pinCanvasTooltip();
    }
  }
  elements.canvas.addEventListener("pointerdown", (e) => {
    canvasClickStart = { x: e.clientX, y: e.clientY };
    // A second finger aborts the scrub and yields to pinch-zoom.
    if (lineDrag) {
      endLineDrag();
      return;
    }
    if (e.pointerType === "mouse" && e.button !== 0) return;
    if (!elements.exprVectors?.checked) return;
    const hit = pickExprVector(canvasNdc(e), activeCamera);
    if (hit === null || hit === undefined) return;
    const param = state.metadata?.groups?.expression?.[hit];
    if (!param || isLocked(param)) return;
    lineDrag = { expr: hit, pointerId: e.pointerId, startX: e.clientX, startY: e.clientY, active: false, seg: null };
    controls.enabled = false;
  });
  function exprTooltipRows(d) {
    const rows = [
      d.name,
      `range ${formatNumber(d.min)} → ${formatNumber(d.max)}`,
    ];
    if (d.value !== null && d.value !== undefined) {
      rows.push(`now ${formatNumber(d.value)}`);
    }
    if (d.deltaMm) rows.push(`translation ${formatDeltaMm(d.deltaMm)} mm`);
    rows.push(`travel ${d.travelMm.toFixed(1)} mm · vertex #${d.vertex}`);
    rows.push(d.mirror ? `mirror: ${d.mirror}` : "mirror: —");
    return rows;
  }
  elements.canvas.addEventListener("pointerup", (e) => {
    if (lineDrag?.active) {
      const expr = lineDrag.expr;
      endLineDrag();
      commitExprDragValue(expr);
      return;
    }
    if (lineDrag) endLineDrag();
    const dx = e.clientX - canvasClickStart.x;
    const dy = e.clientY - canvasClickStart.y;
    if (dx * dx + dy * dy < 16) {
      const ndc = canvasNdc(e);
      // Expression lines first: a line hit selects the shape (tooltip +
      // list sync on the expression tab) and skips actor picking.
      if (elements.exprVectors?.checked) {
        const hit = pickExprVector(ndc, activeCamera);
        if (hit !== null && hit !== undefined) {
          if (setExprVectorSelected(hit)) requestRender();
          const detail = describeExprVector(hit);
          if (detail) {
            // Single click pins a persistent translation readout: hover no
            // longer hides it (dismiss on click-miss, Esc, toggle-off).
            showCanvasTooltip(e, exprTooltipRows(detail));
            pinCanvasTooltip();
          }
          if (state.currentGroup === "expression") selectParameter(hit);
          return;
        }
        if (setExprVectorSelected(null)) requestRender();
        dismissCanvasTooltip();
      }
      const picked = pickAgentAtNdc(ndc, activeCamera);
      if (picked !== undefined) {
        renderActorsList();
        refreshActorCard();
      }
    }
  });
  elements.canvas.addEventListener("pointercancel", () => {
    if (lineDrag?.active) commitExprDragValue(lineDrag.expr);
    if (lineDrag) endLineDrag();
  });

  elements.bodyColor.addEventListener("input", () => {
    // One path for every target: setColor updates the stored color (the LOD
    // paint restores from it) plus the live material (the anchor's is shared).
    const actor = actorManager.activeActor;
    if (actor) actor.setColor(elements.bodyColor.value);
    else bodyMaterial.color.set(elements.bodyColor.value);
    refreshActorCard();
    requestRender();
  });
  elements.backgroundColor.addEventListener("input", () => {
    clearHdriSelection();
    clearBackgroundPanorama(false);
    renderer.setClearColor(elements.backgroundColor.value, 1);
    elements.viewport.style.background = elements.backgroundColor.value;
    requestRender();
  });
  async function applyPanorama(url, name) {
    try {
      setStatus(`Loading panorama ${name}…`);
      await loadBackgroundPanorama(url, name);
      setEnvironmentIntensity(Number(elements.backgroundEnv.value));
      elements.viewport.style.background = "#000";
      setStatus(`Panorama ${name} active (background + env light).`);
    } catch (error) {
      setStatus(`Panorama failed: ${error?.message || error}`, "error");
    }
  }
  if (elements.backgroundFile) {
    elements.backgroundFile.addEventListener("change", () => {
      const file = elements.backgroundFile.files?.[0];
      if (!file) return;
      const url = URL.createObjectURL(file);
      clearHdriSelection();
      const tile = document.getElementById("hdri-upload");
      tile?.classList.add("is-selected");
      tile?.setAttribute("aria-selected", "true");
      applyPanorama(url, file.name).finally(() => URL.revokeObjectURL(url));
      elements.backgroundFile.value = "";
    });
  }
  // Built-in 360 environments: clickable thumbnails from /api/hdri.
  function clearHdriSelection() {
    elements.hdriGrid?.querySelectorAll(".is-selected").forEach((el) => {
      el.classList.remove("is-selected");
      el.setAttribute("aria-selected", "false");
    });
  }
  function prettyHdriName(name) {
    return name.replace(/\.[a-z0-9]+$/i, "").replace(/_\d+k$/i, "").replace(/_\d+$/i, "")
      .split("_").filter(Boolean)
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(" ");
  }
  async function refreshHdriGrid() {
    const grid = elements.hdriGrid;
    if (!grid) return;
    let files = [];
    try {
      const resp = await fetch("/api/hdri/list");
      const data = await resp.json();
      if (data.available) files = data.files ?? [];
    } catch {
      // Server folder missing: leave a quiet note, controls still work.
    }
    grid.replaceChildren();
    const upload = document.createElement("div");
    upload.id = "hdri-upload";
    upload.className = "hdri-thumb hdri-upload";
    upload.title = "Load a panorama file from disk";
    upload.tabIndex = 0;
    upload.setAttribute("role", "option");
    upload.setAttribute("aria-selected", "false");
    const plus = document.createElement("div");
    plus.className = "hdri-upload-plus";
    plus.textContent = "+";
    plus.setAttribute("aria-hidden", "true");
    const uploadCaption = document.createElement("span");
    uploadCaption.textContent = "Upload…";
    upload.append(plus, uploadCaption);
    const pick = () => elements.backgroundFile?.click();
    upload.addEventListener("click", pick);
    upload.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        pick();
      }
    });
    grid.append(upload);
    if (!files.length) {
      const note = document.createElement("span");
      note.className = "quiet";
      note.textContent = "No built-in environments found.";
      grid.append(note);
      return;
    }
    for (const file of files) {
      const label = prettyHdriName(file.name);
      // A div, not a button: buttons misreport content height to grid
      // track sizing (31px rows, captions clipped). Keyboard parity below.
      const btn = document.createElement("div");
      btn.className = "hdri-thumb";
      btn.title = file.name;
      btn.tabIndex = 0;
      btn.setAttribute("role", "option");
      btn.setAttribute("aria-selected", "false");
      const img = document.createElement("img");
      img.src = `/api/hdri/thumb/${encodeURIComponent(file.name)}`;
      img.alt = "";
      img.loading = "lazy";
      img.addEventListener("error", () => img.remove());
      const caption = document.createElement("span");
      caption.textContent = label;
      btn.append(img, caption);
      const choose = () => {
        clearHdriSelection();
        btn.classList.add("is-selected");
        btn.setAttribute("aria-selected", "true");
        applyPanorama(`/api/hdri/file/${encodeURIComponent(file.name)}`, file.name);
      };
      btn.addEventListener("click", choose);
      btn.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          choose();
        }
      });
      grid.append(btn); if (file.name === "aarfontein_sunrise_2k.exr" && !grid.querySelector(".is-selected")) choose();
    }
  }
  refreshHdriGrid();
  if (elements.poseFilter) {
    elements.poseFilter.addEventListener("input", renderPoseTiles);
  }
  if (elements.poseRelease) {
    elements.poseRelease.addEventListener("click", releasePosePreset);
  }
  refreshPoseGrid();
  refreshDecoGrids();
  bindEyesRange(
    elements.backgroundEnv,
    elements.backgroundEnvVal,
    (v) => v.toFixed(2),
    (v) => setEnvironmentIntensity(v),
  );
  elements.wireframe.addEventListener("change", () => {
    // Selection-scoped like Body color: the selected character only.
    actorManager.activeActor?.setWireframe(elements.wireframe.checked);
    requestRender();
  });
  elements.skeleton.addEventListener("change", () => {
    if (!suppressOverlayEvent) tabAutoOverlay.skeleton = false;
    skeletonOverlay.visible = elements.skeleton.checked;
    if (!elements.skeleton.checked) dismissCanvasTooltip();
    else syncDisplayedSkeleton();
    updateSphereVisibility();
    requestRender();
  });
  if (elements.exprVectors) {
    elements.exprVectors.addEventListener("change", () => {
      if (!suppressOverlayEvent) tabAutoOverlay.exprVectors = false;
      // Retry init on enable: a too-early boot call self-heals here.
      if (elements.exprVectors.checked && !initExprVectors()) {
        elements.exprVectors.checked = false;
        setStatus("Visual editing unavailable (see console).");
        return;
      }
      setExprVectorsVisible(elements.exprVectors.checked);
      if (!elements.exprVectors.checked) {
        setExprVectorHover(null);
        setExprVectorSelected(null);
        dismissCanvasTooltip();
      }
      requestRender();
    });
  }
  elements.grid.addEventListener("change", () => {
    gridHelper.visible = elements.grid.checked;
    requestRender();
  });
  elements.ring.addEventListener("change", () => {
    setSelectionRingVisible(elements.ring.checked);
  });
  elements.axes.addEventListener("change", () => {
    axesHelper.visible = elements.axes.checked;
    requestRender();
  });
  elements.frustum.addEventListener("change", () => {
    setFrustumCullingEnabled(elements.frustum.checked);
  });
  const WORLD_TABS = ["world", "camera", "performance", "crowd", "actors"];
  function switchWorldTab(name, { persist = true } = {}) {
    const next = WORLD_TABS.includes(name) ? name : "world";
    elements.worldTabs.querySelectorAll("[data-wtab]").forEach((tab) => {
      const active = tab.dataset.wtab === next;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    for (const panel of WORLD_TABS) {
      document.getElementById(`wtab-${panel}`).hidden = panel !== next;
    }
    if (persist) localStorage.setItem("mhr.worldTab", next);
  }
  elements.worldTabs.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-wtab]");
    if (tab) switchWorldTab(tab.dataset.wtab);
  });
  switchWorldTab(localStorage.getItem("mhr.worldTab"), { persist: false });
  elements.telemetryCollapse.addEventListener("click", () => {
    const collapsed = elements.telemetryBody.hidden;
    elements.telemetryBody.hidden = !collapsed;
    elements.telemetryCollapse.textContent = collapsed ? "▾" : "▸";
    localStorage.setItem("mhr.telemetryCollapsed", collapsed ? "0" : "1");
  });
  if (localStorage.getItem("mhr.telemetryCollapsed") === "1") {
    elements.telemetryBody.hidden = true;
    elements.telemetryCollapse.textContent = "▸";
  }
  if (elements.devTools) {
    elements.devTools.addEventListener("change", () => {
      setDevTools(elements.devTools.checked);
    });
  }
  if (elements.welcomeDev) {
    elements.welcomeDev.addEventListener("click", () => dismissWelcome(true));
  }
  if (elements.welcomePlay) {
    elements.welcomePlay.addEventListener("click", () => dismissWelcome(false));
  }
  applyDevTools();
  elements.canvas.addEventListener("pointermove", updateBoneTooltip);
  elements.canvas.addEventListener("pointermove", (e) => {
    if (lineDrag) {
      updateLineDrag(e);
      return;
    }
    if (!elements.exprVectors?.checked || e.buttons !== 0) {
      setExprVectorHover(null);
      return;
    }
    const hit = pickExprVector(canvasNdc(e), activeCamera);
    if (setExprVectorHover(hit)) requestRender();
    elements.canvas.style.cursor = hit === null ? "" : "pointer";
  });
  elements.canvas.addEventListener("pointerleave", () => {
    if (lineDrag?.active) commitExprDragValue(lineDrag.expr);
    if (lineDrag) endLineDrag();
    setExprVectorHover(null);
    elements.canvas.style.cursor = "";
  });
  elements.parameter.addEventListener("mouseover", (e) => {
    const item = e.target.closest(".parameter-item");
    if (setExprVectorHover(
      item && state.currentGroup === "expression" ? Number(item.dataset.index) : null,
    )) requestRender();
  });
  elements.parameter.addEventListener("mouseleave", () => setExprVectorHover(null));
  elements.canvas.addEventListener("pointerdown", dismissCanvasTooltip);
  elements.canvas.addEventListener("pointerleave", dismissCanvasTooltip);
  document.addEventListener("pointermove", (event) => {
    if (event.target !== elements.canvas) dismissCanvasTooltip();
  });
  window.addEventListener("blur", dismissCanvasTooltip);

  elements.viewButtons.forEach((button) => {
    button.addEventListener("click", () => {
      fitCamera(button.dataset.view, activeActorFocus(button.dataset.view));
      requestRender();
    });
  });
  elements.cameraReset.addEventListener("click", () => {
    fitCamera(state.lastView, activeActorFocus(state.lastView));
    requestRender();
  });
  elements.projection.addEventListener("click", () => {
    toggleProjection();
    syncGizmoCamera();
  });
  elements.cameraProjPersp?.addEventListener("click", () => {
    setProjection("perspective");
    syncGizmoCamera();
  });
  elements.cameraProjOrtho?.addEventListener("click", () => {
    setProjection("orthographic");
    syncGizmoCamera();
  });
  syncProjectionUI();
  setFollowProvider(() => actorManager.activeActor?.position ?? null);
  elements.cameraFollow?.addEventListener("change", () => {
    setFollow(elements.cameraFollow.checked);
  });
  // Effects (WebGPU-only; the chain lives in scene.js).
  if (!state.isWebGPU) {
    const reason = "Post-processing needs WebGPU (WebGL fallback active)";
    for (const el of [elements.fxDof, elements.fxDofFocus, elements.fxDofRange,
        elements.fxDofBokeh, elements.fxBloom, elements.fxBloomStrength,
        elements.fxBloomThreshold]) {
      setDisabledWithReason(el, true, reason);
    }
  }
  elements.fxDof?.addEventListener("change", () => {
    const on = elements.fxDof.checked;
    setEffect("dof", on);
    if (on && elements.fxDofFocus) {
      elements.fxDofFocus.value = getEffect("focus");
      if (elements.fxDofFocusVal) elements.fxDofFocusVal.textContent = `${getEffect("focus").toFixed(1)} m`;
    }
  });
  elements.fxBloom?.addEventListener("change", () => {
    setEffect("bloom", elements.fxBloom.checked);
  });
  if (elements.fxDofFocus) {
    bindEyesRange(elements.fxDofFocus, elements.fxDofFocusVal,
      (v) => `${v.toFixed(1)} m`, (v) => setEffect("focus", v));
  }
  if (elements.fxDofRange) {
    bindEyesRange(elements.fxDofRange, elements.fxDofRangeVal,
      (v) => `${v.toFixed(1)} m`, (v) => setEffect("range", v));
  }
  if (elements.fxDofBokeh) {
    bindEyesRange(elements.fxDofBokeh, elements.fxDofBokehVal,
      (v) => v.toFixed(1), (v) => setEffect("bokeh", v));
  }
  if (elements.fxBloomStrength) {
    bindEyesRange(elements.fxBloomStrength, elements.fxBloomStrengthVal,
      (v) => v.toFixed(2), (v) => setEffect("strength", v));
  }
  if (elements.fxBloomThreshold) {
    bindEyesRange(elements.fxBloomThreshold, elements.fxBloomThresholdVal,
      (v) => v.toFixed(2), (v) => setEffect("threshold", v));
  }
  if (elements.cameraNear) {
    const applyCamInputs = () => {
      setCameraParams({
        near: parseFloat(elements.cameraNear.value),
        far: parseFloat(elements.cameraFar.value),
        fov: parseFloat(elements.cameraFov.value),
      });
    };
    for (const input of [elements.cameraNear, elements.cameraFar, elements.cameraFov]) {
      input.addEventListener("change", applyCamInputs);
    }
    syncCameraInputs();
  }

  elements.screenshot.addEventListener("click", () => {
    renderNow();
    const link = document.createElement("a");
    link.download = `mhr-${state.currentGroup}-${currentParameter().name}.png`;
    link.href = elements.canvas.toDataURL("image/png");
    link.click();
  });
  elements.fullscreen.addEventListener("click", async () => {
    if (!document.fullscreenElement) {
      await elements.viewport.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
    resizeRenderer();
    requestRender();
  });

  window.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, button")) return;
    if (event.key === "Escape" && hasJointSelection()) {
      clearJointSelection();
      return;
    }
    if (event.key === "Escape") {
      dismissCanvasTooltip();
      if (setExprVectorSelected(null)) requestRender();
      return;
    }
    if (hasJointSelection()) {
      const modeKey = event.key.toLowerCase();
      if (modeKey === "g" || modeKey === "r" || modeKey === "s") {
        setJointMode(modeKey === "g" ? "move" : modeKey === "r" ? "rotate" : "scale");
        return;
      }
    }
    if (event.key.toLowerCase() === "f") {
      fitCamera("front", activeActorFocus("front"));
      requestRender();
    }
    if (event.key.toLowerCase() === "r") {
      fitCamera(state.lastView, activeActorFocus(state.lastView));
      requestRender();
    }
  });
  wireEyesMouth();
  wireBenchmark();
}

function formatTris(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}k`;
  return `${Math.round(n)}`;
}

function renderBenchTable() {
  if (!elements.benchBody) return;
  const runs = loadRuns();
  if (runs.length === 0) {
    elements.benchBody.innerHTML = '<tr><td colspan="10" class="quiet">No runs yet.</td></tr>';
    if (elements.benchHeadline) elements.benchHeadline.textContent = "—";
    return;
  }
  const rows = runs.slice(0, 20).map((r) => {
    const date = (r.date || "").slice(5, 16).replace("T", " ");
    const mix = Object.entries(r.lodMix || {}).map(([l, n]) => `L${l}:${n}`).join(" ") || "—";
    const title = `${r.engine || "?"} · ${r.gpuShort || r.gpu || "?"} · ` +
      `${r.frames || 0} frames · frame avg ${(r.frameAvg || 0).toFixed(2)} / p50 ${(r.frameP50 || 0).toFixed(2)} ms · ` +
      `tick agents ${(r.agentsAvg || 0).toFixed(2)} / post ${(r.postAvg || 0).toFixed(2)} / ui ${(r.uiAvg || 0).toFixed(2)} ms · ` +
      `mix ${mix}${r.cpu ? ` + ${r.cpu} CPU` : ""}` +
      (r.gpuExecSamples > 0 ? ` · gpuExec ${(r.gpuExecAvg || 0).toFixed(2)}ms` : " · gpuExec n/a") +
      (r.params?.mode === "ramp" ? ` · mode ramp (step ${r.params.step}, floor ${r.params.minFps}, final ${r.finalAgents ?? "?"})` : "");
    return `<tr title="${title.replace(/"/g, "&quot;")}">` +
      `<td>${r.device || "?"}</td><td>${r.gpuShort || "?"}</td>` +
      `<td>${r.params?.agents ?? "?"}</td><td>${r.params?.lod ?? "?"}</td>` +
      `<td>${(r.fpsAvg || 0).toFixed(0)}</td><td>${(r.frameP95 || 0).toFixed(1)}</td>` +
      `<td>${(r.tickAvg || 0).toFixed(2)}</td><td>${Math.round(r.callsAvg || 0)}</td>` +
      `<td>${formatTris(r.trisAvg || 0)}</td><td>${date}</td></tr>`;
  }).join("");
  elements.benchBody.innerHTML = rows;
  const latest = runs[0];
  if (elements.benchHeadline && latest) {
    const latestRamp = (latest.params?.mode ?? "fixed") === "ramp";
    elements.benchHeadline.textContent =
      (latestRamp ? `ramp max ${latest.params?.agents ?? "?"} @ ` : `${latest.params?.agents ?? "?"} agents · `) +
      `${(latest.fpsAvg || 0).toFixed(0)} fps · tick ${(latest.tickAvg || 0).toFixed(2)}ms`;
  }
}

// URL-driven benchmark: ?bench[=1]&agents=&seconds=&lod=&device= — the same
// path as the Run button, so phones/Quest can trigger runs without menus.
function triggerUrlBenchmark() {
  let params;
  try {
    params = new URLSearchParams(location.search);
  } catch {
    return;
  }
  if (!params.has("bench") && !params.has("benchmark")) return;
  const flag = (params.get("bench") ?? params.get("benchmark") ?? "").toLowerCase();
  if (["0", "false", "off", "no"].includes(flag)) return;
  const setIf = (el, value, ok) => {
    if (el && value !== null && (!ok || ok(value))) el.value = value;
  };
  setIf(elements.benchAgents, params.get("agents"), (v) => /^\d+$/.test(v));
  setIf(elements.benchSeconds, params.get("seconds"), (v) => /^\d+(\.\d+)?$/.test(v));
  setIf(elements.benchLod, params.get("lod"), (v) =>
    [...elements.benchLod.options].some((o) => o.value === v));
  setIf(elements.benchDevice, params.get("device"), (v) => v.length <= 32);
  const rampAsked = (params.get("ramp") ?? params.get("mode") ?? "").toLowerCase();
  if ((params.has("ramp") || params.has("mode")) && elements.benchMode) {
    elements.benchMode.value = ["1", "true", "yes", "ramp"].includes(rampAsked) ? "ramp" : "fixed";
    elements.benchMode.dispatchEvent(new Event("change"));
  }
  setIf(elements.benchStep, params.get("step"), (v) => /^\d+$/.test(v));
  setIf(elements.benchInterval, params.get("interval"), (v) => /^\d+(\.\d+)?$/.test(v));
  setIf(elements.benchMinfps, params.get("minfps"), (v) => /^\d+$/.test(v));
  if (!isDevTools()) setDevTools(true); // Bench UI lives in Performance.
  try {
    const url = new URL(location.href);
    for (const key of ["bench", "benchmark", "agents", "seconds", "lod", "device", "ramp", "mode", "step", "interval", "minfps"]) {
      url.searchParams.delete(key);
    }
    history.replaceState(null, "", url);
  } catch { /* a reload just reruns */ }
  setStatus("Benchmark autostart from URL.");
  elements.benchRun?.click();
}

let benchRunning = false;

function wireBenchmark() {
  if (!elements.benchRun) return;
  if (elements.benchDevice && !elements.benchDevice.value) {
    elements.benchDevice.value = detectEngine().gpuShort;
  }
  renderBenchTable();
  elements.benchRun.addEventListener("click", async () => {
    if (benchRunning) {
      cancelBenchmark();
      elements.benchRun.disabled = true;
      elements.benchRun.textContent = "Cancelling…";
      return;
    }
    benchRunning = true;
    elements.benchRun.textContent = "Cancel";
    window.mhr?.dockApi?.getPanel?.("performance")?.api?.setActive?.();
    const phase = (text) => {
      if (elements.benchStatus) elements.benchStatus.textContent = text;
      setStatus(`Benchmark: ${text}`);
      // Panel DOM renders lazily: re-assert until it exists.
      document.getElementById("bench-details")?.setAttribute("open", "");
    };
    const rampMode = elements.benchMode?.value !== "fixed";
    try {
      const run = await (rampMode ? runRampBenchmark(
        {
          step: elements.benchStep?.value,
          intervalMs: (Number(elements.benchInterval?.value) || 2.5) * 1000,
          minFps: elements.benchMinfps?.value,
          lod: elements.benchLod?.value,
          device: elements.benchDevice?.value,
        },
        phase,
      ) : runBenchmark(
        {
          agents: elements.benchAgents?.value,
          seconds: elements.benchSeconds?.value,
          lod: elements.benchLod?.value,
          device: elements.benchDevice?.value,
        },
        phase,
      ));
      if (run) {
        saveRun(run);
        renderBenchTable();
        const rampDone = (run.params.mode ?? "fixed") === "ramp";
        if (elements.benchStatus) {
          elements.benchStatus.textContent = rampDone
            ? `done: ${run.params.agents} max · ${run.fpsAvg.toFixed(0)} fps · p95 ${run.frameP95.toFixed(1)}ms`
            : `done: ${run.fpsAvg.toFixed(0)} fps · p95 ${run.frameP95.toFixed(1)}ms · tick ${run.tickAvg.toFixed(2)}ms`;
        }
        setStatus(rampDone
          ? `Benchmark saved (${run.device}, ramp max ${run.params.agents} agents)`
          : `Benchmark saved (${run.device}, ${run.params.agents} agents)`);
      } else if (elements.benchStatus) {
        elements.benchStatus.textContent = "cancelled.";
      }
    } catch (err) {
      console.error("[bench]", err);
      if (elements.benchStatus) elements.benchStatus.textContent = `failed: ${err.message}`;
    } finally {
      benchRunning = false;
      elements.benchRun.disabled = false;
      elements.benchRun.textContent = "Run";
    }
  });
  const syncBenchMode = () => {
    const ramp = elements.benchMode?.value !== "fixed";
    elements.benchAgents?.closest("label")?.toggleAttribute("hidden", ramp);
    elements.benchSeconds?.closest("label")?.toggleAttribute("hidden", ramp);
    document.getElementById("bench-ramp-rows")?.toggleAttribute("hidden", !ramp);
  };
  elements.benchMode?.addEventListener("change", syncBenchMode);
  syncBenchMode();
  if (elements.benchExport) {
    elements.benchExport.addEventListener("click", () => {
      exportRuns();
      setStatus("Benchmark runs exported");
    });
  }
  if (elements.benchImport) {
    elements.benchImport.addEventListener("change", async () => {
      const file = elements.benchImport.files?.[0];
      if (!file) return;
      try {
        const { added, total } = await importRuns(file);
        renderBenchTable();
        setStatus(`Imported ${added} run(s), ${total} total`);
      } catch (err) {
        setStatus(`Import failed: ${err.message}`);
      } finally {
        elements.benchImport.value = "";
      }
    });
  }
  if (elements.benchClear) {
    elements.benchClear.addEventListener("click", () => {
      clearRuns();
      renderBenchTable();
      setStatus("Benchmark history cleared");
    });
  }
}

function setSwitchButton(button, stateLabel, on) {
  button.classList.toggle("is-on", on);
  button.setAttribute("aria-checked", String(on));
  stateLabel.textContent = on ? "On" : "Off";
}

function bindEyesRange(input, readout, format, onInput) {
  if (!input) return;
  input.addEventListener("input", () => {
    const val = Number(input.value);
    if (readout) readout.textContent = format(val);
    onInput(val);
  });
}

function wireEyesMouth() {
  if (elements.eyesAutoblink) {
    elements.eyesAutoblink.addEventListener("click", () => {
      const on = !eyesOral.autoBlink;
      eyesOral.setAutoBlink(on);
      setSwitchButton(elements.eyesAutoblink, elements.eyesAutoblinkState, on);
    });
  }
  if (elements.eyesBlink) {
    elements.eyesBlink.addEventListener("click", () => {
      const target = actorManager.activeActor;
      if (!target || target.isAnchor) {
        eyesOral.blink();
        return;
      }
      if (!crowdEyes.triggerAgentBlink(target)) return;
      if (isAgentSimulationActive()) return;
      const t0 = performance.now();
      const drive = () => {
        crowdEyes.update(simulation.agents, simulation.simTime, performance.now());
        crowdMouth.update(simulation.agents);
        refreshGpuAgents();
        requestRender();
        if (performance.now() - t0 < 300) requestAnimationFrame(drive);
      };
      requestAnimationFrame(drive);
    });
  }
  if (elements.actorGazeWander) {
    elements.actorGazeWander.addEventListener("click", () => {
      const target = actorManager.activeActor;
      if (!target || target.isAnchor) return;
      crowdEyes.setAgentGaze(target.id, null);
      refreshActorCard();
      refreshCrowdEyesOnce();
    });
  }
  if (elements.crowdEyesToggle) {
    elements.crowdEyesToggle.addEventListener("click", () => {
      const on = !crowdEyes.enabled;
      crowdEyes.setEnabled(on);
      setSwitchButton(elements.crowdEyesToggle, elements.crowdEyesToggleState, on);
      setStatus(`Crowd eyes ${on ? "enabled" : "disabled"}`);
      requestRender();
    });
  }
  if (elements.crowdBlinkToggle) {
    elements.crowdBlinkToggle.addEventListener("click", () => {
      const on = !crowdEyes.blinkEnabled;
      crowdEyes.setBlinkEnabled(on);
      setSwitchButton(elements.crowdBlinkToggle, elements.crowdBlinkToggleState, on);
    });
  }
  const applyGazeFace = () => {
    setActiveGaze(Number(elements.gazeYaw.value), Number(elements.gazePitch.value));
  };
  bindEyesRange(elements.gazeYaw, elements.gazeYawVal, (v) => `${v}°`, applyGazeFace);
  bindEyesRange(elements.gazePitch, elements.gazePitchVal, (v) => `${v}°`, applyGazeFace);
  // Pupil is per-actor: actor #0 drives its material uniform, agents store a
  // value the instanced eyes read every tick. Iris stays actor-#0-only.
  const applyEyeColors = () => eyesOral.setEyeShader({
    pupil: Number(elements.eyesPupil.value),
    irisInHex: elements.eyesIrisIn.value,
    irisOutHex: elements.eyesIrisOut.value,
  });
  const applyPupil = (v) => {
    const target = actorManager.activeActor;
    if (!target || target.isAnchor) {
      applyEyeColors();
      return;
    }
    target.pupil = v;
    refreshCrowdEyesOnce();
  };
  bindEyesRange(elements.eyesPupil, elements.eyesPupilVal, (v) => v.toFixed(2), applyPupil);
  for (const el of [elements.eyesIrisIn, elements.eyesIrisOut]) {
    if (el) el.addEventListener("input", applyEyeColors);
  }
  const swatchBox = document.querySelector("#iris-swatches");
  if (swatchBox) {
    for (const pr of IRIS_PRESETS) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "iris-swatch";
      b.title = pr.name;
      b.setAttribute("aria-label", `${pr.name} iris`);
      b.innerHTML = `<i style="background:radial-gradient(circle at 50% 50%, #000 0 22%, ${pr.inner} 26%, ${pr.outer} 100%)"></i>`;
      b.addEventListener("click", () => {
        elements.eyesIrisIn.value = pr.inner;
        elements.eyesIrisOut.value = pr.outer;
        applyEyeColors();
      });
      swatchBox.appendChild(b);
    }
  }
  bindEyesRange(
    elements.mouthOpen,
    elements.mouthOpenVal,
    (v) => v.toFixed(2),
    (v) => {
      const target = actorManager.activeActor;
      if (!target || target.isAnchor) {
        eyesOral.setMouthOpen(v);
        return;
      }
      if (!target.isGpuAgent) return;
      const idx = crowdEyes.jawDropIndex();
      if (idx < 0 || target.expressionValues.length <= idx) return;
      target.expressionValues[idx] = v;
      if (!isAgentSimulationActive()) {
        refreshGpuAgents();
        requestRender();
      }
    },
  );
  const applyTeeth = () => eyesOral.setTeeth({
    count: Math.min(
      14,
      Math.max(8, Math.round(Number(elements.teethCount.value) / 2) * 2),
    ),
    crooked: Number(elements.teethCrooked.value),
    gap: Number(elements.teethGap.value),
    overbite: Number(elements.teethOverbite.value),
    fangs: Number(elements.teethFangs.value),
  });
  if (elements.teethCount) {
    elements.teethCount.addEventListener("input", applyTeeth);
  }
  if (elements.fittedTeeth) {
    elements.fittedTeeth.addEventListener("change", () => {
      eyesOral.setFittedTeeth(elements.fittedTeeth.checked);
    });
  }
  if (elements.decoBrowOn) {
    elements.decoBrowOn.addEventListener("change", () => {
      eyesOral.setBrowOn(elements.decoBrowOn.checked);
    });
  }
  if (elements.decoLashOn) {
    elements.decoLashOn.addEventListener("change", () => {
      eyesOral.setLashOn(elements.decoLashOn.checked);
    });
  }
  if (elements.decoBrowTint) {
    elements.decoBrowTint.addEventListener("input", () => {
      eyesOral.setBrowTint(elements.decoBrowTint.value);
    });
  }
  if (elements.decoLashTint) {
    elements.decoLashTint.addEventListener("input", () => {
      eyesOral.setLashTint(elements.decoLashTint.value);
    });
  }
  bindEyesRange(elements.teethCrooked, elements.teethCrookedVal, (v) => v.toFixed(2), applyTeeth);
  bindEyesRange(elements.teethGap, elements.teethGapVal, (v) => v.toFixed(2), applyTeeth);
  bindEyesRange(elements.teethOverbite, elements.teethOverbiteVal, (v) => v.toFixed(2), applyTeeth);
  bindEyesRange(elements.teethFangs, elements.teethFangsVal, (v) => v.toFixed(2), applyTeeth);
  bindEyesRange(elements.teethSize, elements.teethSizeVal, (v) => `${v.toFixed(2)}x`, (v) => eyesOral.setFittedTeethScale(v));
  const applyTongue = () => eyesOral.setTongue({
    out: Number(elements.tongueOut.value),
    curl: Number(elements.tongueCurl.value),
    width: Number(elements.tongueWidth.value),
    roll: Number(elements.tongueRoll.value),
  });
  bindEyesRange(elements.tongueOut, elements.tongueOutVal, (v) => v.toFixed(2), applyTongue);
  bindEyesRange(elements.tongueCurl, elements.tongueCurlVal, (v) => v.toFixed(2), applyTongue);
  bindEyesRange(elements.tongueWidth, elements.tongueWidthVal, (v) => v.toFixed(2), applyTongue);
  bindEyesRange(elements.tongueRoll, elements.tongueRollVal, (v) => v.toFixed(2), applyTongue);
  wireClothing();
}

function wireClothing() {
  if (elements.clothToggle) {
    elements.clothToggle.addEventListener("click", () => {
      const on = shirt.setEnabled(!shirt.enabled);
      setSwitchButton(elements.clothToggle, elements.clothToggleState, on);
      setStatus(`Procedural shirt ${on ? "enabled" : "disabled"}`);
    });
  }
  bindEyesRange(
    elements.clothThickness,
    elements.clothThicknessVal,
    (v) => `${v.toFixed(0)} mm`,
    (v) => {
      shirt.setThickness(v / 1000);
      shorts.setThickness(Math.max(0.002, v / 1000 - 0.002));
    },
  );
  if (elements.clothColor) {
    elements.clothColor.addEventListener("input", () => shirt.setColor(elements.clothColor.value));
  }
  if (elements.clothShortsToggle) {
    elements.clothShortsToggle.addEventListener("click", () => {
      const on = shorts.setEnabled(!shorts.enabled);
      setSwitchButton(elements.clothShortsToggle, elements.clothShortsToggleState, on);
      setStatus(`Procedural shorts ${on ? "enabled" : "disabled"}`);
    });
  }
  if (elements.clothShortsColor) {
    elements.clothShortsColor.addEventListener("input", () => shorts.setColor(elements.clothShortsColor.value));
  }
  if (elements.clothRefit) {
    elements.clothRefit.addEventListener("click", () => {
      const ok = (!shirt.enabled || shirt.refit()) && (!shorts.enabled || shorts.refit());
      if (ok) setStatus("Garments refit to current body shape.");
      else setStatus("Garment refit failed (no body mesh).", "error");
    });
  }
}

await initRenderer();
wireEventListeners();
const menubarActions = {
  snapshot: () => elements.screenshot?.click(),
  fullscreen: () => elements.fullscreen?.click(),
  toggleHeader: () => setHeaderHidden(!document.body.classList.contains("topbar-hidden")),
  isHeaderHidden: () => document.body.classList.contains("topbar-hidden"),
  resetLayout: () => window.mhr.resetDockLayout?.(),
  shortcuts: () => document.getElementById("dlg-shortcuts")?.showModal(),
  about: () => document.getElementById("dlg-about")?.showModal(),
  setStatus,
};
try {
  const { initDock, PANES, setDevPanelsVisible } = await import("./js/dock.js");
  setDevPanelsVisibleFn = setDevPanelsVisible;
  initDock({
    dev: isDevTools(),
    onPanelActivated: (id) => {
      if (PARAM_TABS.includes(id)) showParamGroup(id); if (id === "identity") syncIdentityGrid(); syncTabAutoOverlays(id);
    },
  });
  initMenubar({ dockApi: window.mhr?.dockApi ?? null, panes: PANES, actions: menubarActions });
  if (NOUI) {
    for (const panel of [...(window.mhr?.dockApi?.panels ?? [])]) {
      if (panel.id !== "viewport") panel.api.close();
    }
  }
} catch (err) {
  console.warn("[dock] dockview unavailable, static layout:", err?.message ?? err);
  initMenubar({ dockApi: null, panes: [], actions: menubarActions });
}
initGizmo();
new ResizeObserver(() => {
  resizeRenderer();
  drawPerfGraph();
  requestRender();
}).observe(elements.viewport);
resizeRenderer();
requestRender();
initialize().finally(() => showWelcomeIfFirstRun());
