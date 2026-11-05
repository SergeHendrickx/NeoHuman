/**
 * Identity parameter grid: all 45 shape sliders at once as a flat list in
 * region order. Each cell shows "Region NN" (values are unitless); the full
 * identifier and live value appear on hover, and the card shade shifts
 * whenever the region changes. Commits through the editor's identity path.
 */
import { state, isLocked } from "./state.js";
import { actorManager } from "./actor.js";
import { evaluateBlendshapesClientSide } from "./rig.js";
import { syncBlendshapeServer } from "./sync.js";
import { refreshGpuAgents } from "./crowd.js";
import { requestRender } from "./scene.js";
import { formatNumber } from "./utils.js";

const GROUP = "identity";
const EPSILON = 1e-5;

let rows = [];

export function identityDefault(param) {
  return Math.min(param.max, Math.max(param.min, 0));
}

/** "identity_07" -> "07"; falls back to the 1-based index. */
function shortNumber(param) {
  const match = /(\d+)\s*$/.exec(param.name ?? "");
  return match ? match[1] : String(param.index + 1).padStart(2, "0");
}

function params() {
  return state.metadata?.groups?.identity ?? [];
}

function rowValue(param) {
  const actor = actorManager.activeActor;
  const value = actor ? actor.getParamValue(GROUP, param.index) : param.value;
  param.value = value;
  return value;
}

function refreshRow(row) {
  const value = rowValue(row.param);
  const locked = isLocked(row.param);
  row.slider.disabled = locked;
  row.reset.disabled = locked;
  row.slider.value = value;
  const tip = `${row.param.name} · ${row.param.region ?? "—"} · ${formatNumber(value)}`;
  row.slider.title = tip;
  row.cell.title = tip;
  const modified = Math.abs(value - identityDefault(row.param)) > EPSILON;
  row.reset.classList.toggle("is-modified", modified && !locked);
  row.cell.classList.toggle("is-locked", locked);
}

function commitRow(row, value) {
  if (isLocked(row.param) || !Number.isFinite(value)) {
    refreshRow(row);
    return;
  }
  row.param.value = value;
  const actor = actorManager.activeActor;
  if (actor) {
    actor.setParamValue(GROUP, row.param.index, value);
  }
  if (!actor || actor.isAnchor) {
    evaluateBlendshapesClientSide();
    syncBlendshapeServer(GROUP, row.param.index, value);
  } else {
    refreshGpuAgents();
    requestRender();
  }
  refreshRow(row);
}

function resetAllIdentity() {
  const list = params().filter((p) => !isLocked(p));
  if (list.length === 0) return;
  const actor = actorManager.activeActor;
  for (const param of list) {
    const target = identityDefault(param);
    param.value = target;
    if (actor) actor.setParamValue(GROUP, param.index, target);
  }
  if (!actor || actor.isAnchor) {
    evaluateBlendshapesClientSide();
    fetch("/api/deform", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "reset_group",
        group: GROUP,
        index: 0,
        value: null,
      }),
    }).catch(() => {});
  } else {
    refreshGpuAgents();
  }
  syncIdentityGrid();
  requestRender();
}

function buildRow(param, alt) {
  const cell = document.createElement("div");
  cell.className = `ident-cell${alt ? " is-alt" : ""}`;
  const name = document.createElement("span");
  name.className = "ident-name";
  name.textContent = `${param.region ?? "Other"} ${shortNumber(param)}`;
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = param.min;
  slider.max = param.max;
  slider.step = Math.max((param.max - param.min) / 600, 0.0001);
  slider.setAttribute("aria-label", `${param.name} slider`);
  const reset = document.createElement("button");
  reset.type = "button";
  reset.className = "ident-reset";
  reset.textContent = "↺";
  reset.title = `Reset ${param.name}`;
  reset.setAttribute("aria-label", `Reset ${param.name}`);
  cell.append(name, slider, reset);
  const row = { param, cell, slider, reset };
  slider.addEventListener("input", () => commitRow(row, Number(slider.value)));
  reset.addEventListener("click", () => commitRow(row, identityDefault(param)));
  return row;
}

/**
 * Flat list in identityRegions order (unmatched trailing), with the alt
 * shade flipping at every region boundary. Mirrors the list's grouping.
 */
function orderedParams(list) {
  const regions = state.metadata?.identityRegions ?? [];
  if (regions.length === 0) return list.map((param) => ({ param, alt: false }));
  const byRegion = new Map();
  const unmatched = [];
  for (const param of list) {
    if (param.region && regions.includes(param.region)) {
      if (!byRegion.has(param.region)) byRegion.set(param.region, []);
      byRegion.get(param.region).push(param);
    } else {
      unmatched.push(param);
    }
  }
  const ordered = [];
  let alt = false;
  for (const region of regions) {
    const items = byRegion.get(region);
    if (!items?.length) continue;
    for (const param of items) ordered.push({ param, alt });
    alt = !alt;
  }
  for (const param of unmatched) ordered.push({ param, alt });
  return ordered;
}

export function initIdentityGrid() {
  document.getElementById("identity-reset-all")
    ?.addEventListener("click", resetAllIdentity);
  syncIdentityGrid();
}

/** Build once metadata is ready; refresh values on tab show / actor switch. */
export function syncIdentityGrid() {
  const grid = document.getElementById("identity-grid");
  const list = params();
  if (!grid || list.length === 0) return;
  if (rows.length !== list.length) {
    grid.replaceChildren();
    rows = [];
    for (const { param, alt } of orderedParams(list)) {
      const row = buildRow(param, alt);
      grid.append(row.cell);
      rows.push(row);
    }
  }
  for (const row of rows) refreshRow(row);
}
