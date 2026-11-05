import {
  currentParameter,
  currentParameters,
  elements,
  isLocked,
  state,
} from "./state.js";
import { formatNumber } from "./utils.js";
import { actorManager } from "./actor.js";
import { setExprVectorSelected } from "./exprVectors.js";

export function buildCategories() {
  const categories = [
    "All categories",
    ...new Set(state.metadata.groups.pose.map((parameter) => parameter.category)),
  ];
  elements.category.replaceChildren();
  for (const category of categories) {
    const option = document.createElement("option");
    option.value = category === "All categories" ? "" : category;
    option.textContent = category;
    elements.category.appendChild(option);
  }
}

export function filteredParameters() {
  const query = elements.search.value.trim().toLowerCase();
  const category = elements.category.value;
  return currentParameters().filter((parameter) => {
    const matchesQuery =
      !query ||
      parameter.name.toLowerCase().includes(query) ||
      parameter.label.toLowerCase().includes(query);
    const matchesCategory =
      state.currentGroup !== "pose" || !category || parameter.category === category;
    return matchesQuery && matchesCategory;
  });
}

function createParameterItem(parameter) {
  const locked = isLocked(parameter);
  const item = document.createElement("button");
  item.type = "button";
  item.className = `parameter-item${locked ? " is-locked" : ""}`;
  item.dataset.index = String(parameter.index);
  item.setAttribute("role", "option");
  item.setAttribute("aria-selected", "false");
  if (locked) {
    const reason = parameter.lockedReason || "Locked by model";
    item.title = reason;
    item.setAttribute("aria-label", `${parameter.label}, ${reason}`);
  }

  const label = document.createElement("span");
  label.className = "parameter-item-label";
  label.textContent = state.currentGroup === "pose"
    ? parameter.name
    : parameter.label;
  item.appendChild(label);

  if (state.currentGroup === "pose") {
    const lockSlot = document.createElement("span");
    lockSlot.className = "parameter-lock-slot";
    if (locked) {
      const lock = document.createElement("span");
      lock.className = "parameter-lock";
      lock.setAttribute("aria-hidden", "true");
      lockSlot.appendChild(lock);
    }
    item.appendChild(lockSlot);

    const category = document.createElement("span");
    category.className = "parameter-item-category";
    category.textContent = parameter.category;
    item.appendChild(category);
  } else if (locked) {
    const lock = document.createElement("span");
    lock.className = "parameter-lock";
    lock.setAttribute("aria-hidden", "true");
    item.appendChild(lock);
  }
  return item;
}

function createRegionHeader(region, count) {
  const header = document.createElement("div");
  header.className = "parameter-group-header";
  header.setAttribute("role", "presentation");
  header.textContent = `${region} · ${count}`;
  return header;
}

function appendGroupedItems(filtered) {
  const regions =
    state.metadata[
      state.currentGroup === "identity" ? "identityRegions" : "expressionRegions"
    ] || [];
  const byRegion = new Map();
  const unmatched = [];
  for (const parameter of filtered) {
    const region = parameter.region;
    if (region && regions.includes(region)) {
      if (!byRegion.has(region)) byRegion.set(region, []);
      byRegion.get(region).push(parameter);
    } else {
      unmatched.push(parameter);
    }
  }
  for (const region of regions) {
    const items = byRegion.get(region);
    if (!items || !items.length) continue;
    elements.parameter.appendChild(createRegionHeader(region, items.length));
    for (const parameter of items) {
      elements.parameter.appendChild(createParameterItem(parameter));
    }
  }
  if (unmatched.length) {
    elements.parameter.appendChild(createRegionHeader("Other", unmatched.length));
    for (const parameter of unmatched) {
      elements.parameter.appendChild(createParameterItem(parameter));
    }
  }
}

export function rebuildParameterList() {
  const selected = currentParameter();
  const filtered = filteredParameters();
  elements.parameter.replaceChildren();

  if (state.currentGroup === "identity" || state.currentGroup === "expression") {
    appendGroupedItems(filtered);
  } else {
    for (const parameter of filtered) {
      elements.parameter.appendChild(createParameterItem(parameter));
    }
  }

  elements.filteredCount.textContent = `${filtered.length} shown`;
  let next = selected && filtered.some((item) => item.index === selected.index)
    ? selected
    : filtered[0];

  if (!next) {
    elements.slider.disabled = true;
    elements.value.disabled = true;
    elements.parameter.setAttribute("aria-disabled", "true");
    elements.value.value = "";
    elements.rangeReadout.textContent = "—";
    elements.jointList.textContent = "No matching parameters";
    return;
  }

  state.selectedIndex[state.currentGroup] = next.index;
  elements.parameter.setAttribute("aria-disabled", String(state.requestPending));
  elements.parameter.querySelectorAll(".parameter-item").forEach((item) => {
    const selectedItem = Number(item.dataset.index) === next.index;
    item.classList.toggle("is-selected", selectedItem);
    item.setAttribute("aria-selected", String(selectedItem));
    item.disabled = state.requestPending;
    if (selectedItem) item.scrollIntoView({ block: "nearest" });
  });
  showParameter(next);
}

export function showParameter(parameter) {
  const actor = actorManager.activeActor;
  const val = actor ? actor.getParamValue(state.currentGroup, parameter.index) : parameter.value;
  parameter.value = val;
  const locked = isLocked(parameter);
  const span = Math.max(parameter.max - parameter.min, 0.0001);
  const step = Math.max(span / 600, 0.0001);
  elements.slider.min = parameter.min;
  elements.slider.max = parameter.max;
  elements.slider.step = step;
  elements.slider.value = val;
  elements.value.min = parameter.min;
  elements.value.max = parameter.max;
  elements.value.step = step;
  elements.value.value = formatNumber(val);
  elements.rangeReadout.textContent =
    `${formatNumber(parameter.min, 2)} … ${formatNumber(parameter.max, 2)}`;
  elements.activeParameter.textContent =
    `${state.currentGroup} / ${parameter.name} = ${formatNumber(val)}`;
  elements.jointsField.hidden = state.currentGroup !== "pose";
  elements.jointList.textContent = parameter.joints.length
    ? parameter.joints.join(", ")
    : "No directly affected joints";
  elements.slider.disabled = state.requestPending || locked;
  elements.value.disabled = state.requestPending || locked;
  elements.resetCurrent.disabled = state.requestPending || locked;
  setExprVectorSelected(state.currentGroup === "expression" ? parameter.index : null);
}

export function selectGroup(group) {
  state.currentGroup = group;
  // Tab chrome is owned by switchActorTab (app.js); this stays data-only.
  elements.categoryField.hidden = group !== "pose";
  elements.search.value = "";
  rebuildParameterList();
}
