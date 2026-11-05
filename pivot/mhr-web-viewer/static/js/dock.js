/**
 * Dockview shell: every viewer section is a draggable dock tab. Section DOM
 * is MOVED from the static layout into dock panels on boot; the static
 * strips remain as the no-dockview fallback. Layout persists to
 * localStorage; Reset restores the default.
 *
 * The shared parameter component reparents across the pose/identity/
 * expression mounts: app.js passes onPanelActivated and syncs there, since
 * dock tabs replace the strip clicks.
 */

import { DockviewComponent, themeDark } from "dockview";

const LS_LAYOUT = "mhr.dockLayout.v3";

// Live API for resetDockLayout (the View menu + palette call it directly).
let dockApiRef = null;

// Highly technical panes, hidden from the default (avatar-first) view
// until Developer tools are enabled. The flag also keeps Reset honest:
// rebuilding restores the caller's current mode, not always everything.
export const DEV_PANES = ["performance", "crowd", "actors", "telemetry", "crowdstats", "ai"];
let devVisible = true;

const DEV_HOME = {
  performance: ["world", "within"],
  crowd: ["world", "within"],
  actors: ["world", "within"],
  ai: ["identity", "within"],
  telemetry: ["viewport", "below"],
  crowdstats: ["telemetry", "within"],
};

/** Show or hide the technical panes (no-op without a live dock). */
export function setDevPanelsVisible(on) {
  devVisible = on !== false;
  const api = dockApiRef;
  if (!api) return false;
  if (devVisible) {
    for (const id of DEV_PANES) {
      try {
        if (api.getPanel(id)) continue;
        const title = (PANES.find((p) => p.id === id) ?? {}).title ?? id;
        const [ref, direction] = DEV_HOME[id];
        const options = { id, title, component: id };
        if (api.getPanel(ref)) {
          options.position = { direction, referencePanel: ref };
        }
        api.addPanel(options);
      } catch (err) {
        console.warn(`[dock] cannot restore pane ${id}:`, err?.message ?? err);
      }
    }
  } else {
    for (const id of DEV_PANES) {
      try {
        api.getPanel(id)?.api.close();
      } catch (err) {
        console.warn(`[dock] cannot hide pane ${id}:`, err?.message ?? err);
      }
    }
    if (api.activePanel && DEV_PANES.includes(api.activePanel.id)) {
      try {
        api.getPanel("world")?.api.setActive();
      } catch {
        // Neighbor activation already picked a tab; nothing to do.
      }
    }
  }
  return true;
}

/** Clear the saved layout and rebuild the default tab arrangement. */
export function resetDockLayout() {
  if (!dockApiRef) return false;
  try { localStorage.removeItem(LS_LAYOUT); } catch {
    // Private mode etc: layout was never persisted.
  }
  dockApiRef.clear();
  buildDefault(dockApiRef, { dev: devVisible });
  return true;
}

export const PANES = [
  { id: "viewport", title: "Viewport", selector: "#viewport" },
  { id: "identity", title: "Identity", selector: "#atab-identity" },
  { id: "pose", title: "Pose", selector: "#atab-pose" },
  { id: "expression", title: "Expression", selector: "#atab-expression" },
  { id: "eyes", title: "Eyes", selector: "#atab-eyes" },
  { id: "mouth", title: "Mouth", selector: "#atab-mouth" },
  { id: "materials", title: "Materials", selector: "#atab-materials" },
  { id: "clothing", title: "Clothing", selector: "#atab-clothing" },
  { id: "animation", title: "Animation", selector: "#atab-animation" },
  { id: "ai", title: "AI", selector: "#atab-ai" },
  { id: "world", title: "World", selector: "#wtab-world" },
  { id: "camera", title: "Camera", selector: "#wtab-camera" },
  { id: "performance", title: "Performance", selector: "#wtab-performance" },
  { id: "crowd", title: "Crowd", selector: "#wtab-crowd" },
  { id: "actors", title: "Actors", selector: "#wtab-actors" },
  { id: "telemetry", title: "Telemetry", selector: "#perf-card" },
  { id: "crowdstats", title: "Crowd Stats", selector: "#crowd-stats-details" },
];

const KNOWN = new Set(PANES.map((p) => p.id));

export function initDock({ onPanelActivated, dev = true } = {}) {
  devVisible = dev !== false;
  const root = document.getElementById("dock-root");
  if (!root) return null;
  // Cache nodes: clear()/close() detaches content, so re-querying the
  // document would lose panes on rebuild.
  const nodes = new Map();
  for (const pane of PANES) {
    const el = document.querySelector(pane.selector);
    if (!el) {
      console.warn(`[dock] pane source missing: ${pane.selector}`);
      return null;
    }
    nodes.set(pane.id, el);
  }
  // Shown before construction so the container measures real size; rolled
  // back if the build throws, leaving the static staging layout usable.
  document.body.classList.add("is-docked");
  let dockview;
  try {
    dockview = new DockviewComponent(root, {
      theme: themeDark,
      floatingGroupBounds: "boundedWithinViewport",
      // Single-panel groups: the whole header is the tab (and the drag handle).
      singleTabMode: "fullwidth",
      createComponent: ({ name }) => {
        const element = nodes.get(name);
        // Strips start hidden (only the strip-active panel shows); dock tabs
        // manage visibility themselves.
        element.removeAttribute("hidden");
        return { element, init() {} };
      },
    });
    const api = dockview.api;
    if (!restoreLayout(api)) buildDefault(api, { dev: devVisible });
  } catch (err) {
    document.body.classList.remove("is-docked");
    console.warn("[dock] dockview build failed, static layout:", err?.message ?? err);
    return null;
  }
  const api = dockview.api;
  dockApiRef = api;
  window.mhr = { ...(window.mhr ?? {}), dockApi: api, resetDockLayout };
  api.onDidActivePanelChange(() => onPanelActivated?.(api.activePanel?.id));
  onPanelActivated?.(api.activePanel?.id);
  // Orphans that belong to no pane: the status bar becomes the app footer
  // and the selected-actor header joins the Actors tab. Moved (listeners
  // intact) before the staging shells are dropped.
  const statusbar = document.querySelector(".statusbar");
  if (statusbar) document.body.appendChild(statusbar);
  const actorSelected = document.getElementById("actor-selected");
  if (actorSelected) nodes.get("actors").prepend(actorSelected);
  // Staging shells served first paint; dock owns layout from here.
  document.querySelector("main.workspace")?.remove();
  // Restored layouts may carry technical panes into a clean session.
  if (!devVisible) setDevPanelsVisible(false);

  let saveTimer = 0;
  api.onDidLayoutChange(() => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      try {
        localStorage.setItem(LS_LAYOUT, JSON.stringify(api.toJSON()));
      } catch {
        // Private mode etc: stay ephemeral.
      }
    }, 400);
  });
  return dockview;
}

function addTab(api, id, title, ref, direction) {
  api.addPanel({
    id, title, component: id,
    ...(ref ? { position: { direction, referencePanel: ref } } : {}),
  });
}

function buildDefault(api, { dev = true } = {}) {
  const want = (id) => dev || !DEV_PANES.includes(id);
  addTab(api, "viewport", "Viewport");
  // One right-hand tab group for everything; Identity opens first.
  addTab(api, "identity", "Identity", "viewport", "right");
  for (const [id, title] of [
    ["pose", "Pose"], ["expression", "Expression"], ["eyes", "Eyes"],
    ["mouth", "Mouth"], ["materials", "Materials"], ["clothing", "Clothing"],
    ["animation", "Animation"], ["ai", "AI"], ["world", "World"], ["camera", "Camera"],
  ]) {
    addTab(api, id, title, "identity", "within");
  }
  for (const [id, title] of [["performance", "Performance"], ["crowd", "Crowd"], ["actors", "Actors"]]) {
    if (want(id)) addTab(api, id, title, "identity", "within");
  }
  if (want("telemetry")) addTab(api, "telemetry", "Telemetry", "viewport", "below");
  if (want("crowdstats")) addTab(api, "crowdstats", "Crowd Stats", "telemetry", "within");
  api.getPanel("identity")?.api.setActive();
}

function restoreLayout(api) {
  try {
    const raw = localStorage.getItem(LS_LAYOUT);
    if (!raw) return false;
    const data = JSON.parse(raw);
    const ids = data?.panels ? Object.keys(data.panels) : [];
    // Strict: every saved id must be known (rejects phase-1 layouts and
    // future panes), and at least one tab must remain (else blank dock).
    if (ids.length === 0 || !ids.every((id) => KNOWN.has(id))) return false;
    api.fromJSON(data);
    return true;
  } catch {
    return false;
  }
}
