/**
 * Command menu (Ctrl/Mod+K): the action catalog as a human palette.
 *
 * Entries are built from TOOL_SCHEMAS plus a few UI commands, so the menu
 * can never drift from what the agent (and the console) can do. Fixed-arg
 * entries dispatch immediately; schema-driven forms collect the rest;
 * destructive actions get an explicit confirm step. Every run goes through
 * the shared dispatcher (source "palette"), so validation and the audit
 * trail come for free.
 *
 * Pure parts (buildEntries, filterEntries, formFieldsFor) are node-tested;
 * initCommandMenu owns the DOM + keyboard.
 */

import { TOOL_SCHEMAS } from "./actions.js";

function schemaOf(name) {
  return TOOL_SCHEMAS.find((t) => t.function.name === name)?.function ?? null;
}

/**
 * Flatten the catalog into runnable entries.
 * actors: [{ id, name }]. simRunning/headerHidden tune UI-command titles.
 */
export function buildEntries({ actors = [], simRunning = true, headerHidden = false } = {}) {
  const entries = [];
  const push = (entry) => entries.push({ keywords: "", detail: "", ...entry });

  for (const actor of actors) {
    push({
      key: `select-${actor.id}`,
      title: `Select ${actor.name}`,
      detail: "Camera, skeleton and panel follow",
      keywords: "select actor focus",
      name: "selectActor",
      args: { actorId: actor.id },
    });
  }

  push({
    key: "set-param",
    title: "Set parameter…",
    detail: "Pose / identity / expression value",
    keywords: "pose identity expression slider joint blendshape",
    name: "setParam",
    args: "form",
  });

  for (const count of [1, 6, 20, 100]) {
    push({
      key: `spawn-${count}`,
      title: `Spawn ${count} agent${count === 1 ? "" : "s"}`,
      detail: "GPU-backed crowd agents",
      keywords: "spawn add crowd agents",
      name: "spawnAgents",
      args: { count },
    });
  }
  push({
    key: "spawn-custom",
    title: "Spawn agents…",
    detail: "Custom count, 1–500",
    keywords: "spawn add crowd agents custom",
    name: "spawnAgents",
    args: "form",
  });

  push({
    key: "clear-agents",
    title: "Remove all agents",
    detail: "Actor #0 stays",
    keywords: "clear remove all agents delete",
    name: "clearAgents",
    args: {},
    destructive: true,
  });

  for (const actor of actors.filter((a) => a.id >= 1)) {
    push({
      key: `remove-${actor.id}`,
      title: `Remove ${actor.name}`,
      detail: "Delete one spawned agent",
      keywords: "remove delete agent",
      name: "removeActor",
      args: { actorId: actor.id },
      destructive: true,
    });
  }

  const views = { front: "Front", back: "Back", left: "Left", right: "Right", face: "Face", focus: "Focus" };
  for (const [view, label] of Object.entries(views)) {
    push({
      key: `camera-${view}`,
      title: view === "focus" ? "Focus selection" : `${label} view`,
      detail: "Move the viewport camera",
      keywords: "camera view orbit",
      name: "camera",
      args: { view },
    });
  }

  for (const [label, yaw, pitch] of [
    ["Look center", 0, 0],
    ["Look left", 25, 0],
    ["Look right", -25, 0],
    ["Look up", 0, 20],
    ["Look down", 0, -20],
  ]) {
    push({
      key: `gaze-${label.split(" ")[1].toLowerCase()}`,
      title: label,
      detail: "Eye gaze on the active actor",
      keywords: "gaze eyes look",
      name: "gaze",
      args: { yaw, pitch },
    });
  }
  push({
    key: "gaze-custom",
    title: "Look somewhere…",
    detail: "Custom yaw/pitch in degrees",
    keywords: "gaze eyes look custom",
    name: "gaze",
    args: "form",
  });

  for (const behavior of ["wander", "flock", "circle", "scatter", "idle"]) {
    push({
      key: `behavior-${behavior}`,
      title: `Behavior: ${behavior}`,
      detail: "All agents + future spawns",
      keywords: "behavior steering wander flock",
      name: "setBehavior",
      args: { behavior, actorId: null },
    });
  }

  for (const enabled of [true, false]) {
    push({
      key: `lod-${enabled ? "on" : "off"}`,
      title: `LOD colors ${enabled ? "on" : "off"}`,
      detail: "Tint agents by LOD level",
      keywords: "lod colors tint debug",
      name: "lodColors",
      args: { enabled },
    });
  }

  push({
    key: "bench-quick",
    title: "Run benchmark",
    detail: "60 agents, 5 seconds",
    keywords: "benchmark perf measure fps",
    name: "benchmark",
    args: { agents: 60, seconds: 5 },
  });
  push({
    key: "bench-custom",
    title: "Run benchmark…",
    detail: "Custom agents/seconds",
    keywords: "benchmark perf measure fps custom",
    name: "benchmark",
    args: "form",
  });

  push({
    key: "ui-sim",
    title: simRunning ? "Pause simulation" : "Resume simulation",
    detail: "Freeze or resume time",
    keywords: "play pause sim time",
    ui: "toggleSim",
  });
  push({
    key: "ui-header",
    title: headerHidden ? "Show header" : "Hide header",
    detail: "Collapse the top bar",
    keywords: "header topbar hide show",
    ui: "toggleHeader",
  });
  push({
    key: "ui-layout",
    title: "Reset panel layout",
    detail: "Restore default dock arrangement",
    keywords: "layout reset dock panels",
    ui: "resetLayout",
  });

  return entries;
}

/** Substring-first fuzzy rank; subsequence fallback. Returns top `limit`. */
export function filterEntries(entries, query, limit = 12) {
  const q = query.trim().toLowerCase();
  if (!q) return entries.slice(0, limit);
  const scored = [];
  for (const entry of entries) {
    const hay = `${entry.title} ${entry.detail} ${entry.keywords}`.toLowerCase();
    let score = -1;
    const at = hay.indexOf(q);
    if (at >= 0) {
      score = 100 - at + (entry.title.toLowerCase().startsWith(q) ? 50 : 0);
    } else if (isSubsequence(q, hay)) {
      score = 10;
    }
    if (score >= 0) scored.push([score, entry]);
  }
  scored.sort((a, b) => b[0] - a[0]);
  return scored.slice(0, limit).map(([, entry]) => entry);
}

function isSubsequence(q, hay) {
  let i = 0;
  for (const ch of hay) {
    if (ch === q[i]) i++;
    if (i === q.length) return true;
  }
  return false;
}

/** Schema parameters -> form field models for the arg step. */
export function formFieldsFor(actionName) {
  const schema = schemaOf(actionName);
  const props = schema?.parameters?.properties ?? {};
  const required = new Set(schema?.parameters?.required ?? []);
  return Object.entries(props).map(([key, def]) => ({
    key,
    label: def.description ? `${key} — ${def.description}` : key,
    type: def.enum ? "enum" : def.type === "boolean" ? "boolean" : def.type === "integer" ? "integer" : "number",
    enum: def.enum ?? null,
    min: def.minimum ?? null,
    max: def.maximum ?? null,
    def: def.default ?? (def.enum ? def.enum[0] : null),
    required: required.has(key),
  }));
}

/**
 * Wire the palette DOM. deps:
 * dispatch(name, args) -> { ok, result?, error? } (already source-tagged),
 * listActors() -> [{ id, name }], isSimRunning(), toggleSim(),
 * isHeaderHidden(), setHeaderHidden(bool), resetLayout(), setStatus(msg, kind?).
 */
export function initCommandMenu(deps) {
  const overlay = document.getElementById("cmdk-overlay");
  const input = document.getElementById("cmdk-input");
  const list = document.getElementById("cmdk-list");
  const hint = document.getElementById("cmdk-hint");
  const opener = document.getElementById("cmdk-open");
  if (!overlay || !input || !list) return null;

  let entries = [];
  let matches = [];
  let cursor = 0;
  let formEntry = null;
  let confirmEntry = null;

  function setHint(text) {
    if (hint) hint.textContent = text;
  }

  function renderList() {
    list.replaceChildren();
    matches.forEach((entry, i) => {
      const row = document.createElement("button");
      row.type = "button";
      row.className = `cmdk-row${i === cursor ? " is-active" : ""}${entry.destructive ? " is-danger" : ""}`;
      row.setAttribute("role", "option");
      row.setAttribute("aria-selected", String(i === cursor));
      const title = document.createElement("b");
      title.textContent = entry.title;
      row.append(title);
      if (entry.detail) {
        const detail = document.createElement("span");
        detail.className = "quiet";
        detail.textContent = entry.detail;
        row.append(detail);
      }
      row.addEventListener("click", () => choose(i));
      list.append(row);
    });
    list.querySelector(".is-active")?.scrollIntoView({ block: "nearest" });
  }

  function renderForm() {
    list.replaceChildren();
    const fields = formFieldsFor(formEntry.name);
    const title = document.createElement("div");
    title.className = "cmdk-form-title";
    title.textContent = formEntry.title.replace(/…$/, "");
    list.append(title);
    for (const field of fields) {
      const label = document.createElement("label");
      label.className = "cmdk-field";
      const name = document.createElement("span");
      name.textContent = field.label;
      label.append(name);
      let control;
      if (field.type === "enum") {
        control = document.createElement("select");
        for (const opt of field.enum) {
          const el = document.createElement("option");
          el.value = opt;
          el.textContent = opt;
          control.append(el);
        }
        control.value = field.def ?? field.enum[0];
      } else if (field.type === "boolean") {
        control = document.createElement("select");
        for (const opt of ["true", "false"]) {
          const el = document.createElement("option");
          el.value = opt;
          el.textContent = opt;
          control.append(el);
        }
      } else {
        control = document.createElement("input");
        control.type = "number";
        if (field.min !== null) control.min = String(field.min);
        if (field.max !== null) control.max = String(field.max);
        if (field.def !== null) control.value = String(field.def);
        if (field.type === "integer") control.step = "1";
      }
      control.dataset.field = field.key;
      control.dataset.ftype = field.type;
      label.append(control);
      list.append(label);
    }
    setHint("Enter runs · Esc back");
    list.querySelector("input, select")?.focus();
  }

  function renderConfirm() {
    list.replaceChildren();
    const warn = document.createElement("div");
    warn.className = "cmdk-form-title is-danger";
    warn.textContent = `${confirmEntry.title} — are you sure?`;
    list.append(warn);
    setHint("Enter confirms · Esc back");
  }

  function collectFormArgs() {
    const args = {};
    for (const control of list.querySelectorAll("[data-field]")) {
      const key = control.dataset.field;
      const type = control.dataset.ftype;
      if (type === "boolean") args[key] = control.value === "true";
      else if (type === "integer") args[key] = Number.parseInt(control.value, 10);
      else args[key] = control.value === "" ? undefined : Number(control.value);
      if (args[key] === undefined) delete args[key];
    }
    return { ...defaultArgs(formEntry.name), ...args };
  }

  function defaultArgs() {
    return {};
  }

  async function run(entry, args) {
    if (entry.ui === "toggleSim") {
      deps.toggleSim();
      close();
      return;
    }
    if (entry.ui === "toggleHeader") {
      deps.setHeaderHidden(!deps.isHeaderHidden());
      close();
      return;
    }
    if (entry.ui === "resetLayout") {
      deps.resetLayout();
      close();
      return;
    }
    const out = await deps.dispatch(entry.name, args);
    if (out.ok) {
      deps.setStatus?.(`${entry.title} — done.`, "ok");
      close();
    } else {
      setHint(out.error || "Action failed.");
    }
  }

  function choose(i) {
    const entry = matches[i ?? cursor];
    if (!entry) return;
    if (entry.args === "form") {
      formEntry = entry;
      confirmEntry = null;
      renderForm();
      return;
    }
    if (entry.destructive) {
      confirmEntry = entry;
      formEntry = null;
      renderConfirm();
      return;
    }
    void run(entry, entry.args ?? {});
  }

  function refresh() {
    entries = buildEntries({
      actors: deps.listActors(),
      simRunning: deps.isSimRunning(),
      headerHidden: deps.isHeaderHidden(),
    });
    matches = filterEntries(entries, input.value);
    cursor = 0;
    renderList();
    setHint("↑↓ navigate · Enter run · Esc close");
  }

  function open() {
    formEntry = null;
    confirmEntry = null;
    overlay.hidden = false;
    input.value = "";
    refresh();
    input.focus();
  }

  function close() {
    overlay.hidden = true;
    input.blur();
  }

  function back() {
    // Esc steps out of form/confirm before closing the palette.
    if (formEntry || confirmEntry) {
      formEntry = null;
      confirmEntry = null;
      refresh();
      input.focus();
      return;
    }
    close();
  }

  input.addEventListener("input", () => {
    formEntry = null;
    confirmEntry = null;
    matches = filterEntries(entries, input.value);
    cursor = 0;
    renderList();
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      cursor = Math.min(cursor + 1, matches.length - 1);
      renderList();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      cursor = Math.max(cursor - 1, 0);
      renderList();
    } else if (event.key === "Enter") {
      event.preventDefault();
      choose(cursor);
    } else if (event.key === "Escape") {
      event.preventDefault();
      back();
    }
  });

  // Enter/Escape work from the form controls too.
  list.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (formEntry || confirmEntry)) {
      event.preventDefault();
      if (formEntry) {
        const entry = formEntry;
        if (entry.destructive) {
          confirmEntry = entry;
          formEntry = null;
          confirmEntry._args = collectFormArgs();
          renderConfirm();
        } else {
          formEntry = null;
          void run(entry, collectFormArgs());
        }
      } else if (confirmEntry) {
        const entry = confirmEntry;
        confirmEntry = null;
        void run(entry, entry._args ?? entry.args ?? {});
      }
    } else if (event.key === "Escape") {
      event.preventDefault();
      back();
    }
  });

  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) close();
  });

  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      if (overlay.hidden) open();
      else close();
    }
  });

  opener?.addEventListener("click", () => {
    if (overlay.hidden) open();
    else close();
  });

  return { open, close };
}
