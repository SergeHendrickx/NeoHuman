/**
 * Agent action catalog: the LLM's hands on the viewer.
 *
 * Pure module (no imports): tool schemas in OpenAI function format plus a
 * dependency-injected dispatcher. app.js binds real implementations in slice C;
 * node tests drive it with stubs.
 *
 * dispatch() never throws: every outcome is { ok: true, result } or
 * { ok: false, error }. Results are JSON-fed back to the model.
 */

const GROUPS = ["pose", "identity", "expression"];
const VIEWS = ["front", "back", "left", "right", "face", "focus"];
const BEHAVIORS = ["wander", "flock", "circle", "scatter", "idle"];

export const DESTRUCTIVE_ACTIONS = new Set(["clearAgents", "removeActor"]);

export const TOOL_SCHEMAS = [
  {
    type: "function",
    function: {
      name: "selectActor",
      description:
        "Select an actor as the active one (camera, skeleton, gizmo and actor panel retarget to it). Actor #0 is the initial actor; spawned agents have ids >= 1.",
      parameters: {
        type: "object",
        properties: {
          actorId: { type: "integer", minimum: 0, description: "0 = initial actor, >= 1 = agent id." },
        },
        required: ["actorId"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "setParam",
      description:
        "Set one rig parameter on the active actor: pose (joint rotations), identity (body shape) or expression (blendshapes, 72 params). Use index from the parameter list.",
      parameters: {
        type: "object",
        properties: {
          group: { type: "string", enum: GROUPS },
          index: { type: "integer", minimum: 0, description: "Parameter index within the group." },
          value: { type: "number", description: "New value (clamped to the parameter's own range)." },
        },
        required: ["group", "index", "value"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "spawnAgents",
      description: "Spawn crowd agents into the simulation (GPU-backed). Spawning is batched; large counts take seconds.",
      parameters: {
        type: "object",
        properties: {
          count: { type: "integer", minimum: 1, maximum: 500 },
        },
        required: ["count"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "clearAgents",
      description: "Remove ALL spawned agents (actor #0 stays). Destructive: needs user approval.",
      parameters: { type: "object", properties: {} },
    },
  },
  {
    type: "function",
    function: {
      name: "removeActor",
      description: "Remove one spawned agent by id. Actor #0 cannot be removed. Destructive: needs user approval.",
      parameters: {
        type: "object",
        properties: {
          actorId: { type: "integer", minimum: 1 },
        },
        required: ["actorId"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "camera",
      description:
        "Move the viewport camera: front/back/left/right/face views of the active actor, or focus (orbit target onto the selection).",
      parameters: {
        type: "object",
        properties: {
          view: { type: "string", enum: VIEWS },
        },
        required: ["view"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "gaze",
      description: "Set eye gaze (degrees) on the active actor. Yaw -25..25 (-R/+L), pitch -20..20 (-down/+up).",
      parameters: {
        type: "object",
        properties: {
          yaw: { type: "number", minimum: -25, maximum: 25 },
          pitch: { type: "number", minimum: -20, maximum: 20 },
        },
        required: ["yaw", "pitch"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "setBehavior",
      description:
        "Set crowd steering behavior. Without actorId it applies to all agents and future spawns; with actorId to one agent.",
      parameters: {
        type: "object",
        properties: {
          behavior: { type: "string", enum: BEHAVIORS },
          actorId: { type: "integer", minimum: 1, description: "Optional: single agent id." },
        },
        required: ["behavior"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "lodColors",
      description: "Tint every agent by its LOD level for debugging LOD distribution.",
      parameters: {
        type: "object",
        properties: {
          enabled: { type: "boolean" },
        },
        required: ["enabled"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "benchmark",
      description:
        "Run a timed performance benchmark (spawns agents, measures, restores the scene). Slow: takes seconds plus spawn time. Returns fps/frame-time summary.",
      parameters: {
        type: "object",
        properties: {
          agents: { type: "integer", minimum: 1, maximum: 500, default: 60 },
          seconds: { type: "number", minimum: 3, maximum: 30, default: 5 },
        },
      },
    },
  },
];

function fail(error) {
  return { ok: false, error };
}

function asInt(raw, label) {
  const n = Number(raw);
  if (!Number.isInteger(n)) return { error: `${label} must be an integer, got ${JSON.stringify(raw)}` };
  return { value: n };
}

function asNumber(raw, label) {
  const n = Number(raw);
  if (!Number.isFinite(n)) return { error: `${label} must be a finite number, got ${JSON.stringify(raw)}` };
  return { value: n };
}

function inEnum(raw, allowed, label) {
  if (typeof raw !== "string" || !allowed.includes(raw)) {
    return { error: `${label} must be one of ${allowed.join("/")}, got ${JSON.stringify(raw)}` };
  }
  return { value: raw };
}

function inRange(n, lo, hi, label) {
  if (n < lo || n > hi) return { error: `${label} must be ${lo}..${hi}, got ${n}` };
  return { value: n };
}

/** Per-action validators: raw LLM args -> { args } or { error }. */
const VALIDATORS = {
  selectActor(raw) {
    const id = asInt(raw.actorId, "actorId");
    if (id.error) return id;
    const range = inRange(id.value, 0, 1e6, "actorId");
    if (range.error) return range;
    return { args: { actorId: range.value } };
  },
  setParam(raw) {
    const group = inEnum(raw.group, GROUPS, "group");
    if (group.error) return group;
    const index = asInt(raw.index, "index");
    if (index.error) return index;
    if (index.value < 0) return { error: `index must be >= 0, got ${index.value}` };
    const value = asNumber(raw.value, "value");
    if (value.error) return value;
    return { args: { group: group.value, index: index.value, value: value.value } };
  },
  spawnAgents(raw) {
    const count = asInt(raw.count, "count");
    if (count.error) return count;
    const range = inRange(count.value, 1, 500, "count");
    if (range.error) return range;
    return { args: { count: range.value } };
  },
  clearAgents() {
    return { args: {} };
  },
  removeActor(raw) {
    const id = asInt(raw.actorId, "actorId");
    if (id.error) return id;
    const range = inRange(id.value, 1, 1e6, "actorId");
    if (range.error) return range;
    return { args: { actorId: range.value } };
  },
  camera(raw) {
    const view = inEnum(raw.view, VIEWS, "view");
    if (view.error) return view;
    return { args: { view: view.value } };
  },
  gaze(raw) {
    const yaw = asNumber(raw.yaw, "yaw");
    if (yaw.error) return yaw;
    const pitch = asNumber(raw.pitch, "pitch");
    if (pitch.error) return pitch;
    // Sliders step in whole degrees; rounding is friendlier than rejecting 12.5.
    const y = Math.round(yaw.value);
    const p = Math.round(pitch.value);
    const yr = inRange(y, -25, 25, "yaw");
    if (yr.error) return yr;
    const pr = inRange(p, -20, 20, "pitch");
    if (pr.error) return pr;
    return { args: { yaw: yr.value, pitch: pr.value } };
  },
  setBehavior(raw) {
    const behavior = inEnum(raw.behavior, BEHAVIORS, "behavior");
    if (behavior.error) return behavior;
    if (raw.actorId === undefined || raw.actorId === null) {
      return { args: { behavior: behavior.value, actorId: null } };
    }
    const id = asInt(raw.actorId, "actorId");
    if (id.error) return id;
    const range = inRange(id.value, 1, 1e6, "actorId");
    if (range.error) return range;
    return { args: { behavior: behavior.value, actorId: range.value } };
  },
  lodColors(raw) {
    if (typeof raw.enabled !== "boolean") {
      return { error: `enabled must be a boolean, got ${JSON.stringify(raw.enabled)}` };
    }
    return { args: { enabled: raw.enabled } };
  },
  benchmark(raw) {
    const agents = raw.agents === undefined ? { value: 60 } : asInt(raw.agents, "agents");
    if (agents.error) return agents;
    const ar = inRange(agents.value, 1, 500, "agents");
    if (ar.error) return ar;
    const seconds = raw.seconds === undefined ? { value: 5 } : asNumber(raw.seconds, "seconds");
    if (seconds.error) return seconds;
    const sr = inRange(seconds.value, 3, 30, "seconds");
    if (sr.error) return sr;
    return { args: { agents: ar.value, seconds: sr.value } };
  },
};

/**
 * Append-only audit trail of every dispatch, whatever the source (chat,
 * console, future remotes). Pure; the panel subscribes to render it.
 * Entry: { seq, time, name, args, ok, result, error, ms, source }.
 */
export function createActionLog({ limit = 200 } = {}) {
  const entries = [];
  const listeners = new Set();
  let seq = 0;
  return {
    entries,
    record(entry) {
      const full = { seq: ++seq, time: new Date().toISOString(), ms: 0, source: "api", ...entry };
      entries.push(full);
      while (entries.length > limit) entries.shift();
      for (const fn of listeners) fn(full);
      return full;
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    clear() {
      entries.length = 0;
    },
  };
}

/**
 * deps: { selectActor({actorId}), setParam({group,index,value}),
 *   spawnAgents({count}), clearAgents(), removeActor({actorId}),
 *   camera({view}), gaze({yaw,pitch}), setBehavior({behavior,actorId|null}),
 *   lodColors({enabled}), benchmark({agents,seconds}) }.
 * Each may return a value or a promise; the result is fed back to the model.
 * opts.log (createActionLog) records every outcome incl. rejections.
 * dispatch(name, args, { source }) tags the entry ("chat", "console", ...).
 */
export function createDispatcher(deps, opts = {}) {
  const names = TOOL_SCHEMAS.map((t) => t.function.name);
  const log = opts.log ?? null;
  return {
    actionNames() {
      return [...names];
    },
    isDestructive(name) {
      return DESTRUCTIVE_ACTIONS.has(name);
    },
    async dispatch(name, rawArgs, meta = {}) {
      const source = meta.source ?? "api";
      const validate = VALIDATORS[name];
      if (!validate) {
        const out = fail(`unknown action: ${JSON.stringify(name)}. Available: ${names.join(", ")}`);
        log?.record({ name, args: rawArgs ?? {}, ok: false, error: out.error, source });
        return out;
      }
      const checked = validate(rawArgs ?? {});
      if (checked.error) {
        const out = fail(checked.error);
        log?.record({ name, args: rawArgs ?? {}, ok: false, error: out.error, source });
        return out;
      }
      const impl = deps[name];
      if (typeof impl !== "function") {
        const out = fail(`action not bound in this session: ${name}`);
        log?.record({ name, args: checked.args, ok: false, error: out.error, source });
        return out;
      }
      const t0 = Date.now();
      try {
        const result = await impl(checked.args);
        const out = { ok: true, result: result === undefined ? null : result };
        log?.record({ name, args: checked.args, ok: true, result: out.result, ms: Date.now() - t0, source });
        return out;
      } catch (err) {
        const out = fail(err?.message ? String(err.message) : String(err));
        log?.record({ name, args: checked.args, ok: false, error: out.error, ms: Date.now() - t0, source });
        return out;
      }
    },
  };
}
