import test from "node:test";
import assert from "node:assert/strict";
import { TOOL_SCHEMAS, DESTRUCTIVE_ACTIONS, createActionLog, createDispatcher } from "../../static/js/actions.js";

function stubDispatcher(overrides = {}) {
  const calls = [];
  const record = (name) => async (a) => { calls.push([name, a]); };
  const deps = {
    selectActor: record("selectActor"),
    setParam: record("setParam"),
    spawnAgents: record("spawnAgents"),
    clearAgents: record("clearAgents"),
    removeActor: record("removeActor"),
    camera: record("camera"),
    gaze: record("gaze"),
    setBehavior: record("setBehavior"),
    lodColors: record("lodColors"),
    benchmark: record("benchmark"),
    ...overrides,
  };
  return { d: createDispatcher(deps), calls };
}

test("schemas: unique names, object params, destructive set matches", () => {
  const names = TOOL_SCHEMAS.map((t) => t.function.name);
  assert.equal(new Set(names).size, names.length);
  assert.equal(names.length, 10);
  for (const t of TOOL_SCHEMAS) {
    assert.equal(t.type, "function");
    assert.ok(t.function.description.length > 10, t.function.name);
    assert.equal(t.function.parameters.type, "object");
  }
  assert.deepEqual([...DESTRUCTIVE_ACTIONS].sort(), ["clearAgents", "removeActor"]);
  const { d } = stubDispatcher();
  assert.deepEqual(d.actionNames(), names);
  assert.equal(d.isDestructive("clearAgents"), true);
  assert.equal(d.isDestructive("removeActor"), true);
  assert.equal(d.isDestructive("camera"), false);
});

test("dispatch routes validated args to deps", async () => {
  const { d, calls } = stubDispatcher();
  assert.deepEqual(await d.dispatch("selectActor", { actorId: 3 }), { ok: true, result: null });
  assert.deepEqual(await d.dispatch("setParam", { group: "expression", index: 12, value: 0.6 }), { ok: true, result: null });
  assert.deepEqual(await d.dispatch("spawnAgents", { count: 50 }), { ok: true, result: null });
  assert.deepEqual(await d.dispatch("clearAgents", {}), { ok: true, result: null });
  assert.deepEqual(await d.dispatch("camera", { view: "face" }), { ok: true, result: null });
  assert.deepEqual(await d.dispatch("gaze", { yaw: 10, pitch: -5 }), { ok: true, result: null });
  assert.deepEqual(await d.dispatch("setBehavior", { behavior: "circle" }), { ok: true, result: null });
  assert.deepEqual(await d.dispatch("lodColors", { enabled: true }), { ok: true, result: null });
  assert.deepEqual(calls[0], ["selectActor", { actorId: 3 }]);
  assert.deepEqual(calls[6], ["setBehavior", { behavior: "circle", actorId: null }]);
});

test("dispatch returns dep results and awaits async deps", async () => {
  const { d } = stubDispatcher({ benchmark: async (a) => ({ fpsAvg: 120, ...a }) });
  const res = await d.dispatch("benchmark", {});
  assert.equal(res.ok, true);
  assert.equal(res.result.fpsAvg, 120);
  assert.equal(res.result.agents, 60);
  assert.equal(res.result.seconds, 5);
});

test("validation rejects bad args without calling deps", async () => {
  const { d, calls } = stubDispatcher();
  const bad = [
    ["selectActor", { actorId: -1 }],
    ["selectActor", {}],
    ["setParam", { group: "nope", index: 0, value: 1 }],
    ["setParam", { group: "pose", index: -2, value: 1 }],
    ["setParam", { group: "pose", index: 1.5, value: 1 }],
    ["setParam", { group: "pose", index: 0, value: "far" }],
    ["spawnAgents", { count: 0 }],
    ["spawnAgents", { count: 501 }],
    ["removeActor", { actorId: 0 }],
    ["camera", { view: "top" }],
    ["gaze", { yaw: 30, pitch: 0 }],
    ["gaze", { yaw: 0, pitch: -25 }],
    ["setBehavior", { behavior: "dance" }],
    ["lodColors", { enabled: "yes" }],
    ["benchmark", { agents: 10, seconds: 60 }],
    ["nope", {}],
  ];
  for (const [name, args] of bad) {
    const res = await d.dispatch(name, args);
    assert.equal(res.ok, false, `${name} ${JSON.stringify(args)}`);
    assert.ok(res.error.length > 0);
  }
  assert.equal(calls.length, 0);
});

test("gaze rounds fractional degrees; numeric strings coerce", async () => {
  const { d, calls } = stubDispatcher();
  assert.deepEqual(await d.dispatch("gaze", { yaw: 12.5, pitch: -3.2 }), { ok: true, result: null });
  assert.deepEqual(calls[0], ["gaze", { yaw: 13, pitch: -3 }]);
  assert.deepEqual(await d.dispatch("spawnAgents", { count: "25" }), { ok: true, result: null });
  assert.deepEqual(calls[1], ["spawnAgents", { count: 25 }]);
});

test("action log records every outcome with source and timing", async () => {
  const log = createActionLog({ limit: 3 });
  const seen = [];
  const unsub = log.subscribe((e) => seen.push(e));
  const dd = createDispatcher(
    { camera: async () => ({ moved: true }) },
    { log }
  );
  await dd.dispatch("camera", { view: "front" }, { source: "chat" });
  await dd.dispatch("camera", { view: "up" }, { source: "chat" });
  await dd.dispatch("nope", {}, { source: "console" });
  assert.equal(log.entries.length, 3);
  assert.deepEqual(
    log.entries.map((e) => [e.seq, e.name, e.ok, e.source]),
    [[1, "camera", true, "chat"], [2, "camera", false, "chat"], [3, "nope", false, "console"]]
  );
  assert.ok(log.entries[0].ms >= 0);
  assert.match(log.entries[0].time, /^\d{4}-\d{2}-\d{2}T/);
  assert.equal(seen.length, 3);
  unsub();
  await dd.dispatch("camera", { view: "front" });
  assert.equal(seen.length, 3);
  assert.equal(log.entries.length, 3); // limit holds, oldest evicted
  assert.equal(log.entries[0].seq, 2);
});

test("dep throw becomes ok:false; unbound action reports cleanly", async () => {
  const { d } = stubDispatcher({ camera: async () => { throw new Error("no camera"); } });
  const res = await d.dispatch("camera", { view: "front" });
  assert.deepEqual(res, { ok: false, error: "no camera" });
  const bare = createDispatcher({});
  assert.deepEqual(await bare.dispatch("camera", { view: "front" }), {
    ok: false,
    error: "action not bound in this session: camera",
  });
});
