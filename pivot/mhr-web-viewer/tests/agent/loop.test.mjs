import test from "node:test";
import assert from "node:assert/strict";
import {
  SYSTEM_PROMPT,
  MAX_ITERS,
  DEFAULT_BASE_URL,
  DEFAULT_MODEL,
  formatSnapshot,
  buildBody,
  chatCompletion,
  extractToolCalls,
  runAgentTurn,
} from "../../static/js/agent.js";
import { createDispatcher } from "../../static/js/actions.js";

const CONFIG = { baseUrl: "https://api.meta.ai/v1/", key: "k", model: "m" };

function okFetch(message, capture = {}) {
  return async (url, opts) => {
    capture.url = url;
    capture.opts = opts;
    return {
      ok: true,
      status: 200,
      json: async () => ({ choices: [{ message }] }),
      text: async () => "",
    };
  };
}

function scriptFetch(messages, capture = {}) {
  let i = 0;
  return async (url, opts) => {
    capture.url = url;
    capture.opts = opts;
    const message = messages[Math.min(i++, messages.length - 1)];
    return {
      ok: true,
      status: 200,
      json: async () => ({ choices: [{ message }] }),
      text: async () => "",
    };
  };
}

function stubDispatcher(overrides = {}) {
  const calls = [];
  const record = (name) => async (a) => {
    calls.push([name, a]);
    return { ran: name };
  };
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

const toolMsg = (id, name, args) => ({
  role: "assistant",
  content: null,
  tool_calls: [{ id, type: "function", function: { name, arguments: JSON.stringify(args) } }],
});

test("snapshot formats scene state, skips missing fields", () => {
  assert.equal(formatSnapshot(), "(empty scene)");
  const full = formatSnapshot({
    agents: 12, gpu: 12, cpu: 0, selectedId: 3,
    selected: { behavior: "circle", x: 1.2, z: -0.5 },
    behavior: "wander", lodColors: true, simRunning: false, view: "face",
  });
  assert.match(full, /agents: 12 \(12 GPU, 0 CPU\)/);
  assert.match(full, /selected: agent 3 \(behavior circle, pos 1.2, -0.5\)/);
  assert.match(full, /lodColors: on/);
  assert.match(full, /sim: paused/);
  assert.equal(formatSnapshot({ selectedId: 0 }), "selected: actor #0 (0)");
});

test("request body carries model, tools and low temperature", () => {
  const body = buildBody("m", [{ role: "user", content: "hi" }]);
  assert.equal(body.model, "m");
  assert.equal(body.tools.length, 10);
  assert.equal(body.tool_choice, "auto");
  assert.equal(body.temperature, 0.2);
  assert.ok(SYSTEM_PROMPT.includes("Actor #0"));
});

test("tool calls extract; bad args JSON becomes an error, not a throw", () => {
  const calls = extractToolCalls({
    tool_calls: [
      { id: "1", function: { name: "camera", arguments: '{"view":"face"}' } },
      { id: "2", function: { name: "camera", arguments: "{oops" } },
    ],
  });
  assert.deepEqual(calls[0], { id: "1", name: "camera", args: { view: "face" }, argsError: null });
  assert.equal(calls[1].args, null);
  assert.match(calls[1].argsError, /bad arguments JSON/);
  assert.deepEqual(extractToolCalls({ content: "no tools" }), []);
});

test("chatCompletion joins URL, sends auth, surfaces HTTP errors", async () => {
  const capture = {};
  const msg = await chatCompletion(
    { ...CONFIG, messages: [{ role: "user", content: "hi" }] },
    okFetch({ role: "assistant", content: "yo" }, capture)
  );
  assert.equal(msg.content, "yo");
  assert.equal(capture.url, "https://api.meta.ai/v1/chat/completions");
  const sent = JSON.parse(capture.opts.body);
  assert.equal(sent.model, "m");
  assert.equal(sent.messages.length, 1);
  assert.equal(sent.messages[0].content, "hi");
  assert.equal(capture.opts.headers.Authorization, "Bearer k");
  await assert.rejects(
    chatCompletion({ ...CONFIG, messages: [] }, async () => ({
      ok: false, status: 401, statusText: "nope", text: async () => "bad key",
    })),
    /HTTP 401: bad key/
  );
  await assert.rejects(
    chatCompletion({ ...CONFIG, messages: [] }, async () => { throw new Error("down"); }),
    /request failed: down/
  );
});

test("turn with no tools returns the text", async () => {
  const { d } = stubDispatcher();
  const events = [];
  const { messages, error } = await runAgentTurn({
    dispatcher: d,
    messages: [{ role: "user", content: "hi" }],
    snapshotText: "(empty scene)",
    config: CONFIG,
    fetchImpl: okFetch({ role: "assistant", content: "hello" }),
    onEvent: (e) => events.push(e),
  });
  assert.equal(error, undefined);
  assert.equal(messages.length, 2);
  assert.deepEqual(events, [{ type: "text", text: "hello" }]);
});

test("tool loop dispatches, feeds results back, ends on text", async () => {
  const { d, calls } = stubDispatcher();
  const events = [];
  const { messages } = await runAgentTurn({
    dispatcher: d,
    messages: [{ role: "user", content: "spawn" }],
    snapshotText: "agents: 0",
    config: CONFIG,
    fetchImpl: scriptFetch([
      toolMsg("c1", "spawnAgents", { count: 5 }),
      { role: "assistant", content: "spawned 5" },
    ]),
    onEvent: (e) => events.push(e.type),
  });
  assert.deepEqual(calls, [["spawnAgents", { count: 5 }]]);
  assert.deepEqual(events, ["tool-call", "tool-result", "text"]);
  const toolMsgBack = messages.find((m) => m.role === "tool");
  assert.equal(toolMsgBack.tool_call_id, "c1");
  assert.match(toolMsgBack.content, /"ran":"spawnAgents"/);
});

test("destructive actions need approval; denial skips dispatch", async () => {
  const { d, calls } = stubDispatcher();
  const events = [];
  await runAgentTurn({
    dispatcher: d,
    messages: [{ role: "user", content: "clear" }],
    snapshotText: "agents: 5",
    config: CONFIG,
    fetchImpl: scriptFetch([
      toolMsg("c1", "clearAgents", {}),
      { role: "assistant", content: "kept them" },
    ]),
    approve: async () => false,
    onEvent: (e) => events.push(e.type),
  });
  assert.equal(calls.length, 0);
  assert.deepEqual(events, ["tool-call", "approval", "tool-result", "text"]);

  let approved = null;
  const d2 = stubDispatcher();
  await runAgentTurn({
    dispatcher: d2.d,
    messages: [{ role: "user", content: "clear" }],
    snapshotText: "agents: 5",
    config: CONFIG,
    fetchImpl: scriptFetch([toolMsg("c1", "clearAgents", {}), { role: "assistant", content: "ok" }]),
    approve: async (name, args) => {
      approved = [name, args];
      return true;
    },
    onEvent: () => {},
  });
  assert.deepEqual(approved, ["clearAgents", {}]);
  assert.equal(d2.calls.length, 1);
});

test("loop stops after maxIters; fetch errors surface once", async () => {
  const { d } = stubDispatcher();
  const events = [];
  const res = await runAgentTurn({
    dispatcher: d,
    messages: [{ role: "user", content: "go" }],
    snapshotText: "",
    config: CONFIG,
    maxIters: 2,
    fetchImpl: scriptFetch([toolMsg("c1", "camera", { view: "front" })]),
    onEvent: (e) => events.push(e.type),
  });
  assert.match(res.error, /stopped after 2 tool rounds/);
  assert.equal(events.filter((t) => t === "tool-call").length, 2);

  const events2 = [];
  const res2 = await runAgentTurn({
    dispatcher: d,
    messages: [{ role: "user", content: "go" }],
    snapshotText: "",
    config: CONFIG,
    fetchImpl: async () => { throw new Error("net down"); },
    onEvent: (e) => events2.push(e),
  });
  assert.match(res2.error, /net down/);
  assert.deepEqual(events2, [{ type: "error", error: "request failed: net down" }]);
});

test("defaults point at the Meta Model API", () => {
  assert.equal(DEFAULT_BASE_URL, "https://api.meta.ai/v1");
  assert.equal(DEFAULT_MODEL, "muse-spark-1.3");
});
