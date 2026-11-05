// Regression tests for live progress parsing and folding (no DOM needed).
// Run from the flow-viewer folder:  node tests/progress.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { parseProgressEvents, filterProgressEvents, applyProgressEvents, progressUrl } from "../js/execution.js";

describe("parseProgressEvents", () => {
  it("parses JSON lines and advances the cursor past blanks", () => {
    const text = '{"stage":"pose","msg":"pose done","frac":0.1}\n\n{"stage":"maps","msg":"seg done","frac":0.2}\n';
    const { events, nextLine } = parseProgressEvents(text, 0);
    assert.equal(events.length, 2);
    assert.equal(events[1].stage, "maps");
    assert.equal(nextLine, 4);
    assert.deepEqual(parseProgressEvents(text, nextLine).events, []);
  });

  it("holds the cursor on a partial tail but skips finished garbage", () => {
    const partial = '{"stage":"pose","msg":"pose d';
    const r1 = parseProgressEvents(partial, 0);
    assert.deepEqual(r1.events, []);
    assert.equal(r1.nextLine, 0);
    const completed = partial + 'one","frac":0.1}\n';
    const r2 = parseProgressEvents(completed, r1.nextLine);
    assert.equal(r2.events.length, 1);
    assert.equal(r2.events[0].msg, "pose done");
    const garbage = 'not json\n{"stage":"x","msg":"ok","frac":0.5}\n';
    const r3 = parseProgressEvents(garbage, 0);
    assert.equal(r3.events.length, 1);
    assert.equal(r3.events[0].stage, "x");
  });
});

describe("filterProgressEvents", () => {
  it("keeps only this run's client id", () => {
    const events = [{ cid: "a", msg: "x" }, { cid: "b", msg: "y" }, { msg: "z" }];
    assert.deepEqual(filterProgressEvents(events, "b").map(e => e.msg), ["y"]);
    assert.equal(filterProgressEvents(events, null).length, 3);
  });
});

describe("applyProgressEvents", () => {
  it("folds fractions into bar state and messages into the run log", () => {
    const node = { id: "n", runLog: [] };
    applyProgressEvents(node, [
      { stage: "pose", msg: "pose done", frac: 0.1 },
      { stage: "maps", msg: "seg done", frac: 0.2 },
      { stage: "note", msg: "no fraction here" },
      { stage: "quiet", frac: 0.3 },
    ]);
    assert.deepEqual(node.progress, { frac: 0.3, label: "seg done" });
    assert.deepEqual(node.runLog.map(e => e.msg),
      ["▸ pose done", "▸ seg done", "▸ no fraction here"]);
    assert.ok(node.runLog.every(e => e.level === "info"));
  });

  it("clamps fractions to [0, 1]", () => {
    const node = { id: "n", runLog: [] };
    applyProgressEvents(node, [{ msg: "x", frac: 9 }]);
    assert.equal(node.progress.frac, 1);
    applyProgressEvents(node, [{ msg: "y", frac: -2 }]);
    assert.equal(node.progress.frac, 0);
  });
});

describe("progressUrl", () => {
  it("points at the chat-level live progress file", () => {
    assert.equal(progressUrl("flow-session"), "/api/chats/flow-session/file/progress.jsonl");
  });
});
