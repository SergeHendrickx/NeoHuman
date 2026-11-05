// Regression tests for traversal ordering (no DOM needed).
// Run from the flow-viewer folder:  node tests/traverse.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { orderReadyActions } from "../js/execution.js";

const action = (id, status = "Ready to Run") => ({ id, nodeType: "action", status });
const asset = (id) => ({ id, nodeType: "asset" });

describe("orderReadyActions", () => {
  it("orders ready actions upstream-first along the wires", () => {
    const app = { nodes: [action("c"), action("a"), action("b")], edges: [
      { from: "a", to: "b" }, { from: "b", to: "c" },
    ] };
    assert.deepEqual(orderReadyActions(app).map(n => n.id), ["a", "b", "c"]);
  });

  it("skips actions that already ran, failed, or are running", () => {
    const app = { nodes: [action("a"), action("b", "Passed"), action("c", "Failed"),
      action("d", "Running..."), action("e", "Demo (Offline)")], edges: [] };
    assert.deepEqual(orderReadyActions(app).map(n => n.id), ["a"]);
  });

  it("passes through assets and tolerates cycles", () => {
    const app = { nodes: [action("a"), asset("v"), action("b")], edges: [
      { from: "a", to: "v" }, { from: "v", to: "b" },
    ] };
    assert.deepEqual(orderReadyActions(app).map(n => n.id), ["a", "b"]);
    const cyc = { nodes: [action("a"), action("b")], edges: [
      { from: "a", to: "b" }, { from: "b", to: "a" },
    ] };
    assert.deepEqual(orderReadyActions(cyc).map(n => n.id).sort(), ["a", "b"]);
  });
});
