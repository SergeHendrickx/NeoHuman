// Regression tests for wire status classes and wire deletion (no DOM needed).
// Run from the flow-viewer folder:  node tests/wires.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { edgeStatusClass, deleteEdge } from "../js/wires.js";
import { renderWireMenuHtml } from "../js/context-menu.js";

const action = (id, status) => ({ id, nodeType: "action", status });
const asset = (id, status) => ({ id, nodeType: "asset", status });

describe("edgeStatusClass", () => {
  it("animates only the currently executing card's wires", () => {
    const app = { nodes: [action("a", "Running...")], edges: [] };
    assert.equal(edgeStatusClass(app, { from: "a" }), "active");
  });

  it("paints verified passes solid green, demos solid orange", () => {
    const app = { nodes: [action("a", "Passed"), action("b", "Demo (Offline)")], edges: [] };
    assert.equal(edgeStatusClass(app, { from: "a" }), "success");
    assert.equal(edgeStatusClass(app, { from: "b" }), "pending");
  });

  it("leaves unrun cards' wires idle", () => {
    const app = { nodes: [action("a", "Ready to Run")], edges: [] };
    assert.equal(edgeStatusClass(app, { from: "a" }), "");
    assert.equal(edgeStatusClass(app, { from: "missing" }), "");
  });

  it("paints failed cards' wires solid red", () => {
    const app = { nodes: [action("a", "Failed")], edges: [] };
    assert.equal(edgeStatusClass(app, { from: "a" }), "error");
  });

  it("paints data-holding assets orange, waiting viewers idle", () => {
    const app = { nodes: [asset("p", "Cached in RAM"), asset("v", "Waiting for Input"),
      asset("w", "Viewing ← Box 1"), asset("m", "Artifact")], edges: [] };
    assert.equal(edgeStatusClass(app, { from: "p" }), "pending");
    assert.equal(edgeStatusClass(app, { from: "v" }), "");
    assert.equal(edgeStatusClass(app, { from: "w" }), "pending");
    assert.equal(edgeStatusClass(app, { from: "m" }), "pending");
  });
});

describe("deleteEdge", () => {
  const stubApp = () => ({
    nodes: [
      { id: "a", nodeType: "action", title: "Action: A" },
      { id: "v", nodeType: "asset", templateKey: "asset-photo-viewer",
        title: "Asset: Viewer", status: "Viewing ← Photo", inputs: [{ id: "in-view" }] },
    ],
    edges: [
      { id: "e1", from: "a", to: "v", fromPort: "out-image", toPort: "in-view" },
      { id: "e2", from: "a", to: "a", fromPort: "out-x", toPort: "in-x" },
    ],
    renders: 0,
    closePopover() {},
    renderCanvas() { this.renders++; },
    resolveAssetDims() {},
  });

  it("removes exactly the named wire and repaints", () => {
    const app = stubApp();
    assert.equal(deleteEdge(app, "e1"), true);
    assert.deepEqual(app.edges.map(e => e.id), ["e2"]);
    assert.equal(app.nodes.length, 2);
    assert.equal(app.renders, 1);
  });

  it("no-ops on unknown ids (no repaint)", () => {
    const app = stubApp();
    assert.equal(deleteEdge(app, "ghost"), false);
    assert.equal(app.edges.length, 2);
    assert.equal(app.renders, 0);
  });

  it("falls an unwired viewer back to waiting", () => {
    const app = stubApp();
    deleteEdge(app, "e1");
    const viewer = app.nodes.find(n => n.id === "v");
    assert.equal(viewer.status, "Waiting for Input");
  });
});

describe("renderWireMenuHtml", () => {
  it("identifies the wire route and offers delete", () => {
    const s = renderWireMenuHtml(
      { id: "e1", from: "a", to: "v", label: "raw RGB" },
      "Action: SAM", "Asset: Viewer");
    assert.match(s, /Action: SAM → Asset: Viewer · raw RGB/);
    assert.match(s, /data-wire-action="delete"/);
    assert.match(s, /Delete wire/);
  });

  it("escapes titles and labels", () => {
    const s = renderWireMenuHtml(
      { id: "e1", from: "a", to: "v", label: "<b>" }, "<i>A</i>", "V");
    assert.ok(!s.includes("<i>A</i>") && !s.includes("<b>"));
  });
});
