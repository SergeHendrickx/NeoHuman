// Regression tests for flow persistence helpers (browser APIs stubbed).
// Run from the flow-viewer folder:  node tests/persist.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { serializeFlow, parseFlowData, inlineBlobs, scaleToMax, flowAssetChronology,
  loadFlowList, storeFlowList, FLOW_DATA_VERSION } from "../js/persist.js";

describe("serializeFlow / parseFlowData", () => {
  const app = {
    nodes: [{ id: "a", x: 1 }, { id: "b", x: 2 }],
    edges: [{ id: "e1", from: "a", to: "b" }],
    customPositions: { a: { x: 1, y: 1 } },
    activeRunId: "run-1",
    selectedNodeId: "b",
  };

  it("round-trips the flow graph", () => {
    const snap = serializeFlow(app, "Test flow");
    assert.equal(snap.version, FLOW_DATA_VERSION);
    assert.equal(snap.name, "Test flow");
    assert.deepEqual(snap.nodes, app.nodes);
    assert.deepEqual(snap.edges, app.edges);
    assert.deepEqual(snap.customPositions, app.customPositions);
    assert.equal(snap.activeRunId, "run-1");
    assert.equal(snap.selectedNodeId, "b");
    // Deep clone: mutating the snapshot must not touch the live graph.
    snap.nodes[0].x = 999;
    assert.equal(app.nodes[0].x, 1);
    const revived = parseFlowData(JSON.parse(JSON.stringify(snap)));
    assert.ok(revived);
    assert.equal(revived.nodes[0].x, 999);
  });

  it("rejects corrupt or foreign data", () => {
    assert.equal(parseFlowData(null), null);
    assert.equal(parseFlowData("nope"), null);
    assert.equal(parseFlowData({ version: 999, nodes: [], edges: [] }), null);
    assert.equal(parseFlowData({ version: FLOW_DATA_VERSION, nodes: {}, edges: [] }), null);
    assert.equal(parseFlowData({ version: FLOW_DATA_VERSION, nodes: [{ x: 1 }], edges: [] }), null);
  });
});

describe("inlineBlobs", () => {
  it("inlines blob: URLs and leaves other URLs alone", async () => {
    const blob = new Blob(["fake-png-bytes"], { type: "image/png" });
    const url = URL.createObjectURL(blob);
    const nodes = [
      { id: "a", imgUrl: url, resultImage: "/api/x/combined.jpg" },
      { id: "b", imgUrl: "data:image/png;base64,AAA", resultImage: null },
    ];
    await inlineBlobs(nodes);
    assert.match(nodes[0].imgUrl, /^data:image\/png;base64,/);
    assert.equal(nodes[0].resultImage, "/api/x/combined.jpg");
    assert.equal(nodes[1].imgUrl, "data:image/png;base64,AAA");
    URL.revokeObjectURL(url);
  });

  it("round-trips bytes losslessly", async () => {
    const bytes = new Uint8Array([0, 1, 2, 250, 255, 16, 32]);
    const url = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
    const nodes = [{ id: "a", imgUrl: url }];
    await inlineBlobs(nodes);
    const payload = nodes[0].imgUrl.split(",", 2)[1];
    assert.deepEqual(new Uint8Array(Buffer.from(payload, "base64")), bytes);
    URL.revokeObjectURL(url);
  });

  it("passes non-image blobs through without canvas", async () => {
    const bytes = new Uint8Array([9, 8, 7, 6]);
    const url = URL.createObjectURL(new Blob([bytes], { type: "application/octet-stream" }));
    const nodes = [{ id: "a", imgUrl: url }];
    await inlineBlobs(nodes);
    assert.match(nodes[0].imgUrl, /^data:application\/octet-stream;base64,/);
    const payload = nodes[0].imgUrl.split(",", 2)[1];
    assert.deepEqual(new Uint8Array(Buffer.from(payload, "base64")), bytes);
    URL.revokeObjectURL(url);
  });
});

describe("scaleToMax", () => {
  it("shrinks past the cap preserving aspect, else no-ops", () => {
    assert.deepEqual(scaleToMax(4000, 3000), { w: 2048, h: 1536 });
    assert.deepEqual(scaleToMax(1000, 3000), { w: 683, h: 2048 });
    assert.deepEqual(scaleToMax(512, 512), { w: 512, h: 512 });
    assert.deepEqual(scaleToMax(2048, 2048), { w: 2048, h: 2048 });
  });
});

describe("flowAssetChronology", () => {
  it("lists assets oldest-first, skipping actions", () => {
    const nodes = [
      { id: "act", nodeType: "action", createdAt: 1 },
      { id: "b", nodeType: "asset", createdAt: 30 },
      { id: "a", nodeType: "asset", createdAt: 10 },
      { id: "c", nodeType: "asset", createdAt: 20 },
    ];
    assert.deepEqual(flowAssetChronology(nodes).map(n => n.id), ["a", "c", "b"]);
  });

  it("keeps insertion order on timestamp ties", () => {
    const nodes = [
      { id: "x", nodeType: "asset", createdAt: 5 },
      { id: "y", nodeType: "asset", createdAt: 5 },
    ];
    assert.deepEqual(flowAssetChronology(nodes).map(n => n.id), ["x", "y"]);
  });
});

describe("flow storage", () => {
  it("returns [] and an error without localStorage (node)", () => {
    assert.deepEqual(loadFlowList(), []);
    assert.equal(storeFlowList([]).ok, false);
  });
});
