// Regression tests for describe fills, mask-ref reads and dispatch (no DOM needed).
// Run from the flow-viewer folder:  node tests/describe.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { fillDescribeResultView, upstreamMaskRef, runCard } from "../js/execution.js";

describe("fillDescribeResultView", () => {
  it("fills the description text, input echo and char-count badge", () => {
    const node = { id: "d", outputs: [{ id: "out-text", badge: "—" }] };
    fillDescribeResultView(node, {
      text: "Two people in suits.", imgUrl: "photo.jpg", badge: "description", demo: false,
    });
    assert.equal(node.resultText, "Two people in suits.");
    assert.equal(node.resultImage, "photo.jpg");
    assert.equal(node.outputs[0].badge, "20 chars");
    assert.match(node.resultImageFallback, /^data:image\/svg/);
  });

  it("flags the offline demo", () => {
    const node = { id: "d", outputs: [{ id: "out-text", badge: "—" }] };
    fillDescribeResultView(node, {
      text: "Demo description (offline).", imgUrl: "photo.jpg", badge: "demo", demo: true,
    });
    assert.match(decodeURIComponent(node.resultImageFallback), /Demo/);
    assert.equal(node.outputs[0].badge, "27 chars");
  });
});

describe("upstreamMaskRef", () => {
  const appFor = (pin) => ({ nodes: [{ id: "s", nodeType: "action", outputs: [pin] }], edges: [] });
  const edge = { from: "s", fromPort: "split-out-masks-0" };

  it("reads the backend ref off a live mask pin", () => {
    const app = appFor({ id: "split-out-masks-0", type: "mask",
      ref: { debug_id: "3", mask_id: 1 } });
    assert.deepEqual(upstreamMaskRef(app, edge), { debug_id: "3", mask_id: 1 });
  });

  it("rejects demo pins, missing pins and malformed refs", () => {
    assert.equal(upstreamMaskRef(
      appFor({ id: "split-out-masks-0", type: "mask", ref: { debug_id: null, mask_id: 0 } }),
      edge), null);
    assert.equal(upstreamMaskRef(
      appFor({ id: "split-out-masks-0", type: "mask" }), edge), null);
    assert.equal(upstreamMaskRef(
      appFor({ id: "other", type: "mask", ref: { debug_id: "3", mask_id: 0 } }), edge), null);
    assert.equal(upstreamMaskRef({ nodes: [], edges: [] }, edge), null);
    assert.equal(upstreamMaskRef(
      appFor({ id: "split-out-masks-0", type: "mask", ref: { debug_id: "3", mask_id: -1 } }),
      edge), null);
  });
});

describe("describe dispatch routing", () => {
  const appFor = (templateKey) => {
    const photo = { id: "p", nodeType: "asset", subtitle: "s.jpg",
      imgUrl: "/api/runs/r/source", outputs: [{ id: "out-photo", type: "image" }] };
    const action = { id: "d", nodeType: "action", templateKey, status: "Ready to Run",
      x: 0, y: 0,
      inputs: [{ id: "in-image", type: "image" }],
      outputs: [{ id: "out-text", type: "text" }] };
    return { app: { nodes: [photo, action], activeRunId: "r",
      edges: [{ from: "p", to: "d", fromPort: "out-photo", toPort: "in-image" }],
      renderCanvas() {}, fitView() {} }, action };
  };

  const withDocStub = async (fn) => {
    const doc = globalThis.document;
    globalThis.document = { getElementById: () => null };
    try {
      await fn();
    } finally {
      if (doc === undefined) delete globalThis.document;
      else globalThis.document = doc;
    }
  };

  it("routes describe to /api/debug/describe", async () => {
    await withDocStub(async () => {
      const { app, action } = appFor("action-describe");
      await runCard(app, "d");
      assert.equal(action.status, "Demo (Offline)");
      assert.equal(action.apiCall.hop1.url, "/api/debug/describe");
    });
  });
});
