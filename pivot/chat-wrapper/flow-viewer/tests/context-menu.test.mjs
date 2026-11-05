// Regression tests for node duplication (no DOM needed).
// Run from the flow-viewer folder:  node tests/context-menu.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { duplicateNode } from "../js/context-menu.js";
import { MODEL_CATALOG } from "../js/catalog.js";

const appFor = (node) => ({ nodes: [node], renders: 0,
  renderCanvas() { this.renders++; } });

describe("duplicateNode", () => {
  const samNode = () => ({
    id: "action-meta-sam", nodeType: "action", templateKey: "action-meta-sam",
    title: "Action: Meta SAM 3.1", x: 10, y: 20,
    inputs: [{ id: "in-image", name: "Image Input", type: "image" }],
    outputs: [{ id: "out-boxes", name: "Bounding Boxes", type: "boxes", badge: "2 boxes" }],
    status: "Passed",
    resultData: { count: 2, boxes: [[0, 0, 1, 1], [2, 2, 3, 3]] },
    masks: [{ id: 0 }],
    apiCall: { hop1: {} },
    runLog: [{ t: 1, level: "info", msg: "Passed" }],
  });

  it("deep-clones pins so the copy never shares arrays with the original", () => {
    const app = appFor(samNode());
    duplicateNode(app, "action-meta-sam", 99, 100);
    const [orig, clone] = app.nodes;
    assert.notEqual(clone, orig);
    // Fresh template pins (badges describe the original's run, not this one).
    assert.deepEqual(clone.outputs.map(p => p.id),
      ["out-boxes", "out-masks", "out-confs", "out-labels"]);
    assert.notEqual(clone.outputs, orig.outputs);
    assert.notEqual(clone.inputs, orig.inputs);
    assert.notEqual(clone.inputs[0], orig.inputs[0]);
    // Mutating the copy (split splices pins in) leaves the original alone.
    clone.outputs.push({ id: "split-out-boxes-0" });
    assert.equal(orig.outputs.length, 1);
  });

  it("resets run state on action copies — no stale results ride along", () => {
    const app = appFor(samNode());
    duplicateNode(app, "action-meta-sam", 99, 100);
    const clone = app.nodes[1];
    assert.equal(clone.status, "Ready to Run");
    assert.equal(clone.resultData, null);
    assert.equal(clone.masks, null);
    assert.equal(clone.apiCall, null);
    assert.equal(clone.resultImage, null);
    assert.equal(clone.x, 99);
    assert.equal(clone.y, 100);
    assert.equal(app.selectedNodeId, clone.id);
  });
});

describe("canvas menu", () => {
  it("offers every catalog action, and every offered key resolves", () => {
    const src = readFileSync(new URL("../js/context-menu.js", import.meta.url), "utf8");
    const offered = [...src.matchAll(/data-model-key="([^"]+)"/g)].map(m => m[1]);
    assert.ok(offered.length > 0);
    for (const key of new Set(offered)) {
      assert.ok(MODEL_CATALOG[key], `menu key resolves: ${key}`);
    }
    const catalogActions = Object.values(MODEL_CATALOG)
      .filter(m => m.nodeType === "action").map(m => m.id);
    for (const id of catalogActions) {
      assert.ok(offered.includes(id), `catalog action offered: ${id}`);
    }
  });

  it("lists vision cards before generative ones, sapiens in pipeline order", () => {
    const src = readFileSync(new URL("../js/context-menu.js", import.meta.url), "utf8");
    const offered = [...src.matchAll(/data-model-key="([^"]+)"/g)].map(m => m[1]);
    const pos = (k) => offered.indexOf(k);
    assert.ok(pos("action-meta-sam") < pos("action-describe"));
    assert.ok(pos("action-describe") < pos("action-meta-muse"));
    assert.ok(pos("action-meta-muse") < pos("action-spark"));
    assert.ok(pos("action-spark") < pos("action-apply-mask"));
    assert.ok(pos("action-apply-mask") < pos("action-image-diff"));
    assert.ok(pos("action-image-diff") < pos("action-sapiens-pose"));
    assert.ok(pos("action-sapiens-pose") < pos("action-sapiens-seg"));
    assert.ok(pos("action-sapiens-seg") < pos("action-sapiens-normal"));
    assert.ok(pos("action-sapiens-normal") < pos("action-sapiens-depth"));
    assert.ok(pos("action-body-fit") < pos("action-export"));
    assert.ok(pos("action-export") < pos("action-mesh-render"));
    assert.ok(pos("action-mesh-render") < pos("action-mesh-depth"));
    assert.ok(pos("action-mesh-depth") < pos("action-mesh-normal"));
    assert.ok(pos("action-mesh-normal") < pos("action-mesh-multiview"));
    assert.ok(pos("action-mesh-multiview") < pos("action-fit-overlay"));
  });
});
