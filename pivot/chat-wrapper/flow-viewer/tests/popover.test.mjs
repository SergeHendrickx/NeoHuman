// Regression tests for the downstream model picker filter (no DOM needed).
// Run from the flow-viewer folder:  node tests/popover.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { compatibleModels } from "../js/popover.js";

describe("compatibleModels", () => {
  it("offers cards whose pins accept the type, not just acceptsInput", () => {
    // Sapiens pins take mask via pin-level accepts — the picker must agree.
    const ids = compatibleModels("mask").map(m => m.id);
    assert.ok(ids.includes("action-sapiens-pose"), "sapiens-pose offered for mask");
    assert.ok(ids.includes("asset-photo-viewer"), "viewer offered for mask");
    assert.ok(ids.includes("action-spark"), "spark offered for mask");
  });

  it("keeps the registry compatibilities (box into crop pins)", () => {
    const ids = compatibleModels("box").map(m => m.id);
    assert.ok(ids.includes("action-meta-sam"), "sam offered for box");
    assert.ok(ids.includes("action-sapiens-seg"), "sapiens-seg offered for box");
  });

  it("offers mesh consumers for meshes and nothing for vectors", () => {
    assert.ok(compatibleModels("mesh").some(m => m.id === "action-export"));
    assert.deepEqual(compatibleModels("vector"), []);
  });
});
