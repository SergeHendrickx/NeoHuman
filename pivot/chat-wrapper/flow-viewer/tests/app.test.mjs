// Regression tests for the pristine-preset check behind run switching (no DOM needed).
// The preset is the source photo alone — no default action card.
// Run from the flow-viewer folder:  node tests/app.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { isPristinePresetFlow, needsUnloadGuard } from "../js/app.js";

const photo = (over = {}) => ({ id: "asset-photo", nodeType: "asset",
  imgUrl: "/api/runs/r1/artifact/source", ...over });
const sam = (over = {}) => ({ id: "action-meta-sam", nodeType: "action",
  templateKey: "action-meta-sam", status: "Ready to Run", ...over });
const wire = { from: "asset-photo", to: "action-meta-sam" };

describe("isPristinePresetFlow", () => {
  it("accepts only the untouched preset: the photo alone, no wires", () => {
    assert.equal(isPristinePresetFlow({ nodes: [photo()], edges: [] }), true);
  });

  it("rejects any added card, even unrun", () => {
    assert.equal(isPristinePresetFlow(
      { nodes: [photo(), sam()], edges: [wire] }), false);
    assert.equal(isPristinePresetFlow(
      { nodes: [photo(), sam()], edges: [] }), false);
    assert.equal(isPristinePresetFlow(
      { nodes: [photo(), { id: "v1", nodeType: "asset" }], edges: [] }), false);
  });

  it("rejects user photos, ran cards, wires and empty graphs", () => {
    assert.equal(isPristinePresetFlow(
      { nodes: [photo({ imgUrl: "blob:abc" })], edges: [] }), false);
    assert.equal(isPristinePresetFlow({ nodes: [], edges: [] }), false);
    assert.equal(isPristinePresetFlow(
      { nodes: [sam()], edges: [] }), false);
    assert.equal(isPristinePresetFlow({ nodes: [photo()], edges: [wire] }), false);
  });
});

describe("needsUnloadGuard", () => {
  it("stays silent on empty graphs and the untouched preset", () => {
    assert.equal(needsUnloadGuard({ nodes: [], edges: [] }), false);
    assert.equal(needsUnloadGuard({ nodes: [photo()], edges: [] }), false);
  });

  it("guards anything the user built, ran, or dropped", () => {
    assert.equal(needsUnloadGuard(
      { nodes: [photo(), sam()], edges: [wire] }), true);
    assert.equal(needsUnloadGuard(
      { nodes: [photo(), sam({ status: "Passed" })], edges: [wire] }), true);
    assert.equal(needsUnloadGuard(
      { nodes: [photo({ imgUrl: "blob:abc" })], edges: [] }), true);
    assert.equal(needsUnloadGuard(
      { nodes: [{ id: "asset-image-1", nodeType: "asset" }], edges: [] }), true);
  });
});
