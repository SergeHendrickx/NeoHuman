// Zoom must anchor on the cursor, not the corner: the world point under the
// anchor stays fixed across setZoom. Run:  node tests/canvas.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { setZoom, wheelHitScrollable, replacePhotoOnNode, parseImageDragPayload, spawnImageAssetFromDrag, IMAGE_DRAG_MIME } from "../js/canvas.js";

const stubApp = (scale = 1, panX = 100, panY = 60) => ({
  scale,
  panX,
  panY,
  plane: { style: {} },
  viewport: { style: {}, clientWidth: 800, clientHeight: 600 },
});

const worldAt = (app, x, y) => ({
  x: (x - app.panX) / app.scale,
  y: (y - app.panY) / app.scale,
});

describe("setZoom anchoring", () => {
  it("keeps the world point under an explicit anchor fixed (zoom in)", () => {
    const app = stubApp();
    const before = worldAt(app, 500, 400);
    setZoom(app, 1.5, { x: 500, y: 400 });
    assert.equal(app.scale, 1.5);
    const after = worldAt(app, 500, 400);
    assert.ok(Math.abs(after.x - before.x) < 1e-9, `x drifted: ${before.x} -> ${after.x}`);
    assert.ok(Math.abs(after.y - before.y) < 1e-9, `y drifted: ${before.y} -> ${after.y}`);
  });

  it("keeps the world point under an explicit anchor fixed (zoom out)", () => {
    const app = stubApp(1.5, -40, 200);
    const before = worldAt(app, 120, 90);
    setZoom(app, 0.75, { x: 120, y: 90 });
    assert.equal(app.scale, 0.75);
    const after = worldAt(app, 120, 90);
    assert.ok(Math.abs(after.x - before.x) < 1e-9, `x drifted: ${before.x} -> ${after.x}`);
    assert.ok(Math.abs(after.y - before.y) < 1e-9, `y drifted: ${before.y} -> ${after.y}`);
  });

  it("defaults to the viewport center when no anchor is passed", () => {
    const app = stubApp();
    const before = worldAt(app, 400, 300);
    setZoom(app, 1.25);
    const after = worldAt(app, 400, 300);
    assert.ok(Math.abs(after.x - before.x) < 1e-9, `x drifted: ${before.x} -> ${after.x}`);
    assert.ok(Math.abs(after.y - before.y) < 1e-9, `y drifted: ${before.y} -> ${after.y}`);
  });

  it("holds the anchor even when the scale clamps", () => {
    const app = stubApp();
    setZoom(app, 99, { x: 500, y: 400 });
    assert.equal(app.scale, 2.0);
    setZoom(app, -5, { x: 500, y: 400 });
    assert.equal(app.scale, 0.3);
  });

  it("updates the plane transform and grid", () => {
    const app = stubApp();
    setZoom(app, 1.5, { x: 500, y: 400 });
    assert.match(app.plane.style.transform, /translate\(.+px, .+px\) scale\(1\.5\)/);
    assert.equal(app.viewport.style.backgroundSize, "36px 36px");
  });
});

describe("replacePhotoOnNode", () => {
  const photoNode = () => ({
    id: "asset-photo", nodeType: "asset", assetKind: "image",
    title: "Asset: Source Photo", subtitle: "old.jpg",
    imgUrl: "blob:old", imgBadge: "old.jpg · 10x10",
    tensorShape: "[10, 10, 3] uint8", imageWidth: 10, imageHeight: 10,
    inputs: [{ id: "in-view" }], outputs: [{ id: "out-photo" }],
    x: 80, y: 140,
  });
  const samNode = () => ({
    id: "action-meta-sam", nodeType: "action", templateKey: "action-meta-sam",
    status: "Passed", resultImage: "overlay.jpg",
    inputs: [{ id: "in-image" }], outputs: [{ id: "out-boxes" }],
    x: 520, y: 140,
  });
  const appFor = (nodes, edges) => ({
    nodes, edges, selectedNodeId: "action-meta-sam", customPositions: {},
    userPhoto: null, renders: 0, dimsProbes: [], popovers: 0,
    revokeUserPhoto() {},
    renderCanvas() { this.renders++; },
    resolvePhotoDims(id, src) { this.dimsProbes.push([id, src]); },
    resolveAssetDims() {},
    closePopover() { this.popovers++; },
  });
  const newFile = () => {
    const f = new Blob(["fake-image-bytes"]);
    f.name = "new.jpg";
    return f;
  };

  it("swaps the photo in place and keeps the graph", () => {
    const photo = photoNode();
    const sam = samNode();
    const app = appFor([photo, sam], [
      { id: "e1", from: "asset-photo", to: "action-meta-sam", fromPort: "out-photo", toPort: "in-image" },
    ]);
    assert.equal(replacePhotoOnNode(app, "asset-photo", newFile()), true);
    // Graph survives: same nodes, same edge, same positions.
    assert.deepEqual(app.nodes.map(n => n.id), ["asset-photo", "action-meta-sam"]);
    assert.deepEqual(app.edges.map(e => e.id), ["e1"]);
    assert.deepEqual([photo.x, photo.y], [80, 140]);
    // Photo swapped.
    assert.match(photo.imgUrl, /^blob:/);
    assert.notEqual(photo.imgUrl, "blob:old");
    assert.equal(photo.subtitle, "new.jpg");
    assert.equal(photo.imageWidth, null);
    // Stale downstream results reset (wiring survives).
    assert.equal(sam.status, "Ready to Run");
    assert.equal(sam.resultImage, null);
    // Card re-rendered, selected, dims re-probed.
    assert.ok(app.renders > 0);
    assert.equal(app.selectedNodeId, "asset-photo");
    assert.deepEqual(app.dimsProbes, [["asset-photo", photo.imgUrl]]);
  });

  it("returns false for an unknown card and touches nothing", () => {
    const photo = photoNode();
    const app = appFor([photo], []);
    assert.equal(replacePhotoOnNode(app, "nope", newFile()), false);
    assert.equal(photo.imgUrl, "blob:old");
    assert.equal(app.renders, 0);
  });
});

describe("parseImageDragPayload", () => {
  const dt = (types, data) => ({ types, getData: () => data });

  it("reads a valid internal image payload", () => {
    const p = parseImageDragPayload(dt([IMAGE_DRAG_MIME],
      JSON.stringify({ imgUrl: "/api/x/overlay.jpg", label: "SAM result" })));
    assert.deepEqual(p, { imgUrl: "/api/x/overlay.jpg", label: "SAM result" });
  });

  it("defaults a missing label", () => {
    const p = parseImageDragPayload(dt([IMAGE_DRAG_MIME],
      JSON.stringify({ imgUrl: "blob:abc" })));
    assert.deepEqual(p, { imgUrl: "blob:abc", label: "dragged image" });
  });

  it("returns null for foreign drags and malformed payloads", () => {
    assert.equal(parseImageDragPayload(null), null);
    assert.equal(parseImageDragPayload({}), null);
    assert.equal(parseImageDragPayload(dt(["Files"], "{}")), null);
    assert.equal(parseImageDragPayload(dt([IMAGE_DRAG_MIME], "not-json")), null);
    assert.equal(parseImageDragPayload(dt([IMAGE_DRAG_MIME], "{}")), null);
    assert.equal(parseImageDragPayload(dt([IMAGE_DRAG_MIME],
      JSON.stringify({ imgUrl: "" }))), null);
  });
});

describe("spawnImageAssetFromDrag", () => {
  const dragApp = (nodes, edges) => ({
    nodes, edges, selectedNodeId: null, renders: 0, dimsProbes: [],
    renderCanvas() { this.renders++; },
    resolveAssetDims(id, src) { this.dimsProbes.push([id, src]); },
  });

  it("adds an image asset card and leaves the graph untouched", () => {
    const photo = { id: "asset-photo", nodeType: "asset", x: 80, y: 140 };
    const sam = { id: "action-meta-sam", nodeType: "action", x: 520, y: 140 };
    const edge = { id: "e1", from: "asset-photo", to: "action-meta-sam" };
    const app = dragApp([photo, sam], [edge]);
    const id = spawnImageAssetFromDrag(app,
      { imgUrl: "/api/chats/c/file/debug/0/boxes_overlay.jpg", label: "SAM boxes" }, 300, 400);
    // Additive only: old nodes, edges and positions intact.
    assert.equal(app.nodes.length, 3);
    assert.deepEqual(app.nodes.slice(0, 2), [photo, sam]);
    assert.deepEqual(app.edges, [edge]);
    assert.deepEqual([photo.x, photo.y], [80, 140]);
    // New card carries the dragged image.
    const card = app.nodes[2];
    assert.equal(card.id, id);
    assert.match(card.id, /^asset-image-/);
    assert.equal(card.nodeType, "asset");
    assert.equal(card.assetKind, "image");
    assert.equal(card.imgUrl, "/api/chats/c/file/debug/0/boxes_overlay.jpg");
    assert.equal(card.subtitle, "SAM boxes");
    assert.deepEqual([card.x, card.y], [300, 400]);
    assert.ok(card.outputs.some(o => o.id === "out-photo" && o.type === "image"));
    assert.equal(app.selectedNodeId, id);
    assert.ok(app.renders > 0);
    assert.deepEqual(app.dimsProbes, [[id, card.imgUrl]]);
  });

  it("clamps drop coordinates and falls back cleanly", () => {
    const app = dragApp([], []);
    const id = spawnImageAssetFromDrag(app, { imgUrl: "blob:x" }, -50, 10);
    const card = app.nodes[0];
    assert.equal(card.id, id);
    assert.deepEqual([card.x, card.y], [60, 60]);
    assert.equal(card.subtitle, "dragged image");
  });
});

describe("wheelHitScrollable", () => {
  // Fake DOM chain: leaf -> scroller -> viewport(stop). Geometry only under node.
  const chain = (scrollTop = 40) => {
    const viewport = { nodeType: 1, parentElement: null };
    const scroller = { nodeType: 1, scrollHeight: 500, clientHeight: 200,
      scrollWidth: 200, clientWidth: 200, scrollTop, scrollLeft: 0, parentElement: viewport };
    const leaf = { nodeType: 1, scrollHeight: 20, clientHeight: 20,
      scrollWidth: 20, clientWidth: 20, scrollTop: 0, scrollLeft: 0, parentElement: scroller };
    return { viewport, scroller, leaf };
  };

  it("scrolls content mid-pane instead of zooming (both directions)", () => {
    const { viewport, leaf } = chain(40);
    assert.equal(wheelHitScrollable(leaf, 100, 0, viewport), true);
    assert.equal(wheelHitScrollable(leaf, -100, 0, viewport), true);
  });

  it("falls through to zoom at the scroll limit", () => {
    const { viewport, leaf } = chain(300); // max = 500-200
    assert.equal(wheelHitScrollable(leaf, 100, 0, viewport), false);
    assert.equal(wheelHitScrollable(leaf, -100, 0, viewport), true);
    const top = chain(0);
    assert.equal(wheelHitScrollable(top.leaf, -100, 0, top.viewport), false);
    assert.equal(wheelHitScrollable(top.leaf, 100, 0, top.viewport), true);
  });

  it("zooms over non-scrollable card chrome", () => {
    const viewport = { nodeType: 1, parentElement: null };
    const leaf = { nodeType: 1, scrollHeight: 20, clientHeight: 20,
      scrollWidth: 20, clientWidth: 20, scrollTop: 0, scrollLeft: 0, parentElement: viewport };
    assert.equal(wheelHitScrollable(leaf, 100, 0, viewport), false);
  });

  it("resolves text-node targets to their parent", () => {
    const { viewport, scroller } = chain(40);
    const text = { nodeType: 3, parentElement: scroller };
    assert.equal(wheelHitScrollable(text, 100, 0, viewport), true);
  });

  it("honours horizontal gestures on wide panes", () => {
    const viewport = { nodeType: 1, parentElement: null };
    const wide = { nodeType: 1, scrollHeight: 100, clientHeight: 100,
      scrollWidth: 600, clientWidth: 200, scrollTop: 0, scrollLeft: 50, parentElement: viewport };
    assert.equal(wheelHitScrollable(wide, 0, 60, viewport), true);
    assert.equal(wheelHitScrollable(wide, 0, -60, viewport), true);
    assert.equal(wheelHitScrollable(wide, 100, 0, viewport), false);
  });

  it("never looks past the viewport boundary", () => {
    const outer = { nodeType: 1, scrollHeight: 900, clientHeight: 100,
      scrollWidth: 100, clientWidth: 100, scrollTop: 40, scrollLeft: 0, parentElement: null };
    const viewport = { nodeType: 1, parentElement: outer };
    const leaf = { nodeType: 1, scrollHeight: 10, clientHeight: 10,
      scrollWidth: 10, clientWidth: 10, scrollTop: 0, scrollLeft: 0, parentElement: viewport };
    assert.equal(wheelHitScrollable(leaf, 100, 0, viewport), false);
  });
});
