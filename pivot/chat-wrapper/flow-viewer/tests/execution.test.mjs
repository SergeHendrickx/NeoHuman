// Regression tests for SAM list derivation and Explode item building.
// Run from the flow-viewer folder:  node tests/execution.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { deriveResultData, explodeListItems, buildSplitPins, splitListPin, collapseListPin,
  clearDownstreamOutputs, upstreamImageUrl, collectImageInputs, resolvePromptText,
  fillSparkResultView, fillMuseResultView,
  sapiensInputEdge, fillSapiensResultView,
  produceOutputsForAction, runCard, failUnreachable, executeDescribe,
  fillApplyMaskResultView, executeApplyMask,
  upstreamMeshRef, fillMeshRenderResultView, executeMeshRender,
  meshInputEdge, meshInputDiagnosis, executeMeshMultiview,
  fitDims, absDiffStats, fillImageDiffResultView, executeImageDiff,
  upstreamFitCamera, upstreamFitDiagnosis, fillFitOverlayResultView, executeFitOverlay,
  buildSegClassNode, spawnSegClassCards, validSegLabels, segClassThumbUrl } from "../js/execution.js";

const MASKS = [
  { id: 0, object_id: "0", box: [80, 95, 405, 1000], confidence: 0.987, pixels: 284120,
    crop_url: "/api/x/crop_0.jpg", file_url: "/api/x/mask_0.jpg" },
  { id: 1, object_id: "1", box: [390, 255, 845, 1000], confidence: 0.979, pixels: 219430 },
];

describe("deriveResultData", () => {
  it("builds the four typed lists with derived labels", () => {
    const d = deriveResultData(MASKS, "people", 1000, 1000);
    assert.equal(d.count, 2);
    assert.deepEqual(d.boxes, [[80, 95, 405, 1000], [390, 255, 845, 1000]]);
    assert.equal(d.masks.length, 2);
    assert.deepEqual(d.confidences, [0.987, 0.979]);
    assert.deepEqual(d.labels, ["people 1", "people 2"]);
    assert.equal(d.concept, "people");
  });

  it("tolerates missing fields and non-array input", () => {
    const d = deriveResultData([{ id: 0 }], "", null, undefined);
    assert.equal(d.count, 1);
    assert.deepEqual(d.boxes, [[0, 0, 0, 0]]);
    assert.deepEqual(d.confidences, [0.985]);
    assert.deepEqual(d.labels, ["person 1"]);
    assert.deepEqual(deriveResultData(null, "x", 0, 0).boxes, []);
  });

  it("carries the backend debug session id", () => {
    assert.equal(deriveResultData(MASKS, "people", 1, 1, "dbg-9").debugId, "dbg-9");
    assert.equal(deriveResultData(MASKS, "people", 1, 1).debugId, null);
  });
});

describe("explodeListItems", () => {
  const data = deriveResultData(MASKS, "people", 1000, 1000);

  it("explodes boxes into crop image items", () => {
    const items = explodeListItems(data, "boxes");
    assert.equal(items.length, 2);
    assert.equal(items[0].title, "Box 1");
    assert.equal(items[0].subtitle, "[80, 95, 405, 1000]");
    assert.equal(items[0].assetKind, "image");
    assert.equal(items[0].imgUrl, "/api/x/crop_0.jpg");
    assert.equal(items[0].outputType, "box");
    assert.equal(items[0].outputBadge, "325×905");
    // Record without URLs falls back to an offline placeholder, never "".
    assert.match(items[1].imgUrl, /^data:image\/svg/);
  });

  it("explodes masks into overlay image items", () => {
    const items = explodeListItems(data, "masks");
    assert.equal(items.length, 2);
    assert.equal(items[0].title, "Mask 1");
    assert.equal(items[0].imgUrl, "/api/x/mask_0.jpg");
    assert.equal(items[0].outputType, "mask");
    assert.match(items[1].imgUrl, /^data:image\/svg/);
  });

  it("explodes confidences and labels into scalar data items", () => {
    const confs = explodeListItems(data, "confidences");
    assert.equal(confs.length, 2);
    assert.equal(confs[0].assetKind, "data");
    assert.equal(confs[0].dataValue, 0.987);
    assert.equal(confs[0].dataKind, "number");
    assert.equal(confs[0].outputType, "number");

    const labels = explodeListItems(data, "labels");
    assert.equal(labels[1].dataValue, "people 2");
    assert.equal(labels[1].outputType, "text");
  });

  it("returns [] for unknown kinds and missing data", () => {
    assert.deepEqual(explodeListItems(data, "crops"), []);
    assert.deepEqual(explodeListItems(null, "boxes"), []);
    assert.deepEqual(explodeListItems({ count: 0 }, "boxes"), []);
  });
});

describe("buildSplitPins", () => {
  const data = deriveResultData(MASKS, "people", 1000, 1000);

  it("builds thumbnail item pins for image lists", () => {
    const pins = buildSplitPins(data, "boxes", "out-boxes");
    assert.equal(pins.length, 2);
    assert.equal(pins[0].id, "split-out-boxes-0");
    assert.equal(pins[0].type, "box");
    assert.equal(pins[0].thumb, "/api/x/crop_0.jpg");
    assert.equal(pins[0].thumbValue, null);
    assert.match(pins[1].thumb, /^data:image\/svg/);
  });

  it("builds value thumbnails for scalar lists", () => {
    const pins = buildSplitPins(data, "confidences", "out-confs");
    assert.equal(pins[0].thumb, null);
    assert.equal(pins[0].thumbValue, "0.987");
    assert.equal(pins[0].type, "number");
  });

  it("attaches backend mask refs to every item", () => {
    const withDbg = deriveResultData(MASKS, "people", 1000, 1000, "dbg-9");
    const pins = buildSplitPins(withDbg, "masks", "out-masks");
    assert.deepEqual(pins[0].ref, { debug_id: "dbg-9", mask_id: 0 });
    assert.deepEqual(pins[1].ref, { debug_id: "dbg-9", mask_id: 1 });
    const boxes = buildSplitPins(withDbg, "boxes", "out-boxes");
    assert.deepEqual(boxes[0].ref, { debug_id: "dbg-9", mask_id: 0 });
  });

  it("returns [] for unknown kinds", () => {
    assert.deepEqual(buildSplitPins(data, "crops", "out"), []);
  });
});

describe("splitListPin / collapseListPin", () => {
  const samNode = (over = {}) => ({
    id: "sam", title: "Action: Meta SAM 3.1",
    outputs: [
      { id: "out-boxes", name: "Bounding Boxes", type: "boxes" },
      { id: "out-masks", name: "Masks", type: "masks" },
    ],
    resultData: deriveResultData(MASKS, "people", 1000, 1000),
    ...over,
  });
  const appFor = (node) => ({ nodes: [node], edges: [], renders: 0, probes: [],
    renderCanvas() { this.renders++; },
    resolveAssetDims(id, src, badge) { this.probes.push([id, src, badge]); } });

  it("splits a filled list pin into item pins after it", () => {
    const app = appFor(samNode());
    assert.equal(splitListPin(app, "sam", "out-masks"), true);
    const node = app.nodes[0];
    assert.deepEqual(node.outputs.map(p => p.id),
      ["out-boxes", "out-masks", "split-out-masks-0", "split-out-masks-1"]);
    assert.deepEqual(node.splitPins, { "out-masks": ["split-out-masks-0", "split-out-masks-1"] });
    assert.equal(app.renders, 1);
    // Splitting twice is a no-op.
    assert.equal(splitListPin(app, "sam", "out-masks"), false);
  });

  it("refuses non-list pins and unfilled lists", () => {
    const app = appFor(samNode({ outputs: [{ id: "out-photo", type: "image" }] }));
    assert.equal(splitListPin(app, "sam", "out-photo"), false);
    const unrun = appFor(samNode({ resultData: null }));
    assert.equal(splitListPin(unrun, "sam", "out-masks"), false);
    const empty = appFor(samNode({ resultData: deriveResultData([], "x", 0, 0) }));
    assert.equal(splitListPin(empty, "sam", "out-masks"), false);
    assert.equal(app.renders, 0);
  });

  it("collapses item pins and drops only their wires", () => {
    const app = appFor(samNode());
    splitListPin(app, "sam", "out-masks");
    app.edges = [
      { id: "e1", from: "sam", to: "down", fromPort: "split-out-masks-0", toPort: "in" },
      { id: "e2", from: "sam", to: "other", fromPort: "out-boxes", toPort: "in" },
    ];
    assert.equal(collapseListPin(app, "sam", "out-masks"), true);
    const node = app.nodes[0];
    assert.deepEqual(node.outputs.map(p => p.id), ["out-boxes", "out-masks"]);
    assert.equal(node.splitPins, undefined);
    assert.deepEqual(app.edges.map(e => e.id), ["e2"]);
    assert.equal(collapseListPin(app, "sam", "out-masks"), false);
  });

  it("reverts viewers wired from collapsed pins to Waiting for Input", () => {
    const viewer = { id: "v1", nodeType: "asset", templateKey: "asset-photo-viewer",
      imgUrl: "placeholder", status: "Viewing ← Mask 1", inputs: [{ id: "in-view" }] };
    const app = appFor(samNode());
    app.nodes.push(viewer);
    splitListPin(app, "sam", "out-masks");
    app.edges = [
      { id: "e1", from: "sam", to: "v1", fromPort: "split-out-masks-0", toPort: "in-view" },
    ];
    assert.equal(collapseListPin(app, "sam", "out-masks"), true);
    assert.equal(viewer.status, "Waiting for Input");
    assert.deepEqual(app.probes, [["v1", "placeholder", false]]);
  });
});

describe("clearDownstreamOutputs", () => {
  it("deletes generated assets but spares viewers and the source photo", () => {
    const sam = { id: "sam", nodeType: "action", templateKey: "action-meta-sam", status: "Passed" };
    const gen = { id: "asset-x", nodeType: "asset", status: "Artifact" };
    const viewer = { id: "v1", nodeType: "asset", templateKey: "asset-photo-viewer",
      imgUrl: "placeholder", status: "Viewing ← Box 1", inputs: [{ id: "in-view" }] };
    const photo = { id: "asset-photo", nodeType: "asset", status: "Cached in RAM" };
    const app = { nodes: [sam, gen, viewer, photo], edges: [
      { id: "e1", from: "sam", to: "asset-x", fromPort: "out", toPort: "in" },
      { id: "e2", from: "sam", to: "v1", fromPort: "out", toPort: "in-view" },
      { id: "e3", from: "sam", to: "asset-photo", fromPort: "out", toPort: "in-view" },
    ], selectedNodeId: "sam", customPositions: {},
      renders: 0, probes: [], popovers: 0,
      renderCanvas() { this.renders++; },
      resolveAssetDims(id, src, badge) { this.probes.push([id, src, badge]); },
      closePopover() { this.popovers++; } };
    clearDownstreamOutputs(app, "sam");
    assert.deepEqual(app.nodes.map(n => n.id).sort(), ["asset-photo", "sam", "v1"]);
    assert.equal(app.nodes.find(n => n.id === "v1").status, "Viewing ← Box 1");
  });
});

describe("generic image / prompt gathering", () => {
  const sam = {
    id: "sam", nodeType: "action", resultImage: "overlay.jpg",
    outputs: [
      { id: "out-boxes", name: "Bounding Boxes", type: "boxes" },
      { id: "split-out-boxes-1", name: "Box 2", type: "box", thumb: "crop2.jpg" },
      { id: "split-out-labels-0", name: "Label 1", type: "text", thumbValue: "person 1" },
    ],
  };
  const photo = { id: "asset-photo", nodeType: "asset", imgUrl: "photo.jpg",
    inputs: [], outputs: [{ id: "out-photo", name: "Photo Output" }] };
  const spark = { id: "spark", nodeType: "action", resultText: "upstream reply",
    outputs: [{ id: "out-text", name: "Reply" }] };
  const card = { id: "gen", nodeType: "action",
    inputs: [{ id: "in-prompt" }, { id: "in-images" }] };

  it("upstreamImageUrl prefers pin thumbnails, then chains, then result images", () => {
    const app = { nodes: [sam, photo, card], edges: [] };
    assert.equal(upstreamImageUrl(app, { from: "sam", fromPort: "split-out-boxes-1" }), "crop2.jpg");
    assert.equal(upstreamImageUrl(app, { from: "sam", fromPort: "out-boxes" }), "overlay.jpg");
    assert.equal(upstreamImageUrl(app, { from: "asset-photo", fromPort: "out-photo" }), "photo.jpg");
    assert.equal(upstreamImageUrl(app, { from: "missing", fromPort: "out" }), null);
  });

  it("collectImageInputs gathers every wire on the images pin in edge order", () => {
    const app = { nodes: [sam, photo, spark, card], edges: [
      { to: "gen", toPort: "in-images", from: "sam", fromPort: "split-out-boxes-1" },
      { to: "gen", toPort: "in-images", from: "asset-photo", fromPort: "out-photo" },
      { to: "gen", toPort: "in-prompt", from: "spark", fromPort: "out-text" },
    ] };
    assert.deepEqual(collectImageInputs(app, card, "in-images").map(s => s.url),
      ["crop2.jpg", "photo.jpg"]);
    // Text-only upstream on another port resolves no image.
    assert.deepEqual(collectImageInputs(app, card, "in-prompt"), []);
  });

  it("resolvePromptText reads label values, then upstream reply text", () => {
    const app = { nodes: [sam, spark, card], edges: [
      { to: "gen", toPort: "in-prompt", from: "sam", fromPort: "split-out-labels-0" },
    ] };
    assert.equal(resolvePromptText(app, card, "in-prompt"), "person 1");
    const app2 = { nodes: [sam, spark, card], edges: [
      { to: "gen", toPort: "in-prompt", from: "spark", fromPort: "out-text" },
    ] };
    assert.equal(resolvePromptText(app2, card, "in-prompt"), "upstream reply");
    assert.equal(resolvePromptText({ nodes: [sam, card], edges: [] }, card, "in-prompt"), null);
  });
});

describe("fillSparkResultView / fillMuseResultView", () => {
  it("fills reply text, first image, badge and char count", () => {
    const node = { id: "s", outputs: [{ id: "out-text", badge: "—" }] };
    fillSparkResultView(node, {
      text: "A person in red.", imgUrl: "crop.jpg", badge: "1 img · muse-spark", demo: false,
    });
    assert.equal(node.resultText, "A person in red.");
    assert.equal(node.resultImage, "crop.jpg");
    assert.equal(node.resultImageBadge, "1 img · muse-spark");
    assert.equal(node.outputs[0].badge, "16 chars");
    assert.match(node.resultImageFallback, /^data:image\/svg/);
  });

  it("fills text-only replies without an image", () => {
    const node = { id: "s", outputs: [{ id: "out-text", badge: "—" }] };
    fillSparkResultView(node, { text: "Hi.", imgUrl: null, badge: "muse-spark", demo: false });
    assert.equal(node.resultImage, null);
    assert.equal(node.resultText, "Hi.");
  });

  it("labels the spark demo fallback as demo", () => {
    const node = { id: "s", outputs: [{ id: "out-text", badge: "—" }] };
    fillSparkResultView(node, {
      text: "Demo reply.", imgUrl: null, badge: "demo", demo: true,
    });
    assert.match(decodeURIComponent(node.resultImageFallback), /Demo/);
  });

  it("fills the generated image and dims badge", () => {
    const node = { id: "m", outputs: [{ id: "out-image", badge: "—" }] };
    fillMuseResultView(node, { imgUrl: "gen.png", badge: "1024x1024", demo: false, note: "" });
    assert.equal(node.resultImage, "gen.png");
    assert.equal(node.resultImageBadge, "1024x1024");
    assert.equal(node.outputs[0].badge, "1024x1024");
    assert.equal(node.resultText, "");
  });

  it("fills the muse demo with a flagged note", () => {
    const node = { id: "m", outputs: [{ id: "out-image", badge: "—" }] };
    fillMuseResultView(node, { imgUrl: null, badge: "demo", demo: true, note: "Demo image (offline)." });
    assert.equal(node.resultImage, null);
    assert.equal(node.resultText, "Demo image (offline).");
    assert.equal(node.outputs[0].badge, "demo");
  });
});

describe("sapiensInputEdge / fillSapiensResultView", () => {
  const card = { id: "sap", nodeType: "action" };

  it("prefers the Image pin, falls back to any incoming edge", () => {
    const app = { nodes: [card], edges: [
      { to: "sap", toPort: "in-kpts", from: "pose", fromPort: "out-kpts" },
      { to: "sap", toPort: "in-image", from: "photo", fromPort: "out-photo" },
    ] };
    assert.equal(sapiensInputEdge(app, card).toPort, "in-image");
    const app2 = { nodes: [card], edges: [
      { to: "sap", toPort: "in-kpts", from: "pose", fromPort: "out-kpts" },
    ] };
    assert.equal(sapiensInputEdge(app2, card).toPort, "in-kpts");
    assert.equal(sapiensInputEdge({ nodes: [card], edges: [] }, card), null);
  });

  it("fills pose badges and result view", () => {
    const node = { id: "p", outputs: [
      { id: "out-kpts", badge: "—" }, { id: "out-face", badge: "—" }, { id: "out-body", badge: "—" },
    ] };
    fillSapiensResultView(node, "pose", {
      text: "Sapiens2 pose: 308 keypoints.", imgUrl: "pose.jpg",
      badge: "sapiens2-pose", demo: false, stats: null,
    });
    assert.equal(node.resultText, "Sapiens2 pose: 308 keypoints.");
    assert.equal(node.resultImage, "pose.jpg");
    assert.deepEqual(node.outputs.map(o => o.badge), ["308 pts", "238 pts", "70 pts"]);
  });

  it("fills seg class counts and depth ranges from stats", () => {
    const seg = { id: "s", outputs: [
      { id: "out-parts", badge: "—" }, { id: "out-skin", badge: "—" }, { id: "out-clothing", badge: "—" },
    ] };
    fillSapiensResultView(seg, "seg", {
      text: "t", imgUrl: "seg.jpg", badge: "b", demo: false, stats: { classes: [0, 1, 5] },
    });
    assert.deepEqual(seg.outputs.map(o => o.badge), ["3 classes", "Mask", "Mask"]);
    const depth = { id: "d", outputs: [{ id: "out-depth", badge: "—" }] };
    fillSapiensResultView(depth, "depth", {
      text: "t", imgUrl: "depth.jpg", badge: "b", demo: false, stats: { depth_m: [0.5, 3.0] },
    });
    assert.equal(depth.outputs[0].badge, "0.5–3m");
    const depth2 = { id: "d2", outputs: [{ id: "out-depth", badge: "—" }] };
    fillSapiensResultView(depth2, "depth", {
      text: "t", imgUrl: null, badge: "demo", demo: true, stats: null,
    });
    assert.equal(depth2.outputs[0].badge, "Metres");
    assert.match(decodeURIComponent(depth2.resultImageFallback), /Demo/);
  });

  it("fills normal badges", () => {
    const node = { id: "n", outputs: [{ id: "out-normals", badge: "—" }] };
    fillSapiensResultView(node, "normal", {
      text: "t", imgUrl: "normal.jpg", badge: "b", demo: false, stats: {},
    });
    assert.equal(node.outputs[0].badge, "Normals");
  });
});

describe("produceOutputsForAction (mock/offline producer)", () => {
  const MOCK_MODELS = ["action-sapiens-pose", "action-sapiens-normal", "action-sapiens-depth",
    "action-sapiens-seg", "action-body-fit", "action-export"];
  const appFor = (templateKey) => {
    const action = { id: "a1", nodeType: "action", templateKey, x: 0, y: 0, outputs: [] };
    const app = { nodes: [action], edges: [], activeRunId: "r1",
      renderCanvas() {}, fitView() {} };
    produceOutputsForAction(app, action);
    return { app, action };
  };

  it("wires the export edge from the action's real out-glb port", () => {
    const { app } = appFor("action-export");
    assert.equal(app.edges.length, 1);
    assert.equal(app.edges[0].fromPort, "out-glb");
  });

  it("keeps every mock fallback offline — data-URI placeholders, never remote http", () => {
    for (const key of MOCK_MODELS) {
      const { app } = appFor(key);
      const produced = app.nodes.filter(n => n.id !== "a1");
      assert.equal(produced.length, 1, `${key} produces one asset`);
      assert.match(produced[0].imgFallback, /^data:image\/svg/, `${key} fallback is offline`);
    }
  });

  it("reads the depth mock from the depth artifact, not normal", () => {
    const { app } = appFor("action-sapiens-depth");
    const produced = app.nodes.find(n => n.id !== "a1");
    assert.match(produced.imgUrl, /\/artifact\/depth$/);
  });
});

describe("runCard completion promise", () => {
  it("resolves only after the mock run completes", async () => {
    const doc = globalThis.document;
    globalThis.document = { getElementById: () => null };
    try {
      const node = { id: "a1", nodeType: "action", templateKey: "action-unknown-xyz",
        x: 0, y: 0, outputs: [] };
      const app = { nodes: [node], edges: [], activeRunId: "r1",
        renderCanvas() {}, fitView() {} };
      await runCard(app, "a1");
      assert.equal(node.status, "Passed");
      assert.match((node.runLog || []).map(e => e.msg).join("\n"), /mock outputs/);
    } finally {
      if (doc === undefined) delete globalThis.document;
      else globalThis.document = doc;
    }
  });
});

describe("failUnreachable", () => {
  const liveNode = () => ({ id: "a", status: "Running...", runLog: [],
    apiCall: { at: 1, hop1: { method: "POST", url: "/api/debug/x", status: null, error: null } },
    outputs: [], resultImage: "stale.jpg" });
  const btn = () => ({ classList: { remove() {} }, textContent: "…" });

  it("fails loud with trace + message when the backend was reachable", () => {
    const node = liveNode();
    const b = btn();
    assert.equal(failUnreachable({ backendOnline: true }, node,
      { ok: false, status: 0, error: "Failed to fetch" }, b), true);
    assert.equal(node.status, "Failed");
    assert.match(node.runError, /Backend unreachable/);
    assert.match(node.runError, /Failed to fetch/);
    assert.equal(node.resultImage, null);
    assert.equal(node.apiCall.hop1.status, 0);
    assert.equal(node.apiCall.hop1.error, "Failed to fetch");
    assert.match(node.runLog.map(e => e.msg).join("\n"), /Failed/);
    assert.equal(b.textContent, "✓");
  });

  it("stays quiet (demo path) when the backend was never reached", () => {
    for (const app of [{ backendOnline: false }, {}]) {
      const node = liveNode();
      assert.equal(failUnreachable(app, node, { ok: false, status: 0 }, null), false);
      assert.equal(node.status, "Running...");
      assert.equal(node.resultImage, "stale.jpg");
      assert.equal(node.runError, undefined);
    }
  });
});

describe("executeDescribe input failures", () => {
  const withDoc = async (fn) => {
    const doc = globalThis.document;
    globalThis.document = { getElementById: () => null };
    try {
      await fn();
    } finally {
      if (doc === undefined) delete globalThis.document;
      else globalThis.document = doc;
    }
  };

  it("directs mask wires to the Apply Mask card", async () => {
    await withDoc(async () => {
      const sam = { id: "sam", nodeType: "action", templateKey: "action-meta-sam",
        outputs: [{ id: "split-out-masks-0", type: "mask",
          ref: { debug_id: "d0", mask_id: 2 } }] };
      const node = { id: "d", nodeType: "action", templateKey: "action-describe",
        status: "Running...", runLog: [], outputs: [] };
      const app = { nodes: [sam, node],
        edges: [{ from: "sam", fromPort: "split-out-masks-0", to: "d", toPort: "in-image" }],
        renderCanvas() {} };
      await executeDescribe(app, node, null);
      assert.equal(node.status, "Failed");
      assert.match(node.runError, /Apply Mask/);
      assert.doesNotMatch(node.runError, /Describe-Mask/);
    });
  });

  it("keeps the photo remedy for genuinely imageless input", async () => {
    await withDoc(async () => {
      const node = { id: "d", nodeType: "action", templateKey: "action-describe",
        status: "Running...", runLog: [], outputs: [] };
      const app = { nodes: [node], edges: [], renderCanvas() {} };
      await executeDescribe(app, node, null);
      assert.equal(node.status, "Failed");
      assert.match(node.runError, /No input image/);
      assert.doesNotMatch(node.runError, /Describe-Mask/);
    });
  });
});

describe("fillApplyMaskResultView", () => {
  it("fills the region image, badge and out-image pin", () => {
    const node = { id: "m", outputs: [{ id: "out-image", badge: "—" }] };
    fillApplyMaskResultView(node, { imgUrl: "/api/x/mask_0_crop.jpg", badge: "120×80",
      demo: false, text: "mask M0 applied" });
    assert.equal(node.resultImage, "/api/x/mask_0_crop.jpg");
    assert.equal(node.resultImageBadge, "120×80");
    assert.equal(node.resultText, "mask M0 applied");
    assert.equal(node.outputs[0].badge, "120×80");
  });

  it("flags demo fills honestly", () => {
    const node = { id: "m", outputs: [{ id: "out-image", badge: "—" }] };
    fillApplyMaskResultView(node, { imgUrl: null, badge: "demo", demo: true, text: "Demo region" });
    assert.equal(node.resultImage, null);
    assert.equal(node.outputs[0].badge, "Demo");
  });
});

describe("executeApplyMask", () => {
  const sam = () => ({ id: "sam", nodeType: "action", templateKey: "action-meta-sam",
    outputs: [{ id: "split-out-masks-0", type: "mask", ref: { debug_id: "d0", mask_id: 2 } }] });
  const card = () => ({ id: "m", nodeType: "action", templateKey: "action-apply-mask",
    status: "Running...", runLog: [], outputs: [{ id: "out-image", badge: "—" }] });
  const wired = (node, samNode) => ({ nodes: [samNode, node],
    edges: [{ from: "sam", fromPort: "split-out-masks-0", to: "m", toPort: "in-mask" }],
    backendOnline: false, renderCanvas() {} });
  const withFetch = async (impl, fn) => {
    const real = globalThis.fetch;
    globalThis.fetch = impl;
    try {
      await fn();
    } finally {
      globalThis.fetch = real;
    }
  };

  it("fails fast without a mask edge or a backend ref", async () => {
    const node = card();
    await executeApplyMask({ nodes: [node], edges: [], renderCanvas() {} }, node, null);
    assert.equal(node.status, "Failed");
    assert.match(node.runError, /No mask/);
    const samNoRef = sam();
    samNoRef.outputs[0].ref = null;
    const node2 = card();
    await executeApplyMask(wired(node2, samNoRef), node2, null);
    assert.equal(node2.status, "Failed");
    assert.match(node2.runError, /No backend mask ref/);
  });

  it("passes live with the server crop URL", async () => {
    await withFetch(async () => ({ ok: true, status: 200, json: async () => ({
      chat_id: "flow-session", debug_id: "d0", crop: "mask_2_crop.jpg",
      width: 120, height: 80, assistant_text: "mask M2 applied",
    }) }), async () => {
      const node = card();
      await executeApplyMask(wired(node, sam()), node, null);
      assert.equal(node.status, "Passed");
      assert.equal(node.resultImage, "/api/chats/flow-session/file/debug/d0/mask_2_crop.jpg");
      assert.equal(node.resultImageBadge, "120×80");
      assert.equal(node.latency, "local");
    });
  });

  it("demos honestly when unreachable and unknown to health", async () => {
    await withFetch(async () => { throw new Error("Failed to fetch"); }, async () => {
      const node = card();
      await executeApplyMask(wired(node, sam()), node, null);
      assert.equal(node.status, "Demo (Offline)");
      assert.match(node.resultText, /Demo region/);
    });
  });
});

describe("upstreamMeshRef", () => {
  const app = { nodes: [
    { id: "exp", glbUrl: "/api/x/a.glb", meshObjUrl: "/api/x/a.obj" },
    { id: "fit", meshObjUrl: "/api/x/b.obj" },
    { id: "img", resultImage: "/api/x/c.jpg" },
  ] };

  it("prefers GLB over OBJ, falls back to OBJ, null otherwise", () => {
    assert.deepEqual(upstreamMeshRef(app, { from: "exp" }), { url: "/api/x/a.glb", format: "glb" });
    assert.deepEqual(upstreamMeshRef(app, { from: "fit" }), { url: "/api/x/b.obj", format: "obj" });
    assert.equal(upstreamMeshRef(app, { from: "img" }), null);
    assert.equal(upstreamMeshRef(app, { from: "ghost" }), null);
  });
});

describe("fillMeshRenderResultView", () => {
  const card = (mode) => ({ id: "m", outputs: [{ id:
    mode === "depth" ? "out-depth" : mode === "normal" ? "out-normal" : "out-image", badge: "—" }] });

  it("badges the mode pin and records honest dims", () => {
    for (const mode of ["shaded", "depth", "normal"]) {
      const node = card(mode);
      fillMeshRenderResultView(node, mode, { dataUrl: "data:image/png,xx", width: 512, height: 512, backend: "open3d" });
      assert.equal(node.resultImage, "data:image/png,xx");
      assert.equal(node.resultImageBadge, "512x512");
      assert.equal(node.outputs[0].badge, "512x512");
      assert.match(node.resultText, /Server .* \(open3d\)/);
    }
    const node = card("depth");
    fillMeshRenderResultView(node, "depth", { dataUrl: "d", width: 512, height: 512 });
    assert.match(node.resultText, /depth map/);
  });

  it("throws on a pixel-less still so the run fails loud, never Passed-empty", () => {
    assert.throws(() => fillMeshRenderResultView(card("shaded"), "shaded",
      { dataUrl: null, width: 512, height: 512 }), /no pixels/);
    assert.throws(() => fillMeshRenderResultView(card("shaded"), "shaded",
      { width: 512, height: 512 }), /no pixels/);
  });
});

describe("executeMeshRender", () => {
  const card = () => ({ id: "m", nodeType: "action", templateKey: "action-mesh-depth",
    status: "Running...", runLog: [], outputs: [{ id: "out-depth", badge: "—" }] });
  const wired = (node, src) => ({ nodes: [src, node],
    edges: [{ from: src.id, fromPort: "out-mesh", to: "m", toPort: "in-mesh" }],
    renderCanvas() {} });

  it("fails fast without a mesh edge or a mesh ref", async () => {
    const node = card();
    await executeMeshRender({ nodes: [node], edges: [], renderCanvas() {} }, node, null, "depth");
    assert.equal(node.status, "Failed");
    assert.match(node.runError, /No mesh/);
    const node2 = card();
    await executeMeshRender(wired(node2, { id: "img", resultImage: "c.jpg" }), node2, null, "normal");
    assert.equal(node2.status, "Failed");
    assert.match(node2.runError, /No mesh/);
  });

  const withFetch = async (impl, fn) => {
    const real = globalThis.fetch;
    globalThis.fetch = impl;
    try {
      await fn();
    } finally {
      globalThis.fetch = real;
    }
  };

  it("passes live with the server still and backend note", async () => {
    let seen = null;
    await withFetch(async (url, opts) => {
      seen = JSON.parse(opts.body);
      return { ok: true, status: 200, json: async () => ({
        chat_id: "flow-session", debug_id: "d1", assistant_text: "mesh depth still",
        files: ["mesh_still.png"], image_url: "/mr.png", width: 512, height: 512,
        render_backend: "open3d", stats: {},
      }) };
    }, async () => {
      const node = card();
      await executeMeshRender(wired(node, { id: "exp", glbUrl: "/api/x/a.glb" }), node, null, "depth");
      assert.equal(node.status, "Passed");
      assert.equal(node.resultImage, "/mr.png");
      assert.match(node.runNote, /server-side \(open3d\)/);
      assert.equal(node.apiCall.hop1.url, "/api/debug/mesh-render");
      assert.equal(seen.mode, "depth");
      assert.equal(seen.mesh_url, "/api/x/a.glb");
    });
  });

  it("fails loud (never demo) when unreachable or the server errors", async () => {
    await withFetch(async () => { throw new Error("Failed to fetch"); }, async () => {
      const node = card();
      await executeMeshRender(wired(node, { id: "exp", glbUrl: "/api/x/a.glb" }), node, null, "depth");
      assert.equal(node.status, "Failed");
      assert.ok(node.runLog.some(e => e.level === "error"));
    });
    await withFetch(async () => ({ ok: false, status: 500,
      json: async () => ({ error: "mesh unreadable", remedy: "wire a mesh pin" }) }), async () => {
      const node = card();
      await executeMeshRender(wired(node, { id: "exp", glbUrl: "/api/x/a.glb" }), node, null, "depth");
      assert.equal(node.status, "Failed");
      assert.match(node.runError, /mesh unreadable — wire a mesh pin/);
    });
  });
});

describe("meshInputEdge", () => {
  it("prefers the Mesh pin, else any incoming edge", () => {
    const app = { nodes: [{ id: "m" }], edges: [
      { from: "a", to: "m", toPort: "in-mesh" },
      { from: "b", to: "m", toPort: "in-view" },
    ] };
    assert.equal(meshInputEdge(app, { id: "m" }).from, "a");
    const app2 = { nodes: [{ id: "m" }], edges: [{ from: "b", to: "m", toPort: "in-view" }] };
    assert.equal(meshInputEdge(app2, { id: "m" }).from, "b");
    assert.equal(meshInputEdge({ nodes: [], edges: [] }, { id: "m" }), null);
  });
});

describe("meshInputDiagnosis", () => {
  const app = (src) => ({ nodes: src ? [src] : [] });
  const edge = { from: "up", to: "m", toPort: "in-mesh" };

  it("names the upstream card and reads its state", () => {
    assert.match(meshInputDiagnosis(app(null), { id: "m" }, null), /wire a fitted mesh/);
    assert.match(meshInputDiagnosis(app(null), { id: "m" }, edge), /card is gone/);
    assert.match(meshInputDiagnosis(
      app({ id: "up", title: "Action: Exporter", status: "Demo (Offline)" }), { id: "m" }, edge),
      /demo runs produce no mesh/);
    assert.match(meshInputDiagnosis(
      app({ id: "up", title: "Action: Exporter", status: "Running..." }), { id: "m" }, edge),
      /not Passed/);
    assert.match(meshInputDiagnosis(
      app({ id: "up", title: "Action: Exporter", status: "Passed" }), { id: "m" }, edge),
      /restart chat_server\.py/);
  });
});

describe("executeMeshMultiview", () => {
  const card = (param = "4") => ({ id: "m", nodeType: "action", templateKey: "action-mesh-multiview",
    status: "Running...", runLog: [], param, outputs: [{ id: "out-image", badge: "—" }] });
  const wired = (node, src) => ({ nodes: [src, node],
    edges: [{ from: src.id, fromPort: "out-mesh", to: "m", toPort: "in-mesh" }],
    renderCanvas() {} });

  it("fails fast without a mesh and on bad view counts", async () => {
    const node = card();
    await executeMeshMultiview({ nodes: [node], edges: [], renderCanvas() {} }, node, null);
    assert.equal(node.status, "Failed");
    assert.match(node.runError, /wire a fitted mesh/);
    const node2 = card("7");
    await executeMeshMultiview(wired(node2, { id: "exp", glbUrl: "/a.glb" }), node2, null);
    assert.equal(node2.status, "Failed");
    assert.match(node2.runError, /2, 4, 6, 8/);
  });

  const withFetch = async (impl, fn) => {
    const real = globalThis.fetch;
    globalThis.fetch = impl;
    try {
      await fn();
    } finally {
      globalThis.fetch = real;
    }
  };

  it("passes live with the server sheet for the chosen count", async () => {
    let seen = null;
    await withFetch(async (url, opts) => {
      seen = JSON.parse(opts.body);
      return { ok: true, status: 200, json: async () => ({
        chat_id: "flow-session", debug_id: "d1", assistant_text: "mesh 6-view sheet",
        files: ["mesh_multiview.jpg"], image_url: "/mv.jpg", width: 1152, height: 768,
        render_backend: "numpy", stats: {},
      }) };
    }, async () => {
      const node = card("6");
      await executeMeshMultiview(wired(node, { id: "exp", glbUrl: "/a.glb" }), node, null);
      assert.equal(node.status, "Passed");
      assert.equal(node.resultImage, "/mv.jpg");
      assert.equal(node.outputs[0].badge, "1152x768");
      assert.match(node.runNote, /server-side \(numpy\)/);
      assert.equal(seen.mode, "multiview");
      assert.equal(seen.count, 6);
      assert.equal(seen.framing, "body");
    });
  });

  it("sends the chosen framing and names it in the run note", async () => {
    let seen = null;
    await withFetch(async (url, opts) => {
      seen = JSON.parse(opts.body);
      return { ok: true, status: 200, json: async () => ({
        chat_id: "flow-session", debug_id: "d1", assistant_text: "mesh 4-view face sheet",
        files: ["mesh_multiview.jpg"], image_url: "/mv.jpg", width: 768, height: 768,
        render_backend: "open3d", stats: {},
      }) };
    }, async () => {
      const node = card("4");
      node.framing = "face";
      await executeMeshMultiview(wired(node, { id: "exp", glbUrl: "/a.glb" }), node, null);
      assert.equal(node.status, "Passed");
      assert.equal(seen.framing, "face");
      assert.match(node.runNote, /server-side \(open3d, face\)/);
    });
  });

  it("fails loud (never demo) when unreachable", async () => {
    await withFetch(async () => { throw new Error("Failed to fetch"); }, async () => {
      const node = card("6");
      await executeMeshMultiview(wired(node, { id: "exp", glbUrl: "/a.glb" }), node, null);
      assert.equal(node.status, "Failed");
      assert.ok(node.runLog.some(e => e.level === "error"));
    });
  });

  it("fills the sheet text for multiview mode", () => {
    const node = { id: "m", outputs: [{ id: "out-image", badge: "—" }] };
    fillMeshRenderResultView(node, "multiview", { dataUrl: "d", width: 1536, height: 768 });
    assert.match(node.resultText, /view sheet/);
    assert.equal(node.outputs[0].badge, "1536x768");
  });
});

describe("fitDims", () => {
  it("shrinks past the cap preserving aspect, else no-ops", () => {
    assert.deepEqual(fitDims(2000, 1000, 1024), { w: 1024, h: 512 });
    assert.deepEqual(fitDims(800, 1200, 1024), { w: 683, h: 1024 });
    assert.deepEqual(fitDims(512, 512, 1024), { w: 512, h: 512 });
  });
});

describe("absDiffStats", () => {
  it("amplifies abs diffs and reports mean/peak honestly", () => {
    const a = new Uint8ClampedArray([10, 20, 30, 255, 0, 0, 0, 255]);
    const b = new Uint8ClampedArray([0, 0, 0, 255, 0, 0, 0, 255]);
    const { data, mean01, maxV } = absDiffStats(a, b, 2);
    assert.deepEqual([...data], [20, 40, 60, 255, 0, 0, 0, 255]);
    assert.equal(maxV, 30);
    assert.ok(Math.abs(mean01 - (60 / (2 * 3 * 255))) < 1e-12);
  });

  it("clamps amplified peaks at 255 and throws on mismatch", () => {
    const a = new Uint8ClampedArray([200, 0, 0, 255]);
    const b = new Uint8ClampedArray([0, 0, 0, 255]);
    assert.equal(absDiffStats(a, b, 2).data[0], 255);
    assert.throws(() => absDiffStats(new Uint8ClampedArray(4), new Uint8ClampedArray(8)),
      /mismatched/);
  });
});

describe("executeImageDiff", () => {
  const card = () => ({ id: "m", nodeType: "action", templateKey: "action-image-diff",
    status: "Running...", runLog: [], outputs: [{ id: "out-image", badge: "—" }] });

  it("fails fast unless both A and B resolve to images", async () => {
    const node = card();
    await executeImageDiff({ nodes: [node], edges: [], renderCanvas() {} }, node, null);
    assert.equal(node.status, "Failed");
    assert.match(node.runError, /two images/);
    const half = card();
    const app = { nodes: [{ id: "v", nodeType: "asset", imgUrl: "a.jpg" }, half],
      edges: [{ from: "v", to: "m", toPort: "in-a" }], renderCanvas() {} };
    await executeImageDiff(app, half, null);
    assert.equal(half.status, "Failed");
    assert.match(half.runError, /two images/);
  });

  it("fails loud (never demo) when decode cannot run headless", async () => {
    const node = card();
    const app = { nodes: [
      { id: "a", nodeType: "asset", imgUrl: "a.jpg" },
      { id: "b", nodeType: "asset", imgUrl: "b.jpg" }, node],
    edges: [{ from: "a", to: "m", toPort: "in-a" }, { from: "b", to: "m", toPort: "in-b" }],
    renderCanvas() {} };
    await executeImageDiff(app, node, null);
    assert.equal(node.status, "Failed");
    assert.match(node.runError, /browser page/);
  });

  it("fills Δ stats and badges", () => {
    const node = { id: "m", outputs: [{ id: "out-image", badge: "—" }] };
    fillImageDiffResultView(node, { imgUrl: "d", width: 640, height: 480,
      meanPct: "4.2", maxV: 210, note: "B resampled" });
    assert.match(node.resultText, /Mean Δ 4\.2% · peak 210\/255/);
    assert.equal(node.outputs[0].badge, "640x480");
  });
});

describe("upstreamFitCamera", () => {
  it("reads the fit ref off body-fit and avatar cards", () => {
    const fit = { debug_id: "d", chat_id: "c" };
    const app = { nodes: [
      { id: "fit", fitCamera: fit },
      { id: "exp", glbUrl: "g.glb" },
    ] };
    assert.deepEqual(upstreamFitCamera(app, { from: "fit" }), fit);
    assert.equal(upstreamFitCamera(app, { from: "exp" }), null);
    assert.equal(upstreamFitCamera(app, { from: "ghost" }), null);
  });
});

describe("upstreamFitDiagnosis", () => {
  const app = (src) => ({ nodes: src ? [src] : [] });
  const edge = { from: "up", to: "m", toPort: "in-mesh" };

  it("names the upstream card and reads its state", () => {
    assert.match(upstreamFitDiagnosis(app(null), { id: "m" }, null), /wire the body-fit/);
    assert.match(upstreamFitDiagnosis(app(null), { id: "m" }, edge), /card is gone/);
    assert.match(upstreamFitDiagnosis(
      app({ id: "up", title: "Action: Fit", status: "Demo (Offline)" }), { id: "m" }, edge),
      /demo data/);
    assert.match(upstreamFitDiagnosis(
      app({ id: "up", title: "Action: Exporter", status: "Passed", glbUrl: "g" }),
      { id: "m" }, edge), /exporter drops the fit/);
    assert.match(upstreamFitDiagnosis(
      app({ id: "up", title: "Action: Fit", status: "Running..." }), { id: "m" }, edge),
      /not Passed/);
    assert.match(upstreamFitDiagnosis(
      app({ id: "up", title: "Action: Fit", status: "Passed" }), { id: "m" }, edge),
      /restart chat_server\.py/);
  });
});

describe("fillFitOverlayResultView", () => {
  const card = () => ({ id: "m", outputs: [
    { id: "out-overlay", badge: "—" }, { id: "out-photo", badge: "—" },
    { id: "out-render", badge: "—" },
  ] });

  it("badges all three frame pins with their own thumbs", () => {
    const node = card();
    fillFitOverlayResultView(node, { overlayUrl: "o.jpg", photoUrl: "p.jpg", renderUrl: "r.jpg",
      badge: "640x480", demo: false, text: "blend", width: 640, height: 480 });
    assert.equal(node.resultImage, "o.jpg");
    assert.deepEqual(node.outputs.map(o => o.badge), ["640x480", "640x480", "640x480"]);
    assert.deepEqual(node.outputs.map(o => o.thumb), ["o.jpg", "p.jpg", "r.jpg"]);
  });

  it("leaves missing frames bare on demo", () => {
    const node = card();
    fillFitOverlayResultView(node, { overlayUrl: "p.jpg", photoUrl: null, renderUrl: null,
      badge: "demo", demo: true, text: "demo" });
    assert.equal(node.resultImage, "p.jpg");
    assert.equal(node.outputs[0].thumb, "p.jpg");
    assert.equal(node.outputs[1].badge, "—");
    assert.equal(node.outputs[1].thumb, null);
    assert.equal(node.outputs[2].badge, "—");
    assert.equal(node.outputs[2].thumb, null);
  });
});

describe("executeFitOverlay", () => {
  const card = () => ({ id: "m", nodeType: "action", templateKey: "action-fit-overlay",
    status: "Running...", runLog: [], outputs: [
      { id: "out-overlay", badge: "—" }, { id: "out-photo", badge: "—" },
      { id: "out-render", badge: "—" },
    ] });
  const wired = (node, photo, fit) => ({ nodes: [photo, fit, node],
    edges: [
      { from: photo.id, fromPort: "out-photo", to: "m", toPort: "in-photo" },
      { from: fit.id, fromPort: "out-mesh", to: "m", toPort: "in-mesh" },
    ],
    backendOnline: false, renderCanvas() {} });
  const photo = () => ({ id: "p", nodeType: "asset", imgUrl: "/api/runs/r/artifact/source" });
  const fit = () => ({ id: "f", nodeType: "action", status: "Passed",
    fitCamera: { debug_id: "d0", chat_id: "flow-session" } });
  const withFetch = async (impl, fn) => {
    const real = globalThis.fetch;
    globalThis.fetch = impl;
    try {
      await fn();
    } finally {
      globalThis.fetch = real;
    }
  };

  it("fails fast without a photo or without a fit", async () => {
    const node = card();
    await executeFitOverlay({ nodes: [node], edges: [], renderCanvas() {} }, node, null);
    assert.equal(node.status, "Failed");
    assert.match(node.runError, /No photo/);
    const node2 = card();
    await executeFitOverlay(wired(node2, photo(), { id: "e", status: "Passed", glbUrl: "g" }),
      node2, null);
    assert.equal(node2.status, "Failed");
    assert.match(node2.runError, /exporter drops the fit/);
  });

  it("passes live with three separate frame URLs", async () => {
    await withFetch(async () => ({ ok: true, status: 200, json: async () => ({
      chat_id: "flow-session", debug_id: "d1", assistant_text: "fit overlay",
      files: ["fit_photo.jpg", "fit_render.jpg", "fit_overlay.jpg"],
      photo_url: "/p.jpg", render_url: "/r.jpg", overlay_url: "/o.jpg",
      width: 640, height: 480, stats: {},
    }) }), async () => {
      const node = card();
      await executeFitOverlay(wired(node, photo(), fit()), node, null);
      assert.equal(node.status, "Passed");
      assert.equal(node.resultImage, "/o.jpg");
      assert.deepEqual(node.outputs.map(o => o.thumb), ["/o.jpg", "/p.jpg", "/r.jpg"]);
      assert.match(node.resultText, /fit overlay/);
    });
  });

  it("demos honestly when unreachable and unknown to health", async () => {
    await withFetch(async () => { throw new Error("Failed to fetch"); }, async () => {
      const node = card();
      await executeFitOverlay(wired(node, photo(), fit()), node, null);
      assert.equal(node.status, "Demo (Offline)");
      assert.equal(node.resultImage, "/api/runs/r/artifact/source");
    });
  });

  it("fails loud with the backend remedy on HTTP errors", async () => {
    await withFetch(async () => ({ ok: false, status: 404,
      json: async () => ({ error: "unknown fit debug_id", remedy: "run body-fit" }) }), async () => {
      const node = card();
      await executeFitOverlay(wired(node, photo(), fit()), node, null);
      assert.equal(node.status, "Failed");
      assert.match(node.runError, /unknown fit debug_id — run body-fit/);
    });
  });
});

describe("buildSegClassNode", () => {
  const action = { id: "seg", x: 10, y: 20 };
  const row = { id: "1", name: "T-Shirt", count: 1500, pct: 30, color: [255, 0, 0] };

  it("builds an honest wireable card stacked below the action", () => {
    const n = buildSegClassNode(action, row, "data:image/png,xx", { w: 100, h: 50 }, 2);
    assert.equal(n.nodeType, "asset");
    assert.equal(n.title, "Asset: Seg · T-Shirt");
    assert.equal(n.subtitle, "class_1_t-shirt.png");
    assert.equal(n.imgUrl, "data:image/png,xx");
    assert.equal(n.imgBadge, "1,500 px · 30.0%");
    assert.equal(n.status, "Produced");
    assert.equal(n.segClassId, "1");
    assert.deepEqual([n.x, n.y], [490, 620]);
    assert.equal(n.inputs[0].id, "in-mask");
    assert.equal(n.outputs[0].type, "image");
    assert.ok(!("templateKey" in n), "spawned cards die via the re-run cascade");
  });

  it("falls back to placeholder art without a thumbnail", () => {
    const n = buildSegClassNode(action, row, null, { w: 100, h: 50 }, 0);
    assert.equal(n.imgUrl, null);
    assert.match(n.imgFallback, /T-Shirt/);
  });
});

describe("spawnSegClassCards", () => {
  const legend = (n) => {
    const counts = { 0: 3000 }, names = { 0: "background" }, palette = { 0: [0, 0, 0] };
    for (let i = 1; i <= n; i++) {
      counts[i] = 1000 - i * 10;
      names[i] = `part${i}`;
      palette[i] = [i, i, i];
    }
    return { width: 100, height: 50, counts, names, palette };
  };
  const appFor = (action) => ({ nodes: [action], edges: [], renderCanvas() {} });

  it("spawns one wired card per foreground class (headless: placeholder art)", async () => {
    const action = { id: "seg", x: 0, y: 0, runLog: [], runNote: null };
    const app = appFor(action);
    const ids = await spawnSegClassCards(app, action, legend(2), "/api/x/labels.png");
    assert.deepEqual(ids.length, 2);
    assert.equal(app.nodes.length, 3);
    assert.equal(app.edges.length, 2);
    assert.ok(app.edges.every(e => e.from === "seg" && e.fromPort === "out-parts" && e.toPort === "in-mask"));
    assert.deepEqual(app.edges.map(e => e.label), ["part1", "part2"]);
    // Headless: no labels map, so the cards carry an honest counts-only note.
    assert.match(action.runNote, /labels map unreachable/);
    assert.ok(action.runLog.some(e => /Spawned 2 class cards/.test(e.msg)));
  });

  it("caps the flood and notes the remainder", async () => {
    const action = { id: "seg", x: 0, y: 0, runLog: [], runNote: null };
    const app = appFor(action);
    const ids = await spawnSegClassCards(app, action, legend(15), "/api/x/labels.png");
    assert.equal(ids.length, 12);
    assert.equal(app.edges.length, 12);
    assert.match(action.runNote, /\+3 more classes/);
  });

  it("spawns nothing and says so when no foreground class was detected", async () => {
    const action = { id: "seg", x: 0, y: 0, runLog: [], runNote: null };
    const app = appFor(action);
    const ids = await spawnSegClassCards(app, action, legend(0), "/api/x/labels.png");
    assert.deepEqual(ids, []);
    assert.equal(app.nodes.length, 1);
    assert.match(action.runNote, /No foreground classes/);
  });
});

describe("validSegLabels", () => {
  it("accepts only positive-int dims with a matching buffer", () => {
    const good = { pixels: new Uint8Array(6), w: 3, h: 2 };
    assert.equal(validSegLabels(good), good);
    assert.equal(validSegLabels(null), null);
    assert.equal(validSegLabels({ pixels: new Uint8Array(6), w: 0, h: 2 }), null);
    assert.equal(validSegLabels({ pixels: new Uint8Array(6), w: 3, h: 0 }), null);
    assert.equal(validSegLabels({ pixels: new Uint8Array(6), w: NaN, h: 2 }), null);
    assert.equal(validSegLabels({ pixels: new Uint8Array(5), w: 3, h: 2 }), null);
    assert.equal(validSegLabels({ w: 3, h: 2 }), null);
  });
});

describe("segClassThumbUrl", () => {
  it("never throws headless — null instead of a crash", () => {
    assert.equal(segClassThumbUrl(new Uint8Array(4), 2, 2, "1", [255, 0, 0]), null);
    assert.equal(segClassThumbUrl(null, 0, 0, "1", null), null);
  });
});
