// Custom Sapiens displays: pose SVG tiers, seg legend/isolate math, fills.
// Run from the flow-viewer folder:  node tests/overlays.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  poseDotJobs, poseLinkJobs, renderPoseSvgInner, poseHoverText,
  segLegendRows, segSpawnRows, renderSegLegendHtml, segClassAt, isolateSegPixels,
  POSE_BODY_COLOR, POSE_FACE_COLOR,
} from "../js/overlays.js";
import { fillSapiensResultView } from "../js/execution.js";
import { renderResultPreviewHtml } from "../js/node-renderer.js";

const TOPO = {
  names: ["nose", "left_eye"],
  links: [[0, 1, [51, 153, 255]]],
};
const KPTS = [[10, 20, 0.9], [30, 40, 0.1], [null, null, null]];

describe("poseDotJobs", () => {
  it("keeps plotted points with group + score, drops null coords", () => {
    assert.deepEqual(poseDotJobs(KPTS), [
      { i: 0, x: 10, y: 20, s: 0.9, body: true },
      { i: 1, x: 30, y: 40, s: 0.1, body: true },
    ]);
  });

  it("marks indices >= 70 as face and nulls bad scores", () => {
    const kpts = [[0, 0, 0.9], ...Array(69).fill([null, null, null]), [5, 5, NaN]];
    const dots = poseDotJobs(kpts);
    assert.equal(dots.length, 2);
    assert.equal(dots[1].body, false);
    assert.equal(dots[1].s, null);
  });

  it("tolerates missing input", () => {
    assert.deepEqual(poseDotJobs(null), []);
    assert.deepEqual(poseDotJobs(undefined), []);
  });
});

describe("poseLinkJobs", () => {
  it("resolves links to image pixels with css colors", () => {
    assert.deepEqual(poseLinkJobs(KPTS, TOPO),
      [{ x1: 10, y1: 20, x2: 30, y2: 40, color: "rgb(51,153,255)" }]);
  });

  it("drops links with missing endpoints and empty topologies", () => {
    assert.deepEqual(poseLinkJobs(KPTS, null), []);
    assert.deepEqual(poseLinkJobs(KPTS, {}), []);
    assert.deepEqual(poseLinkJobs([[null, null, null], [1, 1, 0.5]], TOPO), []);
  });
});

describe("renderPoseSvgInner", () => {
  const node = { resultKeypoints: KPTS };

  it("renders dots only without a topology (offline tier)", () => {
    const s = renderPoseSvgInner(node, null);
    assert.equal((s.match(/<circle/g) || []).length, 2);
    assert.ok(!s.includes("<line"));
    assert.ok(s.includes(`fill="${POSE_BODY_COLOR}"`));
    assert.ok(!s.includes("data-kname="));
  });

  it("adds skeleton links + names with a topology", () => {
    const s = renderPoseSvgInner(node, TOPO);
    assert.equal((s.match(/<line/g) || []).length, 1);
    assert.ok(s.includes('data-kname="nose"'));
    assert.ok(s.includes('data-kscore="0.9"'));
  });

  it("fades low-score dots and colors face dots magenta", () => {
    const s = renderPoseSvgInner(node, TOPO);
    assert.equal((s.match(/opacity="0\.35"/g) || []).length, 1);
    const face = { resultKeypoints: [...Array(70).fill([null, null, null]), [5, 5, 0.9]] };
    assert.ok(renderPoseSvgInner(face, null).includes(`fill="${POSE_FACE_COLOR}"`));
  });

  it("escapes topology names and skips empty keypoints", () => {
    const evil = { resultKeypoints: [[1, 2, 0.5]] };
    const s = renderPoseSvgInner(evil, { names: ['"><script>alert(1)</script>'], links: [] });
    assert.ok(!s.includes("<script>"));
    assert.equal(renderPoseSvgInner({ resultKeypoints: null }, TOPO), "");
    assert.equal(renderPoseSvgInner({}, TOPO), "");
  });
});

describe("poseHoverText", () => {
  it("formats name + score, with fallbacks", () => {
    assert.equal(poseHoverText("left_elbow", 7, 0.87), "left_elbow · 0.87");
    assert.equal(poseHoverText(null, 12, null), "#12 · no score");
    assert.equal(poseHoverText("", 3, 0.5), "#3 · 0.5");
  });
});

const LEGEND = {
  width: 10, height: 10,
  counts: { "0": 60, "3": 30, "4": 10 },
  names: { "0": "Background", "3": "Face_Neck", "4": "Hair" },
  palette: { "0": [50, 50, 50], "3": [128, 200, 255], "4": [255, 0, 109] },
};

describe("segLegendRows", () => {
  it("sorts by count desc with pct of frame", () => {
    assert.deepEqual(segLegendRows(LEGEND), [
      { id: "0", count: 60, name: "Background", color: [50, 50, 50], pct: 60 },
      { id: "3", count: 30, name: "Face_Neck", color: [128, 200, 255], pct: 30 },
      { id: "4", count: 10, name: "Hair", color: [255, 0, 109], pct: 10 },
    ]);
  });

  it("falls back on thin legends and empty input", () => {
    const rows = segLegendRows({ width: 2, height: 2, counts: { "5": 4 } });
    assert.deepEqual(rows, [{ id: "5", count: 4, name: "class_5", color: [128, 128, 128], pct: 100 }]);
    assert.deepEqual(segLegendRows(null), []);
    assert.deepEqual(segLegendRows({}), []);
  });
});

describe("renderSegLegendHtml", () => {
  it("renders chips with swatch, name, count + pct", () => {
    const s = renderSegLegendHtml({ resultSegLegend: LEGEND, _segIsolated: null });
    assert.ok(s.includes("Face_Neck"));
    assert.ok(s.includes("rgb(128,200,255)"));
    assert.ok(s.includes("30 · 30.0%"));
    assert.ok(!s.includes(" active\""));
  });

  it("marks the isolated chip active and escapes names", () => {
    const s = renderSegLegendHtml({ resultSegLegend: LEGEND, _segIsolated: "3" });
    assert.ok(s.includes('data-seg-chip="3"'));
    assert.ok(s.includes("flow-seg-chip active"));
    const evil = renderSegLegendHtml({
      resultSegLegend: { width: 1, height: 1, counts: { "1": 1 }, names: { "1": "<img>" } },
    });
    assert.ok(!evil.includes("<img>"));
    assert.equal(renderSegLegendHtml({ resultSegLegend: null }), "");
  });
});

describe("segClassAt", () => {
  const px = new Uint8Array([0, 3, 4, 3]);

  it("reads class ids in bounds, -1 outside", () => {
    assert.equal(segClassAt(px, 2, 2, 0, 0), 0);
    assert.equal(segClassAt(px, 2, 2, 1, 0), 3);
    assert.equal(segClassAt(px, 2, 2, 0, 1), 4);
    assert.equal(segClassAt(px, 2, 2, 1.9, 0.9), 3);
    assert.equal(segClassAt(px, 2, 2, 2, 0), -1);
    assert.equal(segClassAt(px, 2, 2, -1, 0), -1);
    assert.equal(segClassAt(px, 2, 2, 0, 2), -1);
  });
});

describe("isolateSegPixels", () => {
  it("keeps the isolated class and dims the rest, src untouched", () => {
    const src = new Uint8ClampedArray([200, 100, 50, 255, 10, 20, 30, 128]);
    const before = Array.from(src);
    const out = isolateSegPixels(src, new Uint8Array([3, 4]), "3");
    assert.deepEqual(Array.from(out.slice(0, 4)), [200, 100, 50, 255]);
    assert.deepEqual(Array.from(out.slice(4, 8)), [2, 4, 7, 128]);
    assert.deepEqual(Array.from(src), before);
    assert.ok(out instanceof Uint8ClampedArray);
  });
});

describe("fillSapiensResultView overlays", () => {
  const actNode = () => ({ id: "a", outputs: [], _segIsolated: "3" });

  it("stores pose keypoints + dims", () => {
    const n = actNode();
    fillSapiensResultView(n, "pose", {
      text: "t", imgUrl: "u", badge: "b", demo: false,
      stats: { keypoints: KPTS, image_width: 80, image_height: 100 },
    });
    assert.equal(n.resultKeypoints, KPTS);
    assert.equal(n.resultSegLegend, null);
    assert.deepEqual([n.imageWidth, n.imageHeight], [80, 100]);
    assert.equal(n._segIsolated, null);
  });

  it("stores seg legend + labels url + dims", () => {
    const n = actNode();
    fillSapiensResultView(n, "seg", {
      text: "t", imgUrl: "u", badge: "b", demo: false, labelsUrl: "/labels.png",
      stats: { legend: LEGEND, classes: [0, 3, 4] },
    });
    assert.equal(n.resultSegLegend, LEGEND);
    assert.equal(n.resultSegLabelsUrl, "/labels.png");
    assert.equal(n.resultKeypoints, null);
    assert.deepEqual([n.imageWidth, n.imageHeight], [10, 10]);
  });

  it("clears overlays and dims on demo / missing stats", () => {
    const n = actNode();
    n.resultKeypoints = KPTS; n.imageWidth = 80;
    fillSapiensResultView(n, "pose", { text: "t", imgUrl: "u", badge: "b", demo: true, stats: null });
    assert.equal(n.resultKeypoints, null);
    assert.deepEqual([n.imageWidth, n.imageHeight], [0, 0]);
  });
});

describe("renderResultPreviewHtml stages", () => {
  it("mounts the pose stage with a true-size viewBox", () => {
    const s = renderResultPreviewHtml({
      id: "p", templateKey: "action-sapiens-pose", title: "Pose",
      resultImage: "pose.jpg", resultImageBadge: "pose",
      resultKeypoints: KPTS, imageWidth: 80, imageHeight: 100,
    }, TOPO);
    assert.ok(s.includes("data-kpts-stage=\"1\""));
    assert.ok(s.includes('viewBox="0 0 80 100"'));
    assert.ok(s.includes("flow-kpts-overlay"));
  });

  it("mounts the seg stage with canvas + legend", () => {
    const s = renderResultPreviewHtml({
      id: "s", templateKey: "action-sapiens-seg", title: "Seg",
      resultImage: "seg.jpg", resultImageBadge: "seg",
      resultSegLegend: LEGEND, imageWidth: 10, imageHeight: 10,
    });
    assert.ok(s.includes("data-seg-stage=\"1\""));
    assert.ok(s.includes("flow-seg-canvas"));
    assert.ok(s.includes("flow-seg-legend"));
    assert.ok(s.includes("Face_Neck"));
  });

  it("leaves other cards alone", () => {
    const s = renderResultPreviewHtml({
      id: "x", templateKey: "action-meta-sam", title: "SAM",
      resultImage: "sam.jpg", resultImageBadge: "sam",
    });
    assert.ok(!s.includes("kpts-stage"));
    assert.ok(!s.includes("seg-stage"));
  });
});

describe("segSpawnRows", () => {
  const LEGEND = { width: 100, height: 50,
    counts: { 0: 3000, 1: 1500, 2: 500, 3: 0 },
    names: { 0: "background", 1: "face", 2: "hair", 3: "ghost" },
    palette: { 0: [0, 0, 0], 1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255] } };

  it("keeps present classes biggest-first, drops background and zeros", () => {
    const rows = segSpawnRows(LEGEND);
    assert.deepEqual(rows.map(r => r.id), ["1", "2"]);
    assert.equal(rows[0].name, "face");
    assert.equal(rows[0].pct, 30);
  });

  it("matches background case-insensitively, keeps the nameless", () => {
    const rows = segSpawnRows({ width: 10, height: 10, counts: { 0: 60, 7: 40 },
      names: { 0: "Background" }, palette: {} });
    assert.deepEqual(rows.map(r => r.id), ["7"]);
    assert.equal(rows[0].name, "class_7");
  });

  it("returns [] for thin or missing legends", () => {
    assert.deepEqual(segSpawnRows(null), []);
    assert.deepEqual(segSpawnRows({}), []);
    assert.deepEqual(segSpawnRows({ width: 4, height: 4, counts: { 0: 16 },
      names: { 0: "background" }, palette: {} }), []);
  });
});
