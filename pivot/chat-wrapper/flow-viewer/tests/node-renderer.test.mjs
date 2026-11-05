// Regression tests for pure node-renderer helpers (no DOM needed).
// Run from the flow-viewer folder:  node tests/node-renderer.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { statusDotClass, headerStatusClass, assetViewBadge, renderOverlaySvg, renderResultPreviewHtml,
  renderProgressHtml, syncProgressBlock, renderParamEditorHtml } from "../js/node-renderer.js";

describe("statusDotClass", () => {
  it("maps statuses to dot classes", () => {
    assert.equal(statusDotClass("Passed"), "success");
    assert.equal(statusDotClass("Done"), "success");
    assert.equal(statusDotClass("Cached in RAM"), "warning");
    assert.equal(statusDotClass("Artifact"), "warning");
    assert.equal(statusDotClass("Produced"), "warning");
    assert.equal(statusDotClass("Running..."), "active");
    assert.equal(statusDotClass("Viewing ← Box 2"), "warning");
    assert.equal(statusDotClass("Demo (Offline)"), "warning");
    assert.equal(statusDotClass("Failed"), "error");
    assert.equal(statusDotClass("Ready to Run"), "");
    assert.equal(statusDotClass("Waiting for Input"), "");
    assert.equal(statusDotClass(null), "");
  });
});

describe("headerStatusClass", () => {
  it("maps run outcomes to header band classes", () => {
    assert.equal(headerStatusClass("Passed"), "is-passed");
    assert.equal(headerStatusClass("Done"), "is-passed");
    assert.equal(headerStatusClass("Running..."), "is-running");
    assert.equal(headerStatusClass("Failed"), "is-failed");
    assert.equal(headerStatusClass("Demo (Offline)"), "is-demo");
    assert.equal(headerStatusClass("Timed out"), "is-demo");
    assert.equal(headerStatusClass("Produced"), "is-demo");
    assert.equal(headerStatusClass("Ready to Run"), "");
    assert.equal(headerStatusClass("Waiting for Input"), "");
    assert.equal(headerStatusClass(null), "");
  });

  it("agrees with the status dot on every outcome family", () => {
    const pairs = [["Passed", "success"], ["Running...", "active"], ["Failed", "error"],
      ["Demo (Offline)", "warning"]];
    for (const [status, dot] of pairs) {
      assert.equal(statusDotClass(status), dot);
      assert.notEqual(headerStatusClass(status), "",
        `${status} must color the header, like its dot`);
    }
    assert.equal(headerStatusClass("Ready to Run"), "");
  });
});

describe("assetViewBadge", () => {
  const sam = {
    id: "action-meta-sam", nodeType: "action",
    outputs: [{ id: "split-out-boxes-1", name: "Box 2", thumb: "crop2.jpg" }],
  };

  it("shows the upstream pin while a wired input overrides the view", () => {
    const v = { id: "v1", nodeType: "asset", imgBadge: "unwired",
      imageWidth: 320, imageHeight: 480, inputs: [{ id: "in-view" }] };
    const app = { nodes: [sam, v],
      edges: [{ to: "v1", toPort: "in-view", from: "action-meta-sam", fromPort: "split-out-boxes-1" }] };
    assert.equal(assetViewBadge(app, v), "← Box 2 · 320x480");
  });

  it("falls back to the card's own badge when unwired", () => {
    const v = { id: "v1", nodeType: "asset", imgBadge: "unwired", inputs: [{ id: "in-view" }] };
    const app = { nodes: [sam, v], edges: [] };
    assert.equal(assetViewBadge(app, v), "unwired");
  });
});

describe("renderOverlaySvg", () => {
  const sam = {
    masks: [
      { box: [10, 20, 110, 220], confidence: 0.987 },
      { box: [120, 30, 220, 230], confidence: 0.5 },
    ],
    resultData: { concept: "clothes", labels: ["clothes 1", "clothes 2"] },
  };

  it("tags detections with their instance labels, not a hardcoded noun", () => {
    const svg = renderOverlaySvg(sam);
    assert.match(svg, /clothes 1/);
    assert.match(svg, /clothes 2/);
    assert.doesNotMatch(svg, /Person/);
  });

  it("falls back to concept + index when labels are absent", () => {
    const svg = renderOverlaySvg({ masks: sam.masks, resultData: { concept: "face" } });
    assert.match(svg, /face 1/);
    const bare = renderOverlaySvg({ masks: sam.masks });
    assert.match(bare, /P1/);
  });

  it("renders nothing without masks", () => {
    assert.equal(renderOverlaySvg({}), "");
    assert.equal(renderOverlaySvg({ masks: [] }), "");
  });
});

describe("renderResultPreviewHtml error block", () => {
  it("shows the failure message loudly on failed cards", () => {
    const s = renderResultPreviewHtml({
      id: "a", templateKey: "action-sapiens-seg", title: "Seg",
      status: "Failed", runError: "Backend unreachable (<fetch>) — check the server console.",
      apiCall: { at: 1, hop1: { method: "POST", url: "/api/debug/x", status: 0 } },
    });
    assert.ok(s.includes("flow-error-block"));
    assert.ok(s.includes("Backend unreachable"));
    assert.ok(!s.includes("<fetch>"));
  });

  it("shows no error block on healthy cards", () => {
    const s = renderResultPreviewHtml({
      id: "a", templateKey: "action-meta-sam", title: "SAM", status: "Passed",
      resultImage: "u", runLog: [{ t: 1, level: "info", msg: "ok" }],
    });
    assert.ok(!s.includes("flow-error-block"));
  });
});

describe("renderProgressHtml", () => {
  it("renders bar width and stage label from node.progress", () => {
    const s = renderProgressHtml({ progress: { frac: 0.456, label: "running seg" } });
    assert.match(s, /class="flow-progress"/);
    assert.match(s, /width:46%/);
    assert.match(s, /running seg/);
  });

  it("defaults to 0% starting… and escapes the label", () => {
    const s = renderProgressHtml({});
    assert.match(s, /width:0%/);
    assert.match(s, /starting…/);
    assert.ok(!renderProgressHtml({ progress: { frac: 0, label: "<img>" } }).includes("<img>"));
  });
});

describe("syncProgressBlock", () => {
  // Minimal fake card element: strip/prog slots record what the sync did.
  const fakeEl = ({ hasProg = false, hasStrip = true } = {}) => {
    const calls = { inserted: [], removed: 0 };
    const strip = hasStrip ? {
      insertAdjacentHTML(pos, html) { calls.inserted.push([pos, html]); },
    } : null;
    const prog = hasProg ? { remove() { calls.removed++; } } : null;
    return {
      calls,
      querySelector(sel) {
        if (sel === ".flow-node-run-strip") return strip;
        if (sel === ".flow-progress") return prog;
        return null;
      },
    };
  };

  it("injects the block when Running without one (the real-run path)", () => {
    const el = fakeEl();
    syncProgressBlock(el, { status: "Running...", progress: { frac: 0.5, label: "half" } });
    assert.equal(el.calls.inserted.length, 1);
    assert.equal(el.calls.inserted[0][0], "afterend");
    assert.match(el.calls.inserted[0][1], /width:50%/);
  });

  it("never duplicates an existing block mid-run", () => {
    const el = fakeEl({ hasProg: true });
    syncProgressBlock(el, { status: "Running..." });
    assert.equal(el.calls.inserted.length, 0);
    assert.equal(el.calls.removed, 0);
  });

  it("removes the block on every terminal state", () => {
    for (const status of ["Passed", "Failed", "Demo (Offline)", "Ready to Run"]) {
      const el = fakeEl({ hasProg: true });
      syncProgressBlock(el, { status });
      assert.equal(el.calls.removed, 1, status);
    }
  });

  it("no-ops when idle without a block, or when the strip is missing", () => {
    const el = fakeEl();
    syncProgressBlock(el, { status: "Passed" });
    assert.equal(el.calls.removed, 0);
    const noStrip = fakeEl({ hasStrip: false });
    syncProgressBlock(noStrip, { status: "Running..." });
    assert.equal(noStrip.calls.inserted.length, 0);
  });
});

describe("renderParamEditorHtml multiview", () => {
  const node = (param) => ({ id: "m", templateKey: "action-mesh-multiview", param });

  it("offers 2/4/6/8 view chips with the current count active", () => {
    const s = renderParamEditorHtml(node("6"));
    assert.match(s, /Views:/);
    for (const n of ["2", "4", "6", "8"]) {
      assert.ok(s.includes(`data-concept="${n}"`), `chip ${n}`);
    }
    assert.match(s, /data-concept="6">6</);
    assert.ok(!s.includes("flow-param-input"), "chips only, no free text");
  });

  it("leaves other cards on the prompt editor", () => {
    const s = renderParamEditorHtml({ id: "s", templateKey: "action-spark", param: "hi" });
    assert.match(s, /flow-param-input/);
    assert.ok(!s.includes("Views:"));
  });

  it("offers body/waist-up/face framing chips with the current frame active", () => {
    const s = renderParamEditorHtml({ id: "m", templateKey: "action-mesh-multiview",
      param: "4", framing: "face" });
    assert.match(s, /Frame:/);
    for (const f of ["body", "waist", "face"]) {
      assert.ok(s.includes(`data-framing="${f}"`), `chip ${f}`);
    }
    assert.match(s, /data-framing="face">Face</);
    assert.match(s, /flow-param-chip active" data-framing="face"/);
    const d = renderParamEditorHtml({ id: "m", templateKey: "action-mesh-multiview", param: "4" });
    assert.match(d, /flow-param-chip active" data-framing="body"/);
  });
});
