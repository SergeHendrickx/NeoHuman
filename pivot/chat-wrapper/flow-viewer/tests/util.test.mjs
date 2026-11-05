// Focused regression tests for flow-viewer shared helpers.
// Run from the flow-viewer folder:  node --test tests/
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { escapeHtml, uid, svgPlaceholder, formatLatency, minimizePinSide, aspectRatioStyle,
  cardWidthForImage, LIST_TYPES, isListType, itemTypeForList, listCountLabel, redactBlobs, appendRunLog,
  formatRunLogTime, PIN_TYPES, pinTypeColor, canConnect, canWirePorts, sameHtml,
  resolveAssetImageUrl, wiredViewLabel, syncViewerStatus,
  META_BLUE, META_MODEL_FAMILIES, isMetaModel, nodeAccentColor } from "../js/util.js";
import { MODEL_CATALOG } from "../js/catalog.js";

describe("escapeHtml", () => {
  it("escapes markup, quotes and ampersands", () => {
    assert.equal(
      escapeHtml(`<img src=x onerror="alert('pwn')"> & "quoted"`),
      "&lt;img src=x onerror=&quot;alert(&#39;pwn&#39;)&quot;&gt; &amp; &quot;quoted&quot;"
    );
  });

  it("tolerates nullish values", () => {
    assert.equal(escapeHtml(null), "");
    assert.equal(escapeHtml(undefined), "");
  });
});

describe("uid", () => {
  it("generates unique ids in a tight loop (no Date.now collisions)", () => {
    const seen = new Set();
    for (let i = 0; i < 2000; i++) {
      seen.add(uid("asset-boxes"));
    }
    assert.equal(seen.size, 2000);
  });

  it("keeps the requested prefix", () => {
    assert.match(uid("candidate-p0"), /^candidate-p0-/);
  });
});

describe("svgPlaceholder", () => {
  it("returns an offline data URI without external hosts", () => {
    const uri = svgPlaceholder("Vision Boxes");
    assert.match(uri, /^data:image\/svg\+xml/);
    assert.doesNotMatch(uri, /https?:\/\//);
    assert.match(decodeURIComponent(uri), /Vision Boxes/);
  });

  it("escapes the label", () => {
    const uri = decodeURIComponent(svgPlaceholder(`<b>"x"</b>`));
    assert.doesNotMatch(uri, /<b>/);
  });
});

describe("formatLatency", () => {
  it("formats backend latency numbers and strings", () => {
    assert.equal(formatLatency(1.85), "1.85s");
    assert.equal(formatLatency("1.85s"), "1.85s");
    assert.equal(formatLatency(null), "—");
    assert.equal(formatLatency(undefined), "—");
  });
});

describe("minimizePinSide", () => {
  it("minimizes single-pin sides on asset cards only", () => {
    assert.equal(minimizePinSide("asset", [{ id: "out-photo" }]), true);
    assert.equal(minimizePinSide("asset", []), false);
    assert.equal(minimizePinSide("asset", [{ id: "a" }, { id: "b" }]), false);
    assert.equal(minimizePinSide("asset", null), false);
    assert.equal(minimizePinSide("action", [{ id: "out-kpts" }]), false);
  });
});

describe("list types", () => {
  it("covers the four detector lists with item mappings", () => {
    assert.deepEqual(LIST_TYPES, ["boxes", "masks", "confidences", "labels"]);
    assert.equal(isListType("masks"), true);
    assert.equal(isListType("image"), false);
    assert.equal(itemTypeForList("boxes"), "box");
    assert.equal(itemTypeForList("masks"), "mask");
    assert.equal(itemTypeForList("confidences"), "number");
    assert.equal(itemTypeForList("labels"), "text");
    assert.equal(itemTypeForList("crop"), null);
  });

  it("labels counts with singular/plural forms", () => {
    assert.equal(listCountLabel("boxes", 2), "2 boxes");
    assert.equal(listCountLabel("boxes", 1), "1 box");
    assert.equal(listCountLabel("labels", 0), "0 labels");
    assert.equal(listCountLabel("masks", null), "0 masks");
  });
});

describe("redactBlobs", () => {
  it("redacts base64 fields and data URLs, keeps everything else verbatim", () => {
    const input = {
      chat_id: "flow-session",
      concept: "people",
      image_b64: "a".repeat(1000),
      nested: { thumb_b64: "b".repeat(10), encoding: "RLE:" + "01".repeat(500) },
      regular: "x".repeat(500),
    };
    const out = redactBlobs(input);
    assert.equal(out.image_b64, "<1000 base64 chars redacted>");
    assert.equal(out.nested.thumb_b64, "<10 base64 chars redacted>");
    assert.equal(out.nested.encoding, input.nested.encoding);
    assert.equal(out.regular, input.regular);
    assert.equal(out.concept, "people");
    // Input not mutated.
    assert.equal(input.image_b64.length, 1000);
  });

  it("redacts long data URLs only", () => {
    assert.equal(redactBlobs("data:image/jpeg;base64," + "c".repeat(300)),
      "<data URL, 323 chars redacted>");
    assert.equal(redactBlobs("data:,short"), "data:,short");
    assert.equal(redactBlobs("/api/runs/x/artifact/source"), "/api/runs/x/artifact/source");
  });
});

describe("appendRunLog", () => {
  it("appends timestamped entries and caps length", () => {
    const node = {};
    appendRunLog(node, "info", "Run started");
    appendRunLog(node, "error", "Failed: x");
    assert.equal(node.runLog.length, 2);
    assert.equal(node.runLog[0].level, "info");
    assert.equal(node.runLog[1].msg, "Failed: x");
    assert.match(formatRunLogTime(node.runLog[0].t), /^\d{2}:\d{2}:\d{2}$/);
    for (let i = 0; i < 40; i++) appendRunLog(node, "info", `m${i}`);
    assert.equal(node.runLog.length, 30);
    assert.equal(node.runLog[29].msg, "m39");
  });

  it("ignores nullish nodes", () => {
    assert.doesNotThrow(() => appendRunLog(null, "info", "x"));
  });
});

describe("pin types", () => {
  it("declares every wire type with a color and self-acceptance", () => {
    const types = ["image", "crop", "box", "boxes", "mask", "masks", "confidences",
      "labels", "number", "text", "pointcloud", "geometry", "normalmap", "depthmap",
      "mesh", "vector", "list"];
    for (const t of types) {
      assert.ok(PIN_TYPES[t], `missing registry row for ${t}`);
      assert.match(PIN_TYPES[t].color, /^#[0-9a-fA-F]{6}$/);
      if (t !== "list") assert.ok(PIN_TYPES[t].accepts.includes(t), `${t} must accept itself`);
    }
    assert.equal(pinTypeColor("image"), "#10b981");
    assert.equal(pinTypeColor("nope", "#000"), "#000");
  });

  it("enforces strict connections with explicit compatibilities", () => {
    assert.equal(canConnect("image", "image"), true);
    assert.equal(canConnect("mask", "image"), false);
    assert.equal(canConnect("box", "crop"), true);
    assert.equal(canConnect("crop", "box"), false);
    assert.equal(canConnect("pointcloud", "geometry"), true);
    assert.equal(canConnect("geometry", "pointcloud"), true);
    assert.equal(canConnect("boxes", "boxes"), true);
    assert.equal(canConnect("confidences", "image"), false);
    assert.equal(canConnect("masks", "masks"), true);
    assert.equal(canConnect("custom", "custom"), true);
    assert.equal(canConnect("custom", "image"), false);
    assert.equal(canConnect("normalmap", "image"), true);
    assert.equal(canConnect("depthmap", "image"), true);
    assert.equal(canConnect("image", "normalmap"), false);
    assert.equal(canConnect("image", "depthmap"), false);
  });
});

describe("cardWidthForImage", () => {
  it("grows wide images proportionally from the 320px base", () => {
    assert.equal(cardWidthForImage(1536, 768), 640);
    assert.equal(cardWidthForImage(640, 480), 427);
    assert.equal(cardWidthForImage("1920", "1080"), 569);
  });

  it("caps very wide images at 720px", () => {
    assert.equal(cardWidthForImage(3000, 500), 720);
    assert.equal(cardWidthForImage(1536, 256), 720);
  });

  it("keeps the default width for square, tall and unknown dims", () => {
    assert.equal(cardWidthForImage(512, 512), null);
    assert.equal(cardWidthForImage(480, 640), null);
    assert.equal(cardWidthForImage(null, 480), null);
    assert.equal(cardWidthForImage(640, 0), null);
    assert.equal(cardWidthForImage("wide", "tall"), null);
  });
});

describe("aspectRatioStyle", () => {
  it("emits aspect-ratio for valid dimensions", () => {
    assert.equal(aspectRatioStyle(1000, 1500), "aspect-ratio:1000 / 1500;");
    assert.equal(aspectRatioStyle("640", "480"), "aspect-ratio:640 / 480;");
  });

  it("emits nothing for unknown or invalid dimensions", () => {
    assert.equal(aspectRatioStyle(null, 480), "");
    assert.equal(aspectRatioStyle(640, undefined), "");
    assert.equal(aspectRatioStyle(0, 480), "");
    assert.equal(aspectRatioStyle(-3, 480), "");
    assert.equal(aspectRatioStyle("wide", "tall"), "");
  });
});

describe("meta color", () => {
  it("recognizes all three Meta families with the universal fb-blue accent", () => {
    assert.deepEqual([...META_MODEL_FAMILIES].sort(), ["avatar-solver", "meta-ai", "sapiens"]);
    for (const family of META_MODEL_FAMILIES) {
      const node = { nodeType: "action", family };
      assert.equal(isMetaModel(node), true, `${family} must be a Meta model`);
      assert.equal(nodeAccentColor(node), META_BLUE, `${family} must use fb-blue`);
    }
    assert.equal(isMetaModel({ nodeType: "action", family: "utility" }), false);
    assert.equal(isMetaModel({ nodeType: "asset", family: "meta-ai" }), false);
  });

  it("gives every Meta catalog entry the meta badge and icon variants", () => {
    const metas = Object.values(MODEL_CATALOG).filter(n =>
      n.nodeType === "action" && META_MODEL_FAMILIES.includes(n.family));
    assert.ok(metas.length >= 3, "catalog must carry all three Meta families");
    assert.deepEqual(new Set(metas.map(m => m.family)),
      new Set(["meta-ai", "sapiens", "avatar-solver"]));
    for (const m of metas) {
      assert.equal(m.badgeVariant, "meta-light", `${m.id} badge`);
      assert.equal(m.iconVariant, "meta", `${m.id} icon`);
    }
  });

  it("keeps purple out of the pin type registry", () => {
    for (const [t, def] of Object.entries(PIN_TYPES)) {
      assert.notEqual(def.color.toLowerCase(), "#a855f7", `${t} must not be purple`);
    }
    assert.equal(pinTypeColor("box"), META_BLUE);
    assert.equal(pinTypeColor("boxes"), META_BLUE);
    assert.equal(pinTypeColor("normalmap"), META_BLUE);
  });

  it("keeps purple out of Meta catalog pins", () => {
    for (const m of Object.values(MODEL_CATALOG)) {
      for (const pin of [...(m.inputs || []), ...(m.outputs || [])]) {
        assert.ok(!pin.color || pin.color.toLowerCase() !== "#a855f7",
          `${m.id}.${pin.id} must not be purple`);
      }
    }
  });
});

describe("sameHtml", () => {
  it("reports first paint as changed, repeats as identical", () => {
    const el = {};
    assert.equal(sameHtml(el, "<div>a</div>"), false);
    assert.equal(sameHtml(el, "<div>a</div>"), true);
    assert.equal(sameHtml(el, "<div>b</div>"), false);
    assert.equal(sameHtml(el, "<div>b</div>"), true);
  });

  it("tracks elements independently", () => {
    const a = {};
    const b = {};
    assert.equal(sameHtml(a, "x"), false);
    assert.equal(sameHtml(b, "x"), false);
    assert.equal(sameHtml(a, "x"), true);
  });
});

describe("canWirePorts", () => {
  it("defers to the registry when the pin declares no accepts list", () => {
    assert.equal(canWirePorts("image", { type: "image" }), true);
    assert.equal(canWirePorts("box", { type: "crop" }), true);
    assert.equal(canWirePorts("mask", { type: "image" }), false);
    assert.equal(canWirePorts("image", null), false);
  });

  it("honors a pin-level accepts list without loosening the registry", () => {
    const viewPin = { type: "image", accepts: ["image", "box", "mask", "crop"] };
    assert.equal(canWirePorts("box", viewPin), true);
    assert.equal(canWirePorts("mask", viewPin), true);
    assert.equal(canWirePorts("crop", viewPin), true);
    assert.equal(canWirePorts("boxes", viewPin), false);
    assert.equal(canWirePorts("text", viewPin), false);
    // Same source type into a plain image pin still fails.
    assert.equal(canWirePorts("box", { type: "image" }), false);
  });
});

describe("resolveAssetImageUrl", () => {
  const photo = {
    id: "asset-photo", nodeType: "asset", assetKind: "image",
    imgUrl: "photo.jpg", inputs: [{ id: "in-view" }], outputs: [{ id: "out-photo", name: "Photo Output" }],
  };
  const sam = {
    id: "action-meta-sam", nodeType: "action", title: "Action: Meta SAM 3.1",
    resultImage: "overlay.jpg",
    outputs: [
      { id: "out-boxes", name: "Bounding Boxes" },
      { id: "split-out-boxes-0", name: "Box 1", thumb: "crop1.jpg" },
    ],
  };
  const viewer = (id) => ({
    id, nodeType: "asset", assetKind: "image", templateKey: "asset-photo-viewer",
    imgUrl: "placeholder", status: "Waiting for Input",
    inputs: [{ id: "in-view" }], outputs: [{ id: "out-view", name: "Passthrough" }],
  });

  it("returns the card's own photo when unwired", () => {
    const app = { nodes: [photo], edges: [] };
    assert.equal(resolveAssetImageUrl(app, photo), "photo.jpg");
  });

  it("follows a wired input to an upstream asset photo", () => {
    const v = viewer("v1");
    const app = { nodes: [photo, v],
      edges: [{ to: "v1", toPort: "in-view", from: "asset-photo", fromPort: "out-photo" }] };
    assert.equal(resolveAssetImageUrl(app, v), "photo.jpg");
  });

  it("prefers a split item pin's thumbnail crop", () => {
    const v = viewer("v1");
    const app = { nodes: [sam, v],
      edges: [{ to: "v1", toPort: "in-view", from: "action-meta-sam", fromPort: "split-out-boxes-0" }] };
    assert.equal(resolveAssetImageUrl(app, v), "crop1.jpg");
  });

  it("falls back to an action's result image for thumb-less pins", () => {
    const v = viewer("v1");
    const app = { nodes: [sam, v],
      edges: [{ to: "v1", toPort: "in-view", from: "action-meta-sam", fromPort: "out-boxes" }] };
    assert.equal(resolveAssetImageUrl(app, v), "overlay.jpg");
  });

  it("resolves through viewer chains to real pixels", () => {
    const a = viewer("vA");
    const b = viewer("vB");
    const app = { nodes: [photo, a, b], edges: [
      { to: "vA", toPort: "in-view", from: "asset-photo", fromPort: "out-photo" },
      { to: "vB", toPort: "in-view", from: "vA", fromPort: "out-view" },
    ] };
    assert.equal(resolveAssetImageUrl(app, b), "photo.jpg");
  });

  it("terminates on wiring cycles and falls back to the own photo", () => {
    const a = viewer("vA");
    const b = viewer("vB");
    const app = { nodes: [a, b], edges: [
      { to: "vA", toPort: "in-view", from: "vB", fromPort: "out-view" },
      { to: "vB", toPort: "in-view", from: "vA", fromPort: "out-view" },
    ] };
    assert.equal(resolveAssetImageUrl(app, a), "placeholder");
  });

  it("falls back to the own photo when the upstream node is gone", () => {
    const v = viewer("v1");
    const app = { nodes: [v],
      edges: [{ to: "v1", toPort: "in-view", from: "deleted", fromPort: "out-x" }] };
    assert.equal(resolveAssetImageUrl(app, v), "placeholder");
  });
});

describe("wiredViewLabel / syncViewerStatus", () => {
  const sam = {
    id: "action-meta-sam", nodeType: "action", title: "Action: Meta SAM 3.1",
    outputs: [{ id: "split-out-boxes-1", name: "Box 2", thumb: "crop2.jpg" }],
  };
  const viewer = {
    id: "v1", nodeType: "asset", templateKey: "asset-photo-viewer",
    status: "Waiting for Input", inputs: [{ id: "in-view" }],
  };

  it("names the upstream pin and flips viewer status while wired", () => {
    const v = { ...viewer, inputs: [...viewer.inputs] };
    const app = { nodes: [sam, v],
      edges: [{ to: "v1", toPort: "in-view", from: "action-meta-sam", fromPort: "split-out-boxes-1" }] };
    assert.equal(wiredViewLabel(app, v), "Box 2");
    syncViewerStatus(app, v);
    assert.equal(v.status, "Viewing ← Box 2");
  });

  it("clears back to Waiting for Input once unwired", () => {
    const v = { ...viewer, status: "Viewing ← Box 2", inputs: [...viewer.inputs] };
    const app = { nodes: [sam, v], edges: [] };
    assert.equal(wiredViewLabel(app, v), null);
    syncViewerStatus(app, v);
    assert.equal(v.status, "Waiting for Input");
  });

  it("leaves non-viewer cards alone", () => {
    const photo = { id: "asset-photo", nodeType: "asset", status: "Cached in RAM", inputs: [{ id: "in-view" }] };
    const app = { nodes: [sam, photo], edges: [] };
    assert.equal(wiredViewLabel(app, photo), null);
    syncViewerStatus(app, photo);
    assert.equal(photo.status, "Cached in RAM");
  });
});
