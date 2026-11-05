// Node gallery: grouped collection of every placeable card.
// Run from the flow-viewer folder:  node tests/gallery.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  galleryGroupKey, groupCatalogEntries, filterGalleryGroups,
  renderGalleryHtml, gallerySpawnCenter, shortCardTitle,
  assetTypeIndex, filterAssetTypes, renderAssetTypesHtml,
} from "../js/gallery.js";
import { MODEL_CATALOG } from "../js/catalog.js";
import { PIN_TYPES } from "../js/util.js";

describe("galleryGroupKey", () => {
  it("routes actions by family, all assets together, junk to other", () => {
    assert.equal(galleryGroupKey({ nodeType: "action", family: "meta-ai" }), "meta-ai");
    assert.equal(galleryGroupKey({ nodeType: "action", family: "sapiens" }), "sapiens");
    assert.equal(galleryGroupKey({ nodeType: "action", family: "avatar-solver" }), "avatar-solver");
    assert.equal(galleryGroupKey({ nodeType: "action", family: "mesh" }), "mesh");
    assert.equal(galleryGroupKey({ nodeType: "action", family: "utility" }), "utility");
    assert.equal(galleryGroupKey({ nodeType: "action", family: "nope" }), "other");
    assert.equal(galleryGroupKey({ nodeType: "asset", family: "utility" }), "assets");
    assert.equal(galleryGroupKey({ nodeType: "asset" }), "assets");
    assert.equal(galleryGroupKey({ nodeType: "wire" }), "other");
    assert.equal(galleryGroupKey(null), "other");
  });
});

describe("groupCatalogEntries", () => {
  it("orders groups actions-first and drops empties", () => {
    const groups = groupCatalogEntries({
      a: { id: "a", nodeType: "asset" },
      m: { id: "m", nodeType: "action", family: "mesh" },
      s: { id: "s", nodeType: "action", family: "sapiens" },
    });
    assert.deepEqual(groups.map(g => g.key), ["sapiens", "mesh", "assets"]);
    assert.ok(groups.every(g => g.items.length > 0));
  });

  it("places every real catalog entry in exactly one group", () => {
    const groups = groupCatalogEntries(MODEL_CATALOG);
    const placed = groups.flatMap(g => g.items.map(e => e.id)).sort();
    assert.deepEqual(placed, Object.keys(MODEL_CATALOG).sort());
    const byId = new Map(groups.flatMap(g => g.items.map(e => [e.id, g.key])));
    assert.equal(byId.get("action-meta-sam"), "meta-ai");
    assert.equal(byId.get("action-sapiens-seg"), "sapiens");
    assert.equal(byId.get("action-body-fit"), "avatar-solver");
    assert.equal(byId.get("action-mesh-depth"), "mesh");
    assert.equal(byId.get("action-export"), "utility");
    assert.equal(byId.get("asset-photo-viewer"), "assets");
    assert.equal(byId.get("asset-photo"), "assets");
  });

  it("skips null entries without throwing", () => {
    assert.deepEqual(groupCatalogEntries(null), []);
    assert.deepEqual(groupCatalogEntries({ x: null }).length, 0);
  });
});

describe("filterGalleryGroups", () => {
  const groups = groupCatalogEntries(MODEL_CATALOG);

  it("matches titles, subtitles, text and ids case-insensitively", () => {
    const byTitle = filterGalleryGroups(groups, "sapiensseg");
    assert.equal(byTitle.flatMap(g => g.items).length, 1);
    const bySub = filterGalleryGroups(groups, "mesh_depth_map");
    assert.deepEqual(bySub.flatMap(g => g.items).map(e => e.id), ["action-mesh-depth"]);
    const byId = filterGalleryGroups(groups, "ACTION-EXPORT");
    assert.deepEqual(byId.flatMap(g => g.items).map(e => e.id), ["action-export"]);
  });

  it("drops emptied groups and returns the input on blank query", () => {
    const filtered = filterGalleryGroups(groups, "depth");
    assert.ok(filtered.length > 0 && filtered.length < groups.length);
    assert.ok(filtered.every(g => g.items.length > 0));
    assert.equal(filterGalleryGroups(groups, "  "), groups);
    assert.deepEqual(filterGalleryGroups(groups, "zzz-no-such-node"), []);
  });
});

describe("renderGalleryHtml", () => {
  it("renders sections with counts and spawn keys", () => {
    const html = renderGalleryHtml(groupCatalogEntries(MODEL_CATALOG));
    assert.match(html, /Meta AI/);
    assert.match(html, /Assets/);
    assert.match(html, /data-add-key="action-meta-sam"/);
    assert.match(html, /data-add-key="asset-photo"/);
    for (const id of Object.keys(MODEL_CATALOG)) {
      assert.ok(html.includes(`data-add-key="${id}"`), `tile for ${id}`);
    }
  });

  it("shows an empty state and escapes text", () => {
    assert.match(renderGalleryHtml([]), /No nodes match/);
    const html = renderGalleryHtml([{
      key: "other", label: "Other", hint: "",
      items: [{ id: "x", title: "<b>evil</b>", subtitle: null, description: null }],
    }]);
    assert.ok(!html.includes("<b>evil</b>"));
  });
});

describe("gallerySpawnCenter", () => {
  it("maps the viewport center into world coordinates", () => {
    const app = { viewport: { clientWidth: 800, clientHeight: 600 }, panX: 100, panY: 60, scale: 2 };
    assert.deepEqual(gallerySpawnCenter(app), { x: 150, y: 120 });
  });

  it("falls back sanely without geometry", () => {
    assert.deepEqual(gallerySpawnCenter({}), { x: 400, y: 300 });
  });
});

describe("assetTypeIndex", () => {
  it("covers every pin type with its canonical color and accepts", () => {
    const rows = assetTypeIndex(MODEL_CATALOG);
    assert.deepEqual(rows.map(r => r.type), Object.keys(PIN_TYPES));
    const image = rows.find(r => r.type === "image");
    assert.equal(image.label, "Image");
    assert.equal(image.color, "#10b981");
    assert.deepEqual(image.accepts, ["image", "normalmap", "depthmap"]);
  });

  it("derives template-level producers and consumers", () => {
    const rows = assetTypeIndex({
      a: { title: "Action: Maker", outputs: [{ type: "mesh" }], inputs: [] },
      b: { title: "Asset: Taker", outputs: [], inputs: [{ type: "text", accepts: ["mesh"] }] },
    });
    const mesh = rows.find(r => r.type === "mesh");
    assert.deepEqual(mesh.producers, ["Maker"]);
    assert.deepEqual(mesh.consumers, ["Taker"]);
  });

  it("spots-checks the real catalog wiring", () => {
    const rows = assetTypeIndex(MODEL_CATALOG);
    const of = (t) => rows.find(r => r.type === t);
    assert.ok(of("mesh").producers.includes("Character Exporter"));
    assert.ok(of("mesh").consumers.includes("Mesh Depth"));
    assert.ok(of("image").producers.includes("Apply Mask"));
    assert.ok(of("mask").producers.includes("SapiensSeg"));
    assert.ok(of("mask").consumers.includes("Apply Mask"));
    assert.ok(of("text").producers.includes("Meta Spark"));
    assert.deepEqual(of("vector").producers, ["SAM 3D Body & Joint Solver"]);
    assert.deepEqual(of("vector").consumers, []);
  });

  it("shortens card titles", () => {
    assert.equal(shortCardTitle("Action: Meta SAM 3.1"), "Meta SAM 3.1");
    assert.equal(shortCardTitle("Asset: Source Photo"), "Source Photo");
    assert.equal(shortCardTitle(null), "Untitled");
  });
});

describe("filterAssetTypes", () => {
  const rows = assetTypeIndex(MODEL_CATALOG);

  it("matches labels, keys, makers and takers", () => {
    assert.deepEqual(filterAssetTypes(rows, "depthmap").map(r => r.type), ["depthmap"]);
    const byMaker = filterAssetTypes(rows, "sapiensseg");
    assert.ok(byMaker.some(r => r.type === "mask"));
    assert.equal(filterAssetTypes(rows, "  "), rows);
    assert.deepEqual(filterAssetTypes(rows, "zzz-no-such-type"), []);
  });
});

describe("renderAssetTypesHtml", () => {
  it("renders dots, names, makers and takers", () => {
    const html = renderAssetTypesHtml(assetTypeIndex(MODEL_CATALOG));
    assert.match(html, /flow-type-dot/);
    assert.match(html, /Depth Map/);
    assert.match(html, /made by: .*Mesh Depth/);
    assert.match(html, /used by: /);
  });

  it("shows an empty state and escapes text", () => {
    assert.match(renderAssetTypesHtml([]), /No types match/);
    const html = renderAssetTypesHtml([{
      type: "x", label: "<i>", color: "#fff", accepts: [],
      producers: ["<p>"], consumers: [],
    }]);
    assert.ok(!html.includes("<i>") && !html.includes("<p>"));
  });
});
