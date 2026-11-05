// Interactive 3D stage: mount rules, stage markup, failure reasons, dispose
// bookkeeping, and the vendor/importmap contract. three.js itself loads only
// in the browser (dynamic import); node covers the pure view-model.
// Run from the flow-viewer folder:  node tests/meshview.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import {
  meshViewerKind, shouldMountMeshViewer, renderMeshStageHtml,
  meshFailureReason, meshDownloadName, stretchDepthPixels, renderMeshStill,
  viewDirForAzimuth, multiviewAzimuths, multiviewGrid, composeMultiviewSheet,
  MESH_OFFSCREEN_SIZE, disposeMeshViewer, bindMeshViewer, THREE_VERSION,
} from "../js/mesh-view.js";
import { renderResultPreviewHtml } from "../js/node-renderer.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

describe("meshViewerKind", () => {
  it("prefers GLB over OBJ, null when neither", () => {
    assert.deepEqual(meshViewerKind({ glbUrl: "a.glb", meshObjUrl: "b.obj" }),
      { url: "a.glb", format: "glb" });
    assert.deepEqual(meshViewerKind({ meshObjUrl: "b.obj" }),
      { url: "b.obj", format: "obj" });
    assert.equal(meshViewerKind({}), null);
  });
});

describe("shouldMountMeshViewer", () => {
  it("mounts on export/body-fit results and mesh assets", () => {
    assert.equal(shouldMountMeshViewer(
      { nodeType: "action", templateKey: "action-export", glbUrl: "a.glb" }), true);
    assert.equal(shouldMountMeshViewer(
      { nodeType: "action", templateKey: "action-body-fit", meshObjUrl: "b.obj" }), true);
    assert.equal(shouldMountMeshViewer(
      { nodeType: "asset", assetKind: "mesh", meshObjUrl: "b.obj" }), true);
  });

  it("declines cards without a mesh ref or of the wrong kind", () => {
    assert.equal(shouldMountMeshViewer(
      { nodeType: "action", templateKey: "action-export" }), false);
    assert.equal(shouldMountMeshViewer(
      { nodeType: "action", templateKey: "action-meta-sam", glbUrl: "a.glb" }), false);
    assert.equal(shouldMountMeshViewer(
      { nodeType: "asset", assetKind: "image", meshObjUrl: "b.obj" }), false);
    assert.equal(shouldMountMeshViewer(
      { nodeType: "asset", assetKind: "mesh" }), false);
  });
});

describe("renderMeshStageHtml", () => {
  it("emits the stage with format, poster and loading status", () => {
    const s = renderMeshStageHtml({ glbUrl: "/api/x/avatar_character.glb" }, "/api/x/sheet.jpg");
    assert.ok(s.includes('data-mesh-stage="1"'));
    assert.ok(s.includes('data-mesh-format="glb"'));
    assert.ok(s.includes('data-mesh-poster="/api/x/sheet.jpg"'));
    assert.ok(s.includes("Loading 3D GLB…"));
  });

  it("escapes urls and returns empty without a mesh ref", () => {
    const s = renderMeshStageHtml({ meshObjUrl: '"><script>' }, null);
    assert.ok(!s.includes("<script>"));
    assert.equal(renderMeshStageHtml({}), "");
  });
});

describe("meshFailureReason", () => {
  it("maps failure classes to human reasons", () => {
    assert.equal(meshFailureReason(new Error("WebGL context lost")), "WebGL unavailable in this browser");
    assert.equal(meshFailureReason(new Error("404 Not Found")), "model file not found (404)");
    assert.equal(meshFailureReason(new Error("Failed to fetch")), "model fetch failed");
    assert.equal(meshFailureReason(new Error("Cannot import three")), "3D library failed to load");
    assert.equal(meshFailureReason(null), "unknown 3D error");
    assert.ok(meshFailureReason(new Error("x".repeat(500))).length <= 120);
  });
});

describe("disposeMeshViewer", () => {
  it("disposes, unregisters and invalidates in-flight mounts", () => {
    let disposed = 0;
    const app = { _meshViewers: new Map([["n1", { dispose: () => { disposed++; } }]]), _meshGen: {} };
    disposeMeshViewer(app, "n1");
    assert.equal(disposed, 1);
    assert.equal(app._meshViewers.has("n1"), false);
    assert.equal(app._meshGen.n1, 1);
    disposeMeshViewer(app, "n1");
    assert.equal(app._meshGen.n1, 2);
  });

  it("survives missing state and throwing disposers", () => {
    const app = {};
    disposeMeshViewer(app, "zz");
    assert.equal(app._meshGen.zz, 1);
    app._meshViewers = new Map([["zz", { dispose: () => { throw new Error("gone"); } }]]);
    disposeMeshViewer(app, "zz");
    assert.equal(app._meshViewers.has("zz"), false);
  });
});

describe("bindMeshViewer", () => {
  it("is a node-safe no-op that still disposes (no window, no mount)", () => {
    const app = { _meshGen: {} };
    bindMeshViewer(app, { id: "n", glbUrl: "a.glb" }, null);
    assert.equal(app._meshGen.n, 1);
    bindMeshViewer(app, { id: "n" }, { querySelector: () => null });
    assert.equal(app._meshGen.n, 2);
  });
});

describe("renderResultPreviewHtml mesh stages", () => {
  it("mounts a GLB stage on export results (no poster)", () => {
    const s = renderResultPreviewHtml({
      id: "e", nodeType: "action", templateKey: "action-export", title: "Export",
      resultText: "avatar_character.glb · 18439 verts", glbUrl: "/api/x/avatar_character.glb",
      runLog: [{ t: 1, level: "info", msg: "Passed" }],
      outputs: [{ id: "out-glb" }],
    });
    assert.ok(s.includes('data-mesh-format="glb"'));
    assert.ok(!s.includes("data-mesh-poster"));
  });

  it("mounts an OBJ stage below the body-fit sheet (sheet as poster)", () => {
    const s = renderResultPreviewHtml({
      id: "f", nodeType: "action", templateKey: "action-body-fit", title: "Fit",
      resultImage: "/api/x/sheet.jpg", resultImageBadge: "sheet",
      resultText: "fit ok", meshObjUrl: "/api/x/mesh_refined.obj",
      outputs: [{ id: "out-mesh" }],
    });
    assert.ok(s.includes('data-mesh-format="obj"'));
    assert.ok(s.includes('data-mesh-poster="/api/x/sheet.jpg"'));
    assert.ok(s.indexOf("flow-node-img") < s.indexOf("data-mesh-stage"));
  });
});

describe("vendor/importmap contract", () => {
  const VENDOR = [
    "vendor/three/three.module.js",
    "vendor/three/three.core.js",
    "vendor/three/addons/controls/OrbitControls.js",
    "vendor/three/addons/loaders/GLTFLoader.js",
    "vendor/three/addons/loaders/OBJLoader.js",
    "vendor/three/addons/environments/RoomEnvironment.js",
    "vendor/three/addons/utils/BufferGeometryUtils.js",
    "vendor/three/addons/utils/SkeletonUtils.js",
    "vendor/three/LICENSE",
  ];

  it("ships every file the viewer imports, non-empty", () => {
    for (const rel of VENDOR) {
      assert.ok(statSync(join(ROOT, rel)).size > 0, `${rel} missing or empty`);
    }
  });

  it("vendors the pinned three.js revision", () => {
    const core = readFileSync(join(ROOT, "vendor/three/three.core.js"), "utf8");
    assert.ok(core.includes("const REVISION = '186'"), "three.core.js is not r186");
    assert.equal(THREE_VERSION, "0.186.1");
  });

  it("importmap targets resolve to vendored files", () => {
    const html = readFileSync(join(ROOT, "index.html"), "utf8");
    const map = JSON.parse(/<script type="importmap">([\s\S]*?)<\/script>/.exec(html)[1]);
    assert.equal(map.imports.three, "./vendor/three/three.module.js");
    assert.ok(statSync(join(ROOT, map.imports.three)).size > 0);
    const prefix = map.imports["three/addons/"];
    for (const mod of ["controls/OrbitControls.js", "loaders/GLTFLoader.js",
        "loaders/OBJLoader.js", "environments/RoomEnvironment.js"]) {
      assert.ok(statSync(join(ROOT, prefix, mod)).size > 0, `${prefix}${mod} missing`);
    }
  });
});

describe("meshDownloadName", () => {
  it("keeps honest model tails, strips queries, falls back by format", () => {
    assert.equal(meshDownloadName("/api/chats/c/file/debug/d/avatar_character.glb", "glb"),
      "avatar_character.glb");
    assert.equal(meshDownloadName("https://x.test/m.A0_-b.obj?token=1", "obj"), "m.A0_-b.obj");
    assert.equal(meshDownloadName("https://x.test/nameless", "glb"), "model.glb");
    assert.equal(meshDownloadName("https://x.test/nameless", "obj"), "model.obj");
    assert.equal(meshDownloadName(null, "glb"), "model.glb");
  });

  it("rejects path tricks and non-model tails", () => {
    assert.equal(meshDownloadName("/api/x/%2e%2e%2f.glb", "glb"), "model.glb");
    assert.equal(meshDownloadName("/api/x/report.pdf", "glb"), "model.glb");
    assert.equal(meshDownloadName("/api/x/.glb", "glb"), "model.glb");
  });
});

describe("renderMeshStageHtml download", () => {
  it("offers a same-origin download pill per format", () => {
    const glb = renderMeshStageHtml({ glbUrl: "/api/x/a.glb" }, null);
    assert.match(glb, /class="flow-mesh-download"/);
    assert.match(glb, /href="\/api\/x\/a\.glb"/);
    assert.match(glb, /download="a\.glb"/);
    assert.match(glb, /⬇ GLB/);
    const obj = renderMeshStageHtml({ meshObjUrl: "/api/x/b.obj" }, null);
    assert.match(obj, /download="b\.obj"/);
    assert.match(obj, /⬇ OBJ/);
    assert.equal(renderMeshStageHtml({}, null), "");
  });
});

describe("stretchDepthPixels", () => {
  it("stretches the red channel over the full range, bg stays black", () => {
    const src = new Uint8ClampedArray([0, 0, 0, 255, 50, 0, 0, 255, 100, 0, 0, 255, 200, 0, 0, 255]);
    const out = stretchDepthPixels(src);
    assert.deepEqual([...out],
      [0, 0, 0, 255, 64, 64, 64, 255, 128, 128, 128, 255, 255, 255, 255, 255]);
  });

  it("leaves an all-black readback black", () => {
    const src = new Uint8ClampedArray(8);
    assert.deepEqual([...stretchDepthPixels(src)], [...src]);
  });
});

describe("renderMeshStill", () => {
  it("fails loud without a browser page (no silent null)", async () => {
    await assert.rejects(() => renderMeshStill("x.glb", "glb", "shaded"),
      /needs a browser page/);
  });

  it("renders small enough to persist (localStorage budget)", () => {
    assert.ok(MESH_OFFSCREEN_SIZE <= 512, `still is ${MESH_OFFSCREEN_SIZE}px`);
  });
});

describe("viewDirForAzimuth", () => {
  const approx = (a, b) => assert.ok(Math.abs(a - b) < 1e-9, `${a} ≈ ${b}`);

  it("starts front (+Z) at stage elevation and steps around Y", () => {
    const [x0, y0, z0] = viewDirForAzimuth(0);
    approx(x0, 0);
    approx(y0, Math.sin(14.7 * Math.PI / 180));
    approx(z0, Math.cos(14.7 * Math.PI / 180));
    const [x90, , z90] = viewDirForAzimuth(90);
    approx(x90, Math.cos(14.7 * Math.PI / 180));
    approx(z90, 0);
    const [, , z180] = viewDirForAzimuth(180);
    approx(z180, -Math.cos(14.7 * Math.PI / 180));
  });

  it("always returns a unit vector", () => {
    for (const az of [0, 33, 90, 200, 359]) {
      const [x, y, z] = viewDirForAzimuth(az);
      approx(x * x + y * y + z * z, 1);
    }
  });
});

describe("multiviewAzimuths", () => {
  it("spaces views evenly from the front", () => {
    assert.deepEqual(multiviewAzimuths(2), [0, 180]);
    assert.deepEqual(multiviewAzimuths(4), [0, 90, 180, 270]);
    assert.deepEqual(multiviewAzimuths(8), [0, 45, 90, 135, 180, 225, 270, 315]);
  });

  it("rejects junk counts", () => {
    assert.deepEqual(multiviewAzimuths(0), []);
    assert.deepEqual(multiviewAzimuths(-3), []);
    assert.deepEqual(multiviewAzimuths("x"), []);
  });
});

describe("multiviewGrid", () => {
  it("packs supported counts into compact sheets", () => {
    assert.deepEqual(multiviewGrid(2), [2, 1]);
    assert.deepEqual(multiviewGrid(4), [2, 2]);
    assert.deepEqual(multiviewGrid(6), [3, 2]);
    assert.deepEqual(multiviewGrid(8), [4, 2]);
    assert.deepEqual(multiviewGrid(5), [2, 2]);
  });
});

describe("composeMultiviewSheet", () => {
  it("fails loud without a browser page", async () => {
    await assert.rejects(() => composeMultiviewSheet([{ dataUrl: "x", label: "0°" }]),
      /browser page/);
  });
});
