// Regression tests for body-fit / export result fills and mesh refs (no DOM needed).
// Run from the flow-viewer folder:  node tests/bodyfit.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { fillBodyFitResultView, fillExportResultView, upstreamMeshUrl, spawnAvatarAsset } from "../js/execution.js";

describe("fillBodyFitResultView", () => {
  const stats = { verts: 17821, faces: 35640, beta_n: 10, refine_used: "joint" };

  it("fills the sheet render, mesh/beta badges and the mesh ref", () => {
    const node = { id: "b", outputs: [
      { id: "out-mesh", badge: "—" }, { id: "out-beta", badge: "—" },
    ] };
    fillBodyFitResultView(node, {
      text: "SAM 3D Body fit (joint).", imgUrl: "sheet.jpg",
      badge: "joint · 17821 verts", demo: false, stats,
      meshUrl: "/api/chats/c/file/debug/0/mesh_refined.obj",
    });
    assert.equal(node.resultText, "SAM 3D Body fit (joint).");
    assert.equal(node.resultImage, "sheet.jpg");
    assert.equal(node.resultImageBadge, "joint · 17821 verts");
    assert.deepEqual(node.outputs.map(o => o.badge), ["17821 verts", "10 coeffs"]);
    assert.equal(node.meshObjUrl, "/api/chats/c/file/debug/0/mesh_refined.obj");
    assert.match(node.resultImageFallback, /^data:image\/svg/);
  });

  it("stores the fit ref live, drops it on demo", () => {
    const live = { id: "b", outputs: [] };
    fillBodyFitResultView(live, {
      text: "fit", imgUrl: "s.jpg", badge: "b", demo: false,
      stats: { verts: 1, beta_n: 2, focal_px: 900.5, size_wh: [100, 80] },
      meshUrl: "m.obj", debugId: "d9", chatId: "flow-session",
    });
    assert.deepEqual(live.fitCamera,
      { debug_id: "d9", chat_id: "flow-session", focal_px: 900.5, size_wh: [100, 80] });
    const demo = { id: "b", outputs: [] };
    fillBodyFitResultView(demo, {
      text: "demo", imgUrl: "p.jpg", badge: "demo", demo: true,
      stats: null, meshUrl: null, debugId: "d9", chatId: "flow-session",
    });
    assert.equal(demo.fitCamera, null);
  });

  it("flags the offline demo and keeps no mesh ref", () => {
    const node = { id: "b", outputs: [
      { id: "out-mesh", badge: "—" }, { id: "out-beta", badge: "—" },
    ] };
    fillBodyFitResultView(node, {
      text: "Demo body-fit (offline).", imgUrl: "photo.jpg",
      badge: "demo", demo: true, stats: null, meshUrl: null,
    });
    assert.equal(node.meshObjUrl, null);
    assert.match(decodeURIComponent(node.resultImageFallback), /Demo/);
    assert.deepEqual(node.outputs.map(o => o.badge), ["—", "—"]);
  });
});

describe("fillExportResultView", () => {
  it("fills the GLB summary, badge and ref (text-only preview)", () => {
    const node = { id: "e", outputs: [{ id: "out-glb", badge: "—" }] };
    fillExportResultView(node, {
      text: "avatar_character.glb: 17,821 verts.",
      badge: "28.4 MB", demo: false,
      stats: { verts: 17821, glb_bytes: 29790208 },
      glbUrl: "/api/chats/c/file/debug/1/avatar_character.glb",
    });
    assert.equal(node.resultImage, null);
    assert.equal(node.resultText, "avatar_character.glb: 17,821 verts.");
    assert.equal(node.outputs[0].badge, "28.4 MB");
    assert.equal(node.glbUrl, "/api/chats/c/file/debug/1/avatar_character.glb");
  });

  it("flags the offline demo", () => {
    const node = { id: "e", outputs: [{ id: "out-glb", badge: "—" }] };
    fillExportResultView(node, {
      text: "Demo export (offline).", badge: "demo", demo: true, stats: null, glbUrl: null,
    });
    assert.equal(node.glbUrl, null);
    assert.equal(node.outputs[0].badge, "demo");
    assert.match(decodeURIComponent(node.resultImageFallback), /Demo/);
  });
});

describe("upstreamMeshUrl", () => {
  it("reads the fitted mesh ref off the upstream card", () => {
    const fit = { id: "fit", nodeType: "action",
      meshObjUrl: "/api/chats/c/file/debug/0/mesh_refined.obj",
      outputs: [{ id: "out-mesh", name: "3D Parametric Mesh" }] };
    const app = { nodes: [fit], edges: [] };
    assert.equal(upstreamMeshUrl(app, { from: "fit", fromPort: "out-mesh" }),
      "/api/chats/c/file/debug/0/mesh_refined.obj");
    assert.equal(upstreamMeshUrl(app, { from: "missing", fromPort: "out-mesh" }), null);
    assert.equal(upstreamMeshUrl({ nodes: [{ id: "fit", outputs: [] }], edges: [] },
      { from: "fit", fromPort: "out-mesh" }), null);
  });
});

describe("spawnAvatarAsset", () => {
  const appFor = () => ({ nodes: [
    { id: "fit", nodeType: "action", x: 10, y: 20,
      outputs: [{ id: "out-mesh", type: "mesh" }, { id: "out-beta", type: "vector" }] },
  ], edges: [], activeRunId: "r1" });

  it("spawns a mesh asset wired from the fit, carrying the turntable and mesh ref", () => {
    const app = appFor();
    const id = spawnAvatarAsset(app, app.nodes[0], {
      imgUrl: "/api/chats/c/file/debug/0/turntable.jpg",
      badge: "18439 verts · turntable",
      tensorShape: "[18439, 3] vertices + β[45]",
      meshUrl: "/api/chats/c/file/debug/0/mesh_refined.obj",
    });
    const asset = app.nodes.find(n => n.id === id);
    assert.equal(asset.nodeType, "asset");
    assert.equal(asset.assetKind, "mesh");
    assert.equal(asset.title, "Asset: Parametric 3D Avatar");
    assert.equal(asset.imgUrl, "/api/chats/c/file/debug/0/turntable.jpg");
    assert.match(asset.imgFallback, /^data:image\/svg/);
    assert.equal(asset.meshObjUrl, "/api/chats/c/file/debug/0/mesh_refined.obj");
    assert.deepEqual(asset.outputs.map(o => o.type), ["mesh"]);
    assert.deepEqual(app.edges.length, 1);
    assert.deepEqual([app.edges[0].from, app.edges[0].fromPort, app.edges[0].toPort],
      ["fit", "out-mesh", "in-mesh"]);
    // The exporter chain resolves through the spawned asset.
    assert.equal(upstreamMeshUrl(app, { from: id, fromPort: "out-mesh" }),
      "/api/chats/c/file/debug/0/mesh_refined.obj");
  });

  it("passes the fit ref through to the avatar (overlay wires from either)", () => {
    const app = appFor();
    const fit = { debug_id: "d9", chat_id: "flow-session", focal_px: 1, size_wh: [2, 2] };
    const id = spawnAvatarAsset(app, app.nodes[0], {
      imgUrl: "t.jpg", badge: "b", tensorShape: "s", meshUrl: "m.obj", fitCamera: fit,
    });
    assert.deepEqual(app.nodes.find(n => n.id === id).fitCamera, fit);
  });
});
