/**
 * ReUI Flow-2 Modular Node Engine (Main Application Entry)
 * Orchestrates modules for catalog, canvas, wires, context-menu, popovers, execution, and inspector.
 */

import { MODEL_CATALOG } from "./catalog.js";
import { 
  initCanvasPanZoom, setZoom, fitView, updateTransform, 
  initDragAndDrop, handlePhotoDropped, replacePhotoOnNode, renderEmptyDropzone, tidyLayout 
} from "./canvas.js";
import {
  renderNodes, createNodeElement, updateNodeElement,
  renderBlueprintPinsHtml, renderOverlaySvg, updateOverlaySvg, updateHeaderStats,
  assetViewBadge
} from "./node-renderer.js";
import { 
  attachBlueprintPinListeners, startWireDrag, connectNodesWithPort, 
  updateConnectedEdges, updateSingleEdge, renderEdges 
} from "./wires.js";
import {
  openAddStepPopoverForPort, addModelFromPort, closePopover
} from "./popover.js";
import { 
  initContextMenu, openContextMenu, closeContextMenu, 
  spawnModelAt, duplicateNode 
} from "./context-menu.js";
import { 
  runCard, clearDownstreamOutputs, executeRealSamDetection, 
  produceOutputsFromRealSam, produceOutputsForAction, 
  deleteCard, runNextAction, runAllSteps, splitListPin, collapseListPin
} from "./execution.js";
import { selectNode, renderInspectorContent } from "./inspector.js";
import { svgPlaceholder, probeImageSize, fetchJson, resolveAssetImageUrl, wiredViewLabel, escapeHtml } from "./util.js";
import { serializeFlow, inlineBlobs, loadFlowList, storeFlowList, formatFlowTime, formatFlowDate } from "./persist.js";
import { loadPoseTopology } from "./overlays.js";
import { initLightbox } from "./lightbox.js";
import { bindNodesSidebar } from "./gallery.js";

export class DynamicFlowApp {
  constructor() {
    this.scale = 0.75;
    this.panX = 220;
    this.panY = 40;
    this.isDraggingCanvas = false;
    this.canvasDragStart = { x: 0, y: 0 };
    
    // Active graph state: lists of live nodes and live edges on the canvas
    this.nodes = [];
    this.edges = [];
    this.customPositions = {};
    
    this.activeRunId = "20261103-213035-3359";
    this.selectedNodeId = null;
    this.activeTab = "overview";
    this.userPhoto = null; // { url, name, size }
    this.openPopoverSourceId = null;
    this.wireDragFinishedAt = 0;
    this.poseTopology = null;
    // Set by the boot health check: loud failure vs quiet demo on fetch loss.
    // (Start the server after the page loaded? Reload — presets bind at boot.)
    this.backendOnline = false;

    this.initDOM();
    initLightbox();
    initCanvasPanZoom(this);
    initDragAndDrop(this);
    initContextMenu(this);
    initUnloadGuard(this);
    this.renderCanvas();
    this.checkHealthAndRuns();
    document.title = "NeoHuman flow · build 0348 · " + document.title;
    this.startBackendWatch();
    // Pose skeleton links arrive with the topology file; pose cards painted
    // before that show dots only and pick up links on this re-render.
    loadPoseTopology().then(t => {
      this.poseTopology = t;
      if (t && (this.nodes || []).some(n => Array.isArray(n.resultKeypoints) && n.resultKeypoints.length)) {
        this.renderCanvas();
      }
    });
  }

  // Schema Catalog
  getModelCatalog() {
    return MODEL_CATALOG;
  }

  initDOM() {
    this.plane = document.getElementById("canvas-plane");
    this.viewport = document.getElementById("canvas-viewport");
    this.svgLayer = document.getElementById("svg-layer");
    this.inspector = document.getElementById("docked-inspector");
    this.runSelector = document.getElementById("run-selector");
    this.statPill = document.getElementById("stat-pill");
    this.dropOverlay = document.getElementById("viewport-drop-overlay");
    this.fileInput = document.getElementById("file-photo-input");

    // Canvas view controls
    document.getElementById("btn-fit-view").addEventListener("click", () => this.fitView());
    document.getElementById("btn-zoom-in").addEventListener("click", () => this.setZoom(this.scale + 0.12));
    document.getElementById("btn-zoom-out").addEventListener("click", () => this.setZoom(this.scale - 0.12));
    
    // Inspector toggle
    document.getElementById("btn-close-inspector").addEventListener("click", () => {
      this.inspector.classList.toggle("collapsed");
    });

    // Flow controls
    document.getElementById("btn-new-flow").addEventListener("click", () => this.startNewFlow());
    document.getElementById("btn-tidy-layout").addEventListener("click", () => this.tidyLayout());
    const floatingTidy = document.getElementById("btn-floating-tidy");
    if (floatingTidy) {
      floatingTidy.addEventListener("click", () => this.tidyLayout());
    }

    document.getElementById("btn-reset-flow").addEventListener("click", () => this.startNewFlow());
    
    // Auto traverse button runs all runnable nodes sequentially
    document.getElementById("btn-auto-traverse").addEventListener("click", () => this.runAllSteps());

    // Step forward runs the next unrun action
    document.getElementById("btn-step-forward").addEventListener("click", () => this.runNextAction());

    // Replace the input photo without touching the graph wiring.
    const inputPhotoBtn = document.getElementById("btn-load-input-photo");
    if (inputPhotoBtn) inputPhotoBtn.addEventListener("click", () => {
      const photo = (this.nodes || []).find(n => n.id === "asset-photo")
        || (this.nodes || []).find(n => n.nodeType === "asset" && n.assetKind === "image");
      this.replacePhotoTargetId = photo ? photo.id : null;
      if (this.fileInput) this.fileInput.click();
    });

    // Server graph save/load (also the prepared demo graph)
    const saveGraphBtn = document.getElementById("btn-save-graph");
    if (saveGraphBtn) saveGraphBtn.addEventListener("click", () => this.saveGraphToDisk());
    const loadGraphBtn = document.getElementById("btn-load-graph");
    if (loadGraphBtn) loadGraphBtn.addEventListener("click", () => this.loadGraphFromDisk());

    // Saved flows panel
    document.getElementById("btn-flows").addEventListener("click", (e) => {
      e.stopPropagation();
      this.toggleFlowsPanel();
    });

    // Node sidebar: the full card + type library, docked left
    document.getElementById("btn-nodes").addEventListener("click", (e) => {
      e.stopPropagation();
      this.toggleNodesPanel();
    });
    document.getElementById("btn-close-nodes").addEventListener("click", () => {
      this.toggleNodesPanel();
    });
    bindNodesSidebar(this, document.getElementById("nodes-sidebar"));

    // Inspector tabs
    document.querySelectorAll(".inspector-tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".inspector-tab-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        this.activeTab = btn.dataset.tab;
        this.renderInspectorContent();
      });
    });

    // Run selector: a pristine preset graph reloads from the new run (the
    // photo card repoints at its source artifact); a graph the user built
    // or ran keeps its cards and only future productions use the new id.
    this.runSelector.addEventListener("change", (e) => {
      const next = e.target.value;
      if (isPristinePresetFlow(this)) {
        this.loadPreset(next);
      } else {
        this.activeRunId = next;
        this.renderCanvas();
      }
    });

    // File input change: an in-card Replace targets its own card (graph
    // survives); anything else starts a fresh graph from the dropped photo.
    if (this.fileInput) {
      this.fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
          const targetId = this.replacePhotoTargetId;
          this.replacePhotoTargetId = null;
          if (targetId) replacePhotoOnNode(this, targetId, e.target.files[0]);
          else handlePhotoDropped(this, e.target.files[0]);
        }
        e.target.value = "";
      });
    }

    // Dismiss popovers & context menu on outside click
    window.addEventListener("click", (e) => {
      if (Date.now() - (this.wireDragFinishedAt || 0) < 350) {
        return; // Suppress spurious click immediately following drag mouseup
      }
      if (!e.target.closest(".flow-add-popover") && !e.target.closest(".flow-context-menu") && !e.target.closest(".flow-saves-panel")) {
        this.closePopover();
        this.closeContextMenu();
        this.closeFlowsPanel();
      }
    });
  }

  // Canvas & Layout delegates
  setZoom(nextScale) { setZoom(this, nextScale); }
  fitView() { fitView(this); }
  updateTransform() { updateTransform(this); }
  tidyLayout() { tidyLayout(this); }

  // Rendering
  renderCanvas() {
    if (this.nodes.length === 0) {
      renderEmptyDropzone(this);
      updateHeaderStats(this);
      return;
    }

    const dropzone = this.plane.querySelector(".flow-empty-dropzone");
    if (dropzone) dropzone.remove();

    renderNodes(this);
    renderEdges(this);
    updateHeaderStats(this);

    if (this.selectedNodeId) {
      selectNode(this, this.selectedNodeId);
    }
  }

  // Wires & Pins delegates
  attachBlueprintPinListeners(node, el) { attachBlueprintPinListeners(this, node, el); }
  updateConnectedEdges(nodeId) { updateConnectedEdges(this, nodeId); }

  // Popover & Context Menu delegates
  openAddStepPopoverForPort(fromNodeId, fromPort, canvasX, canvasY) { openAddStepPopoverForPort(this, fromNodeId, fromPort, canvasX, canvasY); }
  closePopover() { closePopover(this); }
  closeContextMenu() { closeContextMenu(); }
  openContextMenu(screenX, screenY, canvasX, canvasY, targetNodeId) { openContextMenu(this, screenX, screenY, canvasX, canvasY, targetNodeId); }

  // Execution delegates
  runCard(nodeId) { runCard(this, nodeId); }
  deleteCard(nodeId) { deleteCard(this, nodeId); }
  runNextAction() { runNextAction(this); }
  runAllSteps() { runAllSteps(this); }
  splitListPin(nodeId, portId) { splitListPin(this, nodeId, portId); }
  collapseListPin(nodeId, portId) { collapseListPin(this, nodeId, portId); }

  // Inspector delegates
  selectNode(nodeId) { selectNode(this, nodeId); }
  renderInspectorContent() { renderInspectorContent(this); }

  startNewFlow() {
    this.revokeUserPhoto();
    this.nodes = [];
    this.edges = [];
    this.customPositions = {};
    this.userPhoto = null;
    this.selectedNodeId = null;
    this.closePopover();
    this.closeContextMenu();
    this.panX = 140;
    this.panY = 80;
    this.scale = 0.75;
    this.updateTransform();
    this.renderCanvas();
  }

  revokeUserPhoto() {
    if (this.userPhoto && this.userPhoto.url && this.userPhoto.url.startsWith("blob:")) {
      URL.revokeObjectURL(this.userPhoto.url);
    }
  }

  // ---- Saved flows (localStorage) ----
  toggleFlowsPanel() {
    const existing = document.getElementById("flow-saves-panel");
    if (existing) {
      existing.remove();
      return;
    }
    const panel = document.createElement("div");
    panel.id = "flow-saves-panel";
    panel.className = "flow-saves-panel";
    panel.innerHTML = `
      <div class="flow-saves-head">
        <span>Saved flows</span>
        <button type="button" class="flow-saves-close" title="Close">✕</button>
      </div>
      <div class="flow-saves-save-row">
        <input type="text" class="flow-saves-name" placeholder="Flow name" maxlength="60">
        <button type="button" class="reui-btn sm primary flow-saves-save-btn">Save current</button>
      </div>
      <div class="flow-saves-msg"></div>
      <div class="flow-saves-list"></div>
    `;
    document.body.appendChild(panel);
    panel.querySelector(".flow-saves-close").addEventListener("click", () => this.closeFlowsPanel());
    panel.querySelector(".flow-saves-save-btn").addEventListener("click", () => {
      this.saveFlowAs(panel.querySelector(".flow-saves-name").value.trim());
    });
    panel.querySelector(".flow-saves-name").addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.saveFlowAs(e.target.value.trim());
      e.stopPropagation();
    });
    this.refreshFlowsList();
  }

  closeFlowsPanel() {
    document.getElementById("flow-saves-panel")?.remove();
  }

  toggleNodesPanel() {
    document.getElementById("nodes-sidebar")?.classList.toggle("collapsed");
  }

  flowsPanelMessage(text, isError = false) {
    const msg = document.querySelector("#flow-saves-panel .flow-saves-msg");
    if (msg) {
      msg.textContent = text;
      msg.classList.toggle("is-error", isError);
    }
  }

  refreshFlowsList() {
    const listEl = document.querySelector("#flow-saves-panel .flow-saves-list");
    if (!listEl) return;
    const flows = loadFlowList();
    if (flows.length === 0) {
      listEl.innerHTML = `<div class="flow-saves-empty">No saved flows yet.</div>`;
      return;
    }
    listEl.innerHTML = flows.map(f => `
      <div class="flow-saves-row" data-flow-id="${escapeHtml(f.id)}">
        <div class="flow-saves-row-info">
          <div class="flow-saves-row-name">${escapeHtml(f.name)}</div>
          <div class="flow-saves-row-sub">${escapeHtml(formatFlowDate(f.savedAt))} · ${escapeHtml(formatFlowTime(f.savedAt))} · ${f.nodes.length} cards</div>
        </div>
        <button type="button" class="reui-btn sm flow-saves-load">Open</button>
        <button type="button" class="reui-btn sm flow-saves-del" title="Delete">✕</button>
      </div>
    `).join("");
    listEl.querySelectorAll(".flow-saves-load").forEach(btn => {
      btn.addEventListener("click", () => {
        this.loadFlowById(btn.closest(".flow-saves-row").dataset.flowId);
      });
    });
    listEl.querySelectorAll(".flow-saves-del").forEach(btn => {
      btn.addEventListener("click", () => {
        const id = btn.closest(".flow-saves-row").dataset.flowId;
        storeFlowList(loadFlowList().filter(f => f.id !== id));
        this.refreshFlowsList();
      });
    });
  }

  async saveFlowAs(name) {
    const flows = loadFlowList();
    const snap = serializeFlow(this, name || `Flow ${flows.length + 1}`);
    try {
      await inlineBlobs(snap.nodes);
    } catch (err) {
      this.flowsPanelMessage(`Could not read a photo for saving (${err?.message || err}).`, true);
      return;
    }
    flows.unshift(snap);
    const res = storeFlowList(flows);
    if (!res.ok) {
      this.flowsPanelMessage(res.error === "quota"
        ? "Save failed: browser storage full — delete old flows to make room."
        : `Save failed: ${res.error}`, true);
      return;
    }
    this.flowsPanelMessage(`Saved “${snap.name}”.`);
    this.refreshFlowsList();
  }

  loadFlowById(id) {
    const flow = loadFlowList().find(f => f.id === id);
    if (!flow) return;
    this.revokeUserPhoto();
    this.userPhoto = null;
    this.nodes = JSON.parse(JSON.stringify(flow.nodes));
    this.edges = JSON.parse(JSON.stringify(flow.edges));
    this.customPositions = JSON.parse(JSON.stringify(flow.customPositions || {}));
    this.activeRunId = flow.activeRunId ?? this.activeRunId;
    if (this.runSelector && [...this.runSelector.options].some(o => o.value === this.activeRunId)) {
      this.runSelector.value = this.activeRunId;
    }
    this.selectedNodeId = (flow.selectedNodeId && this.nodes.some(n => n.id === flow.selectedNodeId))
      ? flow.selectedNodeId
      : (this.nodes[0]?.id ?? null);
    this.closePopover();
    this.closeContextMenu();
    this.closeFlowsPanel();
    this.renderCanvas();
  }

  /** Fill in real photo dimensions once the image loads; no-op when offline. */
  resolvePhotoDims(nodeId, expectedSrc) {
    this.resolveAssetDims(nodeId, expectedSrc, true);
  }

  /**
   * Snap an asset card to its image's true aspect ratio once loaded.
   * Probes the effective view (a wired input wins over the card's own photo);
   * photo cards also get their badge/tensor shape, wired views show "← pin".
   */
  resolveAssetDims(nodeId, expectedSrc, updateBadge = false) {
    const node = this.nodes.find(n => n.id === nodeId);
    if (!node) return;
    const src = resolveAssetImageUrl(this, node);
    if (!src) return;
    probeImageSize(src).then(dims => {
      const live = this.nodes.find(n => n.id === nodeId);
      if (!live || !dims || !dims.w || !dims.h) return;
      if (resolveAssetImageUrl(this, live) !== expectedSrc) return;
      live.imageWidth = dims.w;
      live.imageHeight = dims.h;
      const wrap = document.querySelector(`#${CSS.escape(nodeId)} .flow-node-img-wrapper`);
      if (wrap) wrap.style.aspectRatio = `${dims.w} / ${dims.h}`;
      if (updateBadge) {
        if (!wiredViewLabel(this, live)) {
          live.tensorShape = `[${dims.h}, ${dims.w}, 3] uint8`;
          live.imgBadge = `${live.subtitle} · ${dims.w}x${dims.h}`;
        }
        const badgeEl = document.querySelector(`#${CSS.escape(nodeId)} .flow-node-img-badge`);
        if (badgeEl) badgeEl.textContent = assetViewBadge(this, live);
      }
      this.updateConnectedEdges(nodeId);
      this.renderInspectorContent();
    });
  }

  loadPreset(runId, photoName = "source.jpg") {
    this.activeRunId = runId;
    this.runSelector.value = runId;
    this.revokeUserPhoto();
    this.userPhoto = null;
    this.nodes = [];
    this.edges = [];
    this.customPositions = {};
    this.closeContextMenu();
    this.closePopover();

    // 1. Asset: Source Photo
    const photoNode = {
      id: "asset-photo",
      nodeType: "asset",
      assetKind: "image",
      title: "Asset: Source Photo",
      subtitle: photoName,
      category: "Image Asset",
      badgeVariant: "success-light",
      iconVariant: "outline",
      iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>`,
      imgUrl: `/api/runs/${runId}/artifact/source`,
      imgFallback: svgPlaceholder(photoName),
      imgBadge: `${photoName} · …`,
      description: "High-resolution input photo received. Wire its Photo Output into any card from the sidebar or the right-click menu.",
      latency: "Ready",
      status: "Cached in RAM",
      tensorShape: "RGB image (? × ?)",
      format: "RGB Bitmap / JPEG",
      inputs: [
        { id: "in-view", name: "View", type: "image", color: "#10b981", badge: "View", accepts: ["image", "box", "mask", "crop"] }
      ],
      outputs: [
        { id: "out-photo", name: "Photo Output", type: "image", color: "#10b981", badge: "RGB" }
      ],
      x: 80,
      y: 140
    };

    // No default action: the preset is the photo alone, and the user picks
    // the first card from the sidebar or the right-click menu.
    this.nodes.push(photoNode);
    this.selectedNodeId = "asset-photo";
    this.renderCanvas();
    this.fitView();
    this.resolvePhotoDims(photoNode.id, photoNode.imgUrl);
  }

  /**
   * Optional preset: same stack + a Meta Muse Image card that turns the
   * textured 2x3 guide into a photoreal 6-view sheet, which VGGT then
   * validates (photoreal views match far better than clay/texture-fill).
   * One paid Muse call per run; use a photo of a private person (public
   * figures are rejected by the content policy).
   */
  loadMuseTurnaroundPreset(photoName = "source.jpg") {
    this.loadMetaAvatarPreset(photoName);
    const photo = this.nodes.find(n => n.id === "asset-photo");
    const tex = this.nodes.find(n => n.templateKey === "action-texture");
    const vggt = this.nodes.find(n => n.templateKey === "action-vggt-check");
    const tpl = MODEL_CATALOG["action-meta-muse"];
    if (!tpl) return;
    const muse = Object.assign(JSON.parse(JSON.stringify(tpl)), {
      id: "action-meta-muse-preset",
      templateKey: "action-meta-muse",
      x: 1480, y: 1320,
      status: "Ready to Run",
      createdAt: Date.now(),
      museSize: "1536x1024",
      param:
        "Image 1 is a flat-shaded 2x3 GRID guide of six views of the SAME person in a relaxed A-POSE (arms slightly away from the body, palms relaxed, legs straight, standing), in this order: row 1 = left profile (-90), 3/4 left (-45), front (0); row 2 = 3/4 right (+45), right profile (+90), back (180). Image 2 is the real person. " +
        "Make image 1 photorealistic as the SAME PERSON as image 2: same identity and face, same skin tone and texture, same hairstyle and hair colour, same clothing and colours in every panel. " +
        "COPY THE POSE OF IMAGE 1 EXACTLY: do NOT raise the arms, do NOT put hands on the head, do NOT change leg or torso position; the silhouette per panel must match image 1. " +
        "Keep the SAME 2x3 grid, panel order, boundaries, full-body framing and scale. " +
        "FLAT, EVEN, SHADOWLESS LIGHTING, plain light-grey background, sharp focus, photographic, no text, no labels, no numbers, no borders, no watermark.",
    });
    this.nodes.push(muse);
    const link = (fromNode, fromPortId, toNode, toPortId) => {
      if (!fromNode || !toNode) return;
      const pin = (fromNode.outputs || []).find(p => p.id === fromPortId);
      if (pin) connectNodesWithPort(this, fromNode.id, toNode.id, pin, toPortId || null);
    };
    // Two refs into the same images pin: the A-pose guide sheet + the identity photo.
    const apose = this.nodes.find(n => n.templateKey === "action-apose");
    link(photo, "out-photo", muse, "in-images");
    if (apose) link(apose, "out-sheet", muse, "in-images");
    else if (tex) link(tex, "out-sheet", muse, "in-images");
    // Re-route VGGT: drop the texture->vggt edge, feed it Muse instead.
    this.edges = (this.edges || []).filter(e => !(e.from === tex.id && e.to === vggt.id));
    link(muse, "out-image", vggt, "in-image");
    this.selectedNodeId = muse.id;
    this.renderCanvas();
    this.fitView();
  }

  /**
   * Demo preset: the "Meta avatar stack" - photo -> SAM 3D Body + joint solver (identity+expression
   * on the MHR base mesh) -> Canonical A-Pose -> VGGT multi-view check + character export.
   * Wires the cards so judges can press Run on the fit and then walk the graph left to right.
   */
  loadMetaAvatarPreset(photoName = "source.jpg") {
    this.loadPreset("20261103-213035-3359", photoName);
    const photo = this.nodes.find(n => n.id === "asset-photo");
    // Private-person demo photo (the news run is a bad default and Muse
    // refuses public figures). Drop your own onto the card to replace it.
    if (photo) {
      photo.imgUrl = "/api/chats/flow-session/file/debug/166/demo_photo.webp";
      photo.imgFallback = "";
      photo.subtitle = "demo_photo.webp";
      photo.imgBadge = "demo_photo.webp · private person";
    }
    // Staged ladder, left to right: baseline -> +Sapiens -> +expression -> FINAL.
    // Six labelled columns, left -> right, one horizontal story line.
    const spec = [
      ["action-sapiens-pose", 420, 300, "pose"],
      ["action-sapiens-seg", 420, 640, "seg"],
      ["action-sapiens-normal", 420, 980, "normal"],
      ["action-sapiens-depth", 420, 1320, "depth"],
      ["action-body-fit-raw", 840, 300, "raw"],
      ["action-body-fit", 840, 700, "neutral"],
      ["action-body-fit", 840, 1120, "expr"],
      ["action-texture", 1300, 340, "tex"],
      ["action-apose", 2020, 300, "apose"],
      ["action-meta-muse", 2020, 700, "muse6"],
      ["action-vggt-check", 2020, 1100, "vggt"],
      ["action-export", 2020, 1500, "exp"],
    ];
    const made = {};
    spec.forEach(([key, x, y, tag], i) => {
      const template = MODEL_CATALOG[key];
      if (!template) return;
      const node = Object.assign(JSON.parse(JSON.stringify(template)), {
        id: template.id + "-preset-" + tag,
        templateKey: key,
        x, y,
        status: template.nodeType === "action" ? "Ready to Run" : template.status,
        createdAt: Date.now() + i,
      });
      this.nodes.push(node);
      made[tag] = node;
    });
    const link = (fromNode, fromPortId, toNode, toPortId) => {
      if (!fromNode || !toNode) return;
      const pin = (fromNode.outputs || []).find(p => p.id === fromPortId);
      if (!pin) return;
      connectNodesWithPort(this, fromNode.id, toNode.id, pin, toPortId || null);
    };
    const { raw, neutral, expr, tex, apose, vggt, exp, pose, seg, normal, depth } = made;

    if (raw) raw.param = "off";
    if (neutral) {
      neutral.param = "off";
      neutral.title = "LOCAL: SAM 3D Body · + Sapiens2 (identity, neutral)";
      neutral.subtitle = "smpl_mhr_fitter · supervised identity, no expression";
    }
    if (expr) {
      expr.param = "on";
      expr.title = "LOCAL: SAM 3D Body · + Sapiens2 + Expression";
      expr.subtitle = "smpl_mhr_fitter · identity + 72 expression (jaw/lips)";
    }
    if (tex) {
      tex.bigCard = true;                 // FINAL result: 4x the area
      tex.title = "FINAL · Textured Avatar";
      tex.subtitle = "texture_bake · textured GLB + UV texture + 6-view sheet";
    }

    link(photo, "out-photo", raw, "in-image");
    link(photo, "out-photo", neutral, "in-image");
    link(photo, "out-photo", expr, "in-image");
    link(photo, "out-photo", pose, "in-image");
    link(photo, "out-photo", seg, "in-image");
    link(photo, "out-photo", normal, "in-image");
    link(photo, "out-photo", depth, "in-image");
    for (const fitNode of [neutral, expr]) {
      link(pose, "out-kpts", fitNode, "in-pose");
      link(seg, "out-skin", fitNode, "in-seg");
      link(normal, "out-normals", fitNode, "in-normal");
      link(depth, "out-depth", fitNode, "in-depth");
    }
    // Warm-start chain (raw -> neutral -> expr) is implemented in the backend and
    // the in-init pin exists, but it stays unwired until the init path is verified;
    // today each stage is an independent fit of the same photo.
    link(expr, "out-mesh", tex, "in-mesh");
    link(photo, "out-photo", tex, "in-photo");
    link(tex, "out-glb", exp, "in-mesh");
    link(tex, "out-sheet", vggt, "in-image");
    link(expr, "out-mesh", apose, "in-mesh");

    // ---- Meta API stages (muse-image-1.0): the more Meta, the better ----
    const museTpl = MODEL_CATALOG["action-meta-muse"];
    const addMuse = (tag, x, y, prompt, size, extra) => {
      if (!museTpl) return null;
      const n = Object.assign(JSON.parse(JSON.stringify(museTpl)), {
        id: "action-meta-muse-preset-" + tag,
        templateKey: "action-meta-muse",
        x, y,
        status: "Ready to Run",
        createdAt: Date.now() + 100 + this.nodes.length,
        param: prompt,
        museSize: size,
      }, extra || {});
      this.nodes.push(n);
      made[tag] = n;
      return n;
    };
    const hexRef = "Image 1 is the private person in the reference photo; keep their identity, face, skin tone, hair and clothing exactly as photographed. ";
    const TURN_PROMPT =
      "Image 1 is a flat-shaded 2x3 GRID guide of six views of a person in a relaxed A-POSE (arms slightly away from the body, palms relaxed, legs straight, standing), in this order: row 1 = left profile (-90), 3/4 left (-45), front (0); row 2 = 3/4 right (+45), right profile (+90), back (180). Image 2 is the real person. " +
      "Make image 1 photorealistic as the SAME PERSON as image 2: same identity and face, same skin tone and texture, same hairstyle and hair colour, same clothing and colours in every panel. " +
      "COPY THE POSE OF IMAGE 1 EXACTLY: do NOT raise the arms, do NOT put hands on the head, do NOT change leg or torso position; the silhouette per panel must match image 1. " +
      "Keep the SAME 2x3 grid, panel order, boundaries, full-body framing and scale. FLAT, EVEN, SHADOWLESS LIGHTING, plain light-grey background, sharp focus, photographic, no text, no labels, no numbers, no borders, no watermark.";
    const HAT_PROMPT = hexRef + "Edit the photo: completely REMOVE the hat from the person's head, so the hat is gone and the full hair/head underneath is visible exactly as it is (same hairstyle, hair colour, hairline, no hat shadow). Keep EVERYTHING else EXACTLY identical: same identity, same face and expression, same skin tone, same head pose and tilt, same clothing, same background, same camera distance, framing, lighting and image quality. No text, no labels, no borders.";
    const NEUTRAL_PROMPT = hexRef + "Edit the photo: change ONLY the person's facial expression to a neutral, relaxed expression (lips gently closed, no smile, no visible teeth, eyebrows relaxed, eyes open and looking at the camera). Keep EVERYTHING else EXACTLY identical: same identity and face shape, same head pose, tilt and angle, same hairstyle, same clothing, same background, same camera distance, framing, lighting and image quality. No text, no labels, no borders.";
    const BALD_PROMPT = hexRef + "Edit the photo: completely remove the person's hair and any headwear. Make them fully bald with a smooth, clean-shaven scalp and a natural hairline; no hair, no hair strands, no hair shadow on the forehead or scalp. Keep EVERYTHING else EXACTLY identical: same identity, face and features, skin tone, makeup, same facial expression, same head pose and tilt, same clothing, same background, same camera distance, framing, lighting and image quality. No text, no labels, no borders.";
    const SAD_PROMPT = hexRef + "Edit the photo: change ONLY the facial expression to a clearly sad, downcast expression (slightly downturned mouth, inner eyebrows raised, eyes looking down or away, no smile, no teeth). Keep EVERYTHING else EXACTLY identical: same identity and face shape, same head pose and tilt, same hairstyle and hair colour, same clothing, same background, same camera distance, framing, lighting and image quality. No text, no labels, no borders.";

    // The 6-view turnaround card is part of the main column (tag "muse6").
    const muse6 = made["muse6"];
    if (muse6) { muse6.param = TURN_PROMPT; muse6.museSize = "1536x1024"; }
    // Character-variant row along the bottom (Meta Image edits).
    const museHat = addMuse("musehat", 420, 1760, HAT_PROMPT, "1024x1024");
    const museNeutral = addMuse("museneutral", 840, 1760, NEUTRAL_PROMPT, "1024x1024");
    const museSad = addMuse("musesad", 1260, 1760, SAD_PROMPT, "1024x1024");
    // Chained multi-turn edit: the sad variant is the input to the bald edit.
    const museBald = addMuse("musebald", 1680, 1760, BALD_PROMPT, "1024x1024");
    if (muse6) {
      link(photo, "out-photo", muse6, "in-images");
      link(apose, "out-sheet", muse6, "in-images");
      this.edges = (this.edges || []).filter(e => !(e.from === tex.id && e.to === vggt.id));
      link(muse6, "out-image", vggt, "in-image");   // VGGT validates photoreal 6 views
    }
    for (const m of [museHat, museNeutral, museSad]) {
      if (m) link(photo, "out-photo", m, "in-images");
    }
    if (museBald && museSad) link(museSad, "out-image", museBald, "in-images");

    const noteTpl = MODEL_CATALOG["asset-note"];
    if (noteTpl) {
      const headers = [
        [60, 40, "STAGE 1 - INPUT\nOne private-person photo. Everything below is derived from it."],
        [420, 40, "STAGE 2 - SAPIENS2 SUPERVISION\n308 keypoints, segmentation, skin normals, metric depth. Local Meta open models."],
        [840, 40, "STAGE 3 - SOLVER LADDER\nSame photo, three levels of supervision: HMR baseline -> +Sapiens2 (neutral) -> +expression."],
        [1300, 40, "STAGE 4 - FINAL ASSET\nThe one deliverable: textured GLB + 1024 UV texture + textured 6-view turnaround."],
        [2020, 40, "STAGE 5 - VALIDATE & EXPORT\nMeta A-pose, Meta Image photoreal turnaround, VGGT consistency check, character GLB."],
        [420, 1560, "STAGE 6 - CHARACTER VARIANTS (Meta Image edits)\nSame person, different states: hat off, neutral, sad, and a chained sad -> bald edit."],
      ];
      headers.forEach(([x, y, text], i) => {
        this.nodes.push(Object.assign(JSON.parse(JSON.stringify(noteTpl)), {
          id: "asset-note-stage-" + (i + 1),
          x, y,
          noteWidth: 340,
          noteKind: "header",
          noteText: text,
          status: "Stage",
        }));
      });
      this.nodes.push(Object.assign(JSON.parse(JSON.stringify(noteTpl)), {
        id: "asset-note-preset",
        x: 2420,
        y: 1600,
        noteText:
          "GNM-V - the honest ladder.\n\n" +
          "1. HMR baseline: one photo -> SAM 3D Body single pass.\n" +
          "2. + Sapiens2: 308 keypoints, seg, normals, metric depth supervise identity + pose.\n" +
          "3. + Expression: 72 coefficients (jaw/lips) solved from 0.\n" +
          "4. FINAL: textured GLB + UV texture + 6-view sheet.\n\n" +
          "5. Meta API stages (5 muse-image-1.0 calls): 6-view photoreal turnaround -> VGGT, " +
          "hat-off, neutral, sad, and a CHAINED edit (sad -> bald) that shows multi-turn character control.\n\n" +
          "Card colours: blue = Meta cloud API; blue edge + yellow inner = local Meta open model (HuggingFace); " +
          "slate = local render/utility (not a model).\n" +
          "Press Auto Traverse to run the ladder upstream-first.",
      }));
    }
    this.selectedNodeId = (expr || neutral || raw || this.nodes[0]).id;
    this.renderCanvas();
    const sidebar = document.getElementById("nodes-sidebar");
    if (sidebar && !sidebar.classList.contains("collapsed")) this.toggleNodesPanel();
    this.fitView();
    setTimeout(() => this.fitView(), 600);
  }

  /**
   * Watch the backend while the page is open. A server restart (or death)
   * used to leave cards spinning forever; now any in-flight card is marked
   * Failed with an explicit reason and a toast says what happened.
   */
  startBackendWatch() {
    if (this._backendWatch) return;
    const tick = async () => {
      let health = null;
      try {
        const res = await fetchJson("/api/health", { timeoutMs: 6000 });
        health = res.ok ? (res.data || null) : null;
      } catch (err) {
        health = null;
      }
      const online = !!health;
      const running = (this.nodes || []).filter(n => n.status === "Running...");
      // A card that has been "Running..." for over 3 minutes with the backend
      // reachable is a lost response, not a computation: fail it visibly.
      const now = Date.now();
      for (const n of running) {
        const started = n._runStartedAt || 0;
        if (started && now - started > 180000) {
          n.status = "Failed";
          n.runError = "No response after 3 minutes. The server usually finished the work (see Logs / debug dir) - the HTTP response was lost. Click Run again.";
          n.runNote = null;
          this.renderCanvas();
          this.showFlowToast("A run got no response for 3 minutes - marked failed. Click Run again.", null, true);
        }
      }
      if (!online) {
        this.backendOnline = false;
        if (running.length > 0) {
          for (const n of running) {
            n.status = "Failed";
            n.runError = "Backend unreachable - the server died or was restarted mid-run. Click Run again.";
            n.runNote = null;
          }
          this.renderCanvas();
          this.showFlowToast("Backend unreachable - in-flight runs marked failed. Re-run them.", null, true);
        }
        return;
      }
      const started = health.server_started || null;
      if (this._serverStarted && started && started !== this._serverStarted && running.length > 0) {
        for (const n of running) {
          n.status = "Failed";
          n.runError = "Server restarted mid-run (old process exited) - the result was lost. Click Run again.";
          n.runNote = null;
        }
        this.renderCanvas();
        this.showFlowToast("Server restarted mid-run - the card was reset. Click Run again.", null, true);
      }
      this._serverStarted = started;
      this.backendOnline = true;
    };
    this._backendWatch = setInterval(tick, 8000);
    tick();
  }

  /**
   * Boot the graph the judges should see: the prepared demo graph from
   * flow-viewer/data/prepared_flow.json, else whatever a saved graph holds,
   * else the plain photo preset.
   */
  async bootFlow(runId) {
    this.activeRunId = runId;
    const params = (typeof window !== "undefined" && window.location && window.location.search)
      ? new URLSearchParams(window.location.search) : new URLSearchParams();
    if (params.get("preset") === "blank") {
      this.loadPreset(runId);
      return;
    }
    if (params.get("shared") === "1" || params.get("flow") === "shared") {
      const ok = await this.loadGraphFromDisk();
      this.startSharedGraphWatch();
      if (ok) return;
    }
    let prepared = null;
    try {
      const res = await fetchJson("/api/flow/prepared", { timeoutMs: 4000 });
      if (res.ok && res.data) prepared = res.data;
    } catch (err) {
      prepared = null;
    }
    if (prepared && prepared.preset === "meta-avatar") {
      this.loadMetaAvatarPreset(prepared.photo || "source.jpg");
      this.startSharedGraphWatch();
      return;
    }
    if (prepared && Array.isArray(prepared.nodes)) {
      this.restoreFlow(prepared);
      this.startSharedGraphWatch();
      return;
    }
    this.loadPreset(runId);
    this.startSharedGraphWatch();
  }

  /** Restore a graph snapshot (nodes/edges/positions) onto the canvas. */
  restoreFlow(data) {
    if (!data || !Array.isArray(data.nodes)) return false;
    this.revokeUserPhoto();
    this.userPhoto = null;
    this.nodes = JSON.parse(JSON.stringify(data.nodes));
    this.edges = JSON.parse(JSON.stringify(data.edges || []));
    this.customPositions = JSON.parse(JSON.stringify(data.customPositions || {}));
    if (data.activeRunId) {
      this.activeRunId = data.activeRunId;
      if (this.runSelector && [...this.runSelector.options].some(o => o.value === this.activeRunId)) {
        this.runSelector.value = this.activeRunId;
      }
    }
    this.selectedNodeId = data.selectedNodeId && this.nodes.some(n => n.id === data.selectedNodeId)
      ? data.selectedNodeId
      : (this.nodes[0] ? this.nodes[0].id : null);
    this.closePopover();
    this.closeContextMenu();
    this.renderCanvas();
    this.fitView();
    return true;
  }

  /** Shared graph name on the server ("shared" = the team slot). */
  sharedGraphName() {
    let who = "shared";
    try { who = localStorage.getItem("reui.flow.sharedName") || "shared"; } catch { /* private mode */ }
    return who;
  }

  /** Save the current graph to the server, into the shared slot. */
  async saveGraphToDisk() {
    const snap = serializeFlow(this, this.sharedGraphName());
    try {
      await inlineBlobs(snap.nodes);   // browser-owned photos become data-URLs
    } catch (err) {
      this.flowsPanelMessage(`Could not inline a photo for saving (${err?.message || err}).`, true);
      return false;
    }
    const name = this.sharedGraphName();
    const res = await fetchJson("/api/flow/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ flow: snap, name, saved_by: "flow-viewer" }),
      timeoutMs: 20000,
    });
    if (!res.ok) {
      this.flowsPanelMessage(`Save failed: ${res.error || "server error"}`, true);
      this.showFlowToast(`Shared save failed: ${res.error || "server error"}`, null, true);
      return false;
    }
    this.sharedSavedAt = res.data.savedAt || Date.now();
    this.flowsPanelMessage(`Saved to shared slot "${name}" (${res.data.nodes} nodes, ${res.data.edges} edges).`);
    this.showFlowToast(`Saved shared graph "${name}" - other users can load it now.`, null);
    return true;
  }

  /** Load the graph from the shared server slot. */
  async loadGraphFromDisk() {
    const name = this.sharedGraphName();
    try {
      const res = await fetchJson(`/api/flow/saved?name=${encodeURIComponent(name)}`, { timeoutMs: 10000 });
      if (!res.ok || !res.data) {
        this.flowsPanelMessage("No shared graph on the server yet.", true);
        this.showFlowToast("No shared graph on the server yet - hit Save Graph first.", null, true);
        return false;
      }
      this.sharedSavedAt = res.data.savedAt || Date.now();
      const ok = this.restoreFlow(res.data);
      const count = Array.isArray(res.data.nodes) ? res.data.nodes.length : 0;
      this.flowsPanelMessage(`Loaded shared graph "${name}".`);
      // Always give visible feedback: silence here used to look like "the button does nothing".
      this.showFlowToast(ok
        ? `Loaded shared graph "${name}" - ${count} nodes.`
        : `Shared graph "${name}" had no node list (${count} nodes).`, null, !ok);
      return true;
    } catch (err) {
      this.showFlowToast(`Load failed: ${err?.message || err}`, null, true);
      return false;
    }
  }

  /** Small floating toast; onClick loads the shared graph. */
  showFlowToast(text, onClick, isError = false) {
    let el = document.getElementById("flow-toast");
    if (!el) {
      el = document.createElement("div");
      el.id = "flow-toast";
      el.className = "flow-toast";
      document.body.appendChild(el);
    }
    el.textContent = text;
    el.classList.toggle("is-error", !!isError);
    el.hidden = false;
    if (el._timer) clearTimeout(el._timer);
    el._timer = setTimeout(() => { el.hidden = true; }, 9000);
    el.onclick = () => {
      el.hidden = true;
      if (onClick) onClick();
    };
  }

  /** Poll the server for a newer shared graph saved by the other user. */
  startSharedGraphWatch() {
    if (this._sharedWatch) return;
    const tick = async () => {
      try {
        const res = await fetchJson("/api/flow/list", { timeoutMs: 8000 });
        if (!res.ok || !res.data) return;
        const name = this.sharedGraphName();
        const mine = (res.data.graphs || []).find(g => g.name === name);
        if (!mine || !mine.savedAt) return;
        if (!this.sharedSavedAt) {
          this.sharedSavedAt = mine.savedAt;
          const t0 = new Date(mine.savedAt).toLocaleTimeString();
          this.showFlowToast(`Shared graph on the server (${mine.nodes} nodes, saved ${t0}) - click to load it.`,
            () => this.loadGraphFromDisk());
          return;
        }
        if (mine.savedAt > this.sharedSavedAt + 1500) {
          this.sharedSavedAt = mine.savedAt;
          const t = new Date(mine.savedAt).toLocaleTimeString();
          this.showFlowToast(
            `Shared graph updated at ${t} (${mine.nodes} nodes) - click to load it.`,
            () => this.loadGraphFromDisk());
        }
      } catch (err) { /* server offline: stay quiet */ }
    };
    this._sharedWatch = setInterval(tick, 12000);
    tick();
  }

  async checkHealthAndRuns() {
    try {
      const health = await fetchJson("/api/health", { timeoutMs: 8000 });
      if (health.ok && health.data) {
        this.backendOnline = true;
        const h = health.data;
        // Old builds answer /api/health without these fields. Their JS is the
        // same on-disk bundle, so the page looks current while the backend is
        // stale - that mismatch is what looks like "crashed" cards.
        if (!h.server_started && typeof window !== "undefined" && window.location) {
          const port = window.location.port;
          const target = `${window.location.protocol}//${window.location.hostname}:8152${window.location.pathname}`;
          if (port !== "8152") {
            this.showFlowToast(`Old backend on port ${port} - switching to the live server on 8151...`, null);
            setTimeout(() => { window.location.replace(target); }, 1200);
            return;
          }
        }
        if (this.statPill) {
          this.statPill.textContent = h.muse ? "Meta Muse: Ready" : "Meta Muse: Replay";
        }
      } else {
        this.backendOnline = false;
        if (this.statPill) {
          this.statPill.textContent = "Backend: Offline (Demo)";
          this.statPill.dataset.variant = "warning-light";
        }
      }

      const runsRes = await fetchJson("/api/runs", { timeoutMs: 8000 });
      if (runsRes.ok && runsRes.data) {
        const data = runsRes.data;
        const runs = data.runs || [];
        this.runSelector.innerHTML = "";
        runs.forEach(r => {
          const opt = document.createElement("option");
          opt.value = r;
          opt.textContent = r;
          this.runSelector.appendChild(opt);
        });
        if (data.active_run) {
          this.runSelector.value = data.active_run;
          await this.bootFlow(data.active_run);
        } else if (runs.length > 0) {
          const chosen = runs[runs.length - 1];
          this.runSelector.value = chosen;
          await this.bootFlow(chosen);
        } else {
          await this.bootFlow(this.activeRunId);
        }
      } else {
        await this.bootFlow(this.activeRunId);
      }
    } catch (err) {
      this.backendOnline = false;
      console.log("Using standalone demo pipeline data");
      await this.bootFlow(this.activeRunId);
    }
  }
}

/**
 * Pure: true when the graph is exactly the untouched preset — the
 * preset-backed source photo alone, no cards, no wires. Run switching
 * reloads these graphs; anything the user touched keeps its cards.
 */
export function isPristinePresetFlow(app) {
  const nodes = app.nodes || [];
  const edges = app.edges || [];
  if (nodes.length !== 1 || edges.length !== 0) return false;
  const photo = nodes[0];
  if (photo.id !== "asset-photo") return false;
  if (!photo.imgUrl || !photo.imgUrl.startsWith("/api/runs/")) return false;
  return true;
}

/**
 * Pure: true when leaving the page would lose user work — any graph that is
 * not the untouched preset (nothing to lose there: it reloads identically).
 */
export function needsUnloadGuard(app) {
  return (app.nodes || []).length > 0 && !isPristinePresetFlow(app);
}

export function initUnloadGuard(app) {
  if (typeof window === "undefined") return;
  window.addEventListener("beforeunload", (e) => {
    if (!needsUnloadGuard(app)) return;
    e.preventDefault();
    e.returnValue = "";
  });
}

if (typeof window !== "undefined") {
  window.addEventListener("DOMContentLoaded", () => {
    window.flowApp = new DynamicFlowApp();
  });
}
