/**
 * Canvas Engine & Layout Manager
 * Handles viewport panning, zooming, dot grid alignment, file drag-and-drop,
 * empty dropzones, and topological DAG auto-alignment.
 */

import { svgPlaceholder, uid, resolveAssetImageUrl } from "./util.js";
import { clearDownstreamOutputs } from "./execution.js";

export function initCanvasPanZoom(app) {
  let panRafId = null;

  app.viewport.addEventListener("mousedown", (e) => {
    // Only left click (0) or middle click (1) for canvas panning; right click is for context menu
    if (e.button !== 0 && e.button !== 1) return;

    if (e.target.closest(".flow-node") || 
        e.target.closest(".flow-empty-dropzone") || 
        e.target.closest(".flow-floating-controls") ||
        e.target.closest(".flow-add-popover") ||
        e.target.closest(".flow-context-menu") ||
        e.target.closest("#viewport-drop-overlay")) {
      return;
    }
    app.closePopover();
    app.closeContextMenu();
    app.isDraggingCanvas = true;
    app.viewport.classList.add("is-panning");
    app.canvasDragStart = { x: e.clientX - app.panX, y: e.clientY - app.panY };
  });

  window.addEventListener("mousemove", (e) => {
    if (!app.isDraggingCanvas) return;
    app.panX = e.clientX - app.canvasDragStart.x;
    app.panY = e.clientY - app.canvasDragStart.y;
    if (!panRafId) {
      panRafId = requestAnimationFrame(() => {
        updateTransform(app);
        panRafId = null;
      });
    }
  });

  window.addEventListener("mouseup", () => {
    if (app.isDraggingCanvas) {
      app.isDraggingCanvas = false;
      app.viewport.classList.remove("is-panning");
    }
  });

  app.viewport.addEventListener("wheel", (e) => {
    // A wheel over scrollable card content (logs, legends, traces) scrolls
    // it; only zoom when nothing under the cursor can take the gesture.
    // Pinch-zoom (ctrlKey) always zooms — that is the zoom gesture.
    if (!e.ctrlKey && wheelHitScrollable(e.target, e.deltaY, e.deltaX || 0, app.viewport)) return;
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.08 : 0.92;
    const rect = app.viewport.getBoundingClientRect();
    setZoom(app, app.scale * zoomFactor, {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  }, { passive: false });

  updateTransform(app);
}

/**
 * Pure given DOM nodes: true when the wheel gesture should go to page
 * content instead of canvas zoom — i.e. some ancestor below stopAt can
 * scroll further in the gesture's direction. Text-node targets resolve to
 * their parent. Overflow style is consulted only where geometry already
 * says scrollable (and only in a browser; node tests use geometry alone).
 */
export function wheelHitScrollable(target, deltaY, deltaX, stopAt) {
  let el = target && target.nodeType === 3 ? target.parentElement : target;
  const canStyle = typeof getComputedStyle === "function";
  while (el && el !== stopAt) {
    if (el.nodeType === 1) {
      const vBar = (el.scrollHeight || 0) > (el.clientHeight || 0) + 1;
      const hBar = (el.scrollWidth || 0) > (el.clientWidth || 0) + 1;
      if ((deltaY && vBar) || (deltaX && hBar)) {
        let ovY = "auto", ovX = "auto";
        if (canStyle) {
          try {
            const st = getComputedStyle(el);
            ovY = st.overflowY || "auto";
            ovX = st.overflowX || "auto";
          } catch { /* closed shadow edge: fall through on geometry */ }
        }
        if (deltaY && vBar && /auto|scroll|overlay/i.test(ovY)) {
          const max = el.scrollHeight - el.clientHeight;
          if ((deltaY > 0 && (el.scrollTop || 0) < max) || (deltaY < 0 && (el.scrollTop || 0) > 0)) return true;
        }
        if (deltaX && hBar && /auto|scroll|overlay/i.test(ovX)) {
          const maxX = el.scrollWidth - el.clientWidth;
          if ((deltaX > 0 && (el.scrollLeft || 0) < maxX) || (deltaX < 0 && (el.scrollLeft || 0) > 0)) return true;
        }
      }
    }
    el = el.parentElement;
  }
  return false;
}

export function setZoom(app, nextScale, anchor) {
  const clamped = Math.max(0.3, Math.min(2.0, nextScale));
  // Zoom toward the anchor (cursor for wheel, viewport center for buttons):
  // screen = pan + scale·world, so hold the world point under the anchor fixed.
  const ax = anchor ? anchor.x : app.viewport.clientWidth / 2;
  const ay = anchor ? anchor.y : app.viewport.clientHeight / 2;
  const wx = (ax - app.panX) / app.scale;
  const wy = (ay - app.panY) / app.scale;
  app.scale = clamped;
  app.panX = ax - wx * clamped;
  app.panY = ay - wy * clamped;
  updateTransform(app);
}

export function fitView(app) {
  if (app.nodes.length === 0) {
    app.scale = 0.75;
    app.panX = 140;
    app.panY = 80;
    updateTransform(app);
    return;
  }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  app.nodes.forEach(n => {
    minX = Math.min(minX, n.x);
    minY = Math.min(minY, n.y);
    const el = document.getElementById(n.id);
    const w = el ? el.offsetWidth : 320;
    const h = el ? el.offsetHeight : 360;
    maxX = Math.max(maxX, n.x + w);
    maxY = Math.max(maxY, n.y + h);
  });

  const vWidth = app.viewport.clientWidth || (window.innerWidth - 380);
  const vHeight = app.viewport.clientHeight || (window.innerHeight - 52);
  const graphWidth = Math.max(100, maxX - minX + 160);
  const graphHeight = Math.max(100, maxY - minY + 160);

  const scaleX = vWidth / graphWidth;
  const scaleY = vHeight / graphHeight;
  app.scale = Math.max(0.3, Math.min(1.0, Math.min(scaleX, scaleY)));

  app.panX = Math.round((vWidth - (maxX + minX) * app.scale) / 2);
  app.panY = Math.round((vHeight - (maxY + minY) * app.scale) / 2);
  updateTransform(app);
}

export function updateTransform(app) {
  app.plane.style.transform = `translate(${app.panX}px, ${app.panY}px) scale(${app.scale})`;
  app.viewport.style.backgroundPosition = `${app.panX}px ${app.panY}px`;
  app.viewport.style.backgroundSize = `${24 * app.scale}px ${24 * app.scale}px`;
}

export function initDragAndDrop(app) {
  const hasImageFile = (e) => {
    if (!e.dataTransfer || !e.dataTransfer.types) return false;
    return Array.from(e.dataTransfer.types).includes("Files");
  };

  app.viewport.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (hasImageFile(e)) {
      app.viewport.classList.add("file-drag-over");
    }
  });

  app.viewport.addEventListener("dragleave", (e) => {
    if (e.relatedTarget === null || !app.viewport.contains(e.relatedTarget)) {
      app.viewport.classList.remove("file-drag-over");
    }
  });

  // Internal card-image drags carry our MIME payload; the drop handler turns
  // them into new image asset cards instead of navigating away (which used
  // to wipe the in-memory graph).
  app.viewport.addEventListener("dragstart", (e) => {
    const t = e.target;
    if (!t || !t.closest || !e.dataTransfer) return;
    const img = t.closest("img.flow-node-img");
    if (!img) return;
    const card = img.closest(".flow-node");
    const node = card ? (app.nodes || []).find(n => n.id === card.id) : null;
    let url = null;
    let label = "dragged image";
    if (node) {
      label = node.subtitle || node.title || label;
      url = img.hasAttribute("data-result-img")
        ? (node.resultImage || node.imgUrl || null)
        : (resolveAssetImageUrl(app, node) || node.imgUrl || null);
    }
    if (!url) url = img.currentSrc || img.src || null;
    if (!url) return;
    try {
      e.dataTransfer.setData(IMAGE_DRAG_MIME, JSON.stringify({ imgUrl: url, label }));
      e.dataTransfer.effectAllowed = "copy";
    } catch { /* MIME unsupported — native drag proceeds */ }
  });

  // The viewport drop listener only covers the canvas: a drop on the
  // header, inspector or any other chrome used to navigate the tab to the
  // dragged URL and lose the graph. Swallow those drops everywhere except
  // inside text fields (where dropping text must keep working).
  if (typeof document !== "undefined" && !document.__flowDropGuard) {
    document.__flowDropGuard = true;
    const inField = (e) => !!(e.target && e.target.closest && e.target.closest("input, textarea, [contenteditable]"));
    document.addEventListener("dragover", (e) => { if (!inField(e)) e.preventDefault(); });
    document.addEventListener("drop", (e) => { if (!inField(e)) e.preventDefault(); });
  }

  app.viewport.addEventListener("drop", (e) => {
    e.preventDefault();
    app.viewport.classList.remove("file-drag-over");

    const internal = parseImageDragPayload(e.dataTransfer);
    if (internal) {
      const rect = app.plane.getBoundingClientRect();
      spawnImageAssetFromDrag(app, internal,
        Math.round((e.clientX - rect.left) / app.scale),
        Math.round((e.clientY - rect.top) / app.scale));
      return;
    }

    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith("image/")) {
        const rect = app.plane.getBoundingClientRect();
        const dropX = Math.round((e.clientX - rect.left) / app.scale);
        const dropY = Math.round((e.clientY - rect.top) / app.scale);
        handlePhotoDropped(app, file, dropX, dropY);
      }
    }
  });
}

/**
 * Swap the photo on an EXISTING source card in place: the graph (nodes,
 * edges, positions) survives; downstream results are invalidated exactly as
 * on re-run, since they described the old pixels.
 */
export function replacePhotoOnNode(app, nodeId, file) {
  const node = (app.nodes || []).find(n => n.id === nodeId);
  if (!node) return false;
  app.revokeUserPhoto();
  const url = URL.createObjectURL(file);
  const sizeKB = (file.size / 1024).toFixed(1) + " KB";
  app.userPhoto = { url, name: file.name, size: sizeKB };
  node.imgUrl = url;
  node.imgFallback = svgPlaceholder(file.name);
  node.imgBadge = `${file.name} · ${sizeKB}`;
  node.subtitle = file.name;
  node.tensorShape = "RGB image (? × ?)";
  node.imageWidth = null;
  node.imageHeight = null;
  // Old downstream results described the old pixels — clear them (wiring
  // survives; user-placed cards survive; generated artifacts are deleted).
  clearDownstreamOutputs(app, nodeId);
  app.selectedNodeId = nodeId;
  app.renderCanvas();
  app.resolvePhotoDims(nodeId, url);
  return true;
}

export const IMAGE_DRAG_MIME = "application/x-flow-image";

/**
 * Read an internal card-image drag payload off a drop event's dataTransfer.
 * Returns { imgUrl, label } or null when this is not one of our drags.
 */
export function parseImageDragPayload(dataTransfer) {
  if (!dataTransfer || !dataTransfer.types) return null;
  if (!Array.from(dataTransfer.types).includes(IMAGE_DRAG_MIME)) return null;
  try {
    const p = JSON.parse(dataTransfer.getData(IMAGE_DRAG_MIME));
    if (!p || typeof p.imgUrl !== "string" || !p.imgUrl) return null;
    return { imgUrl: p.imgUrl, label: typeof p.label === "string" && p.label ? p.label : "dragged image" };
  } catch {
    return null;
  }
}

/**
 * Spawn an image asset card from an internal card-image drag. Additive
 * only: the existing graph (nodes, edges, positions) is never touched.
 */
export function spawnImageAssetFromDrag(app, payload, dropX = null, dropY = null) {
  const posX = dropX !== null ? Math.max(60, dropX) : 80;
  const posY = dropY !== null ? Math.max(60, dropY) : 140;
  const label = (payload && payload.label) || "dragged image";
  const node = {
    id: uid("asset-image"),
    nodeType: "asset",
    assetKind: "image",
    title: "Asset: Image",
    subtitle: label,
    category: "Image Asset",
    badgeVariant: "success-light",
    iconVariant: "outline",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>`,
    imgUrl: payload.imgUrl,
    imgFallback: svgPlaceholder(label),
    imgBadge: label,
    description: `Image dragged from a card. Feeds raw RGB raster frames into vision actions.`,
    latency: "Ready",
    status: "Cached in RAM",
    tensorShape: "RGB image (? × ?)",
    format: "RGB Bitmap",
    inputs: [
      { id: "in-view", name: "View", type: "image", color: "#10b981", badge: "View", accepts: ["image", "box", "mask", "crop"] }
    ],
    outputs: [
      { id: "out-photo", name: "Photo Output", type: "image", color: "#10b981", badge: "RGB" }
    ],
    x: posX,
    y: posY
  };
  (app.nodes = app.nodes || []).push(node);
  app.selectedNodeId = node.id;
  app.renderCanvas();
  if (typeof app.resolveAssetDims === "function") app.resolveAssetDims(node.id, node.imgUrl);
  return node.id;
}

export function handlePhotoDropped(app, file, dropX = null, dropY = null) {
  // Dropping onto a wired graph means "new input photo", not "throw the graph
  // away": replace the existing source-photo card and keep every card/edge.
  const wiredPhoto = (app.nodes || []).find(n => n.id === "asset-photo")
    || (app.nodes || []).find(n => n.nodeType === "asset" && n.assetKind === "image");
  if (wiredPhoto && (app.edges || []).some(e => e.from === wiredPhoto.id)) {
    return replacePhotoOnNode(app, wiredPhoto.id, file);
  }
  app.revokeUserPhoto();
  const url = URL.createObjectURL(file);
  const sizeKB = (file.size / 1024).toFixed(1) + " KB";
  app.userPhoto = { url, name: file.name, size: sizeKB };

  // Clear canvas and start fresh with ONLY this photo asset card
  app.nodes = [];
  app.edges = [];
  app.customPositions = {};

  const posX = dropX !== null ? Math.max(60, dropX) : 80;
  const posY = dropY !== null ? Math.max(60, dropY) : 140;

  const photoNode = {
    id: "asset-photo",
    nodeType: "asset",
    assetKind: "image",
    title: "Asset: Source Photo",
    subtitle: file.name,
    category: "Image Asset",
    badgeVariant: "success-light",
    iconVariant: "outline",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>`,
    imgUrl: url,
    imgFallback: svgPlaceholder(file.name),
    imgBadge: `${file.name} · ${sizeKB}`,
    description: `Dropped source photo. Feeds raw RGB raster frames into Meta SAM 3.1 segmentation.`,
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
    x: posX,
    y: posY
  };

  const samTemplate = app.getModelCatalog()["action-meta-sam"];
  const samNode = {
    id: "action-meta-sam",
    templateKey: "action-meta-sam",
    ...JSON.parse(JSON.stringify(samTemplate)),
    x: posX + 440,
    y: posY
  };

  const wireEdge = {
    id: "e-asset-photo-action-meta-sam",
    from: "asset-photo",
    to: "action-meta-sam",
    fromPort: "out-photo",
    toPort: "in-image",
    label: "raw RGB stream"
  };

  app.nodes.push(photoNode, samNode);
  app.edges.push(wireEdge);
  app.selectedNodeId = "action-meta-sam";
  app.renderCanvas();

  fitView(app);
  app.resolvePhotoDims(photoNode.id, photoNode.imgUrl);
}

export function renderEmptyDropzone(app) {
  app.plane.querySelectorAll(".flow-node, .flow-edge-label, .flow-add-popover").forEach(n => n.remove());
  app.svgLayer.innerHTML = "";

  let dropzone = app.plane.querySelector(".flow-empty-dropzone");
  if (!dropzone) {
    dropzone = document.createElement("div");
    dropzone.className = "flow-empty-dropzone";
    dropzone.style.left = "420px";
    dropzone.style.top = "120px";

    dropzone.innerHTML = `
      <div class="reui-icon-tile" data-variant="purple" data-size="lg" style="width:52px;height:52px">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style="width:26px;height:26px">
          <rect x="3" y="3" width="18" height="18" rx="2"></rect>
          <circle cx="8.5" cy="8.5" r="1.5"></circle>
          <polyline points="21 15 16 10 5 21"></polyline>
        </svg>
      </div>

      <div>
        <h2 class="flow-empty-title">Drop a Photo to Start Flow</h2>
        <p class="flow-empty-desc">
          Drag and drop any portrait photo directly onto this canvas to spawn the initial photo asset. You can then connect models piece-by-piece.
        </p>
      </div>

      <div class="flow-empty-actions">
        <button id="btn-browse-photo" class="reui-btn primary">📁 Browse Photo...</button>
      </div>

      <div class="flow-empty-presets">
        <span class="flow-empty-presets-title">Or start with sample run</span>
        <div class="flow-presets-grid">
          <button id="btn-load-suit" class="reui-btn sm">⚡ Run 20261103 (Dark Suit & Blazer)</button>
          <button id="btn-load-meta-avatar" class="reui-btn sm">🧬 Meta Avatar stack (fit + A-pose + VGGT)</button>
          <button id="btn-load-muse-6view" class="reui-btn sm">🎨 Muse 6-view → VGGT (1 API call)</button>
        </div>
      </div>
    `;

    dropzone.querySelector("#btn-browse-photo").addEventListener("click", () => {
      // Fresh graph from here: drop any stale in-card replace target (e.g.
      // the user opened Replace and cancelled the dialog earlier).
      app.replacePhotoTargetId = null;
      if (app.fileInput) app.fileInput.click();
    });

    dropzone.querySelector("#btn-load-suit").addEventListener("click", () => {
      app.loadPreset("20261103-213035-3359", "dark_suit_pair.jpg");
    });

    const metaBtn = dropzone.querySelector("#btn-load-meta-avatar");
    if (metaBtn) metaBtn.addEventListener("click", () => {
      app.loadMetaAvatarPreset("meta_avatar.jpg");
    });
    const muse6Btn = dropzone.querySelector("#btn-load-muse-6view");
    if (muse6Btn) muse6Btn.addEventListener("click", () => {
      app.loadMuseTurnaroundPreset("meta_avatar.jpg");
    });

    app.plane.appendChild(dropzone);
  }
}

export function tidyLayout(app) {
  app.customPositions = {};
  if (app.nodes.length === 0) return;

  // Calculate topological depth using DAG edges
  const inDegree = {};
  const childrenMap = {};
  app.nodes.forEach(n => {
    inDegree[n.id] = 0;
    childrenMap[n.id] = [];
  });

  app.edges.forEach(e => {
    if (inDegree[e.to] !== undefined) {
      inDegree[e.to]++;
    }
    if (childrenMap[e.from]) {
      childrenMap[e.from].push(e.to);
    }
  });

  const depth = {};
  const queue = [];
  app.nodes.forEach(n => {
    if (inDegree[n.id] === 0) {
      depth[n.id] = 0;
      queue.push(n.id);
    }
  });

  app.nodes.forEach(n => {
    if (depth[n.id] === undefined) depth[n.id] = 0;
  });

  while (queue.length > 0) {
    const currId = queue.shift();
    const currDepth = depth[currId];
    (childrenMap[currId] || []).forEach(childId => {
      if ((depth[childId] || 0) < currDepth + 1) {
        depth[childId] = currDepth + 1;
        queue.push(childId);
      }
    });
  }

  // Group nodes by depth column
  const columns = {};
  app.nodes.forEach(n => {
    const col = depth[n.id] || 0;
    if (!columns[col]) columns[col] = [];
    columns[col].push(n);
  });

  const colKeys = Object.keys(columns).map(Number).sort((a, b) => a - b);
  colKeys.forEach((colIdx) => {
    const colNodes = columns[colIdx];
    const colX = 80 + colIdx * 480;

    const totalHeight = colNodes.reduce((acc, n) => {
      const el = document.getElementById(n.id);
      return acc + (el ? el.offsetHeight + 40 : 420);
    }, 0);

    let startY = Math.max(60, 360 - totalHeight / 2);
    colNodes.forEach(n => {
      n.x = colX;
      n.y = Math.round(startY);
      const el = document.getElementById(n.id);
      const h = el ? el.offsetHeight + 40 : 420;
      startY += h;
    });
  });

  app.renderCanvas();
  fitView(app);
}
