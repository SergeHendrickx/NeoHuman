/**
 * Node Card & Blueprint Socket Renderer
 * Builds DOM elements for Action/Asset nodes, Blueprint side-socket rows,
 * confidence tags, live SVG detection overlays, and display toggle chips.
 */

import { escapeHtml, minimizePinSide, aspectRatioStyle, cardWidthForImage, pinTypeColor, isMetaModel, nodeAccentColor, resolveAssetImageUrl, wiredViewLabel, sameHtml } from "./util.js";
import { renderApiTraceHtml, renderRunLogHtml } from "./trace-view.js";
import { renderPoseSvgInner, renderSegLegendHtml, bindResultStages } from "./overlays.js";
import { renderMeshStageHtml, bindMeshViewer, meshViewerKind, shouldMountMeshViewer } from "./mesh-view.js";

/**
 * Action-header band class: the top bar reads the run outcome at a glance —
 * green verified pass, red failure, amber demo/unverified, striped amber
 * while running, default blue while untouched. Actions only; asset cards
 * never run, so they keep their kind tint.
 */
export function headerStatusClass(status) {
  const s = String(status || "");
  if (s === "Done" || s === "Passed") return "is-passed";
  if (s === "Running...") return "is-running";
  if (/Fail|Error/i.test(s)) return "is-failed";
  if (/Demo|Baseline|Timed out/i.test(s)) return "is-demo";
  if (s === "Cached in RAM" || s === "Artifact" || s === "Produced" || s.startsWith("Viewing")) return "is-demo";
  return "";
}

/** Status-strip dot class: success / active / warning / error / none. */
export function statusDotClass(status) {
  const s = String(status || "");
  if (s === "Done" || s === "Passed") return "success";
  if (s === "Running...") return "active";
  if (/Fail|Error/i.test(s)) return "error";
  if (/Demo|Offline|Timed out/i.test(s)) return "warning";
  if (s === "Cached in RAM" || s === "Artifact" || s === "Produced" || s.startsWith("Viewing")) return "warning";
  return "";
}

/** Pure: the live progress block (bar + stage label), direct-DOM updated mid-run. */
export function renderProgressHtml(node) {
  const pct = node.progress ? Math.round(node.progress.frac * 100) : 0;
  const label = (node.progress && node.progress.label) || "starting…";
  return `<div class="flow-progress">`
    + `<div class="flow-progress-track"><div class="flow-progress-fill" style="width:${pct}%"></div></div>`
    + `<div class="flow-progress-label">${escapeHtml(label)}</div>`
    + `</div>`;
}

/**
 * Keep the progress block in sync with run state: present while Running
 * (polls drive it directly), removed on every terminal state. Without
 * this the bar markup only exists for cards *created* mid-run — i.e. the
 * live bar never appeared in real use.
 */
export function syncProgressBlock(el, node) {
  const strip = el.querySelector(".flow-node-run-strip");
  const prog = el.querySelector(".flow-progress");
  if (node.status === "Running...") {
    if (!prog && strip && typeof strip.insertAdjacentHTML === "function") {
      strip.insertAdjacentHTML("afterend", renderProgressHtml(node));
    }
  } else if (prog && typeof prog.remove === "function") {
    prog.remove();
  }
}

/**
 * Badge under an asset card's image: "← <pin> · WxH" while a wired input
 * overrides the view, otherwise the card's own badge. Pure given (app, node).
 */
export function assetViewBadge(app, node) {
  const label = wiredViewLabel(app, node);
  if (!label) return node.imgBadge || "";
  const dims = node.imageWidth ? ` · ${node.imageWidth}x${node.imageHeight}` : "";
  return `← ${label}${dims}`;
}

export function renderNodes(app) {
  const nodeIds = new Set(app.nodes.map(n => n.id));

  // Remove obsolete DOM nodes
  app.plane.querySelectorAll(".flow-node").forEach(el => {
    if (!nodeIds.has(el.id)) el.remove();
  });

  // Render or update each node
  app.nodes.forEach(node => {
    let el = document.getElementById(node.id);
    if (!el) {
      el = createNodeElement(app, node);
      app.plane.appendChild(el);
    } else {
      updateNodeElement(app, el, node);
    }
  });
}

export function createNodeElement(app, node) {
  // Creation stamp for the chronological asset listing (loaded flows keep it).
  node.createdAt = node.createdAt || Date.now();
  const el = document.createElement("div");
  el.id = node.id;
  
  const isAsset = node.nodeType === "asset";
  const assetKindClass = isAsset ? `asset-${node.assetKind || "image"}` : "";
  const isAction = node.nodeType === "action";
  const headerStatusCls = isAction ? headerStatusClass(node.status) : "";

  el.className = `flow-node ${node.id === app.selectedNodeId ? "selected" : ""} ${assetKindClass}${isMetaModel(node) ? " is-meta-model" : ""}`
    + (node.styleKind ? ` style-${node.styleKind}` : "")
    + (node.bigCard ? " flow-node-big" : "");
  el.style.setProperty("--node-accent", nodeAccentColor(node));
  el.dataset.nodeType = node.nodeType;
  if (node.family) {
    el.dataset.family = node.family;
  }
  el.style.left = `${node.x}px`;
  el.style.top = `${node.y}px`;
  if (node.bigCard) el.style.width = "640px";

  // Free-form note card (post-it): no pins, editable text, drag by its bar.
  if (node.nodeType === "note") {
    el.className = `flow-node flow-note${node.noteKind === "header" ? " flow-note-header" : ""}${node.id === app.selectedNodeId ? " selected" : ""}`;
    el.dataset.nodeType = "note";
    el.style.width = `${node.noteWidth || 320}px`;
    el.innerHTML = `
      <div class="flow-note-bar">
        <span class="flow-note-grip">Note</span>
        <button type="button" class="flow-note-del" title="Delete note">x</button>
      </div>
      <div class="flow-note-text" contenteditable="true" spellcheck="false">${escapeHtml(node.noteText || "")}</div>
    `;
    const txt = el.querySelector(".flow-note-text");
    txt.addEventListener("input", () => { node.noteText = txt.innerText; });
    txt.addEventListener("mousedown", (e) => e.stopPropagation());
    el.querySelector(".flow-note-del").addEventListener("click", (e) => {
      e.stopPropagation();
      app.deleteCard(node.id);
    });
    el.addEventListener("mousedown", (e) => {
      if (e.target.closest("button, [contenteditable]")) return;
      app.closePopover();
      app.closeContextMenu();
      const sx = e.clientX, sy = e.clientY, ox = node.x, oy = node.y;
      let moved = false;
      const onMove = (ev) => {
        const dx = (ev.clientX - sx) / app.scale;
        const dy = (ev.clientY - sy) / app.scale;
        if (!moved && Math.hypot(dx, dy) > 3) moved = true;
        if (moved) {
          node.x = Math.round(ox + dx);
          node.y = Math.round(oy + dy);
          el.style.left = `${node.x}px`;
          el.style.top = `${node.y}px`;
          app.customPositions[node.id] = { x: node.x, y: node.y };
        }
      };
      const onUp = () => {
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
        if (!moved) app.selectNode(node.id);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });
    return el;
  }

  // Family Banner (for Action models)
  let familyBarHtml = "";
  if (node.familyLabel) {
    familyBarHtml = `
      <div class="flow-node-family-bar">
        <span>✦ ${escapeHtml(node.familyLabel)}</span>
      </div>
    `;
  }

  // (Concept/prompt editor renders at input level, inside the INPUTS column.)

  // Consumes & Produces IO Slots (for Actions)
  let ioHtml = "";
  if (node.consumes || node.produces) {
    const consumesHtml = (node.consumes || []).map(c => `<span class="flow-io-tag tag-consumes">⇲ ${escapeHtml(c)}</span>`).join("");
    const producesHtml = (node.produces || []).map(p => `<span class="flow-io-tag tag-produces">⇱ ${escapeHtml(p)}</span>`).join("");
    ioHtml = `
      <div class="flow-node-io">
        ${node.consumes ? `<div class="flow-io-row"><span class="flow-io-label">Consumes:</span><div class="flow-io-tags">${consumesHtml}</div></div>` : ""}
        ${node.produces ? `<div class="flow-io-row"><span class="flow-io-label">Produces:</span><div class="flow-io-tags">${producesHtml}</div></div>` : ""}
      </div>
    `;
  }

  // (Overlay display toggles render inside the in-card result preview.)

  // On-Graph Visual Preview: a wired input wins over the card's own photo,
  // so viewers (and the source photo itself) show what is plugged in.
  // (Detection overlays render in the model card's in-card result view.)
  const viewUrl = isAsset ? resolveAssetImageUrl(app, node) : node.imgUrl;
  const meshStageAny = shouldMountMeshViewer(node)
    ? renderMeshStageHtml(node, node.resultImage || viewUrl || null) : "";
  let visualHtml = "";
  if (!viewUrl && meshStageAny) {
    visualHtml = `
      <div class="flow-node-img-wrapper flow-mesh-asset">
        ${meshStageAny}
        <span class="flow-node-img-badge">3D</span>
      </div>
    `;
  }
  if (viewUrl) {
    const replaceBtnHtml = node.id === "asset-photo" ? `
      <button class="flow-photo-replace-btn" title="Upload or choose a new photo">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style="width:11px;height:11px"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
        <span>Change Photo</span>
      </button>
    ` : "";

    const fallbackAttr = escapeHtml(node.imgFallback || "");
    const ratioAttr = aspectRatioStyle(node.imageWidth, node.imageHeight);
    const assetMeshStage = shouldMountMeshViewer(node) ? renderMeshStageHtml(node, viewUrl) : "";
    visualHtml = assetMeshStage ? `
      <div class="flow-node-img-wrapper flow-mesh-asset">
        ${assetMeshStage}
        <span class="flow-node-img-badge">${escapeHtml(assetViewBadge(app, node))}</span>
      </div>
    ` : `
      <div class="flow-node-img-wrapper"${ratioAttr ? ` style="${ratioAttr}"` : ""}>
        <img class="flow-node-img" src="${escapeHtml(viewUrl)}" data-state="primary" onerror="this.onerror=null;this.dataset.state='fallback';this.src='${fallbackAttr}'" alt="${escapeHtml(node.title)}" title="Double-click for full size">
        ${replaceBtnHtml}
        <span class="flow-node-img-badge">${escapeHtml(assetViewBadge(app, node))}</span>
      </div>
    `;
  }

  // Wide images grow the card (multiview sheets, side-by-sides): the
  // width follows the pixels, capped, instead of squeezing to 320px.
  const showsImage = isAsset ? !!viewUrl : !!node.resultImage;
  const widePx = showsImage ? cardWidthForImage(node.imageWidth, node.imageHeight) : null;
  if (widePx) el.style.width = `${widePx}px`;

  // Confidence badge on Candidate Person cards
  let confPillHtml = "";
  if (node.confidence !== undefined) {
    const pct = (node.confidence * 100).toFixed(1) + "%";
    confPillHtml = `<span class="flow-conf-pill">✦ ${pct} conf</span>`;
  }

  // Asset cards are photo-dominant: no description body (the inspector has it),
  // no Connect button (legacy — wire off the edge port instead).
  const bodyHtml = isAsset ? "" : `
    <div class="flow-node-body">
      ${escapeHtml(node.description)}
    </div>
  `;

  const blueprintPinsHtml = renderBlueprintPinsHtml(node, app.poseTopology);
  const minimalPortsHtml = renderMinimalPortsHtml(node);

  el.innerHTML = `
    ${familyBarHtml}
    <div class="flow-node-header ${headerStatusCls}">
      <div class="reui-icon-tile" data-variant="${escapeHtml(node.iconVariant)}" data-size="sm">
        ${node.iconSvg}
      </div>
      <div class="flow-node-header-info">
        <div class="flow-node-title" style="display:flex;align-items:center;gap:6px">
          <span>${escapeHtml(node.title)}</span>
          ${confPillHtml}
        </div>
        <div class="flow-node-subtitle">${escapeHtml(node.subtitle)}</div>
      </div>
      
      <!-- Card Action Buttons: Play/Run & Delete -->
      <div class="flow-node-header-actions">
        ${isAction ? `<button class="flow-card-btn btn-run" title="Run / Execute model" data-node-id="${node.id}">▶</button>` : ""}
        <button class="flow-card-btn btn-delete" title="Delete card" data-node-id="${node.id}">✕</button>
      </div>
    </div>

    ${visualHtml}
    ${bodyHtml}
    ${blueprintPinsHtml}
    ${minimalPortsHtml}

    <div class="flow-node-run-strip">
      <div class="status-indicator">
        <span class="status-dot ${statusDotClass(node.status)}"></span>
        <span>${escapeHtml(node.status)}</span>
      </div>
      <span style="color:var(--text)">${escapeHtml(node.latency)}</span>
    </div>
    ${isAction && node.status === "Running..." ? renderProgressHtml(node) : ""}
  `;

  // Dragging Handler (60 FPS with requestAnimationFrame)
  el.addEventListener("mousedown", (e) => {
    if (e.target.closest("button, input, select, a, .reui-btn, .flow-card-btn, .flow-opt-chip, .blueprint-port-circle, .blueprint-pin")) return;
    app.closePopover();
    app.closeContextMenu();

    const startClientX = e.clientX;
    const startClientY = e.clientY;
    const origX = node.x;
    const origY = node.y;
    let hasMoved = false;
    let dragRaf = null;

    const onMouseMove = (moveEvent) => {
      const dx = (moveEvent.clientX - startClientX) / app.scale;
      const dy = (moveEvent.clientY - startClientY) / app.scale;

      if (!hasMoved && Math.hypot(dx, dy) > 3) {
        hasMoved = true;
        el.classList.add("is-dragging");
        app.viewport.classList.add("is-node-dragging");
      }

      if (hasMoved) {
        node.x = Math.round(origX + dx);
        node.y = Math.round(origY + dy);
        el.style.left = `${node.x}px`;
        el.style.top = `${node.y}px`;
        app.customPositions[node.id] = { x: node.x, y: node.y };

        if (!dragRaf) {
          dragRaf = requestAnimationFrame(() => {
            app.updateConnectedEdges(node.id);
            dragRaf = null;
          });
        }
      }
    };

    const onMouseUp = () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
      el.classList.remove("is-dragging");
      app.viewport.classList.remove("is-node-dragging");
      if (dragRaf) {
        cancelAnimationFrame(dragRaf);
        dragRaf = null;
      }
      app.updateConnectedEdges(node.id);

      if (!hasMoved) {
        app.selectNode(node.id);
      }
    };

    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
  });

  // Action Button Listeners
  const btnRun = el.querySelector(".btn-run");
  if (btnRun) {
    btnRun.addEventListener("click", (e) => {
      e.stopPropagation();
      app.runCard(node.id);
    });
  }

  const btnDelete = el.querySelector(".btn-delete");
  if (btnDelete) {
    btnDelete.addEventListener("click", (e) => {
      e.stopPropagation();
      app.deleteCard(node.id);
    });
  }

  bindDisplayOpts(app, node, el);
  bindResultStages(app, node, el);
  bindMeshViewer(app, node, el);

  // Photo replace button listener
  const btnReplacePhoto = el.querySelector(".flow-photo-replace-btn");
  if (btnReplacePhoto) {
    btnReplacePhoto.addEventListener("click", (e) => {
      e.stopPropagation();
      app.replacePhotoTargetId = node.id;
      if (app.fileInput) app.fileInput.click();
    });
  }

  bindParamEditor(app, node, el);
  app.attachBlueprintPinListeners(node, el);
  bindResultTabs(app, node, el);
  bindPinsCollapse(app, node, el);

  return el;
}

/** Bind column collapse toggles (called on create and pins refresh). */
export function bindPinsCollapse(app, node, el) {
  el.querySelectorAll("[data-collapse-side]").forEach(btn => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (btn.dataset.collapseSide === "inputs") {
        node.inputsCollapsed = !node.inputsCollapsed;
      } else {
        node.outputsCollapsed = !node.outputsCollapsed;
      }
      app.renderCanvas();
    });
  });
}

/** Bind overlay display toggles (called on create and pins refresh). */
export function bindDisplayOpts(app, node, el) {
  el.querySelectorAll(".flow-opt-chip").forEach(chip => {
    chip.addEventListener("click", (e) => {
      e.stopPropagation();
      const opt = chip.dataset.opt;
      if (node.displayOpts && typeof node.displayOpts[opt] !== "undefined") {
        node.displayOpts[opt] = !node.displayOpts[opt];
        chip.classList.toggle("active", node.displayOpts[opt]);
        updateOverlaySvg(node, el);
        app.renderInspectorContent();
      }
    });
  });
}

/**
 * Bind result tabs (called on create and pins refresh). Tab switches are
 * direct DOM toggles; expanding trace blocks re-routes wires (height change).
 */
export function bindResultTabs(app, node, el) {
  const bar = el.querySelector(".flow-result-tabbar");
  if (bar) {
    const paint = () => {
      const collapsed = !!node.resultCollapsed;
      const active = node.activeResultTab || "preview";
      bar.querySelectorAll(".flow-result-tab[data-rtab]").forEach(b => {
        b.classList.toggle("active", b.dataset.rtab === active);
      });
      el.querySelectorAll(".flow-result-tabpage").forEach(page => {
        page.hidden = collapsed || page.dataset.rtabPage !== active;
      });
      const chev = bar.querySelector("[data-rtab-toggle]");
      if (chev) chev.textContent = collapsed ? "▸" : "▾";
      app.updateConnectedEdges(node.id);
    };
    bar.querySelectorAll(".flow-result-tab[data-rtab]").forEach(btn => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        node.activeResultTab = btn.dataset.rtab;
        node.resultCollapsed = false;
        paint();
      });
    });
    const toggle = bar.querySelector("[data-rtab-toggle]");
    if (toggle) {
      toggle.addEventListener("click", (e) => {
        e.stopPropagation();
        node.resultCollapsed = !node.resultCollapsed;
        paint();
      });
    }
  }
  el.querySelectorAll(".flow-result-tabpage details").forEach(d => {
    d.addEventListener("toggle", () => app.updateConnectedEdges(node.id));
  });
}

/**
 * Re-attempt a card image stuck on its fallback (e.g. it errored while the
 * backend was down). Called on selection, so the card heals the moment the
 * user looks at it. Returns true when a retry was started.
 */
export function retryCardImage(app, nodeId) {
  const node = app.nodes.find(n => n.id === nodeId);
  const el = document.getElementById(nodeId);
  if (!node || !el) return false;
  let retried = false;
  el.querySelectorAll(".flow-node-img").forEach(img => {
    const primary = img.hasAttribute("data-result-img")
      ? node.resultImage
      : (node.nodeType === "asset" ? resolveAssetImageUrl(app, node) : node.imgUrl);
    if (!primary) return;
    if (img.dataset.state !== "fallback" && img.getAttribute("src") === primary) return;
    img.dataset.state = "primary";
    const fallback = img.hasAttribute("data-result-img")
      ? (node.resultImageFallback || "")
      : (node.imgFallback || "");
    img.onerror = () => {
      img.onerror = null;
      img.dataset.state = "fallback";
      img.src = fallback;
    };
    img.src = primary;
    retried = true;
  });
  return retried;
}

/** Bind the input-level concept/prompt editor (called on create and pins refresh). */
export function bindParamEditor(app, node, el) {
  const paramInput = el.querySelector(".flow-param-input");
  if (paramInput) {
    paramInput.addEventListener("click", (e) => e.stopPropagation());
    paramInput.addEventListener("mousedown", (e) => e.stopPropagation());
    paramInput.addEventListener("input", (e) => {
      node.param = e.target.value;
      const promptPin = node.inputs?.find(inp => inp.id === "in-prompt");
      if (promptPin) {
        promptPin.badge = `'${e.target.value}'`;
        const badgeEl = el.querySelector(`.blueprint-pin[data-port-id="in-prompt"] .blueprint-pin-meta`);
        if (badgeEl) badgeEl.textContent = `'${e.target.value}'`;
      }
      el.querySelectorAll(".flow-param-chip").forEach(ch => {
        ch.classList.toggle("active", ch.dataset.concept === e.target.value.trim());
      });
    });
    paramInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        app.runCard(node.id);
      }
    });
  }

  el.querySelectorAll(".flow-param-chip").forEach(chip => {
    chip.addEventListener("click", (e) => {
      e.stopPropagation();
      // Framing chips (multiview row 2) write node.framing; active state
      // stays inside the clicked row so the count row is never cleared.
      const row = chip.closest(".flow-param-chips") || el;
      if (chip.dataset.framing) {
        node.framing = chip.dataset.framing;
        row.querySelectorAll(".flow-param-chip").forEach(ch => ch.classList.toggle("active", ch === chip));
        return;
      }
      const concept = chip.dataset.concept;
      if (paramInput) {
        paramInput.value = concept;
      }
      node.param = concept;
      row.querySelectorAll(".flow-param-chip").forEach(ch => ch.classList.remove("active"));
      chip.classList.add("active");
      const promptPin = node.inputs?.find(inp => inp.id === "in-prompt");
      if (promptPin) {
        promptPin.badge = `'${concept}'`;
        const badgeEl = el.querySelector(`.blueprint-pin[data-port-id="in-prompt"] .blueprint-pin-meta`);
        if (badgeEl) badgeEl.textContent = `'${concept}'`;
      }
    });
  });
}

export function updateNodeElement(app, el, node) {
  if (node.nodeType === "note") {
    el.style.left = `${node.x}px`;
    el.style.top = `${node.y}px`;
    el.style.width = `${node.noteWidth || 320}px`;
    el.classList.toggle("selected", node.id === app.selectedNodeId);
    const t = el.querySelector(".flow-note-text");
    if (t && document.activeElement !== t && t.innerText !== (node.noteText || "")) {
      t.innerText = node.noteText || "";
    }
    return;
  }
  el.style.left = `${node.x}px`;
  el.style.top = `${node.y}px`;
  el.classList.toggle("selected", node.id === app.selectedNodeId);
  if (node.styleKind) el.classList.add(`style-${node.styleKind}`);
  if (node.bigCard) { el.classList.add("flow-node-big"); el.style.width = "640px"; }

  // Card width tracks wide images (dims often resolve after creation).
  const updShowsImage = node.nodeType === "asset"
    ? !!resolveAssetImageUrl(app, node)
    : !!node.resultImage;
  const updWide = updShowsImage ? cardWidthForImage(node.imageWidth, node.imageHeight) : null;
  const wantWidth = updWide ? `${updWide}px` : "";
  if (el.style.width !== wantWidth) el.style.width = wantWidth;

  // Update status strip
  const dot = el.querySelector(".status-dot");
  const statusText = el.querySelector(".flow-node-run-strip .status-indicator span:last-child");
  if (dot && statusText) {
    dot.className = `status-dot ${statusDotClass(node.status)}`;
    statusText.textContent = node.status;
  }

  // The header band follows the run outcome (actions only).
  const header = el.querySelector(".flow-node-header");
  if (header) {
    header.className = `flow-node-header ${node.nodeType === "action" ? headerStatusClass(node.status) : ""}`;
  }

  // The progress block follows the Running state (polls drive it directly).
  syncProgressBlock(el, node);

  // Refresh blueprint pins only when the markup actually changed. Identical
  // markup skips the rebuild entirely: no re-parse, and listeners plus typing
  // focus survive untouched. The pins section hosts the concept/prompt
  // editor, so a real rebuild still preserves typing focus across it.
  let pinsSection = el.querySelector(".blueprint-pins-section");
  const newPinsHtml = renderBlueprintPinsHtml(node, app.poseTopology);
  let caret = null;
  if (!sameHtml(el, newPinsHtml)) {
    const focusedInput = pinsSection?.querySelector(".flow-param-input:focus");
    caret = focusedInput ? [focusedInput.selectionStart, focusedInput.selectionEnd] : null;
    if (pinsSection) {
      pinsSection.outerHTML = newPinsHtml;
      app.attachBlueprintPinListeners(node, el);
      bindParamEditor(app, node, el);
      bindDisplayOpts(app, node, el);
      bindResultStages(app, node, el);
      bindMeshViewer(app, node, el);
      bindResultTabs(app, node, el);
      bindPinsCollapse(app, node, el);
    } else if (newPinsHtml) {
      const strip = el.querySelector(".flow-node-run-strip");
      if (strip) {
        strip.insertAdjacentHTML("beforebegin", newPinsHtml);
        app.attachBlueprintPinListeners(node, el);
        bindParamEditor(app, node, el);
        bindDisplayOpts(app, node, el);
        bindResultStages(app, node, el);
        bindMeshViewer(app, node, el);
        bindResultTabs(app, node, el);
        bindPinsCollapse(app, node, el);
      }
    }
  }

  // Sync card images when the underlying URL changed. Images sitting on
  // fallback are left alone here (no retry storms while offline) — selection
  // retries them deliberately via retryCardImage.
  el.querySelectorAll(".flow-node-img").forEach(img => {
    const primary = img.hasAttribute("data-result-img")
      ? node.resultImage
      : (node.nodeType === "asset" ? resolveAssetImageUrl(app, node) : node.imgUrl);
    if (!primary || img.dataset.state === "fallback") return;
    if (img.getAttribute("src") !== primary) img.src = primary;
  });
  // Asset badges follow the wiring too ("← Box 2" while viewing upstream).
  if (node.nodeType === "asset") {
    const badgeEl = el.querySelector(".flow-node-img-badge");
    if (badgeEl) {
      const badge = assetViewBadge(app, node);
      if (badgeEl.textContent !== badge) badgeEl.textContent = badge;
    }
  }
  if (caret) {
    const restored = el.querySelector(".flow-param-input");
    if (restored) {
      restored.focus();
      try { restored.setSelectionRange(caret[0], caret[1]); } catch { /* non-text input */ }
    }
  }
}

/** Bare unlabeled edge ports for minimized single-pin asset sides. */
export function renderMinimalPortsHtml(node) {
  const ports = [];
  if (minimizePinSide(node.nodeType, node.inputs)) {
    ports.push(renderMinimalPortHtml(node, node.inputs[0], "in"));
  }
  if (minimizePinSide(node.nodeType, node.outputs)) {
    ports.push(renderMinimalPortHtml(node, node.outputs[0], "out"));
  }
  return ports.join("");
}

function renderMinimalPortHtml(node, pin, dir) {
  const cls = dir === "in" ? "port-input" : "port-output";
  const verb = dir === "in" ? "Wire into" : "Drag wire from";
  const color = pin.color || pinTypeColor(pin.type);
  return `<div class="blueprint-port-circle ${cls} port-minimal" data-port-id="${escapeHtml(pin.id)}" data-port-dir="${dir}" data-node-id="${escapeHtml(node.id)}" title="${verb}: ${escapeHtml(pin.name)}" style="border-color:${color};color:${color}"><div class="blueprint-port-inner"></div></div>`;
}

/** Concept/prompt editor, rendered at input level inside the INPUTS column. */
export function renderParamEditorHtml(node) {
  const val = node.param !== undefined ? node.param : node.promptText;
  const isSam = node.id.includes("sam") || (node.templateKey && node.templateKey.includes("sam"));
  const isMultiview = node.templateKey && node.templateKey.includes("multiview");
  const isBodyFit = node.templateKey && node.templateKey === "action-body-fit";
  const chipsHtml = isSam ? `
    <div class="flow-param-chips">
      <button type="button" class="flow-param-chip ${val === 'people' ? 'active' : ''}" data-concept="people">people</button>
      <button type="button" class="flow-param-chip ${val === 'person' ? 'active' : ''}" data-concept="person">person</button>
      <button type="button" class="flow-param-chip ${val === 'face' ? 'active' : ''}" data-concept="face">face</button>
      <button type="button" class="flow-param-chip ${val === 'clothes' ? 'active' : ''}" data-concept="clothes">clothes</button>
    </div>
  ` : "";
  const viewsHtml = isMultiview ? `
    <div class="flow-param-chips">
      <button type="button" class="flow-param-chip ${String(val) === '2' ? 'active' : ''}" data-concept="2">2</button>
      <button type="button" class="flow-param-chip ${String(val) === '4' ? 'active' : ''}" data-concept="4">4</button>
      <button type="button" class="flow-param-chip ${String(val) === '6' ? 'active' : ''}" data-concept="6">6</button>
      <button type="button" class="flow-param-chip ${String(val) === '8' ? 'active' : ''}" data-concept="8">8</button>
    </div>
  ` : "";

  if (isBodyFit) {
    const exprOn = String(node.param || "on") !== "off";
    return `
    <div class="flow-action-param-box">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:2px">
        <span class="flow-param-title">Expression fit:</span>
        <span style="font-size:9px;color:var(--text-subtle);font-family:var(--font-mono)">72 MHR coefficients</span>
      </div>
      <div class="flow-param-chips">
        <button type="button" class="flow-param-chip ${exprOn ? 'active' : ''}" data-concept="on">ON (jaw/lips solved)</button>
        <button type="button" class="flow-param-chip ${!exprOn ? 'active' : ''}" data-concept="off">OFF (neutral)</button>
      </div>
      <div style="font-size:9px;color:var(--text-subtle);margin-top:4px;font-family:var(--font-mono)">identity 45 · pose 130 · scale 28 · camera 3</div>
    </div>
  `;
  }

  if (isMultiview) {
    const frameVal = node.framing || "body";
    const frameChip = (v, label) =>
      `<button type="button" class="flow-param-chip ${frameVal === v ? 'active' : ''}" data-framing="${v}">${label}</button>`;
    return `
    <div class="flow-action-param-box">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:2px">
        <span class="flow-param-title">Views:</span>
        <span style="font-size:9px;color:var(--text-subtle);font-family:var(--font-mono)">even azimuths from front</span>
      </div>
      ${viewsHtml}
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:2px;margin-top:6px">
        <span class="flow-param-title">Frame:</span>
        <span style="font-size:9px;color:var(--text-subtle);font-family:var(--font-mono)">subject window</span>
      </div>
      <div class="flow-param-chips">
        ${frameChip("body", "Body")}${frameChip("waist", "Waist-up")}${frameChip("face", "Face")}
      </div>
    </div>
  `;
  }

  return `
    <div class="flow-action-param-box">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:2px">
        <span class="flow-param-title">${isSam ? "Segment Concept:" : "Prompt / Concept:"}</span>
        <span style="font-size:9px;color:var(--text-subtle);font-family:var(--font-mono)">${isSam ? "sam-3.1 noun phrase" : "prompt"}</span>
      </div>
      <input type="text" class="flow-param-input" value="${escapeHtml(val)}" placeholder="${isSam ? "e.g. people, person, face, clothes" : "Type a prompt, or wire one in"}" />
      ${chipsHtml}
    </div>
  `;
}

/** Overlay toggle toolbar (masks / boxes / labels / confidence). */
export function renderDisplayOptsHtml(node) {
  if (!node.displayOpts) return "";
  return `
    <div class="flow-display-opts" data-node-id="${node.id}">
      <span class="flow-display-opts-title">Display:</span>
      <button class="flow-opt-chip ${node.displayOpts.showMasks ? 'active' : ''}" data-opt="showMasks" title="Toggle Segmentation Masks">
        <span class="flow-opt-dot"></span>Masks
      </button>
      <button class="flow-opt-chip ${node.displayOpts.showBoxes ? 'active' : ''}" data-opt="showBoxes" title="Toggle Bounding Boxes">
        <span class="flow-opt-dot"></span>Boxes
      </button>
      <button class="flow-opt-chip ${node.displayOpts.showLabels ? 'active' : ''}" data-opt="showLabels" title="Toggle Object Labels">
        <span class="flow-opt-dot"></span>Labels
      </button>
      <button class="flow-opt-chip ${node.displayOpts.showConfidence ? 'active' : ''}" data-opt="showConfidence" title="Toggle Confidence Scores">
        <span class="flow-opt-dot"></span>Confidence
      </button>
    </div>
  `;
}

/**
 * In-card result view for model cards, under OUTPUTS: tabbed Preview / API /
 * Logs. Renders once the card has a result image or a captured API trace.
 */
export function renderResultPreviewHtml(node, topology = null) {
  const hasHistory = node.runLog && node.runLog.length > 0;
  if (!node.resultImage && !node.apiCall && !hasHistory) return "";
  const active = node.activeResultTab || "preview";

  const overlayW = Number(node.imageWidth) || 1000;
  const overlayH = Number(node.imageHeight) || 1000;
  const overlaySvg = node.masks ? `
    <svg class="flow-detection-overlay" viewBox="0 0 ${overlayW} ${overlayH}" preserveAspectRatio="none">
      ${renderOverlaySvg(node)}
    </svg>
  ` : "";
  const isPoseStage = node.templateKey === "action-sapiens-pose"
    && Array.isArray(node.resultKeypoints) && node.resultKeypoints.length > 0;
  const isSegStage = node.templateKey === "action-sapiens-seg" && !!node.resultSegLegend;
  const stageAttr = isPoseStage ? ` data-kpts-stage="1"` : (isSegStage ? ` data-seg-stage="1"` : "");
  const kptsSvg = isPoseStage ? `
    <svg class="flow-kpts-overlay" viewBox="0 0 ${overlayW} ${overlayH}" preserveAspectRatio="none">
      ${renderPoseSvgInner(node, topology)}
    </svg>
    <div class="flow-chart-tip" hidden></div>
  ` : "";
  const segCanvas = isSegStage ? `
    <canvas class="flow-seg-canvas" width="${Number(node.resultSegLegend.width) || 16}" height="${Number(node.resultSegLegend.height) || 16}" hidden></canvas>
    <div class="flow-chart-tip" hidden></div>
  ` : "";
  const segLegend = isSegStage ? renderSegLegendHtml(node) : "";
  const meshStage = shouldMountMeshViewer(node)
    ? renderMeshStageHtml(node, node.resultImage || null) : "";
  const ratioAttr = aspectRatioStyle(node.imageWidth, node.imageHeight);
  const imageBlock = node.resultImage ? `
    <div class="flow-node-img-wrapper"${stageAttr}${ratioAttr ? ` style="${ratioAttr}"` : ""}>
      <img class="flow-node-img" data-result-img="1" src="${escapeHtml(node.resultImage)}" data-state="primary" onerror="this.onerror=null;this.dataset.state='fallback';this.src='${escapeHtml(node.resultImageFallback || "")}'" alt="${escapeHtml(node.title)} result" title="Double-click for full size">
      ${overlaySvg}${kptsSvg}${segCanvas}
      <span class="flow-node-img-badge">${escapeHtml(node.resultImageBadge || "")}</span>
    </div>
  ` : "";
  // Text-producing cards (Spark Describe) show their narration under the image.
  const textBlock = node.resultText
    ? `<div class="flow-result-text">${escapeHtml(node.resultText)}</div>`
    : "";
  const linksBlock = (Array.isArray(node.resultLinks) && node.resultLinks.length)
    ? `<div class="flow-result-links">${node.resultLinks.map(l => `
        <a class="flow-result-link" href="${escapeHtml(l.url)}" target="_blank" rel="noopener"${l.download ? " download" : ""} title="${escapeHtml(l.url)}">${escapeHtml(l.label)}</a>`).join("")}</div>`
    : "";
  const errorBlock = (node.status === "Failed" && node.runError)
    ? `<div class="flow-error-block"><div class="flow-error-title">✕ Run failed</div><div class="flow-error-msg">${escapeHtml(node.runError)}</div></div>`
    : "";
  const previewPage = (imageBlock || textBlock || errorBlock || linksBlock)
    ? `${errorBlock}${imageBlock}${segLegend}${meshStage}${linksBlock}${textBlock}${renderDisplayOptsHtml(node)}`
    : `<div class="flow-api-empty">No preview — the run produced no image.</div>`;

  const collapsed = !!node.resultCollapsed;
  const tab = (id, label) => `
    <button type="button" class="flow-result-tab${active === id ? " active" : ""}" data-rtab="${id}">${label}</button>`;
  const pageHidden = (id) => (collapsed || active !== id ? " hidden" : "");

  return `
    <div class="flow-result-preview">
      <div class="flow-result-tabbar" role="tablist">
        ${tab("preview", "Preview")}
        ${tab("api", "API")}
        ${tab("logs", "Logs")}
        <button type="button" class="flow-result-tab flow-result-collapse" data-rtab-toggle title="Collapse section">${collapsed ? "▸" : "▾"}</button>
      </div>
      <div class="flow-result-tabpage" data-rtab-page="preview"${pageHidden("preview")}>
        ${previewPage}
      </div>
      <div class="flow-result-tabpage" data-rtab-page="api"${pageHidden("api")}>
        ${renderApiTraceHtml(node.apiCall)}
      </div>
      <div class="flow-result-tabpage" data-rtab-page="logs"${pageHidden("logs")}>
        ${renderRunLogHtml(node.runLog)}
      </div>
    </div>
  `;
}

/**
 * Compact port-dot strip for a collapsed pin column. Circles keep their data
 * attributes, so wires stay attached, routable and draggable while collapsed.
 */
export function renderPinsStrip(node, pins, dir) {
  const cls = dir === "in" ? "port-input" : "port-output";
  const circles = pins.map(p => {
    const color = p.color || pinTypeColor(p.type);
    return `
    <div class="blueprint-port-circle ${cls}" data-port-id="${escapeHtml(p.id)}" data-port-dir="${dir}" data-node-id="${escapeHtml(node.id)}" title="${escapeHtml(p.name)}${p.badge ? ` · ${escapeHtml(p.badge)}` : ""}" style="border-color:${color};color:${color}">
      <div class="blueprint-port-inner"></div>
    </div>`;
  }).join("");
  return `<div class="blueprint-pins-strip">${circles}</div>`;
}

export function renderBlueprintPinsHtml(node, topology = null) {
  // Minimized sides render as bare edge ports instead of labeled rows.
  const inputs = minimizePinSide(node.nodeType, node.inputs) ? [] : (node.inputs || []);
  const outputs = minimizePinSide(node.nodeType, node.outputs) ? [] : (node.outputs || []);

  if (inputs.length === 0 && outputs.length === 0) return "";

  const inCollapsed = !!node.inputsCollapsed;
  const outCollapsed = !!node.outputsCollapsed;

  let inputsHtml = "";
  if (inputs.length > 0) {
    const pinRows = inputs.map(inp => {
      const dotColor = inp.color || pinTypeColor(inp.type);
      const badgeHtml = inp.badge ? `<span class="blueprint-pin-meta">${escapeHtml(inp.badge)}</span>` : "";
      return `
        <div class="blueprint-pin input-pin" data-port-id="${escapeHtml(inp.id)}" data-port-dir="in" title="Input Pin: ${escapeHtml(inp.name)}">
          <div class="blueprint-port-circle port-input" data-port-id="${escapeHtml(inp.id)}" data-port-dir="in" data-node-id="${escapeHtml(node.id)}" title="Connect to input: ${escapeHtml(inp.name)}">
            <div class="blueprint-port-inner"></div>
          </div>
          <div class="blueprint-pin-info">
            <span class="blueprint-pin-dot" style="background:${dotColor}; color:${dotColor}"></span>
            <span class="blueprint-pin-name">${escapeHtml(inp.name)}</span>
            ${badgeHtml}
          </div>
        </div>
      `;
    }).join("");

    const paramEditorHtml = (node.param !== undefined || node.promptText !== undefined)
      ? renderParamEditorHtml(node)
      : "";
    inputsHtml = `
      <div class="blueprint-pins-col inputs-col${inCollapsed ? " collapsed" : ""}">
        <button type="button" class="blueprint-col-title blueprint-col-toggle" data-collapse-side="inputs" title="Collapse section">
          <span class="sec-label"><span class="sec-dot"></span><span>INPUTS (${inputs.length})</span></span><span class="chev">${inCollapsed ? "▸" : "▾"}</span>
        </button>
        ${inCollapsed ? renderPinsStrip(node, inputs, "in") : `${pinRows}${paramEditorHtml}`}
      </div>
    `;
  }

  let outputsHtml = "";
  if (outputs.length > 0) {
    const pinRows = outputs.map(out => {
      const dotColor = out.color || pinTypeColor(out.type);
      const badgeHtml = out.badge ? `<span class="blueprint-pin-meta">${escapeHtml(out.badge)}</span>` : "";
      const thumbHtml = out.thumb
        ? `<img class="blueprint-pin-thumb" src="${escapeHtml(out.thumb)}" alt="" loading="lazy" onerror="this.style.display='none'">`
        : (out.thumbValue ? `<span class="blueprint-pin-thumbvalue">${escapeHtml(out.thumbValue)}</span>` : "");
      return `
        <div class="blueprint-pin output-pin" data-port-id="${escapeHtml(out.id)}" data-port-dir="out" title="Drag wire from output pin: ${escapeHtml(out.name)}">
          <div class="blueprint-pin-info">
            ${badgeHtml}
            ${thumbHtml}
            <span class="blueprint-pin-name">${escapeHtml(out.name)}</span>
            <span class="blueprint-pin-dot" style="background:${dotColor}; color:${dotColor}"></span>
          </div>
          <div class="blueprint-port-circle port-output" data-port-id="${escapeHtml(out.id)}" data-port-dir="out" data-node-id="${escapeHtml(node.id)}" title="Drag connection from: ${escapeHtml(out.name)}">
            <div class="blueprint-port-inner"></div>
          </div>
        </div>
      `;
    }).join("");

    outputsHtml = `
      <div class="blueprint-pins-col outputs-col${outCollapsed ? " collapsed" : ""}">
        <button type="button" class="blueprint-col-title blueprint-col-toggle" data-collapse-side="outputs" title="Collapse section">
          <span class="sec-label"><span class="sec-dot"></span><span>OUTPUTS (${outputs.length})</span></span><span class="chev">${outCollapsed ? "▸" : "▾"}</span>
        </button>
        ${outCollapsed ? renderPinsStrip(node, outputs, "out") : pinRows}
      </div>
    `;
  }

  const resultHtml = renderResultPreviewHtml(node, topology);
  const divider = `<div class="blueprint-pins-divider"></div>`;
  const parts = [inputsHtml, outputsHtml, resultHtml].filter(s => s);

  return `
    <div class="blueprint-pins-section stacked">
      ${parts.join(divider)}
    </div>
  `;
}

export function renderOverlaySvg(node) {
  if (!node.masks || node.masks.length === 0) return "";
  const opts = node.displayOpts || { showMasks: true, showBoxes: true, showLabels: true, showConfidence: true };

  const labels = (node.resultData && Array.isArray(node.resultData.labels))
    ? node.resultData.labels
    : [];
  const concept = (node.resultData && node.resultData.concept) || null;

  return node.masks.map((m, idx) => {
    const bx = m.box;
    if (!bx || bx.length < 4) return "";
    const x = Math.round(bx[0]);
    const y = Math.round(bx[1]);
    const w = Math.round(bx[2] - bx[0]);
    const h = Math.round(bx[3] - bx[1]);
    const conf = (m.confidence ? (m.confidence * 100).toFixed(1) : "98.5") + "%";
    const label = labels[idx] || (concept ? `${concept} ${idx + 1}` : `P${idx + 1}`);

    let maskRect = "";
    if (opts.showMasks) {
      maskRect = `<rect class="flow-det-mask" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>`;
    }

    let boxRect = "";
    if (opts.showBoxes) {
      boxRect = `<rect class="flow-det-box" x="${x}" y="${y}" width="${w}" height="${h}" rx="4"/>`;
    }

    let labelTag = "";
    if (opts.showLabels) {
      const tagY = Math.max(22, y - 6);
      const tagWidth = label.length * 8 + 14;
      labelTag = `
        <g class="flow-det-label-group">
          <rect class="flow-det-tag-bg" x="${x}" y="${tagY - 18}" width="${tagWidth}" height="20" rx="3"/>
          <text class="flow-det-tag-text" x="${x + 6}" y="${tagY - 4}">${label}</text>
        </g>
      `;
    }

    let confTag = "";
    if (opts.showConfidence) {
      const confY = Math.max(22, y - 6);
      const offsetX = opts.showLabels ? (label.length * 8 + 18) : 0;
      confTag = `
        <g class="flow-det-label-group">
          <rect class="flow-det-conf-bg" x="${x + offsetX}" y="${confY - 18}" width="62" height="20" rx="3"/>
          <text class="flow-det-conf-text" x="${x + offsetX + 6}" y="${confY - 4}">${conf}</text>
        </g>
      `;
    }

    return `${maskRect}${boxRect}${labelTag}${confTag}`;
  }).join("");
}

export function updateOverlaySvg(node, cardEl) {
  const overlay = cardEl.querySelector(".flow-detection-overlay");
  if (overlay) {
    overlay.innerHTML = renderOverlaySvg(node);
  }
}

export function updateHeaderStats(app) {
  const activeCount = app.nodes.length;
  const actionCount = app.nodes.filter(n => n.nodeType === "action").length;
  const assetCount = app.nodes.filter(n => n.nodeType === "asset").length;

  const pill1 = document.getElementById("stat-stage-pill");
  const pill2 = document.getElementById("stat-candidate-pill");
  if (pill1) pill1.textContent = `${activeCount} Active Nodes`;
  if (pill2) pill2.textContent = `${actionCount} Models · ${assetCount} Assets`;
}
