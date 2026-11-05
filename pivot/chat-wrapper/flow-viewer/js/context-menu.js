/**
 * Blueprint Right-Click Context Menu Engine
 * Displays categorized models with fuzzy search on canvas background right-click,
 * and card-specific actions (Run, Duplicate, Tidy, Inspect, Delete) on node right-click.
 */

import { MODEL_CATALOG } from "./catalog.js";
import { uid, isListType, escapeHtml } from "./util.js";
import { deleteEdge } from "./wires.js";

export function initContextMenu(app) {
  // Right-click context menu on canvas viewport
  app.viewport.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    e.stopPropagation();

    const planeRect = app.plane.getBoundingClientRect();
    const canvasX = Math.round((e.clientX - planeRect.left) / app.scale);
    const canvasY = Math.round((e.clientY - planeRect.top) / app.scale);

    // Right-clicked on an output pin gets the pin menu (explode/collapse);
    // anything else on a card gets the card menu.
    const pinEl = e.target.closest(".blueprint-pin[data-port-id], .blueprint-port-circle[data-port-id]");
    const pinCard = pinEl ? pinEl.closest(".flow-node") : null;
    if (pinEl && pinCard && pinEl.dataset.portDir === "out") {
      openPinMenu(app, e.clientX, e.clientY, pinCard.id, pinEl.dataset.portId, canvasX, canvasY);
      return;
    }

    // Right-clicked on a wire (path or label) gets the wire menu.
    const wireEl = e.target.closest("[data-edge-id]");
    if (wireEl) {
      openWireMenu(app, e.clientX, e.clientY, wireEl.dataset.edgeId);
      return;
    }

    const targetCard = e.target.closest(".flow-node");
    if (targetCard) {
      openContextMenu(app, e.clientX, e.clientY, canvasX, canvasY, targetCard.id);
    } else {
      openContextMenu(app, e.clientX, e.clientY, canvasX, canvasY, null);
    }
  });

  // Dismiss context menus and popovers on Escape
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeContextMenu();
      app.closePopover();
    }
  });
}

export function closeContextMenu() {
  const existing = document.getElementById("active-flow-context-menu");
  if (existing) {
    existing.remove();
  }
}

export function openContextMenu(app, screenX, screenY, canvasX, canvasY, targetNodeId = null) {
  closeContextMenu();
  app.closePopover();

  const menuEl = document.createElement("div");
  menuEl.id = "active-flow-context-menu";
  menuEl.className = "flow-context-menu";

  if (targetNodeId) {
    // Right-clicked on an existing node
    const node = app.nodes.find(n => n.id === targetNodeId);
    if (!node) return;

    const isAction = node.nodeType === "action";
    menuEl.innerHTML = `
      <div class="flow-context-header">
        <div style="font-size:12px;font-weight:700;color:var(--text);display:flex;align-items:center;gap:6px">
          <span style="color:var(--primary)">✦</span>
          <span>${node.title}</span>
        </div>
        <div style="font-size:10px;font-family:var(--font-mono);color:var(--text-muted)">${node.subtitle} · ${node.category}</div>
      </div>
      <div class="flow-context-body">
        ${isAction ? `
          <button class="flow-context-item" data-action="run">
            <span style="color:var(--primary);font-size:12px;width:16px;text-align:center">▶</span>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Run / Execute Model</div>
              <div class="flow-context-item-sub">Execute forward pass and produce downstream assets</div>
            </div>
          </button>
        ` : ""}
        <button class="flow-context-item" data-action="duplicate">
          <span style="color:#06b6d4;font-size:12px;width:16px;text-align:center">📋</span>
          <div class="flow-context-item-info">
            <div class="flow-context-item-title">Duplicate Node</div>
            <div class="flow-context-item-sub">Create copy with identical configuration</div>
          </div>
        </button>
        <button class="flow-context-item" data-action="inspect">
          <span style="color:#a855f7;font-size:12px;width:16px;text-align:center">🔍</span>
          <div class="flow-context-item-info">
            <div class="flow-context-item-title">Inspect Details</div>
            <div class="flow-context-item-sub">Open in side drawer inspector</div>
          </div>
        </button>
        <div class="flow-context-divider"></div>
        <button class="flow-context-item" data-action="tidy">
          <span style="color:#f59e0b;font-size:12px;width:16px;text-align:center">🪄</span>
          <div class="flow-context-item-info">
            <div class="flow-context-item-title">Auto-align Layout</div>
            <div class="flow-context-item-sub">Tidy up cards topologically</div>
          </div>
        </button>
        <div class="flow-context-divider"></div>
        <button class="flow-context-item" data-action="delete" style="color:var(--destructive)">
          <span style="font-size:12px;width:16px;text-align:center">✕</span>
          <div class="flow-context-item-info">
            <div class="flow-context-item-title" style="color:var(--destructive)">Delete Node</div>
            <div class="flow-context-item-sub">Remove card and all attached wires</div>
          </div>
        </button>
      </div>
    `;

    menuEl.querySelectorAll(".flow-context-item").forEach(item => {
      item.addEventListener("click", () => {
        const act = item.dataset.action;
        if (act === "run") app.runCard(targetNodeId);
        else if (act === "duplicate") duplicateNode(app, targetNodeId, canvasX + 30, canvasY + 30);
        else if (act === "inspect") {
          app.selectNode(targetNodeId);
          app.inspector.classList.remove("collapsed");
        }
        else if (act === "tidy") app.tidyLayout();
        else if (act === "delete") app.deleteCard(targetNodeId);
        closeContextMenu();
      });
    });

  } else {
    // Right-clicked on canvas background
    menuEl.innerHTML = `
      <div class="flow-context-header">
        <div class="flow-context-search-wrapper">
          <svg class="flow-context-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
          <input type="text" id="context-search-input" class="flow-context-search-input" placeholder="Search models, actions & tools..." autocomplete="off">
        </div>
      </div>
      <div class="flow-context-body" id="context-items-list">
        <!-- Meta AI Foundation -->
        <div class="flow-context-group" data-group="meta">
          <div class="flow-context-group-title">
            <span>✦ Meta AI Foundation</span>
            <span class="reui-badge" data-variant="meta-light" data-size="xs">Cloud API</span>
          </div>
          <button class="flow-context-item" data-model-key="action-meta-sam" data-search="meta sam 3.1 segment anything media segmentation bboxes masks">
            <div class="reui-icon-tile" data-variant="meta" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Meta SAM 3.1</div>
              <div class="flow-context-item-sub">sam-3.1 · Noun Concept Segmentation</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-describe" data-search="meta describe vision scene caption whole image appearance read">
            <div class="reui-icon-tile" data-variant="meta" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Vision Describe</div>
              <div class="flow-context-item-sub">Meta Describe · Whole-Scene Read</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-meta-muse" data-search="meta muse image generate edit prompt picture responses api">
            <div class="reui-icon-tile" data-variant="meta" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M20.4 14.5L16 10 4 20"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Meta Muse Image</div>
              <div class="flow-context-item-sub">muse-image-1.0 · Generate & Edit</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-spark" data-search="meta spark prompt chat text vision image multimodal llm">
            <div class="reui-icon-tile" data-variant="meta" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Meta Spark</div>
              <div class="flow-context-item-sub">muse-spark · Prompt + Images → Text</div>
            </div>
          </button>
        </div>

        <div class="flow-context-divider"></div>

        <!-- Local Tools (no API calls) -->
        <div class="flow-context-group" data-group="local">
          <div class="flow-context-group-title">
            <span>✦ Local Tools</span>
            <span class="reui-badge" data-variant="meta-light" data-size="xs">This Machine</span>
          </div>
          <button class="flow-context-item" data-model-key="action-apply-mask" data-search="apply mask crop region isolate cut out selection photo sam local">
            <div class="reui-icon-tile" data-variant="green" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2" stroke-dasharray="3 2"/><circle cx="12" cy="12" r="4"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Apply Mask</div>
              <div class="flow-context-item-sub">Mask × Photo · Region Crop</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-image-diff" data-search="image difference diff compare subtract delta photos local pixels change">
            <div class="reui-icon-tile" data-variant="green" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="8" height="18" rx="1"/><rect x="13" y="3" width="8" height="18" rx="1"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Image Difference</div>
              <div class="flow-context-item-sub">A − B · Δ Map & Stats</div>
            </div>
          </button>
        </div>

        <div class="flow-context-divider"></div>

        <!-- Sapiens Foundation -->
        <div class="flow-context-group" data-group="sapiens">
          <div class="flow-context-group-title">
            <span>✦ Sapiens Foundation (0.4B)</span>
            <span class="reui-badge" data-variant="meta-light" data-size="xs">RTX 4090</span>
          </div>
          <button class="flow-context-item" data-model-key="action-sapiens-pose" data-search="sapiens pose 308 keypoints anatomical landmarks skeleton face hands">
            <div class="reui-icon-tile" data-variant="meta" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="5" r="2"/><path d="m9 20 3-6 3 6"/><path d="m6 8 6 2 6-2"/><path d="M12 10v4"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">SapiensPose</div>
              <div class="flow-context-item-sub">308 Anatomical Keypoints & Pose Cloud</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-sapiens-seg" data-search="sapiens seg 28-class body part segmentation garments face skin">
            <div class="reui-icon-tile" data-variant="meta" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zm1 14.93V17a1 1 0 0 1-2 0v-.07A8 8 0 0 1 4.07 11H5a1 1 0 0 1 0 2H4.07A8 8 0 0 1 11 4.07V5a1 1 0 0 1 2 0v-.93A8 8 0 0 1 19.93 11H19a1 1 0 0 1 0-2h.93A8 8 0 0 1 13 16.93z"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">SapiensSeg</div>
              <div class="flow-context-item-sub">28-Class Body & Apparel Part Segmentation</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-sapiens-normal" data-search="sapiens normal surface normals orientation vectors skin clothing">
            <div class="reui-icon-tile" data-variant="meta" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 14.14 14.14"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">SapiensNormal</div>
              <div class="flow-context-item-sub">Dense Surface Normal Vector Map [Nx, Ny, Nz]</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-sapiens-depth" data-search="sapiens depth metric distance euclidean pointmap meters 0.5-3.0m">
            <div class="reui-icon-tile" data-variant="meta" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M2 12h20"/><path d="M20 12v6a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-6"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">SapiensDepth</div>
              <div class="flow-context-item-sub">Metric Depth Map (Metres Euclidean)</div>
            </div>
          </button>
        </div>

        <div class="flow-context-divider"></div>

        <!-- Avatar Solver & Exporter -->
        <div class="flow-context-group" data-group="avatar">
          <div class="flow-context-group-title">
            <span>✦ Parametric Avatar Suite</span>
            <span class="reui-badge" data-variant="meta-light" data-size="xs">CUDA / SMPL</span>
          </div>
          <button class="flow-context-item" data-model-key="action-body-fit" data-search="sam 3d body joint solver smpl mhr beta shape coefficients mesh">
            <div class="reui-icon-tile" data-variant="meta" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">SAM 3D Body & Joint Solver</div>
              <div class="flow-context-item-sub">SMPL / MHR 17,821-vertex Avatar Solver</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-export" data-search="character exporter glb gltf 3d package download">
            <div class="reui-icon-tile" data-variant="green" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Character Exporter</div>
              <div class="flow-context-item-sub">Bake & Package GLB 3D Character Asset</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-mesh-render" data-search="mesh renderer render shaded still image three.js local webgl turntable view">
            <div class="reui-icon-tile" data-variant="soft" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><circle cx="12" cy="12" r="3"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Mesh Renderer</div>
              <div class="flow-context-item-sub">Shaded Still · Local three.js</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-mesh-depth" data-search="mesh depth map white near grayscale local webgl three.js geometry">
            <div class="reui-icon-tile" data-variant="soft" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 15l6-6 4 4 3-3 5 5"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Mesh Depth</div>
              <div class="flow-context-item-sub">Depth Map · Local three.js</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-mesh-normal" data-search="mesh normal map normals rgb xyz surface orientation local webgl three.js">
            <div class="reui-icon-tile" data-variant="soft" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="9"/><path d="M12 3v18M3 12h18"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Mesh Normals</div>
              <div class="flow-context-item-sub">Normal Map · Local three.js</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-mesh-multiview" data-search="mesh multiview multi view angles turntable sheet contact photos camera orbit local webgl three.js">
            <div class="reui-icon-tile" data-variant="soft" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="8" height="8" rx="1"/><rect x="13" y="3" width="8" height="8" rx="1"/><rect x="3" y="13" width="8" height="8" rx="1"/><rect x="13" y="13" width="8" height="8" rx="1"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Multi-View Renderer</div>
              <div class="flow-context-item-sub">2/4/6/8 Views · Local three.js</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="action-fit-overlay" data-search="fit overlay matched render photo blend side by side alignment check honest sam 3d">
            <div class="reui-icon-tile" data-variant="meta" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="12" cy="12" r="4"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Fit Overlay</div>
              <div class="flow-context-item-sub">Matched Render × Photo · 50/50</div>
            </div>
          </button>
        </div>

        <div class="flow-context-divider"></div>

        <!-- Canvas Utilities -->
        <div class="flow-context-group" data-group="utils">
          <div class="flow-context-group-title">
            <span>✦ Canvas Utilities</span>
          </div>
          <button class="flow-context-item" data-action="tidy" data-search="tidy align layout clean graph auto-align">
            <span style="color:#f59e0b;font-size:13px;width:16px;text-align:center">🪄</span>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Auto-align / Tidy Layout</div>
              <div class="flow-context-item-sub">Organize nodes into neat DAG columns</div>
            </div>
          </button>
          <button class="flow-context-item" data-action="fit" data-search="fit view zoom center frame all nodes">
            <span style="color:#06b6d4;font-size:13px;width:16px;text-align:center">⛶</span>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Fit View to Nodes</div>
              <div class="flow-context-item-sub">Zoom and center all cards into view</div>
            </div>
          </button>
          <button class="flow-context-item" data-action="new-flow" data-search="new clear reset empty flow canvas photo">
            <span style="color:var(--text-muted);font-size:13px;width:16px;text-align:center">🗑️</span>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Clear Canvas / New Flow</div>
              <div class="flow-context-item-sub">Reset to empty canvas dropzone</div>
            </div>
          </button>
          <button class="flow-context-item" data-model-key="asset-photo-viewer" data-search="photo viewer preview inspect visualize image mid-flow">
            <div class="reui-icon-tile" data-variant="outline" data-size="xs">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
            </div>
            <div class="flow-context-item-info">
              <div class="flow-context-item-title">Photo Viewer</div>
              <div class="flow-context-item-sub">Drop mid-flow, wire in to visualize</div>
            </div>
          </button>
        </div>
      </div>
    `;

    // Item click handler
    menuEl.querySelectorAll(".flow-context-item").forEach(item => {
      item.addEventListener("click", () => {
        const modelKey = item.dataset.modelKey;
        const actionKey = item.dataset.action;

        if (modelKey) {
          spawnModelAt(app, modelKey, canvasX, canvasY);
        } else if (actionKey === "tidy") {
          app.tidyLayout();
        } else if (actionKey === "fit") {
          app.fitView();
        } else if (actionKey === "new-flow") {
          app.startNewFlow();
        }
        closeContextMenu();
      });
    });

    // Filter input search handler
    const searchInput = menuEl.querySelector("#context-search-input");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        const query = e.target.value.trim().toLowerCase();
        const items = menuEl.querySelectorAll(".flow-context-item");
        let firstVisible = null;

        items.forEach(item => {
          const text = (item.dataset.search || "") + " " + (item.innerText || "");
          const match = !query || text.toLowerCase().includes(query);
          item.style.display = match ? "flex" : "none";
          if (match && !firstVisible) {
            firstVisible = item;
          }
        });

        // Show/hide section group titles & dividers
        menuEl.querySelectorAll(".flow-context-group").forEach(grp => {
          const visibleChild = grp.querySelector('.flow-context-item[style*="display: flex"], .flow-context-item:not([style*="display: none"])');
          grp.style.display = visibleChild ? "block" : "none";
        });

        menuEl.querySelectorAll(".flow-context-divider").forEach(div => {
          div.style.display = query ? "none" : "block";
        });

        // Focus indicator on first visible
        items.forEach(it => it.classList.remove("is-focused"));
        if (firstVisible) {
          firstVisible.classList.add("is-focused");
        }
      });

      // Enter key to trigger top visible result
      searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          const focused = menuEl.querySelector(".flow-context-item.is-focused") || 
                          menuEl.querySelector('.flow-context-item:not([style*="display: none"])');
          if (focused) {
            focused.click();
          }
        }
      });
    }
  }

  document.body.appendChild(menuEl);
  placeMenuEl(menuEl, screenX, screenY);

  // Focus search input on canvas right-click
  const input = menuEl.querySelector("#context-search-input");
  if (input) {
    setTimeout(() => input.focus(), 20);
  }
}

/** Clamp a floating menu inside the window viewport. */
export function placeMenuEl(menuEl, screenX, screenY, menuWidth = 320) {
  const menuHeight = Math.min(460, menuEl.offsetHeight || 380);
  let posX = screenX;
  let posY = screenY;

  if (posX + menuWidth > window.innerWidth - 12) {
    posX = Math.max(12, window.innerWidth - menuWidth - 12);
  }
  if (posY + menuHeight > window.innerHeight - 12) {
    posY = Math.max(12, window.innerHeight - menuHeight - 12);
  }

  menuEl.style.left = `${posX}px`;
  menuEl.style.top = `${posY}px`;
}

/**
 * Right-click menu for an output pin. List pins offer UE-style Explode
 * (split into item pins with thumbnails) and Collapse; anything else falls
 * back to the card menu.
 */
export function openPinMenu(app, screenX, screenY, nodeId, portId, canvasX, canvasY) {
  const node = app.nodes.find(n => n.id === nodeId);
  const pin = node && (node.outputs || []).find(p => p.id === portId);
  if (!node || !pin || !isListType(pin.type)) {
    openContextMenu(app, screenX, screenY, canvasX, canvasY, nodeId);
    return;
  }

  closeContextMenu();
  app.closePopover();

  const data = node.resultData && node.resultData[pin.type];
  const count = Array.isArray(data) ? data.length : 0;
  const isSplit = !!(node.splitPins && node.splitPins[portId]);

  const menuEl = document.createElement("div");
  menuEl.id = "active-flow-context-menu";
  menuEl.className = "flow-context-menu";

  let actionHtml = "";
  if (isSplit) {
    actionHtml = `
      <button class="flow-context-item" data-pin-action="collapse">
        <span style="color:#a855f7;font-size:12px;width:16px;text-align:center">▸</span>
        <div class="flow-context-item-info">
          <div class="flow-context-item-title">Collapse list</div>
          <div class="flow-context-item-sub">Remove item pins and their wires</div>
        </div>
      </button>
    `;
  } else if (count > 0) {
    actionHtml = `
      <button class="flow-context-item" data-pin-action="explode">
        <span style="color:var(--primary);font-size:12px;width:16px;text-align:center">▾</span>
        <div class="flow-context-item-info">
          <div class="flow-context-item-title">Explode list (${count})</div>
          <div class="flow-context-item-sub">Split into item pins with previews</div>
        </div>
      </button>
    `;
  } else {
    actionHtml = `
      <button class="flow-context-item" data-pin-action="none" disabled title="Run the card first">
        <span style="font-size:12px;width:16px;text-align:center">▾</span>
        <div class="flow-context-item-info">
          <div class="flow-context-item-title">Explode list</div>
          <div class="flow-context-item-sub">Run this card first — the list is empty</div>
        </div>
      </button>
    `;
  }

  menuEl.innerHTML = `
    <div class="flow-context-header">
      <div style="font-size:12px;font-weight:700;color:var(--text);display:flex;align-items:center;gap:6px">
        <span class="blueprint-pin-dot" style="background:${pin.color || "#94a3b8"};color:${pin.color || "#94a3b8"}"></span>
        <span>${escapeHtml(pin.name)}</span>
      </div>
      <div style="font-size:10px;font-family:var(--font-mono);color:var(--text-muted)">${escapeHtml(pin.type)} · ${escapeHtml(node.title)}</div>
    </div>
    <div class="flow-context-body">
      ${actionHtml}
    </div>
  `;

  menuEl.querySelectorAll(".flow-context-item").forEach(item => {
    item.addEventListener("click", () => {
      const act = item.dataset.pinAction;
      if (act === "explode") app.splitListPin(nodeId, portId);
      else if (act === "collapse") app.collapseListPin(nodeId, portId);
      closeContextMenu();
    });
  });

  document.body.appendChild(menuEl);
  placeMenuEl(menuEl, screenX, screenY, 300);
}

/** Pure: wire right-click menu — identifies the wire, offers delete. */
export function renderWireMenuHtml(edge, fromTitle, toTitle) {
  const route = edge.label
    ? `${fromTitle} → ${toTitle} · ${edge.label}`
    : `${fromTitle} → ${toTitle}`;
  return `
    <div class="flow-context-group">
      <div class="flow-context-group-title"><span>✦ Wire</span></div>
      <div class="flow-wire-menu-route">${escapeHtml(route)}</div>
      <button class="flow-context-item" data-wire-action="delete">
        <span style="color:#ef4444;font-size:13px;width:16px;text-align:center">✕</span>
        <div class="flow-context-item-info">
          <div class="flow-context-item-title">Delete wire</div>
          <div class="flow-context-item-sub">Alt+click a wire deletes it directly</div>
        </div>
      </button>
    </div>`;
}

/** Right-click menu for a wire. Unknown ids no-op (never an empty menu). */
export function openWireMenu(app, screenX, screenY, edgeId) {
  const edge = (app.edges || []).find(e => e.id === edgeId);
  if (!edge) return;
  closeContextMenu();
  app.closePopover();
  const from = (app.nodes || []).find(n => n.id === edge.from);
  const to = (app.nodes || []).find(n => n.id === edge.to);
  const menuEl = document.createElement("div");
  menuEl.id = "active-flow-context-menu";
  menuEl.className = "flow-context-menu";
  menuEl.innerHTML = renderWireMenuHtml(edge, from?.title || edge.from, to?.title || edge.to);
  menuEl.querySelector("[data-wire-action=delete]").addEventListener("click", () => {
    deleteEdge(app, edgeId);
    closeContextMenu();
  });
  document.body.appendChild(menuEl);
  placeMenuEl(menuEl, screenX, screenY, 300);
}

export function spawnModelAt(app, modelKey, posX, posY) {
  const template = MODEL_CATALOG[modelKey];
  if (!template) return;

  const newId = uid(template.id);

  const newNode = {
    ...template,
    id: newId,
    templateKey: modelKey,
    x: Math.max(40, posX),
    y: Math.max(40, posY),
    status: template.nodeType === "action" ? "Ready to Run" : template.status
  };

  app.nodes.push(newNode);
  app.selectedNodeId = newId;
  app.renderCanvas();
}

export function duplicateNode(app, nodeId, posX, posY) {
  const node = app.nodes.find(n => n.id === nodeId);
  if (!node) return;

  const baseKey = node.templateKey || node.id.split('-').slice(0, 2).join('-');
  const newId = uid(baseKey);

  // Deep-clone: a spread would share the inputs/outputs arrays (and their pin
  // objects) with the original, so splits and badge updates would leak across.
  const cloned = {
    ...JSON.parse(JSON.stringify(node)),
    id: newId,
    createdAt: Date.now(),
    x: posX !== undefined ? posX : node.x + 40,
    y: posY !== undefined ? posY : node.y + 40,
  };
  if (cloned.nodeType === "action") {
    // Fresh card, no stale results: badges, splits, traces and logs describe
    // the original's run, not this one's.
    cloned.status = "Ready to Run";
    cloned.runError = null;
    cloned.runNote = null;
    cloned.apiCall = null;
    cloned.resultData = null;
    cloned.masks = null;
    cloned.resultImage = null;
    cloned.resultImageFallback = null;
    cloned.resultImageBadge = null;
    cloned.resultText = null;
    cloned.meshObjUrl = null;
    cloned.glbUrl = null;
    cloned.fitCamera = null;
    cloned.progress = null;
    cloned._progressLine = 0;
    cloned._progressCid = null;
    cloned.runLog = [];
    delete cloned.splitPins;
    const template = cloned.templateKey ? MODEL_CATALOG[cloned.templateKey] : null;
    if (template) {
      cloned.outputs = JSON.parse(JSON.stringify(template.outputs || []));
    }
  }

  app.nodes.push(cloned);
  app.selectedNodeId = newId;
  app.renderCanvas();
}
