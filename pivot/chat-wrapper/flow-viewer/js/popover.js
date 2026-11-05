/**
 * Downstream Model Picker Popover
 * Opens interactive popovers when releasing dragged wires on the canvas or clicking "+" buttons.
 */

import { MODEL_CATALOG } from "./catalog.js";
import { uid, canConnect, canWirePorts, escapeHtml, resolveAssetImageUrl, syncViewerStatus } from "./util.js";

/**
 * Pure: catalog cards that can take a wire of this pin type. Union of the
 * registry check over acceptsInput and the pin-aware check over each card's
 * real input pins — so pin-level accepts (mask into Sapiens, box into
 * viewers) are offered exactly when a drop would connect.
 */
export function compatibleModels(portType) {
  return Object.values(MODEL_CATALOG).filter(m =>
    (m.acceptsInput || []).some(t => canConnect(portType, t)) ||
    (m.inputs || []).some(p => canWirePorts(portType, p))
  );
}

export function openAddStepPopoverForPort(app, fromNodeId, fromPort, canvasX, canvasY) {
  closePopover(app);
  app.openPopoverSourceId = fromNodeId;

  const sourceNode = app.nodes.find(n => n.id === fromNodeId);
  if (!sourceNode) return;

  const popover = document.createElement("div");
  popover.className = "flow-add-popover";
  popover.id = "active-add-popover";

  // Position directly at cursor in canvas plane coordinates
  popover.style.left = `${Math.max(40, canvasX - 165)}px`;
  popover.style.top = `${Math.max(20, canvasY - 15)}px`;

  // Strict type filter: a model is offered only when it can take this
  // pin's type — same rule a drop would apply.
  const portType = fromPort?.type || sourceNode.assetKind || "image";
  const offered = compatibleModels(portType);

  const optionsHtml = offered.length === 0
    ? `<div class="flow-add-popover-empty">No card accepts “${escapeHtml(portType)}” yet.</div>`
    : offered.map(m => `
    <button class="flow-add-option-btn" data-model-key="${m.id}">
      <div class="reui-icon-tile" data-variant="${m.iconVariant}" data-size="xs">
        ${m.iconSvg}
      </div>
      <div class="flow-add-option-info">
        <div class="flow-add-option-title">${m.title.replace('Action: ', '').replace('Asset: ', '')}</div>
        <div class="flow-add-option-sub">✦ ${m.familyLabel.split('·')[0]} · Pin: ${fromPort.name}</div>
      </div>
    </button>
  `).join("");

  popover.innerHTML = `
    <div class="flow-add-popover-header">
      <span class="flow-add-popover-title">
        <span class="blueprint-pin-dot" style="background:${fromPort.color || '#10b981'};width:8px;height:8px"></span>
        Wire Pin: ${fromPort.name}
      </span>
      <button class="reui-btn sm" id="btn-close-popover" style="padding:2px 6px">✕</button>
    </div>
    <div class="flow-add-popover-list">
      ${optionsHtml}
    </div>
  `;

  popover.querySelector("#btn-close-popover").addEventListener("click", () => closePopover(app));

  popover.querySelectorAll(".flow-add-option-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const modelKey = btn.dataset.modelKey;
      addModelFromPort(app, fromNodeId, fromPort, modelKey, canvasX, canvasY);
      closePopover(app);
    });
  });

  app.plane.appendChild(popover);
}

export function addModelFromPort(app, fromNodeId, fromPort, modelKey, posX, posY) {
  const template = MODEL_CATALOG[modelKey];
  if (!template) return;

  const sourceNode = app.nodes.find(n => n.id === fromNodeId);
  if (!sourceNode) return;

  const newId = uid(template.id);

  const newNode = {
    ...template,
    id: newId,
    templateKey: modelKey,
    x: Math.max(40, posX),
    y: Math.max(40, posY - 60),
    status: template.nodeType === "action" ? "Ready to Run" : template.status
  };

  app.nodes.push(newNode);

  let targetPort = null;
  if (newNode.inputs && newNode.inputs.length > 0) {
    targetPort = newNode.inputs.find(p => canWirePorts(fromPort.type, p)) || null;
  }

  // The picker only offers compatible models, but if none of the new card's
  // actual pins take this type, place it unwired rather than forcing a wire.
  if (targetPort) {
    app.edges.push({
      id: `e-${fromNodeId}-${fromPort.id}-${newId}-${targetPort.id}`,
      from: fromNodeId,
      to: newId,
      fromPort: fromPort.id,
      toPort: targetPort.id,
      label: `${fromPort.name} → ${targetPort.name}`
    });
    syncViewerStatus(app, newNode);
  }

  app.selectedNodeId = newId;
  app.renderCanvas();
  if (targetPort && newNode.nodeType === "asset") {
    app.resolveAssetDims(newId, resolveAssetImageUrl(app, newNode), true);
  }
}

export function closePopover(app) {
  const p = document.getElementById("active-add-popover");
  if (p) p.remove();
  if (app) app.openPopoverSourceId = null;
}
