/**
 * Blueprint Wire Routing & SVG Connection Engine
 * Handles interactive wire dragging, socket hover snapping, horizontal S-curve bezier routing,
 * pin-to-pin connections, and live edge updates during node motion.
 */

import { canWirePorts, appendRunLog, isListType, resolveAssetImageUrl, syncViewerStatus } from "./util.js";

/**
 * Pure: a wire's status class from its source card. Only the currently
 * executing card's wires animate (`active`); completed cards — live or
 * demo — read solid green (`success`); everything else stays idle. Assets
 * count as completed unless they are still waiting for input.
 */
export function edgeStatusClass(app, edge) {
  // Green means verified success only. Anything holding data that never ran
  // (demo fills, cached/viewed artifacts) is solid orange; running wires keep
  // the animated stripe; untouched cards stay idle gray.
  const src = (app.nodes || []).find(n => n.id === edge.from);
  if (!src) return "";
  const st = String(src.status || "");
  if (st === "Running...") return "active";
  if (st === "Failed") return "error";
  if (st === "Passed" || st === "Done") return "success";
  if (st.includes("Demo") || st === "Cached in RAM" || st === "Artifact"
      || st === "Produced" || st.startsWith("Viewing")) return "pending";
  return "";
}

/** Resolve an input circle element to its pin def (null when unknown). */
function inputPort(app, circleEl) {
  if (!circleEl) return null;
  const node = app.nodes.find(n => n.id === circleEl.dataset.nodeId);
  return (node && (node.inputs || []).find(p => p.id === circleEl.dataset.portId)) || null;
}

export function attachBlueprintPinListeners(app, node, el) {
  if (!node.outputs || node.outputs.length === 0) return;

  const bindPort = (portId, anchorEl, circleEl) => {
    const outputDef = (node.outputs || []).find(o => o.id === portId) || { id: portId, name: "Output", color: "#10b981" };
    const handleStart = (e) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();
      // Second press of a double-click: skip the wire drag so dblclick owns it.
      const now = Date.now();
      if (anchorEl._lastDownAt && now - anchorEl._lastDownAt < 350) {
        anchorEl._lastDownAt = 0;
        return;
      }
      anchorEl._lastDownAt = now;
      startWireDrag(app, node, outputDef, e, circleEl);
    };
    anchorEl.addEventListener("mousedown", handleStart);
    if (circleEl !== anchorEl) {
      circleEl.addEventListener("mousedown", handleStart);
    }
  };

  // Double-click a filled list pin auto-splits it (split only — collapse
  // deletes wires, so it stays behind the deliberate right-click menu).
  const bindDoubleSplit = (pinEl) => {
    pinEl.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      const pin = (node.outputs || []).find(o => o.id === pinEl.dataset.portId);
      if (!pin || !isListType(pin.type)) return;
      if (node.splitPins && node.splitPins[pin.id]) return;
      const filled = node.resultData && Array.isArray(node.resultData[pin.type]) &&
        node.resultData[pin.type].length > 0;
      if (!filled) {
        appendRunLog(node, "info", `Cannot split ${pin.name} yet — run the card first.`);
        app.renderCanvas();
        return;
      }
      app.closePopover();
      app.splitListPin(node.id, pin.id);
    });
  };

  el.querySelectorAll(".blueprint-pin.output-pin").forEach(pinEl => {
    const circle = pinEl.querySelector(".blueprint-port-circle.port-output") || pinEl;
    bindPort(pinEl.dataset.portId, pinEl, circle);
    bindDoubleSplit(pinEl);
  });

  // Bare edge ports on minimized single-output asset cards.
  el.querySelectorAll(".blueprint-port-circle.port-output.port-minimal").forEach(circle => {
    bindPort(circle.dataset.portId, circle, circle);
    bindDoubleSplit(circle);
  });

  // Port dots on collapsed pin-column strips.
  el.querySelectorAll(".blueprint-pins-strip .blueprint-port-circle.port-output").forEach(circle => {
    bindPort(circle.dataset.portId, circle, circle);
    bindDoubleSplit(circle);
  });
}

export function startWireDrag(app, fromNode, fromPort, startEvent, sourceCircleEl) {
  app.closePopover();
  app.closeContextMenu();
  app.viewport.classList.add("is-wire-dragging");

  const planeRect = app.plane.getBoundingClientRect();
  let targetCircle = sourceCircleEl;
  if (!targetCircle.classList.contains("blueprint-port-circle")) {
    targetCircle = sourceCircleEl.querySelector(".blueprint-port-circle.port-output") || sourceCircleEl;
  }
  const circleRect = targetCircle.getBoundingClientRect();
  const x1 = Math.round((circleRect.left + circleRect.width / 2 - planeRect.left) / app.scale);
  const y1 = Math.round((circleRect.top + circleRect.height / 2 - planeRect.top) / app.scale);

  const startClientX = startEvent.clientX;
  const startClientY = startEvent.clientY;

  // Create temporary active wire in SVG layer
  const tempWireId = "temp-blueprint-wire";
  let tempWire = document.getElementById(tempWireId);
  if (!tempWire) {
    tempWire = document.createElementNS("http://www.w3.org/2000/svg", "path");
    tempWire.id = tempWireId;
    tempWire.setAttribute("class", "flow-edge-path blueprint-wire-active");
    app.svgLayer.appendChild(tempWire);
  }
  tempWire.setAttribute("d", `M ${x1} ${y1} C ${x1 + 30} ${y1}, ${x1 + 30} ${y1}, ${x1} ${y1}`);

  let hoveredNodeId = null;
  let hoveredInputPortEl = null;

  // Mousemove can fire far faster than frames: coalesce to one paint per
  // frame (SVG path write + hit-test + highlights), like node dragging.
  let moveRaf = null;
  let lastMove = { x: startEvent.clientX, y: startEvent.clientY };
  const onPointerMove = (moveEvt) => {
    lastMove = { x: moveEvt.clientX, y: moveEvt.clientY };
    if (moveRaf) return;
    moveRaf = requestAnimationFrame(() => {
      moveRaf = null;
      paintWireDrag(lastMove.x, lastMove.y);
    });
  };

  const paintWireDrag = (clientX, clientY) => {
    const curX = Math.round((clientX - planeRect.left) / app.scale);
    const curY = Math.round((clientY - planeRect.top) / app.scale);

    // Horizontal cubic bezier curve
    const dx = Math.max(50, Math.abs(curX - x1) * 0.5);
    tempWire.setAttribute("d", `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${curX - dx} ${curY}, ${curX} ${curY}`);

    // Check element under point
    const elUnderPoint = document.elementFromPoint(clientX, clientY);
    const targetInputCircle = elUnderPoint?.closest(".blueprint-port-circle.port-input");
    const targetCard = elUnderPoint?.closest(".flow-node");

    // Clear previous input circle highlight
    if (hoveredInputPortEl && hoveredInputPortEl !== targetInputCircle) {
      hoveredInputPortEl.classList.remove("socket-drag-target", "drop-invalid");
      hoveredInputPortEl = null;
    }

    if (targetInputCircle && targetInputCircle.dataset.nodeId !== fromNode.id) {
      hoveredInputPortEl = targetInputCircle;
      const ok = canWirePorts(fromPort.type, inputPort(app, targetInputCircle));
      targetInputCircle.classList.add(ok ? "socket-drag-target" : "drop-invalid");
    }

    if (targetCard && targetCard.id !== fromNode.id) {
      if (hoveredNodeId !== targetCard.id) {
        if (hoveredNodeId) {
          document.getElementById(hoveredNodeId)?.classList.remove("blueprint-target-hover");
        }
        hoveredNodeId = targetCard.id;
        targetCard.classList.add("blueprint-target-hover");
      }
    } else {
      if (hoveredNodeId) {
        document.getElementById(hoveredNodeId)?.classList.remove("blueprint-target-hover");
        hoveredNodeId = null;
      }
    }
  };

  const onPointerUp = (upEvt) => {
    window.removeEventListener("mousemove", onPointerMove);
    window.removeEventListener("mouseup", onPointerUp);
    if (moveRaf) {
      cancelAnimationFrame(moveRaf);
      moveRaf = null;
    }
    app.viewport.classList.remove("is-wire-dragging");

    // Suppress subsequent click event from immediately dismissing the spawned popover
    app.wireDragFinishedAt = Date.now();

    if (tempWire) {
      tempWire.remove();
    }

    if (hoveredInputPortEl) {
      hoveredInputPortEl.classList.remove("socket-drag-target", "drop-invalid");
    }

    if (hoveredNodeId) {
      document.getElementById(hoveredNodeId)?.classList.remove("blueprint-target-hover");
    }

    const dist = Math.hypot(upEvt.clientX - startClientX, upEvt.clientY - startClientY);
    const elUnderPoint = document.elementFromPoint(upEvt.clientX, upEvt.clientY);
    const targetInputCircle = elUnderPoint?.closest(".blueprint-port-circle.port-input");
    const targetCard = elUnderPoint?.closest(".flow-node");

    if (targetInputCircle && targetInputCircle.dataset.nodeId !== fromNode.id) {
      const toNodeId = targetInputCircle.dataset.nodeId;
      const toPortId = targetInputCircle.dataset.portId;
      connectNodesWithPort(app, fromNode.id, toNodeId, fromPort, toPortId);
    } else if (targetCard && targetCard.id !== fromNode.id) {
      // Dropped onto card body: auto-connect to compatible input
      connectNodesWithPort(app, fromNode.id, targetCard.id, fromPort, null);
    } else {
      if (dist < 6) {
        // Clicked output pin directly: open downstream model picker neatly beside this card
        const sourceEl = document.getElementById(fromNode.id);
        const popX = fromNode.x + (sourceEl ? sourceEl.offsetWidth + 24 : 360);
        const popY = fromNode.y + 40;
        app.openAddStepPopoverForPort(fromNode.id, fromPort, popX, popY);
      } else {
        // Dragged into open space: open downstream model picker right at release point
        const dropX = Math.round((upEvt.clientX - planeRect.left) / app.scale);
        const dropY = Math.round((upEvt.clientY - planeRect.top) / app.scale);
        app.openAddStepPopoverForPort(fromNode.id, fromPort, dropX, dropY);
      }
    }
  };

  window.addEventListener("mousemove", onPointerMove);
  window.addEventListener("mouseup", onPointerUp);
}

export function connectNodesWithPort(app, fromNodeId, toNodeId, fromPort, toPortId = null) {
  const fromNode = app.nodes.find(n => n.id === fromNodeId);
  const toNode = app.nodes.find(n => n.id === toNodeId);
  if (!toNode) return;
  const reject = (msg) => {
    if (fromNode) appendRunLog(fromNode, "error", `Rejected wire: ${msg}`);
  };

  let targetPort = null;
  if (toPortId && toNode.inputs) {
    targetPort = toNode.inputs.find(p => p.id === toPortId);
    // Explicit pin requested: validate strictly, never silently re-route.
    if (targetPort && !canWirePorts(fromPort.type, targetPort)) {
      reject(`${fromPort.type} → ${targetPort.type} (type mismatch)`);
      return;
    }
  }
  if (!targetPort && toNode.inputs && toNode.inputs.length > 0) {
    targetPort = toNode.inputs.find(p => canWirePorts(fromPort.type, p)) || null;
  }
  if (!targetPort) {
    const toTitle = toNode.title.replace('Action: ', '').replace('Asset: ', '');
    reject(`no input on ${toTitle} accepts ${fromPort.type}`);
    return;
  }

  const edgeId = `e-${fromNodeId}-${fromPort.id}-${toNodeId}-${targetPort.id}`;

  if (app.edges.some(e => e.id === edgeId)) return;

  app.edges.push({
    id: edgeId,
    from: fromNodeId,
    to: toNodeId,
    fromPort: fromPort.id,
    toPort: targetPort.id,
    label: `${fromPort.name} → ${targetPort.name}`
  });

  // A wired asset input changes what the card shows — full refresh so the
  // viewer image, badge and aspect snap to the upstream pixels.
  if (toNode.nodeType === "asset") {
    syncViewerStatus(app, toNode);
    app.renderCanvas();
    app.resolveAssetDims(toNode.id, resolveAssetImageUrl(app, toNode), true);
  }

  renderEdges(app);
}

export function updateConnectedEdges(app, nodeId) {
  const connectedEdges = app.edges.filter(e => e.from === nodeId || e.to === nodeId);
  // One shared plane measurement per pass — never one per edge.
  const planeRect = app.plane.getBoundingClientRect();
  connectedEdges.forEach(edge => {
    updateSingleEdge(app, edge, planeRect);
  });
}

export function updateSingleEdge(app, edge, planeRect = null) {
  const fromNode = app.nodes.find(n => n.id === edge.from);
  const toNode = app.nodes.find(n => n.id === edge.to);
  if (!fromNode || !toNode) return;

  const nodeWidth = 320;
  const fromEl = document.getElementById(fromNode.id);
  const toEl = document.getElementById(toNode.id);

  // Default right edge of fromNode, left edge of toNode
  let x1 = fromNode.x + nodeWidth;
  let y1 = fromNode.y + (fromEl ? fromEl.offsetHeight / 2 : 160);

  let x2 = toNode.x;
  let y2 = toNode.y + (toEl ? toEl.offsetHeight / 2 : 160);

  const pr = planeRect || app.plane.getBoundingClientRect();

  // Read phase: resolve both sockets and measure before any DOM writes, so
  // one edge never forces a reflow between its own reads.
  let fromCircle = null;
  if (fromEl) {
    if (edge.fromPort) {
      fromCircle = fromEl.querySelector(`.blueprint-port-circle.port-output[data-port-id="${edge.fromPort}"]`);
    }
    if (!fromCircle) {
      fromCircle = fromEl.querySelector(`.blueprint-port-circle.port-output`);
    }
    if (fromCircle) {
      const cRect = fromCircle.getBoundingClientRect();
      x1 = Math.round((cRect.left + cRect.width / 2 - pr.left) / app.scale);
      y1 = Math.round((cRect.top + cRect.height / 2 - pr.top) / app.scale);
    }
  }

  let toCircle = null;
  if (toEl) {
    if (edge.toPort) {
      toCircle = toEl.querySelector(`.blueprint-port-circle.port-input[data-port-id="${edge.toPort}"]`);
    }
    if (!toCircle) {
      toCircle = toEl.querySelector(`.blueprint-port-circle.port-input`);
    }
    if (toCircle) {
      const cRect = toCircle.getBoundingClientRect();
      x2 = Math.round((cRect.left + cRect.width / 2 - pr.left) / app.scale);
      y2 = Math.round((cRect.top + cRect.height / 2 - pr.top) / app.scale);
    }
  }

  // Write phase.
  if (fromCircle) fromCircle.classList.add("connected");
  if (toCircle) toCircle.classList.add("connected");

  // Pure horizontal S-curve bezier routing from right to left
  const dx = Math.max(60, Math.abs(x2 - x1) * 0.5);

  // Never poison the path with NaN (e.g. mid-layout measurements): skip the
  // update and keep the last valid curve instead of hiding the wire.
  if (!Number.isFinite(x1 + y1 + x2 + y2 + dx)) return;

  const pathData = `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;

  const path = document.getElementById(`edge-path-${edge.id}`);
  if (path) {
    path.setAttribute("d", pathData);
    const cls = edgeStatusClass(app, edge);
    path.setAttribute("class", cls ? `flow-edge-path ${cls}` : "flow-edge-path");
  }

  if (edge.label) {
    const midX = (x1 + x2) / 2;
    const midY = (y1 + y2) / 2;
    const lbl = document.getElementById(`edge-label-${edge.id}`);
    if (lbl) {
      lbl.style.left = `${midX}px`;
      lbl.style.top = `${midY - 8}px`;
    }
  }
}

export function renderEdges(app) {
  const edgeIds = new Set(app.edges.map(e => e.id));

  // Remove obsolete SVG paths
  Array.from(app.svgLayer.querySelectorAll(".flow-edge-path")).forEach(path => {
    const id = path.id.replace("edge-path-", "");
    if (!edgeIds.has(id)) path.remove();
  });

  // Remove obsolete labels
  Array.from(app.plane.querySelectorAll(".flow-edge-label")).forEach(lbl => {
    const id = lbl.id.replace("edge-label-", "");
    if (!edgeIds.has(id)) lbl.remove();
  });

  // One shared plane measurement for the whole pass.
  const planeRect = app.plane.getBoundingClientRect();

  // Create missing SVG paths and labels, then update
  app.edges.forEach(edge => {
    let path = document.getElementById(`edge-path-${edge.id}`);
    if (!path) {
      path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.id = `edge-path-${edge.id}`;
      const cls = edgeStatusClass(app, edge);
      path.setAttribute("class", cls ? `flow-edge-path ${cls}` : "flow-edge-path");
      path.setAttribute("data-edge-id", edge.id);
      const tip = document.createElementNS("http://www.w3.org/2000/svg", "title");
      tip.textContent = "Alt+click to delete · right-click for options";
      path.appendChild(tip);
      path.addEventListener("click", (e) => {
        if (!e.altKey) return;
        e.stopPropagation();
        e.preventDefault();
        deleteEdge(app, edge.id);
      });
      app.svgLayer.appendChild(path);
    }

    if (edge.label) {
      let lbl = document.getElementById(`edge-label-${edge.id}`);
      if (!lbl) {
        lbl = document.createElement("div");
        lbl.id = `edge-label-${edge.id}`;
        lbl.className = "flow-edge-label";
        lbl.textContent = edge.label;
        lbl.setAttribute("data-edge-id", edge.id);
        lbl.title = "Alt+click to delete · right-click for options";
        lbl.addEventListener("click", (e) => {
          if (!e.altKey) return;
          e.stopPropagation();
          e.preventDefault();
          deleteEdge(app, edge.id);
        });
        app.plane.appendChild(lbl);
      }
    }

    updateSingleEdge(app, edge, planeRect);
  });
}

/**
 * Delete one wire by id (Alt+click or the wire menu). Viewers left unwired
 * fall back to their own image, mirroring node deletion. Returns false
 * when the id is unknown (nothing happens).
 */
export function deleteEdge(app, edgeId) {
  const edge = (app.edges || []).find(e => e.id === edgeId);
  if (!edge) return false;
  app.edges = app.edges.filter(e => e.id !== edgeId);
  const target = (app.nodes || []).find(n => n.id === edge.to);
  if (target && target.nodeType === "asset") syncViewerStatus(app, target);
  app.closePopover();
  app.renderCanvas();
  if (target && target.nodeType === "asset") {
    app.resolveAssetDims(target.id, resolveAssetImageUrl(app, target), false);
  }
  return true;
}
