/**
 * Pipeline Execution Engine & SAM 3.1 Detector
 * Executes model passes, invokes live Meta SAM 3.1 (/api/debug/detect) endpoint,
 * generates downstream asset nodes (crops, normal maps, depth maps, meshes), and handles deletion cascades.
 */

import { uid, svgPlaceholder, fetchJson, formatLatency, isListType, itemTypeForList, listCountLabel, redactBlobs, appendRunLog, resolveAssetImageUrl, syncViewerStatus } from "./util.js";
import { MODEL_CATALOG } from "./catalog.js";
import { disposeMeshViewer, MULTIVIEW_COUNTS, MULTIVIEW_CELL } from "./mesh-view.js";
import { loadSegLabels, segSpawnRows } from "./overlays.js";

const DETECT_TIMEOUT_MS = 200000;

export async function runCard(app, nodeId) {
  const node = app.nodes.find(n => n.id === nodeId);
  if (!node || node.nodeType !== "action") return;

  const cardEl = document.getElementById(nodeId);
  const btnRun = cardEl?.querySelector(".btn-run");
  node._runStartedAt = Date.now();
  if (btnRun) {
    btnRun.classList.add("is-running");
    btnRun.textContent = "⏳";
  }

  node.status = "Running...";
  node.runError = null;
  node.runNote = null;
  appendRunLog(node, "info", "Run started");
  app.renderCanvas();

  // Clear existing downstream output cards if re-running
  clearDownstreamOutputs(app, nodeId);

  const modelType = node.templateKey || node.id;

  if (modelType.startsWith("action-describe")) {
    await executeDescribe(app, node, btnRun);
  } else if (modelType.startsWith("action-apply-mask")) {
    await executeApplyMask(app, node, btnRun);
  } else if (modelType.startsWith("action-image-diff")) {
    await executeImageDiff(app, node, btnRun);
  } else if (modelType.startsWith("action-meta-sam")) {
    await executeRealSamDetection(app, node, btnRun);
  } else if (modelType.startsWith("action-spark")) {
    await executeSpark(app, node, btnRun);
  } else if (modelType.startsWith("action-meta-muse")) {
    await executeMuse(app, node, btnRun);
  } else if (modelType.startsWith("action-sapiens-pose")) {
    await executeSapiens(app, node, btnRun, "pose");
  } else if (modelType.startsWith("action-sapiens-seg")) {
    await executeSapiens(app, node, btnRun, "seg");
  } else if (modelType.startsWith("action-sapiens-normal")) {
    await executeSapiens(app, node, btnRun, "normal");
  } else if (modelType.startsWith("action-sapiens-depth")) {
    await executeSapiens(app, node, btnRun, "depth");
  } else if (modelType.startsWith("action-body-fit")) {
    await executeBodyFit(app, node, btnRun);
  } else if (modelType.startsWith("action-vggt-check")) {
    await executeVGGTCheck(app, node, btnRun);
  } else if (modelType.startsWith("action-mhr-match")) {
    await executeMHRMatch(app, node, btnRun);
  } else if (modelType.startsWith("action-apose")) {
    await executeAPose(app, node, btnRun);
  } else if (modelType.startsWith("action-texture")) {
    await executeTexture(app, node, btnRun);
  } else if (modelType.startsWith("action-export")) {
    await executeExport(app, node, btnRun);
  } else if (modelType.startsWith("action-mesh-render")) {
    await executeMeshRender(app, node, btnRun, "shaded");
  } else if (modelType.startsWith("action-mesh-depth")) {
    await executeMeshRender(app, node, btnRun, "depth");
  } else if (modelType.startsWith("action-mesh-normal")) {
    await executeMeshRender(app, node, btnRun, "normal");
  } else if (modelType.startsWith("action-mesh-multiview")) {
    await executeMeshMultiview(app, node, btnRun);
  } else if (modelType.startsWith("action-fit-overlay")) {
    await executeFitOverlay(app, node, btnRun);
  } else {
    // Unknown template: clearly-labelled mock fallback. Returns a completion
    // promise like every other path, so traversal can await it.
    await new Promise(resolve => setTimeout(() => {
      node.status = "Passed";
      if (btnRun) {
        btnRun.classList.remove("is-running");
        btnRun.textContent = "✓";
      }
      appendRunLog(node, "info", "Passed in 0.7s (mock outputs)");
      produceOutputsForAction(app, node);
      app.renderCanvas();
      resolve();
    }, 700));
  }
}

export function clearDownstreamOutputs(app, actionNodeId) {
  // Invalidate everything downstream of a re-run: generated assets are deleted,
  // while user-placed actions are reset to Ready (outputs restored from their
  // template) so wiring survives. Traversal passes THROUGH reset actions.
  const seen = new Set([actionNodeId]);
  const queue = [actionNodeId];
  const doomedAssets = [];
  const resetActions = [];
  while (queue.length > 0) {
    const curr = queue.shift();
    app.edges.filter(e => e.from === curr).forEach(e => {
      if (seen.has(e.to)) return;
      seen.add(e.to);
      queue.push(e.to);
      const target = app.nodes.find(n => n.id === e.to);
      if (!target) return;
      if (target.nodeType === "action") resetActions.push(target);
      // User-placed assets (viewer cards, the source photo) survive re-runs —
      // only generated artifacts are deleted. Traversal still passes through.
      else if (target.templateKey || target.id === "asset-photo") { /* keep */ }
      else doomedAssets.push(e.to);
    });
  }
  resetActions.forEach(t => resetActionNode(app, t));
  // deleteCard mutates app.edges, so collect first, delete leaves-first.
  doomedAssets.reverse().forEach(childId => deleteCard(app, childId));
}

function resetActionNode(app, node) {
  collapseAllSplits(app, node, false);
  stopProgressPoll(node);
  node.progress = null;
  node._progressLine = 0;
  node._progressCid = null;
  node.status = "Ready to Run";
  node.runError = null;
  node.runNote = null;
  node.apiCall = null;
  node.resultData = null;
  node.masks = null;
  node.resultImage = null;
  node.resultImageFallback = null;
  node.resultImageBadge = null;
  node.resultText = null;
  node.resultKeypoints = null;
  node.resultSegLegend = null;
  node.resultSegLabelsUrl = null;
  node._segIsolated = null;
  node.meshObjUrl = null;
  node.glbUrl = null;
  node.fitCamera = null;
  const template = node.templateKey ? MODEL_CATALOG[node.templateKey] : null;
  if (template) {
    node.outputs = JSON.parse(JSON.stringify(template.outputs || []));
  }
}

export async function executeRealSamDetection(app, actionNode, btnRun) {
  const incomingEdge = app.edges.find(e => e.to === actionNode.id);
  const inputNode = incomingEdge 
    ? app.nodes.find(n => n.id === incomingEdge.from)
    : app.nodes.find(n => n.id === "asset-photo") || app.nodes[0];

  let imageB64 = null;
  let imageUrl = null;

  // Resolve through viewer cards, so a viewer between photo and SAM is
  // transparent: SAM segments what the viewer shows, not its placeholder.
  const inputSrc = inputNode ? resolveAssetImageUrl(app, inputNode) : null;
  if (inputSrc) {
    if (inputSrc.startsWith("data:image")) {
      imageB64 = inputSrc;
    } else if (inputSrc.startsWith("blob:")) {
      try {
        imageB64 = await readUrlAsDataUrl(inputSrc);
      } catch (err) {
        console.warn("Could not read blob, using fallback", err);
      }
    } else {
      imageUrl = inputSrc;
    }
  }

  if (!inputSrc || inputSrc.startsWith("data:image/svg")) {
    // Nothing real to segment (e.g. SAM wired from an unwired viewer card):
    // fail fast without a network call.
    if (btnRun) {
      btnRun.classList.remove("is-running");
      btnRun.textContent = "✓";
    }
    resetActionNode(app, actionNode);
    actionNode.status = "Failed";
    actionNode.runError = "No input image — wire the source photo (or a viewer showing one) into Image Input.";
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const concept = (actionNode.param || "people").trim() || "people";

  const requestBody = {
    chat_id: "flow-session",
    concept: concept,
    image_b64: imageB64 || undefined,
    image_url: imageUrl || undefined,
    filename: inputNode?.subtitle || "photo.jpg"
  };
  markRequestSent(app, actionNode, "/api/debug/detect", requestBody);
  const res = await fetchJson("/api/debug/detect", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
    timeoutMs: DETECT_TIMEOUT_MS
  });

  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }

  // Capture hop 1 (viewer → backend) for the transparency views; hop 2
  // (backend → Meta) rides along when the backend returns its api_trace.
  const captureTrace = () => {
    actionNode.apiCall = {
      at: Date.now(),
      hop1: {
        method: "POST",
        url: "/api/debug/detect",
        status: res.status,
        request: redactBlobs(requestBody),
        response: res.data ? redactBlobs(res.data) : null,
        error: res.ok ? null : (res.error || "request failed"),
      },
      hop2: res.ok && res.data && res.data.api_trace ? res.data.api_trace : null,
    };
  };

  if (!res.ok && res.status === 0) {
    // Backend unreachable: stay usable with clearly-labelled demo data.
    if (failUnreachable(app, actionNode, res, btnRun)) { app.renderCanvas(); return; }
    console.warn("Live SAM 3.1 request failed, showing demo outputs:", res.error);
    captureTrace();
    actionNode.status = "Demo (Offline)";
    actionNode.runError = res.error || "Detection request failed";
    actionNode.runNote = null;
    appendRunLog(actionNode, "error", `Backend unreachable (${actionNode.runError}) — demo data`);
    produceOutputsForAction(app, actionNode);
    app.renderCanvas();
    return;
  }

  if (!res.ok && /zero matches/i.test(res.error || "")) {
    // Live empty lane: a successful call that found nothing. Not an error,
    // and no mock data — the empty result IS the truth.
    captureTrace();
    collapseAllSplits(app, actionNode, false);
    const resultData = deriveResultData([], concept, 0, 0, null);
    fillSamOutputs(actionNode, resultData);
    actionNode.status = "Passed";
    actionNode.runError = null;
    actionNode.runNote = (res.data && res.data.remedy) || "sam-3.1 returned zero matches.";
    actionNode.latency = "0 items";
    actionNode.masks = [];
    actionNode.imageWidth = 0;
    actionNode.imageHeight = 0;
    actionNode.resultImage = svgPlaceholder("0 detections");
    actionNode.resultImageFallback = actionNode.resultImage;
    actionNode.resultImageBadge = "0 boxes";
    appendRunLog(actionNode, "info", `Passed — 0 matches (${actionNode.runNote})`);
    app.renderCanvas();
    return;
  }

  if (!res.ok) {
    // Backend answered with an error: fail loudly, no mock data.
    resetActionNode(app, actionNode);
    captureTrace();
    actionNode.status = "Failed";
    const remedy = res.data && res.data.remedy ? ` — ${res.data.remedy}` : "";
    actionNode.runError = `${res.error || `HTTP ${res.status}`}${remedy}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const data = res.data || {};
  captureTrace();
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.runNote = null;
  actionNode.latency = formatLatency(data.api_trace?.meta?.latency_s);
  appendRunLog(actionNode, "info", `Passed in ${actionNode.latency} — ${(data.masks || []).length} masks`);

  produceOutputsFromRealSam(app, actionNode, data);
  app.renderCanvas();
}

/**
 * Normalize backend SAM masks into the four typed lists the action produces.
 * Pure (no DOM): shared by the live path and the offline demo path.
 */
export function deriveResultData(masks, concept, imageWidth, imageHeight, debugId = null) {
  const list = Array.isArray(masks) ? masks : [];
  const c = (concept || "person").trim() || "person";
  return {
    concept: c,
    count: list.length,
    debugId: debugId || null,
    imageWidth: Number(imageWidth) || 0,
    imageHeight: Number(imageHeight) || 0,
    boxes: list.map(m => (m && Array.isArray(m.box) ? m.box : [0, 0, 0, 0])),
    masks: list,
    confidences: list.map(m => (m && typeof m.confidence === "number" ? m.confidence : 0.985)),
    labels: list.map((m, i) => `${c} ${i + 1}`),
  };
}

/** Attach list data to a SAM action node and refresh its four pin badges. */
export function fillSamOutputs(actionNode, resultData) {
  actionNode.resultData = resultData;
  const n = resultData.count;
  const setBadge = (id, kind) => {
    const p = (actionNode.outputs || []).find(o => o.id === id);
    if (p) p.badge = listCountLabel(kind, n);
  };
  setBadge("out-boxes", "boxes");
  setBadge("out-masks", "masks");
  setBadge("out-confs", "confidences");
  setBadge("out-labels", "labels");
}

/**
 * Fill the in-card result view: combined detection image, overlay masks and
 * toggle state. Overlay toggles persist across re-runs (view preference).
 */
export function fillSamResultView(actionNode, resultData, { imgUrl, demo }) {
  const count = resultData.count;
  const meanConf = count > 0
    ? resultData.confidences.reduce((acc, c) => acc + c, 0) / count
    : 0.985;
  const confPct = (meanConf * 100).toFixed(1) + "%";
  const countLabel = listCountLabel("boxes", count);

  actionNode.masks = resultData.masks;
  actionNode.imageWidth = resultData.imageWidth || 0;
  actionNode.imageHeight = resultData.imageHeight || 0;
  actionNode.displayOpts = actionNode.displayOpts || {
    showMasks: true,
    showBoxes: true,
    showLabels: true,
    showConfidence: true
  };
  actionNode.resultImage = imgUrl;
  actionNode.resultImageFallback = svgPlaceholder(demo ? `Demo · ${countLabel}` : countLabel);
  actionNode.resultImageBadge = `${countLabel} · Conf ${confPct}`;
}

export function produceOutputsFromRealSam(app, actionNode, data) {
  const runId = app.activeRunId;
  collapseAllSplits(app, actionNode, false);
  const resultData = deriveResultData(
    data.masks, data.concept || actionNode.param, data.image_width, data.image_height, data.debug_id
  );
  fillSamOutputs(actionNode, resultData);
  fillSamResultView(actionNode, resultData, {
    imgUrl: data.combined_url || `/api/runs/${runId}/artifact/boxes`,
    demo: false,
  });

  // Reframe canvas smoothly to reveal newly generated outputs
  setTimeout(() => {
    app.fitView();
  }, 60);
}

const SPARK_TIMEOUT_MS = 200000;
const MUSE_TIMEOUT_MS = 200000;
const DESCRIBE_TIMEOUT_MS = 200000;
const APPLY_MASK_TIMEOUT_MS = 60000;

/**
 * Pure: displayable image URL at the upstream end of an edge — split-item
 * thumbnails first, then through viewer chains, then action result images.
 */
export function upstreamImageUrl(app, edge) {
  const src = (app.nodes || []).find(n => n.id === edge.from);
  if (!src) return null;
  const pin = (src.outputs || []).find(p => p.id === edge.fromPort);
  if (pin && pin.thumb) return pin.thumb;
  if (src.nodeType === "asset") return resolveAssetImageUrl(app, src);
  return src.resultImage || src.imgUrl || null;
}

/**
 * Pure: every image wired into a multi-wire images pin, in edge order —
 * one input_image block per entry downstream.
 */
export function collectImageInputs(app, actionNode, portId) {
  return (app.edges || [])
    .filter(e => e.to === actionNode.id && e.toPort === portId)
    .map(e => ({ url: upstreamImageUrl(app, e), pinName: e.fromPort }))
    .filter(s => !!s.url);
}

/**
 * Pure: prompt text wired into a prompt pin — label values, then upstream
 * reply text — or null when unwired/unresolvable (caller falls back to the
 * typed prompt).
 */
export function resolvePromptText(app, actionNode, portId) {
  const edge = (app.edges || []).find(e => e.to === actionNode.id && e.toPort === portId);
  if (!edge) return null;
  const src = (app.nodes || []).find(n => n.id === edge.from);
  if (!src) return null;
  const pin = (src.outputs || []).find(p => p.id === edge.fromPort);
  if (pin && pin.thumbValue != null) return String(pin.thumbValue);
  if (src.resultText) return src.resultText;
  return null;
}

/**
 * Encode one viewer-side image URL for the backend: blob: URLs are fetched
 * and base64'd (the backend cannot reach them); data: URLs pass through as
 * image_b64; http(s) and relative artifact URLs pass through as image_url
 * for the backend to resolve.
 */
/** Read a blob: URL into a data-URL (the backend cannot reach blobs). Browser-only. */
export async function readUrlAsDataUrl(url) {
  const res = await fetch(url);
  const blob = await res.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export async function encodeImageInput(url) {
  if (url.startsWith("blob:")) return { image_b64: await readUrlAsDataUrl(url) };
  if (url.startsWith("data:")) return { image_b64: url };
  return { image_url: url };
}

/** Mesh-side twin of encodeImageInput: blob:/data: URLs base64, the rest pass as mesh_url. */
export async function encodeMeshInput(url) {
  if (url.startsWith("blob:")) return { mesh_b64: await readUrlAsDataUrl(url) };
  if (url.startsWith("data:")) return { mesh_b64: url };
  return { mesh_url: url };
}

/** Fill a Spark card's result view: reply text, first image, badges. */
export function fillSparkResultView(actionNode, { text, imgUrl, badge, demo }) {
  actionNode.resultText = text || "";
  actionNode.resultImage = imgUrl || null;
  actionNode.resultImageFallback = svgPlaceholder(demo ? "Demo · reply" : "reply");
  actionNode.resultImageBadge = badge || "";
  const p = (actionNode.outputs || []).find(o => o.id === "out-text");
  if (p) p.badge = `${(text || "").length} chars`;
}

/** Fill a Muse card's result view: generated image and badges. */
export function fillMuseResultView(actionNode, { imgUrl, badge, demo, note }) {
  actionNode.resultText = note || "";
  actionNode.resultImage = imgUrl || null;
  actionNode.resultImageFallback = svgPlaceholder(demo ? "Demo · generated" : "generated");
  actionNode.resultImageBadge = badge || "";
  const p = (actionNode.outputs || []).find(o => o.id === "out-image");
  if (p) p.badge = badge || "—";
}

/**
 * Publish the outgoing POST to the trace views the moment it is sent, so a
 * long call shows its hop 1 request while awaiting the response instead of
 * only after the return lands. captureTrace fills in the rest on arrival.
 */
function markRequestSent(app, actionNode, url, requestBody) {
  actionNode.apiCall = {
    at: Date.now(),
    hop1: {
      method: "POST",
      url,
      status: null,
      request: redactBlobs(requestBody),
      response: null,
      error: null,
    },
    hop2: null,
  };
  app.renderCanvas();
}

/** Shared fail-fast: reset the run button, fail the card with a remedy. */
function failRun(app, actionNode, btnRun, message) {
  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }
  resetActionNode(app, actionNode);
  actionNode.status = "Failed";
  actionNode.runError = message;
  appendRunLog(actionNode, "error", `Failed: ${message}`);
  app.renderCanvas();
}

/**
 * Unreachable-backend fork for status-0 failures. When the backend answered
 * health at boot, a dropped request means the server died mid-run — fail
 * loud (red wire + message) instead of swapping in quiet demo data. Returns
 * true when the caller must stop (loud path taken). Standalone or
 * never-reached backends return false and keep the demo path.
 */
export function failUnreachable(app, actionNode, res, btnRun) {
  if (!app.backendOnline) return false;
  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }
  const trace = actionNode.apiCall;
  resetActionNode(app, actionNode);
  actionNode.apiCall = trace;
  actionNode.status = "Failed";
  const cause = (res && res.error) || "Failed to fetch";
  actionNode.runError = `Backend unreachable (${cause}) — the server stopped responding. Check the server console, then run again.`;
  appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
  if (actionNode.apiCall && actionNode.apiCall.hop1) {
    actionNode.apiCall.hop1.status = 0;
    actionNode.apiCall.hop1.error = cause;
  }
  return true;
}

export async function executeSpark(app, actionNode, btnRun) {
  const sources = collectImageInputs(app, actionNode, "in-images");
  const prompt = resolvePromptText(app, actionNode, "in-prompt")?.trim()
    || (actionNode.param || "").trim()
    || (sources.length > 0 ? "Describe this image in detail." : null);
  if (!prompt) {
    failRun(app, actionNode, btnRun, "No prompt — wire text into Prompt, type one, or wire images to describe.");
    return;
  }

  let images = [];
  try {
    images = await Promise.all(sources.map(s => encodeImageInput(s.url)));
  } catch (err) {
    failRun(app, actionNode, btnRun, `Could not read an input image (${err?.message || err}).`);
    return;
  }

  const requestBody = { chat_id: "flow-session", prompt };
  if (actionNode.museSize) requestBody.size = actionNode.museSize;
  if (images.length > 0) requestBody.images = images;
  markRequestSent(app, actionNode, "/api/debug/spark", requestBody);
  const res = await fetchJson("/api/debug/spark", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
    timeoutMs: SPARK_TIMEOUT_MS,
  });

  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }

  // Capture hop 1 (viewer → backend) for the transparency views; hop 2
  // (backend → Meta) rides along when the backend returns its api_trace.
  const captureTrace = () => {
    actionNode.apiCall = {
      at: Date.now(),
      hop1: {
        method: "POST",
        url: "/api/debug/spark",
        status: res.status,
        request: redactBlobs(requestBody),
        response: res.data ? redactBlobs(res.data) : null,
        error: res.ok ? null : (res.error || "request failed"),
      },
      hop2: res.ok && res.data && res.data.api_trace ? res.data.api_trace : null,
    };
  };

  if (!res.ok && res.status === 0) {
    // Backend unreachable: stay usable with clearly-labelled demo data.
    if (failUnreachable(app, actionNode, res, btnRun)) { app.renderCanvas(); return; }
    console.warn("Live spark request failed, showing demo outputs:", res.error);
    captureTrace();
    actionNode.status = "Demo (Offline)";
    actionNode.runError = res.error || "Spark request failed";
    actionNode.runNote = null;
    appendRunLog(actionNode, "error", `Backend unreachable (${actionNode.runError}) — demo data`);
    fillSparkResultView(actionNode, {
      text: `Demo reply (offline) for "${prompt.slice(0, 80)}": the live muse-spark-1.3-contributor reply appears here once the backend is reachable.`,
      imgUrl: sources[0]?.url || null,
      badge: sources.length > 0 ? `${sources.length} img · demo` : "demo",
      demo: true,
    });
    app.renderCanvas();
    return;
  }

  if (!res.ok) {
    // Backend answered with an error: fail loudly, no mock data.
    resetActionNode(app, actionNode);
    captureTrace();
    actionNode.status = "Failed";
    const remedy = res.data && res.data.remedy ? ` — ${res.data.remedy}` : "";
    actionNode.runError = `${res.error || `HTTP ${res.status}`}${remedy}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const data = res.data || {};
  captureTrace();
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.runNote = null;
  actionNode.latency = formatLatency(data.api_trace?.meta?.latency_s);
  appendRunLog(actionNode, "info", `Passed in ${actionNode.latency} — ${(data.assistant_text || "").length} chars`);
  fillSparkResultView(actionNode, {
    text: data.assistant_text || "(empty reply)",
    imgUrl: sources[0]?.url || null,
    badge: sources.length > 0 ? `${sources.length} img · muse-spark` : "muse-spark",
    demo: false,
  });
  app.renderCanvas();
}

export async function executeMuse(app, actionNode, btnRun) {
  const prompt = resolvePromptText(app, actionNode, "in-prompt")?.trim()
    || (actionNode.param || "").trim()
    || null;
  if (!prompt) {
    failRun(app, actionNode, btnRun, "No prompt — Muse needs text: wire some into Prompt or type it.");
    return;
  }

  const sources = collectImageInputs(app, actionNode, "in-images");
  let images = [];
  try {
    images = await Promise.all(sources.map(s => encodeImageInput(s.url)));
  } catch (err) {
    failRun(app, actionNode, btnRun, `Could not read an input image (${err?.message || err}).`);
    return;
  }

  const requestBody = { chat_id: "flow-session", prompt };
  if (images.length > 0) requestBody.images = images;
  markRequestSent(app, actionNode, "/api/debug/muse", requestBody);
  const res = await fetchJson("/api/debug/muse", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
    timeoutMs: MUSE_TIMEOUT_MS,
  });

  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }

  const captureTrace = () => {
    actionNode.apiCall = {
      at: Date.now(),
      hop1: {
        method: "POST",
        url: "/api/debug/muse",
        status: res.status,
        request: redactBlobs(requestBody),
        response: res.data ? redactBlobs(res.data) : null,
        error: res.ok ? null : (res.error || "request failed"),
      },
      hop2: res.ok && res.data && res.data.api_trace ? res.data.api_trace : null,
      // The backend routes by image presence; record which upstream endpoint
      // this return came from so hop 2 doesn't mislabel it /responses.
      hop2url: res.ok && res.data && res.data.api_trace
        ? `https://api.meta.ai/v1/images/${requestBody.images ? "edits" : "generations"}`
        : null,
    };
  };

  if (!res.ok && res.status === 0) {
    if (failUnreachable(app, actionNode, res, btnRun)) { app.renderCanvas(); return; }
    console.warn("Live muse request failed, showing demo outputs:", res.error);
    captureTrace();
    actionNode.status = "Demo (Offline)";
    actionNode.runError = res.error || "Muse request failed";
    actionNode.runNote = null;
    appendRunLog(actionNode, "error", `Backend unreachable (${actionNode.runError}) — demo data`);
    fillMuseResultView(actionNode, {
      imgUrl: sources[0]?.url || null,
      badge: "demo",
      demo: true,
      note: `Demo image (offline) for "${prompt.slice(0, 80)}": the live muse-image-1.0 render appears here once the backend is reachable.`,
    });
    app.renderCanvas();
    return;
  }

  if (!res.ok) {
    resetActionNode(app, actionNode);
    captureTrace();
    actionNode.status = "Failed";
    const remedy = res.data && res.data.remedy ? ` — ${res.data.remedy}` : "";
    actionNode.runError = `${res.error || `HTTP ${res.status}`}${remedy}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const data = res.data || {};
  captureTrace();
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.runNote = null;
  actionNode.latency = formatLatency(data.api_trace?.meta?.latency_s);
  appendRunLog(actionNode, "info", `Passed in ${actionNode.latency} — ${data.output_format || "image"}`);
  const dims = data.image_width ? `${data.image_width}x${data.image_height}` : (data.output_format || "muse-image");
  fillMuseResultView(actionNode, {
    imgUrl: data.image_url || null,
    badge: dims,
    demo: false,
    note: "",
  });
  app.renderCanvas();
}

/** Fill a describe card's result view: description text, input echo, badges. */
export function fillDescribeResultView(actionNode, { text, imgUrl, badge, demo }) {
  actionNode.resultText = text || "";
  actionNode.resultImage = imgUrl || null;
  actionNode.resultImageFallback = svgPlaceholder(demo ? "Demo · description" : "description");
  actionNode.resultImageBadge = badge || "";
  const p = (actionNode.outputs || []).find(o => o.id === "out-text");
  if (p) p.badge = `${(text || "").length} chars`;
}

/**
 * Pure: backend mask ref at the upstream end of an edge — split item pins
 * carry {debug_id, mask_id} from the SAM run that produced them. Null when
 * the pin has no live ref (demo data, or never a SAM item).
 */
export function upstreamMaskRef(app, edge) {
  const src = (app.nodes || []).find(n => n.id === edge.from);
  if (!src) return null;
  const pin = (src.outputs || []).find(p => p.id === edge.fromPort);
  const ref = pin && pin.ref;
  if (!ref || !ref.debug_id || !Number.isInteger(ref.mask_id) || ref.mask_id < 0) return null;
  return { debug_id: String(ref.debug_id), mask_id: ref.mask_id };
}

export async function executeDescribe(app, actionNode, btnRun) {
  const incomingEdge = app.edges.find(e => e.to === actionNode.id);
  const inputNode = incomingEdge
    ? app.nodes.find(n => n.id === incomingEdge.from)
    : app.nodes.find(n => n.id === "asset-photo") || app.nodes[0];

  let imageB64 = null;
  let imageUrl = null;
  const inputSrc = inputNode ? resolveAssetImageUrl(app, inputNode) : null;
  if (inputSrc) {
    if (inputSrc.startsWith("data:image")) {
      imageB64 = inputSrc;
    } else if (inputSrc.startsWith("blob:")) {
      try {
        imageB64 = await readUrlAsDataUrl(inputSrc);
      } catch (err) {
        console.warn("Could not read blob, using fallback", err);
      }
    } else {
      imageUrl = inputSrc;
    }
  }

  if (!inputSrc || inputSrc.startsWith("data:image/svg")) {
    if (btnRun) {
      btnRun.classList.remove("is-running");
      btnRun.textContent = "✓";
    }
    resetActionNode(app, actionNode);
    actionNode.status = "Failed";
    const wiredPin = incomingEdge && inputNode
      ? (inputNode.outputs || []).find(p => p.id === incomingEdge.fromPort) : null;
    const maskWired = !!incomingEdge
      && (wiredPin?.type === "mask" || !!upstreamMaskRef(app, incomingEdge));
    actionNode.runError = maskWired
      ? "A mask is wired into Image Input — put an Apply Mask card between, then wire its region into Describe."
      : "No input image — wire the source photo (or a viewer showing one) into Image Input.";
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const requestBody = {
    chat_id: "flow-session",
    image_b64: imageB64 || undefined,
    image_url: imageUrl || undefined,
    filename: inputNode?.subtitle || "photo.jpg"
  };
  const endpoint = "/api/debug/describe";
  markRequestSent(app, actionNode, endpoint, requestBody);
  const res = await fetchJson(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
    timeoutMs: DESCRIBE_TIMEOUT_MS
  });

  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }

  const captureTrace = () => {
    actionNode.apiCall = {
      at: Date.now(),
      hop1: {
        method: "POST",
        url: endpoint,
        status: res.status,
        request: redactBlobs(requestBody),
        response: res.data ? redactBlobs(res.data) : null,
        error: res.ok ? null : (res.error || "request failed"),
      },
      hop2: res.ok && res.data && res.data.api_trace ? res.data.api_trace : null,
    };
  };

  if (!res.ok && res.status === 0) {
    if (failUnreachable(app, actionNode, res, btnRun)) { app.renderCanvas(); return; }
    console.warn("Live describe request failed, showing demo outputs:", res.error);
    captureTrace();
    actionNode.status = "Demo (Offline)";
    actionNode.runError = res.error || "Describe request failed";
    actionNode.runNote = null;
    appendRunLog(actionNode, "error", `Backend unreachable (${actionNode.runError}) — demo data`);
    fillDescribeResultView(actionNode, {
      text: "Demo description (offline): the live whole-scene read appears here once the backend is reachable.",
      imgUrl: inputSrc,
      badge: "demo",
      demo: true,
    });
    app.renderCanvas();
    return;
  }

  if (!res.ok) {
    resetActionNode(app, actionNode);
    captureTrace();
    actionNode.status = "Failed";
    const remedy = res.data && res.data.remedy ? ` — ${res.data.remedy}` : "";
    actionNode.runError = `${res.error || `HTTP ${res.status}`}${remedy}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const data = res.data || {};
  captureTrace();
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.runNote = null;
  actionNode.latency = formatLatency(data.api_trace?.meta?.latency_s);
  appendRunLog(actionNode, "info", `Passed in ${actionNode.latency} — scene described`);
  fillDescribeResultView(actionNode, {
    text: data.assistant_text || "",
    imgUrl: inputSrc,
    badge: "description",
    demo: false,
  });
  app.renderCanvas();
}

export function fillApplyMaskResultView(actionNode, { imgUrl, badge, demo, text }) {
  actionNode.resultImage = imgUrl || null;
  actionNode.resultImageFallback = svgPlaceholder("Mask region");
  actionNode.resultImageBadge = badge || "region";
  actionNode.resultText = text || "";
  const p = (actionNode.outputs || []).find(o => o.id === "out-image");
  if (p) p.badge = demo ? "Demo" : (badge || "Crop");
}

export async function executeApplyMask(app, actionNode, btnRun) {
  // Mask-ref contract: the backend crops one mask from a prior SAM
  // debug dir into a plain RGB region image (no model).
  const failFast = (message) => {
    if (btnRun) {
      btnRun.classList.remove("is-running");
      btnRun.textContent = "✓";
    }
    resetActionNode(app, actionNode);
    actionNode.status = "Failed";
    actionNode.runError = message;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
  };

  const incomingEdge = (app.edges || []).find(e => e.to === actionNode.id && e.toPort === "in-mask")
    || (app.edges || []).find(e => e.to === actionNode.id);
  if (!incomingEdge) {
    failFast("No mask — run SAM, explode Masks, and wire one Mask item pin into Mask.");
    return;
  }
  let maskRef = upstreamMaskRef(app, incomingEdge);
  if (!maskRef) {
    // Wired from the "Masks" list pin: use the first mask of the upstream
    // SAM card's last live run instead of forcing a manual explode first.
    const src = (app.nodes || []).find(n => n.id === incomingEdge.from);
    const resp = (src && src.apiCall && src.apiCall.hop1 && src.apiCall.hop1.response) || {};
    const debugId = resp.debug_id || src && src.lastDebugId;
    const masks = resp.masks || (src && src.masks) || [];
    const first = Array.isArray(masks) ? masks[0] : null;
    if (debugId && first) {
      maskRef = { debug_id: String(debugId), mask_id: Number(first.id != null ? first.id : 0) };
      appendRunLog(actionNode, "info", `Using mask ${maskRef.mask_id} from the "${src.title}" list.`);
    }
  }
  if (!maskRef) {
    failFast("No mask ref on that pin - run SAM live on this photo, then wire its Masks pin (or one mask item pin) into Mask.");
    return;
  }

  const requestBody = {
    chat_id: "flow-session",
    debug_id: maskRef.debug_id,
    mask_id: maskRef.mask_id,
  };
  const endpoint = "/api/debug/apply-mask";
  markRequestSent(app, actionNode, endpoint, requestBody);
  const res = await fetchJson(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
    timeoutMs: APPLY_MASK_TIMEOUT_MS
  });

  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }

  const captureTrace = () => {
    actionNode.apiCall = {
      at: Date.now(),
      hop1: {
        method: "POST",
        url: endpoint,
        status: res.status,
        request: redactBlobs(requestBody),
        response: res.data ? redactBlobs(res.data) : null,
        error: res.ok ? null : (res.error || "request failed"),
      },
      hop2: null,
    };
  };

  if (!res.ok && res.status === 0) {
    if (failUnreachable(app, actionNode, res, btnRun)) { app.renderCanvas(); return; }
    console.warn("Live apply-mask request failed, showing demo outputs:", res.error);
    captureTrace();
    actionNode.status = "Demo (Offline)";
    actionNode.runError = res.error || "Apply-mask request failed";
    actionNode.runNote = null;
    appendRunLog(actionNode, "error", `Backend unreachable (${actionNode.runError}) — demo data`);
    fillApplyMaskResultView(actionNode, {
      imgUrl: null,
      badge: "demo",
      demo: true,
      text: "Demo region (offline): the live mask crop appears here once the backend is reachable.",
    });
    app.renderCanvas();
    return;
  }

  if (!res.ok) {
    resetActionNode(app, actionNode);
    captureTrace();
    actionNode.status = "Failed";
    const remedy = res.data && res.data.remedy ? ` — ${res.data.remedy}` : "";
    actionNode.runError = `${res.error || `HTTP ${res.status}`}${remedy}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const data = res.data || {};
  captureTrace();
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.runNote = null;
  actionNode.latency = "local";
  appendRunLog(actionNode, "info", `Passed — mask M${maskRef.mask_id} applied`);
  fillApplyMaskResultView(actionNode, {
    imgUrl: data.crop
      ? `/api/chats/${data.chat_id || "flow-session"}/file/debug/${data.debug_id}/${data.crop}`
      : null,
    badge: (data.width && data.height) ? `${data.width}×${data.height}` : `M${maskRef.mask_id}`,
    demo: false,
    text: data.assistant_text || "",
  });
  app.renderCanvas();
}

const PROGRESS_POLL_MS = 2000;
const PROGRESS_LOG_MARK = "▸ ";

/** Chat-level live progress file the backend appends stage events to mid-run. */
export function progressUrl(chatId) {
  return `/api/chats/${chatId}/file/progress.jsonl`;
}

/**
 * Pure: parse newly arrived progress.jsonl lines. Blank lines pass; a corrupt
 * finished line is skipped; an unterminated tail is partial, so the cursor
 * holds for the next poll.
 */
export function parseProgressEvents(text, fromLine) {
  const lines = String(text || "").split("\n");
  const terminated = String(text || "").endsWith("\n");
  const events = [];
  let next = fromLine;
  for (let i = fromLine; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) {
      next = i + 1;
      continue;
    }
    try {
      const ev = JSON.parse(line);
      if (ev && typeof ev === "object") events.push(ev);
      next = i + 1;
    } catch {
      const isTail = i === lines.length - 1 && !terminated;
      if (isTail) break;
      next = i + 1;
    }
  }
  return { events, nextLine: next };
}

/** Pure: keep only this run's events (the live file multiplexes concurrent runs). */
export function filterProgressEvents(events, cid) {
  if (!cid) return events;
  return events.filter(e => e.cid === cid);
}

/** Fold progress events into a card: live bar state plus run-log lines. */
export function applyProgressEvents(node, events) {
  node.runLog = node.runLog || [];
  for (const ev of events) {
    if (typeof ev.frac === "number") {
      node.progress = {
        frac: Math.min(1, Math.max(0, ev.frac)),
        label: ev.msg || (node.progress && node.progress.label) || ev.stage || "",
      };
    }
    if (ev.msg) appendRunLog(node, "info", `${PROGRESS_LOG_MARK}${ev.msg}`);
  }
}

/** Direct-DOM bar update (no canvas re-render: cheap and drag-safe mid-run). */
export function updateProgressBar(node) {
  if (typeof document === "undefined") return;
  const cardEl = document.getElementById(node.id);
  if (!cardEl || !node.progress) return;
  const fill = cardEl.querySelector(".flow-progress-fill");
  const label = cardEl.querySelector(".flow-progress-label");
  if (fill) fill.style.width = `${Math.round(node.progress.frac * 100)}%`;
  if (label) label.textContent = node.progress.label;
}

async function drainProgressOnce(app, node, url, cid) {
  const bust = `${url}?n=${node._progressLine || 0}`;
  const res = await fetch(bust, { cache: "no-store" });
  if (!res.ok) return;
  const { events, nextLine } = parseProgressEvents(await res.text(), node._progressLine || 0);
  node._progressLine = nextLine;
  const mine = filterProgressEvents(events, cid);
  if (mine.length === 0) return;
  applyProgressEvents(node, mine);
  updateProgressBar(node);
  if (app.selectedNodeId === node.id && typeof app.renderInspectorContent === "function") {
    app.renderInspectorContent();
  }
}

export function startProgressPoll(app, node, chatId, cid) {
  stopProgressPoll(node);
  node._progressLine = 0;
  node._progressCid = cid;
  node.progress = { frac: 0, label: "starting…" };
  updateProgressBar(node);
  const url = progressUrl(chatId);
  node._pollTimer = setInterval(() => {
    drainProgressOnce(app, node, url, cid).catch(() => {});
  }, PROGRESS_POLL_MS);
}

export function stopProgressPoll(node) {
  if (node._pollTimer) {
    clearInterval(node._pollTimer);
    node._pollTimer = null;
  }
}

/**
 * Final drain after the call resolves: replaces polled lines with the
 * per-call record so the history is complete even when live polls missed.
 */
export async function finalProgressDrain(app, node, chatId, debugId) {
  stopProgressPoll(node);
  if (!debugId) return;
  try {
    const res = await fetch(`/api/chats/${chatId}/file/debug/${debugId}/progress.jsonl`,
      { cache: "no-store" });
    if (!res.ok) return;
    const { events } = parseProgressEvents(await res.text(), 0);
    node.runLog = (node.runLog || []).filter(e => !String(e.msg || "").startsWith(PROGRESS_LOG_MARK));
    applyProgressEvents(node, events);
  } catch {
    // Progress is best-effort; the run result stands on its own.
  }
}

/**
 * POST with live progress: polls the chat-level progress file while the call
 * runs, then backfills the per-call record so the stage history is complete
 * even when live polls missed. Best-effort — failures never break the run.
 */
async function fetchWithProgress(app, node, chatId, endpoint, requestBody, timeoutMs) {
  startProgressPoll(app, node, chatId, requestBody.client_debug_id);
  let res;
  try {
    res = await fetchJson(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
      timeoutMs,
    });
  } finally {
    stopProgressPoll(node);
  }
  await finalProgressDrain(app, node, chatId, res && res.data && res.data.debug_id);
  return res;
}

const SAPIENS_TIMEOUT_MS = 180000;

/** Overlay render per Sapiens kind, as written by the backend debug dir. */
const SAPIENS_OVERLAY = {
  pose: "pose_keypoints.jpg",
  seg: "seg_overlay.jpg",
  normal: "normal.jpg",
  depth: "depth.jpg",
};
/** Grayscale class-id map backing seg hover + isolate, beside the overlay. */
const SAPIENS_SEG_LABELS = "seg_labels.png";
/** Raw Sapiens2 artifacts a downstream solver can consume (not the JPG previews). */
const SAPIENS_ARTIFACT = { pose: "pose.npz", seg: "seg_labels.png", normal: "normals.npy", depth: "depth_m.npy" };

/**
 * Pure: the image edge into a Sapiens card — the Image pin first, else any
 * incoming edge (pose/normal/depth/seg all take one image).
 */
export function sapiensInputEdge(app, actionNode) {
  const incoming = (app.edges || []).filter(e => e.to === actionNode.id);
  return incoming.find(e => e.toPort === "in-image") || incoming[0] || null;
}

/** Fill a Sapiens card's result view: overlay render, summary, pin badges. */
export function fillSapiensResultView(actionNode, kind, { text, imgUrl, badge, demo, stats, labelsUrl }) {
  actionNode.resultText = text || "";
  actionNode.resultKeypoints = (kind === "pose" && stats && Array.isArray(stats.keypoints)) ? stats.keypoints : null;
  actionNode.resultSegLegend = (kind === "seg" && stats && stats.legend) ? stats.legend : null;
  actionNode.resultSegLabelsUrl = (kind === "seg" && labelsUrl) ? labelsUrl : null;
  actionNode._segIsolated = null;
  const dims = kind === "pose" ? stats : actionNode.resultSegLegend;
  const dw = dims ? Number(dims.image_width || dims.width) : 0;
  const dh = dims ? Number(dims.image_height || dims.height) : 0;
  actionNode.imageWidth = dw > 0 ? dw : 0;
  actionNode.imageHeight = dh > 0 ? dh : 0;
  actionNode.resultImage = imgUrl || null;
  actionNode.resultImageFallback = svgPlaceholder(demo ? `Demo · ${kind}` : kind);
  actionNode.resultImageBadge = badge || "";
  const setBadge = (id, value) => {
    const p = (actionNode.outputs || []).find(o => o.id === id);
    if (p) p.badge = value;
  };
  if (kind === "pose") {
    setBadge("out-kpts", "308 pts");
    setBadge("out-face", "238 pts");
    setBadge("out-body", "70 pts");
  } else if (kind === "seg") {
    const n = stats && Array.isArray(stats.classes) ? stats.classes.length : "—";
    setBadge("out-parts", `${n} classes`);
    setBadge("out-skin", "Mask");
    setBadge("out-clothing", "Mask");
  } else if (kind === "normal") {
    setBadge("out-normals", "Normals");
  } else if (kind === "depth") {
    const r = stats && Array.isArray(stats.depth_m) ? stats.depth_m : null;
    setBadge("out-depth", r ? `${r[0]}–${r[1]}m` : "Metres");
  }
}

const SEG_SPAWN_CAP = 12;
const SEG_THUMB_MAX = 192;

/**
 * Pure: one spawned asset card for a detected seg class. The thumbnail is a
 * same-shape mask render (or null when the labels map was unreachable —
 * the card then shows its placeholder and the counts stay real).
 */
export function buildSegClassNode(actionNode, row, thumbUrl, dims, index) {
  const slug = String(row.name || `class_${row.id}`).toLowerCase()
    .replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || `class-${row.id}`;
  return {
    id: uid("asset-segclass"),
    nodeType: "asset",
    assetKind: "image",
    title: `Asset: Seg · ${row.name}`,
    subtitle: `class_${row.id}_${slug}.png`,
    category: "Mask Asset",
    badgeVariant: "success-light",
    iconVariant: "green",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><path d="M8 12a4 4 0 0 1 8 0"/></svg>`,
    imgUrl: thumbUrl || null,
    imgFallback: svgPlaceholder(`Seg · ${row.name}`),
    imgBadge: `${row.count.toLocaleString("en-US")} px · ${row.pct.toFixed(1)}%`,
    description: `Detected part class "${row.name}" (${row.count.toLocaleString("en-US")} px, ${row.pct.toFixed(1)}% of frame). Wire its image pin onward to work on this region.`,
    latency: "Produced",
    status: "Produced",
    imageWidth: dims.w,
    imageHeight: dims.h,
    tensorShape: `[${dims.h}, ${dims.w}] uint8 mask`,
    format: "PNG mask",
    segClassId: String(row.id),
    inputs: [
      { id: "in-mask", name: "Class Mask", type: "mask", color: "#ec4899", badge: "Mask" }
    ],
    outputs: [
      { id: "out-image", name: `${row.name} Region`, type: "image", color: "#10b981", badge: `${dims.w}x${dims.h}` }
    ],
    x: actionNode.x + 480,
    y: actionNode.y + index * 300
  };
}

/**
 * Pure: a decoded labels record is usable only with positive integer dims
 * and a pixel buffer of exactly w*h — anything else (a 0x0 record from a
 * mishandled decode, a truncated buffer) degrades to placeholder thumbs
 * instead of crashing the spawn.
 */
export function validSegLabels(rec) {
  if (!rec || !Number.isInteger(rec.w) || !Number.isInteger(rec.h)) return null;
  if (rec.w <= 0 || rec.h <= 0) return null;
  if (!rec.pixels || rec.pixels.length !== rec.w * rec.h) return null;
  return rec;
}

/**
 * Browser-only: isolated-mask thumbnail for one class — class pixels in
 * the legend color, everything else transparent, downscaled to fit
 * SEG_THUMB_MAX. Null outside a browser or on bad dims (never throws).
 */
export function segClassThumbUrl(labels, w, h, classId, color) {
  if (typeof document === "undefined") return null;
  if (!Number.isInteger(w) || !Number.isInteger(h) || w <= 0 || h <= 0) return null;
  if (!labels || labels.length !== w * h) return null;
  const keep = Number(classId);
  const [r, g, b] = color || [128, 128, 128];
  const full = document.createElement("canvas");
  full.width = w; full.height = h;
  const fctx = full.getContext("2d");
  const img = fctx.createImageData(w, h);
  for (let p = 0, o = 0; p < labels.length; p++, o += 4) {
    if (labels[p] === keep) {
      img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  fctx.putImageData(img, 0, 0);
  const scale = Math.min(1, SEG_THUMB_MAX / Math.max(w, h));
  const tw = Math.max(1, Math.round(w * scale));
  const th = Math.max(1, Math.round(h * scale));
  const thumb = document.createElement("canvas");
  thumb.width = tw; thumb.height = th;
  thumb.getContext("2d").drawImage(full, 0, 0, tw, th);
  return thumb.toDataURL("image/png");
}

/**
 * Spawn one asset card per detected seg class (background dropped, biggest
 * first, capped). Re-runs clear prior spawns via clearDownstreamOutputs
 * (spawned cards carry no templateKey), so this only ever adds. Resolves
 * the spawned ids; an empty legend spawns nothing and says so loudly.
 */
export async function spawnSegClassCards(app, actionNode, legend, labelsUrl) {
  const rows = segSpawnRows(legend);
  if (rows.length === 0) {
    actionNode.runNote = "No foreground classes detected — nothing to spawn.";
    appendRunLog(actionNode, "info", "Seg passed with no foreground classes; no cards spawned.");
    return [];
  }
  let rec = null;
  try {
    rec = labelsUrl ? await loadSegLabels(labelsUrl) : null;
  } catch {
    rec = null;
  }
  rec = validSegLabels(rec);
  const dims = { w: Number(legend.width) || 0, h: Number(legend.height) || 0 };
  if (rec) {
    dims.w = rec.w;
    dims.h = rec.h;
  }
  const shown = rows.slice(0, SEG_SPAWN_CAP);
  const ids = [];
  shown.forEach((row, i) => {
    const thumb = rec ? segClassThumbUrl(rec.pixels, rec.w, rec.h, row.id, row.color) : null;
    const node = buildSegClassNode(actionNode, row, thumb, dims, i);
    app.nodes.push(node);
    app.edges.push({
      id: `e-${actionNode.id}-${node.id}`, from: actionNode.id, to: node.id,
      fromPort: "out-parts", toPort: "in-mask", label: row.name
    });
    ids.push(node.id);
  });
  const extra = rows.length - shown.length;
  const notes = [];
  if (!rec) {
    notes.push("labels map unreachable — cards show counts only.");
    appendRunLog(actionNode, "info", "Labels map unreachable — class cards show counts only (hover/isolate off too).");
  }
  if (extra > 0) notes.push(`+${extra} more classes in Result → legend.`);
  actionNode.runNote = notes.length > 0 ? notes.join(" ") : null;
  appendRunLog(actionNode, "info",
    `Spawned ${ids.length} class card${ids.length === 1 ? "" : "s"}${extra > 0 ? ` (${extra} more in legend)` : ""}.`);
  return ids;
}

export async function executeSapiens(app, actionNode, btnRun, kind) {
  const edge = sapiensInputEdge(app, actionNode);
  const url = edge ? upstreamImageUrl(app, edge) : null;
  if (!edge || !url) {
    failRun(app, actionNode, btnRun, "No image — wire a photo, crop, box or mask into Image.");
    return;
  }
  const src = (app.nodes || []).find(n => n.id === edge.from);

  let encoded;
  try {
    encoded = await encodeImageInput(url);
  } catch (err) {
    failRun(app, actionNode, btnRun, `Could not read the input image (${err?.message || err}).`);
    return;
  }

  const requestBody = { chat_id: "flow-session", ...encoded, filename: src?.subtitle || "crop.jpg" };
  const endpoint = `/api/debug/sapiens-${kind}`;
  markRequestSent(app, actionNode, endpoint, requestBody);
  actionNode.apiCall.local = true;
  requestBody.client_debug_id = uid("flow");
  const res = await fetchWithProgress(app, actionNode, "flow-session", endpoint, requestBody,
    SAPIENS_TIMEOUT_MS);

  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }

  // Local GPU call: hop 1 only, no upstream trace by design (flagged local
  // so the views say so instead of blaming errors).
  const captureTrace = () => {
    actionNode.apiCall = {
      at: Date.now(),
      local: true,
      hop1: {
        method: "POST",
        url: endpoint,
        status: res.status,
        request: redactBlobs(requestBody),
        response: res.data ? redactBlobs(res.data) : null,
        error: res.ok ? null : (res.error || "request failed"),
      },
      hop2: null,
    };
  };

  if (!res.ok && res.status === 0) {
    if (failUnreachable(app, actionNode, res, btnRun)) { app.renderCanvas(); return; }
    console.warn(`Live sapiens-${kind} request failed, showing demo outputs:`, res.error);
    captureTrace();
    // Demo reuses the mock asset producer, then flags the card honestly.
    produceOutputsForAction(app, actionNode);
    actionNode.status = "Demo (Offline)";
    actionNode.runError = res.error || `Sapiens ${kind} request failed`;
    actionNode.runNote = null;
    appendRunLog(actionNode, "error", `Backend unreachable (${actionNode.runError}) — demo data`);
    fillSapiensResultView(actionNode, kind, {
      text: `Demo ${kind} (offline): the live Sapiens2 render appears here once the backend is reachable.`,
      imgUrl: url,
      badge: "demo",
      demo: true,
      stats: null,
    });
    app.renderCanvas();
    return;
  }

  if (!res.ok) {
    resetActionNode(app, actionNode);
    captureTrace();
    actionNode.status = "Failed";
    const remedy = res.data && res.data.remedy ? ` — ${res.data.remedy}` : "";
    actionNode.runError = `${res.error || `HTTP ${res.status}`}${remedy}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const data = res.data || {};
  captureTrace();
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.runNote = null;
  // Backend embeds its measured seconds in the summary ("(12.3s, local …)").
  const secs = / \(([\d.]+)s,/.exec(data.assistant_text || "");
  actionNode.latency = secs ? formatLatency(Number(secs[1])) : "local";
  appendRunLog(actionNode, "info", `Passed in ${actionNode.latency} — Sapiens2 ${kind}`);
  if (data.debug_id) {
    actionNode.artifacts = actionNode.artifacts || {};
    actionNode.artifacts[kind] = `/api/chats/${data.chat_id || "flow-session"}/file/debug/${data.debug_id}/${SAPIENS_ARTIFACT[kind]}`;
  }
  fillSapiensResultView(actionNode, kind, {
    text: data.assistant_text || "",
    imgUrl: `/api/chats/${data.chat_id || "flow-session"}/file/debug/${data.debug_id}/${SAPIENS_OVERLAY[kind]}`,
    badge: `sapiens2-${kind}`,
    demo: false,
    stats: data.stats || null,
    labelsUrl: kind === "seg" ? `/api/chats/${data.chat_id || "flow-session"}/file/debug/${data.debug_id}/${SAPIENS_SEG_LABELS}` : null,
  });
  if (kind === "seg" && data.stats && data.stats.legend) {
    // Keep the board clean: the class legend lives inside the seg card. The
    // per-class cards are available via the library/apply-mask if a specific
    // class region is needed, but a seg run should not litter the canvas.
    const legend = data.stats.legend || {};
    const n = Object.keys(legend.counts || {}).length;
    appendRunLog(actionNode, "info", `Seg: ${n} classes kept in the in-card legend (no class cards spawned).`);
    actionNode.runNote = null;
  }
  app.renderCanvas();
}

/**
 * Pure: mesh ref at the upstream end of an edge — GLB wins over OBJ.
 * Export cards hold glbUrl; body-fit and avatar cards hold meshObjUrl.
 */
export function upstreamMeshRef(app, edge) {
  const src = (app.nodes || []).find(n => n.id === edge.from);
  if (!src) return null;
  if (src.glbUrl) return { url: src.glbUrl, format: "glb" };
  if (src.meshObjUrl) return { url: src.meshObjUrl, format: "obj" };
  return null;
}

const MESH_RENDER_TIMEOUT_MS = 120000;
const MESH_MULTIVIEW_TIMEOUT_MS = 300000;

/**
 * Fill a mesh-render card's result view: still, summary, pin badges.
 * Takes the server still ({ dataUrl, width, height }) and throws when it
 * carries no pixels, so an empty response fails loud instead of passing
 * with an empty result view.
 */
export function fillMeshRenderResultView(actionNode, mode, { dataUrl, width, height, backend }) {
  if (!dataUrl) throw new Error("renderer returned no pixels");
  const label = mode === "depth" ? "depth map" : mode === "normal" ? "normal map"
    : mode === "multiview" ? "view sheet" : "shaded still";
  actionNode.resultText = `Server ${label} (${width}x${height}) rendered from the wired mesh (${backend || "cpu"}).`;
  actionNode.resultImage = dataUrl;
  actionNode.resultImageFallback = svgPlaceholder(`Mesh ${mode}`);
  actionNode.resultImageBadge = `${width}x${height}`;
  actionNode.imageWidth = width || 0;
  actionNode.imageHeight = height || 0;
  const pinId = mode === "depth" ? "out-depth" : mode === "normal" ? "out-normal" : "out-image";
  const p = (actionNode.outputs || []).find(o => o.id === pinId);
  if (p) p.badge = `${width}x${height}`;
}

/**
 * Pure: the mesh edge into a render card — the Mesh pin first, else any
 * incoming edge.
 */
export function meshInputEdge(app, actionNode) {
  const incoming = (app.edges || []).filter(e => e.to === actionNode.id);
  return incoming.find(e => e.toPort === "in-mesh") || incoming[0] || null;
}

/**
 * Pure: why a render card has no mesh to work with — names the upstream
 * card and reads its state, so demo/stale-backend cases diagnose
 * themselves instead of a bare "no mesh".
 */
export function meshInputDiagnosis(app, actionNode, edge) {
  if (!edge) return "No mesh — wire a fitted mesh (body-fit, avatar, exporter) into Mesh.";
  const src = (app.nodes || []).find(n => n.id === edge.from);
  const name = src ? (src.title || src.id) : edge.from;
  const status = src ? (src.status || "unknown state") : "missing card";
  if (!src) return `No mesh — the wired card is gone; rewire a fitted mesh into Mesh.`;
  if (/Demo/i.test(status)) {
    return `No mesh — "${name}" holds demo data (${status}), and demo runs produce no mesh. Run it live first.`;
  }
  if (!/Passed|Done/.test(status)) {
    return `No mesh — "${name}" is at "${status}", not Passed. Run it first, then run this card.`;
  }
  return `No mesh — "${name}" passed without a mesh ref. The server predates mesh refs: restart chat_server.py, re-run "${name}", then run this card.`;
}

/**
 * Server-side mesh render: shaded still, depth map, or normal map from
 * POST /api/debug/mesh-render (open3d offscreen, numpy fallback). CPU
 * call, hop-1-only trace. No demo path: a render is real pixels or a
 * loud failure, never a placeholder.
 */
export async function executeMeshRender(app, actionNode, btnRun, mode) {
  const edge = meshInputEdge(app, actionNode);
  const ref = edge ? upstreamMeshRef(app, edge) : null;
  if (!edge || !ref) {
    failRun(app, actionNode, btnRun, meshInputDiagnosis(app, actionNode, edge));
    return;
  }
  let encoded;
  try {
    encoded = await encodeMeshInput(ref.url);
  } catch (err) {
    failRun(app, actionNode, btnRun, `Could not read the mesh (${err?.message || err}).`);
    return;
  }
  const requestBody = { chat_id: "flow-session", ...encoded,
    filename: ref.format === "glb" ? "mesh.glb" : "mesh.obj", mode, size: 512 };
  const endpoint = "/api/debug/mesh-render";
  appendRunLog(actionNode, "info", `Rendering ${mode} still (server)…`);
  markRequestSent(app, actionNode, endpoint, requestBody);
  actionNode.apiCall.local = true;
  app.renderCanvas();
  const res = await fetchJson(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
    timeoutMs: MESH_RENDER_TIMEOUT_MS,
  });

  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }

  // Local CPU call: hop 1 only, no upstream trace by design.
  const captureTrace = () => {
    actionNode.apiCall = {
      at: Date.now(),
      local: true,
      hop1: {
        method: "POST",
        url: endpoint,
        status: res.status,
        request: redactBlobs(requestBody),
        response: res.data ? redactBlobs(res.data) : null,
        error: res.ok ? null : (res.error || "request failed"),
      },
      hop2: null,
    };
  };

  if (!res.ok && res.status === 0) {
    if (failUnreachable(app, actionNode, res, btnRun)) { app.renderCanvas(); return; }
    captureTrace();
    actionNode.status = "Failed";
    actionNode.runError = res.error || "Mesh-render request failed";
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  if (!res.ok) {
    resetActionNode(app, actionNode);
    captureTrace();
    actionNode.status = "Failed";
    const remedy = res.data && res.data.remedy ? ` — ${res.data.remedy}` : "";
    actionNode.runError = `${res.error || `HTTP ${res.status}`}${remedy}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const data = res.data || {};
  captureTrace();
  try {
    fillMeshRenderResultView(actionNode, mode, {
      dataUrl: data.image_url || null,
      width: Number(data.width) || 0,
      height: Number(data.height) || 0,
      backend: data.render_backend || "cpu",
    });
  } catch (err) {
    failRun(app, actionNode, btnRun, err?.message || "renderer returned no pixels");
    return;
  }
  const backend = data.render_backend || "cpu";
  actionNode.status = "Passed";
  actionNode.latency = backend;
  actionNode.runError = null;
  actionNode.runNote = `Rendered server-side (${backend}) — no GPU slot.`;
  appendRunLog(actionNode, "info", `Passed — ${mode} still ${data.width}x${data.height} (${backend})`);
  app.renderCanvas();
}

/**
 * Server-side multiview render: 2/4/6/8 shaded stills at even azimuths
 * composed into one labelled contact sheet by POST /api/debug/mesh-render
 * (open3d offscreen, numpy fallback). CPU call, hop-1-only trace. No
 * demo path: failures fail loud.
 */
export async function executeMeshMultiview(app, actionNode, btnRun) {
  const edge = meshInputEdge(app, actionNode);
  const ref = edge ? upstreamMeshRef(app, edge) : null;
  if (!edge || !ref) {
    failRun(app, actionNode, btnRun, meshInputDiagnosis(app, actionNode, edge));
    return;
  }
  const count = parseInt(actionNode.param ?? "4", 10);
  if (!MULTIVIEW_COUNTS.includes(count)) {
    failRun(app, actionNode, btnRun, `Views must be one of 2, 4, 6, 8 (got "${actionNode.param}").`);
    return;
  }
  const framing = ["body", "waist", "face"].includes(actionNode.framing) ? actionNode.framing : "body";
  let encoded;
  try {
    encoded = await encodeMeshInput(ref.url);
  } catch (err) {
    failRun(app, actionNode, btnRun, `Could not read the mesh (${err?.message || err}).`);
    return;
  }
  const requestBody = { chat_id: "flow-session", ...encoded,
    filename: ref.format === "glb" ? "mesh.glb" : "mesh.obj",
    mode: "multiview", count, framing, size: MULTIVIEW_CELL };
  const endpoint = "/api/debug/mesh-render";
  appendRunLog(actionNode, "info", `Rendering ${count} views (${framing}, server)…`);
  markRequestSent(app, actionNode, endpoint, requestBody);
  actionNode.apiCall.local = true;
  app.renderCanvas();
  const res = await fetchJson(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
    timeoutMs: MESH_MULTIVIEW_TIMEOUT_MS,
  });

  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }

  // Local CPU call: hop 1 only, no upstream trace by design.
  const captureTrace = () => {
    actionNode.apiCall = {
      at: Date.now(),
      local: true,
      hop1: {
        method: "POST",
        url: endpoint,
        status: res.status,
        request: redactBlobs(requestBody),
        response: res.data ? redactBlobs(res.data) : null,
        error: res.ok ? null : (res.error || "request failed"),
      },
      hop2: null,
    };
  };

  if (!res.ok && res.status === 0) {
    if (failUnreachable(app, actionNode, res, btnRun)) { app.renderCanvas(); return; }
    captureTrace();
    actionNode.status = "Failed";
    actionNode.runError = res.error || "Mesh-render request failed";
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  if (!res.ok) {
    resetActionNode(app, actionNode);
    captureTrace();
    actionNode.status = "Failed";
    const remedy = res.data && res.data.remedy ? ` — ${res.data.remedy}` : "";
    actionNode.runError = `${res.error || `HTTP ${res.status}`}${remedy}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const data = res.data || {};
  captureTrace();
  try {
    fillMeshRenderResultView(actionNode, "multiview", {
      dataUrl: data.image_url || null,
      width: Number(data.width) || 0,
      height: Number(data.height) || 0,
      backend: data.render_backend || "cpu",
    });
  } catch (err) {
    failRun(app, actionNode, btnRun, err?.message || "renderer returned no pixels");
    return;
  }
  const backend = data.render_backend || "cpu";
  actionNode.status = "Passed";
  actionNode.latency = backend;
  actionNode.runError = null;
  actionNode.runNote = `Rendered server-side (${backend}${framing !== "body" ? `, ${framing}` : ""}) — no GPU slot.`;
  appendRunLog(actionNode, "info", `Passed — ${count}-view ${framing} sheet ${data.width}x${data.height} (${backend})`);
  app.renderCanvas();
}

const FITOVERLAY_TIMEOUT_MS = 120000;

/**
 * Pure: fit ref at the upstream end of a mesh edge — set by live body-fit
 * fills and passed through the avatar asset. Exporters drop it by design.
 */
export function upstreamFitCamera(app, edge) {
  const src = (app.nodes || []).find(n => n.id === edge.from);
  return (src && src.fitCamera) || null;
}

/**
 * Pure: why an overlay card has no fit — names the upstream card and reads
 * its state, so demo/exporter/stale-backend cases diagnose themselves.
 */
export function upstreamFitDiagnosis(app, actionNode, meshEdge) {
  if (!meshEdge) return "No fit — wire the body-fit (or avatar) card into Fit.";
  const src = (app.nodes || []).find(n => n.id === meshEdge.from);
  const name = src ? (src.title || src.id) : meshEdge.from;
  const status = src ? (src.status || "unknown state") : "missing card";
  if (!src) return "No fit — the wired card is gone; rewire the body-fit card into Fit.";
  if (/Demo/i.test(status)) {
    return `No fit — "${name}" holds demo data (${status}). Run it live first.`;
  }
  if (src.glbUrl && !src.fitCamera) {
    return `No fit — "${name}" is an export; the exporter drops the fit. Wire the body-fit (or avatar) card instead.`;
  }
  if (!/Passed|Done/.test(status)) {
    return `No fit — "${name}" is at "${status}", not Passed. Run it first.`;
  }
  return `No fit — "${name}" passed without a fit ref. The server predates fit refs: restart chat_server.py, re-run "${name}", then run this card.`;
}

/** Fill a fit-overlay card's result view: blend plus the two frames, badges. */
export function fillFitOverlayResultView(actionNode, { overlayUrl, photoUrl, renderUrl, badge, demo, text, width, height }) {
  actionNode.resultText = text || "";
  actionNode.resultImage = overlayUrl || null;
  actionNode.resultImageFallback = svgPlaceholder(demo ? "Demo · fit-overlay" : "fit-overlay");
  actionNode.resultImageBadge = badge || "";
  actionNode.imageWidth = width || 0;
  actionNode.imageHeight = height || 0;
  const setPin = (id, url) => {
    const p = (actionNode.outputs || []).find(o => o.id === id);
    if (p) {
      p.badge = url ? badge || "—" : "—";
      p.thumb = url || null;
    }
  };
  setPin("out-overlay", overlayUrl);
  setPin("out-photo", photoUrl);
  setPin("out-render", renderUrl);
}

/**
 * Fit overlay: the backend re-renders the fitted mesh in the photo frame
 * (same matched-frame painter body-fit uses) and blends it 50/50 over the
 * photo, plus a photo/render/overlay side-by-side. CPU call, hop-1-only
 * trace; demo when unreachable like every backend card.
 */
export async function executeFitOverlay(app, actionNode, btnRun) {
  const incoming = (app.edges || []).filter(e => e.to === actionNode.id);
  const photoEdge = incoming.find(e => e.toPort === "in-photo") || null;
  const meshEdge = incoming.find(e => e.toPort === "in-mesh") || null;
  const photoUrl = photoEdge ? upstreamImageUrl(app, photoEdge) : null;
  const fit = meshEdge ? upstreamFitCamera(app, meshEdge) : null;
  if (!photoUrl) {
    failRun(app, actionNode, btnRun, "No photo — wire the fitted photo into Photo.");
    return;
  }
  if (!fit) {
    failRun(app, actionNode, btnRun, upstreamFitDiagnosis(app, actionNode, meshEdge));
    return;
  }
  const src = (app.nodes || []).find(n => n.id === photoEdge.from);
  let encoded;
  try {
    encoded = await encodeImageInput(photoUrl);
  } catch (err) {
    failRun(app, actionNode, btnRun, `Could not read the photo (${err?.message || err}).`);
    return;
  }
  const requestBody = { chat_id: "flow-session", debug_id: fit.debug_id,
    ...encoded, filename: src?.subtitle || "photo.jpg" };
  const endpoint = "/api/debug/fit-overlay";
  markRequestSent(app, actionNode, endpoint, requestBody);
  actionNode.apiCall.local = true;
  const res = await fetchJson(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
    timeoutMs: FITOVERLAY_TIMEOUT_MS,
  });

  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }

  // Local CPU call: hop 1 only, no upstream trace by design.
  const captureTrace = () => {
    actionNode.apiCall = {
      at: Date.now(),
      local: true,
      hop1: {
        method: "POST",
        url: endpoint,
        status: res.status,
        request: redactBlobs(requestBody),
        response: res.data ? redactBlobs(res.data) : null,
        error: res.ok ? null : (res.error || "request failed"),
      },
      hop2: null,
    };
  };

  if (!res.ok && res.status === 0) {
    if (failUnreachable(app, actionNode, res, btnRun)) { app.renderCanvas(); return; }
    console.warn("Live fit-overlay request failed, showing demo outputs:", res.error);
    captureTrace();
    actionNode.status = "Demo (Offline)";
    actionNode.runError = res.error || "Fit-overlay request failed";
    actionNode.runNote = null;
    appendRunLog(actionNode, "error", `Backend unreachable (${actionNode.runError}) — demo data`);
    fillFitOverlayResultView(actionNode, {
      text: "Demo overlay (offline): the live matched blend appears here once the backend is reachable.",
      overlayUrl: photoUrl,
      // No fit ran, so no fit frame exists: only the preview echoes the
      // input (once), the frame pins stay bare instead of triplicating it.
      photoUrl: null,
      renderUrl: null,
      badge: "demo",
      demo: true,
    });
    app.renderCanvas();
    return;
  }

  if (!res.ok) {
    resetActionNode(app, actionNode);
    captureTrace();
    actionNode.status = "Failed";
    const remedy = res.data && res.data.remedy ? ` — ${res.data.remedy}` : "";
    actionNode.runError = `${res.error || `HTTP ${res.status}`}${remedy}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const data = res.data || {};
  captureTrace();
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.runNote = null;
  actionNode.latency = "local";
  const W = Number(data.width) || 0, H = Number(data.height) || 0;
  appendRunLog(actionNode, "info", `Passed — fit overlay ${W}x${H} (local CPU)`);
  fillFitOverlayResultView(actionNode, {
    text: data.assistant_text || "",
    overlayUrl: data.overlay_url || null,
    photoUrl: data.photo_url || null,
    renderUrl: data.render_url || null,
    badge: W > 0 ? `${W}x${H}` : "overlay",
    demo: false,
    width: W,
    height: H,
  });
  app.renderCanvas();
}

const BODYFIT_TIMEOUT_MS = 240000;
const EXPORT_TIMEOUT_MS = 120000;

/**
 * Pure: fitted mesh URL at the upstream end of an edge — the body-fit card
 * stores its mesh ref on itself when the live fit lands.
 */
export function upstreamMeshUrl(app, edge) {
  const src = (app.nodes || []).find(n => n.id === edge.from);
  if (!src) return null;
  return src.meshObjUrl || null;
}

/** Fill a body-fit card's result view: mesh sheet, summary, pin badges, mesh ref. */
export function fillBodyFitResultView(actionNode, { text, imgUrl, badge, demo, stats, meshUrl, debugId, chatId }) {
  actionNode.resultText = text || "";
  actionNode.resultImage = imgUrl || null;
  actionNode.resultImageFallback = svgPlaceholder(demo ? "Demo · body-fit" : "body-fit");
  actionNode.resultImageBadge = badge || "";
  actionNode.meshObjUrl = meshUrl || null;
  const setPinBadge = (id, value) => {
    const p = (actionNode.outputs || []).find(o => o.id === id);
    if (p) p.badge = value;
  };
  setPinBadge("out-beta", "45 coeffs");
  if (stats && typeof stats.expr_l2 === "number") setPinBadge("out-expr", `|e| ${stats.expr_l2.toFixed(2)}`);
  actionNode.fitCamera = (!demo && debugId) ? {
    debug_id: debugId,
    chat_id: chatId || "flow-session",
    focal_px: stats && typeof stats.focal_px === "number" ? stats.focal_px : null,
    size_wh: stats && Array.isArray(stats.size_wh) ? stats.size_wh : null,
  } : null;
  const setBadge = (id, value) => {
    const p = (actionNode.outputs || []).find(o => o.id === id);
    if (p) p.badge = value;
  };
  const verts = stats && typeof stats.verts === "number" ? stats.verts : null;
  const faceErr = stats && typeof stats.face_err_px === "number" ? stats.face_err_px : null;
  const betaN = stats && typeof stats.beta_n === "number" ? stats.beta_n : null;
  setBadge("out-mesh", verts != null ? `${verts} verts` : "—");
  setBadge("out-beta", betaN != null ? `${betaN} coeffs` : "—");
}

/**
 * Spawn the live Avatar asset card for a body fit: shows the server-side
 * turntable render and passes the fitted mesh through, so the exporter wires
 * from the asset. Re-runs delete and respawn it via the cascade.
 */
export function spawnAvatarAsset(app, actionNode, { imgUrl, badge, tensorShape, meshUrl, fitCamera }) {
  const assetId = uid("asset-avatar");
  const node = {
    id: assetId,
    nodeType: "asset",
    assetKind: "mesh",
    title: "Asset: Parametric 3D Avatar",
    subtitle: "avatar turntable",
    category: "3D Mesh Asset",
    badgeVariant: "primary-light",
    iconVariant: "soft",
    iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>`,
    imgUrl,
    imgFallback: svgPlaceholder("Avatar turntable"),
    imgBadge: badge,
    description: "Fitted 3D humanoid mesh: server-side turntable of the live body fit. Wire its mesh pin into the exporter.",
    latency: "Produced",
    status: "Artifact",
    tensorShape,
    format: "Turntable render + OBJ ref",
    meshObjUrl: meshUrl || null,
    fitCamera: fitCamera || null,
    inputs: [
      { id: "in-mesh", name: "Fitted Mesh", type: "mesh", color: "#3b82f6", badge: "Mesh" }
    ],
    outputs: [
      { id: "out-mesh", name: "Parametric Mesh", type: "mesh", color: "#3b82f6", badge: "Mesh" }
    ],
    x: actionNode.x + 480,
    y: actionNode.y
  };
  app.nodes.push(node);
  app.edges.push({ id: `e-${actionNode.id}-${assetId}`, from: actionNode.id, to: assetId, fromPort: "out-mesh", toPort: "in-mesh", label: "fitted avatar" });
  return assetId;
}

/** Max diff dimension: full-res diffs would blow the localStorage budget. */
export const DIFF_MAX_DIM = 1024;
/** Diff amplification: per-channel abs diffs are doubled for visibility. */
export const DIFF_AMPLIFY = 2;

/**
 * Pure: fit (w, h) inside maxDim, preserving aspect. No-op when small.
 */
export function fitDims(w, h, maxDim = DIFF_MAX_DIM) {
  const m = Math.max(w, h);
  if (!(m > maxDim)) return { w, h };
  const s = maxDim / m;
  return { w: Math.max(1, Math.round(w * s)), h: Math.max(1, Math.round(h * s)) };
}

/**
 * Pure: amplified absolute RGB difference of two equal RGBA buffers.
 * Returns { data, mean01, maxV } — mean abs diff as 0..1, peak as 0..255.
 * Throws on mismatched buffers (the caller aligns sizes first).
 */
export function absDiffStats(a, b, amplify = DIFF_AMPLIFY) {
  if (!a || !b || a.length !== b.length) throw new Error("mismatched diff buffers");
  const out = new Uint8ClampedArray(a.length);
  let sum = 0, max = 0;
  const n = a.length / 4;
  for (let p = 0, o = 0; p < n; p++, o += 4) {
    for (let c = 0; c < 3; c++) {
      const d = Math.abs(a[o + c] - b[o + c]);
      sum += d;
      if (d > max) max = d;
      out[o + c] = Math.min(255, Math.round(d * amplify));
    }
    out[o + 3] = 255;
  }
  return { data: out, mean01: n > 0 ? sum / (n * 3 * 255) : 0, maxV: max };
}

/**
 * Browser-only: decode an image URL to raw pixels. Dimensions are read
 * before the bitmap closes (a closed bitmap reports 0x0).
 */
export async function fetchImagePixels(url) {
  if (typeof document === "undefined") throw new Error("image decode needs a browser page");
  const res = await fetch(url);
  if (!res.ok) throw new Error(`image fetch failed (HTTP ${res.status})`);
  const bm = await createImageBitmap(await res.blob());
  const w = bm.width, h = bm.height;
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bm, 0, 0);
  const data = ctx.getImageData(0, 0, w, h).data;
  bm.close();
  return { data, w, h };
}

/** Browser-only: resample an RGBA buffer to (nw, nh) via canvas. */
export function scaleImagePixels(px, w, h, nw, nh) {
  const src = document.createElement("canvas");
  src.width = w; src.height = h;
  src.getContext("2d").putImageData(new ImageData(px, w, h), 0, 0);
  const dst = document.createElement("canvas");
  dst.width = nw; dst.height = nh;
  dst.getContext("2d").drawImage(src, 0, 0, nw, nh);
  return dst.getContext("2d").getImageData(0, 0, nw, nh).data;
}

/** Fill an image-diff card's result view: diff map, Δ stats, pin badge. */
export function fillImageDiffResultView(actionNode, { imgUrl, width, height, meanPct, maxV, note }) {
  actionNode.resultText = `Mean Δ ${meanPct}% · peak ${maxV}/255${note ? ` — ${note}` : ""}. Local pixel diff, no backend call.`;
  actionNode.resultImage = imgUrl || null;
  actionNode.resultImageFallback = svgPlaceholder("Image diff");
  actionNode.resultImageBadge = `${width}x${height}`;
  actionNode.imageWidth = width || 0;
  actionNode.imageHeight = height || 0;
  const p = (actionNode.outputs || []).find(o => o.id === "out-image");
  if (p) p.badge = `${width}x${height}`;
}

/**
 * Local two-photo difference: amplified abs-diff map plus mean/peak Δ.
 * B is resampled to A when sizes differ (noted); both shrink past
 * DIFF_MAX_DIM. No backend, no demo path: failures fail loud.
 */
export async function executeImageDiff(app, actionNode, btnRun) {
  const incoming = (app.edges || []).filter(e => e.to === actionNode.id);
  const edgeA = incoming.find(e => e.toPort === "in-a") || null;
  const edgeB = incoming.find(e => e.toPort === "in-b") || null;
  const urlA = edgeA ? upstreamImageUrl(app, edgeA) : null;
  const urlB = edgeB ? upstreamImageUrl(app, edgeB) : null;
  if (!urlA || !urlB) {
    failRun(app, actionNode, btnRun, "Need two images — wire one photo into A and one into B.");
    return;
  }
  appendRunLog(actionNode, "info", "Diffing A against B (local canvas)…");
  app.renderCanvas();
  try {
    const pa = await fetchImagePixels(urlA);
    const pb = await fetchImagePixels(urlB);
    const notes = [];
    let bData = pb.data, bw = pb.w, bh = pb.h;
    if (pb.w !== pa.w || pb.h !== pa.h) {
      bData = scaleImagePixels(pb.data, pb.w, pb.h, pa.w, pa.h);
      bw = pa.w; bh = pa.h;
      notes.push(`B resampled ${pb.w}x${pb.h}→${pa.w}x${pa.h}`);
    }
    let aData = pa.data, W = pa.w, H = pa.h;
    const fit = fitDims(W, H);
    if (fit.w !== W || fit.h !== H) {
      aData = scaleImagePixels(aData, W, H, fit.w, fit.h);
      bData = scaleImagePixels(bData, bw, bh, fit.w, fit.h);
      notes.push(`downscaled to ${fit.w}x${fit.h}`);
      W = fit.w; H = fit.h;
    }
    const { data, mean01, maxV } = absDiffStats(aData, bData);
    const out = document.createElement("canvas");
    out.width = W; out.height = H;
    out.getContext("2d").putImageData(new ImageData(data, W, H), 0, 0);
    actionNode.status = "Passed";
    actionNode.latency = "local";
    actionNode.runError = null;
    actionNode.runNote = "Diffed locally — no backend call.";
    appendRunLog(actionNode, "info", `Passed — mean Δ ${(mean01 * 100).toFixed(1)}% (local)`);
    fillImageDiffResultView(actionNode, {
      imgUrl: out.toDataURL("image/png"), width: W, height: H,
      meanPct: (mean01 * 100).toFixed(1), maxV, note: notes.join("; ") || null,
    });
  } catch (err) {
    actionNode.status = "Failed";
    actionNode.runError = String((err && err.message) || err || "diff failed");
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
  }
  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }
  app.renderCanvas();
}

/** Fill an exporter card's result view: GLB summary, badge, GLB ref (text-only preview). */
export function fillExportResultView(actionNode, { text, badge, demo, stats, glbUrl }) {
  actionNode.resultText = text || "";
  actionNode.resultImage = null;
  actionNode.resultImageFallback = svgPlaceholder(demo ? "Demo · export" : "export");
  actionNode.resultImageBadge = "";
  actionNode.glbUrl = glbUrl || null;
  actionNode.resultLinks = glbUrl ? [{ label: "avatar_character.glb", url: glbUrl, download: true }] : [];
  const p = (actionNode.outputs || []).find(o => o.id === "out-glb");
  if (p) p.badge = badge || "—";
}

export async function executeBodyFit(app, actionNode, btnRun) {
  const edge = sapiensInputEdge(app, actionNode);
  const url = edge ? upstreamImageUrl(app, edge) : null;
  if (!edge || !url) {
    failRun(app, actionNode, btnRun, "No image — wire a photo, crop, box or mask into Image.");
    return;
  }
  const src = (app.nodes || []).find(n => n.id === edge.from);

  let encoded;
  try {
    encoded = await encodeImageInput(url);
  } catch (err) {
    failRun(app, actionNode, btnRun, `Could not read the input image (${err?.message || err}).`);
    return;
  }

  const upstreamArtifacts = { pose_url: null, seg_url: null, normal_url: null, depth_url: null };
  const upstreamNodes = (app.edges || [])
    .filter(e => e.to === actionNode.id)
    .map(e => (app.nodes || []).find(n => n.id === e.from))
    .filter(Boolean);
  for (const n of upstreamNodes) {
    const a = n.artifacts;
    if (!a) continue;
    if (!upstreamArtifacts.pose_url && a.pose) upstreamArtifacts.pose_url = a.pose;
    if (!upstreamArtifacts.seg_url && a.seg) upstreamArtifacts.seg_url = a.seg;
    if (!upstreamArtifacts.normal_url && a.normal) upstreamArtifacts.normal_url = a.normal;
    if (!upstreamArtifacts.depth_url && a.depth) upstreamArtifacts.depth_url = a.depth;
  }
  const wired = Object.entries(upstreamArtifacts).filter(([, v]) => !!v).map(([k]) => k.replace("_url", ""));
  const isRaw = String(actionNode.templateKey || "").endsWith("-raw");
  const initEdge = (app.edges || []).find(e => e.to === actionNode.id && e.toPort === "in-init");
  const initSrc = initEdge ? (app.nodes || []).find(n => n.id === initEdge.from) : null;
  const requestBody = { chat_id: "flow-session", ...encoded, filename: src?.subtitle || "photo.jpg",
                        stage: isRaw ? "raw" : "full",
                        fit_expression: String(actionNode.param || "on") !== "off",
                        ...upstreamArtifacts, wired_maps: wired,
                        init_mesh_url: initSrc ? initSrc.meshObjUrl : null };
  const endpoint = "/api/debug/body-fit";
  markRequestSent(app, actionNode, endpoint, requestBody);
  actionNode.apiCall.local = true;
  requestBody.client_debug_id = uid("flow");
  const res = await fetchWithProgress(app, actionNode, "flow-session", endpoint, requestBody,
    BODYFIT_TIMEOUT_MS);

  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }

  // Local GPU call: hop 1 only, no upstream trace by design.
  const captureTrace = () => {
    actionNode.apiCall = {
      at: Date.now(),
      local: true,
      hop1: {
        method: "POST",
        url: endpoint,
        status: res.status,
        request: redactBlobs(requestBody),
        response: res.data ? redactBlobs(res.data) : null,
        error: res.ok ? null : (res.error || "request failed"),
      },
      hop2: null,
    };
  };

  if (!res.ok && res.status === 0) {
    if (failUnreachable(app, actionNode, res, btnRun)) { app.renderCanvas(); return; }
    console.warn("Live body-fit request failed, showing demo outputs:", res.error);
    captureTrace();
    produceOutputsForAction(app, actionNode);
    actionNode.status = "Demo (Offline)";
    actionNode.runError = res.error || "Body-fit request failed";
    actionNode.runNote = null;
    appendRunLog(actionNode, "error", `Backend unreachable (${actionNode.runError}) — demo data`);
    fillBodyFitResultView(actionNode, {
      text: "Demo body-fit (offline): the live SAM 3D Body fit appears here once the backend is reachable.",
      imgUrl: url,
      badge: "demo",
      demo: true,
      stats: null,
      meshUrl: null,
    });
    app.renderCanvas();
    return;
  }

  if (!res.ok) {
    resetActionNode(app, actionNode);
    captureTrace();
    actionNode.status = "Failed";
    const remedy = res.data && res.data.remedy ? ` — ${res.data.remedy}` : "";
    actionNode.runError = `${res.error || `HTTP ${res.status}`}${remedy}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const data = res.data || {};
  captureTrace();
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.runNote = null;
  const secs = / \(([\d.]+)s,/.exec(data.assistant_text || "");
  actionNode.latency = secs ? formatLatency(Number(secs[1])) : "local";
  const stats = data.stats || null;
  const verts = stats && typeof stats.verts === "number" ? stats.verts : null;
  appendRunLog(actionNode, "info", `Passed in ${actionNode.latency} — body fit${verts != null ? `, ${verts} verts` : ""}`);
  const bfCid = data.chat_id || "flow-session";
  const bfBase = `/api/chats/${bfCid}/file/debug/${data.debug_id}`;
  const faceErr = stats && typeof stats.face_err_px === "number" ? stats.face_err_px : null;
  actionNode.resultLinks = [
    { label: "06 keypoints", url: `${bfBase}/06_3db_keypoints.jpg` },
    { label: "07 mesh + keypoints", url: `${bfBase}/07_3db_mesh_keypoints.jpg` },
    { label: "08 head calibration", url: `${bfBase}/08_head_calibration.jpg` },
    { label: "09 expression", url: `${bfBase}/09_expression.jpg` },
    { label: "Mesh (OBJ)", url: data.mesh_url || `${bfBase}/mesh_refined.obj`, download: true },
  ];
  fillBodyFitResultView(actionNode, {
    text: data.assistant_text || "",
    imgUrl: `/api/chats/${data.chat_id || "flow-session"}/file/debug/${data.debug_id}/${data.sheet || "07_3db_mesh_keypoints.jpg"}`,
    badge: `${stats?.refine_used || "fit"} · ${verts != null ? `${verts} verts` : "mesh"}${faceErr != null ? ` · face ${faceErr}px` : ""}`,
    demo: false,
    stats,
    meshUrl: data.mesh_url || null,
    debugId: data.debug_id || null,
    chatId: data.chat_id || "flow-session",
  });
  if (stats && stats.turntable) {
    const betaN = typeof stats.beta_n === "number" ? stats.beta_n : null;
    const ttUrl = `/api/chats/${data.chat_id || "flow-session"}/file/debug/${data.debug_id}/${stats.turntable}`;
    const avatarId = spawnAvatarAsset(app, actionNode, {
      imgUrl: ttUrl,
      badge: `${verts != null ? `${verts} verts` : "mesh"} · turntable`,
      tensorShape: verts != null ? `[${verts}, 3] vertices${betaN != null ? ` + β[${betaN}]` : ""}` : "3D mesh + β",
      meshUrl: data.mesh_url || null,
      fitCamera: actionNode.fitCamera || null,
    });
    appendRunLog(actionNode, "info", `Avatar asset spawned (${stats.render_backend || "server"} turntable)`);
    app.renderCanvas();
    app.resolveAssetDims(avatarId, ttUrl, true);
    return;
  }
  if (stats && stats.turntable_error) {
    appendRunLog(actionNode, "info", `Turntable render skipped (${stats.turntable_error})`);
  }
  app.renderCanvas();
}

/** VGGT multi-view check: sheet -> cameras, fused cloud, consistency report. */
export async function executeVGGTCheck(app, actionNode, btnRun) {
  const edge = sapiensInputEdge(app, actionNode);
  const url = edge ? upstreamImageUrl(app, edge) : null;
  if (!edge || !url) {
    failRun(app, actionNode, btnRun, "No view sheet - wire a 2x2 / 2x3 / 1x3 render grid into View Sheet.");
    return;
  }
  const src = (app.nodes || []).find(n => n.id === edge.from);
  let encoded;
  try {
    encoded = await encodeImageInput(url);
  } catch (err) {
    failRun(app, actionNode, btnRun, `Could not read the sheet (${err?.message || err}).`);
    return;
  }
  const endpoint = "/api/debug/vggt-check";
  const requestBody = { chat_id: "flow-session", ...encoded, filename: src?.subtitle || "sheet.png" };
  markRequestSent(app, actionNode, endpoint, requestBody);
  actionNode.status = "Running";
  actionNode.runNote = "VGGT-Omega: cameras + depth + fused cloud";
  app.renderCanvas();
  const res = await fetchJson(endpoint, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody), timeoutMs: 600000 });
  if (btnRun) { btnRun.classList.remove("is-running"); btnRun.textContent = "✓"; }
  actionNode.apiCall = { at: Date.now(), local: true,
    hop1: { method: "POST", url: endpoint, status: res.status,
            request: redactBlobs(requestBody), response: res.data ? redactBlobs(res.data) : null,
            error: res.ok ? null : (res.error || "request failed") }, hop2: null };
  if (!res.ok) {
    actionNode.status = "Failed";
    actionNode.runError = res.error || `HTTP ${res.status}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }
  const data = res.data || {};
  const stats = data.stats || {};
  const summary = stats.summary || {};
  const secs = / \(([\d.]+)s,/.exec(data.assistant_text || "");
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.runNote = null;
  actionNode.latency = secs ? formatLatency(Number(secs[1])) : "local";
  actionNode.resultText = data.assistant_text || "";
  actionNode.resultImage = data.preview_url || null;
  actionNode.plyUrl = data.cloud_url || null;
  actionNode.resultLinks = [
    { label: "Point cloud (PLY)", url: data.cloud_url, download: true },
    { label: "Report (JSON)", url: data.report_url, download: true },
    { label: "Cloud image", url: data.preview_url },
  ].filter(l => !!l.url);
  actionNode.resultImageFallback = svgPlaceholder("VGGT cloud");
  actionNode.resultImageBadge = summary.median_mm != null
    ? `median ${summary.median_mm} mm · ${stats.n_views || "?"} views · ${stats.grid ? stats.grid[0] + "x" + stats.grid[1] : "grid"}`
    : "point cloud";
  const pin = (actionNode.outputs || []).find(o => o.id === "out-cloud");
  if (pin) pin.badge = summary.median_mm != null ? `${summary.median_mm} mm` : "PLY";
  appendRunLog(actionNode, "info", `Passed in ${actionNode.latency} - cross-view median ${summary.median_mm} mm (best ${summary.best_mm}, worst ${summary.worst_mm})`);
  app.renderCanvas();
}

/** MHR multi-view match: per-panel independent MHR fits, head-landmark alignment, mm residuals. */
export async function executeMHRMatch(app, actionNode, btnRun) {
  const edge = sapiensInputEdge(app, actionNode);
  const url = edge ? upstreamImageUrl(app, edge) : null;
  if (!edge || !url) {
    failRun(app, actionNode, btnRun, "No view sheet - wire a 2x2 / 2x3 / 2x4 grid into View Sheet.");
    return;
  }
  const src = (app.nodes || []).find(n => n.id === edge.from);
  let encoded;
  try {
    encoded = await encodeImageInput(url);
  } catch (err) {
    failRun(app, actionNode, btnRun, `Could not read the sheet (${err?.message || err}).`);
    return;
  }
  const endpoint = "/api/debug/mhr-match";
  const requestBody = { chat_id: "flow-session", ...encoded, filename: src?.subtitle || "sheet.png" };
  markRequestSent(app, actionNode, endpoint, requestBody);
  actionNode.status = "Running";
  actionNode.runNote = "MHR: one fit per panel, aligned by head landmarks";
  app.renderCanvas();
  const res = await fetchJson(endpoint, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody), timeoutMs: 900000 });
  if (btnRun) { btnRun.classList.remove("is-running"); btnRun.textContent = "✓"; }
  actionNode.apiCall = { at: Date.now(), local: true,
    hop1: { method: "POST", url: endpoint, status: res.status,
            request: redactBlobs(requestBody), response: res.data ? redactBlobs(res.data) : null,
            error: res.ok ? null : (res.error || "request failed") }, hop2: null };
  if (!res.ok) {
    actionNode.status = "Failed";
    actionNode.runError = res.error || `HTTP ${res.status}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }
  const data = res.data || {};
  const stats = data.stats || {};
  const summary = stats.summary || {};
  const secs = / \(([\d.]+)s,/.exec(data.assistant_text || "");
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.runNote = null;
  actionNode.latency = secs ? formatLatency(Number(secs[1])) : "local";
  actionNode.resultText = data.assistant_text || "";
  actionNode.resultImage = data.preview_url || null;
  actionNode.plyUrl = data.cloud_url || null;
  actionNode.resultLinks = [
    { label: "Aligned cloud (PLY)", url: data.cloud_url, download: true },
    { label: "Report (JSON)", url: data.report_url, download: true },
    { label: "Cloud image", url: data.preview_url },
  ].filter(l => !!l.url);
  actionNode.resultImageFallback = svgPlaceholder("MHR match");
  actionNode.resultImageBadge = summary.median_mm != null
    ? `median ${summary.median_mm} mm · ${stats.n_panels || "?"} MHR fits · ${stats.grid ? stats.grid[0] + "x" + stats.grid[1] : "grid"}`
    : "aligned cloud";
  const pin = (actionNode.outputs || []).find(o => o.id === "out-cloud");
  if (pin) pin.badge = summary.median_mm != null ? `${summary.median_mm} mm` : "PLY";
  appendRunLog(actionNode, "info", `Passed in ${actionNode.latency} - MHR residual median ${summary.median_mm} mm (best ${summary.best_mm}, worst ${summary.worst_mm})`);
  app.renderCanvas();
}

/** Canonical A-pose export from a live body-fit mesh. */
/** Texture bake: fitted mesh + source photo -> vertex-coloured textured GLB. */
export async function executeTexture(app, actionNode, btnRun) {
  const incoming = (app.edges || []).filter(e => e.to === actionNode.id)
    .map(e => ({ edge: e, src: (app.nodes || []).find(n => n.id === e.from) }))
    .filter(x => !!x.src);
  const meshSrc = incoming.find(x => x.src.meshObjUrl);
  const imgSrc = incoming.find(x => !x.src.meshObjUrl);
  if (!meshSrc || !imgSrc) {
    failRun(app, actionNode, btnRun, "Wire the fitted mesh and the source photo into this card.");
    return;
  }
  const imageUrl = upstreamImageUrl(app, imgSrc.edge);
  if (!imageUrl) {
    failRun(app, actionNode, btnRun, "Source photo card has no image.");
    return;
  }
  let encoded;
  try {
    encoded = await encodeImageInput(imageUrl);
  } catch (err) {
    failRun(app, actionNode, btnRun, `Could not read the photo (${err?.message || err}).`);
    return;
  }
  const endpoint = "/api/debug/texture";
  const requestBody = { chat_id: "flow-session", mesh_url: meshSrc.src.meshObjUrl,
                        ...encoded, filename: imgSrc.src.subtitle || "photo.jpg" };
  markRequestSent(app, actionNode, endpoint, requestBody);
  actionNode.status = "Running";
  app.renderCanvas();
  const res = await fetchJson(endpoint, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody), timeoutMs: 900000 });
  if (btnRun) { btnRun.classList.remove("is-running"); btnRun.textContent = "✓"; }
  actionNode.apiCall = { at: Date.now(), local: true,
    hop1: { method: "POST", url: endpoint, status: res.status,
            request: redactBlobs(requestBody), response: res.data ? redactBlobs(res.data) : null,
            error: res.ok ? null : (res.error || "request failed") }, hop2: null };
  if (!res.ok) {
    actionNode.status = "Failed";
    actionNode.runError = res.error || `HTTP ${res.status}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }
  const data = res.data || {};
  const stats = data.stats || {};
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.latency = stats.seconds ? `${stats.seconds}s` : "local";
  actionNode.resultText = data.assistant_text || "";
  actionNode.resultImage = data.sheet_url || data.preview_url || null;
  actionNode.resultImageFallback = svgPlaceholder("Texture bake");
  actionNode.resultImageBadge = `textured · ${Math.round((stats.projected_coverage || 0) * 100)}% projected`;
  actionNode.meshObjUrl = data.glb_url || null;
  actionNode.resultLinks = [
    { label: "Textured GLB", url: data.glb_url, download: true },
    { label: "UV texture (1024)", url: data.texture_url, download: true },
    { label: "Turnaround", url: data.sheet_url },
    { label: "Preview", url: data.preview_url },
  ].filter(l => !!l.url);
  appendRunLog(actionNode, "info", `Passed in ${actionNode.latency} - textured GLB with vertex colours + 1024 UV texture`);
  app.renderCanvas();
}

export async function executeAPose(app, actionNode, btnRun) {
  const edge = sapiensInputEdge(app, actionNode);
  const url = edge ? upstreamMeshUrl(app, edge) : null;
  if (!edge || !url) {
    failRun(app, actionNode, btnRun, "No mesh - wire the body-fit Mesh Output into Fitted Mesh (run the fit live first).");
    return;
  }
  const endpoint = "/api/debug/apose";
  const requestBody = { chat_id: "flow-session", mesh_url: url };
  markRequestSent(app, actionNode, endpoint, requestBody);
  actionNode.status = "Running";
  app.renderCanvas();
  const res = await fetchJson(endpoint, { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody), timeoutMs: 600000 });
  if (btnRun) { btnRun.classList.remove("is-running"); btnRun.textContent = "✓"; }
  actionNode.apiCall = { at: Date.now(), local: true,
    hop1: { method: "POST", url: endpoint, status: res.status,
            request: redactBlobs(requestBody), response: res.data ? redactBlobs(res.data) : null,
            error: res.ok ? null : (res.error || "request failed") }, hop2: null };
  if (!res.ok) {
    actionNode.status = "Failed";
    actionNode.runError = res.error || `HTTP ${res.status}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }
  const data = res.data || {};
  const stats = data.stats || {};
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.latency = "local";
  actionNode.resultText = data.assistant_text || "";
  actionNode.resultImage = data.sheet ? `/api/chats/${data.chat_id || "flow-session"}/file/debug/${data.debug_id}/${data.sheet}` : null;
  actionNode.sheetWideUrl = data.sheet_wide ? `/api/chats/${data.chat_id || "flow-session"}/file/debug/${data.debug_id}/${data.sheet_wide}` : null;
  actionNode.resultImageFallback = svgPlaceholder("A-pose");
  actionNode.resultImageBadge = `A-pose · ${stats.verts || "?"} verts · 6 views`;
  actionNode.meshObjUrl = data.obj_url || null;
  actionNode.plyUrl = data.mesh_url || null;
  const apBase = `/api/chats/${data.chat_id || "flow-session"}/file/debug/${data.debug_id}`;
  actionNode.resultLinks = [
    { label: "A-pose PLY", url: data.mesh_url, download: true },
    { label: "A-pose OBJ", url: data.obj_url, download: true },
    { label: "Turnaround 2x3", url: `${apBase}/turnaround_apose_2x3.jpg` },
    { label: "Turnaround 1x6", url: `${apBase}/turnaround_apose.jpg` },
    { label: "Pose JSON", url: `${apBase}/pose_canonical.json`, download: true },
  ].filter(l => !!l.url);
  const mPin = (actionNode.outputs || []).find(o => o.id === "out-mesh");
  if (mPin) mPin.badge = "A-pose PLY";
  const sPin = (actionNode.outputs || []).find(o => o.id === "out-sheet");
  if (sPin) sPin.badge = "6 views";
  appendRunLog(actionNode, "info", `A-pose export ready (${stats.verts || "?"} verts)`);
  app.renderCanvas();
}

export async function executeExport(app, actionNode, btnRun) {
  const edge = sapiensInputEdge(app, actionNode);
  const meshUrl = edge ? upstreamMeshUrl(app, edge) : null;
  if (!edge || !meshUrl) {
    failRun(app, actionNode, btnRun, "No mesh — wire a body-fit mesh into Mesh (run the fit live first).");
    return;
  }
  let encoded;
  try {
    encoded = await encodeMeshInput(meshUrl);
  } catch (err) {
    failRun(app, actionNode, btnRun, `Could not read the input mesh (${err?.message || err}).`);
    return;
  }

  const filename = meshUrl.split("?")[0].split("/").pop() || "mesh_refined.obj";
  const requestBody = { chat_id: "flow-session", ...encoded, filename };
  const endpoint = "/api/debug/export";
  markRequestSent(app, actionNode, endpoint, requestBody);
  actionNode.apiCall.local = true;
  const res = await fetchJson(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
    timeoutMs: EXPORT_TIMEOUT_MS,
  });

  if (btnRun) {
    btnRun.classList.remove("is-running");
    btnRun.textContent = "✓";
  }

  // Local CPU call: hop 1 only, no upstream trace by design.
  const captureTrace = () => {
    actionNode.apiCall = {
      at: Date.now(),
      local: true,
      hop1: {
        method: "POST",
        url: endpoint,
        status: res.status,
        request: redactBlobs(requestBody),
        response: res.data ? redactBlobs(res.data) : null,
        error: res.ok ? null : (res.error || "request failed"),
      },
      hop2: null,
    };
  };

  if (!res.ok && res.status === 0) {
    if (failUnreachable(app, actionNode, res, btnRun)) { app.renderCanvas(); return; }
    console.warn("Live export request failed, showing demo outputs:", res.error);
    captureTrace();
    produceOutputsForAction(app, actionNode);
    actionNode.status = "Demo (Offline)";
    actionNode.runError = res.error || "Export request failed";
    actionNode.runNote = null;
    appendRunLog(actionNode, "error", `Backend unreachable (${actionNode.runError}) — demo data`);
    fillExportResultView(actionNode, {
      text: "Demo export (offline): the live avatar_character.glb packages here once the backend is reachable.",
      badge: "demo",
      demo: true,
      stats: null,
      glbUrl: null,
    });
    app.renderCanvas();
    return;
  }

  if (!res.ok) {
    resetActionNode(app, actionNode);
    captureTrace();
    actionNode.status = "Failed";
    const remedy = res.data && res.data.remedy ? ` — ${res.data.remedy}` : "";
    actionNode.runError = `${res.error || `HTTP ${res.status}`}${remedy}`;
    appendRunLog(actionNode, "error", `Failed: ${actionNode.runError}`);
    app.renderCanvas();
    return;
  }

  const data = res.data || {};
  captureTrace();
  actionNode.status = "Passed";
  actionNode.runError = null;
  actionNode.runNote = null;
  actionNode.latency = "local";
  const stats = data.stats || null;
  const mb = stats && typeof stats.glb_bytes === "number" ? stats.glb_bytes / 1048576 : null;
  appendRunLog(actionNode, "info", `Passed — avatar_character.glb${mb != null ? `, ${mb.toFixed(1)} MB` : ""}`);
  fillExportResultView(actionNode, {
    text: data.assistant_text || "",
    badge: mb != null ? `${mb.toFixed(1)} MB` : "GLB",
    demo: false,
    stats,
    glbUrl: data.glb_url || null,
  });
  app.renderCanvas();
}

/**
 * Build one item-card spec per list entry. Pure (no DOM): boxes/masks become
 * image items (crop / mask overlay, placeholder when URLs are absent, e.g.
 * offline demo records); confidences/labels become scalar data items.
 */
export function explodeListItems(resultData, kind) {
  const itemType = itemTypeForList(kind);
  if (!resultData || !itemType || !Array.isArray(resultData[kind])) return [];
  const list = resultData[kind];
  const records = Array.isArray(resultData.masks) ? resultData.masks : [];
  const labelFor = (i) => (resultData.labels && resultData.labels[i]) || `${resultData.concept || "item"} ${i + 1}`;

  return list.map((entry, i) => {
    const n = i + 1;
    const rec = records[i] || {};
    const conf = typeof rec.confidence === "number" ? rec.confidence : resultData.confidences?.[i];
    const confPct = typeof conf === "number" ? (conf * 100).toFixed(1) + "%" : "—";
    // Backend provenance for this instance (debug run + mask index).
    const ref = { debug_id: resultData.debugId || null, mask_id: (rec && rec.id != null) ? rec.id : i };

    if (kind === "boxes") {
      const bx = Array.isArray(entry) ? entry : [0, 0, 0, 0];
      const bw = Math.round(bx[2] - bx[0]);
      const bh = Math.round(bx[3] - bx[1]);
      const coords = `[${bx.map(v => Math.round(v)).join(", ")}]`;
      return {
        title: `Box ${n}`, subtitle: coords, assetKind: "image", ref,
        imgUrl: rec.crop_url || rec.file_url || svgPlaceholder(`Box ${n} ${coords}`),
        imgFallback: svgPlaceholder(`Box ${n}`),
        imgBadge: `${bw}×${bh}px · ${confPct}`,
        imageWidth: bw + 24, imageHeight: bh + 24,
        box: bx, confidence: conf, pixels: rec.pixels,
        outputType: "box", outputName: `Box ${n}`, outputBadge: `${bw}×${bh}`,
        tensorShape: "[4] xyxy", format: "Bounding box + crop ref",
        badgeVariant: "meta-light", iconVariant: "meta",
      };
    }

    if (kind === "masks") {
      const bx = Array.isArray(rec.box) ? rec.box : [0, 0, 0, 0];
      return {
        title: `Mask ${n}`, subtitle: `${rec.pixels ?? "—"} px · M${rec.id ?? i}`,
        assetKind: "image", ref,
        imgUrl: rec.file_url || rec.crop_url || svgPlaceholder(`Mask ${n}`),
        imgFallback: svgPlaceholder(`Mask ${n}`),
        imgBadge: `M${rec.id ?? i} · ${confPct}`,
        imageWidth: resultData.imageWidth || 0, imageHeight: resultData.imageHeight || 0,
        box: bx, confidence: conf, pixels: rec.pixels,
        outputType: "mask", outputName: `Mask ${n}`, outputBadge: confPct,
        tensorShape: "[H, W] bool", format: "Binary mask + overlay",
        badgeVariant: "success-light", iconVariant: "green",
      };
    }

    if (kind === "confidences") {
      return {
        title: `Confidence ${n}`, subtitle: labelFor(i), assetKind: "data", ref,
        dataValue: typeof entry === "number" ? entry : conf, dataKind: "number",
        outputType: "number", outputName: `Confidence ${n}`,
        outputBadge: typeof entry === "number" ? String(entry) : "—",
        tensorShape: "scalar float", format: "Confidence score",
        badgeVariant: "primary-light", iconVariant: "soft",
      };
    }

    // labels
    return {
      title: `Label ${n}`, subtitle: `item ${n} of ${list.length}`, assetKind: "data", ref,
      dataValue: String(entry ?? "—"), dataKind: "text",
      outputType: "text", outputName: `Label ${n}`, outputBadge: String(entry ?? "—"),
      tensorShape: "string", format: "Instance label",
      badgeVariant: "info-light", iconVariant: "outline",
    };
  });
}

const ITEM_PIN_COLORS = { box: "#0866FF", mask: "#10b981", number: "#f59e0b", text: "#ec4899" };

/**
 * Build item-pin defs for a split list pin. Pure (no DOM): image items carry
 * a thumbnail URL, scalar items a thumbnail value.
 */
export function buildSplitPins(resultData, kind, listPortId) {
  return explodeListItems(resultData, kind).map((s, i) => ({
    id: `split-${listPortId}-${i}`,
    name: s.outputName,
    type: s.outputType,
    color: ITEM_PIN_COLORS[s.outputType] || "#94a3b8",
    badge: s.outputBadge,
    thumb: s.assetKind === "image" ? s.imgUrl : null,
    thumbValue: s.assetKind === "data" ? String(s.dataValue ?? "—") : null,
    ref: s.ref || null,
  }));
}

/** UE-style split: expand a list pin into item pins with thumbnails. */
export function splitListPin(app, nodeId, listPortId) {
  const node = app.nodes.find(n => n.id === nodeId);
  const pin = node && (node.outputs || []).find(p => p.id === listPortId);
  if (!node || !pin || !isListType(pin.type)) return false;
  const data = node.resultData;
  if (!data || !Array.isArray(data[pin.type]) || data[pin.type].length === 0) return false;
  node.splitPins = node.splitPins || {};
  if (node.splitPins[listPortId]) return false;
  const itemPins = buildSplitPins(data, pin.type, listPortId);
  if (itemPins.length === 0) return false;
  const at = node.outputs.findIndex(p => p.id === listPortId);
  node.outputs.splice(at + 1, 0, ...itemPins);
  node.splitPins[listPortId] = itemPins.map(p => p.id);
  appendRunLog(node, "info", `Exploded ${pin.name} → ${itemPins.length} pins`);
  app.renderCanvas();
  return true;
}

/**
 * Collapse a split list pin. Item pins and their wires are removed;
 * downstream cards stay in place, unwired.
 */
export function collapseListPin(app, nodeId, listPortId, render = true) {
  const node = app.nodes.find(n => n.id === nodeId);
  const ids = node && node.splitPins && node.splitPins[listPortId];
  if (!node || !ids) return false;
  const gone = new Set(ids);
  const before = app.edges.length;
  const downstreamIds = app.edges
    .filter(e => e.from === nodeId && gone.has(e.fromPort))
    .map(e => e.to);
  app.edges = app.edges.filter(e => !(e.from === nodeId && gone.has(e.fromPort)));
  const dropped = before - app.edges.length;
  node.outputs = (node.outputs || []).filter(o => !gone.has(o.id));
  delete node.splitPins[listPortId];
  if (Object.keys(node.splitPins).length === 0) delete node.splitPins;
  const pin = (node.outputs || []).find(p => p.id === listPortId);
  appendRunLog(node, "info", `Collapsed ${pin ? pin.name : listPortId} (${dropped} wires removed)`);
  downstreamIds.forEach(id => {
    const n = app.nodes.find(x => x.id === id);
    if (!n) return;
    syncViewerStatus(app, n);
    // Re-probe even when the caller skips rendering (re-run reset path):
    // the probe patches the live DOM itself once the image loads.
    if (n.nodeType === "asset") app.resolveAssetDims(id, resolveAssetImageUrl(app, n), false);
  });
  if (render) app.renderCanvas();
  return true;
}

export function collapseAllSplits(app, node, render = true) {
  if (!node || !node.splitPins) return;
  Object.keys(node.splitPins).forEach(pid => collapseListPin(app, node.id, pid, false));
  if (render) app.renderCanvas();
}

export function produceOutputsForAction(app, actionNode) {
  const runId = app.activeRunId;
  const modelType = actionNode.templateKey || actionNode.id;

  if (modelType.startsWith("action-meta-sam")) {
    const defaultMasks = [
      { id: 0, object_id: "0", box: [80, 95, 405, 1000], confidence: 0.987, pixels: 284120 },
      { id: 1, object_id: "1", box: [390, 255, 845, 1000], confidence: 0.979, pixels: 219430 }
    ];

    collapseAllSplits(app, actionNode, false);
    const resultData = deriveResultData(defaultMasks, actionNode.param, 1000, 1000);
    fillSamOutputs(actionNode, resultData);
    fillSamResultView(actionNode, resultData, {
      imgUrl: `/api/runs/${runId}/artifact/boxes`,
      demo: true,
    });

    setTimeout(() => app.fitView(), 60);
  } else if (modelType.startsWith("action-sapiens-pose")) {
    const kptsId = uid("asset-pose");
    const kptsNode = {
      id: kptsId,
      nodeType: "asset",
      assetKind: "pointcloud",
      title: "Asset: 308-Kpts Pointcloud",
      subtitle: "pose_308.json",
      category: "Pointcloud Asset",
      badgeVariant: "cyan-light",
      iconVariant: "cyan",
      iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"/><circle cx="19" cy="6" r="2"/><circle cx="5" cy="18" r="2"/><path d="M12 9v-3"/><path d="m14 14 3 3"/></svg>`,
      imgUrl: `/api/runs/${runId}/artifact/pose`,
      imgFallback: svgPlaceholder("Pose demo (offline)"),
      imgBadge: "308 keypoints · median 0.968",
      description: "Dense 308 landmark coordinates (x, y, confidence) across body, face, hands, and feet.",
      latency: "Produced",
      status: "RAM Buffer",
      demo: true,
      tensorShape: "[308, 3] float32",
      format: "Keypoint coordinate array",
      inputs: [
        { id: "in-crop", name: "Person Crop", type: "crop", color: "#f59e0b", badge: "Crop" }
      ],
      outputs: [
        { id: "out-kpts", name: "308 Keypoints Array", type: "pointcloud", color: "#06b6d4", badge: "308 Kpts" }
      ],
      x: actionNode.x + 480,
      y: actionNode.y
    };
    app.nodes.push(kptsNode);
    app.edges.push({ id: `e-${actionNode.id}-${kptsId}`, from: actionNode.id, to: kptsId, fromPort: "out-kpts", toPort: "in-crop", label: "produces 308 kpts" });

  } else if (modelType.startsWith("action-sapiens-normal")) {
    const normalId = uid("asset-normal");
    const normalNode = {
      id: normalId,
      nodeType: "asset",
      assetKind: "normalmap",
      title: "Asset: Surface Normal Map",
      subtitle: "surface_normals.png",
      category: "Normalmap Asset",
      badgeVariant: "meta-light",
      iconVariant: "meta",
      iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>`,
      imgUrl: `/api/runs/${runId}/artifact/normal`,
      imgFallback: svgPlaceholder("Normals demo (offline)"),
      imgBadge: "Normals 3D · Unit Vectors",
      description: "Pixel-aligned surface orientation vectors representing skin and clothing curvatures.",
      latency: "Produced",
      status: "Artifact",
      demo: true,
      tensorShape: "[H, W, 3] float32",
      format: "RGB Normal Map",
      inputs: [
        { id: "in-crop", name: "Person Crop", type: "crop", color: "#f59e0b", badge: "Crop" }
      ],
      outputs: [
        { id: "out-normals", name: "Surface Normal Map", type: "normalmap", badge: "RGB Normals" }
      ],
      x: actionNode.x + 480,
      y: actionNode.y
    };
    app.nodes.push(normalNode);
    app.edges.push({ id: `e-${actionNode.id}-${normalId}`, from: actionNode.id, to: normalId, fromPort: "out-normals", toPort: "in-crop", label: "produces normal map" });

  } else if (modelType.startsWith("action-sapiens-depth")) {
    const depthId = uid("asset-depth");
    const depthNode = {
      id: depthId,
      nodeType: "asset",
      assetKind: "depthmap",
      title: "Asset: Metric Depth Map",
      subtitle: "metric_depth.npy",
      category: "Depthmap Asset",
      badgeVariant: "cyan-light",
      iconVariant: "cyan",
      iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>`,
      imgUrl: `/api/runs/${runId}/artifact/depth`,
      imgFallback: svgPlaceholder("Depth demo (offline)"),
      imgBadge: "Depth [0.34m, 0.52m]",
      description: "Euclidean metric distance array in meters for each pixel.",
      latency: "Produced",
      status: "Memory Array",
      demo: true,
      tensorShape: "[H, W] float32 meters",
      format: "Metric Float Array",
      inputs: [
        { id: "in-crop", name: "Person Crop", type: "crop", color: "#f59e0b", badge: "Crop" }
      ],
      outputs: [
        { id: "out-depth", name: "Metric Depth Map", type: "depthmap", color: "#06b6d4", badge: "Float Array" }
      ],
      x: actionNode.x + 480,
      y: actionNode.y
    };
    app.nodes.push(depthNode);
    app.edges.push({ id: `e-${actionNode.id}-${depthId}`, from: actionNode.id, to: depthId, fromPort: "out-depth", toPort: "in-crop", label: "produces depth map" });

  } else if (modelType.startsWith("action-sapiens-seg")) {
    const segId = uid("asset-seg");
    const segNode = {
      id: segId,
      nodeType: "asset",
      assetKind: "image",
      title: "Asset: 28-Class Part Seg",
      subtitle: "part_segmentation.png",
      category: "Mask Asset",
      badgeVariant: "success-light",
      iconVariant: "green",
      iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><path d="M8 12a4 4 0 0 1 8 0"/></svg>`,
      imgUrl: `/api/runs/${runId}/artifact/boxes`,
      imgFallback: svgPlaceholder("Seg demo (offline)"),
      imgBadge: "28 classes · Skin & Clothing",
      description: "Categorical part segmentation separating facial skin, hair, and clothing.",
      latency: "Produced",
      status: "Artifact",
      demo: true,
      tensorShape: "[H, W] uint8 (0..27)",
      format: "Indexed PNG",
      inputs: [
        { id: "in-crop", name: "Person Crop", type: "crop", color: "#f59e0b", badge: "Crop" }
      ],
      outputs: [
        { id: "out-parts", name: "28 Part Masks", type: "image", color: "#10b981", badge: "28 Classes" }
      ],
      x: actionNode.x + 480,
      y: actionNode.y
    };
    app.nodes.push(segNode);
    app.edges.push({ id: `e-${actionNode.id}-${segId}`, from: actionNode.id, to: segId, fromPort: "out-parts", toPort: "in-crop", label: "produces part seg" });

  } else if (modelType.startsWith("action-body-fit")) {
    const meshId = uid("asset-mesh");
    const meshNode = {
      id: meshId,
      nodeType: "asset",
      assetKind: "mesh",
      title: "Asset: Parametric 3D Avatar",
      subtitle: "avatar_mesh.npy",
      category: "3D Mesh Asset",
      badgeVariant: "primary-light",
      iconVariant: "soft",
      iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>`,
      imgUrl: `/api/runs/${runId}/artifact/mesh`,
      imgFallback: svgPlaceholder("Mesh demo (offline)"),
      imgBadge: "Mesh Fit · ||β||₂ 5.981 · focal 1414px",
      description: "Fitted 3D humanoid mesh with vertices, bone weights, UV coordinates, and shape parameters.",
      latency: "Produced",
      status: "GPU VBO Buffer",
      demo: true,
      tensorShape: "[17821, 3] vertices + β[10]",
      format: "Wavefront OBJ / NumPy",
      inputs: [
        { id: "in-features", name: "Pose & Geometry", type: "geometry", color: "#06b6d4", badge: "Fusion" }
      ],
      outputs: [
        { id: "out-mesh", name: "Parametric Mesh", type: "mesh", color: "#f59e0b", badge: "17821 Verts" }
      ],
      x: actionNode.x + 480,
      y: actionNode.y
    };
    app.nodes.push(meshNode);
    app.edges.push({ id: `e-${actionNode.id}-${meshId}`, from: actionNode.id, to: meshId, fromPort: "out-mesh", toPort: "in-features", label: "produces avatar mesh" });

  } else if (modelType.startsWith("action-export")) {
    const specId = uid("asset-spec");
    const specNode = {
      id: specId,
      nodeType: "asset",
      assetKind: "mesh",
      title: "Asset: avatar_character.glb",
      subtitle: "stats.json & GLB Asset",
      category: "Binary 3D Spec",
      badgeVariant: "success-light",
      iconVariant: "green",
      iconSvg: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>`,
      imgUrl: `/api/runs/${runId}/artifact/mesh`,
      imgFallback: svgPlaceholder("Export demo (offline)"),
      imgBadge: "avatar_character.glb · 28.9MB",
      description: "Governance-compliant character asset package ready for articulation.",
      latency: "Final Spec",
      status: "Disk Artifact",
      demo: true,
      tensorShape: "3D Asset (GLB + JSON)",
      format: "glTF Binary 2.0",
      inputs: [
        { id: "in-mesh", name: "Parametric Mesh", type: "mesh", color: "#3b82f6", badge: "Mesh" }
      ],
      outputs: [
        { id: "out-spec", name: "GLB Package", type: "mesh", color: "#10b981", badge: "28.9MB" }
      ],
      x: actionNode.x + 480,
      y: actionNode.y
    };
    app.nodes.push(specNode);
    app.edges.push({ id: `e-${actionNode.id}-${specId}`, from: actionNode.id, to: specId, fromPort: "out-glb", toPort: "in-mesh", label: "produces character spec" });
  }
}

export function deleteCard(app, nodeId) {
  disposeMeshViewer(app, nodeId);
  const downstreamIds = app.edges.filter(e => e.from === nodeId).map(e => e.to);
  app.nodes = app.nodes.filter(n => n.id !== nodeId);
  app.edges = app.edges.filter(e => e.from !== nodeId && e.to !== nodeId);
  delete app.customPositions[nodeId];

  if (app.selectedNodeId === nodeId) {
    app.selectedNodeId = app.nodes.length > 0 ? app.nodes[0].id : null;
  }

  // Viewers wired from the deleted card fall back to their own image.
  downstreamIds.forEach(id => {
    const n = app.nodes.find(x => x.id === id);
    if (!n || n.nodeType !== "asset") return;
    syncViewerStatus(app, n);
  });

  app.closePopover();
  app.renderCanvas();
  downstreamIds.forEach(id => {
    const n = app.nodes.find(x => x.id === id);
    if (n && n.nodeType === "asset") app.resolveAssetDims(id, resolveAssetImageUrl(app, n), false);
  });
}

/**
 * Pure: ready actions in upstream-first wire order. Depth over the full edge
 * graph (assets pass through), stable for ties, cycle-safe — traversal order
 * for Step Forward and Auto Traverse.
 */
export function orderReadyActions(app) {
  const nodes = app.nodes || [];
  const edges = app.edges || [];
  const ready = nodes.filter(n => n.nodeType === "action" && n.status === "Ready to Run");
  if (ready.length === 0) return [];
  // Longest-path depth: relax every edge until stable so a card with several
  // parents runs after ALL of them (a BFS visit-once under-counts that case:
  // Muse gets photo at depth 1 and A-pose at depth 2, and VGGT used to slip
  // in before Muse because the early visit pinned Muse at depth 1).
  const depth = new Map(nodes.map(n => [n.id, 0]));
  for (let pass = 0; pass <= nodes.length; pass++) {
    let changed = false;
    for (const e of edges) {
      if (!depth.has(e.from) || !depth.has(e.to)) continue;
      const d = depth.get(e.from) + 1;
      if (depth.get(e.to) < d) {
        depth.set(e.to, d);
        changed = true;
      }
    }
    if (!changed) break;
  }
  const order = new Map(nodes.map((n, i) => [n.id, i]));
  return ready.slice().sort((a, b) =>
    (depth.get(a.id) - depth.get(b.id)) || (order.get(a.id) - order.get(b.id)));
}

export function runNextAction(app) {
  const next = orderReadyActions(app)[0];
  if (next) {
    runCard(app, next.id);
  }
}

export async function runAllSteps(app) {
  // Sequential, upstream-first: each card finishes before the next starts, so
  // Sapiens stages never collide on the single-GPU ModelSlot and downstream
  // cards run on fresh upstream outputs. Stops at the first failure.
  for (;;) {
    const next = orderReadyActions(app)[0];
    if (!next) return;
    await runCard(app, next.id);
    const done = (app.nodes || []).find(n => n.id === next.id);
    if (!done || done.status === "Failed") return;
  }
}
