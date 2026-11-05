/**
 * Shared flow-viewer utilities (DOM-free except probeImageSize).
 * Single home for HTML escaping, unique node ids, offline placeholders,
 * timeout fetch, and image dimension probing.
 */

/** Escape a value for safe interpolation into HTML or an attribute. */
export function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

let uidCounter = 0;

/** Unique id for graph nodes/edges. Monotonic per session, no Date.now collisions. */
export function uid(prefix) {
  uidCounter += 1;
  const rand = Math.random().toString(36).slice(2, 7);
  return `${prefix}-${Date.now().toString(36)}-${uidCounter.toString(36)}${rand}`;
}

/** Offline-safe SVG placeholder as a data URI (no external faces, no network). */
export function svgPlaceholder(label = "No preview", w = 600, h = 400) {
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">` +
    `<rect width="100%" height="100%" fill="#161b22"/>` +
    `<rect x="14" y="14" width="${w - 28}" height="${h - 28}" rx="10" fill="none" stroke="#30363d" stroke-dasharray="8 6"/>` +
    `<text x="50%" y="50%" fill="#8b949e" font-family="monospace" font-size="22" text-anchor="middle">${escapeHtml(label)}</text>` +
    `</svg>`;
  // encodeURIComponent leaves ' untouched; percent-encode it so the URI is
  // safe inside single-quoted onerror handlers as well as src attributes.
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg).replace(/'/g, "%27")}`;
}

/**
 * fetch() with a timeout that resolves (never rejects) to
 * { ok, status, data } — data is parsed JSON or null, error describes failures.
 */
export async function fetchJson(url, { method = "GET", headers, body, timeoutMs = 15000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(url, {
      method,
      headers,
      body,
      signal: ctrl.signal,
    });
    const data = await resp.json().catch(() => null);
    if (!resp.ok) {
      const detail = (data && (data.error || data.remedy)) || `HTTP ${resp.status}`;
      return { ok: false, status: resp.status, data, error: String(detail) };
    }
    return { ok: true, status: resp.status, data, error: null };
  } catch (err) {
    const timedOut = err && err.name === "AbortError";
    return {
      ok: false,
      status: 0,
      data: null,
      error: timedOut ? `Request timed out after ${Math.round(timeoutMs / 1000)}s` : String((err && err.message) || err),
    };
  } finally {
    clearTimeout(timer);
  }
}

/** Resolve an image URL to { w, h } or null (never rejects). Browser-only. */
export function probeImageSize(url) {
  return new Promise((resolve) => {
    if (!url) {
      resolve(null);
      return;
    }
    const done = (v) => resolve(v);
    let settled = false;
    const finish = (v) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        done(v);
      }
    };
    const timer = setTimeout(() => finish(null), 10000);
    try {
      const img = new Image();
      img.onload = () => finish({ w: img.naturalWidth || 0, h: img.naturalHeight || 0 });
      img.onerror = () => finish(null);
      img.src = url;
    } catch {
      finish(null);
    }
  });
}

/** Format backend latency (number seconds or preformatted string) for display. */
export function formatLatency(latency) {
  if (latency === null || latency === undefined || latency === "") return "—";
  if (typeof latency === "number") return `${latency}s`;
  return String(latency);
}

/**
 * Single-pin sides on asset cards render as one unlabeled edge port —
 * there is nothing to choose, so labels are noise. Action cards always
 * keep labeled pins; multi-pin asset sides (e.g. N person crops) keep them too.
 */
export function minimizePinSide(nodeType, pins) {
  return nodeType === "asset" && Array.isArray(pins) && pins.length === 1;
}

/** Inline aspect-ratio style for a photo-dominant card, or "" when unknown. */
export function aspectRatioStyle(imageWidth, imageHeight) {
  const w = Number(imageWidth);
  const h = Number(imageHeight);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return "";
  return `aspect-ratio:${w} / ${h};`;
}

/**
 * Pure: card width for a wide image, or null to keep the default 320px.
 * Wide stills grow the card proportionally (a 2:1 sheet reads at 640px,
 * not as a 160px sliver) up to a 720px cap; square and tall images
 * never shrink it.
 */
export const CARD_BASE_WIDTH_PX = 320;
export const CARD_MAX_WIDTH_PX = 720;
export function cardWidthForImage(imageWidth, imageHeight) {
  const w = Number(imageWidth);
  const h = Number(imageHeight);
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return null;
  if (w / h <= 1) return null;
  return Math.min(CARD_MAX_WIDTH_PX, Math.round(CARD_BASE_WIDTH_PX * (w / h)));
}

/** Typed list kinds produced by detectors; Explode accepts any of these. */
export const LIST_TYPES = ["boxes", "masks", "confidences", "labels"];

/** Item type produced when a list kind is exploded. */
export const ITEM_TYPE_FOR_LIST = {
  boxes: "box",
  masks: "mask",
  confidences: "number",
  labels: "text",
};

export function isListType(t) {
  return LIST_TYPES.includes(t);
}

export function itemTypeForList(kind) {
  return ITEM_TYPE_FOR_LIST[kind] || null;
}

/** "2 boxes", "1 mask" — kinds arrive plural, singularized for 1. */
export function listCountLabel(kind, count) {
  const n = Number(count) || 0;
  const singular = { boxes: "box", masks: "mask", confidences: "confidence", labels: "label" };
  const plural = { boxes: "boxes", masks: "masks", confidences: "confidences", labels: "labels" };
  const word = n === 1 ? (singular[kind] || "item") : (plural[kind] || "items");
  return `${n} ${word}`;
}

/**
 * Deep-clone a request/response payload with image blobs redacted, mirroring
 * the backend's `_redact_post` (which keeps full bodies minus inline base64).
 * Everything else — including mask encodings — stays verbatim: full return.
 */
export function redactBlobs(value) {
  if (Array.isArray(value)) return value.map(redactBlobs);
  if (value && typeof value === "object") {
    const out = {};
    for (const [k, v] of Object.entries(value)) {
      if ((k === "image_b64" || k.endsWith("_b64")) && typeof v === "string") {
        out[k] = `<${v.length} base64 chars redacted>`;
      } else {
        out[k] = redactBlobs(v);
      }
    }
    return out;
  }
  if (typeof value === "string" && value.length > 200 && value.startsWith("data:")) {
    return `<data URL, ${value.length} chars redacted>`;
  }
  return value;
}

const RUN_LOG_LIMIT = 30;

/** Append a timestamped entry to a card's run log (shown in its Logs tab). */
export function appendRunLog(node, level, msg) {
  if (!node) return;
  node.runLog = node.runLog || [];
  node.runLog.push({ t: Date.now(), level: level === "error" ? "error" : "info", msg: String(msg ?? "") });
  if (node.runLog.length > RUN_LOG_LIMIT) {
    node.runLog.splice(0, node.runLog.length - RUN_LOG_LIMIT);
  }
}

export function formatRunLogTime(t) {
  const d = new Date(t);
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

/**
 * Pin type registry (ComfyUI-strict): every wire type declared once with its
 * canonical color and the exact list of types that may plug into it. A mask
 * is not an image; a box item carries its crop, so `crop` accepts `box`;
 * point clouds and geometry are interchangeable. Anything unlisted is
 * rejected at drop time with a log entry — never silently re-routed.
 */
export const PIN_TYPES = {
  image:       { label: "Image",       color: "#10b981", accepts: ["image", "normalmap", "depthmap"] },
  crop:        { label: "Crop",        color: "#f59e0b", accepts: ["crop", "box"] },
  box:         { label: "Box",         color: "#0866FF", accepts: ["box"] },
  boxes:       { label: "Boxes",       color: "#0866FF", accepts: ["boxes"] },
  mask:        { label: "Mask",        color: "#10b981", accepts: ["mask"] },
  masks:       { label: "Masks",       color: "#10b981", accepts: ["masks"] },
  confidences: { label: "Confidences", color: "#f59e0b", accepts: ["confidences"] },
  labels:      { label: "Labels",      color: "#ec4899", accepts: ["labels"] },
  number:      { label: "Number",      color: "#f59e0b", accepts: ["number"] },
  text:        { label: "Text",        color: "#ec4899", accepts: ["text"] },
  pointcloud:  { label: "Point Cloud", color: "#06b6d4", accepts: ["pointcloud", "geometry"] },
  geometry:    { label: "Geometry",    color: "#06b6d4", accepts: ["geometry", "pointcloud"] },
  normalmap:   { label: "Normal Map",  color: "#0866FF", accepts: ["normalmap"] },
  depthmap:    { label: "Depth Map",   color: "#06b6d4", accepts: ["depthmap"] },
  mesh:        { label: "Mesh",        color: "#3b82f6", accepts: ["mesh"] },
  vector:      { label: "Vector",      color: "#f59e0b", accepts: ["vector"] },
  list:        { label: "List",        color: "#94a3b8", accepts: ["boxes", "masks", "confidences", "labels"] },
};

export function pinTypeDef(type) {
  return PIN_TYPES[type] || null;
}

export function pinTypeColor(type, fallback = "#94a3b8") {
  const def = PIN_TYPES[type];
  return def ? def.color : fallback;
}

/** Strict check: may fromType plug into toType? Unknown types match only themselves. */
export function canConnect(fromType, toType) {
  if (fromType === toType) return true;
  const def = PIN_TYPES[toType];
  if (!def) return false;
  return def.accepts.includes(fromType);
}

/**
 * Pin-aware wire check: the type registry first, then the target pin's own
 * `accepts` list when it declares one. Viewer pins take image/box/mask/crop
 * this way while every other pin stays registry-strict.
 */
export function canWirePorts(fromType, toPin) {
  if (!toPin) return false;
  if (canConnect(fromType, toPin.type)) return true;
  return Array.isArray(toPin.accepts) && toPin.accepts.includes(fromType);
}

/**
 * Effective display image for an asset card: a wired input wins over the
 * card's own photo, so viewer cards (and the source photo itself) show
 * whatever is plugged in. Recurses through viewer chains with a cycle guard;
 * falls back to the card's own imgUrl when unwired or unresolvable.
 */
export function resolveAssetImageUrl(app, node, seen = null) {
  if (!node || node.nodeType !== "asset") return node?.imgUrl || null;
  const visited = seen || new Set();
  if (visited.has(node.id)) return node.imgUrl || null;
  visited.add(node.id);
  const edge = (app.edges || []).find(e =>
    e.to === node.id && (node.inputs || []).some(p => p.id === e.toPort));
  if (!edge) return node.imgUrl || null;
  const src = (app.nodes || []).find(n => n.id === edge.from);
  if (!src) return node.imgUrl || null;
  const pin = (src.outputs || []).find(p => p.id === edge.fromPort);
  if (pin && pin.thumb) return pin.thumb;
  if (src.nodeType === "asset") {
    return resolveAssetImageUrl(app, src, visited) || src.imgUrl || node.imgUrl || null;
  }
  return src.resultImage || src.imgUrl || node.imgUrl || null;
}

/**
 * Viewer liveness status follows the wiring: "Viewing ← <pin>" when wired,
 * "Waiting for Input" when not. Only viewer-template cards flip; the source
 * photo keeps its own status. Call before render so the strip picks it up.
 */
export function syncViewerStatus(app, node) {
  if (!node || node.templateKey !== "asset-photo-viewer") return;
  const label = wiredViewLabel(app, node);
  node.status = label ? `Viewing ← ${label}` : "Waiting for Input";
}

/** Upstream pin name a viewer card is showing, or null when unwired. */
export function wiredViewLabel(app, node) {
  if (!node || node.nodeType !== "asset" || !node.inputs?.length) return null;
  const edge = (app.edges || []).find(e =>
    e.to === node.id && node.inputs.some(p => p.id === e.toPort));
  if (!edge) return null;
  const src = (app.nodes || []).find(n => n.id === edge.from);
  if (!src) return null;
  const pin = (src.outputs || []).find(p => p.id === edge.fromPort);
  return pin ? pin.name : src.title || null;
}

/** Facebook blue — every Meta model card wears it on its header. */
export const META_BLUE = "#0866FF";

/** Catalog families that are Meta models (utility/graph-op cards excluded). */
export const META_MODEL_FAMILIES = ["meta-ai", "sapiens", "avatar-solver"];

export function isMetaModel(node) {
  return !!node && node.nodeType === "action" && META_MODEL_FAMILIES.includes(node.family);
}

/**
 * Paint memo: true when this exact HTML string was already painted here, so
 * the caller can skip a redundant innerHTML/outerHTML write (no re-parse, no
 * image re-decode, listeners and focus survive). Compares our own strings —
 * never DOM reads — so a skip is always pixel-identical.
 */
export function sameHtml(el, html) {
  if (el._paintedHtml === html) return true;
  el._paintedHtml = html;
  return false;
}

/** Card accent color: Meta models are fb-blue, assets follow their kind color. */
export function nodeAccentColor(node) {
  if (!node) return "#94a3b8";
  if (isMetaModel(node)) return META_BLUE;
  if (node.nodeType === "asset") {
    const byKind = {
      image: "#10b981", pointcloud: "#3b82f6", depthmap: "#06b6d4",
      normalmap: "#0866FF", mesh: "#f59e0b", prompt: "#ec4899",
    };
    return byKind[node.assetKind] || "#94a3b8";
  }
  return "#94a3b8";
}
