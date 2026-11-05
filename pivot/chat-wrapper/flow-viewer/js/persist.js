/**
 * Flow Persistence & Asset Chronology (viewer-local).
 * Flows (graph + positions + run state) save to localStorage so previous
 * flows reopen intact; blob: photos inline as data-URLs on save because blob
 * URLs die with the session, while backend artifact URLs persist as-is.
 * Saved photos downscale to INLINE_BLOB_MAX_DIM (fast saves, small quota);
 * the live canvas keeps full resolution until reload.
 */

export const FLOW_STORAGE_KEY = "reui.flow-viewer.flows.v1";
export const FLOW_DATA_VERSION = 1;

/** Drop per-run trace/buffers from a node copy: saved flows stay small. */
export function slimNodeForSave(node) {
  const copy = JSON.parse(JSON.stringify(node));
  delete copy.apiCall;
  delete copy.progress;
  delete copy._progressLine;
  delete copy._progressCid;
  delete copy.resultData;
  delete copy.splitPins;
  delete copy.masks;
  if (Array.isArray(copy.runLog)) copy.runLog = copy.runLog.slice(-8);
  return copy;
}

/** Plain-data snapshot of the flow; JSON-safe after inlineBlobs(). */
export function serializeFlow(app, name) {
  return {
    version: FLOW_DATA_VERSION,
    id: `flow-${Date.now()}-${Math.floor(Math.random() * 1e6)}`,
    name: name || "Untitled flow",
    savedAt: Date.now(),
    nodes: (app.nodes || []).map(slimNodeForSave),
    edges: JSON.parse(JSON.stringify(app.edges || [])),
    customPositions: JSON.parse(JSON.stringify(app.customPositions || {})),
    activeRunId: app.activeRunId ?? null,
    selectedNodeId: app.selectedNodeId ?? null,
  };
}

/** Validate parsed flow data; returns the object or null when unusable. */
export function parseFlowData(raw) {
  if (!raw || typeof raw !== "object") return null;
  if (raw.version !== FLOW_DATA_VERSION) return null;
  if (!Array.isArray(raw.nodes) || !Array.isArray(raw.edges)) return null;
  if (!raw.nodes.every(n => n && typeof n.id === "string")) return null;
  return raw;
}

/** Max persisted photo dimension: phone photos shrink on save (fast + small). */
export const INLINE_BLOB_MAX_DIM = 2048;

/** Pure: fit (w, h) inside maxDim, preserving aspect. No-op when small. */
export function scaleToMax(w, h, maxDim = INLINE_BLOB_MAX_DIM) {
  const m = Math.max(w, h);
  if (!(m > maxDim)) return { w, h };
  const s = maxDim / m;
  return { w: Math.max(1, Math.round(w * s)), h: Math.max(1, Math.round(h * s)) };
}

/**
 * One blob: URL to a data-URL. Decodable images downscale to
 * INLINE_BLOB_MAX_DIM via canvas (PNG stays PNG for masks/transparency,
 * everything else becomes JPEG); anything undecodable goes through
 * FileReader's native base64 — linear, no quadratic string loop. Node
 * runtimes (no canvas/FileReader) take a Buffer passthrough instead.
 */
export async function inlineOneBlob(url) {
  const res = await fetch(url);
  const blob = await res.blob();
  if (typeof createImageBitmap === "function" && (blob.type || "").startsWith("image/")) {
    try {
      const bm = await createImageBitmap(blob);
      const fit = scaleToMax(bm.width, bm.height);
      const c = document.createElement("canvas");
      c.width = fit.w; c.height = fit.h;
      c.getContext("2d").drawImage(bm, 0, 0, fit.w, fit.h);
      bm.close();
      return blob.type === "image/png"
        ? c.toDataURL("image/png")
        : c.toDataURL("image/jpeg", 0.92);
    } catch {
      // Not actually decodable — fall through to FileReader.
    }
  }
  if (typeof FileReader !== "undefined") {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }
  const buf = Buffer.from(await blob.arrayBuffer());
  return `data:${blob.type || "image/png"};base64,${buf.toString("base64")}`;
}

/**
 * Inline blob: image URLs as data-URLs (mutates the given nodes, which must
 * already be clones). Backend artifact URLs and data-URIs pass through.
 */
export async function inlineBlobs(nodes) {
  for (const node of nodes || []) {
    for (const key of ["imgUrl", "resultImage"]) {
      const url = node[key];
      if (typeof url === "string" && url.startsWith("blob:")) {
        node[key] = await inlineOneBlob(url);
      }
    }
  }
  return nodes;
}

/** Asset nodes oldest-first (stable for equal timestamps). */
export function flowAssetChronology(nodes) {
  return (nodes || [])
    .filter(n => n && n.nodeType === "asset")
    .slice()
    .sort((a, b) => (a.createdAt || 0) - (b.createdAt || 0));
}

export function loadFlowList() {
  try {
    if (typeof localStorage === "undefined") return [];
    const raw = localStorage.getItem(FLOW_STORAGE_KEY);
    if (!raw) return [];
    const list = JSON.parse(raw);
    return Array.isArray(list) ? list.filter(parseFlowData) : [];
  } catch {
    return [];
  }
}

export function storeFlowList(list) {
  try {
    if (typeof localStorage === "undefined") return { ok: false, error: "no local storage" };
    localStorage.setItem(FLOW_STORAGE_KEY, JSON.stringify(list));
    return { ok: true };
  } catch (err) {
    const quota = err && (err.name === "QuotaExceededError" || err.code === 22);
    return { ok: false, error: quota ? "quota" : String(err?.message || err) };
  }
}

/** Human clock time for lists ("14:03"). */
export function formatFlowTime(ts) {
  const d = new Date(Number(ts) || 0);
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

/** Human date for lists ("Nov 4"). */
export function formatFlowDate(ts) {
  return new Date(Number(ts) || 0).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
