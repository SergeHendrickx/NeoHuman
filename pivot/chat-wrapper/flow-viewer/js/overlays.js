/**
 * Custom result displays for Sapiens cards.
 *
 * Pose: SVG skeleton + keypoint dots over the result image (same overlay
 * pattern as the detection boxes). Seg: class legend chips, click-to-isolate
 * canvas, hover tooltips. Pure view-models up top (node-tested), thin DOM
 * wiring at the bottom.
 */
import { escapeHtml } from "./util.js";

// 308 split from sam3dbody-ingest/run_all.py draw_pose: indices 0-69 are
// body/foot/hand, 70-307 are dense face. Backend colors are BGR dots:
// (0,255,0) green body, (200,70,255) magenta face.
export const POSE_BODY_COUNT = 70;
export const POSE_BODY_COLOR = "#22c55e";
export const POSE_FACE_COLOR = "#c846ff";
export const POSE_BODY_R = 5;
export const POSE_FACE_R = 2;
export const POSE_DIM_SCORE = 0.3;
export const SEG_DIM = 0.22;

export function isPlottedKpt(t) {
  return Array.isArray(t) && Number.isFinite(t[0]) && Number.isFinite(t[1]);
}

function kptScore(t) {
  return (typeof t[2] === "number" && Number.isFinite(t[2])) ? t[2] : null;
}

/** Pure: plotted dots with group + score, null-coordinate points dropped. */
export function poseDotJobs(keypoints) {
  const dots = [];
  (keypoints || []).forEach((t, i) => {
    if (!isPlottedKpt(t)) return;
    dots.push({ i, x: t[0], y: t[1], s: kptScore(t), body: i < POSE_BODY_COUNT });
  });
  return dots;
}

function cssColor(rgb, fallback = "#ffffff") {
  return (Array.isArray(rgb) && rgb.length >= 3)
    ? `rgb(${rgb[0]},${rgb[1]},${rgb[2]})` : fallback;
}

/**
 * Pure: skeleton links in image pixels. Links whose endpoints are missing
 * (null coords) are dropped; a null topology yields no links (dots-only
 * tier, e.g. before data/pose_topology.json loads or while offline).
 */
export function poseLinkJobs(keypoints, topology) {
  if (!topology || !Array.isArray(topology.links)) return [];
  const jobs = [];
  for (const link of topology.links) {
    const [a, b, color] = link;
    const pa = keypoints ? keypoints[a] : null;
    const pb = keypoints ? keypoints[b] : null;
    if (!isPlottedKpt(pa) || !isPlottedKpt(pb)) continue;
    jobs.push({ x1: pa[0], y1: pa[1], x2: pb[0], y2: pb[1], color: cssColor(color) });
  }
  return jobs;
}

/**
 * Pure: inner SVG for the pose overlay — skeleton lines under score-faded
 * dots. Circles carry data attrs for the hover tooltip; a transparent fat
 * stroke makes the 2px face dots hoverable.
 */
export function renderPoseSvgInner(node, topology) {
  const kpts = node.resultKeypoints;
  if (!Array.isArray(kpts) || kpts.length === 0) return "";
  const names = (topology && Array.isArray(topology.names)) ? topology.names : null;
  let s = "";
  for (const l of poseLinkJobs(kpts, topology)) {
    s += `<line x1="${l.x1}" y1="${l.y1}" x2="${l.x2}" y2="${l.y2}" class="flow-kpts-link" stroke="${l.color}"/>`;
  }
  for (const d of poseDotJobs(kpts)) {
    const fill = d.body ? POSE_BODY_COLOR : POSE_FACE_COLOR;
    const r = d.body ? POSE_BODY_R : POSE_FACE_R;
    const dim = (d.s !== null && d.s < POSE_DIM_SCORE) ? ` opacity="0.35"` : "";
    const nm = (names && names[d.i]) ? ` data-kname="${escapeHtml(names[d.i])}"` : "";
    const sc = d.s !== null ? ` data-kscore="${d.s}"` : "";
    s += `<circle cx="${d.x}" cy="${d.y}" r="${r}" fill="${fill}"${dim} data-kidx="${d.i}"${nm}${sc}/>`;
  }
  return s;
}

/** Pure: hover label for one keypoint. */
export function poseHoverText(name, idx, score) {
  const who = name || `#${idx}`;
  const how = (score === null || score === undefined || score === "") ? "no score" : score;
  return `${who} · ${how}`;
}

/**
 * Pure: legend rows sorted by pixel count desc. Missing names/palette fall
 * back to class_<id> / gray so a thin legend still renders honestly.
 */
export function segLegendRows(legend) {
  if (!legend || !legend.counts) return [];
  const total = (Number(legend.width) || 0) * (Number(legend.height) || 0);
  return Object.entries(legend.counts).map(([id, count]) => ({
    id: String(id),
    count,
    name: (legend.names && legend.names[id]) || `class_${id}`,
    color: (legend.palette && legend.palette[id]) || [128, 128, 128],
    pct: total > 0 ? (100 * count / total) : 0,
  })).sort((a, b) => b.count - a.count);
}

/**
 * Pure: legend rows worth a spawned asset card — present classes, biggest
 * first, background dropped (it is detected in every frame and would be a
 * noise card). The caller caps the count and notes the remainder.
 */
export function segSpawnRows(legend) {
  return segLegendRows(legend).filter(r => r.count > 0 && !/background/i.test(String(r.name || "")));
}

/** Pure: legend chips HTML; the isolated class (if any) reads active. */
export function renderSegLegendHtml(node) {
  const rows = segLegendRows(node.resultSegLegend);
  if (rows.length === 0) return "";
  const chips = rows.map(r => `
    <button type="button" class="flow-seg-chip${String(node._segIsolated) === r.id ? " active" : ""}" data-seg-chip="${escapeHtml(r.id)}" title="Click to isolate ${escapeHtml(r.name)}">
      <span class="flow-seg-swatch" style="background:${cssColor(r.color, "#808080")}"></span>
      <span class="flow-seg-name">${escapeHtml(r.name)}</span>
      <span class="flow-seg-count">${r.count.toLocaleString("en-US")} · ${r.pct.toFixed(1)}%</span>
    </button>`).join("");
  return `<div class="flow-seg-legend" data-seg-legend="1">${chips}</div>`;
}

/** Pure: class id at image pixel (x, y), or -1 outside the frame. */
export function segClassAt(pixels, w, h, x, y) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  if (ix < 0 || iy < 0 || ix >= w || iy >= h) return -1;
  return pixels[iy * w + ix];
}

/**
 * Pure: isolate one class — kept pixels verbatim, everything else dimmed
 * toward black. Returns a new array; src (live ImageData) is untouched.
 */
export function isolateSegPixels(src, pixels, isolatedId, dim = SEG_DIM) {
  const out = new Uint8ClampedArray(src.length);
  const keep = Number(isolatedId);
  for (let p = 0, o = 0; p < pixels.length; p++, o += 4) {
    if (pixels[p] === keep) {
      out[o] = src[o]; out[o + 1] = src[o + 1]; out[o + 2] = src[o + 2]; out[o + 3] = src[o + 3];
    } else {
      out[o] = src[o] * dim; out[o + 1] = src[o + 1] * dim; out[o + 2] = src[o + 2] * dim;
      out[o + 3] = src[o + 3];
    }
  }
  return out;
}

// ---- async loaders (browser only; resolve null under node) ----

let topoPromise = null;

/** Cached pose topology ({names, links, groups}) from data/. */
export function loadPoseTopology() {
  if (typeof document === "undefined") return Promise.resolve(null);
  if (!topoPromise) {
    topoPromise = fetch(new URL("data/pose_topology.json", document.baseURI).toString())
      .then(r => (r.ok ? r.json() : null))
      .catch(() => null);
  }
  return topoPromise;
}

const labelsCache = new Map(); // url -> Promise<{pixels,w,h} | null>
const labelsReady = new Map(); // url -> resolved {pixels,w,h} (sync hover path)

/**
 * Cached decode of a seg_labels.png into per-pixel class ids (R channel of
 * the grayscale PNG). Resolves null when the fetch or decode fails.
 */
export function loadSegLabels(url) {
  if (typeof document === "undefined" || !url) return Promise.resolve(null);
  if (!labelsCache.has(url)) {
    labelsCache.set(url, fetch(url)
      .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.blob(); })
      .then(b => createImageBitmap(b))
      .then(bm => {
        // Read dimensions BEFORE close(): a closed bitmap reports 0x0,
        // which used to poison every cached record (dead hover, and a
        // createImageData crash in the first consumer that used w/h).
        const W = bm.width, H = bm.height;
        const c = document.createElement("canvas");
        c.width = W; c.height = H;
        const ctx = c.getContext("2d", { willReadFrequently: true });
        ctx.drawImage(bm, 0, 0);
        const d = ctx.getImageData(0, 0, W, H).data;
        const px = new Uint8Array(W * H);
        for (let i = 0; i < px.length; i++) px[i] = d[i * 4];
        bm.close();
        const rec = { pixels: px, w: W, h: H };
        labelsReady.set(url, rec);
        return rec;
      })
      .catch(() => null));
  }
  return labelsCache.get(url);
}

export function clearSegLabelsCache() {
  labelsCache.clear();
  labelsReady.clear();
}

// ---- DOM wiring (per-card, called from the node-renderer bind pass) ----

function showTip(stage, clientX, clientY, text) {
  const tip = stage.querySelector(".flow-chart-tip");
  if (!tip) return;
  const rect = stage.getBoundingClientRect();
  tip.textContent = text;
  tip.hidden = false;
  const x = Math.min(Math.max(4, clientX - rect.left + 12), Math.max(4, rect.width - 140));
  const y = Math.max(4, clientY - rect.top - 30);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}

function hideTip(stage) {
  const tip = stage.querySelector(".flow-chart-tip");
  if (tip) tip.hidden = true;
}

function imgReady(img) {
  if (img.complete && img.naturalWidth) return Promise.resolve();
  return new Promise(res => {
    img.addEventListener("load", () => res(), { once: true });
    img.addEventListener("error", () => res(), { once: true });
  });
}

function poseHover(e, stage) {
  const t = e.target;
  const dot = t && t.closest ? t.closest("circle[data-kidx]") : null;
  if (!dot) { hideTip(stage); return; }
  showTip(stage, e.clientX, e.clientY,
    poseHoverText(dot.dataset.kname || null, dot.dataset.kidx, dot.dataset.kscore ?? null));
}

function segHover(e, node, stage) {
  const legend = node.resultSegLegend;
  const url = node.resultSegLabelsUrl;
  if (!legend || !url) return;
  loadSegLabels(url); // warms the cache; the sync map serves once decoded
  const d = labelsReady.get(url);
  if (!d) return;
  const rect = stage.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return;
  const cid = segClassAt(d.pixels, d.w, d.h,
    (e.clientX - rect.left) / rect.width * d.w,
    (e.clientY - rect.top) / rect.height * d.h);
  if (cid < 0) { hideTip(stage); return; }
  const row = segLegendRows(legend).find(r => r.id === String(cid));
  const label = row
    ? `${row.name} · ${row.count.toLocaleString("en-US")} px · ${row.pct.toFixed(1)}%`
    : `class_${cid}`;
  showTip(stage, e.clientX, e.clientY, label);
}

function paintSegIsolate(node, el) {
  const stage = el.querySelector("[data-seg-stage]");
  const canvas = stage ? stage.querySelector("canvas.flow-seg-canvas") : null;
  const img = stage ? stage.querySelector("img.flow-node-img") : null;
  const id = node._segIsolated;
  const syncChips = () => {
    el.querySelectorAll("[data-seg-chip]").forEach(ch => {
      ch.classList.toggle("active", String(id) !== "null" && ch.dataset.segChip === String(id));
    });
  };
  if (id === null || id === undefined || !stage || !canvas || !img) {
    if (canvas) canvas.hidden = true;
    syncChips();
    return Promise.resolve();
  }
  return Promise.all([loadSegLabels(node.resultSegLabelsUrl), imgReady(img)]).then(([d]) => {
    if (node._segIsolated === null || node._segIsolated === undefined || !d) {
      node._segIsolated = (d && node._segIsolated !== null && node._segIsolated !== undefined)
        ? node._segIsolated : null;
      canvas.hidden = true;
      syncChips();
      return;
    }
    canvas.width = d.w;
    canvas.height = d.h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, d.w, d.h);
    const frame = ctx.getImageData(0, 0, d.w, d.h);
    const out = isolateSegPixels(frame.data, d.pixels, node._segIsolated);
    ctx.putImageData(new ImageData(out, d.w, d.h), 0, 0);
    canvas.hidden = false;
    syncChips();
  }).catch(() => {
    node._segIsolated = null;
    canvas.hidden = true;
    syncChips();
  });
}

/**
 * Per-card bind (mirrors bindDisplayOpts): pose hover, seg hover, seg
 * chip isolate. Re-runs restore a persisted isolation without renderCanvas.
 */
export function bindResultStages(app, node, el) {
  const kpts = el.querySelector("[data-kpts-stage]");
  if (kpts) {
    kpts.addEventListener("mousemove", (e) => poseHover(e, kpts));
    kpts.addEventListener("mouseleave", () => hideTip(kpts));
  }
  const stage = el.querySelector("[data-seg-stage]");
  if (stage) {
    stage.addEventListener("mousemove", (e) => segHover(e, node, stage));
    stage.addEventListener("mouseleave", () => hideTip(stage));
  }
  el.querySelectorAll("[data-seg-chip]").forEach(ch => {
    ch.addEventListener("click", (e) => {
      e.stopPropagation();
      const cid = ch.dataset.segChip;
      node._segIsolated = (String(node._segIsolated) === cid) ? null : cid;
      paintSegIsolate(node, el);
    });
  });
  if (node._segIsolated !== null && node._segIsolated !== undefined) {
    paintSegIsolate(node, el);
  }
}
