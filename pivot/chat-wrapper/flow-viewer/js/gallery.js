/**
 * Node sidebar: every catalog entry plus every wire/asset type in one
 * grouped, searchable dock.
 *
 * The right-click menu is a task-ordered subset tuned for flow building;
 * this is the complete collection — actions by family, assets together,
 * then the pin-type reference with makers and takers — so nothing
 * placeable hides and no menu-diving is needed. Tiles spawn through the
 * same spawnModelAt path as the menu. Pure builders carry the logic; the
 * sidebar binding is a thin delegation layer.
 */
import { MODEL_CATALOG } from "./catalog.js";
import { escapeHtml, PIN_TYPES } from "./util.js";
import { spawnModelAt } from "./context-menu.js";

/** Gallery group order: actions by family, then assets, then the unknown. */
export const GALLERY_GROUP_ORDER = [
  { key: "meta-ai", label: "Meta AI", hint: "Cloud API" },
  { key: "sapiens", label: "Sapiens", hint: "Local GPU" },
  { key: "avatar-solver", label: "Avatar Solver", hint: "Local GPU" },
  { key: "mesh", label: "Mesh Render", hint: "Local WebGL" },
  { key: "utility", label: "Utilities", hint: "Local" },
  { key: "assets", label: "Assets", hint: "Cards" },
  { key: "other", label: "Other", hint: "" },
];

const KNOWN_FAMILIES = new Set(["meta-ai", "sapiens", "avatar-solver", "mesh", "utility"]);

/** Pure: gallery group key for one catalog entry. */
export function galleryGroupKey(entry) {
  if (!entry || !entry.nodeType) return "other";
  if (entry.nodeType === "asset") return "assets";
  if (entry.nodeType !== "action") return "other";
  return KNOWN_FAMILIES.has(entry.family) ? entry.family : "other";
}

/** Pure: catalog entries bucketed into ordered, non-empty groups. */
export function groupCatalogEntries(catalog) {
  const buckets = new Map(GALLERY_GROUP_ORDER.map(g => [g.key, []]));
  for (const entry of Object.values(catalog || {})) {
    if (!entry || !entry.id) continue;
    buckets.get(galleryGroupKey(entry)).push(entry);
  }
  return GALLERY_GROUP_ORDER
    .map(g => ({ ...g, items: buckets.get(g.key) }))
    .filter(g => g.items.length > 0);
}

/** Pure: keep groups/items matching the query (title, sub, text, id). */
export function filterGalleryGroups(groups, query) {
  const q = String(query || "").trim().toLowerCase();
  if (!q) return groups;
  return groups
    .map(g => ({
      ...g,
      items: g.items.filter(e => [e.title, e.subtitle, e.description, e.id]
        .filter(Boolean).join(" ").toLowerCase().includes(q)),
    }))
    .filter(g => g.items.length > 0);
}

/** Pure: gallery list markup — one section per group, one tile per entry. */
export function renderGalleryHtml(groups) {
  if (!groups || groups.length === 0) {
    return `<div class="flow-nodes-empty">No nodes match.</div>`;
  }
  return groups.map(g => `
    <div class="flow-nodes-group">
      <div class="flow-nodes-group-title">
        <span>${escapeHtml(g.label)}</span>
        <span class="reui-badge" data-variant="meta-light" data-size="xs">${g.items.length}${g.hint ? ` · ${escapeHtml(g.hint)}` : ""}</span>
      </div>
      ${g.items.map(e => `
        <button type="button" class="flow-nodes-tile" data-add-key="${escapeHtml(e.id)}" title="${escapeHtml(e.description || e.title)}">
          <div class="reui-icon-tile" data-variant="${escapeHtml(e.iconVariant || "soft")}" data-size="xs">
            ${e.iconSvg || ""}
          </div>
          <div class="flow-context-item-info">
            <div class="flow-context-item-title">${escapeHtml(e.title)}</div>
            <div class="flow-context-item-sub">${escapeHtml(e.subtitle || "")}</div>
          </div>
        </button>`).join("")}
    </div>`).join("");
}

/** Pure given app geometry: viewport-center world point for gallery spawns. */
export function gallerySpawnCenter(app) {
  const w = app.viewport ? app.viewport.clientWidth : 800;
  const h = app.viewport ? app.viewport.clientHeight : 600;
  const scale = app.scale || 1;
  return {
    x: (w / 2 - (app.panX || 0)) / scale,
    y: (h / 2 - (app.panY || 0)) / scale,
  };
}

/** Short card name for the type index ("Action: X" → "X"). */
export function shortCardTitle(title) {
  return String(title || "").replace(/^(Action|Asset):\s*/, "") || "Untitled";
}

/**
 * Pure: every PIN_TYPES wire/asset type with its canonical color, accepts
 * list, and the template-level producers/consumers (which catalog cards
 * emit or take it). Exploded split pins are runtime-only, so template
 * pins are the whole story here.
 */
export function assetTypeIndex(catalog) {
  const rows = Object.entries(PIN_TYPES).map(([type, def]) => ({
    type,
    label: def.label,
    color: def.color,
    accepts: [...def.accepts],
    producers: [],
    consumers: [],
  }));
  const byType = new Map(rows.map(r => [r.type, r]));
  for (const entry of Object.values(catalog || {})) {
    if (!entry) continue;
    const name = shortCardTitle(entry.title);
    for (const o of entry.outputs || []) {
      const row = o && byType.get(o.type);
      if (row && !row.producers.includes(name)) row.producers.push(name);
    }
    for (const i of entry.inputs || []) {
      if (!i) continue;
      const takes = new Set([i.type, ...((i && i.accepts) || [])]);
      for (const t of takes) {
        const row = byType.get(t);
        if (row && !row.consumers.includes(name)) row.consumers.push(name);
      }
    }
  }
  for (const row of rows) {
    row.producers.sort();
    row.consumers.sort();
  }
  return rows;
}

/** Pure: keep types matching the query (label, key, producers, consumers). */
export function filterAssetTypes(rows, query) {
  const q = String(query || "").trim().toLowerCase();
  if (!q) return rows;
  return rows.filter(r => [r.label, r.type, ...r.producers, ...r.consumers]
    .join(" ").toLowerCase().includes(q));
}

/** Pure: type reference rows — color dot, label, makers and takers. */
export function renderAssetTypesHtml(rows) {
  if (!rows || rows.length === 0) {
    return `<div class="flow-nodes-empty">No types match.</div>`;
  }
  return rows.map(r => `
    <div class="flow-type-row" title="${escapeHtml(r.type)} accepts: ${(r.accepts || []).join(", ")}">
      <span class="flow-type-dot" style="background:${escapeHtml(r.color)}"></span>
      <div class="flow-type-info">
        <div class="flow-type-name">${escapeHtml(r.label)} <span class="flow-type-key">${escapeHtml(r.type)}</span></div>
        <div class="flow-type-io">made by: ${r.producers.map(n => escapeHtml(n)).join(", ") || "—"} · used by: ${r.consumers.map(n => escapeHtml(n)).join(", ") || "—"}</div>
      </div>
    </div>`).join("");
}

/** Wire the docked sidebar: one search filters cards and types; tiles spawn. */
export function bindNodesSidebar(app, panel) {
  const nodesList = panel.querySelector(".flow-nodes-list");
  const typesList = panel.querySelector(".flow-types-list");
  const search = panel.querySelector(".flow-nodes-search");
  const groups = groupCatalogEntries(MODEL_CATALOG);
  const types = assetTypeIndex(MODEL_CATALOG);
  const paint = (q) => {
    nodesList.innerHTML = renderGalleryHtml(filterGalleryGroups(groups, q));
    typesList.innerHTML = renderAssetTypesHtml(filterAssetTypes(types, q));
  };
  search.addEventListener("input", (e) => paint(e.target.value));
  search.addEventListener("keydown", (e) => e.stopPropagation());
  nodesList.addEventListener("click", (e) => {
    const tile = e.target.closest("[data-add-key]");
    if (!tile) return;
    const c = gallerySpawnCenter(app);
    spawnModelAt(app, tile.dataset.addKey, c.x, c.y);
  });
  paint("");
}
