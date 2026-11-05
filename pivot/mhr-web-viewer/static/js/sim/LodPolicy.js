// LOD auto-assignment policy (pure: no THREE, no DOM, Node-testable).
//
// Bands are camera distances in meters; the pen is ±5.5m (diag ~15.6m) so
// the top bands only trigger on wide zooms. LOD 1 is the source mesh.
// LOD 0 is deliberately excluded: a detail-free subdivision of LOD 1, so
// the pool system pretends it does not exist (agents never want it, no
// pool is ever built for it, overflow skips it).
export const POOL_LODS = [1, 2, 3, 4, 5, 6];

// Max GPU slots per LOD pool. Uniform 256: auto-LOD never regresses the
// old single-pool capacity, and every pool fits default WebGPU limits.
export const LOD_POOL_SLOTS = { 1: 256, 2: 256, 3: 256, 4: 256, 5: 256, 6: 256 };

// Band edges: band i covers (BANDS[i-1].max, BANDS[i].max].
export const LOD_BANDS = [
  { lod: 1, max: 5 },
  { lod: 2, max: 7.5 },
  { lod: 3, max: 10 },
  { lod: 4, max: 13 },
  { lod: 5, max: 17 },
  { lod: 6, max: Infinity },
];

// Hysteresis overlap at band edges: an agent must cross 10% past the edge
// before migrating, so orbit wobble never churns pools.
export const LOD_HYSTERESIS = 0.1;

// Migrations per tick. Each move swaps the mesh and pays first-render
// init on its next frame; saturated drain at 600 agents measured holding
// render at 3-13ms with occasional spikes, so 8 clears camera-swing
// backlogs (hundreds of invalidated bands) in seconds, not minutes.
export const LOD_MIGRATION_BUDGET = 8;

// Debug palette for the "color by LOD" toggle (hex): L1 red → L6 purple.
// CPU agents (no pool) render gray so the toggle also answers "CPU or GPU?".
export const LOD_DEBUG_COLORS = {
  1: 0xff3838,
  2: 0xff9500,
  3: 0xffcc00,
  4: 0x34c759,
  5: 0x007aff,
  6: 0xaf52de,
};
export const CPU_DEBUG_COLOR = 0x8e8e93;

export function bandForDistance(d) {
  for (const band of LOD_BANDS) {
    if (d <= band.max) return band.lod;
  }
  return 6;
}

/**
 * Hysteresis-aware desired LOD: returns currentLod unless the distance is
 * past the band edge + margin in the move direction (then jumps straight
 * to the correct band — multi-band migration costs the same single move).
 * Unknown current LODs (stale pins) resolve as the finest band.
 */
export function migrateTarget(currentLod, d) {
  const idx = LOD_BANDS.findIndex((b) => b.lod === currentLod);
  const safeIdx = idx === -1 ? 0 : idx;
  const lo = safeIdx === 0 ? 0 : LOD_BANDS[safeIdx - 1].max;
  const hi = LOD_BANDS[safeIdx].max;
  if (d > hi * (1 + LOD_HYSTERESIS)) return bandForDistance(d);
  if (d < lo * (1 - LOD_HYSTERESIS)) return bandForDistance(d);
  return currentLod;
}

/**
 * Pool preference order for an agent wanting `preferred`: the band pool,
 * then coarser pools (perf-safe overflow), then CPU as the last resort
 * (caller-side; not in this list). Never finer: a full preferred pool
 * must shed detail, not steal another band's slots.
 */
export function overflowLods(preferred) {
  return POOL_LODS.filter((l) => l >= preferred);
}

/**
 * Nearest-first migration priority: after a camera swing hundreds of
 * bands invalidate at once, and id order would stall the visible crowd
 * behind far agents. Sorts candidate {dist} records ascending, in place
 * (stable, so ties keep id order).
 */
export function sortNearestFirst(cands) {
  cands.sort((a, b) => a.dist - b.dist);
  return cands;
}
