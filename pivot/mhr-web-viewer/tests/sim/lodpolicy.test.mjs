import test from "node:test";
import assert from "node:assert/strict";
import {
  CPU_DEBUG_COLOR,
  LOD_BANDS,
  LOD_DEBUG_COLORS,
  LOD_HYSTERESIS,
  LOD_MIGRATION_BUDGET,
  LOD_POOL_SLOTS,
  POOL_LODS,
  bandForDistance,
  migrateTarget,
  overflowLods,
  sortNearestFirst,
} from "../../static/js/sim/LodPolicy.js";

test("pool slots cover every pool lod (LOD0 excluded)", () => {
  assert.deepEqual(POOL_LODS, [1, 2, 3, 4, 5, 6]);
  for (const l of POOL_LODS) {
    assert.ok(Number.isInteger(LOD_POOL_SLOTS[l]) && LOD_POOL_SLOTS[l] > 0, `lod ${l}`);
  }
  assert.ok(!("0" in LOD_POOL_SLOTS), "LOD0 must have no pool");
  // LOD1 keeps the old single-pool capacity: auto-LOD must not regress it.
  assert.equal(LOD_POOL_SLOTS[1], 256);
  // 8 clears camera-swing backlogs in seconds; saturated drain measured
  // holding render at 3-13ms (600 agents). Raise deliberately, not casually.
  assert.equal(LOD_MIGRATION_BUDGET, 8);
});

test("migration candidates sort nearest-first (stable)", () => {
  const cands = [
    { id: "far", dist: 30 },
    { id: "near", dist: 2 },
    { id: "mid", dist: 10 },
    { id: "near2", dist: 2 },
  ];
  const out = sortNearestFirst(cands);
  assert.equal(out, cands, "sorts in place");
  assert.deepEqual(out.map((c) => c.id), ["near", "near2", "mid", "far"]);
});

test("bands tile distance without gaps", () => {
  assert.equal(LOD_BANDS.length, POOL_LODS.length);
  assert.equal(LOD_BANDS[LOD_BANDS.length - 1].max, Infinity);
  for (const [d, want] of [[0, 1], [2.5, 1], [5, 1], [5.1, 2], [7.5, 2], [10, 3], [13, 4], [17, 5], [18, 6], [1e9, 6]]) {
    assert.equal(bandForDistance(d), want, `d=${d}`);
  }
});

test("hysteresis holds the band at edges, jumps when past margin", () => {
  assert.ok(LOD_HYSTERESIS > 0 && LOD_HYSTERESIS < 0.5);
  // LOD1 band is [0, 5]: inside stays, margin band stays, past margin moves.
  assert.equal(migrateTarget(1, 0), 1);
  assert.equal(migrateTarget(1, 4), 1);
  assert.equal(migrateTarget(1, 5), 1);
  assert.equal(migrateTarget(1, 5.4), 1);
  assert.equal(migrateTarget(1, 5.6), 2);
  // LOD2 band is (5, 7.5].
  assert.equal(migrateTarget(2, 6), 2);
  assert.equal(migrateTarget(2, 4.6), 2);
  assert.equal(migrateTarget(2, 4.4), 1);
  // Far jumps land directly (multi-band migration costs the same).
  assert.equal(migrateTarget(1, 30), 6);
  assert.equal(migrateTarget(5, 1), 1);
  // Extremes never migrate outward; unknown LODs resolve finest.
  assert.equal(migrateTarget(1, 0), 1);
  assert.equal(migrateTarget(6, 1e9), 6);
  assert.equal(migrateTarget(0, 30), 6);
});

test("debug palette covers every pool with distinct colors", () => {
  const seen = new Set([CPU_DEBUG_COLOR]);
  for (const l of POOL_LODS) {
    const hex = LOD_DEBUG_COLORS[l];
    assert.ok(Number.isInteger(hex) && hex >= 0 && hex <= 0xffffff, `lod ${l}`);
    assert.ok(!seen.has(hex), `duplicate color for lod ${l}`);
    seen.add(hex);
  }
});

test("overflow sheds coarser, never steals finer", () => {
  assert.deepEqual(overflowLods(1), [1, 2, 3, 4, 5, 6]);
  assert.deepEqual(overflowLods(5), [5, 6]);
  assert.deepEqual(overflowLods(6), [6]);
  assert.deepEqual(overflowLods(0), [1, 2, 3, 4, 5, 6]);
});
