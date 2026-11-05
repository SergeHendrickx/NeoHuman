// Pure expression-vector math (no imports): mirror pairing over FACS names
// and strongest-vertex scan over the blendshape delta table. Split from
// exprVectors.js so node tests can import it without browser/three modules.
export { mirrorPairs, strongestVertex, formatDeltaMm };

// Signed millimetre vector for the translation tooltip row:
// "(+0.3, -1.2, +7.8)".
function formatDeltaMm(deltaMm) {
  return `(${deltaMm.map((v) => `${v < 0 ? "" : "+"}${v.toFixed(1)}`).join(", ")})`;
}

// Mirror pairing over raw FACS names (lipFunneler_LB): same base, same
// non-L/R suffix letters, L variant paired with R variant. Returns
// pairOf[i] (self when solo). Generic fallback names never pair.
function mirrorPairs(names) {
  const pairOf = names.map((_, i) => i);
  const groups = new Map();
  names.forEach((raw, i) => {
    const name = String(raw);
    const us = name.indexOf("_");
    if (us < 0) return;
    const base = name.slice(0, us);
    const suffix = name.slice(us + 1);
    const hasL = suffix.includes("L");
    const hasR = suffix.includes("R");
    if (!hasL && !hasR) return;
    const key = `${base}_${suffix.replaceAll("L", "X").replaceAll("R", "X")}`;
    let g = groups.get(key);
    if (!g) {
      g = { L: [], R: [] };
      groups.set(key, g);
    }
    (hasL && !hasR ? g.L : g.R).push(i);
  });
  for (const g of groups.values()) {
    const n = Math.min(g.L.length, g.R.length);
    for (let k = 0; k < n; k++) {
      pairOf[g.L[k]] = g.R[k];
      pairOf[g.R[k]] = g.L[k];
    }
  }
  return pairOf;
}

// Strongest vertex of one blendshape column (argmax |delta|), or -1.
function strongestVertex(deltas, vCount, totalBs, shapeIdx) {
  let best = -1;
  let bestSq = 0;
  for (let v = 0; v < vCount; v++) {
    const idx = (v * totalBs + shapeIdx) * 4;
    const d = deltas[idx] * deltas[idx]
      + deltas[idx + 1] * deltas[idx + 1]
      + deltas[idx + 2] * deltas[idx + 2];
    if (d > bestSq) {
      bestSq = d;
      best = v;
    }
  }
  return bestSq > 1e-12 ? best : -1;
}
