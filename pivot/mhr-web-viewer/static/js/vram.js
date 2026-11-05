// DOM-free VRAM accounting (extracted from perf.js for testability).
//
// Dedupe is by ARRAY, not geometry: every GPU pool slot shell is its own
// BufferGeometry sharing the pool's mega attributes, so geometry-level
// dedupe recounts ~150MB per shell (the 77GB bug). Distinct views over one
// buffer still count per view — three uploads each attribute separately.
export function computeVramBytes(root, blendshapes, correctiveDeltas) {
  const seenArrays = new Set();
  let bytes = 0;
  const add = (arr) => {
    if (!arr || !Number.isFinite(arr.byteLength)) return;
    if (seenArrays.has(arr)) return;
    seenArrays.add(arr);
    bytes += arr.byteLength;
  };
  let geometries = 0;
  root.traverse((obj) => {
    const geom = obj && obj.geometry;
    if (geom) {
      geometries++;
      const attrs = geom.attributes || {};
      for (const name in attrs) {
        const attr = attrs[name];
        if (attr) add(attr.array);
      }
      if (geom.index) add(geom.index.array);
    }
    if (obj && obj.isInstancedMesh) {
      if (obj.instanceMatrix) add(obj.instanceMatrix.array);
      if (obj.instanceColor) add(obj.instanceColor.array);
    }
  });
  if (blendshapes) {
    add(blendshapes.neutralVertices);
    add(blendshapes.blendshapeDeltas);
  }
  add(correctiveDeltas);
  return { bytes, geometries };
}
