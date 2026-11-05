import test from "node:test";
import assert from "node:assert/strict";
import { computeVramBytes } from "../../static/js/vram.js";

function rootOf(items) {
  return { traverse(fn) { for (const obj of items) fn(obj); } };
}

function attr(n) {
  return { array: new Float32Array(n) };
}

test("shared pool mega attributes count once, not once per shell", () => {
  // Every pool slot shell is its own geometry sharing the pool mega arrays.
  const mega = attr(100); // 400 bytes, shared by all shells
  const shells = [];
  for (let i = 0; i < 10; i++) {
    shells.push({
      geometry: {
        attributes: { position: mega, normal: attr(100) },
        index: { array: new Uint32Array(30) },
      },
    });
  }
  const { bytes, geometries } = computeVramBytes(rootOf(shells), null, null);
  // mega once (400) + 10 normals (400 each) + 10 indices (120 each)
  assert.equal(bytes, 400 + 10 * 400 + 10 * 120);
  assert.equal(geometries, 10);
});

test("instanced attributes and client arrays are included once", () => {
  const objects = [
    {
      isInstancedMesh: true,
      geometry: { attributes: { position: attr(12) }, index: null },
      instanceMatrix: { array: new Float32Array(32) },
      instanceColor: null,
    },
  ];
  const blendshapes = {
    neutralVertices: new Float32Array(6),
    blendshapeDeltas: new Float32Array(9),
  };
  const correctiveDeltas = new Float32Array(3);
  const { bytes } = computeVramBytes(rootOf(objects), blendshapes, correctiveDeltas);
  assert.equal(bytes, 48 + 128 + 24 + 36 + 12);
});

test("missing and non-typed arrays never produce NaN", () => {
  const objects = [
    { geometry: { attributes: { position: attr(4), broken: { array: [1, 2, 3] } } } },
    { geometry: null },
    {},
  ];
  const { bytes } = computeVramBytes(rootOf(objects), {}, undefined);
  assert.equal(bytes, 16);
});
