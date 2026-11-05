import test from "node:test";
import assert from "node:assert/strict";
import {
  GPU_MAX_AGENTS,
  GPU_MAX_CHAIN,
  allocBoneMega,
  allocOutputMega,
  allocShapeBuffer,
  allocThetaBuffer,
  allocWeightBuffer,
  biasSlotIndex,
  buildBodyBuffer,
  buildLodBuffer,
  buildRigBuffer,
  computeBodyLayout,
  computeLodLayout,
  computeRigLayout,
  neutralOutputNormals,
  neutralOutputPositions,
} from "../../static/js/gpu/layouts.js";

const J = 4;
const V = 8;
const V1 = 8;
const B = 3;

test("rig layout packs sections without overlap", () => {
  const { offsets, totalFloats } = computeRigLayout({ jointCount: J, vertCount: V, srcVertCount: V1 });
  assert.equal(offsets.jointOffsets, 0);
  assert.equal(offsets.jointPreRots, J * 3);
  assert.equal(offsets.jointParents, J * 3 + J * 4);
  assert.equal(offsets.inverses, J * 3 + J * 4 + J);
  assert.equal(offsets.skinIdx, J * 3 + J * 4 + J + J * 16);
  assert.equal(offsets.skinW, J * 3 + J * 4 + J + J * 16 + V * 4);
  assert.equal(offsets.neutralNrm, J * 3 + J * 4 + J + J * 16 + V * 8);
  assert.equal(totalFloats, J * 3 + J * 4 + J + J * 16 + V * 8 + V1 * 3);
  assert.ok(GPU_MAX_CHAIN >= 12);
  assert.equal(GPU_MAX_AGENTS, 256);
});

test("rig buffer round-trips all sections", () => {
  const layout = computeRigLayout({ jointCount: J, vertCount: V, srcVertCount: V1 });
  const src = {
    jointOffsets: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    jointPreRots: new Float32Array(16).fill(0.5),
    jointParents: new Int32Array([-1, 0, 1, 1]),
    inverses: new Float32Array(64).map((_, i) => i),
    skinIdx: new Uint16Array(32).map((_, i) => i % J),
    skinW: new Float32Array(32).fill(0.25),
    neutralNrm: new Float32Array(24).map((_, i) => i * 0.1),
  };
  const out = buildRigBuffer(
    layout, { jointCount: J, vertCount: V, srcVertCount: V1 }, src,
  );
  const o = layout.offsets;
  assert.deepEqual(Array.from(out.slice(o.jointOffsets, o.jointOffsets + 3)), [1, 2, 3]);
  assert.equal(out[o.jointParents], -1);
  assert.equal(out[o.jointParents + 2], 1);
  assert.equal(out[o.inverses + 63], 63);
  assert.equal(out[o.skinIdx + 5], 1);
  assert.equal(out[o.skinW + 31], 0.25);
  assert.ok(Math.abs(out[o.neutralNrm + 23] - 2.3) < 1e-6);
});

test("body buffer strips w components", () => {
  const layout = computeBodyLayout({ srcVertCount: V1, shapeCount: B });
  const neutral4 = new Float32Array(V1 * 4).map((_, i) => i);
  const deltas4 = new Float32Array(V1 * B * 4).map((_, i) => i + 1000);
  const out = buildBodyBuffer(layout, { srcVertCount: V1, shapeCount: B }, { neutral4, deltas4 });
  assert.deepEqual(Array.from(out.slice(0, 3)), [0, 1, 2]);
  // vertex 1, shape 2, xyz
  const d = layout.offsets.deltas + (1 * B + 2) * 3;
  const s = (1 * B + 2) * 4;
  assert.deepEqual(
    Array.from(out.slice(d, d + 3)),
    [1000 + s, 1001 + s, 1002 + s],
  );
});

test("lod buffer builds identity for null mapping", () => {
  const layout = computeLodLayout({ targetCount: V });
  const out = buildLodBuffer(
    layout, { targetCount: V, sourceCount: V1, mapping: null },
  );
  assert.equal(out[layout.offsets.indices + 3 * 5], 5);
  assert.equal(out[layout.offsets.indices + 3 * 5 + 1], 5);
  assert.equal(out[layout.offsets.indices + 3 * 5 + 2], 5);
  assert.deepEqual(
    Array.from(out.slice(layout.offsets.barycentric + 3 * 5, layout.offsets.barycentric + 3 * 6)),
    [1, 0, 0],
  );
});

test("lod buffer passes explicit mappings through", () => {
  const layout = computeLodLayout({ targetCount: V });
  const mapping = {
    sourceIndices: new Uint32Array(V * 3).map((_, i) => (i * 7) % V1),
    barycentric: new Float32Array(V * 3).fill(0.5),
  };
  const out = buildLodBuffer(
    layout, { targetCount: V, sourceCount: V1, mapping },
  );
  assert.equal(out[layout.offsets.indices + 10], (10 * 7) % V1);
  assert.equal(out[layout.offsets.barycentric + 10], 0.5);
});

test("neutral outputs gather through the mapping", () => {
  const neutral4 = new Float32Array(V1 * 4);
  for (let v = 0; v < V1; v++) {
    neutral4[v * 4] = v;
    neutral4[v * 4 + 1] = v * 10;
    neutral4[v * 4 + 2] = v * 100;
    neutral4[v * 4 + 3] = 1;
  }
  const mapping = {
    sourceIndices: new Uint32Array([0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    barycentric: new Float32Array([0.5, 0.25, 0.25, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]),
  };
  const pos = neutralOutputPositions({
    targetCount: V, sourceCount: V1, mapping, neutral4,
  });
  assert.equal(pos[0], 0 * 0.5 + 1 * 0.25 + 2 * 0.25);
  assert.equal(pos[1], 0 * 0.5 + 10 * 0.25 + 20 * 0.25);
  const nrmIn = new Float32Array(V1 * 3).map((_, i) => i);
  const nrm = neutralOutputNormals({
    targetCount: V, sourceCount: V1, mapping, neutralNrm: nrmIn,
  });
  assert.equal(nrm[0], 0 * 0.5 + 3 * 0.25 + 6 * 0.25);
});

test("dynamic buffer allocators size correctly", () => {
  assert.equal(allocThetaBuffer(GPU_MAX_AGENTS, J).length, GPU_MAX_AGENTS * J * 7);
  assert.equal(allocWeightBuffer(GPU_MAX_AGENTS, B).length, GPU_MAX_AGENTS * B);
  assert.equal(allocShapeBuffer(GPU_MAX_AGENTS).length, GPU_MAX_AGENTS);
  assert.equal(allocBoneMega(GPU_MAX_AGENTS, J).length, GPU_MAX_AGENTS * J * 16);
  assert.equal(allocOutputMega(GPU_MAX_AGENTS, V).length, GPU_MAX_AGENTS * V * 3);
});

test("slot index bias offsets into the mega slice", () => {
  const anchor = new Uint16Array([0, 5, 7, 3]);
  const biased = biasSlotIndex(anchor, 3, V);
  assert.ok(biased instanceof Uint32Array);
  assert.deepEqual(Array.from(biased), [3 * V, 5 + 3 * V, 7 + 3 * V, 3 + 3 * V]);
  assert.deepEqual(Array.from(biasSlotIndex(anchor, 0, V)), [0, 5, 7, 3]);
});
