import test from "node:test";
import assert from "node:assert/strict";
import { buildBatchedKernels } from "../../static/js/gpu/kernels.js";
import {
  computeBodyLayout,
  computeLodLayout,
  computeRigLayout,
} from "../../static/js/gpu/layouts.js";

const WEBGPU_URL = new URL(
  "../../node_modules/three/build/three.webgpu.js",
  import.meta.url,
);

const S = 4;
const J = 4;
const V = 8;
const V1 = 8;
const B = 3;

async function buildTestKernels() {
  const W = await import(WEBGPU_URL);
  const TSL = W.TSL;
  assert.ok(TSL, "three/webgpu must export the TSL namespace");
  const SBA = W.StorageBufferAttribute;
  const rigLayout = computeRigLayout({ jointCount: J, vertCount: V, srcVertCount: V1 });
  const bodyLayout = computeBodyLayout({ srcVertCount: V1, shapeCount: B });
  const lodLayout = computeLodLayout({ targetCount: V });
  const io = {
    rig: new SBA(new Float32Array(rigLayout.totalFloats), 1),
    rigCount: rigLayout.totalFloats,
    body: new SBA(new Float32Array(bodyLayout.totalFloats), 1),
    bodyCount: bodyLayout.totalFloats,
    theta: new SBA(new Float32Array(S * J * 7), 1),
    thetaCount: S * J * 7,
    weights: new SBA(new Float32Array(S * B), 1),
    weightCount: S * B,
    shape: new SBA(new Float32Array(S), 1),
    bones: new SBA(new Float32Array(S * J * 16), 1),
    outPos: new SBA(new Float32Array(S * V * 3), 3),
    outNormal: new SBA(new Float32Array(S * V * 3), 3),
    lod: new SBA(new Float32Array(lodLayout.totalFloats), 1),
    lodCount: lodLayout.totalFloats,
    uFkTotal: TSL.uniform(0),
    uSkinRowW: TSL.uniform(0),
    uSkinTotal: TSL.uniform(0),
    dims: {
      S,
      J,
      V,
      V1,
      B,
      rig: rigLayout.offsets,
      body: bodyLayout.offsets,
      lod: lodLayout.offsets,
    },
  };
  return buildBatchedKernels(TSL, io);
}

test("TSL surface used by the kernels exists", async () => {
  const W = await import(WEBGPU_URL);
  const TSL = W.TSL;
  for (const name of [
    "Fn", "If", "Loop", "float", "int", "sin", "cos", "pow", "sqrt",
    "vec3", "storage", "uniform", "globalId", "compute",
  ]) {
    assert.ok(TSL[name] !== undefined, `TSL.${name} must exist`);
  }
  assert.equal(typeof W.StorageBufferAttribute, "function");
});

test("batched kernels build to count-free compute nodes", async () => {
  // Stage C passes explicit dispatch sizes per tick, so the nodes carry no
  // baked count; both use the default 64-thread workgroup.
  const { fkAll, skinAll } = await buildTestKernels();
  assert.equal(fkAll.isComputeNode, true);
  assert.equal(skinAll.isComputeNode, true);
  assert.equal(fkAll.count, null);
  assert.equal(skinAll.count, null);
  assert.equal(fkAll.workgroupSize[0], 64);
  assert.equal(skinAll.workgroupSize[0], 64);
});

test("every TSL method used by the kernel bodies exists", async () => {
  // Fn bodies are lazy: construction alone never executes them, so exercise
  // each method name kernels.js relies on against real nodes instead.
  const W = await import(WEBGPU_URL);
  const TSL = W.TSL;
  const a = TSL.float(1).toVar();
  const b = TSL.float(2).toVar();
  const i = TSL.int(TSL.globalId.x).toVar();
  assert.ok(typeof a.add(b).sub(b).mul(b).div(b) === "object");
  assert.ok(typeof i.add(TSL.int(1)).mul(TSL.int(2)) === "object");
  assert.ok(typeof i.div(TSL.int(4)) === "object");
  assert.ok(typeof i.lessThan(TSL.int(8)) === "object");
  assert.ok(typeof TSL.int(TSL.globalId.y) === "object");
  assert.ok(typeof i.greaterThanEqual(TSL.int(0)) === "object");
  assert.ok(typeof a.greaterThan(TSL.float(0)) === "object");
  assert.ok(typeof a.notEqual(TSL.float(0)) === "object");
  assert.ok(typeof TSL.sin(a) === "object");
  assert.ok(typeof TSL.cos(a) === "object");
  assert.ok(typeof TSL.pow(TSL.float(2), a) === "object");
  assert.ok(typeof TSL.sqrt(a) === "object");
  assert.ok(typeof TSL.vec3(a, b, a) === "object");
  const store = TSL.storage(new W.StorageBufferAttribute(new Float32Array(8), 1), "float", 8);
  assert.ok(typeof store.element(i).assign === "function");
  a.assign(b);
  a.addAssign(b);
  // NOTE: If/Loop/Fn bodies only execute inside a shader build (browser);
  // their names are pinned by the surface test above, their behavior by the
  // browser run.
});
