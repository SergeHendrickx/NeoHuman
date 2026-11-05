import test from "node:test";
import assert from "node:assert/strict";
import { formatDeltaMm, mirrorPairs, strongestVertex } from "../../static/js/exprVectorMath.js";

test("mirrorPairs pairs L/R variants of one base", () => {
  assert.deepEqual(
    mirrorPairs(["eyeBlink_L", "eyeBlink_R", "jawDrop"]),
    [1, 0, 2],
  );
});

test("mirrorPairs matches on non-side suffix letters", () => {
  assert.deepEqual(
    mirrorPairs(["lipFunneler_LB", "lipFunneler_RB", "chinRaise_B"]),
    [1, 0, 2],
  );
});

test("mirrorPairs leaves generic and unmatched names solo", () => {
  assert.deepEqual(
    mirrorPairs(["expression_01", "expression_02", "browRaise_L", "jawDrop"]),
    [0, 1, 2, 3],
  );
});

test("strongestVertex finds the argmax delta vertex", () => {
  const vCount = 3, totalBs = 2;
  const deltas = new Float32Array(vCount * totalBs * 4);
  const set = (v, b, x, y, z) => {
    const i = (v * totalBs + b) * 4;
    deltas[i] = x; deltas[i + 1] = y; deltas[i + 2] = z;
  };
  set(0, 1, 1, 0, 0);
  set(1, 1, 0, 2, 0);
  set(2, 1, 0, 0, 0);
  assert.equal(strongestVertex(deltas, vCount, totalBs, 1), 1);
  assert.equal(strongestVertex(deltas, vCount, totalBs, 0), -1);
});

test("formatDeltaMm signs every component", () => {
  assert.equal(formatDeltaMm([0.34, -1.26, 7.83]), "(+0.3, -1.3, +7.8)");
  assert.equal(formatDeltaMm([-0.04, 0, 2]), "(-0.0, +0.0, +2.0)");
});
