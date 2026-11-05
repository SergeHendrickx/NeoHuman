import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import {
  GAIT_PARAMS,
  GaitController,
} from "../../static/js/sim/GaitController.js";

// Small synthetic rig mapping: 8 joints, enough columns for the gait params.
const J = 8;
const C = 70;

function syntheticMapping() {
  const TpMatrix = new Float32Array(J * 7 * C);
  // A few known-nonzero entries on gait columns.
  TpMatrix[(3 * 7 + 4) * C + GAIT_PARAMS.spineTwist] = 0.5;
  TpMatrix[(5 * 7 + 5) * C + GAIT_PARAMS.rKneeBend] = -0.25;
  TpMatrix[(2 * 7 + 0) * C + GAIT_PARAMS.lUplegPitch] = 1.5;
  return {
    TpMatrix,
    tpCols: C,
    jointCount: J,
    jointOffsets: new Float32Array(J * 3),
    jointPreRotations: new Float32Array(J * 4).map((_, i) => (i % 4 === 3 ? 1 : 0)),
  };
}

test("gaitValues returns the 11 locomotion params", () => {
  const values = GaitController.gaitValues(Math.PI / 3, 0.8);
  assert.equal(Object.keys(values).length, 11);
  for (const p of Object.values(GAIT_PARAMS)) {
    assert.ok(Number.isFinite(values[p]), `param ${p} must be finite`);
  }
  // Idle (speed 0): elbows rest at 0.18, legs at 0.
  const idle = GaitController.gaitValues(1.0, 0.0);
  assert.equal(idle[GAIT_PARAMS.rElbowBend], 0.18);
  assert.equal(idle[GAIT_PARAMS.rKneeBend], 0);
});

test("accumulateThetaJ matches direct Tp column math", () => {
  const mapping = syntheticMapping();
  const thetaJ = new Float32Array(J * 7);
  const values = GaitController.gaitValues(0.7, 0.9);
  GaitController.accumulateThetaJ(values, thetaJ, mapping);
  const expected = new Float32Array(J * 7);
  for (const [paramIdx, val] of Object.entries(values)) {
    if (Math.abs(val) < 1e-5) continue;
    const p = Number(paramIdx);
    for (let row = 0; row < J * 7; row++) {
      const w = mapping.TpMatrix[row * C + p];
      if (Math.abs(w) > 1e-5) expected[row] += w * val;
    }
  }
  assert.deepEqual(Array.from(thetaJ), Array.from(expected));
  // And the known entries actually fired.
  assert.notEqual(thetaJ[3 * 7 + 4], 0);
});

test("evaluateAndApply still poses bones end to end", () => {
  const mapping = syntheticMapping();
  const bones = Array.from({ length: J }, () => new THREE.Bone());
  const boneIndexMap = bones.map((_, i) => i);
  const thetaJ = new Float32Array(J * 7);
  GaitController.evaluateAndApply(Math.PI / 4, 1.0, bones, boneIndexMap, mapping, thetaJ);
  assert.ok(thetaJ.some((v) => v !== 0), "thetaJ must be nonzero");
  // Joint 3 got a nonzero rotation channel, so its bone quaternion moved.
  const q = bones[3].quaternion;
  assert.ok(
    Math.abs(q.x) + Math.abs(q.y) + Math.abs(q.z) > 1e-6,
    "active joint bone must rotate",
  );
});
