import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { MH_CHAINS, retargetPoseToTheta } from "../../static/js/poses.js";

test("chain table covers the body, left/right symmetric", () => {
  assert.equal(MH_CHAINS.length, 7 + 24 * 2);
  const byMhr = new Map(MH_CHAINS.map((c) => [c.mhr, c.src]));
  for (const core of [
    "root",
    "c_spine0", "c_spine1", "c_spine2", "c_spine3",
    "c_neck", "c_head",
  ]) {
    assert.ok(byMhr.has(core), `missing ${core}`);
  }
  for (const side of ["l_", "r_"]) {
    for (const joint of [
      "upleg", "lowleg", "foot", "ball", "clavicle", "uparm", "lowarm",
      "wrist", "eye", "thumb1", "index1", "middle1", "ring1", "pinky1",
    ]) {
      assert.ok(byMhr.has(side + joint), `missing ${side}${joint}`);
    }
  }
  // Symmetry: every l_ chain mirrors an r_ chain source-for-source.
  for (const { mhr, src } of MH_CHAINS) {
    if (!mhr.startsWith("l_")) continue;
    const twin = byMhr.get("r_" + mhr.slice(2));
    assert.deepEqual(
      twin,
      src.map((n) => n.replace(".L", ".R")),
      `${mhr} has no R twin`,
    );
  }
});

test("composed chains multiply parent-first, bob is decimeters to cm", () => {
  const qx = (deg) => new THREE.Quaternion().setFromEuler(
    new THREE.Euler(THREE.MathUtils.degToRad(deg), 0, 0),
  );
  const pose = {
    quats: new Map([
      ["spine05", qx(30)],
      ["spine04", qx(30)],
      ["head", qx(90)],
      ["root", new THREE.Quaternion()],
    ]),
    rootMotionY: 1.5,
  };
  const jointIndexByName = new Map([["root", 1], ["c_spine0", 2], ["c_head", 3]]);
  const { theta, joints } = retargetPoseToTheta(pose, jointIndexByName, 4);
  assert.deepEqual([...joints].sort((a, b) => a - b), [1, 2, 3]);
  const deg = (rad) => THREE.MathUtils.radToDeg(rad);
  assert.ok(Math.abs(deg(theta[2 * 7 + 3]) - 60) < 1e-4, "spine composes 30+30");
  assert.ok(Math.abs(deg(theta[3 * 7 + 3]) - 90) < 1e-4, "head takes 90");
  assert.equal(theta[1 * 7 + 1], 15, "1.5dm bob = 15cm");
  assert.equal(theta[1 * 7 + 0], 0, "no X translation");
  assert.equal(theta[0 * 7 + 3], 0, "unmapped joints stay zero");
});


test("direction-exact solve hits source dirs despite rest mismatch", () => {
  const qx = (deg) => new THREE.Quaternion().setFromEuler(
    new THREE.Euler(THREE.MathUtils.degToRad(deg), 0, 0),
  );
  // MH: root -> spine05 -> spine04 -> spine03, Z-up bind.
  const parents = new Map([
    ["root", null],
    ["spine05", "root"],
    ["spine04", "spine05"],
    ["spine03", "spine04"],
  ]);
  const Z = [0, 0, 1];
  const pose = {
    quats: new Map([[
      "root", new THREE.Quaternion()], ["spine05", qx(90)],
    ]),
    parents,
    offsets: new Map([
      ["root", [0, 0, 0]], ["spine05", Z], ["spine04", Z], ["spine03", Z],
    ]),
    joints: ["root", "spine05", "spine04", "spine03"],
    rootMotionY: 0,
    rootMotion: [0, 0, 2],
  };
  // Ours: root -> c_spine0 -> c_spine1, Y-up bind (deliberately NOT
  // axis-matching MH: the solver must still hit exact directions).
  const jointIndexByName = new Map([["root", 0], ["c_spine0", 1], ["c_spine1", 2]]);
  const I4 = [0, 0, 0, 1];
  const rig = {
    jointCount: 3,
    pre: new Float32Array([...I4, ...I4, ...I4]),
    parents: new Int32Array([-1, 0, 1]),
    neutral: new Float32Array([...I4, ...I4, ...I4]),
    order: [0, 1, 2],
  };
  const offsets = new Float32Array([0, 0, 0, 0, 1, 0, 0, 1, 0]);
  const baseTheta = new Float32Array(3 * 7);
  const { theta, joints } = retargetPoseToTheta(pose, jointIndexByName, 3, {
    targetRig: rig, baseTheta, offsets,
  });
  assert.deepEqual([...joints].sort((a, b) => a - b), [0, 1, 2]);
  assert.equal(theta[0 * 7 + 1], 20, "MH-Z bob in cm");
  // FK our result and compare bone dirs against MH dirs in our frame.
  const P = [];
  const e = new THREE.Euler(0, 0, 0, "ZYX"), q = new THREE.Quaternion(), v = new THREE.Vector3();
  const Q = [];
  for (const ji of rig.order) {
    const b = ji * 7, p = rig.parents[ji];
    q.identity();
    e.set(theta[b + 3], theta[b + 4], theta[b + 5]);
    q.multiply(new THREE.Quaternion().setFromEuler(e));
    v.set(offsets[ji * 3], offsets[ji * 3 + 1], offsets[ji * 3 + 2]);
    if (p >= 0) { v.applyQuaternion(Q[p]).add(P[p]); q.premultiply(Q[p]); }
    P[ji] = v.clone(); Q[ji] = q.clone();
  }
  // MH spine04->spine03 dir: R_x(90) * +Z = -Y (MH). In our frame
  // (MH_Y -> our -Z): (0,-1,0) -> (0,0,1) = +Z.
  const got = P[2].clone().sub(P[1]).normalize();
  assert.ok(got.distanceTo(new THREE.Vector3(0, 0, 1)) < 1e-4, `spine dir ${got.toArray()}`);
  // Root took no rotation (identity delta), leaf continued its parent.
  const deg = (rad) => THREE.MathUtils.radToDeg(rad);
  assert.ok(Math.abs(deg(theta[0 * 7 + 3])) < 1e-3, "root stays");
  assert.ok(Math.abs(deg(theta[2 * 7 + 3])) < 1e-3, "leaf continues parent");
});
test("head ignores eye descendants and keeps level orientation", () => {
  // Eyes are side-feature offsets, not bone directions: aiming the head
  // via them pitches it systematically (face to the sky on every pose).
  // The head must take the orientation rule instead.
  const I = () => new THREE.Quaternion();
  const pose = {
    quats: new Map([["root", I()], ["neck03", I()], ["head", I()]]),
    parents: new Map([
      ["root", null],
      ["neck03", "root"],
      ["head", "neck03"],
      ["eye.R", "head"],
    ]),
    offsets: new Map([
      ["root", [0, 0, 0]], ["neck03", [0, 0, 1]], ["head", [0, 0, 1]],
      // MH eye: forward of the head (MH +Y is forward).
      ["eye.R", [0.3, 1, 0.2]],
    ]),
    joints: ["root", "neck03", "head", "eye.R"],
    rootMotionY: 0,
    rootMotion: [0, 0, 0],
  };
  const jointIndexByName = new Map([["root", 0], ["c_neck", 1], ["c_head", 2], ["r_eye", 3]]);
  const I4 = [0, 0, 0, 1];
  const rig = {
    jointCount: 4,
    pre: new Float32Array([...I4, ...I4, ...I4, ...I4]),
    parents: new Int32Array([-1, 0, 1, 2]),
    neutral: new Float32Array([...I4, ...I4, ...I4, ...I4]),
    order: [0, 1, 2, 3],
  };
  // Our eye sits sideways-dominant off the head, like real anatomy —
  // ~70 degrees away from the MH eye direction, so aiming via the eye
  // would visibly pitch the head.
  const offsets = new Float32Array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0.2, 0.1]);
  const baseTheta = new Float32Array(4 * 7);
  const { theta, joints } = retargetPoseToTheta(pose, jointIndexByName, 4, {
    targetRig: rig, baseTheta, offsets,
  });
  assert.deepEqual([...joints].sort((a, b) => a - b), [0, 1, 2, 3]);
  const e = new THREE.Euler(0, 0, 0, "ZYX"), q = new THREE.Quaternion();
  const Q = [];
  for (const ji of rig.order) {
    const b = ji * 7, p = rig.parents[ji];
    q.identity();
    e.set(theta[b + 3], theta[b + 4], theta[b + 5]);
    q.multiply(new THREE.Quaternion().setFromEuler(e));
    if (p >= 0) q.premultiply(Q[p]);
    Q[ji] = q.clone();
  }
  // Level MH head -> level our head (identity neutral here).
  const tilt = 2 * Math.acos(Math.min(1, Math.abs(Q[2].dot(new THREE.Quaternion()))));
  assert.ok(tilt < 1e-3, `head tilted ${THREE.MathUtils.radToDeg(tilt).toFixed(1)}deg`);
});
