import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import {
  MIXAMO_JOINT_MAP,
  MixamoClip,
  canonicalMixamoName,
} from "../../static/js/mixamo.js";

test("joint map drives 52 unique joints, left/right symmetric", () => {
  const values = Object.values(MIXAMO_JOINT_MAP);
  assert.equal(values.length, 52);
  assert.equal(new Set(values).size, values.length, "each joint driven once");
  const left = values.filter((v) => v.startsWith("l_")).length;
  const right = values.filter((v) => v.startsWith("r_")).length;
  assert.equal(left, right);
  assert.ok(left > 0);
});

test("canonical names tolerate namespaces and stripped prefixes", () => {
  assert.equal(canonicalMixamoName("mixamorig:Hips"), "mixamorigHips");
  assert.equal(canonicalMixamoName("mixamorigHips"), "mixamorigHips");
  assert.equal(canonicalMixamoName("Hips"), "mixamorigHips");
  assert.equal(canonicalMixamoName("LeftEye"), "LeftEye");
  assert.equal(canonicalMixamoName("SomethingElse"), "SomethingElse");
});

function syntheticRig() {
  const root = new THREE.Group();
  const hips = new THREE.Bone();
  hips.name = "mixamorigHips";
  const spine = new THREE.Bone();
  spine.name = "mixamorigSpine";
  root.add(hips);
  hips.add(spine);
  return { root, hips, spine };
}

test("sampling writes first-frame-relative ZYX eulers into thetaJ", () => {
  const { root } = syntheticRig();
  const q0 = new THREE.Quaternion();
  const q1 = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI / 2, 0));
  const track = new THREE.QuaternionKeyframeTrack(
    "mixamorigHips.quaternion",
    [0, 1],
    [...q0.toArray(), ...q1.toArray()],
  );
  const animation = new THREE.AnimationClip("test", 1, [track]);
  const jointNames = ["body_world", "root", "c_spine1"];
  const clip = new MixamoClip("test.fbx", root, animation, jointNames);
  assert.equal(clip.mappedCount, 2);
  const player = clip.createPlayer();
  const thetaJ = new Float32Array(jointNames.length * 7);
  player.update(0.5, thetaJ);
  // Halfway through a 0→90° yaw: rx/rz ~ 0, ry ~ 45°.
  assert.ok(Math.abs(thetaJ[1 * 7 + 3]) < 1e-4, `rx ~ 0, got ${thetaJ[1 * 7 + 3]}`);
  assert.ok(
    Math.abs(thetaJ[1 * 7 + 4] - Math.PI / 4) < 1e-3,
    `ry ~ 45°, got ${thetaJ[1 * 7 + 4]}`,
  );
  assert.ok(Math.abs(thetaJ[1 * 7 + 5]) < 1e-4, `rz ~ 0, got ${thetaJ[1 * 7 + 5]}`);
  // Unanimated spine holds zeros.
  for (let k = 3; k < 6; k++) {
    assert.ok(Math.abs(thetaJ[2 * 7 + k]) < 1e-6, `spine channel ${k} ~ 0`);
  }
  player.dispose();
});

test("rest offset is stripped (animation matching rest = no-op)", () => {
  const { root, hips } = syntheticRig();
  const rest = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI / 6, 0));
  hips.quaternion.copy(rest);
  const track = new THREE.QuaternionKeyframeTrack(
    "mixamorigHips.quaternion",
    [0, 1],
    [...rest.toArray(), ...rest.toArray()],
  );
  const animation = new THREE.AnimationClip("rest", 1, [track]);
  const clip = new MixamoClip("rest.fbx", root, animation, ["body_world", "root"]);
  const player = clip.createPlayer();
  const thetaJ = new Float32Array(2 * 7);
  player.update(0.5, thetaJ);
  assert.ok(Math.abs(thetaJ[7 + 3]) < 1e-4);
  assert.ok(Math.abs(thetaJ[7 + 4]) < 1e-4);
  assert.ok(Math.abs(thetaJ[7 + 5]) < 1e-4);
  player.dispose();
});

test("clip exposes animation metadata", () => {
  const { root } = syntheticRig();
  const tip = new THREE.Bone();
  tip.name = "mixamorigHeadTop_End";
  root.add(tip);
  const q = new THREE.Quaternion();
  const track = new THREE.QuaternionKeyframeTrack(
    "mixamorigHips.quaternion",
    [0, 0.5, 1],
    [...q.toArray(), ...q.toArray(), ...q.toArray()],
  );
  const animation = new THREE.AnimationClip("mixamo.com", 1, [track]);
  const clip = new MixamoClip("dance.fbx", root, animation, ["body_world", "root", "c_spine1"]);
  const md = clip.getMetadata();
  assert.equal(md.name, "dance.fbx");
  assert.equal(md.takeName, "mixamo.com");
  assert.equal(md.duration, 1);
  assert.equal(md.trackCount, 1);
  assert.equal(md.keyCount, 3);
  assert.equal(md.fps, 2);
  assert.equal(md.boneCount, 3);
  assert.equal(md.mappedCount, 2);
  assert.deepEqual(md.unmapped, ["mixamorigHeadTop_End"]);
  assert.equal(md.restHipsY, 0);
});

test("channels resolve late when joints arrive after parse", () => {
  const { root } = syntheticRig();
  const animation = new THREE.AnimationClip("x", 1, []);
  const clip = new MixamoClip("late.fbx", root, animation);
  assert.equal(clip.mappedCount, 0);
  assert.deepEqual(clip.unmappedSourceBones, []);
  assert.equal(clip.getMetadata().fps, null);
  clip.resolveChannels(["body_world", "root", "c_spine1"]);
  assert.equal(clip.mappedCount, 2);
});

test("euler channels hold branch on genuine wraps (roll sweep)", () => {
  // Continuous roll 170° → 200°: raw atan2 output jumps at ±180°; the
  // player must emit a branch-stable 0° → 30° ramp (first-frame-relative:
  // the 170° first frame is the reference, not a baked offset).
  const root = new THREE.Group();
  const hips = new THREE.Bone();
  hips.name = "mixamorigHips";
  root.add(hips);
  const roll = (deg) => new THREE.Quaternion().setFromEuler(
    new THREE.Euler((deg * Math.PI) / 180, 0, 0),
  );
  const track = new THREE.QuaternionKeyframeTrack(
    "mixamorigHips.quaternion",
    [0, 1],
    [...roll(170).toArray(), ...roll(200).toArray()],
  );
  const animation = new THREE.AnimationClip("sweep", 1, [track]);
  const clip = new MixamoClip("sweep.fbx", root, animation, ["body_world", "root"]);
  const player = clip.createPlayer({ loop: false });
  const thetaJ = new Float32Array(2 * 7);
  let maxStep = 0;
  let prevX = null;
  for (let i = 0; i < 20; i++) {
    player.update(0.05, thetaJ);
    const x = thetaJ[7 + 3];
    if (prevX !== null) maxStep = Math.max(maxStep, Math.abs(x - prevX));
    prevX = x;
  }
  assert.ok(maxStep < 0.2, `ramp must not jump, maxStep=${maxStep}`);
  const end = new THREE.Euler(thetaJ[7 + 3], thetaJ[7 + 4], thetaJ[7 + 5], "ZYX");
  const got = new THREE.Quaternion().setFromEuler(end);
  const expected = roll(170).invert().multiply(roll(200));
  assert.ok(Math.abs(got.dot(expected)) > 0.999, "end pose reconstructs relative motion");
  player.dispose();
});

test("loop closure detection (+ rest preserved by the probe)", () => {
  const root = new THREE.Group();
  const hips = new THREE.Bone();
  hips.name = "mixamorigHips";
  root.add(hips);
  const quat = (deg) => new THREE.Quaternion().setFromEuler(
    new THREE.Euler(0, (deg * Math.PI) / 180, 0),
  );
  const mk = (endDeg, tag) => {
    const track = new THREE.QuaternionKeyframeTrack(
      "mixamorigHips.quaternion",
      [0, 1],
      [...quat(0).toArray(), ...quat(endDeg).toArray()],
    );
    return new MixamoClip(tag, root, new THREE.AnimationClip(tag, 1, [track]), ["body_world", "root"]);
  };
  const closed = mk(0, "closed");
  assert.equal(closed.isLooping, true);
  assert.ok(closed.loopGapDeg < 1, `gap ${closed.loopGapDeg}`);
  const open = mk(90, "open");
  assert.equal(open.isLooping, false);
  assert.ok(open.loopGapDeg > 80, `gap ${open.loopGapDeg}`);
  const identity = new THREE.Quaternion();
  assert.ok(hips.quaternion.angleTo(identity) < 1e-6, "source rest preserved");
});
