import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { MixamoClip, captureFirstFrameRest, setSpineBoost, setMixamoTargetRig } from "../../static/js/mixamo.js";

function quat(xDeg) {
  return new THREE.Quaternion().setFromEuler(new THREE.Euler(THREE.MathUtils.degToRad(xDeg), 0, 0));
}

function quatNear(a, b, eps = 1e-4) {
  return Math.abs(a.x - b.x) < eps && Math.abs(a.y - b.y) < eps &&
    Math.abs(a.z - b.z) < eps && Math.abs(a.w - b.w) < eps;
}

// Two-bone rig whose bind pose DIFFERS from the animation's first frame
// (the ActorCore case: bind A-pose vs animated reference).
function makeRig() {
  const root = new THREE.Group();
  const boneA = new THREE.Bone();
  boneA.name = "mixamorigLeftArm";
  boneA.quaternion.copy(quat(20)); // bind: 20 deg
  const boneB = new THREE.Bone();
  boneB.name = "mixamorigLeftForeArm";
  root.add(boneA);
  boneA.add(boneB);
  return { root, boneA, boneB };
}

function makeClip() {
  const q0 = quat(30); // first frame: 30 deg (NOT the bind pose)
  const q1 = quat(60);
  const track = new THREE.QuaternionKeyframeTrack(
    "mixamorigLeftArm.quaternion",
    [0, 1],
    [q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w],
  );
  return new THREE.AnimationClip("take", 1, [track]);
}

test("rest comes from the animation first frame, not the bind pose", () => {
  const { root, boneA, boneB } = makeRig();
  const rest = captureFirstFrameRest(root, makeClip());
  // Animated bone: rest == first-frame sample (30 deg), not bind (20 deg).
  assert.ok(quatNear(rest.get(boneA), quat(30)), "arm rest should be the 30deg first frame");
  // Unanimated bone: rest == bind pose (mixer leaves it untouched).
  assert.ok(quatNear(rest.get(boneB), new THREE.Quaternion()), "forearm rest should be bind");
});

test("channels sample the animated rig when bone names duplicate", () => {
  // ActorCore files nest a second, static rig inside the animated one
  // (129 bones, 65 unique names). The mixer binds tracks to the FIRST
  // traversal match; our channels must resolve the same bone, or the
  // player samples the static duplicate and emits zeros forever.
  const root = new THREE.Group();
  const hipsLive = new THREE.Bone();
  hipsLive.name = "mixamorigHips";
  const hipsDead = new THREE.Bone();
  hipsDead.name = "mixamorigHips";
  root.add(hipsLive);
  hipsLive.add(hipsDead); // nested duplicate, never animated
  const q0 = new THREE.Quaternion();
  const q1 = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
  const track = new THREE.QuaternionKeyframeTrack(
    "mixamorigHips.quaternion",
    [0, 1],
    [...q0.toArray(), ...q1.toArray()],
  );
  const clip = new MixamoClip("dup.fbx", root, new THREE.AnimationClip("dup", 1, [track]), ["root"]);
  const player = clip.createPlayer({ loop: false });
  const thetaJ = new Float32Array(7);
  player.update(1.0, thetaJ);
  const mag = Math.abs(thetaJ[3]) + Math.abs(thetaJ[4]) + Math.abs(thetaJ[5]);
  assert.ok(mag > 1.0, `expected ~90deg of motion, got ${mag} rad`);
  player.dispose();
});

test("hips vertical bob transfers to the root slot (cm + m files)", () => {
  // Rotations-only playback pins the root: crouches read as dangling
  // feet and the dance looks stiff. Hips Y relative to the first frame
  // rides the root slot instead (thetaJ translations are centimeters).
  const mk = (y0, y1) => {
    const root = new THREE.Group();
    const hips = new THREE.Bone();
    hips.name = "mixamorigHips";
    root.add(hips);
    const track = new THREE.VectorKeyframeTrack(
      "mixamorigHips.position", [0, 1], [0, y0, 0, 0, y1, 0],
    );
    const clip = new MixamoClip("bob.fbx", root,
      new THREE.AnimationClip("bob", 1, [track]), ["root"]);
    const player = clip.createPlayer({ loop: false });
    const thetaJ = new Float32Array(7);
    thetaJ[1] = 5; // authored base rides along (additive)
    player.update(1.0, thetaJ);
    const slot = thetaJ[1];
    player.dispose();
    return slot;
  };
  assert.ok(Math.abs(mk(100, 80) - (5 - 20)) < 1e-3, "cm file: -20 crouch");
  assert.ok(Math.abs(mk(1.0, 0.8) - (5 - 20)) < 1e-3, "meter file: -20 crouch");
});

test("spine boost exaggerates spine channels live, others untouched", () => {
  const root = new THREE.Group();
  const spine = new THREE.Bone();
  spine.name = "mixamorigSpine";
  const hips = new THREE.Bone();
  hips.name = "mixamorigHips";
  root.add(spine, hips);
  const q0 = new THREE.Quaternion();
  const q1 = new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 6, 0, 0));
  const mkTrack = (name) => new THREE.QuaternionKeyframeTrack(
    `${name}.quaternion`, [0, 1], [...q0.toArray(), ...q1.toArray()],
  );
  const clip = new MixamoClip("boost.fbx", root,
    new THREE.AnimationClip("boost", 1, [mkTrack("mixamorigSpine"), mkTrack("mixamorigHips")]),
    ["c_spine1", "root"]);
  const player = clip.createPlayer({ loop: false });
  const thetaJ = new Float32Array(14);
  player.update(1.0, thetaJ);
  assert.ok(Math.abs(thetaJ[3] - Math.PI / 6) < 1e-3, `unboosted spine ~30deg, got ${thetaJ[3]}`);
  setSpineBoost(2);
  player.update(0.0001, thetaJ); // clamped end pose, live gain
  assert.ok(Math.abs(thetaJ[3] - Math.PI / 3) < 1e-2, `boosted spine ~60deg, got ${thetaJ[3]}`);
  assert.ok(Math.abs(thetaJ[10] - Math.PI / 6) < 1e-3, `hips must stay ~30deg, got ${thetaJ[10]}`);
  player.dispose();
  setSpineBoost(1);
});

test("global match retargets world motion onto our rest frames", () => {
  // Local-relative transfer copies the source triple verbatim: when rest
  // conventions twist (our X-down-bone vs Mixamo), hinge bends leak onto
  // twist axes (frozen knees). Global match solves our local rotation from
  // OUR rest frames + the source WORLD delta, so the composed world result
  // equals our rest posed by the source motion. Here: source knee flexes
  // 50deg about its local X; our rig uses real MHR thigh pre-rotations.
  const PRE_U = [-0.5159, 0.4933, -0.4441, 0.5416]; // real r/knee-chain pre
  setMixamoTargetRig(
    new Float32Array([0, 0, 0, 1, ...PRE_U, 0, 0, 0, 1]),
    [-1, 0, 1],
  );
  try {
    const root = new THREE.Group();
    const hips = new THREE.Bone();
    hips.name = "mixamorigHips";
    const thigh = new THREE.Bone();
    thigh.name = "mixamorigLeftUpLeg";
    const shin = new THREE.Bone();
    shin.name = "mixamorigLeftLeg";
    root.add(hips);
    hips.add(thigh);
    thigh.add(shin);
    const flex = new THREE.QuaternionKeyframeTrack(
      "mixamorigLeftLeg.quaternion", [0, 1],
      [...new THREE.Quaternion().toArray(),
       ...new THREE.Quaternion().setFromEuler(
         new THREE.Euler(THREE.MathUtils.degToRad(50), 0, 0)).toArray()],
    );
    const clip = new MixamoClip("glob.fbx", root,
      new THREE.AnimationClip("glob", 1, [flex]), ["root", "l_upleg", "l_lowleg"]);
    const player = clip.createPlayer({ loop: false });
    const thetaJ = new Float32Array(21);
    player.update(1.0, thetaJ);
    // Forward-kinematics the WRITTEN eulers through our test rig.
    const fk = (ji) => {
      const b = ji * 7;
      const e = new THREE.Euler(thetaJ[b + 3], thetaJ[b + 4], thetaJ[b + 5], "ZYX");
      return new THREE.Quaternion().setFromEuler(e);
    };
    const preQ = (q) => new THREE.Quaternion(...q);
    const gRoot = preQ([0, 0, 0, 1]).multiply(fk(0));
    const gThigh = gRoot.clone().multiply(preQ(PRE_U)).multiply(fk(1));
    const gShin = gThigh.clone().multiply(preQ([0, 0, 0, 1])).multiply(fk(2));
    // 1. The knee visibly bends ~50deg in world (thigh-vs-shin angle change).
    const bend = gThigh.clone().invert().multiply(gShin);
    const bendDeg = (2 * Math.acos(Math.min(1, Math.abs(bend.w))) * 180) / Math.PI;
    assert.ok(Math.abs(bendDeg - 50) < 2, `knee should bend ~50deg, got ${bendDeg}`);
    // 2. World result equals our rest posed by the source delta (independent
    // path: neutral(lowleg) x flexion, no player internals).
    const expected = preQ(PRE_U).multiply(
      new THREE.Quaternion().setFromEuler(new THREE.Euler(THREE.MathUtils.degToRad(50), 0, 0)));
    assert.ok(gShin.angleTo(expected) < 0.05, `world result off by ${gShin.angleTo(expected)}`);
    player.dispose();
  } finally {
    setMixamoTargetRig(null);
  }
});

test("global match keeps the first frame's own pose (crouch survives)", () => {
  // First-frame-relative transfer drops the first frame's pose and only
  // conveys variation: a dance opening in a 30deg crouch plays back with
  // straight legs. Bind-relative transfer keeps absolute poses: ~30deg at
  // frame 0, ~80deg at the end (not 0/50deg relative).
  const PRE_U = [-0.5159, 0.4933, -0.4441, 0.5416];
  setMixamoTargetRig(
    new Float32Array([0, 0, 0, 1, ...PRE_U, 0, 0, 0, 1]),
    [-1, 0, 1],
    new Float32Array([0, 0, 0, 0, -1, 0, 0, -1, 0]),
  );
  try {
    const root = new THREE.Group();
    const hips = new THREE.Bone();
    hips.name = "mixamorigHips";
    const thigh = new THREE.Bone();
    thigh.name = "mixamorigLeftUpLeg";
    thigh.position.set(0, -1, 0);
    const shin = new THREE.Bone();
    shin.name = "mixamorigLeftLeg";
    shin.position.set(0, -1, 0);
    root.add(hips);
    hips.add(thigh);
    thigh.add(shin);
    const q = (deg) => new THREE.Quaternion().setFromEuler(
      new THREE.Euler(THREE.MathUtils.degToRad(deg), 0, 0));
    const flex = new THREE.QuaternionKeyframeTrack(
      "mixamorigLeftLeg.quaternion", [0, 1],
      [...q(30).toArray(), ...q(80).toArray()],
    );
    const clip = new MixamoClip("crouch.fbx", root,
      new THREE.AnimationClip("crouch", 1, [flex]), ["root", "l_upleg", "l_lowleg"]);
    const player = clip.createPlayer({ loop: false });
    const thetaJ = new Float32Array(21);
    const bendDeg = () => {
      const fk = (ji) => {
        const b = ji * 7;
        const e = new THREE.Euler(thetaJ[b + 3], thetaJ[b + 4], thetaJ[b + 5], "ZYX");
        return new THREE.Quaternion().setFromEuler(e);
      };
      const preQ = (a) => new THREE.Quaternion(...a);
      const gRoot = preQ([0, 0, 0, 1]).multiply(fk(0));
      const gThigh = gRoot.clone().multiply(preQ(PRE_U)).multiply(fk(1));
      const gShin = gThigh.clone().multiply(preQ([0, 0, 0, 1])).multiply(fk(2));
      const bend = gThigh.clone().invert().multiply(gShin);
      return (2 * Math.acos(Math.min(1, Math.abs(bend.w))) * 180) / Math.PI;
    };
    player.update(0.0001, thetaJ);
    assert.ok(Math.abs(bendDeg() - 30) < 2, `frame 0 should crouch ~30deg, got ${bendDeg()}`);
    player.update(1.0, thetaJ);
    assert.ok(Math.abs(bendDeg() - 80) < 2, `end should bend ~80deg absolute, got ${bendDeg()}`);
    player.dispose();
  } finally {
    setMixamoTargetRig(null);
  }
});

test("aligned transfer remaps flexion across twisted binds (no twist leak)", () => {
  // Source shin bind twisted 90deg about Y (Y-down rig vs our frame);
  // flexion runs about source-local X. The written euler must carry the
  // same 50deg about OUR corresponding axis (-Z here), not source-X and
  // not the untwisted world axis: world-orientation transfer and the
  // backwards conjugation both put it ~90deg off (frozen knees /
  // twisting feet in production).
  setMixamoTargetRig(
    new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]),
    [-1, 0, 1],
  );
  try {
    const root = new THREE.Group();
    const hips = new THREE.Bone();
    hips.name = "mixamorigHips";
    const thigh = new THREE.Bone();
    thigh.name = "mixamorigLeftUpLeg";
    const shin = new THREE.Bone();
    shin.name = "mixamorigLeftLeg";
    root.add(hips);
    hips.add(thigh);
    thigh.add(shin);
    const bind = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI / 2, 0));
    shin.quaternion.copy(bind);
    const at = (deg) => bind.clone().multiply(
      new THREE.Quaternion().setFromEuler(new THREE.Euler(THREE.MathUtils.degToRad(deg), 0, 0)));
    const flex = new THREE.QuaternionKeyframeTrack(
      "mixamorigLeftLeg.quaternion", [0, 1],
      [...at(0).toArray(), ...at(50).toArray()],
    );
    const clip = new MixamoClip("twist.fbx", root,
      new THREE.AnimationClip("twist", 1, [flex]), ["root", "l_upleg", "l_lowleg"]);
    const player = clip.createPlayer({ loop: false });
    const thetaJ = new Float32Array(21);
    player.update(0.0001, thetaJ);
    assert.ok(Math.abs(thetaJ[2 * 7 + 3]) < 0.035, `frame 0 ~ rest, got ${thetaJ[2 * 7 + 3]}`);
    player.update(1.0, thetaJ);
    const ex = thetaJ[2 * 7 + 3];
    const ey = thetaJ[2 * 7 + 4];
    const ez = thetaJ[2 * 7 + 5];
    assert.ok(Math.abs(ex) < 0.035, `no source-X leak, got ex=${ex}`);
    assert.ok(Math.abs(ey) < 0.035, `no twist leak, got ey=${ey}`);
    assert.ok(Math.abs(ez + Math.PI * 50 / 180) < 0.035, `50deg about -Z, got ez=${ez}`);
    player.dispose();
  } finally {
    setMixamoTargetRig(null);
  }
});

test("direction rule bends segments exactly across twisted binds", () => {
  // Production geometry, idealized: Y-down source with a 90deg-twisted
  // shin bind (flexion about source-local X) vs our X-down-bone frames
  // (segments along local X, bind pointing down like the source).
  // World-orientation transfer spins this 50deg flexion about our bone
  // axis (0deg bend: frozen knees); the direction rule matches segments,
  // so our shin bends the full 50deg from bind.
  const RZ90 = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, Math.PI / 2));
  const ID = [0, 0, 0, 1];
  setMixamoTargetRig(
    new Float32Array([...ID, ...RZ90.toArray(), ...ID, ...ID]),
    [-1, 0, 1, 2],
    new Float32Array([0, 0, 0, 0, -1, 0, -1, 0, 0, -1, 0, 0]),
  );
  try {
    const root = new THREE.Group();
    const mk = (name, y) => {
      const b = new THREE.Bone();
      b.name = name;
      b.position.set(0, y, 0);
      return b;
    };
    const hips = mk("mixamorigHips", 0);
    const thigh = mk("mixamorigLeftUpLeg", -1);
    const shin = mk("mixamorigLeftLeg", -1);
    const foot = mk("mixamorigLeftFoot", -1);
    root.add(hips);
    hips.add(thigh);
    thigh.add(shin);
    shin.add(foot);
    const bind = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, Math.PI / 2, 0));
    shin.quaternion.copy(bind);
    const at = (deg) => bind.clone().multiply(
      new THREE.Quaternion().setFromEuler(new THREE.Euler(THREE.MathUtils.degToRad(deg), 0, 0)));
    const flex = new THREE.QuaternionKeyframeTrack(
      "mixamorigLeftLeg.quaternion", [0, 1],
      [...at(0).toArray(), ...at(50).toArray()],
    );
    const clip = new MixamoClip("dirseg.fbx", root,
      new THREE.AnimationClip("dirseg", 1, [flex]),
      ["root", "l_upleg", "l_lowleg", "l_foot"]);
    const player = clip.createPlayer({ loop: false });
    const thetaJ = new Float32Array(28);
    player.update(1.0, thetaJ);
    // FK our written eulers over our pre-rotations + offsets; shin segment vs bind.
    const g = [];
    const pos = [new THREE.Vector3(0, 0, 0)];
    const offs = [[0, 0, 0], [0, -1, 0], [-1, 0, 0], [-1, 0, 0]];
    const pres = [
      new THREE.Quaternion(),
      RZ90.clone(),
      new THREE.Quaternion(),
      new THREE.Quaternion(),
    ];
    const parents = [-1, 0, 1, 2];
    for (let j = 0; j < 4; j++) {
      const b = j * 7;
      const e = new THREE.Euler(thetaJ[b + 3], thetaJ[b + 4], thetaJ[b + 5], "ZYX");
      const q = pres[j].clone().multiply(new THREE.Quaternion().setFromEuler(e));
      g[j] = j === 0 ? q : g[parents[j]].clone().multiply(q);
      if (j > 0) {
        const o = offs[j];
        pos[j] = pos[parents[j]].clone().add(
          new THREE.Vector3(o[0], o[1], o[2]).applyQuaternion(g[parents[j]]));
      }
    }
    const seg = pos[3].clone().sub(pos[2]).normalize();
    const bend = Math.acos(Math.min(1, Math.max(-1, seg.dot(new THREE.Vector3(0, -1, 0)))))
      * 180 / Math.PI;
    assert.ok(Math.abs(bend - 50) < 2, `shin segment should bend 50deg, got ${bend}`);
    player.dispose();
  } finally {
    setMixamoTargetRig(null);
  }
});

test("rest capture leaves the source rig at bind", () => {
  const { root, boneA, boneB } = makeRig();
  captureFirstFrameRest(root, makeClip());
  assert.ok(quatNear(boneA.quaternion, quat(20)), "arm bind pose not restored");
  assert.ok(quatNear(boneB.quaternion, new THREE.Quaternion()), "forearm bind pose not restored");
});
