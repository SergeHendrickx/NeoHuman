/**
 * MakeHuman BVH pose presets: static retarget onto any actor's rig.
 *
 * Mirrors the Mixamo clip conventions (rotations everywhere, first-frame-
 * relative hips-Y bob, X/Z steering-owned) but for single-frame BVH poses.
 * Transfer is direction-exact: ourGlobal(j) is solved parent-first so every
 * mapped bone points exactly where its MH source points (in the shared
 * MH_FRAME below), with minimal twist from our rest. Plain orientation
 * transfer cannot work here: our rest conventions differ per bone from
 * MH's (identical vs mirrored L/R pre-rotations, A-pose angles), so only
 * directions transfer losslessly. A naive local copy stays as fallback
 * when no target rig is available.
 * Unmapped joints keep their live base pose.
 *
 * Units: MakeHuman BVH is decimeters; thetaJ translations are centimeters.
 */

import * as THREE from "three";
import { BVHLoader } from "three/addons/loaders/BVHLoader.js";

// [MHR joint, MH source chain parent-first]. {S} expands per side:
// .L/.R on the MH side, l_/r_ on the MHR side. MH finger1 is the thumb
// (only digit without a metacarpal joint); 2..5 run index to pinky.
const SIDED = [
  ["{S}upleg", ["pelvis{S}", "upperleg01{S}", "upperleg02{S}"]],
  ["{S}lowleg", ["lowerleg01{S}", "lowerleg02{S}"]],
  ["{S}foot", ["foot{S}"]],
  ["{S}ball", ["toe1-1{S}"]],
  ["{S}clavicle", ["clavicle{S}"]],
  ["{S}uparm", ["shoulder01{S}", "upperarm01{S}", "upperarm02{S}"]],
  ["{S}lowarm", ["lowerarm01{S}", "lowerarm02{S}"]],
  ["{S}wrist", ["wrist{S}"]],
  ["{S}thumb1", ["finger1-1{S}"]],
  ["{S}thumb2", ["finger1-2{S}"]],
  ["{S}thumb3", ["finger1-3{S}"]],
  ["{S}index1", ["finger2-1{S}"]],
  ["{S}index2", ["finger2-2{S}"]],
  ["{S}index3", ["finger2-3{S}"]],
  ["{S}middle1", ["finger3-1{S}"]],
  ["{S}middle2", ["finger3-2{S}"]],
  ["{S}middle3", ["finger3-3{S}"]],
  ["{S}ring1", ["finger4-1{S}"]],
  ["{S}ring2", ["finger4-2{S}"]],
  ["{S}ring3", ["finger4-3{S}"]],
  ["{S}pinky1", ["finger5-1{S}"]],
  ["{S}pinky2", ["finger5-2{S}"]],
  ["{S}pinky3", ["finger5-3{S}"]],
  ["{S}eye", ["eye{S}"]],
];

const SINGLES = [
  ["root", ["root"]],
  ["c_spine0", ["spine05", "spine04"]],
  ["c_spine1", ["spine03"]],
  ["c_spine2", ["spine02"]],
  ["c_spine3", ["spine01"]],
  ["c_neck", ["neck01", "neck02", "neck03"]],
  ["c_head", ["head"]],
];

/**
 * Retarget chains: [{ mhr, src }] with src parent-first. Built once so the
 * joint map stays a reviewable table above.
 */
export const MH_CHAINS = (() => {
  const chains = SINGLES.map(([mhr, src]) => ({ mhr, src: [...src] }));
  for (const [mhrTpl, srcTpl] of SIDED) {
    for (const [mhSide, mhrSide] of [[".L", "l_"], [".R", "r_"]]) {
      chains.push({
        mhr: mhrTpl.replace("{S}", mhrSide),
        src: srcTpl.map((n) => n.replace("{S}", mhSide)),
      });
    }
  }
  return chains;
})();

/** All MH source joints referenced by the map (for coverage checks). */
export const MH_MAPPED_SOURCES = new Set(MH_CHAINS.flatMap((c) => c.src));

/**
 * Fixed MakeHuman -> viewer frame. MH BVH is Z-up with mirrored-side
 * conventions; we are Y-up. Found by searching all 24 proper axis
 * rotations for best bind-pose bone-direction agreement.
 * Columns are the images of MH +X/+Y/+Z in our space.
 */
export const MH_FRAME = new THREE.Quaternion().setFromRotationMatrix(
  new THREE.Matrix4().makeBasis(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 0, -1),
    new THREE.Vector3(0, 1, 0),
  ),
);

/**
 * Mapped joints that must never aim their parent's direction target. Eyes
 * are side-feature offsets, not bone directions: aiming the head via them
 * pitches it ~40 degrees on every pose (face to the sky). They stay solved
 * (orientation rule) - they just don't steer.
 */
const MH_NO_AIM = new Set(["l_eye", "r_eye"]);

export async function fetchPoseList() {
  const resp = await fetch("/api/pose/list");
  const data = await resp.json();
  return data.available ? (data.poses ?? []) : [];
}

/**
 * Load a preset by server id. Returns frame-0 local quats per MH joint plus
 * the root motion (decimeters, rest-relative) for the hips-Y bob.
 */
/** Encode a nested pose id (slashes stay, segments escape). */
export function encodePoseId(id) {
  return id.split("/").map(encodeURIComponent).join("/");
}

export async function loadPosePreset(id) {
  const resp = await fetch(`/api/pose/file/${encodePoseId(id)}`);
  if (!resp.ok) throw new Error(`pose ${id}: HTTP ${resp.status}`);
  return parsePoseText(id, await resp.text());
}

/** Parse BVH text into frame-0 pose data (offsets in decimeters). */
export function parsePoseText(id, text) {
  const { clip, skeleton } = new BVHLoader().parse(text);
  const quats = new Map();
  for (const track of clip.tracks) {
    if (!track.name.endsWith(".quaternion") || track.values.length < 4) continue;
    const bone = track.name.slice(0, -".quaternion".length);
    quats.set(bone, new THREE.Quaternion(
      track.values[0], track.values[1], track.values[2], track.values[3],
    ));
  }
  let rootMotionY = 0;
  const rootMotion = [0, 0, 0];
  const rootBone = skeleton.bones.find((b) => b.name === "root");
  const posTrack = clip.tracks.find((t) => t.name === "root.position");
  if (rootBone && posTrack && posTrack.values.length >= 3) {
    // Track positions are offset+motion; the bob is motion-only.
    rootMotionY = posTrack.values[1] - rootBone.position.y;
    rootMotion[0] = posTrack.values[0] - rootBone.position.x;
    rootMotion[1] = rootMotionY;
    rootMotion[2] = posTrack.values[2] - rootBone.position.z;
  }
  const parents = new Map();
  for (const bone of skeleton.bones) {
    parents.set(bone.name, bone.parent?.isBone ? bone.parent.name : null);
  }
  const offsets = new Map();
  for (const bone of skeleton.bones) {
    offsets.set(bone.name, [bone.position.x, bone.position.y, bone.position.z]);
  }
  return {
    id,
    quats,
    parents,
    offsets,
    rootMotionY,
    rootMotion,
    joints: skeleton.bones.map((b) => b.name),
    mapped: MH_CHAINS.filter((c) => c.src.every((n) => quats.has(n))).length,
    chains: MH_CHAINS.length,
  };
}

const _q = new THREE.Quaternion();
const _e = new THREE.Euler(0, 0, 0, "ZYX");
const _gPre = new THREE.Quaternion();
const _gNeu = new THREE.Quaternion();
const _gTmp = new THREE.Quaternion();
const _gOur = new THREE.Quaternion();
const _IDENTITY_Q = new THREE.Quaternion();
const _gA = new THREE.Quaternion();
const _vDir = new THREE.Vector3();
const _vBind = new THREE.Vector3();
const _MH_FRAME_INV = MH_FRAME.clone().invert();

/**
 * Retarget a loaded preset into thetaJ layout (7 floats/joint: mapped
 * rotations + root Y bob in centimeters). jointIndexByName maps MHR joint
 * names to indices; joints lists the actually written indices so the hold
 * layer never zeroes an unmapped base rotation.
 *
 * opts.targetRig/baseTheta/groupQuat/offsets select the direction-exact
 * path; without them it falls back to naive local transfer. baseTheta
 * is the authored base (unmapped joints rest-continue from it);
 * groupQuat is the actor's world orientation; offsets are our joint
 * offsets (centimeters) for bind segment directions.
 */
export function retargetPoseToTheta(pose, jointIndexByName, jointCount, opts = {}) {
  const { targetRig = null, baseTheta = null, groupQuat = null, offsets = null } = opts;
  if (targetRig && baseTheta && offsets) {
    return retargetGlobal(pose, jointIndexByName, jointCount, targetRig, baseTheta, groupQuat, offsets);
  }
  return retargetNaive(pose, jointIndexByName, jointCount);
}

function retargetNaive(pose, jointIndexByName, jointCount) {
  const theta = new Float32Array(jointCount * 7);
  const joints = [];
  for (const { mhr, src } of MH_CHAINS) {
    const joint = jointIndexByName.get(mhr);
    if (joint === undefined) continue;
    _q.identity();
    let ok = true;
    for (const name of src) {
      const q = pose.quats.get(name);
      if (!q) { ok = false; break; }
      _q.multiply(q);
    }
    if (!ok) continue;
    _e.setFromQuaternion(_q, "ZYX");
    const base = joint * 7;
    theta[base + 3] = _e.x;
    theta[base + 4] = _e.y;
    theta[base + 5] = _e.z;
    joints.push(joint);
  }
  writeRootBob(pose, jointIndexByName, theta);
  return { theta, joints };
}

function writeRootBob(pose, jointIndexByName, theta) {
  const root = jointIndexByName.get("root");
  if (root !== undefined) theta[root * 7 + 1] = pose.rootMotionY * 10;
}

function retargetGlobal(pose, jointIndexByName, jointCount, rig, baseTheta, groupQuat, offsets) {
  const theta = new Float32Array(jointCount * 7);
  const joints = [];
  const groupQ = groupQuat ?? _IDENTITY_Q;
  const namesByIndex = new Map();
  for (const [name, joint] of jointIndexByName) namesByIndex.set(joint, name);
  const rootIdx = jointIndexByName.get("root");
  // MH posed world positions (dm) and quats.
  const mhP = new Map(), mhQ = new Map();
  const mhSolve = (name) => {
    if (mhP.has(name)) return;
    const parent = pose.parents.get(name);
    if (parent) mhSolve(parent);
    const o = pose.offsets.get(name) || [0, 0, 0];
    const lq = pose.quats.get(name);
    const wq = parent ? mhQ.get(parent).clone() : new THREE.Quaternion();
    if (lq) wq.multiply(lq);
    const wp = new THREE.Vector3(o[0], o[1], o[2]);
    if (parent) wp.applyQuaternion(mhQ.get(parent)).add(mhP.get(parent));
    mhP.set(name, wp); mhQ.set(name, wq);
  };
  for (const name of pose.joints) mhSolve(name);
  // Our children lists + bind FK (zero eulers) for bind segment dirs.
  const kids = new Map();
  for (let j = 0; j < jointCount; j++) {
    const parent = rig.parents[j];
    if (parent < 0) continue;
    if (!kids.has(parent)) kids.set(parent, []);
    kids.get(parent).push(j);
  }
  const byMhr = new Map(MH_CHAINS.map((c) => [c.mhr, c.src[c.src.length - 1]]));
  const bindPos = [], bindWq = [];
  for (const ji of rig.order) {
    if (ji < 0 || ji >= jointCount) continue;
    const p = rig.parents[ji];
    _gTmp.set(rig.pre[ji * 4], rig.pre[ji * 4 + 1], rig.pre[ji * 4 + 2], rig.pre[ji * 4 + 3]);
    const wq = p >= 0 && bindWq[p] ? bindWq[p].clone().multiply(_gTmp) : _gTmp.clone();
    _vDir.set(offsets[ji * 3], offsets[ji * 3 + 1], offsets[ji * 3 + 2]);
    if (p >= 0 && bindPos[p]) _vDir.applyQuaternion(bindWq[p]).add(bindPos[p]);
    bindPos[ji] = _vDir.clone(); bindWq[ji] = wq;
  }
  const stepDown = (from, toward) => {
    let n = toward, prev = null;
    const guard = new Set();
    while (n && n !== from && !guard.has(n)) { guard.add(n); prev = n; n = pose.parents.get(n); }
    return n === from ? prev : null;
  };
  // Nearest mapped descendant (skips unmapped middles like thumb0).
  const mappedDesc = (joint) => {
    const seen = new Set([joint]);
    let frontier = (kids.get(joint) || []).filter((c) => !seen.has(c));
    while (frontier.length) {
      const next = [];
      for (const c of frontier) {
        const nm = namesByIndex.get(c);
        if (byMhr.has(nm) && !MH_NO_AIM.has(nm)) return c;
        for (const g of kids.get(c) || []) if (!seen.has(g)) { seen.add(g); next.push(g); }
      }
      frontier = next;
    }
    return undefined;
  };
  // Direction target + bind dir per mapped joint, our frame (root and
  // leaves use the orientation rule instead — see the solve loop).
  const targets = new Map();
  for (const [mhr, mhBone] of byMhr) {
    const joint = jointIndexByName.get(mhr);
    if (joint === undefined || joint >= jointCount || joint === rootIdx) continue;
    if (!mhP.has(mhBone) || !bindPos[joint]) continue;
    const desc = mappedDesc(joint);
    const mhDesc = desc !== undefined ? byMhr.get(namesByIndex.get(desc)) : undefined;
    if (!mhDesc || mhDesc === mhBone || !bindPos[desc]) continue;
    const step = stepDown(mhBone, mhDesc);
    if (!step || !mhP.has(step)) continue;
    _vDir.copy(mhP.get(step)).sub(mhP.get(mhBone));
    if (_vDir.lengthSq() < 1e-12) continue;
    _vBind.copy(bindPos[desc]).sub(bindPos[joint]);
    if (_vBind.lengthSq() < 1e-12) continue;
    targets.set(joint, {
      dir: _vDir.normalize().applyQuaternion(MH_FRAME).clone(),
      bind: _vBind.normalize().clone(),
    });
  }
  const restContinue = (ji, ourParent) => {
    const b = ji * 7;
    _e.set(baseTheta[b + 3], baseTheta[b + 4], baseTheta[b + 5]);
    _gTmp.setFromEuler(_e);
    _gPre.set(rig.pre[ji * 4], rig.pre[ji * 4 + 1], rig.pre[ji * 4 + 2], rig.pre[ji * 4 + 3]);
    ourG[ji].copy(ourParent).multiply(_gPre).multiply(_gTmp);
  };
  // World-frame MH delta, conjugated into our frame.
  const conjDelta = (delta) => _gTmp.copy(MH_FRAME).multiply(delta).multiply(_MH_FRAME_INV).clone();
  const ourG = [];
  for (let j = 0; j < jointCount; j++) ourG.push(new THREE.Quaternion());
  for (const ji of rig.order) {
    if (ji < 0 || ji >= jointCount) continue;
    const p = rig.parents[ji];
    const ourParent = p >= 0 && p < jointCount ? ourG[p] : groupQ;
    _gNeu.set(rig.neutral[ji * 4], rig.neutral[ji * 4 + 1], rig.neutral[ji * 4 + 2], rig.neutral[ji * 4 + 3]);
    const mhr = namesByIndex.get(ji);
    const mhBone = mhr !== undefined ? byMhr.get(mhr) : undefined;
    const delta = mhBone && mhQ.has(mhBone) ? mhQ.get(mhBone) : null;
    const t = targets.get(ji);
    if (ji === rootIdx) {
      // Root: orientation rule (MH root is a ground dummy, ours is the
      // pelvis, so directions don't correspond — relative facing does).
      if (!delta) { restContinue(ji, ourParent); continue; }
      ourG[ji].copy(groupQ).multiply(conjDelta(delta)).multiply(_gNeu);
    } else if (t) {
      // Direction-exact with minimal twist from our rest.
      _gOur.copy(groupQ).invert();
      _vDir.copy(t.dir).applyQuaternion(_gOur);
      _gA.setFromUnitVectors(t.bind, _vDir);
      ourG[ji].copy(groupQ).multiply(_gA).multiply(_gNeu);
    } else if (delta) {
      // Leaf (ball, fingertips): orientation rule.
      ourG[ji].copy(groupQ).multiply(conjDelta(delta)).multiply(_gNeu);
    } else {
      restContinue(ji, ourParent);
      continue;
    }
    _gPre.set(rig.pre[ji * 4], rig.pre[ji * 4 + 1], rig.pre[ji * 4 + 2], rig.pre[ji * 4 + 3]);
    _gOur.copy(_gPre).invert().multiply(_gTmp.copy(ourParent).invert()).multiply(ourG[ji]);
    _e.setFromQuaternion(_gOur, "ZYX");
    const base = ji * 7;
    theta[base + 3] = _e.x;
    theta[base + 4] = _e.y;
    theta[base + 5] = _e.z;
    joints.push(ji);
  }
  // Hips-Y bob from MH vertical (+Z) root motion, decimeters to cm.
  if (rootIdx !== undefined) theta[rootIdx * 7 + 1] = (pose.rootMotion?.[2] ?? 0) * 10;
  return { theta, joints };
}
