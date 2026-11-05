import * as THREE from "three";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

/**
 * Mixamo motion clips: .fbx/.glb files from the server folder, retargeted
 * onto any actor's rig. Rotations plus first-frame-relative hips-Y bob
 * (crouches plant the feet); X/Z stay in place (steering keeps owning
 * translation); unmapped joints hold their base pose.
 *
 * Retarget math per mapped joint (direction rule, poses.js recipe): each
 * joint's segment points where the source segment points, minimal twist
 * from our rest; root/leaves use frame-aligned orientation instead.
 * Converted to the ZYX euler channels our rig expects. This feeds CPU
 * and GPU actors identically. Segments match even when local frames
 * twist between rigs (our X-down-bone vs Mixamo's Y-down): orientation
 * transfer and verbatim local copies both leak hinge bends onto twist
 * axes (frozen knees, twisting feet). Absolute source segments, so
 * sustained poses like dance crouches survive. Playing snaps to the
 * clip's current pose (like a held preset); stopping restores base.
 */

// Source bone (as FBXLoader yields it: `mixamorig<Bone>`, colon stripped)
// -> our joint name. 54 entries: hips/spine/neck/head, both limbs with
// hands+feet, fingers 1-3 (4 = tip, no visible rotation), eyes.
// Joint correspondence follows the authoritative NAME_MAP in
// SAM3DBody-cpp src/core/bvh_writer.cpp: Mixamo's 3 spine bones drive
// c_spine1/2/3 (c_spine0 holds rest), ToeBase stays unmapped (feet flat).
export const MIXAMO_JOINT_MAP = {
  mixamorigHips: "root",
  mixamorigSpine: "c_spine1",
  mixamorigSpine1: "c_spine2",
  mixamorigSpine2: "c_spine3",
  mixamorigNeck: "c_neck",
  mixamorigHead: "c_head",
  mixamorigLeftShoulder: "l_clavicle",
  mixamorigLeftArm: "l_uparm",
  mixamorigLeftForeArm: "l_lowarm",
  mixamorigLeftHand: "l_wrist",
  mixamorigLeftUpLeg: "l_upleg",
  mixamorigLeftLeg: "l_lowleg",
  mixamorigLeftFoot: "l_foot",
  mixamorigLeftHandIndex1: "l_index1",
  mixamorigLeftHandIndex2: "l_index2",
  mixamorigLeftHandIndex3: "l_index3",
  mixamorigLeftHandMiddle1: "l_middle1",
  mixamorigLeftHandMiddle2: "l_middle2",
  mixamorigLeftHandMiddle3: "l_middle3",
  mixamorigLeftHandRing1: "l_ring1",
  mixamorigLeftHandRing2: "l_ring2",
  mixamorigLeftHandRing3: "l_ring3",
  mixamorigLeftHandPinky1: "l_pinky1",
  mixamorigLeftHandPinky2: "l_pinky2",
  mixamorigLeftHandPinky3: "l_pinky3",
  mixamorigLeftHandThumb1: "l_thumb1",
  mixamorigLeftHandThumb2: "l_thumb2",
  mixamorigLeftHandThumb3: "l_thumb3",
  mixamorigRightShoulder: "r_clavicle",
  mixamorigRightArm: "r_uparm",
  mixamorigRightForeArm: "r_lowarm",
  mixamorigRightHand: "r_wrist",
  mixamorigRightUpLeg: "r_upleg",
  mixamorigRightLeg: "r_lowleg",
  mixamorigRightFoot: "r_foot",
  mixamorigRightHandIndex1: "r_index1",
  mixamorigRightHandIndex2: "r_index2",
  mixamorigRightHandIndex3: "r_index3",
  mixamorigRightHandMiddle1: "r_middle1",
  mixamorigRightHandMiddle2: "r_middle2",
  mixamorigRightHandMiddle3: "r_middle3",
  mixamorigRightHandRing1: "r_ring1",
  mixamorigRightHandRing2: "r_ring2",
  mixamorigRightHandRing3: "r_ring3",
  mixamorigRightHandPinky1: "r_pinky1",
  mixamorigRightHandPinky2: "r_pinky2",
  mixamorigRightHandPinky3: "r_pinky3",
  mixamorigRightHandThumb1: "r_thumb1",
  mixamorigRightHandThumb2: "r_thumb2",
  mixamorigRightHandThumb3: "r_thumb3",
  LeftEye: "l_eye",
  RightEye: "r_eye",
};

/**
 * Tolerate exporter variants: `ns:Name` namespaces and a stripped
 * `mixamorig` prefix both resolve to the map key when possible.
 */
export function canonicalMixamoName(name) {
  const bare = name.includes(":") ? name.slice(name.lastIndexOf(":") + 1) : name;
  if (MIXAMO_JOINT_MAP[bare] !== undefined) return bare;
  const prefixed = `mixamorig${bare}`;
  if (MIXAMO_JOINT_MAP[prefixed] !== undefined) return prefixed;
  return bare;
}

let _clipsCache = null;

/** Server clip listing, cached until forced. */
export async function fetchMixamoClips(force = false) {
  if (_clipsCache && !force) return _clipsCache;
  const resp = await fetch("/api/mixamo/clips");
  if (!resp.ok) throw new Error(`Mixamo list failed: HTTP ${resp.status}`);
  _clipsCache = await resp.json();
  return _clipsCache;
}

const _clipPromises = new Map();

// Motion exaggeration for stiff torsos: output gain on the spine channels
// (c_spine1/2/3). 1 = faithful, 0 = frozen spine, up to 4x. Live: players
// read clip.spineBoost every frame, so the UI slider retunes mid-dance.
const SPINE_JOINTS = new Set(["c_spine1", "c_spine2", "c_spine3"]);
let _spineBoost = 1;
const _liveClips = new Set();

export function setSpineBoost(v) {
  _spineBoost = Number.isFinite(v) ? Math.min(4, Math.max(0, v)) : 1;
  for (const clip of _liveClips) clip.spineBoost = _spineBoost;
  return _spineBoost;
}

// Direction-rule retarget (poses.js recipe, per frame): per mapped joint,
// the segment points where the source segment points (minimal twist from
// our rest); root/leaves use frame-aligned orientation (e = A x m x A^-1).
// Rest conventions twist between rigs (our X-down-bone vs Mixamo's
// Y-down), which leaks hinge bends onto twist axes under orientation
// transfer and verbatim local copies (frozen knees, twisting feet).
// Per frame, parent-first:
//   ourGlobal(j) = group x segDelta(j) x neutral(j)
//   ourLocal(j)  = pre(j)^-1 x ourParent(j)^-1 x ourGlobal(j)
// where segDelta rotates our bind segment onto the source segment.
// Absolute segments, so sustained poses (crouches) survive; unmapped
// joints hold their live base euler (authored pose kept); group heading
// preserved (we never touch the group). Needs the target rig (boot);
// without it, players fall back to the legacy local-relative path.
let _targetRig = null;
let _globalMatch = true;

export function setGlobalMatch(v) {
  _globalMatch = v !== false;
  return _globalMatch;
}

export function setMixamoTargetRig(preArray, parentArray, offsetArray = null) {
  if (!preArray || !parentArray) {
    _targetRig = null;
    return null;
  }
  const J = parentArray.length;
  if (preArray.length !== J * 4) {
    console.warn("[mixamo] target rig shape mismatch, global match disabled.");
    _targetRig = null;
    return null;
  }
  const pre = Float32Array.from(preArray);
  const parents = Int32Array.from(parentArray);
  const offsets = offsetArray && offsetArray.length === J * 3
    ? Float32Array.from(offsetArray)
    : null;
  const order = [];
  const done = new Uint8Array(J);
  const visit = (j) => {
    if (done[j]) return;
    done[j] = 1;
    const p = parents[j];
    if (p >= 0 && p < J) visit(p);
    order.push(j);
  };
  for (let j = 0; j < J; j++) visit(j);
  const neutral = new Float32Array(J * 4);
  const q = new THREE.Quaternion();
  const qp = new THREE.Quaternion();
  for (const j of order) {
    const p = parents[j];
    if (p >= 0 && p < J) qp.set(neutral[p * 4], neutral[p * 4 + 1], neutral[p * 4 + 2], neutral[p * 4 + 3]);
    else qp.identity();
    q.set(pre[j * 4], pre[j * 4 + 1], pre[j * 4 + 2], pre[j * 4 + 3]);
    qp.multiply(q);
    neutral[j * 4] = qp.x;
    neutral[j * 4 + 1] = qp.y;
    neutral[j * 4 + 2] = qp.z;
    neutral[j * 4 + 3] = qp.w;
  }
  _targetRig = { jointCount: J, pre, parents, neutral, order, offsets };
  return _targetRig;
}

/** Target rig for retargeting (null until boot sets it). Shared with poses.js. */
export function getMixamoTargetRig() {
  return _targetRig;
}

/**
 * Fetch + parse a clip by server file name (cached). jointNames resolves
 * the map to joint indices; throws when the file holds no animation.
 */
export async function loadMixamoClip(name, jointNames = []) {
  if (!_clipPromises.has(name)) {
    _clipPromises.set(name, _loadMixamoClipUncached(name, jointNames));
  }
  try {
    const clip = await _clipPromises.get(name);
    clip.resolveChannels(jointNames);
    return clip;
  } catch (error) {
    _clipPromises.delete(name);
    throw error;
  }
}

async function _loadMixamoClipUncached(name, jointNames) {
  const resp = await fetch(`/api/mixamo/file/${encodeURIComponent(name)}`);
  if (!resp.ok) throw new Error(`Mixamo fetch failed (${name}): HTTP ${resp.status}`);
  const buffer = await resp.arrayBuffer();
  const lower = name.toLowerCase();
  let root = null;
  let animations = [];
  if (lower.endsWith(".fbx")) {
    const group = new FBXLoader().parse(buffer, "");
    root = group;
    animations = group.animations;
  } else if (lower.endsWith(".glb")) {
    const gltf = await new GLTFLoader().parseAsync(buffer, "");
    root = gltf.scene;
    animations = gltf.animations;
  } else {
    throw new Error(`Unsupported clip extension: ${name}`);
  }
  if (!animations || animations.length === 0) {
    throw new Error(`No animation found in ${name}`);
  }
  if (animations.length > 1) {
    console.info(`[mixamo] ${name}: ${animations.length} takes, using "${animations[0].name}".`);
  }
  return new MixamoClip(name, root, animations[0], jointNames);
}

/**
 * A parsed clip with its joint channels resolved. First-frame rest is
 * captured via a scratch mixer (bind restored after); each player clones it.
 */
// End→start worst-joint gap (degrees) below which a clip loops seamlessly.
// Measured: closed performances ≈2°, open ones ≈110° — 10 splits safely.
const LOOP_GAP_DEG = 10;

/**
 * Sample every bone quat at time t. Preserves the source bind pose:
 * captures every bone quat/pos around the scratch mixer and restores it.
 */
function sampleBoneQuats(root, animation, t) {
  const bones = [];
  root.traverse((o) => { if (o.isBone) bones.push(o); });
  if (bones.length === 0 || !(animation.duration > 0)) return { bones, quats: [] };
  const restQ = bones.map((b) => b.quaternion.clone());
  const restP = bones.map((b) => b.position.clone());
  const mixer = new THREE.AnimationMixer(root);
  const action = mixer.clipAction(animation);
  action.loop = THREE.LoopOnce;
  action.clampWhenFinished = true;
  action.play();
  mixer.setTime(t);
  mixer.update(0);
  const quats = bones.map((b) => b.quaternion.clone());
  mixer.stopAllAction();
  mixer.uncacheClip(animation);
  mixer.uncacheRoot(root);
  for (let i = 0; i < bones.length; i++) {
    bones[i].quaternion.copy(restQ[i]);
    bones[i].position.copy(restP[i]);
  }
  return { bones, quats };
}

/**
 * First-frame rest reference: bone -> quat sampled at t=0. Unanimated
 * bones keep their bind quat (the mixer leaves them untouched).
 */
export function captureFirstFrameRest(root, animation) {
  const { bones, quats } = sampleBoneQuats(root, animation, 0);
  const rest = new Map();
  for (let i = 0; i < bones.length; i++) rest.set(bones[i], quats[i]);
  return rest;
}

/** Worst-joint end→start angle (degrees). */
function measureLoopGap(root, animation) {
  if (!(animation.duration > 0)) return 0;
  const { bones, quats: first } = sampleBoneQuats(root, animation, 0);
  const { quats: last } = sampleBoneQuats(root, animation, Math.max(0, animation.duration - 1 / 60));
  let worst = 0;
  for (let i = 0; i < bones.length; i++) {
    const dot = Math.abs(first[i].dot(last[i]));
    const deg = (2 * Math.acos(Math.min(1, dot)) * 180) / Math.PI;
    if (deg > worst) worst = deg;
  }
  return worst;
}

export class MixamoClip {
  constructor(name, sourceRoot, animation, jointNames = []) {
    this.name = name;
    this.sourceRoot = sourceRoot;
    this.animation = animation;
    this.duration = animation.duration;
    this.takeName = animation.name;
    this.trackCount = animation.tracks.length;
    this.keyCount = 0;
    this.fps = null;
    let fpsKeys = 0;
    for (const track of animation.tracks) {
      const times = track.times;
      if (!times) continue;
      this.keyCount += times.length;
      if (times.length < 2) continue;
      const span = times[times.length - 1] - times[0];
      if (span <= 0 || times.length <= fpsKeys) continue;
      fpsKeys = times.length;
      this.fps = Math.round(((times.length - 1) / span) * 10) / 10;
    }
    this.jointNames = jointNames;
    this.spineBoost = _spineBoost;
    _liveClips.add(this);
    this._firstFrameRest = captureFirstFrameRest(sourceRoot, animation);
    this.resolveChannels();
    this.loopGapDeg = measureLoopGap(sourceRoot, animation);
    this.isLooping = this.loopGapDeg <= LOOP_GAP_DEG;
    if (jointNames.length > 0 && this.unmappedSourceBones.length > 0) {
      console.info(
        `[mixamo] ${name}: ${this.mappedCount}/${this.boneCount} bones mapped; ` +
        `unmapped: ${this.unmappedSourceBones.join(", ")}.`,
      );
    }
  }

  /**
   * (Re)binds source bones to joint indices. Re-runnable: inspecting a
   * clip before the rig loads resolves zero channels, and the later pass
   * with real joint names fills them (the source rig is never posed, so
   * rest capture stays valid).
   */
  resolveChannels(jointNames = this.jointNames) {
    this.jointNames = jointNames;
    const jointIndexByName = new Map(jointNames.map((jointName, index) => [jointName, index]));
    const bonesByName = new Map();
    this.sourceRoot.traverse((node) => {
      // First match wins: matches the mixer's track binding (findNode).
      // ActorCore files nest a second, static rig inside the animated one
      // (129 bones, 65 unique names) — last-wins would sample the dead rig.
      if (node.isBone) {
        const key = canonicalMixamoName(node.name);
        if (!bonesByName.has(key)) bonesByName.set(key, node);
      }
    });
    this.channels = [];
    this.unmappedSourceBones = [];
    this.restHipsY = null;
    for (const [boneName, bone] of bonesByName) {
      if (boneName === "mixamorigHips") this.restHipsY = bone.position.y;
      const jointName = MIXAMO_JOINT_MAP[boneName];
      if (jointName === undefined) {
        this.unmappedSourceBones.push(boneName);
        continue;
      }
      const joint = jointIndexByName.get(jointName);
      if (joint === undefined) continue;
      // First-frame rest (bind fallback for zero-duration clips).
      const restQ = this._firstFrameRest?.get(bone) ?? bone.quaternion;
      this.channels.push({
        boneName,
        bone,
        joint,
        restInv: restQ.clone().invert(),
        // Bind-local quat (sourceRoot is restored to bind after capture):
        // axis-alignment reference for twist-robust transfer.
        bindQ: bone.quaternion.clone(),
      });
    }
    this.mappedCount = this.channels.length;
    this.boneCount = bonesByName.size;
    // Hips-Y bob: crouches/jumps ride the root slot (thetaJ translations
    // are centimeters on both CPU and GPU paths). Unit-robust: hips ride
    // ~102 units in cm files, ~1.02 in meter files — 10 splits safely.
    this.hipsTranslate = null;
    this.hipsBobCm = 0;
    const hipsChannel = this.channels.find((c) => MIXAMO_JOINT_MAP[c.boneName] === "root");
    const posTrack = hipsChannel
      ? this.animation.tracks.find((t) => t.name === `${hipsChannel.bone.name}.position`)
      : null;
    if (hipsChannel && posTrack && posTrack.times.length > 0) {
      const n = posTrack.times.length;
      let min = Infinity;
      let max = -Infinity;
      let maxAbs = 0;
      for (let i = 0; i < n; i++) {
        const y = posTrack.values[i * 3 + 1];
        if (y < min) min = y;
        if (y > max) max = y;
        const a = Math.abs(y);
        if (a > maxAbs) maxAbs = a;
      }
      const toCm = maxAbs > 10 ? 1 : 100;
      this.hipsTranslate = {
        key: hipsChannel.boneName,
        joint: hipsChannel.joint,
        firstY: posTrack.values[1],
        toCm,
      };
      this.hipsBobCm = Math.round((max - min) * toCm * 10) / 10;
    }
  }

  /** Plain-object metadata for the UI panel and console paste-back. */
  getMetadata() {
    return {
      name: this.name,
      takeName: this.takeName,
      duration: this.duration,
      fps: this.fps,
      trackCount: this.trackCount,
      keyCount: this.keyCount,
      boneCount: this.boneCount,
      mappedCount: this.mappedCount,
      unmapped: [...this.unmappedSourceBones],
      restHipsY: this.restHipsY,
      hipsBobCm: this.hipsBobCm,
      loopGapDeg: Math.round(this.loopGapDeg * 10) / 10,
      isLooping: this.isLooping,
    };
  }

  createPlayer({ loop = true } = {}) {
    return new ClipPlayer(this, loop);
  }
}

const _sampleQ = new THREE.Quaternion();
const _sampleE = new THREE.Euler(0, 0, 0, "ZYX");
const _TAU = Math.PI * 2;
const _IDENTITY_Q = new THREE.Quaternion();
const _gPre = new THREE.Quaternion();
const _gNeu = new THREE.Quaternion();
const _gTmp = new THREE.Quaternion();
const _gOur = new THREE.Quaternion();
const _gAInv = new THREE.Quaternion();
const _gDir = new THREE.Quaternion();
const _vA = new THREE.Vector3();
const _vB = new THREE.Vector3();
const _vDir = new THREE.Vector3();
const _vBind = new THREE.Vector3();
const _qBind = new THREE.Quaternion();

/** Nearest 2π-branch of `angle` to `prev` (euler continuity). */
function _unwrapAngle(angle, prev) {
  let d = (angle - prev) % _TAU;
  if (d > Math.PI) d -= _TAU;
  else if (d < -Math.PI) d += _TAU;
  return prev + d;
}

/**
 * One playing instance: owns a cloned source rig + mixer so several
 * actors can play the same clip at independent phases.
 */
export class ClipPlayer {
  constructor(clip, loop = true) {
    this.clip = clip;
    this.root = clip.sourceRoot.clone(true);
    const bonesByName = new Map();
    this.root.traverse((node) => {
      // First match wins, same as resolveChannels (see note there).
      if (node.isBone) {
        const key = canonicalMixamoName(node.name);
        if (!bonesByName.has(key)) bonesByName.set(key, node);
      }
    });
    this.channels = [];
    this._chByJoint = new Map();
    this._byJoint = new Map();
    this._ourG = [];
    for (let j = 0; j < clip.jointNames.length; j++) this._ourG.push(new THREE.Quaternion());
    this._dirRig = null;
    for (const source of clip.channels) {
      const bone = bonesByName.get(source.boneName);
      if (!bone) continue;
      const ci = this.channels.length;
      this.channels.push({
        ci,
        bone,
        boneName: source.boneName,
        joint: source.joint,
        restInv: source.restInv,
        bindQ: source.bindQ ?? null,
        isSpine: SPINE_JOINTS.has(MIXAMO_JOINT_MAP[source.boneName] ?? ""),
        _align: new THREE.Quaternion(),
        _alignRig: null,
        _descBone: null,
        _bindDir: null,
      });
      if (!this._chByJoint.has(source.joint)) {
        const added = this.channels[this.channels.length - 1];
        this._chByJoint.set(source.joint, added);
        this._byJoint.set(source.joint, added);
      }
    }
    this._bindOk = this.channels.length > 0 && this.channels.every((c) => c.bindQ);
    this.hips = null;
    if (clip.hipsTranslate) {
      const bone = bonesByName.get(clip.hipsTranslate.key);
      if (bone) this.hips = { bone, ...clip.hipsTranslate };
    }
    this.mixer = new THREE.AnimationMixer(this.root);
    this.action = this.mixer.clipAction(clip.animation);
    this.setLoop(loop);
    this.action.play();
    this.time = 0;
    this._actionTime = 0;
    this._prevEuler = null;
    this._prevValid = false;
  }

  setLoop(loop) {
    this.loop = Boolean(loop);
    this.action.loop = this.loop ? THREE.LoopRepeat : THREE.LoopOnce;
    this.action.clampWhenFinished = !this.loop;
  }

  /**
   * Source-side limb angles (degrees, 180 = straight) from the clone's
   * current posed state. Ground truth for retarget diagnosis: compare
   * against our limbFlex at the same moment.
   */
  sourceFlex() {
    const find = (name) => {
      let hit = null;
      this.root.traverse((o) => {
        if (!hit && o.isBone && canonicalMixamoName(o.name) === name) hit = o;
      });
      return hit;
    };
    const bones = {
      uplegL: find("mixamorigLeftUpLeg"),
      legL: find("mixamorigLeftLeg"),
      footL: find("mixamorigLeftFoot"),
      uplegR: find("mixamorigRightUpLeg"),
      legR: find("mixamorigRightLeg"),
      footR: find("mixamorigRightFoot"),
      uparmL: find("mixamorigLeftArm"),
      lowarmL: find("mixamorigLeftForeArm"),
      wristL: find("mixamorigLeftHand"),
      uparmR: find("mixamorigRightArm"),
      lowarmR: find("mixamorigRightForeArm"),
      wristR: find("mixamorigRightHand"),
    };
    if (Object.values(bones).some((b) => !b)) return null;
    this.root.updateMatrixWorld(true);
    const P = (b) => b.getWorldPosition(new THREE.Vector3());
    const ang = (A, B, C) => {
      const u = A.clone().sub(B);
      const v = C.clone().sub(B);
      if (u.lengthSq() < 1e-12 || v.lengthSq() < 1e-12) return -1;
      const c = Math.min(1, Math.max(-1, u.normalize().dot(v.normalize())));
      return Math.round((Math.acos(c) * 180) / Math.PI * 10) / 10;
    };
    const p = {};
    for (const [k, b] of Object.entries(bones)) p[k] = P(b);
    return {
      kneeL: ang(p.uplegL, p.legL, p.footL),
      kneeR: ang(p.uplegR, p.legR, p.footR),
      elbowL: ang(p.uparmL, p.lowarmL, p.wristL),
      elbowR: ang(p.uparmR, p.lowarmR, p.wristR),
    };
  }

  /**
   * Per-channel retarget audit: source delta magnitude vs written euler.
   * Pinpoints whether stiff joints come from dead source data, lost
   * deltas, or misrouted writes. Reads live scratch (call after update).
   */
  /**
   * Per-channel retarget audit: source local motion vs written euler.
   * (Direction joints match segments, so the two legitimately differ;
   * alignDeg is the fallback orientation alignment, null unless used.)
   */
  channelDebug() {
    const m = new THREE.Quaternion();
    const e = new THREE.Quaternion();
    const eu = new THREE.Euler(0, 0, 0, "ZYX");
    const angOf = (qq) => 2 * Math.acos(Math.min(1, Math.abs(qq.w))) * 180 / Math.PI;
    return this.channels.map((c) => {
      m.copy(c.bindQ).invert().multiply(c.bone.quaternion);
      const o = c.ci * 3;
      const ex = this._prevEuler ? this._prevEuler[o] : 0;
      const ey = this._prevEuler ? this._prevEuler[o + 1] : 0;
      const ez = this._prevEuler ? this._prevEuler[o + 2] : 0;
      e.setFromEuler(eu.set(ex, ey, ez));
      return {
        bone: c.boneName,
        joint: this.clip.jointNames?.[c.joint] ?? c.joint,
        motionDeg: +angOf(m).toFixed(1),
        eulerDeg: +angOf(e).toFixed(1),
        alignDeg: c._alignRig ? +angOf(c._align).toFixed(1) : null,
        euler: [+ex.toFixed(3), +ey.toFixed(3), +ez.toFixed(3)],
      };
    });
  }

  _useGlobal() {
    return _globalMatch
      && _targetRig !== null
      && _targetRig.jointCount === this.clip.jointNames.length
      && this._bindOk;
  }

  _writeChannel(channel, q, thetaJ) {
    _sampleE.setFromQuaternion(q);
    const base = channel.joint * 7;
    const o = channel.ci * 3;
    let x = _sampleE.x;
    let y = _sampleE.y;
    let z = _sampleE.z;
    if (this._prevValid) {
      x = _unwrapAngle(x, this._prevEuler[o]);
      y = _unwrapAngle(y, this._prevEuler[o + 1]);
      z = _unwrapAngle(z, this._prevEuler[o + 2]);
    }
    this._prevEuler[o] = x;
    this._prevEuler[o + 1] = y;
    this._prevEuler[o + 2] = z;
    // Unwrap continuity lives in true rotation space; the spine gain is
    // a pure output multiplier applied at write time.
    const gain = channel.isSpine ? this.clip.spineBoost : 1;
    thetaJ[base + 3] = x * gain;
    thetaJ[base + 4] = y * gain;
    thetaJ[base + 5] = z * gain;
  }

  /** Legacy path: source local triple copied verbatim (needs no rig). */
  _updateLegacy(thetaJ) {
    for (const channel of this.channels) {
      _sampleQ.copy(channel.restInv).multiply(channel.bone.quaternion);
      this._writeChannel(channel, _sampleQ, thetaJ);
    }
  }

  /**
   * Global-match path: ourGlobal(j) = group x neutral(j) x bindDelta(j),
   * solved parent-first; ourLocal from our own updated parents. See the
   * module note above setMixamoTargetRig.
   */
  /**
   * Refresh the per-channel direction cache: nearest mapped-descendant
   * source bone plus our bind segment direction (bind-position FK).
   */
  _refreshDirCache(rig) {
    for (const channel of this.channels) {
      channel._descBone = null;
      channel._bindDir = null;
    }
    if (!rig.offsets) return;
    const J = rig.jointCount;
    const bindPos = new Float32Array(J * 3);
    for (const j of rig.order) {
      const p = rig.parents[j];
      const o = j * 3;
      if (p >= 0 && p < J) {
        _qBind.set(rig.neutral[p * 4], rig.neutral[p * 4 + 1],
          rig.neutral[p * 4 + 2], rig.neutral[p * 4 + 3]);
        _vBind.set(rig.offsets[o], rig.offsets[o + 1], rig.offsets[o + 2])
          .applyQuaternion(_qBind);
        bindPos[o] = bindPos[p * 3] + _vBind.x;
        bindPos[o + 1] = bindPos[p * 3 + 1] + _vBind.y;
        bindPos[o + 2] = bindPos[p * 3 + 2] + _vBind.z;
      } else {
        bindPos[o] = rig.offsets[o];
        bindPos[o + 1] = rig.offsets[o + 1];
        bindPos[o + 2] = rig.offsets[o + 2];
      }
    }
    for (const channel of this.channels) {
      channel._descBone = null;
      channel._bindDir = null;
      // Nearest descendant joint with a channel (BFS over the kids).
      const seen = new Set([channel.joint]);
      const queue = [channel.joint];
      let found = -1;
      while (queue.length > 0 && found < 0) {
        const cur = queue.shift();
        for (let k = 0; k < J; k++) {
          if (rig.parents[k] !== cur || seen.has(k)) continue;
          seen.add(k);
          if (this._chByJoint.has(k)) { found = k; break; }
          queue.push(k);
        }
      }
      if (found < 0) continue;
      channel._descBone = this._chByJoint.get(found).bone;
      _vBind.set(bindPos[found * 3] - bindPos[channel.joint * 3],
        bindPos[found * 3 + 1] - bindPos[channel.joint * 3 + 1],
        bindPos[found * 3 + 2] - bindPos[channel.joint * 3 + 2]);
      if (_vBind.lengthSq() > 1e-12) channel._bindDir = _vBind.clone().normalize();
    }
    this._dirRig = rig;
  }

  /** Aligned-local fallback for leaves: e = A x m x A^-1, A = pre^-1 x bindQ. */
  _writeAlignedFallback(channel, thetaJ) {
    const rig = _targetRig;
    if (channel._alignRig !== rig) {
      _gPre.set(rig.pre[channel.joint * 4], rig.pre[channel.joint * 4 + 1],
        rig.pre[channel.joint * 4 + 2], rig.pre[channel.joint * 4 + 3]);
      channel._align.copy(_gPre).invert().multiply(channel.bindQ);
      channel._alignRig = rig;
    }
    _gTmp.copy(channel.bindQ).invert().multiply(channel.bone.quaternion);
    _gOur.copy(channel._align).multiply(_gTmp)
      .multiply(_gAInv.copy(channel._align).invert());
    this._writeChannel(channel, _gOur, thetaJ);
  }

  /**
   * Direction-rule transfer (poses.js recipe, per frame): each mapped
   * joint's segment points where the source segment points, minimal
   * twist from our rest. Convention-immune: segments match even when
   * local frames twist between rigs (our X-down-bone vs Mixamo's
   * Y-down). Root/leaves fall back to aligned-local orientation.
   */
  _updateAligned(thetaJ, groupQuat) {
    const rig = _targetRig;
    if (this._dirRig !== rig) this._refreshDirCache(rig);
    this.root.updateMatrixWorld(true);
    for (const ji of rig.order) {
      const p = rig.parents[ji];
      const ourParent = p >= 0 && p < rig.jointCount ? this._ourG[p] : groupQuat;
      const channel = this._byJoint.get(ji);
      if (!channel) {
        // Unmapped: rest-continue from the live base euler (authored kept).
        const b = ji * 7;
        _sampleE.set(thetaJ[b + 3], thetaJ[b + 4], thetaJ[b + 5]);
        _gTmp.setFromEuler(_sampleE);
        _gPre.set(rig.pre[ji * 4], rig.pre[ji * 4 + 1], rig.pre[ji * 4 + 2], rig.pre[ji * 4 + 3]);
        this._ourG[ji].copy(ourParent).multiply(_gPre).multiply(_gTmp);
        continue;
      }
      let directed = false;
      if (channel._descBone && channel._bindDir) {
        channel.bone.getWorldPosition(_vA);
        channel._descBone.getWorldPosition(_vB);
        _vDir.copy(_vB).sub(_vA);
        if (_vDir.lengthSq() > 1e-12) {
          _gDir.setFromUnitVectors(channel._bindDir, _vDir.normalize());
          _gNeu.set(rig.neutral[ji * 4], rig.neutral[ji * 4 + 1],
            rig.neutral[ji * 4 + 2], rig.neutral[ji * 4 + 3]);
          this._ourG[ji].copy(groupQuat).multiply(_gDir).multiply(_gNeu);
          _gPre.set(rig.pre[ji * 4], rig.pre[ji * 4 + 1],
            rig.pre[ji * 4 + 2], rig.pre[ji * 4 + 3]);
          _gOur.copy(_gPre).invert().multiply(_gTmp.copy(ourParent).invert())
            .multiply(this._ourG[ji]);
          this._writeChannel(channel, _gOur, thetaJ);
          directed = true;
        }
      }
      if (!directed) {
        // Leaf/orientation fallback (also refreshes _ourG for the chain).
        this._writeAlignedFallback(channel, thetaJ);
        const b = ji * 7;
        _sampleE.set(thetaJ[b + 3], thetaJ[b + 4], thetaJ[b + 5]);
        _gTmp.setFromEuler(_sampleE);
        _gPre.set(rig.pre[ji * 4], rig.pre[ji * 4 + 1],
          rig.pre[ji * 4 + 2], rig.pre[ji * 4 + 3]);
        this._ourG[ji].copy(ourParent).multiply(_gPre).multiply(_gTmp);
      }
    }
  }

  /**
   * Advance the performance and write ZYX eulers into thetaJ (mapped
   * joints only). Channels unwrap against the previous frame: raw euler
   * conversion flips branches (e.g. +179° → -179°) on fast rotations,
   * which reads as violent twitching. Loop wraps snap instead (a real
   * cut the unwrap must not smear). groupQuat is kept for API
   * compatibility (steering heading owns the group, clips never move it).
   */
  update(dt, thetaJ, groupQuat = _IDENTITY_Q) {
    this.time += dt;
    this.mixer.update(dt);
    const wrapped = this.action.time < this._actionTime - 1e-6;
    this._actionTime = this.action.time;
    if (wrapped || !this._prevEuler || this._prevEuler.length !== this.channels.length * 3) {
      this._prevEuler = new Float32Array(this.channels.length * 3);
      this._prevValid = false;
    }
    if (this._useGlobal()) this._updateAligned(thetaJ, groupQuat);
    else this._updateLegacy(thetaJ);
    if (this.hips) {
      // Additive onto the authored base: the root drops/rises with the
      // performance while X/Z stay steering-owned.
      thetaJ[this.hips.joint * 7 + 1] += (this.hips.bone.position.y - this.hips.firstY) * this.hips.toCm;
    }
    this._prevValid = true;
    return this.time;
  }

  dispose() {
    this.action.stop();
    this.mixer.uncacheClip(this.clip.animation);
    this.mixer.uncacheRoot(this.root);
  }
}
