import * as THREE from "three";
import { MeshBasicNodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";
import { currentParameter, state } from "./state.js";
import { requestRender } from "./scene.js";
import { actorManager } from "./actor.js";
import { showParameter } from "./params.js";
import {
  addBlendshapeUpdateListener,
  addPoseUpdateListener,
  evaluateBlendshapesClientSide,
  evaluatePoseClientSide,
} from "./rig.js";
import { BROW_ANCHOR, BROW_REF_DM, CARD_BROW_PUSH, CARD_BROW_TILT, CARD_PROUD, LASH_ANCHOR, LASH_CARD_PROUD, LASH_REF_DM, MH_DM_TO_M, STRAND_LASH_PUSH, isCardDecoId, loadMhCardTexture, loadMhMesh, loadMhObjMesh, placeDecoPosition } from "./mhAssets.js";

// Eyes + oral cavity for the anchor actor (#0).
//
// Geometry builders are adapted from v2/mhr_eyes_oral_cavity_module (dental
// arches, skinned tongue, throat/palate/floor, eyeball proportions, iris
// shading). Deliberate deviations from that module, verified against our rig:
// - No prosthetic eyelid caps: our face already closes via the eyesClosed /
//   lidTightener / upperLidRaiser blendshapes, so blink and gaze droop write
//   expression values instead of rotating extra meshes.
// - No FACS index table: expression channels resolve by name
//   (eyesClosed_L, jawDrop, ...) against metadata.groups.expression.
// - No jaw angle read from bone rotation: our jaw quaternion is a pre-rotated
//   ZYX euler, so the lower assembly follows the jawDrop expression value
//   plus live jaw-bone motion, all in head space.
// - The GLSL eye shader is ported to TSL so it runs on WebGPURenderer (and
//   the WebGL fallback, which also accepts node materials).
// - Module-space content (iris +Z, up +Y) is anchored to OUR head frame via a
//   runtime basis built from the l_eye / r_eye / c_head_null joints, with the
//   module eye midpoint pinned to our eye midpoint. Nothing assumes the demo
//   proxy's absolute offsets.

// Module-space landmarks (meters, iris-forward +Z, up +Y). Exported for
// crowdMouth.js, which re-anchors the same assemblies under agent heads.
export const MODULE_EYE_MID = new THREE.Vector3(0, 0.032, 0.055);
export const UPPER_ARCH_POS = new THREE.Vector3(0, -0.028, 0.036);
export const TONGUE_REL_LOWER_ARCH = new THREE.Vector3(0, 0.005, -0.019);
export const FLOOR_REL_LOWER_ARCH = new THREE.Vector3(0, -0.0035, -0.006);
// Lower-arch rest gap below the upper-arch origin, and jawDrop follow travel.
export const LOWER_GAP_REST = 0.016;
// Fitted MakeHuman arches render small against our head; uniform grow factor.
export const FITTED_TEETH_SCALE = 1.2;
// Eye-decoration placement lives in mhAssets.js (node-testable).
export const CARD_TINT_DEFAULT = "#ffffff";
export const STRAND_TINT_DEFAULT = "#201a16";
export const DEFAULT_BROW_ID =
  "eye_decoration_modelled/eyebrows01_cc0/eyebrows/mindfront_eyebrows_01/mind_eyebrows_01.obj";
export const DEFAULT_LASH_ID =
  "eye_decoration_modelled/eyelashes01_cc0/eyelashes/mindfront_eyelashes_01/mind_eyelashes_01.obj";
export const DROP_DOWN = 0.024;
export const DROP_BACK = 0.014;
export const LOWER_VISIBLE_AT = 0.02;

// ---------------------------------------------------------------------------
// TSL eye shader (branchless port of the module's iris/pupil/sclera GLSL)
// ---------------------------------------------------------------------------

export function createEyeMaterial({ pupil = null } = {}) {
  const uIrisColor = TSL.uniform(new THREE.Color("#2e77bb"));
  // Pupil size node: a uniform for actor #0, a per-instance attribute for crowds.
  const uPupilSize = pupil ?? TSL.uniform(0.45);
  const uScleraRedness = TSL.uniform(0.15);

  const dir = TSL.positionLocal.normalize();
  const radialDist = dir.xy.length();
  const angle = TSL.atan(dir.y, dir.x);
  const viewDir = TSL.positionView.mul(-1).normalize();
  const normal = TSL.normalView.normalize();

  const refractedXY = dir.xy.sub(viewDir.xy.mul(0.06));
  const rRefract = refractedXY.length();
  const irisRadius = TSL.float(0.47);
  const pupilRadius = TSL.mix(0.08, 0.34, uPupilSize);

  const front = TSL.smoothstep(0.0, 0.03, dir.z);
  const inIris = TSL.float(1).sub(
    TSL.smoothstep(irisRadius.sub(0.015), irisRadius.add(0.015), rRefract),
  );
  const irisZone = front.mul(inIris);
  const pupilMask = TSL.float(1).sub(
    TSL.smoothstep(pupilRadius.sub(0.008), pupilRadius.add(0.008), rRefract),
  );

  const t = rRefract
    .sub(pupilRadius)
    .div(irisRadius.sub(pupilRadius).max(1e-4))
    .clamp(0, 1);
  const fibers = TSL.sin(angle.mul(70).add(t.mul(12).sin()))
    .mul(0.5)
    .add(0.5);
  const crypts = TSL.sin(angle.mul(23).sub(t.mul(8)))
    .mul(0.5)
    .add(0.5);
  const innerCol = uIrisColor.mul(1.35).add(TSL.vec3(0.12, 0.06, 0.0));
  const outerCol = uIrisColor.mul(fibers.mul(0.35).add(0.65));
  const irisCol = TSL.mix(innerCol, outerCol, TSL.smoothstep(0.0, 0.85, t)).mul(
    crypts.mul(0.37).add(0.78),
  );
  const limbus = TSL.smoothstep(0.75, 1.0, t);
  const irisShaded = TSL.mix(irisCol, uIrisColor.mul(0.18), limbus);
  const irisFinal = TSL.mix(
    irisShaded,
    TSL.vec3(0.01, 0.01, 0.015),
    pupilMask,
  );

  const veinPattern = TSL.sin(angle.mul(17).add(radialDist.mul(18)))
    .mul(TSL.cos(angle.mul(11)))
    .abs();
  const veinMask = TSL.smoothstep(0.52, 0.95, radialDist).mul(
    veinPattern.pow(5),
  );
  const scleraCol = TSL.mix(
    TSL.vec3(0.95, 0.93, 0.91),
    TSL.vec3(0.82, 0.22, 0.22),
    uScleraRedness.mul(0.35).add(veinMask.mul(uScleraRedness).mul(0.55)),
  );
  const borderSoft = TSL.smoothstep(
    irisRadius,
    irisRadius.add(0.04),
    rRefract,
  );
  const scleraFinal = scleraCol.mul(
    TSL.mix(1.0, TSL.mix(0.6, 1.0, borderSoft), front),
  );
  const albedo = TSL.mix(scleraFinal, irisFinal, irisZone);

  const light1 = TSL.vec3(0.5, 0.7, 1.0).normalize();
  const light2 = TSL.vec3(-0.6, 0.2, 0.8).normalize();
  const diff = normal
    .dot(light1)
    .max(0)
    .mul(0.75)
    .add(normal.dot(light2).max(0).mul(0.35))
    .add(0.25);
  const halfway = light1.add(viewDir).normalize();
  const spec = normal.dot(halfway).max(0).pow(180).mul(1.35);
  const fresnel = TSL.float(1).sub(normal.dot(viewDir).max(0)).pow(4).mul(0.25);

  const material = new MeshBasicNodeMaterial();
  material.colorNode = albedo.mul(diff).add(spec.add(fresnel));
  material.userData.uniforms = { uIrisColor, uPupilSize, uScleraRedness };
  return material;
}

// ---------------------------------------------------------------------------
// Anatomical eye (TSL port of the "Procedural Eye (1).html" demo iris/sclera).
// Ball-local coords: dir.xy plays the role of the demo's eye-space q, and
// mesh rotation carries gaze, so the demo's 2D gaze/blink/lid cheats are
// intentionally not ported. Lids, lashes and skin stay real geometry.
// ---------------------------------------------------------------------------

function mhHash(p) {
  const q = p.mul(TSL.vec2(123.34, 456.21)).fract();
  const r = q.add(q.dot(q.add(45.32)));
  return r.x.mul(r.y).fract();
}

function mhNoise(p) {
  const i = p.floor();
  const f0 = p.fract();
  const f = f0.mul(f0).mul(TSL.float(3).sub(f0.mul(2)));
  const a = mhHash(i);
  const b = mhHash(i.add(TSL.vec2(1, 0)));
  const c = mhHash(i.add(TSL.vec2(0, 1)));
  const d = mhHash(i.add(TSL.vec2(1, 1)));
  return TSL.mix(TSL.mix(a, b, f.x), TSL.mix(c, d, f.x), f.y);
}

function mhRot(p) {
  return TSL.vec2(
    p.x.mul(1.6).sub(p.y.mul(1.2)),
    p.x.mul(1.2).add(p.y.mul(1.6)),
  );
}

// 5-octave fbm, unrolled: pure node expressions, no Fn scope needed.
function mhFbm(p) {
  const p1 = mhRot(p);
  const p2 = mhRot(p1);
  const p3 = mhRot(p2);
  const p4 = mhRot(p3);
  return mhNoise(p).mul(0.5)
    .add(mhNoise(p1).mul(0.25))
    .add(mhNoise(p2).mul(0.125))
    .add(mhNoise(p3).mul(0.0625))
    .add(mhNoise(p4).mul(0.03125));
}

function mhSplat(x) {
  return TSL.vec2(x, x);
}

/** Rich hero-eye material. Uniform set mirrors the demo's defaults. */
export function createAnatomicalEyeMaterial() {
  const uIrisIn = TSL.uniform(new THREE.Color("#9fc7ea"));
  const uIrisOut = TSL.uniform(new THREE.Color("#1f5499"));
  const uPupilSize = TSL.uniform(0.45);
  const uVeins = TSL.uniform(0.5);
  const uTime = TSL.uniform(0);
  // Demo defaults, kept as constants (wired to UI only on request).
  const uDetail = TSL.float(5);
  const uFibre = TSL.float(1);
  const uGrain = TSL.float(0.5);
  const uRadial = TSL.float(1.4);
  const uCollar = TSL.float(0.55);
  const uCrypts = TSL.float(0.35);
  const uLimbal = TSL.float(0.85);
  const uVeinScale = TSL.float(1);
  const uVeinThin = TSL.float(12);
  const uVeinOpac = TSL.float(0.65);
  const irisRadius = TSL.float(0.47);

  const dir = TSL.positionLocal.normalize();
  const radialDist = dir.xy.length();
  const viewDir = TSL.positionView.mul(-1).normalize();
  const normal = TSL.normalView.normalize();

  // Cheap parallax cue, as in the basic eye: the iris pattern lags the view.
  const refractedXY = dir.xy.sub(viewDir.xy.mul(0.06));
  const pR = refractedXY.length();
  const dirN = refractedXY.div(pR.max(1e-4));
  const pupilFrac = TSL.mix(0.17, 0.62, uPupilSize);
  const wobble = TSL.float(1)
    .add(uTime.mul(1.9).sin().mul(0.025))
    .add(uTime.mul(4.3).sin().mul(0.015));
  const pr = irisRadius.mul(pupilFrac).mul(wobble);
  const t = pR.sub(pr).div(irisRadius.sub(pr).max(1e-3)).clamp(0, 1);

  const fib = mhFbm(dirN.mul(uDetail).add(mhSplat(pR.mul(uRadial))));
  const fib2 = mhNoise(dirN.mul(uDetail.mul(3)).add(mhSplat(pR.mul(uRadial).mul(2.1))));
  const fine = mhFbm(dirN.mul(uDetail.mul(2.2)).add(mhSplat(pR.mul(uRadial).mul(0.6))));

  const irisBase = TSL.mix(uIrisIn, uIrisOut, TSL.smoothstep(0, 0.85, t));
  const irisFibred = irisBase.mul(TSL.float(0.96).add(uFibre.mul(fib.mul(1.15).sub(0.54))));
  const irisGrained = irisFibred.mul(TSL.float(1).add(uGrain.mul(0.48).mul(fine.sub(0.5))));
  const collarT = t.sub(0.30).sub(fib.sub(0.5).mul(0.05)).abs();
  const collar = TSL.float(1).sub(TSL.smoothstep(0, 0.07, collarT));
  const irisCollared = TSL.mix(irisGrained, uIrisIn.mul(1.35).add(0.06), collar.mul(uCollar));
  const cryptZone = TSL.smoothstep(0.1, 0.4, t).mul(TSL.float(1).sub(TSL.smoothstep(0.5, 0.8, t)));
  const irisCrypt = irisCollared.mul(
    TSL.float(1).sub(uCrypts.mul(TSL.smoothstep(0.55, 0.78, fib2)).mul(cryptZone)));
  const limb = TSL.smoothstep(0.62, 1.0, pR.div(irisRadius));
  const irisLimbed = irisCrypt.mul(TSL.float(1).sub(uLimbal.mul(limb).mul(limb)));
  const irisCol = irisLimbed.mul(TSL.float(0.55).add(TSL.float(0.45).mul(TSL.smoothstep(0, 0.14, t))));

  const front = TSL.smoothstep(0.0, 0.03, dir.z);
  const irisMask = TSL.float(1).sub(TSL.smoothstep(irisRadius.sub(0.006), irisRadius.add(0.006), pR));
  const pupilMask = TSL.float(1).sub(TSL.smoothstep(pr.sub(0.004), pr.add(0.004), pR));
  const irisFinal = TSL.mix(irisCol, TSL.vec3(0.008, 0.008, 0.008), pupilMask);
  const irisZone = front.mul(irisMask);

  // Sclera with gaze-coupled single-layer veins (mesh rotation carries gaze).
  const scleraBase = TSL.vec3(0.93, 0.91, 0.88)
    .mul(TSL.float(1).sub(TSL.float(0.38).mul(TSL.smoothstep(0.55, 1.0, radialDist))));
  const sq = dir.xy.mul(1.32);
  const warp = mhFbm(sq.mul(1.3).mul(uVeinScale)).mul(2);
  const vn = mhFbm(sq.mul(TSL.vec2(2.6, 3.4)).mul(uVeinScale).add(TSL.vec2(4, 1)).add(mhSplat(warp)));
  const veins = TSL.float(1).sub(vn.mul(2).sub(0.95).abs()).clamp(0, 1).pow(uVeinThin);
  const vmask = TSL.smoothstep(irisRadius.mul(1.05), irisRadius.mul(1.9), radialDist).mul(front).mul(uVeins);
  const scleraVeined = TSL.mix(scleraBase, TSL.vec3(0.75, 0.18, 0.16), veins.mul(vmask).mul(uVeinOpac));
  const limbusShadow = TSL.float(1).sub(TSL.smoothstep(irisRadius, irisRadius.mul(1.55), radialDist));
  const scleraCol = scleraVeined.mul(TSL.float(1).sub(TSL.float(0.3).mul(limbusShadow)));
  const albedo = TSL.mix(scleraCol, irisFinal, irisZone);

  // View-space key/fill lighting, as in the basic eye.
  const light1 = TSL.vec3(0.5, 0.7, 1.0).normalize();
  const light2 = TSL.vec3(-0.6, 0.2, 0.8).normalize();
  const diff = normal
    .dot(light1)
    .max(0)
    .mul(0.75)
    .add(normal.dot(light2).max(0).mul(0.35))
    .add(0.25);
  const halfway = light1.add(viewDir).normalize();
  const spec = normal.dot(halfway).max(0).pow(180).mul(1.35);
  const fresnel = TSL.float(1).sub(normal.dot(viewDir).max(0)).pow(4).mul(0.25);

  const material = new MeshBasicNodeMaterial();
  material.colorNode = albedo.mul(diff).add(spec.add(fresnel));
  material.userData.uniforms = { uIrisIn, uIrisOut, uPupilSize, uVeins, uTime };
  return material;
}

// Eyeball radius tuned so the lids fully cover the ball at the canthi;
// smaller leaves see-through wedges into the skull at the corners.
export const EYEBALL_RADIUS = 0.0135;
export function createEyeballGeometry() {
  const geo = new THREE.SphereGeometry(EYEBALL_RADIUS, 48, 32);
  const pos = geo.attributes.position;
  const bulgeAt = EYEBALL_RADIUS * 0.84;
  for (let i = 0; i < pos.count; i++) {
    const z = pos.getZ(i);
    if (z > bulgeAt) pos.setZ(i, z + (z - bulgeAt) * 0.45);
  }
  geo.computeVertexNormals();
  return geo;
}

function createEyeballMesh(material) {
  const mesh = new THREE.Mesh(createEyeballGeometry(), material);
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return mesh;
}

// ---------------------------------------------------------------------------
// Dental arches (ported builder; morphs: crooked, gap, overbite, fangs)
// ---------------------------------------------------------------------------

export function createDentalArchMesh(isUpper) {
  const maxTeeth = 14;
  const baseCrownGeo = new THREE.BoxGeometry(1, 1, 1, 3, 3, 3);
  const bPos = baseCrownGeo.attributes.position;
  for (let i = 0; i < bPos.count; i++) {
    const vx = bPos.getX(i);
    const vy = bPos.getY(i);
    const vz = bPos.getZ(i);
    const rootFactor = isUpper ? vy + 0.5 : 0.5 - vy;
    const taper = 0.78 + 0.22 * (1.0 - rootFactor * 0.5);
    bPos.setXYZ(i, vx * taper, vy, vz * (0.85 + 0.15 * Math.cos(vx * 1.5)));
  }
  baseCrownGeo.computeVertexNormals();
  const vertsPerTooth = baseCrownGeo.attributes.position.count;
  const idxPerTooth = baseCrownGeo.index.count;
  const totalVerts = vertsPerTooth * maxTeeth;
  const neutralPositions = new Float32Array(totalVerts * 3);
  const normals = new Float32Array(totalVerts * 3);
  const indices = new Uint16Array(idxPerTooth * maxTeeth);
  const morphCrooked = new Float32Array(totalVerts * 3);
  const morphGap = new Float32Array(totalVerts * 3);
  const morphOverbite = new Float32Array(totalVerts * 3);
  const morphFangs = new Float32Array(totalVerts * 3);
  const dummy = new THREE.Object3D();
  const vTemp = new THREE.Vector3();
  const nTemp = new THREE.Vector3();
  for (let pairIdx = 0; pairIdx < 7; pairIdx++) {
    for (let sideIdx = 0; sideIdx < 2; sideIdx++) {
      const toothSlot = pairIdx * 2 + sideIdx;
      const sign = sideIdx === 0 ? -1 : 1;
      const u = pairIdx + 0.48;
      const angle = ((u / 6.8) * (Math.PI * 0.46)) * sign;
      const archWidth = 0.025;
      const archDepth = 0.032;
      const tx = Math.sin(angle) * archWidth;
      const tz = Math.cos(angle) * archDepth;
      let w = 0.0062;
      let h = 0.0085;
      let d = 0.0042;
      if (pairIdx === 0) {
        w = 0.0072; h = 0.0092; d = 0.0038;
      } else if (pairIdx === 1) {
        w = 0.0058; h = 0.0086; d = 0.0038;
      } else if (pairIdx === 2) {
        w = 0.0064; h = 0.0098; d = 0.0046;
      } else if (pairIdx >= 5) {
        w = 0.0076; h = 0.0075; d = 0.0068;
      }
      if (!isUpper && pairIdx <= 1) w *= 0.82;
      const ty = isUpper ? -h * 0.45 : h * 0.45;
      dummy.position.set(tx, ty, tz);
      dummy.rotation.set(0, angle, 0);
      dummy.scale.set(w, h, d);
      dummy.updateMatrix();
      const vOffset = toothSlot * vertsPerTooth;
      const iOffset = toothSlot * idxPerTooth;
      for (let k = 0; k < idxPerTooth; k++) {
        indices[iOffset + k] = baseCrownGeo.index.array[k] + vOffset;
      }
      const seed = (pairIdx + 1) * 1.7 + sideIdx * 3.1 + (isUpper ? 0.5 : 2.2);
      const crookedYaw =
        Math.sin(seed * 2.3) * 0.42 * (pairIdx <= 3 ? 1.0 : 0.4);
      const crookedPitch = Math.cos(seed * 1.9) * 0.22;
      const crookedShiftZ =
        (pairIdx % 2 === 0 ? 1 : -1) * 0.0022 * (pairIdx <= 2 ? 1.0 : 0.3);
      const crookedDummy = dummy.clone();
      crookedDummy.rotation.y += crookedYaw;
      crookedDummy.rotation.x += crookedPitch;
      crookedDummy.position.z += crookedShiftZ;
      crookedDummy.updateMatrix();
      for (let v = 0; v < vertsPerTooth; v++) {
        const idx3 = (vOffset + v) * 3;
        const lx = bPos.getX(v);
        const ly = bPos.getY(v);
        const lz = bPos.getZ(v);
        vTemp.set(lx, ly, lz).applyMatrix4(dummy.matrix);
        neutralPositions[idx3] = vTemp.x;
        neutralPositions[idx3 + 1] = vTemp.y;
        neutralPositions[idx3 + 2] = vTemp.z;
        nTemp
          .fromBufferAttribute(baseCrownGeo.attributes.normal, v)
          .transformDirection(dummy.matrix);
        normals[idx3] = nTemp.x;
        normals[idx3 + 1] = nTemp.y;
        normals[idx3 + 2] = nTemp.z;
        const cPos = new THREE.Vector3(lx, ly, lz).applyMatrix4(
          crookedDummy.matrix,
        );
        morphCrooked[idx3] = cPos.x;
        morphCrooked[idx3 + 1] = cPos.y;
        morphCrooked[idx3 + 2] = cPos.z;
        const gapFactor = pairIdx === 0 ? 0.0028 : pairIdx === 1 ? 0.0014 : 0.0;
        morphGap[idx3] = vTemp.x + sign * gapFactor;
        morphGap[idx3 + 1] = vTemp.y;
        morphGap[idx3 + 2] = vTemp.z;
        const archShift = isUpper ? 0.0045 : -0.0045;
        morphOverbite[idx3] = vTemp.x;
        morphOverbite[idx3 + 1] = vTemp.y;
        morphOverbite[idx3 + 2] = vTemp.z + archShift;
        let fangY = 0;
        let fangTapX = vTemp.x;
        let fangTapZ = vTemp.z;
        if (pairIdx === 2) {
          const tipMask = isUpper
            ? Math.max(0, -ly + 0.2)
            : Math.max(0, ly + 0.2);
          fangY = (isUpper ? -1 : 1) * tipMask * 0.0075;
          fangTapX = THREE.MathUtils.lerp(
            vTemp.x,
            dummy.position.x,
            tipMask * 0.55,
          );
          fangTapZ = THREE.MathUtils.lerp(
            vTemp.z,
            dummy.position.z,
            tipMask * 0.55,
          );
        }
        morphFangs[idx3] = fangTapX;
        morphFangs[idx3 + 1] = vTemp.y + fangY;
        morphFangs[idx3 + 2] = fangTapZ;
      }
    }
  }
  const archGeo = new THREE.BufferGeometry();
  archGeo.setAttribute("position", new THREE.BufferAttribute(neutralPositions, 3));
  archGeo.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  archGeo.setIndex(new THREE.BufferAttribute(indices, 1));
  archGeo.morphAttributes.position = [
    new THREE.BufferAttribute(morphCrooked, 3),
    new THREE.BufferAttribute(morphGap, 3),
    new THREE.BufferAttribute(morphOverbite, 3),
    new THREE.BufferAttribute(morphFangs, 3),
  ];
  archGeo.computeBoundingSphere();
  const gumCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.028, isUpper ? 0.004 : -0.004, -0.002),
    new THREE.Vector3(-0.025, isUpper ? 0.004 : -0.004, 0.005),
    new THREE.Vector3(-0.019, isUpper ? 0.004 : -0.004, 0.024),
    new THREE.Vector3(0.0, isUpper ? 0.004 : -0.004, 0.033),
    new THREE.Vector3(0.019, isUpper ? 0.004 : -0.004, 0.024),
    new THREE.Vector3(0.025, isUpper ? 0.004 : -0.004, 0.005),
    new THREE.Vector3(0.028, isUpper ? 0.004 : -0.004, -0.002),
  ]);
  const gumGeo = new THREE.TubeGeometry(gumCurve, 40, 0.0058, 12, false);
  const capGeo = new THREE.SphereGeometry(0.0058, 12, 10);
  const gumMat = new THREE.MeshPhysicalMaterial({
    color: 0xd96b75,
    roughness: 0.32,
    clearcoat: 0.65,
    clearcoatRoughness: 0.2,
  });
  const gumMesh = new THREE.Mesh(gumGeo, gumMat);
  const capL = new THREE.Mesh(capGeo, gumMat);
  capL.position.copy(gumCurve.getPoint(0));
  const capR = new THREE.Mesh(capGeo, gumMat);
  capR.position.copy(gumCurve.getPoint(1));
  const enamelMat = new THREE.MeshPhysicalMaterial({
    color: 0xf8f5eb,
    roughness: 0.16,
    clearcoat: 0.85,
    clearcoatRoughness: 0.12,
    reflectivity: 0.7,
  });
  const teethMesh = new THREE.Mesh(archGeo, enamelMat);
  teethMesh.updateMorphTargets();
  for (const mesh of [gumMesh, capL, capR, teethMesh]) {
    mesh.castShadow = false;
    mesh.receiveShadow = false;
  }
  const group = new THREE.Group();
  group.add(gumMesh, capL, capR, teethMesh);
  return { group, teethMesh, idxPerTooth };
}

// ---------------------------------------------------------------------------
// Skinned tongue (ported builder; morphs: width, roll, thick)
// ---------------------------------------------------------------------------

export function createSkinnedTongue() {
  const length = 0.044;
  const segsZ = 24;
  const segsX = 14;
  const segsY = 6;
  const geo = new THREE.BoxGeometry(0.028, 0.009, length, segsX, segsY, segsZ);
  geo.translate(0, 0, length * 0.5);
  const pos = geo.attributes.position;
  const vCount = pos.count;
  for (let i = 0; i < vCount; i++) {
    let x = pos.getX(i);
    let y = pos.getY(i);
    const z = pos.getZ(i);
    const u = z / length;
    const widthProfile = Math.sin(
      Math.acos(Math.min(0.98, Math.max(0, u * 0.92))),
    );
    x *= widthProfile;
    const normX = Math.abs(x) / 0.014;
    const roundY = Math.sqrt(Math.max(0.05, 1.0 - normX * normX * 0.65));
    const tipThin = 1.0 - Math.pow(u, 2.2) * 0.55;
    y *= roundY * tipThin;
    if (y > 0) y -= Math.exp(-normX * normX * 12.0) * 0.0012 * (1.0 - u * 0.5);
    pos.setXYZ(i, x, y, z);
  }
  geo.computeVertexNormals();
  const morphWide = new Float32Array(vCount * 3);
  const morphRoll = new Float32Array(vCount * 3);
  const morphHump = new Float32Array(vCount * 3);
  for (let i = 0; i < vCount; i++) {
    const idx3 = i * 3;
    const x = pos.getX(i);
    const y = pos.getY(i);
    const z = pos.getZ(i);
    const u = z / length;
    const activeBlade = Math.sin(u * Math.PI * 0.85);
    morphWide[idx3] = x * (1.0 + 0.42 * activeBlade);
    morphWide[idx3 + 1] = y * (1.0 - 0.25 * activeBlade);
    morphWide[idx3 + 2] = z;
    const edgeDist = Math.abs(x) / 0.013;
    const rollLift = (edgeDist * edgeDist * 0.0085 - 0.0022) * activeBlade;
    const rollPinch = x * (1.0 - 0.32 * activeBlade * edgeDist);
    morphRoll[idx3] = rollPinch;
    morphRoll[idx3 + 1] = y + rollLift;
    morphRoll[idx3 + 2] = z;
    const humpBell = Math.sin(u * Math.PI);
    morphHump[idx3] = x;
    morphHump[idx3 + 1] = y + humpBell * 0.0065;
    morphHump[idx3 + 2] = z * 0.94;
  }
  geo.morphAttributes.position = [
    new THREE.BufferAttribute(morphWide, 3),
    new THREE.BufferAttribute(morphRoll, 3),
    new THREE.BufferAttribute(morphHump, 3),
  ];
  geo.computeBoundingSphere();
  const boneBase = new THREE.Bone();
  boneBase.name = "tongue_base";
  boneBase.position.set(0, -0.004, 0.044);
  const boneMid = new THREE.Bone();
  boneMid.name = "tongue_mid";
  boneMid.position.set(0, 0, length * 0.36);
  boneBase.add(boneMid);
  const boneTip = new THREE.Bone();
  boneTip.name = "tongue_tip";
  boneTip.position.set(0, 0, length * 0.36);
  boneMid.add(boneTip);
  const skinIndices = new Uint16Array(vCount * 4);
  const skinWeights = new Float32Array(vCount * 4);
  for (let i = 0; i < vCount; i++) {
    const z = pos.getZ(i);
    const u = THREE.MathUtils.clamp(z / length, 0, 1);
    let w0 = 0;
    let w1 = 0;
    let w2 = 0;
    if (u < 0.45) {
      const t = THREE.MathUtils.smoothstep(u, 0.1, 0.45);
      w0 = 1.0 - t;
      w1 = t;
    } else {
      const t = THREE.MathUtils.smoothstep(u, 0.45, 0.85);
      w1 = 1.0 - t;
      w2 = t;
    }
    const i4 = i * 4;
    skinIndices[i4] = 0;
    skinIndices[i4 + 1] = 1;
    skinIndices[i4 + 2] = 2;
    skinIndices[i4 + 3] = 0;
    skinWeights[i4] = w0;
    skinWeights[i4 + 1] = w1;
    skinWeights[i4 + 2] = w2;
    skinWeights[i4 + 3] = 0;
  }
  geo.setAttribute("skinIndex", new THREE.BufferAttribute(skinIndices, 4));
  geo.setAttribute("skinWeight", new THREE.BufferAttribute(skinWeights, 4));
  const tongueMat = new THREE.MeshPhysicalMaterial({
    color: 0xc84b59,
    roughness: 0.28,
    clearcoat: 0.78,
    clearcoatRoughness: 0.18,
  });
  const mesh = new THREE.SkinnedMesh(geo, tongueMat);
  mesh.updateMorphTargets();
  mesh.frustumCulled = false;
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  const skeleton = new THREE.Skeleton([boneBase, boneMid, boneTip]);
  mesh.add(boneBase);
  mesh.bind(skeleton);
  return { mesh, bones: [boneBase, boneMid, boneTip] };
}

export function createMouthCavity() {
  const group = new THREE.Group();
  group.name = "mouthCavity";
  const throatGeo = new THREE.SphereGeometry(0.052, 24, 16);
  throatGeo.scale(0.85, 0.75, 0.9);
  const throatMat = new THREE.MeshStandardMaterial({
    color: 0x2b0b0e,
    roughness: 0.95,
    side: THREE.BackSide,
  });
  const throat = new THREE.Mesh(throatGeo, throatMat);
  throat.position.set(0, -0.024, 0.008);
  throat.castShadow = false;
  throat.receiveShadow = false;
  group.add(throat);
  const palateGeo = new THREE.SphereGeometry(
    0.03,
    24,
    12,
    0,
    Math.PI * 2,
    0,
    Math.PI * 0.5,
  );
  palateGeo.scale(1.0, 0.55, 1.25);
  const palateMat = new THREE.MeshStandardMaterial({
    color: 0xa2545a,
    roughness: 0.6,
    side: THREE.DoubleSide,
  });
  const palate = new THREE.Mesh(palateGeo, palateMat);
  palate.rotation.x = Math.PI;
  palate.position.set(0, -0.0205, 0.036);
  palate.castShadow = false;
  palate.receiveShadow = false;
  group.add(palate);
  return { group, throat, palate };
}

export function createMouthFloor() {
  const geo = new THREE.SphereGeometry(0.024, 20, 12);
  geo.scale(1.0, 0.28, 1.35);
  const mat = new THREE.MeshStandardMaterial({
    color: 0x8e3a44,
    roughness: 0.55,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.copy(FLOOR_REL_LOWER_ARCH);
  mesh.castShadow = false;
  mesh.receiveShadow = false;
  return mesh;
}

const _v0 = new THREE.Vector3();
const _m0 = new THREE.Matrix4();
const _m1 = new THREE.Matrix4();
const _s0 = new THREE.Vector3();

function jointWorld(joint, out) {
  return out.fromArray(state.jointPositions, joint * 3);
}

/**
 * Anchor-actor eyes + oral cavity. Attach once after setRiggedMesh; the
 * upper assembly (eyes, upper teeth, throat) rides the head bone rigidly
 * while the lower assembly (lower teeth, tongue, floor) follows jawDrop plus
 * live jaw-bone motion, recomputed on every pose/blendshape evaluation.
 */
export class EyesOral {
  constructor() {
    this.attached = false;
    this.enabled = true;
    this.headBone = null;
    this.jawBone = null;
    this.eyeMaterial = null;
    this.eyeL = null;
    this.eyeR = null;
    this.upperRoot = null;
    this.lowerRoot = null;
    this.upperArch = null;
    this.lowerArch = null;
    this.tongue = null;
    this.fitted = { upper: null, lower: null, loaded: false, enabled: true, scale: FITTED_TEETH_SCALE, archGap: null };
    this.deco = {
      brow: null, lash: null, browId: null, lashId: null,
      browOn: true, lashOn: true,
      browTint: STRAND_TINT_DEFAULT, lashTint: STRAND_TINT_DEFAULT,
      browToken: 0, lashToken: 0,
    };
    this.upH = new THREE.Vector3(0, 1, 0);
    this.fwdH = new THREE.Vector3(0, 0, 1);
    this.modQuat = new THREE.Quaternion();
    this.modQuatInv = new THREE.Quaternion();
    this.lowerRest = new THREE.Matrix4();
    this.jawRestRel = new THREE.Matrix4();
    this.gaze = { yaw: 0, pitch: 0 };
    this.teeth = { count: 14, crooked: 0, gap: 0, overbite: 0, fangs: 0 };
    this.tonguePose = { out: 0, curl: 0, width: 0, roll: 0 };
    this.autoBlink = true;
    this._paused = false;
    this._blinkRaf = 0;
    this._blinkTimer = 0;
    this._exprByName = null;
  }

  get jawDrop() {
    const entry = this._exprEntry("jawDrop");
    return entry ? entry.value : 0;
  }

  _exprEntry(name) {
    if (!this._exprByName) {
      this._exprByName = new Map();
      const params = state.metadata?.groups?.expression || [];
      for (const entry of params) this._exprByName.set(entry.name, entry);
    }
    return this._exprByName.get(name) || null;
  }

  _writeExpressions(writes) {
    // writes: Array<[name, value]>. Batches into one blendshape evaluation.
    const touched = new Set();
    const actor = actorManager.getActor(0);
    const params = state.metadata?.groups?.expression || [];
    for (const [name, value] of writes) {
      const entry = this._exprEntry(name);
      if (!entry) continue;
      const clamped = THREE.MathUtils.clamp(value, entry.min, entry.max);
      entry.value = clamped;
      const slot = params.indexOf(entry);
      if (slot >= 0 && actor && actor.expressionValues.length > slot) {
        actor.expressionValues[slot] = clamped;
      }
      touched.add(entry.index);
    }
    if (!touched.size) return;
    evaluateBlendshapesClientSide();
    if (state.currentGroup === "expression") {
      const selected = currentParameter();
      if (selected && touched.has(selected.index)) showParameter(selected);
    }
  }

  attach() {
    if (this.attached || !state.skeleton || !state.boneIndexMap) return false;
    const names = state.metadata.jointNames;
    const head = names.indexOf("c_head");
    const jaw = names.indexOf("c_jaw");
    const lEye = names.indexOf("l_eye");
    const rEye = names.indexOf("r_eye");
    const crown = names.indexOf("c_head_null");
    if (head < 0 || jaw < 0 || lEye < 0 || rEye < 0 || crown < 0) return false;
    if (!state.jointPositions) evaluatePoseClientSide();
    if (!state.jointPositions) return false;
    const bones = state.skeleton.bones;
    this.headBone = bones[state.boneIndexMap[head]];
    this.jawBone = bones[state.boneIndexMap[jaw]];
    if (!this.headBone || !this.jawBone) return false;
    state.rootBone.updateWorldMatrix(true, true);

    // Head-local landmarks from live world positions (no hierarchy guess).
    const toHead = (joint, out) => {
      jointWorld(joint, out);
      return this.headBone.worldToLocal(out);
    };
    const headLocal = toHead(head, new THREE.Vector3());
    const lLocal = toHead(lEye, new THREE.Vector3());
    const rLocal = toHead(rEye, new THREE.Vector3());
    const crownLocal = toHead(crown, new THREE.Vector3());
    const eyeMid = lLocal.clone().add(rLocal).multiplyScalar(0.5);

    // Module basis in head space: X = our-left, Y = up, Z = face-forward.
    const upH = crownLocal.clone().sub(headLocal).normalize();
    const latH = lLocal.clone().sub(rLocal).normalize();
    const fwdH = new THREE.Vector3().crossVectors(latH, upH).normalize();
    if (eyeMid.clone().sub(headLocal).dot(fwdH) < 0) {
      latH.negate();
      fwdH.negate();
    }
    this.upH.copy(upH);
    this.fwdH.copy(fwdH);
    _m0.makeBasis(latH, upH, fwdH);
    this.modQuat.setFromRotationMatrix(_m0);
    this.modQuatInv.copy(this.modQuat).invert();

    // Upper assembly: module-space content pinned so the module eye
    // midpoint lands on our eye midpoint.
    this.eyeMaterial = createAnatomicalEyeMaterial();
    this.upperRoot = new THREE.Group();
    this.upperRoot.name = "eyesOralUpper";
    this.upperRoot.quaternion.copy(this.modQuat);
    this.upperRoot.position
      .copy(MODULE_EYE_MID)
      .applyQuaternion(this.modQuat)
      .negate()
      .add(eyeMid);
    this.headBone.add(this.upperRoot);

    const toModule = (headPoint) =>
      headPoint
        .clone()
        .sub(this.upperRoot.position)
        .applyQuaternion(this.modQuatInv);
    this.eyeL = new THREE.Group();
    this.eyeL.name = "eyeL";
    this.eyeL.position.copy(toModule(lLocal));
    this.eyeL.add(createEyeballMesh(this.eyeMaterial));
    this.eyeR = new THREE.Group();
    this.eyeR.name = "eyeR";
    this.eyeR.position.copy(toModule(rLocal));
    this.eyeR.add(createEyeballMesh(this.eyeMaterial));
    this.upperRoot.add(this.eyeL, this.eyeR);

    this.upperArch = createDentalArchMesh(true);
    this.upperArch.group.position.copy(UPPER_ARCH_POS);
    this.upperRoot.add(this.upperArch.group);
    this.upperRoot.add(createMouthCavity().group);

    // Lower assembly: head-parented, re-posed every update from live jaw
    // motion plus the jawDrop follow offset (both in head space).
    this.lowerRoot = new THREE.Group();
    this.lowerRoot.name = "eyesOralLower";
    this.headBone.add(this.lowerRoot);
    this.lowerArch = createDentalArchMesh(false);
    this.lowerRoot.add(this.lowerArch.group);
    this.tongue = createSkinnedTongue();
    this.tongue.mesh.position.copy(TONGUE_REL_LOWER_ARCH);
    this.lowerRoot.add(this.tongue.mesh);
    this.lowerRoot.add(createMouthFloor());

    const upperArchHead = UPPER_ARCH_POS.clone()
      .applyQuaternion(this.modQuat)
      .add(this.upperRoot.position);
    const lowerRestHead = upperArchHead
      .clone()
      .addScaledVector(this.upH, -LOWER_GAP_REST);
    this.lowerRest.compose(lowerRestHead, this.modQuat, _s0.set(1, 1, 1));

    this.headBone.updateWorldMatrix(true, false);
    this.jawBone.updateWorldMatrix(true, false);
    this.jawRestRel
      .copy(this.headBone.matrixWorld)
      .invert()
      .multiply(this.jawBone.matrixWorld);

    this.attached = true;
    this._loadFittedTeeth();
    this._loadDeco();
    this.setTeeth(this.teeth);
    this.setTongue(this.tonguePose);
    this.setGaze(0, 0, { silent: true });
    addPoseUpdateListener(() => this.update());
    addBlendshapeUpdateListener(() => this.update());
    this.update();
    this._scheduleBlink();
    return true;
  }

  update() {
    if (!this.attached) return;
    const eyeUniforms = this.eyeMaterial?.userData?.uniforms;
    if (eyeUniforms?.uTime) eyeUniforms.uTime.value = performance.now() / 1000;
    this.headBone.updateWorldMatrix(true, false);
    this.jawBone.updateWorldMatrix(true, false);
    // Jaw motion since attach, expressed in head space.
    _m0.copy(this.headBone.matrixWorld).invert().multiply(this.jawBone.matrixWorld);
    _m1.copy(_m0).multiply(_m0.copy(this.jawRestRel).invert());
    const jawDelta = _m1;
    // jawDrop follow: rigid drop + retrusion along head axes.
    const drop = THREE.MathUtils.clamp(this.jawDrop, 0, 1);
    _v0
      .copy(this.upH)
      .multiplyScalar(-drop * DROP_DOWN)
      .addScaledVector(this.fwdH, -drop * DROP_BACK);
    const follow = _m0.makeTranslation(_v0.x, _v0.y, _v0.z);
    _m0.multiplyMatrices(jawDelta, follow).multiply(this.lowerRest);
    _m0.decompose(this.lowerRoot.position, this.lowerRoot.quaternion, _s0);

    const showLower =
      this.enabled && (drop > LOWER_VISIBLE_AT || this.tonguePose.out > 0.05);
    this.lowerRoot.visible = showLower;
    this.upperRoot.visible = this.enabled;
    this.eyeL.visible = this.enabled;
    this.eyeR.visible = this.enabled;
  }

  /**
   * Swap the procedural dental arches for the fitted MakeHuman teeth
   * (user-split upper/lower). Fire-and-forget: procedural stays until
   * the GLBs arrive, and stays permanently when they fail to load.
   */
  async _loadFittedTeeth() {
    try {
      const [upper, lower] = await Promise.all([
        loadMhMesh("makehuman_system_assets_cc0/teeth/jaw_upper.glb"),
        loadMhMesh("makehuman_system_assets_cc0/teeth/jaw_lower.glb"),
      ]);
      upper.mesh.position.copy(UPPER_ARCH_POS);
      this.upperRoot.add(upper.mesh);
      this.fitted.upper = upper.mesh;
      // Preserve the authored inter-arch gap: lowerRoot already carries
      // LOWER_GAP_REST along module -Y, so only the residual applies.
      this.fitted.archGap = lower.mhCentroid.clone().sub(upper.mhCentroid).multiplyScalar(MH_DM_TO_M);
      this.lowerRoot.add(lower.mesh);
      this.fitted.lower = lower.mesh;
      this.setFittedTeethScale(this.fitted.scale);
      this.fitted.loaded = true;
      this._applyTeethVisibility();
      requestRender();
    } catch (error) {
      console.warn("[eyesOral] fitted teeth unavailable, procedural fallback:", error);
    }
  }

  setFittedTeeth(on) {
    this.fitted.enabled = on !== false;
    this._applyTeethVisibility();
    requestRender();
    return this.fitted.enabled;
  }

  setFittedTeethScale(s = FITTED_TEETH_SCALE) {
    this.fitted.scale = s;
    if (this.fitted.upper) this.fitted.upper.scale.setScalar(MH_DM_TO_M * s);
    if (this.fitted.lower) {
      this.fitted.lower.scale.setScalar(MH_DM_TO_M * s);
      if (this.fitted.archGap) {
        const gap = this.fitted.archGap.clone().multiplyScalar(s);
        gap.y += LOWER_GAP_REST;
        this.fitted.lower.position.copy(gap);
      }
    }
    requestRender();
    return s;
  }

  /**
   * Fitted MakeHuman brows + lashes on the anchor actor. Fire-and-forget
   * like the teeth: nothing shows until the OBJs arrive, and nothing
   * breaks when they fail to load.
   */
  _loadDeco() {
    this.setBrowStyle(DEFAULT_BROW_ID);
    this.setLashStyle(DEFAULT_LASH_ID);
  }

  _placeDeco(mesh, mhCentroid, anchor, refDm, proud) {
    mesh.position.copy(placeDecoPosition(mhCentroid, anchor, refDm, proud));
  }

  _disposeDeco(mesh) {
    if (!mesh) return;
    this.upperRoot?.remove(mesh);
    mesh.geometry?.dispose();
    const mat = mesh.material;
    if (mat) {
      if (mat.map) mat.map.dispose();
      mat.dispose();
    }
  }

  async _setDecoStyle(slot, id, anchor, refDm) {
    const tokenKey = slot === "brow" ? "browToken" : "lashToken";
    const mine = ++this.deco[tokenKey];
    const tintKey = slot === "brow" ? "browTint" : "lashTint";
    try {
      const card = isCardDecoId(id);
      const [{ mesh, mhCentroid }, tex] = await Promise.all([
        loadMhObjMesh(id),
        card ? loadMhCardTexture(id) : Promise.resolve(null),
      ]);
      if (mine !== this.deco[tokenKey] || !this.upperRoot) {
        mesh.geometry?.dispose();
        tex?.dispose();
        return false;
      }
      mesh.material = card
        ? new THREE.MeshStandardMaterial({
          map: tex, alphaTest: 0.5, side: THREE.DoubleSide,
          roughness: 1.0, metalness: 0.0,
          color: new THREE.Color(this.deco[tintKey]),
        })
        : new THREE.MeshStandardMaterial({
          color: new THREE.Color(this.deco[tintKey]),
          roughness: 0.9, metalness: 0.0, side: THREE.DoubleSide,
        });
      let proud = 0;
      if (slot === "brow" && card) {
        mesh.geometry.rotateX(CARD_BROW_TILT);
        proud = CARD_PROUD + CARD_BROW_PUSH;
      } else if (slot === "lash" && card) {
        proud = LASH_CARD_PROUD;
      } else if (slot === "lash") {
        proud = STRAND_LASH_PUSH;
      }
      this._placeDeco(mesh, mhCentroid, anchor, refDm, proud);
      this._disposeDeco(this.deco[slot]);
      this.deco[slot] = mesh;
      this.deco[slot === "brow" ? "browId" : "lashId"] = id;
      this.upperRoot.add(mesh);
      this._applyDecoVisibility();
      requestRender();
      return true;
    } catch (error) {
      console.warn(`[eyesOral] decoration ${id} unavailable:`, error);
      return false;
    }
  }

  setBrowStyle(id) {
    return this._setDecoStyle("brow", id, BROW_ANCHOR, BROW_REF_DM);
  }

  setLashStyle(id) {
    return this._setDecoStyle("lash", id, LASH_ANCHOR, LASH_REF_DM);
  }

  setDecoTint(slot, hex) {
    this.deco[slot === "brow" ? "browTint" : "lashTint"] = hex;
    const mesh = this.deco[slot];
    // Cards: tint multiplies the alpha texture. Strands: tint IS the color.
    if (mesh?.material?.color) mesh.material.color.set(hex);
    requestRender();
  }

  setBrowTint(hex) { this.setDecoTint("brow", hex); }
  setLashTint(hex) { this.setDecoTint("lash", hex); }

  setDecoOn(slot, on) {
    this.deco[slot === "brow" ? "browOn" : "lashOn"] = on !== false;
    this._applyDecoVisibility();
    requestRender();
  }

  setBrowOn(on) { this.setDecoOn("brow", on); }
  setLashOn(on) { this.setDecoOn("lash", on); }

  _applyDecoVisibility() {
    if (this.deco.brow) this.deco.brow.visible = this.deco.browOn && this.enabled;
    if (this.deco.lash) this.deco.lash.visible = this.deco.lashOn && this.enabled;
  }

  _applyTeethVisibility() {
    const fitted = this.fitted.enabled && this.fitted.loaded;
    if (this.upperArch) this.upperArch.group.visible = !fitted;
    if (this.lowerArch) this.lowerArch.group.visible = !fitted;
    if (this.fitted.upper) this.fitted.upper.visible = fitted;
    if (this.fitted.lower) this.fitted.lower.visible = fitted;
  }

  setEnabled(on) {
    this.enabled = Boolean(on);
    this.update();
    requestRender();
  }

  setGaze(yawDeg, pitchDeg, { silent = false } = {}) {
    this.gaze.yaw = yawDeg;
    this.gaze.pitch = pitchDeg;
    if (this.attached) {
      const pitchRad = THREE.MathUtils.degToRad(-pitchDeg);
      const yawRad = THREE.MathUtils.degToRad(yawDeg);
      this.eyeL.rotation.set(pitchRad, yawRad, 0, "YXZ");
      this.eyeR.rotation.set(pitchRad, yawRad, 0, "YXZ");
    }
    if (silent || !this.enabled) {
      if (this.attached) requestRender();
      return;
    }
    // Module +yaw looks our-left; keep lid/tissue blendshapes in sync.
    const yawL = THREE.MathUtils.clamp(yawDeg / 25, 0, 1);
    const yawR = THREE.MathUtils.clamp(-yawDeg / 25, 0, 1);
    const lookUp = THREE.MathUtils.clamp(pitchDeg / 20, 0, 1);
    const lookDown = THREE.MathUtils.clamp(-pitchDeg / 20, 0, 1);
    const lidRaise = Math.max(0, Math.min(pitchDeg / 25, 1)) * 0.3;
    const lidTighten = Math.max(0, Math.min(-pitchDeg / 25, 1)) * 0.2;
    this._writeExpressions([
      ["eyesLookLeft_L", yawL],
      ["eyesLookLeft_R", yawL],
      ["eyesLookRight_L", yawR],
      ["eyesLookRight_R", yawR],
      ["eyesLookUp_L", lookUp],
      ["eyesLookUp_R", lookUp],
      ["eyesLookDown_L", lookDown],
      ["eyesLookDown_R", lookDown],
      ["upperLidRaiser_L", lidRaise],
      ["upperLidRaiser_R", lidRaise],
      ["lidTightener_L", lidTighten],
      ["lidTightener_R", lidTighten],
    ]);
  }

  setBlink(v) {
    if (!this.attached || !this.enabled) return;
    const clamped = THREE.MathUtils.clamp(v, 0, 1);
    this._writeExpressions([
      ["eyesClosed_L", clamped],
      ["eyesClosed_R", clamped],
    ]);
  }

  blink(durationMs = 200) {
    if (!this.attached || !this.enabled || this._blinkRaf) return;
    const baseL = this._exprEntry("eyesClosed_L")?.value || 0;
    const baseR = this._exprEntry("eyesClosed_R")?.value || 0;
    const t0 = performance.now();
    const step = (now) => {
      const t = Math.min((now - t0) / durationMs, 1);
      const raw = t < 0.35 ? t / 0.35 : 1 - (t - 0.35) / 0.65;
      const curve = raw * raw * (3 - 2 * raw);
      this._writeExpressions([
        ["eyesClosed_L", Math.max(baseL, curve)],
        ["eyesClosed_R", Math.max(baseR, curve)],
      ]);
      if (t < 1) {
        this._blinkRaf = requestAnimationFrame(step);
      } else {
        this._blinkRaf = 0;
      }
    };
    this._blinkRaf = requestAnimationFrame(step);
  }

  setAutoBlink(on) {
    this.autoBlink = Boolean(on);
    if (this.autoBlink) this._scheduleBlink();
    else if (this._blinkTimer) clearTimeout(this._blinkTimer);
  }

  // Global pause (sim loop stopped): no NEW blinks. An in-flight blink
  // finishes its 200ms arc instead of freezing mid-lid.
  setPaused(paused) {
    paused = Boolean(paused);
    if (paused === this._paused) return;
    this._paused = paused;
    if (paused) {
      if (this._blinkTimer) clearTimeout(this._blinkTimer);
      this._blinkTimer = 0;
    } else if (this.autoBlink) {
      this._scheduleBlink();
    }
  }

  _scheduleBlink() {
    if (this._blinkTimer) clearTimeout(this._blinkTimer);
    if (!this.autoBlink || this._paused) return;
    this._blinkTimer = setTimeout(
      () => {
        this.blink();
        this._scheduleBlink();
      },
      2200 + Math.random() * 2600,
    );
  }

  setTeeth({ count = 14, crooked = 0, gap = 0, overbite = 0, fangs = 0 } = {}) {
    this.teeth = { count, crooked, gap, overbite, fangs };
    if (!this.attached) return;
    for (const arch of [this.upperArch, this.lowerArch]) {
      arch.teethMesh.geometry.setDrawRange(0, count * arch.idxPerTooth);
      arch.teethMesh.morphTargetInfluences[0] = crooked;
      arch.teethMesh.morphTargetInfluences[1] = gap;
      arch.teethMesh.morphTargetInfluences[2] = overbite;
      arch.teethMesh.morphTargetInfluences[3] = fangs;
    }
    requestRender();
  }

  setTongue({ out = 0, curl = 0, width = 0, roll = 0 } = {}) {
    this.tonguePose = { out, curl, width, roll };
    if (!this.attached) return;
    const [bBase, bMid, bTip] = this.tongue.bones;
    bBase.position.z = 0.044 + out * 0.018;
    bMid.position.z = 0.0158 + out * 0.008;
    bTip.position.z = 0.0158 + out * 0.008;
    bBase.rotation.set(-curl * 0.22, 0, 0);
    bMid.rotation.set(-curl * 0.52, 0, 0);
    bTip.rotation.set(-curl * 0.72, 0, 0);
    this.tongue.mesh.morphTargetInfluences[0] = width;
    this.tongue.mesh.morphTargetInfluences[1] = roll;
    this.update();
    requestRender();
  }

  setEyeShader({ pupil = 0.45, redness = 0.5, irisInHex = "#9fc7ea", irisOutHex = "#1f5499" } = {}) {
    if (!this.attached) return;
    const { uIrisIn, uIrisOut, uPupilSize, uVeins } =
      this.eyeMaterial.userData.uniforms;
    uPupilSize.value = pupil;
    uVeins.value = redness;
    uIrisIn.value.set(irisInHex);
    uIrisOut.value.set(irisOutHex);
    requestRender();
  }

  setMouthOpen(v) {
    if (!this.attached || !this.enabled) return;
    this._writeExpressions([["jawDrop", v]]);
  }
}

/** Iris colour presets (inner = peripupillary, outer = ciliary). */
export const IRIS_PRESETS = [
  { name: "Hazel", inner: "#c78530", outer: "#5c6628" },
  { name: "Blue", inner: "#9fc7ea", outer: "#1f5499" },
  { name: "Green", inner: "#b3bd4c", outer: "#1a6640" },
  { name: "Brown", inner: "#9e5c24", outer: "#38190c" },
  { name: "Grey", inner: "#b8c4cc", outer: "#475a66" },
];

export const eyesOral = new EyesOral();
