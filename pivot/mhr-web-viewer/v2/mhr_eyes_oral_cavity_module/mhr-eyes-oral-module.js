// mhr-eyes-oral-module.js — reusable MHR eyes + oral cavity sub-rig.
// Matches the Integration Guide in mhr_eyes_oral_cavity_module.html:
//   headBone.add(sys.eyeLeftBone, sys.eyeRightBone, sys.upperJawGroup);
//   jawBone.add(sys.lowerJawGroup);
//   sys.setGaze(...) -> { upperLidFollow, lowerLidFollow }
import * as THREE from 'three';

export const FACS_IDX = { EYE_LOOK_UP: 0, EYE_LOOK_DOWN: 1, JAW_OPEN: 2 };

const EYE_VERT = /* glsl */`
  varying vec3 vNormalView;
  varying vec3 vViewPos;
  varying vec3 vLocalPos;
  varying vec2 vUv;
  void main() {
    vUv = uv;
    vLocalPos = position;
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vViewPos = -mvPosition.xyz;
    vNormalView = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const EYE_FRAG = /* glsl */`
  uniform vec3 uIrisColor;
  uniform float uPupilSize;
  uniform float uScleraRedness;
  varying vec3 vNormalView;
  varying vec3 vViewPos;
  varying vec3 vLocalPos;
  varying vec2 vUv;
  void main() {
    vec3 N = normalize(vNormalView);
    vec3 V = normalize(vViewPos);
    vec3 dir = normalize(vLocalPos);
    float radialDist = length(dir.xy);
    float angle = atan(dir.y, dir.x);
    vec2 refractedXY = dir.xy - V.xy * 0.06;
    float rRefract = length(refractedXY);
    float irisRadius = 0.47;
    float pupilRadius = mix(0.08, 0.34, uPupilSize);
    vec3 color;
    if (dir.z > 0.0 && rRefract < irisRadius) {
      if (rRefract < pupilRadius) {
        color = vec3(0.01, 0.01, 0.015);
      } else {
        float t = (rRefract - pupilRadius) / (irisRadius - pupilRadius);
        float fibers = sin(angle * 70.0 + sin(t * 12.0)) * 0.5 + 0.5;
        float crypts = sin(angle * 23.0 - t * 8.0) * 0.5 + 0.5;
        vec3 innerCol = uIrisColor * 1.35 + vec3(0.12, 0.06, 0.0);
        vec3 outerCol = uIrisColor * (0.65 + 0.35 * fibers);
        color = mix(innerCol, outerCol, smoothstep(0.0, 0.85, t));
        color *= mix(0.78, 1.15, crypts);
        float limbus = smoothstep(0.75, 1.0, t);
        color = mix(color, uIrisColor * 0.18, limbus);
      }
    } else {
      float veinPattern = abs(sin(angle * 17.0 + radialDist * 18.0) * cos(angle * 11.0));
      float veinMask = smoothstep(0.52, 0.95, radialDist) * pow(veinPattern, 5.0);
      vec3 baseSclera = vec3(0.95, 0.93, 0.91);
      vec3 bloodTint = vec3(0.82, 0.22, 0.22);
      color = mix(baseSclera, bloodTint, uScleraRedness * 0.35 + veinMask * uScleraRedness * 0.55);
      float borderSoft = smoothstep(irisRadius, irisRadius + 0.04, rRefract);
      if (dir.z > 0.0) color *= mix(0.6, 1.0, borderSoft);
    }
    vec3 L1 = normalize(vec3(0.5, 0.7, 1.0));
    vec3 L2 = normalize(vec3(-0.6, 0.2, 0.8));
    float diff = max(dot(N, L1), 0.0) * 0.75 + max(dot(N, L2), 0.0) * 0.35 + 0.25;
    vec3 H1 = normalize(L1 + V);
    float spec1 = pow(max(dot(N, H1), 0.0), 180.0) * 1.35;
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 4.0) * 0.25;
    vec3 finalColor = color * diff + vec3(spec1 + fresnel);
    gl_FragColor = vec4(finalColor, 1.0);
    #include <tonemapping_fragment>
    #include <colorspace_fragment>
  }
`;

export function createEyeMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader: EYE_VERT,
    fragmentShader: EYE_FRAG,
    uniforms: {
      uIrisColor: { value: new THREE.Color('#2e77bb') },
      uPupilSize: { value: 0.45 },
      uScleraRedness: { value: 0.15 }
    }
  });
}

function createEyeballRig(xOffset, name, material) {
  const eyeBone = new THREE.Bone();
  eyeBone.name = name;
  eyeBone.position.set(xOffset, 0.032, 0.055);
  const geo = new THREE.SphereGeometry(0.0125, 48, 32);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const z = pos.getZ(i);
    if (z > 0.0105) pos.setZ(i, z + (z - 0.0105) * 0.45);
  }
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, material);
  eyeBone.add(mesh);
  return { bone: eyeBone, mesh };
}

// --- Eyelids: shutter-pivot caps parented to HEAD (not eye) so gaze stays independent ---
function createEyelidForEye(xOffset, lidMaterial) {
  const group = new THREE.Group();
  group.position.set(xOffset, 0.032, 0.055);

  const upperPivot = new THREE.Group();
  const lowerPivot = new THREE.Group();
  group.add(upperPivot, lowerPivot);

  // Upper lid: cap facing forward, hinge at top. Open = rotated up/back, closed = drops down.
  const upperGeo = new THREE.SphereGeometry(0.0146, 32, 16, 0, Math.PI * 2, 0, Math.PI * 0.52);
  upperGeo.rotateX(Math.PI * 0.5 - 0.35);
  const upperMesh = new THREE.Mesh(upperGeo, lidMaterial);
  upperMesh.position.set(0, 0.0012, 0.0012);
  upperPivot.add(upperMesh);

  const lowerGeo = new THREE.SphereGeometry(0.0146, 32, 16, 0, Math.PI * 2, Math.PI * 0.55, Math.PI * 0.45);
  lowerGeo.rotateX(Math.PI * 0.5 + 0.35);
  const lowerMesh = new THREE.Mesh(lowerGeo, lidMaterial);
  lowerMesh.position.set(0, -0.0012, 0.0012);
  lowerPivot.add(lowerMesh);

  return { group, upperPivot, lowerPivot, upperMesh, lowerMesh };
}

function createDentalArchMesh(isUpper) {
  const maxTeeth = 14;
  const baseCrownGeo = new THREE.BoxGeometry(1, 1, 1, 3, 3, 3);
  const bPos = baseCrownGeo.attributes.position;
  for (let i = 0; i < bPos.count; i++) {
    const vx = bPos.getX(i), vy = bPos.getY(i), vz = bPos.getZ(i);
    const rootFactor = isUpper ? (vy + 0.5) : (0.5 - vy);
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
      const angle = (u / 6.8) * (Math.PI * 0.46) * sign;
      const archWidth = 0.025, archDepth = 0.032;
      const tx = Math.sin(angle) * archWidth;
      const tz = Math.cos(angle) * archDepth;
      let w = 0.0062, h = 0.0085, d = 0.0042;
      if (pairIdx === 0) { w = 0.0072; h = 0.0092; d = 0.0038; }
      else if (pairIdx === 1) { w = 0.0058; h = 0.0086; d = 0.0038; }
      else if (pairIdx === 2) { w = 0.0064; h = 0.0098; d = 0.0046; }
      else if (pairIdx >= 5) { w = 0.0076; h = 0.0075; d = 0.0068; }
      if (!isUpper && pairIdx <= 1) w *= 0.82;
      const ty = isUpper ? (-h * 0.45) : (h * 0.45);
      dummy.position.set(tx, ty, tz);
      dummy.rotation.set(0, angle, 0);
      dummy.scale.set(w, h, d);
      dummy.updateMatrix();
      const vOffset = toothSlot * vertsPerTooth;
      const iOffset = toothSlot * idxPerTooth;
      for (let k = 0; k < idxPerTooth; k++) indices[iOffset + k] = baseCrownGeo.index.array[k] + vOffset;
      const seed = (pairIdx + 1) * 1.7 + (sideIdx * 3.1) + (isUpper ? 0.5 : 2.2);
      const crookedYaw = Math.sin(seed * 2.3) * 0.42 * (pairIdx <= 3 ? 1.0 : 0.4);
      const crookedPitch = Math.cos(seed * 1.9) * 0.22;
      const crookedShiftZ = (pairIdx % 2 === 0 ? 1 : -1) * 0.0022 * (pairIdx <= 2 ? 1.0 : 0.3);
      const crookedDummy = dummy.clone();
      crookedDummy.rotation.y += crookedYaw;
      crookedDummy.rotation.x += crookedPitch;
      crookedDummy.position.z += crookedShiftZ;
      crookedDummy.updateMatrix();
      for (let v = 0; v < vertsPerTooth; v++) {
        const idx3 = (vOffset + v) * 3;
        const lx = bPos.getX(v), ly = bPos.getY(v), lz = bPos.getZ(v);
        vTemp.set(lx, ly, lz).applyMatrix4(dummy.matrix);
        neutralPositions[idx3] = vTemp.x; neutralPositions[idx3 + 1] = vTemp.y; neutralPositions[idx3 + 2] = vTemp.z;
        nTemp.fromBufferAttribute(baseCrownGeo.attributes.normal, v).transformDirection(dummy.matrix);
        normals[idx3] = nTemp.x; normals[idx3 + 1] = nTemp.y; normals[idx3 + 2] = nTemp.z;
        const cPos = new THREE.Vector3(lx, ly, lz).applyMatrix4(crookedDummy.matrix);
        morphCrooked[idx3] = cPos.x; morphCrooked[idx3 + 1] = cPos.y; morphCrooked[idx3 + 2] = cPos.z;
        const gapFactor = pairIdx === 0 ? 0.0028 : (pairIdx === 1 ? 0.0014 : 0.0);
        morphGap[idx3] = vTemp.x + sign * gapFactor; morphGap[idx3 + 1] = vTemp.y; morphGap[idx3 + 2] = vTemp.z;
        const archShift = isUpper ? 0.0045 : -0.0045;
        morphOverbite[idx3] = vTemp.x; morphOverbite[idx3 + 1] = vTemp.y; morphOverbite[idx3 + 2] = vTemp.z + archShift;
        let fangY = 0, fangTapX = vTemp.x, fangTapZ = vTemp.z;
        if (pairIdx === 2) {
          const tipMask = isUpper ? Math.max(0, -ly + 0.2) : Math.max(0, ly + 0.2);
          fangY = (isUpper ? -1 : 1) * tipMask * 0.0075;
          fangTapX = THREE.MathUtils.lerp(vTemp.x, dummy.position.x, tipMask * 0.55);
          fangTapZ = THREE.MathUtils.lerp(vTemp.z, dummy.position.z, tipMask * 0.55);
        }
        morphFangs[idx3] = fangTapX; morphFangs[idx3 + 1] = vTemp.y + fangY; morphFangs[idx3 + 2] = fangTapZ;
      }
    }
  }
  const archGeo = new THREE.BufferGeometry();
  archGeo.setAttribute('position', new THREE.BufferAttribute(neutralPositions, 3));
  archGeo.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  archGeo.setIndex(new THREE.BufferAttribute(indices, 1));
  archGeo.morphAttributes.position = [
    new THREE.BufferAttribute(morphCrooked, 3),
    new THREE.BufferAttribute(morphGap, 3),
    new THREE.BufferAttribute(morphOverbite, 3),
    new THREE.BufferAttribute(morphFangs, 3)
  ];
  archGeo.computeBoundingSphere();
  const gumCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.028, isUpper ? 0.004 : -0.004, -0.002),
    new THREE.Vector3(-0.025, isUpper ? 0.004 : -0.004, 0.005),
    new THREE.Vector3(-0.019, isUpper ? 0.004 : -0.004, 0.024),
    new THREE.Vector3(0.0, isUpper ? 0.004 : -0.004, 0.033),
    new THREE.Vector3(0.019, isUpper ? 0.004 : -0.004, 0.024),
    new THREE.Vector3(0.025, isUpper ? 0.004 : -0.004, 0.005),
    new THREE.Vector3(0.028, isUpper ? 0.004 : -0.004, -0.002)
  ]);
  const gumGeo = new THREE.TubeGeometry(gumCurve, 40, 0.0058, 12, false);
  // Close tube ends with gum bulbs so no holes show behind molars
  const capGeo = new THREE.SphereGeometry(0.0058, 12, 10);
  const gumMat = new THREE.MeshPhysicalMaterial({ color: 0xd96b75, roughness: 0.32, clearcoat: 0.65, clearcoatRoughness: 0.2 });
  const gumMesh = new THREE.Mesh(gumGeo, gumMat);
  const capL = new THREE.Mesh(capGeo, gumMat);
  capL.position.copy(gumCurve.getPoint(0));
  const capR = new THREE.Mesh(capGeo, gumMat);
  capR.position.copy(gumCurve.getPoint(1));
  const enamelMat = new THREE.MeshPhysicalMaterial({ color: 0xf8f5eb, roughness: 0.16, clearcoat: 0.85, clearcoatRoughness: 0.12, reflectivity: 0.7 });
  const teethMesh = new THREE.Mesh(archGeo, enamelMat);
  teethMesh.updateMorphTargets();
  const group = new THREE.Group();
  group.add(gumMesh, capL, capR, teethMesh);
  return { group, teethMesh, gumMesh, idxPerTooth };
}

function createSkinnedTongueModule() {
  const length = 0.044;
  const segsZ = 24, segsX = 14, segsY = 6;
  const geo = new THREE.BoxGeometry(0.028, 0.009, length, segsX, segsY, segsZ);
  geo.translate(0, 0, length * 0.5);
  const pos = geo.attributes.position;
  const vCount = pos.count;
  for (let i = 0; i < vCount; i++) {
    let x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const u = z / length;
    const widthProfile = Math.sin(Math.acos(Math.min(0.98, Math.max(0, u * 0.92))));
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
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    const u = z / length;
    const activeBlade = Math.sin(u * Math.PI * 0.85);
    morphWide[idx3] = x * (1.0 + 0.42 * activeBlade);
    morphWide[idx3 + 1] = y * (1.0 - 0.25 * activeBlade);
    morphWide[idx3 + 2] = z;
    const edgeDist = Math.abs(x) / 0.013;
    const rollLift = (edgeDist * edgeDist * 0.0085 - 0.0022) * activeBlade;
    const rollPinch = x * (1.0 - 0.32 * activeBlade * edgeDist);
    morphRoll[idx3] = rollPinch; morphRoll[idx3 + 1] = y + rollLift; morphRoll[idx3 + 2] = z;
    const humpBell = Math.sin(u * Math.PI);
    morphHump[idx3] = x; morphHump[idx3 + 1] = y + humpBell * 0.0065; morphHump[idx3 + 2] = z * 0.94;
  }
  geo.morphAttributes.position = [
    new THREE.BufferAttribute(morphWide, 3),
    new THREE.BufferAttribute(morphRoll, 3),
    new THREE.BufferAttribute(morphHump, 3)
  ];
  geo.computeBoundingSphere();
  const boneBase = new THREE.Bone(); boneBase.name = 'tongue_base';
  boneBase.position.set(0, -0.004, 0.044);
  const boneMid = new THREE.Bone(); boneMid.name = 'tongue_mid';
  boneMid.position.set(0, 0, length * 0.36); boneBase.add(boneMid);
  const boneTip = new THREE.Bone(); boneTip.name = 'tongue_tip';
  boneTip.position.set(0, 0, length * 0.36); boneMid.add(boneTip);
  const skinIndices = new Uint16Array(vCount * 4);
  const skinWeights = new Float32Array(vCount * 4);
  for (let i = 0; i < vCount; i++) {
    const z = pos.getZ(i);
    const u = THREE.MathUtils.clamp(z / length, 0, 1);
    let w0 = 0, w1 = 0, w2 = 0;
    if (u < 0.45) { const t = THREE.MathUtils.smoothstep(u, 0.1, 0.45); w0 = 1.0 - t; w1 = t; }
    else { const t = THREE.MathUtils.smoothstep(u, 0.45, 0.85); w1 = 1.0 - t; w2 = t; }
    const i4 = i * 4;
    skinIndices[i4] = 0; skinIndices[i4 + 1] = 1; skinIndices[i4 + 2] = 2; skinIndices[i4 + 3] = 0;
    skinWeights[i4] = w0; skinWeights[i4 + 1] = w1; skinWeights[i4 + 2] = w2; skinWeights[i4 + 3] = 0;
  }
  geo.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, 4));
  geo.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, 4));
  const tongueMat = new THREE.MeshPhysicalMaterial({ color: 0xc84b59, roughness: 0.28, clearcoat: 0.78, clearcoatRoughness: 0.18 });
  const mesh = new THREE.SkinnedMesh(geo, tongueMat);
  mesh.updateMorphTargets();
  const skeleton = new THREE.Skeleton([boneBase, boneMid, boneTip]);
  mesh.add(boneBase);
  mesh.bind(skeleton);
  return { mesh, bones: [boneBase, boneMid, boneTip] };
}

function createMouthCavity() {
  const group = new THREE.Group();
  group.name = 'mouthCavity';
  // Dark inner throat shell — BackSide so you see darkness through open mouth, never through cheeks
  const throatGeo = new THREE.SphereGeometry(0.058, 24, 16);
  throatGeo.scale(0.85, 0.75, 0.9);
  const throatMat = new THREE.MeshStandardMaterial({ color: 0x2b0b0e, roughness: 0.95, side: THREE.BackSide });
  const throat = new THREE.Mesh(throatGeo, throatMat);
  throat.position.set(0, -0.028, 0.008);
  group.add(throat);
  // Hard palate vault above upper teeth
  const palateGeo = new THREE.SphereGeometry(0.03, 24, 12, 0, Math.PI * 2, 0, Math.PI * 0.5);
  palateGeo.scale(1.0, 0.55, 1.25);
  const palateMat = new THREE.MeshStandardMaterial({ color: 0xa2545a, roughness: 0.6, side: THREE.DoubleSide });
  const palate = new THREE.Mesh(palateGeo, palateMat);
  palate.rotation.x = Math.PI;
  palate.position.set(0, -0.0205, 0.036);
  group.add(palate);
  return { group, throat, palate };
}

function createMouthFloor() {
  const geo = new THREE.SphereGeometry(0.024, 20, 12);
  geo.scale(1.0, 0.28, 1.35);
  const mat = new THREE.MeshStandardMaterial({ color: 0x8e3a44, roughness: 0.55 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(0, -0.0085, 0.055);
  return mesh;
}

// ============================================================================
// Public factory — this is what the Integration Guide promises
// ============================================================================
export function createMHREyesAndMouthModule(options = {}) {
  const eyeMaterial = createEyeMaterial();
  const lidMaterial = new THREE.MeshStandardMaterial({ color: 0xdb9f83, roughness: 0.62, side: THREE.DoubleSide });

  const eyeLeft = createEyeballRig(-0.031, 'eye_l', eyeMaterial);
  const eyeRight = createEyeballRig(0.031, 'eye_r', eyeMaterial);
  const eyeLeftBone = eyeLeft.bone;
  const eyeRightBone = eyeRight.bone;

  const lidLeft = createEyelidForEye(-0.031, lidMaterial);
  const lidRight = createEyelidForEye(0.031, lidMaterial);

  // Upper assembly rides on head; lower assembly rides on jaw
  const upperJawGroup = new THREE.Group();
  upperJawGroup.name = 'upperJawGroup';
  const lowerJawGroup = new THREE.Group();
  lowerJawGroup.name = 'lowerJawGroup';

  const upperArch = createDentalArchMesh(true);
  upperArch.group.position.set(0, -0.028, 0.036);
  upperJawGroup.add(upperArch.group);

  const lowerArch = createDentalArchMesh(false);
  lowerArch.group.position.set(0, -0.005, 0.061);
  lowerJawGroup.add(lowerArch.group);

  const cavity = createMouthCavity();
  upperJawGroup.add(cavity.group);

  const floor = createMouthFloor();
  lowerJawGroup.add(floor);

  const tongue = createSkinnedTongueModule();
  lowerJawGroup.add(tongue.mesh);

  const state = {
    gaze: { yaw: 0, pitch: 0, verg: 0 },
    blink: 0, lidOpen: 1, squint: 0,
    jawDeg: 12,
    teeth: { count: 14, crooked: 0, gap: 0, overbite: 0, fangs: 0 },
    tonguePose: { out: 0, curl: 0, swing: 0, width: 0, roll: 0, thick: 0 },
    occlusionEnabled: true,
    mouthVisible: true
  };

  function setGaze(yawDeg = 0, pitchDeg = 0, vergDeg = 0) {
    state.gaze = { yaw: yawDeg, pitch: pitchDeg, verg: vergDeg };
    const pitchRad = THREE.MathUtils.degToRad(-pitchDeg);
    const yawRad = THREE.MathUtils.degToRad(yawDeg);
    const vergRad = THREE.MathUtils.degToRad(vergDeg);
    eyeLeftBone.rotation.set(pitchRad, yawRad + vergRad, 0, 'YXZ');
    eyeRightBone.rotation.set(pitchRad, yawRad - vergRad, 0, 'YXZ');
    const upperLidFollow = THREE.MathUtils.clamp(pitchDeg / 25, -1, 1);
    const lowerLidFollow = THREE.MathUtils.clamp(-pitchDeg / 25, -1, 1);
    applyEyelids();
    return { upperLidFollow, lowerLidFollow };
  }

  function applyEyelids() {
    // blink 0..1 (1 = fully closed), lidOpen 0..1, squint 0..1 narrows aperture
    const blinkClose = THREE.MathUtils.clamp(state.blink, 0, 1);
    const gazeDroop = THREE.MathUtils.clamp(-state.gaze.pitch / 25, -0.35, 0.55);
    // aperture: 1 = wide open, 0 = shut
    let aperture = THREE.MathUtils.clamp(state.lidOpen - blinkClose, 0, 1);
    aperture *= (1.0 - state.squint * 0.55);
    for (const lid of [lidLeft, lidRight]) {
      // open pose vs closed pose, tuned so open clears the pupil
      const upOpen = -0.95, upClosed = 0.42;
      const loOpen = 0.72, loClosed = -0.28;
      lid.upperPivot.rotation.x = THREE.MathUtils.lerp(upClosed, upOpen, aperture) + gazeDroop * 0.35;
      lid.lowerPivot.rotation.x = THREE.MathUtils.lerp(loClosed, loOpen, aperture) + gazeDroop * -0.2;
    }
  }

  function setBlink(v) { state.blink = THREE.MathUtils.clamp(v, 0, 1); applyEyelids(); }
  function setLids(open01 = 1, squint = 0) {
    state.lidOpen = THREE.MathUtils.clamp(open01, 0, 1);
    state.squint = THREE.MathUtils.clamp(squint, 0, 1);
    applyEyelids();
  }

  function setJaw(jawDeg) {
    state.jawDeg = jawDeg;
    return updateOcclusion();
  }

  function updateOcclusion(jawDeg = state.jawDeg, mouthOpenBeta = 0) {
    const vis = !state.occlusionEnabled || jawDeg > 1.0 || mouthOpenBeta > 0.03 || state.tonguePose.out > 0.05;
    state.mouthVisible = vis;
    upperJawGroup.visible = vis;
    lowerJawGroup.visible = vis;
    // Eyelids stay visible — they are NOT gated. Only oral groups cull.
    return vis;
  }

  function setTeeth({ count = 14, crooked = 0, gap = 0, overbite = 0, fangs = 0 } = {}) {
    state.teeth = { count, crooked, gap, overbite, fangs };
    const drawCount = count * upperArch.idxPerTooth;
    let skipped = true;
    for (const arch of [upperArch, lowerArch]) {
      if (!arch.teethMesh.visible) continue;
      skipped = false;
      arch.teethMesh.geometry.setDrawRange(0, drawCount);
      arch.teethMesh.morphTargetInfluences[0] = crooked;
      arch.teethMesh.morphTargetInfluences[1] = gap;
      arch.teethMesh.morphTargetInfluences[2] = overbite;
      arch.teethMesh.morphTargetInfluences[3] = fangs;
    }
    return !skipped;
  }

  function setTongue({ out = 0, curl = 0, swing = 0, width = 0, roll = 0, thick = 0 } = {}) {
    state.tonguePose = { out, curl, swing, width, roll, thick };
    if (!state.mouthVisible && state.occlusionEnabled) return false; // culled — skip skinning cost
    const [bBase, bMid, bTip] = tongue.bones;
    bBase.position.z = 0.044 + out * 0.018;
    bMid.position.z = 0.0158 + out * 0.008;
    bTip.position.z = 0.0158 + out * 0.008;
    bBase.rotation.set(-curl * 0.22, swing * 0.35, 0);
    bMid.rotation.set(-curl * 0.52, swing * 0.42, 0);
    bTip.rotation.set(-curl * 0.72, swing * 0.35, 0);
    tongue.mesh.morphTargetInfluences[0] = width;
    tongue.mesh.morphTargetInfluences[1] = roll;
    tongue.mesh.morphTargetInfluences[2] = thick;
    return true;
  }

  function setEyeShader({ pupil = 0.45, redness = 0.15, irisHex = '#2e77bb' } = {}) {
    eyeMaterial.uniforms.uPupilSize.value = pupil;
    eyeMaterial.uniforms.uScleraRedness.value = redness;
    eyeMaterial.uniforms.uIrisColor.value.set(irisHex);
  }

  // Initial pose
  setGaze(0, 0, 0);
  setTeeth(state.teeth);

  return {
    eyeLeftBone, eyeRightBone,
    eyeLeftMesh: eyeLeft.mesh, eyeRightMesh: eyeRight.mesh,
    eyeMaterial,
    lidLeft, lidRight,
    upperJawGroup, lowerJawGroup,
    upperArch, lowerArch,
    tongueMesh: tongue.mesh, tongueBones: tongue.bones,
    cavity: cavity.group, mouthFloor: floor,
    state,
    setGaze, setBlink, setLids, setJaw, setTeeth, setTongue, setEyeShader,
    updateOcclusion, applyEyelids
  };
}
