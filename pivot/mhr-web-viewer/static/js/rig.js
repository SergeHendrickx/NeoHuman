import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { elements, state } from "./state.js";
import { fetchBinaryAsset } from "./cache.js";
import { formatStageMs, recordPerfMetric, recordStageTimes } from "./perf.js";
import { Actor, actorManager, syncDisplayedSkeleton } from "./actor.js";
import {
  bodyMaterial,
  fitCamera,
  requestRender,
  scene,
} from "./scene.js";

export function parseRigMapping(buffer) {
  const view = new DataView(buffer);
  const rows = view.getUint32(0, true);
  const cols = view.getUint32(4, true);
  const joints = view.getUint32(8, true);
  const floatOffset = 12;
  const tpLength = rows * cols;
  const offsetsLength = joints * 3;
  const prerotLength = joints * 4;
  const floatArray = new Float32Array(
    buffer,
    floatOffset,
    tpLength + offsetsLength + prerotLength,
  );
  return {
    tpRows: rows,
    tpCols: cols,
    jointCount: joints,
    TpMatrix: floatArray.subarray(0, tpLength),
    jointOffsets: floatArray.subarray(tpLength, tpLength + offsetsLength),
    jointPreRotations: floatArray.subarray(tpLength + offsetsLength),
  };
}

export function parseBlendshapes(buffer) {
  const view = new DataView(buffer);
  const vCount = view.getUint32(0, true);
  const idCount = view.getUint32(4, true);
  const feCount = view.getUint32(8, true);
  const totalBs = idCount + feCount;
  const neutralOffset = 12;
  const neutralLength = vCount * 4;
  const neutralVertices = new Float32Array(buffer, neutralOffset, neutralLength);
  const deltasOffset = neutralOffset + neutralLength * 4;
  const deltasLength = vCount * totalBs * 4;
  const blendshapeDeltas = new Float32Array(buffer, deltasOffset, deltasLength);

  return {
    vCount,
    idCount,
    feCount,
    totalBs,
    neutralVertices,
    blendshapeDeltas,
  };
}

export function parseLodMapping(buffer) {
  const view = new DataView(buffer);
  const targetCount = view.getUint32(0, true);
  if (targetCount === 0) return null;
  const indicesOffset = 4;
  const indicesLength = targetCount * 3;
  const sourceIndices = new Uint32Array(buffer, indicesOffset, indicesLength);
  const barycOffset = indicesOffset + indicesLength * 4;
  const barycLength = targetCount * 3;
  const barycentric = new Float32Array(buffer, barycOffset, barycLength);
  return { targetCount, sourceIndices, barycentric };
}

export function parsePoseCorrectives(buffer) {
  const view = new DataView(buffer);
  const magic = view.getUint32(0, true);
  if (magic !== 0x4350484D) {
    throw new Error("Invalid pose correctives binary format.");
  }
  const l0Nnz = view.getUint32(4, true);
  const l0InDim = view.getUint32(8, true);
  const l0OutDim = view.getUint32(12, true);
  const l2Rows = view.getUint32(16, true);
  const l2Cols = view.getUint32(20, true);
  const l2Nnz = view.getUint32(24, true);

  let offset = 28;
  const l0Rows = new Uint16Array(buffer, offset, l0Nnz);
  offset += l0Nnz * 2;
  const l0Cols = new Uint16Array(buffer, offset, l0Nnz);
  offset += l0Nnz * 2;
  const l0Weights = new Float32Array(buffer, offset, l0Nnz);
  offset += l0Nnz * 4;

  const l2Indptr = new Uint32Array(buffer, offset, l2Rows + 1);
  offset += (l2Rows + 1) * 4;
  const l2Indices = new Uint16Array(buffer, offset, l2Nnz);
  offset += l2Nnz * 2;
  const l2Data = new Float32Array(buffer, offset, l2Nnz);

  return {
    l0Nnz,
    l0InDim,
    l0OutDim,
    l0Rows,
    l0Cols,
    l0Weights,
    l2Rows,
    l2Cols,
    l2Nnz,
    l2Indptr,
    l2Indices,
    l2Data,
  };
}

function evaluatePoseCorrectivesClientSide(thetaJ, outDeltas) {
  if (!state.poseCorrectives) return;
  const {
    l0Nnz,
    l0InDim,
    l0OutDim,
    l0Rows,
    l0Cols,
    l0Weights,
    l2Rows,
    l2Indptr,
    l2Indices,
    l2Data,
  } = state.poseCorrectives;

  if (!state.poseFeatures || state.poseFeatures.length !== l0InDim) {
    state.poseFeatures = new Float32Array(l0InDim);
  }
  if (!state.poseHidden || state.poseHidden.length !== l0OutDim) {
    state.poseHidden = new Float32Array(l0OutDim);
  }
  const feats = state.poseFeatures;
  const hidden = state.poseHidden;
  const jointCount = thetaJ.length / 7;
  if ((jointCount - 2) * 6 !== l0InDim) {
    throw new Error(
      `Pose corrective feature layout mismatch: ${jointCount} joints need ` +
      `${(jointCount - 2) * 6} features, model provides ${l0InDim}.`,
    );
  }

  // 1. Extract 6D Euler rotation features for joints 2..(jointCount - 1)
  for (let j = 2; j < jointCount; j++) {
    const base = j * 7;
    const rx = thetaJ[base + 3];
    const ry = thetaJ[base + 4];
    const rz = thetaJ[base + 5];

    const cx = Math.cos(rx), sx = Math.sin(rx);
    const cy = Math.cos(ry), sy = Math.sin(ry);
    const cz = Math.cos(rz), sz = Math.sin(rz);

    const off = (j - 2) * 6;
    feats[off + 0] = cy * cz - 1.0;
    feats[off + 1] = cy * sz;
    feats[off + 2] = -sy;
    feats[off + 3] = sx * sy * cz - cx * sz;
    feats[off + 4] = sx * sy * sz + cx * cz - 1.0;
    feats[off + 5] = sx * cy;
  }

  // 2. Layer 0 SparseLinear
  hidden.fill(0);
  for (let k = 0; k < l0Nnz; k++) {
    hidden[l0Rows[k]] += l0Weights[k] * feats[l0Cols[k]];
  }

  // 3. Layer 1 ReLU
  for (let i = 0; i < l0OutDim; i++) {
    if (hidden[i] < 0) hidden[i] = 0;
  }

  // 4. Layer 2 Linear CSR SpMV (convert cm to meters: divide by 100)
  for (let r = 0; r < l2Rows; r++) {
    const start = l2Indptr[r];
    const end = l2Indptr[r + 1];
    let sum = 0.0;
    for (let idx = start; idx < end; idx++) {
      sum += l2Data[idx] * hidden[l2Indices[idx]];
    }
    outDeltas[r] = sum / 100.0;
  }
}

function evaluateBlendshapesCPU(blendshapes, weights, outPos) {
  const { vCount, totalBs, neutralVertices, blendshapeDeltas } = blendshapes;
  const activeIndices = [];
  const activeWeights = [];
  for (let b = 0; b < totalBs; b++) {
    const w = weights[b];
    if (Math.abs(w) > 1e-6) {
      activeIndices.push(b);
      activeWeights.push(w);
    }
  }
  const numActive = activeIndices.length;

  for (let v = 0; v < vCount; v++) {
    const v4 = v * 4;
    let x = neutralVertices[v4];
    let y = neutralVertices[v4 + 1];
    let z = neutralVertices[v4 + 2];
    const vBase = v * totalBs * 4;
    for (let k = 0; k < numActive; k++) {
      const b = activeIndices[k];
      const w = activeWeights[k];
      const idx = vBase + b * 4;
      x += blendshapeDeltas[idx] * w;
      y += blendshapeDeltas[idx + 1] * w;
      z += blendshapeDeltas[idx + 2] * w;
    }
    const o3 = v * 3;
    outPos[o3] = x;
    outPos[o3 + 1] = y;
    outPos[o3 + 2] = z;
  }
}

function applyLodMapping(sourceVertices, targetPos, mapping) {
  const { targetCount, sourceIndices, barycentric } = mapping;
  for (let v = 0; v < targetCount; v++) {
    const v3 = v * 3;
    const i0 = sourceIndices[v3] * 3;
    const i1 = sourceIndices[v3 + 1] * 3;
    const i2 = sourceIndices[v3 + 2] * 3;
    const b0 = barycentric[v3];
    const b1 = barycentric[v3 + 1];
    const b2 = barycentric[v3 + 2];
    targetPos[v3]     = sourceVertices[i0] * b0 + sourceVertices[i1] * b1 + sourceVertices[i2] * b2;
    targetPos[v3 + 1] = sourceVertices[i0 + 1] * b0 + sourceVertices[i1 + 1] * b1 + sourceVertices[i2 + 1] * b2;
    targetPos[v3 + 2] = sourceVertices[i0 + 2] * b0 + sourceVertices[i1 + 2] * b1 + sourceVertices[i2 + 2] * b2;
  }
}

export function evaluateBlendshapesClientSide() {
  if (!state.blendshapes || !state.skinnedMesh) return;
  const t0 = performance.now();
  const idParams = state.metadata.groups.identity;
  const exprParams = state.metadata.groups.expression;

  const totalBs = idParams.length + exprParams.length;
  if (!state.activeWeights || state.activeWeights.length !== totalBs) {
    state.activeWeights = new Float32Array(totalBs);
  }

  for (let i = 0; i < idParams.length; i++) {
    state.activeWeights[i] = idParams[i].value;
  }
  for (let j = 0; j < exprParams.length; j++) {
    state.activeWeights[idParams.length + j] = exprParams[j].value;
  }

  if (!state.unposedBase) {
    state.unposedBase = new Float32Array(state.blendshapes.vCount * 3);
  }

  const tBlend0 = performance.now();
  evaluateBlendshapesCPU(state.blendshapes, state.activeWeights, state.unposedBase);
  recordStageTimes({ blend: performance.now() - tBlend0 });
  updateMeshVertices();
  const elapsed = performance.now() - t0;
  if (elements.renderTime) {
    elements.renderTime.textContent = elapsed < 0.1 ? "< 0.1 ms" : `${elapsed.toFixed(1)} ms`;
  }
  recordPerfMetric(elapsed, "blend");
  notifyBlendshapesUpdated();
  requestRender();
}

function updateMeshVertices() {
  if (!state.skinnedMesh || !state.blendshapes) return;
  const posAttr = state.skinnedMesh.geometry.getAttribute("position");
  if (!posAttr) return;
  const tMesh0 = performance.now();

  const currentLod = state.metadata?.model?.lod ?? 1;
  const applyCorr = Boolean(
    state.metadata?.model?.applyCorrectives &&
    state.poseCorrectives &&
    state.correctiveDeltas,
  );

  if (!state.deformedTemplate) {
    state.deformedTemplate = new Float32Array(state.blendshapes.vCount * 3);
  }

  const base = state.unposedBase || state.blendshapes.neutralVertices;
  const target = state.deformedTemplate;

  if (applyCorr) {
    const corr = state.correctiveDeltas;
    for (let i = 0; i < target.length; i++) {
      target[i] = base[i] + corr[i];
    }
  } else {
    target.set(base);
  }

  let transferMs = 0;
  if (currentLod === 1 || !state.lodMappings[currentLod]) {
    if (posAttr.array.length === target.length) {
      posAttr.array.set(target);
      posAttr.needsUpdate = true;
    }
  } else {
    const mapping = state.lodMappings[currentLod];
    if (posAttr.array.length === mapping.targetCount * 3) {
      const tTransfer0 = performance.now();
      applyLodMapping(target, posAttr.array, mapping);
      transferMs = performance.now() - tTransfer0;
      posAttr.needsUpdate = true;
    }
  }

  const tNormals0 = performance.now();
  state.skinnedMesh.geometry.computeVertexNormals();
  const tBounds0 = performance.now();
  state.skinnedMesh.geometry.computeBoundingBox();
  state.skinnedMesh.geometry.computeBoundingSphere();
  state.bounds = state.skinnedMesh.geometry.boundingBox.clone();
  const tMeshEnd = performance.now();
  recordStageTimes(
    { mesh: tMeshEnd - tMesh0 },
    {
      mesh:
        `normals ${formatStageMs(tBounds0 - tNormals0)}, ` +
        `bounds ${formatStageMs(tMeshEnd - tBounds0)}, ` +
        `transfer ${formatStageMs(transferMs)}`,
    },
  );
}

const _bonePos = new THREE.Vector3();
// MHR composes joint rotations as R = Rz * Ry * Rx, which is THREE's "ZYX"
// Euler order (verified against joint_parameters_to_skeleton_state).
const _boneEuler = new THREE.Euler(0, 0, 0, "ZYX");
const _boneQEuler = new THREE.Quaternion();
const _boneQPre = new THREE.Quaternion();
const _boneScale = new THREE.Vector3();
const _jointWorldPos = new THREE.Vector3();

const poseUpdateListeners = new Set();

// Runs after every client-side pose evaluation with fresh bones and joint
// positions. Lets the joint gizmo follow without an import cycle.
export function addPoseUpdateListener(fn) {
  poseUpdateListeners.add(fn);
  return () => poseUpdateListeners.delete(fn);
}

function notifyPoseUpdated() {
  for (const fn of poseUpdateListeners) fn();
}

const blendshapeUpdateListeners = new Set();

// Runs after every client-side blendshape evaluation. Lets the eyes/mouth
// attachments follow expression state (jawDrop, blinks) without an import
// cycle, mirroring addPoseUpdateListener.
export function addBlendshapeUpdateListener(fn) {
  blendshapeUpdateListeners.add(fn);
  return () => blendshapeUpdateListeners.delete(fn);
}

function notifyBlendshapesUpdated() {
  for (const fn of blendshapeUpdateListeners) fn();
}

const meshReplaceListeners = new Set();

// Runs before setRiggedMesh disposes the outgoing mesh, so crowd clones
// (which share its geometry) can drop it first without an import cycle.
export function addMeshReplaceListener(fn) {
  meshReplaceListeners.add(fn);
  return () => meshReplaceListeners.delete(fn);
}

function notifyMeshReplacing() {
  for (const fn of meshReplaceListeners) fn();
}

// Linear rig mapping: 204 pose parameters -> 127 x 7 joint channels.
export function computeThetaJ() {
  const { TpMatrix, jointCount, tpCols } = state.rigMapping;
  const poseParams = state.metadata.groups.pose;
  const thetaP = new Float32Array(poseParams.length);
  for (let i = 0; i < poseParams.length; i++) {
    thetaP[i] = poseParams[i].value;
  }

  const thetaJ = new Float32Array(jointCount * 7);
  for (let row = 0; row < jointCount * 7; row++) {
    let sum = 0;
    const rowOffset = row * tpCols;
    for (let col = 0; col < tpCols; col++) {
      sum += TpMatrix[rowOffset + col] * thetaP[col];
    }
    thetaJ[row] = sum;
  }
  return thetaJ;
}

export function evaluatePoseClientSide() {
  if (!state.rigMapping || !state.skeleton || !state.boneIndexMap) return;
  const t0 = performance.now();
  const { jointOffsets, jointPreRotations, jointCount } = state.rigMapping;
  const thetaJ = computeThetaJ();

  const bones = state.skeleton.bones;
  const boneIndex = state.boneIndexMap;
  for (let j = 0; j < jointCount; j++) {
    const base = j * 7;
    const tx = thetaJ[base + 0];
    const ty = thetaJ[base + 1];
    const tz = thetaJ[base + 2];
    const rx = thetaJ[base + 3];
    const ry = thetaJ[base + 4];
    const rz = thetaJ[base + 5];
    const s = 2.0 ** thetaJ[base + 6];

    _bonePos.set(
      (jointOffsets[j * 3 + 0] + tx) / 100.0,
      (jointOffsets[j * 3 + 1] + ty) / 100.0,
      (jointOffsets[j * 3 + 2] + tz) / 100.0,
    );

    _boneEuler.set(rx, ry, rz);
    _boneQEuler.setFromEuler(_boneEuler);
    _boneQPre.set(
      jointPreRotations[j * 4 + 0],
      jointPreRotations[j * 4 + 1],
      jointPreRotations[j * 4 + 2],
      jointPreRotations[j * 4 + 3],
    );
    _boneQPre.multiply(_boneQEuler);

    _boneScale.set(s, s, s);

    const bone = bones[boneIndex[j]];
    bone.position.copy(_bonePos);
    bone.quaternion.copy(_boneQPre);
    bone.scale.copy(_boneScale);
  }

  state.rootBone.updateWorldMatrix(true, true);
  state.skeleton.update();
  const tFkBones = performance.now();

  const applyCorr = Boolean(
    state.metadata?.model?.applyCorrectives && state.poseCorrectives,
  );
  if (applyCorr) {
    if (!state.correctiveDeltas) {
      state.correctiveDeltas = new Float32Array(state.poseCorrectives.l2Rows);
    }
    const tCorr0 = performance.now();
    evaluatePoseCorrectivesClientSide(thetaJ, state.correctiveDeltas);
    recordStageTimes({ corr: performance.now() - tCorr0 });
    updateMeshVertices();
    state.correctivesActive = true;
  } else if (state.correctivesActive) {
    state.correctivesActive = false;
    updateMeshVertices();
    recordStageTimes({ corr: 0 });
  } else {
    recordStageTimes({ corr: 0 });
  }

  const tJoints0 = performance.now();
  if (!state.jointPositions || state.jointPositions.length !== jointCount * 3) {
    state.jointPositions = new Float32Array(jointCount * 3);
  }
  for (let j = 0; j < jointCount; j++) {
    bones[boneIndex[j]].getWorldPosition(_jointWorldPos);
    state.jointPositions[j * 3 + 0] = _jointWorldPos.x;
    state.jointPositions[j * 3 + 1] = _jointWorldPos.y;
    state.jointPositions[j * 3 + 2] = _jointWorldPos.z;
  }
  const mainActor = actorManager.getActor(0);
  if (mainActor) {
    mainActor.jointPositions.set(state.jointPositions);
  }

  syncDisplayedSkeleton();
  recordStageTimes({
    fk: tFkBones - t0 + (performance.now() - tJoints0),
  });
  notifyPoseUpdated();
  const elapsed = performance.now() - t0;
  if (elements.renderTime) {
    elements.renderTime.textContent = elapsed < 0.1 ? "< 0.1 ms" : `${elapsed.toFixed(1)} ms`;
  }
  recordPerfMetric(elapsed, "pose");
  requestRender();
}

const gltfLoader = new GLTFLoader();

export async function loadRiggedModel(lod) {
  // Topology version pins the mesh bytes: after a seal cut the URL changes
  // so no stale GLB survives in HTTP or IndexedDB caches.
  const version = state.metadata?.model?.topologyVersion ?? "unversioned";
  const url = `/api/model/rigged-glb?lod=${lod}&v=${version}`;
  const buffer = await fetchBinaryAsset(url, "Could not load rigged GLB model.");
  return new Promise((resolve, reject) => {
    gltfLoader.parse(
      buffer,
      "",
      (gltf) => {
        let skinnedMesh = null;
        let rootBone = null;
        gltf.scene.traverse((obj) => {
          if (obj.isSkinnedMesh && !skinnedMesh) skinnedMesh = obj;
          if (obj.isBone && !rootBone && (!obj.parent || !obj.parent.isBone)) {
            rootBone = obj;
          }
        });
        if (!skinnedMesh || !rootBone) {
          reject(new Error("Rigged model did not contain a SkinnedMesh and Armature."));
          return;
        }
        resolve({ gltf, skinnedMesh, rootBone });
      },
      reject,
    );
  });
}

export async function setRiggedMesh(lod, { fit = false } = {}) {
  const { gltf, skinnedMesh, rootBone } = await loadRiggedModel(lod);

  if (lod !== 1 && !state.lodMappings[lod]) {
    try {
      const buffer = await fetchBinaryAsset(
        `/api/lod-mapping?lod=${lod}`,
        `Could not load LOD ${lod} mapping.`,
      );
      state.lodMappings[lod] = parseLodMapping(buffer);
    } catch (err) {
      console.warn(`Could not load LOD ${lod} mapping:`, err);
    }
  }

  notifyMeshReplacing();

  const existingActor0 = actorManager.getActor(0);
  if (state.skinnedMesh) {
    state.skinnedMesh.parent?.remove(state.skinnedMesh);
    state.rootBone?.parent?.remove(state.rootBone);
    state.skinnedMesh.geometry.dispose();
  }
  // The previous anchor group (if any) is emptied above; drop it from the
  // graph so re-rigs don't stack orphan groups in the scene.
  if (existingActor0?.group.parent) {
    existingActor0.group.parent.remove(existingActor0.group);
  }

  state.skinnedMesh = skinnedMesh;
  state.rootBone = rootBone;
  state.skeleton = skinnedMesh.skeleton;

  // Map metadata joint order to skeleton bones by name. Bone array order is an
  // accident of exporter layout; names are the contract.
  const bonesByName = new Map(state.skeleton.bones.map((bone) => [bone.name, bone]));
  state.boneIndexMap = state.metadata.jointNames.map((name, joint) => {
    const bone = bonesByName.get(name);
    if (!bone) {
      throw new Error(`Rigged model is missing joint "${name}" (index ${joint}).`);
    }
    return state.skeleton.bones.indexOf(bone);
  });

  skinnedMesh.material = bodyMaterial;
  skinnedMesh.castShadow = true;
  skinnedMesh.receiveShadow = true;

  skinnedMesh.geometry.computeBoundingBox();
  skinnedMesh.geometry.computeBoundingSphere();
  state.bounds = skinnedMesh.geometry.boundingBox.clone();

  // Register the anchor actor as Actor #0 in actorManager
  const poseCount = state.metadata?.groups?.pose?.length || 204;
  const idCount = state.metadata?.groups?.identity?.length || 0;
  const exprCount = state.metadata?.groups?.expression?.length || 0;
  const jointCount = state.metadata?.jointNames?.length || 127;

  const initialPose = existingActor0 ? existingActor0.poseValues : new Float32Array(poseCount);
  if (!existingActor0 && state.metadata?.groups?.pose) {
    for (let i = 0; i < state.metadata.groups.pose.length; i++) {
      initialPose[i] = state.metadata.groups.pose[i].value;
    }
  }

  const actor0 = new Actor(0, "Actor #0", {
    mesh: skinnedMesh,
    rootBone: rootBone,
    skeleton: skinnedMesh.skeleton,
    material: bodyMaterial,
    boneIndexMap: state.boneIndexMap,
    rigMapping: state.rigMapping,
    poseCount,
    idCount,
    exprCount,
    jointCount,
    initialPoseValues: initialPose,
    behavior: "manual",
    color: bodyMaterial.color.clone(),
    scale: existingActor0?.scale ?? 1,
  });
  actorManager.registerActor(actor0);
  // Actor #0 renders under its actor group (identity at build, so this is a
  // no-op transform): Scale and any future group transform then apply to
  // the anchor exactly like they do to agents.
  actor0.group.add(rootBone, skinnedMesh);
  scene.add(actor0.group);

  evaluateBlendshapesClientSide();
  evaluatePoseClientSide();

  if (fit) fitCamera("front");
  requestRender();
}
