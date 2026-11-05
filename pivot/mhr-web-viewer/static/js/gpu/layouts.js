// Buffer layouts for the GPU crowd path (compute FK + skinning).
//
// Pure typed-array code: no THREE, no DOM, so Node can test it directly.
// All values are float32; integer quantities (joint indices, parents, skin
// indices) are stored as exact floats to keep every buffer one type.

export const GPU_MAX_AGENTS = 256;
// Ancestor-walk iterations in the FK kernel; must exceed the deepest joint
// chain (the MHR rig is well under this).
export const GPU_MAX_CHAIN = 16;

/**
 * Float offsets inside the static rig buffer (S1).
 * Concatenation: offsets | prerots | parents | inverses | skinIdx | skinW |
 * neutralNrm.
 */
export function computeRigLayout({ jointCount, vertCount, srcVertCount }) {
  const offsets = {};
  let n = 0;
  offsets.jointOffsets = n;
  n += jointCount * 3;
  offsets.jointPreRots = n;
  n += jointCount * 4;
  offsets.jointParents = n;
  n += jointCount;
  offsets.inverses = n;
  n += jointCount * 16;
  offsets.skinIdx = n;
  n += vertCount * 4;
  offsets.skinW = n;
  n += vertCount * 4;
  offsets.neutralNrm = n;
  n += srcVertCount * 3;
  return { offsets, totalFloats: n };
}

/**
 * Float offsets inside the static body buffer (S2).
 * Concatenation: neutral(V1*3) | deltas(V1*B*3, vertex-major).
 */
export function computeBodyLayout({ srcVertCount, shapeCount }) {
  const offsets = { neutral: 0, deltas: srcVertCount * 3 };
  return { offsets, totalFloats: srcVertCount * 3 + srcVertCount * shapeCount * 3 };
}

/**
 * Float offsets inside the static LOD buffer (S3).
 * Concatenation: indices(V*3) | bary(V*3).
 */
export function computeLodLayout({ targetCount }) {
  const offsets = { indices: 0, barycentric: targetCount * 3 };
  return { offsets, totalFloats: targetCount * 6 };
}

/**
 * Fills the static rig buffer. parents/skinIdx accept any integer array.
 */
export function buildRigBuffer(
  layout,
  { jointCount, vertCount, srcVertCount },
  { jointOffsets, jointPreRots, jointParents, inverses, skinIdx, skinW, neutralNrm },
) {
  const out = new Float32Array(layout.totalFloats);
  const { offsets } = layout;
  out.set(jointOffsets.subarray(0, jointCount * 3), offsets.jointOffsets);
  out.set(jointPreRots.subarray(0, jointCount * 4), offsets.jointPreRots);
  for (let j = 0; j < jointCount; j++) out[offsets.jointParents + j] = jointParents[j];
  out.set(inverses.subarray(0, jointCount * 16), offsets.inverses);
  for (let i = 0; i < vertCount * 4; i++) out[offsets.skinIdx + i] = skinIdx[i];
  out.set(skinW.subarray(0, vertCount * 4), offsets.skinW);
  out.set(neutralNrm.subarray(0, srcVertCount * 3), offsets.neutralNrm);
  return out;
}

/**
 * Fills the static body buffer, stripping the w component of the vec4 CPU
 * blendshape arrays (neutral V1*4, deltas V1*B*4).
 */
export function buildBodyBuffer(layout, { srcVertCount, shapeCount }, { neutral4, deltas4 }) {
  const out = new Float32Array(layout.totalFloats);
  const { offsets } = layout;
  for (let v = 0; v < srcVertCount; v++) {
    const s4 = v * 4;
    const d3 = offsets.neutral + v * 3;
    out[d3] = neutral4[s4];
    out[d3 + 1] = neutral4[s4 + 1];
    out[d3 + 2] = neutral4[s4 + 2];
    const srcBase = v * shapeCount * 4;
    const dstBase = offsets.deltas + v * shapeCount * 3;
    for (let b = 0; b < shapeCount; b++) {
      const s = srcBase + b * 4;
      const d = dstBase + b * 3;
      out[d] = deltas4[s];
      out[d + 1] = deltas4[s + 1];
      out[d + 2] = deltas4[s + 2];
    }
  }
  return out;
}

/**
 * Fills the static LOD buffer. A null mapping builds the identity transfer
 * (target v reads source v with weight 1), so LOD 1 needs no special case.
 */
export function buildLodBuffer(layout, { targetCount, sourceCount, mapping }) {
  const out = new Float32Array(layout.totalFloats);
  const { offsets } = layout;
  if (!mapping) {
    for (let v = 0; v < targetCount; v++) {
      const o3 = v * 3;
      const src = Math.min(v, sourceCount - 1);
      out[offsets.indices + o3] = src;
      out[offsets.indices + o3 + 1] = src;
      out[offsets.indices + o3 + 2] = src;
      out[offsets.barycentric + o3] = 1;
      out[offsets.barycentric + o3 + 1] = 0;
      out[offsets.barycentric + o3 + 2] = 0;
    }
    return out;
  }
  out.set(mapping.sourceIndices.subarray(0, targetCount * 3), offsets.indices);
  out.set(mapping.barycentric.subarray(0, targetCount * 3), offsets.barycentric);
  return out;
}

export function allocThetaBuffer(maxAgents, jointCount) {
  return new Float32Array(maxAgents * jointCount * 7);
}

export function allocWeightBuffer(maxAgents, shapeCount) {
  return new Float32Array(maxAgents * shapeCount);
}

export function allocShapeBuffer(maxAgents) {
  return new Float32Array(maxAgents);
}

export function allocBoneMega(maxAgents, jointCount) {
  return new Float32Array(maxAgents * jointCount * 16);
}

export function allocOutputMega(maxAgents, vertCount) {
  return new Float32Array(maxAgents * vertCount * 3);
}

/**
 * Per-slot biased index copy: slot s reads verts [s*V, s*V+V) of the shared
 * mega position/normal buffers. Always Uint32: slot*V exceeds 65535 quickly.
 */
export function biasSlotIndex(indexArray, slot, vertCount) {
  const out = new Uint32Array(indexArray.length);
  const base = slot * vertCount;
  for (let i = 0; i < indexArray.length; i++) out[i] = indexArray[i] + base;
  return out;
}

/**
 * Gathers neutral base positions at target-LOD resolution on the CPU, used to
 * initialize per-agent output buffers so the first render is never garbage.
 */
export function neutralOutputNormals({ targetCount, sourceCount, mapping, neutralNrm }) {
  const out = new Float32Array(targetCount * 3);
  for (let v = 0; v < targetCount; v++) {
    let i0 = v;
    let i1 = v;
    let i2 = v;
    let b0 = 1;
    let b1 = 0;
    let b2 = 0;
    if (mapping) {
      const o3 = v * 3;
      i0 = mapping.sourceIndices[o3];
      i1 = mapping.sourceIndices[o3 + 1];
      i2 = mapping.sourceIndices[o3 + 2];
      b0 = mapping.barycentric[o3];
      b1 = mapping.barycentric[o3 + 1];
      b2 = mapping.barycentric[o3 + 2];
    } else {
      i0 = Math.min(v, sourceCount - 1);
      i1 = i0;
      i2 = i0;
    }
    const o = v * 3;
    out[o] = neutralNrm[i0 * 3] * b0 + neutralNrm[i1 * 3] * b1 + neutralNrm[i2 * 3] * b2;
    out[o + 1] =
      neutralNrm[i0 * 3 + 1] * b0 + neutralNrm[i1 * 3 + 1] * b1 + neutralNrm[i2 * 3 + 1] * b2;
    out[o + 2] =
      neutralNrm[i0 * 3 + 2] * b0 + neutralNrm[i1 * 3 + 2] * b1 + neutralNrm[i2 * 3 + 2] * b2;
  }
  return out;
}

export function neutralOutputPositions({ targetCount, sourceCount, mapping, neutral4 }) {
  const out = new Float32Array(targetCount * 3);
  for (let v = 0; v < targetCount; v++) {
    let i0 = v;
    let i1 = v;
    let i2 = v;
    let b0 = 1;
    let b1 = 0;
    let b2 = 0;
    if (mapping) {
      const o3 = v * 3;
      i0 = mapping.sourceIndices[o3];
      i1 = mapping.sourceIndices[o3 + 1];
      i2 = mapping.sourceIndices[o3 + 2];
      b0 = mapping.barycentric[o3];
      b1 = mapping.barycentric[o3 + 1];
      b2 = mapping.barycentric[o3 + 2];
    } else {
      i0 = Math.min(v, sourceCount - 1);
      i1 = i0;
      i2 = i0;
    }
    const o = v * 3;
    out[o] = neutral4[i0 * 4] * b0 + neutral4[i1 * 4] * b1 + neutral4[i2 * 4] * b2;
    out[o + 1] =
      neutral4[i0 * 4 + 1] * b0 + neutral4[i1 * 4 + 1] * b1 + neutral4[i2 * 4 + 1] * b2;
    out[o + 2] =
      neutral4[i0 * 4 + 2] * b0 + neutral4[i1 * 4 + 2] * b1 + neutral4[i2 * 4 + 2] * b2;
  }
  return out;
}
