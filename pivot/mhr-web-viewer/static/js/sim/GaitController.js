import * as THREE from "three";

/**
 * Procedural locomotion and idle pose generator for MHR skeletons.
 * Evaluates walking, running, and idle motions via MHR's linear rig mapping.
 */

// Key MHR pose parameter indices for locomotion
export const GAIT_PARAMS = {
  spineTwist: 7,   // spine_twist0: [-0.9, 0.9]
  rUparmPitch: 35, // r_uparm_rz: [-1.2, 2.0]
  rElbowBend: 36,  // r_elbow_bend: [-0.5, 2.0]
  lUparmPitch: 45, // l_uparm_rz: [-1.2, 2.0]
  lElbowBend: 46,  // l_elbow_bend: [-0.5, 2.0]
  rUplegPitch: 52, // r_upleg_rz: [-2.5, 1.0]
  rKneeBend: 53,   // r_knee_bend: [0.0, 2.8]
  rFootBend: 55,   // r_foot_bend: [-1.0, 1.0]
  lUplegPitch: 61, // l_upleg_rz: [-2.5, 1.0]
  lKneeBend: 62,   // l_knee_bend: [0.0, 2.8]
  lFootBend: 64,   // l_foot_bend: [-1.0, 1.0]
};

// Reusable scratch objects
const _bonePos = new THREE.Vector3();
const _boneEuler = new THREE.Euler(0, 0, 0, "ZYX");
const _boneQEuler = new THREE.Quaternion();
const _boneQPre = new THREE.Quaternion();
const _boneScale = new THREE.Vector3();

export class GaitController {
  /**
   * Builds sparse lookup tables for the gait parameters to allow near-zero-cost
   * evaluation of thetaJ (only 13 active matrix entries).
   */
  static initializeSparseMapping(rigMapping) {
    if (this._initialized && this._rigMapping === rigMapping) return;

    const { TpMatrix, tpCols, jointCount } = rigMapping;
    this._paramEntries = new Map();
    this._activeJoints = new Set();

    const paramIndices = Object.values(GAIT_PARAMS);
    for (const p of paramIndices) {
      const entries = [];
      for (let row = 0; row < jointCount * 7; row++) {
        const weight = TpMatrix[row * tpCols + p];
        if (Math.abs(weight) > 1e-5) {
          const jointIndex = Math.floor(row / 7);
          entries.push({ row, weight, jointIndex });
          this._activeJoints.add(jointIndex);
        }
      }
      this._paramEntries.set(p, entries);
    }

    this._activeJointList = Array.from(this._activeJoints);
    this._rigMapping = rigMapping;
    this._initialized = true;
  }

  /**
   * Initializes all 127 skeleton bones to their neutral MHR rest pose.
   */
  static applyRestPoseToAllBones(bones, boneIndexMap, rigMapping) {
    const { jointOffsets, jointPreRotations, jointCount } = rigMapping;

    for (let j = 0; j < jointCount; j++) {
      _bonePos.set(
        jointOffsets[j * 3 + 0] / 100.0,
        jointOffsets[j * 3 + 1] / 100.0,
        jointOffsets[j * 3 + 2] / 100.0,
      );

      _boneQPre.set(
        jointPreRotations[j * 4 + 0],
        jointPreRotations[j * 4 + 1],
        jointPreRotations[j * 4 + 2],
        jointPreRotations[j * 4 + 3],
      );

      _boneScale.set(1, 1, 1);

      const bone = bones[boneIndexMap[j]];
      if (bone) {
        bone.position.copy(_bonePos);
        bone.quaternion.copy(_boneQPre);
        bone.scale.copy(_boneScale);
      }
    }
  }

  /**
   * Evaluates the gait cycle for an agent and applies joint rotations directly
   * to its active skeleton bones.
   *
   * @param {number} phase - Gait cycle phase in radians [0, 2pi].
   * @param {number} speedFactor - Normalized movement speed [0, 1].
   * @param {Array<THREE.Bone>} bones - The agent's cloned skeleton bones.
   * @param {Array<number>} boneIndexMap - Mapping from joint index to bone index.
   * @param {object} rigMapping - MHR linear rig mapping.
   * @param {Float32Array} thetaJ - Pre-allocated buffer of length jointCount * 7.
   */
  static gaitValues(phase, speedFactor) {
    // 1. Calculate parametric gait values based on speed
    const s = Math.min(1.0, Math.max(0.0, speedFactor));
    const isWalking = s > 0.05;

    // Sinusoidal gait curves
    const sinP = Math.sin(phase);
    const cosP = Math.cos(phase);

    // Leg swings (hip pitch): negative is forward swing, positive is back extension
    const rUpleg = isWalking ? sinP * 0.42 * s : 0;
    const lUpleg = isWalking ? -sinP * 0.42 * s : 0;

    // Knee bends: flexion during back swing
    const rKnee = isWalking ? Math.max(0, cosP) * 0.65 * s : 0;
    const lKnee = isWalking ? Math.max(0, -cosP) * 0.65 * s : 0;

    // Ankle flexions
    const rFoot = isWalking ? -sinP * 0.22 * s : 0;
    const lFoot = isWalking ? sinP * 0.22 * s : 0;

    // Arm swings: counter-phase to legs
    const rUparm = isWalking ? -sinP * 0.35 * s : 0;
    const lUparm = isWalking ? sinP * 0.35 * s : 0;

    // Elbows: natural resting bend + extra flexion during arm forward swing
    const rElbow = 0.18 + (isWalking ? Math.max(0, -sinP) * 0.3 * s : 0);
    const lElbow = 0.18 + (isWalking ? Math.max(0, sinP) * 0.3 * s : 0);

    // Spine counter-twist: subtle torso sway
    const spineTwist = isWalking
      ? sinP * 0.06 * s
      : Math.sin(phase * 0.6) * 0.015; // Gentle idle breathing sway

    const values = {
      [GAIT_PARAMS.spineTwist]: spineTwist,
      [GAIT_PARAMS.rUparmPitch]: rUparm,
      [GAIT_PARAMS.rElbowBend]: rElbow,
      [GAIT_PARAMS.lUparmPitch]: lUparm,
      [GAIT_PARAMS.lElbowBend]: lElbow,
      [GAIT_PARAMS.rUplegPitch]: rUpleg,
      [GAIT_PARAMS.rKneeBend]: rKnee,
      [GAIT_PARAMS.rFootBend]: rFoot,
      [GAIT_PARAMS.lUplegPitch]: lUpleg,
      [GAIT_PARAMS.lKneeBend]: lKnee,
      [GAIT_PARAMS.lFootBend]: lFoot,
    };
    return values;
  }

  static accumulateThetaJ(values, thetaJ, rigMapping, additive = false) {
    this.initializeSparseMapping(rigMapping);
    // 2. Accumulate non-zero entries into thetaJ (only 13 operations).
    // Additive mode layers gait onto the caller's authored base pose.
    if (!additive) thetaJ.fill(0);
    for (const [paramIdx, val] of Object.entries(values)) {
      if (Math.abs(val) < 1e-5) continue;
      const entries = this._paramEntries.get(Number(paramIdx));
      if (!entries) continue;
      for (let i = 0; i < entries.length; i++) {
        const e = entries[i];
        thetaJ[e.row] += e.weight * val;
      }
    }
  }

  static evaluateAndApply(phase, speedFactor, bones, boneIndexMap, rigMapping, thetaJ, additive = false) {
    const values = GaitController.gaitValues(phase, speedFactor);
    GaitController.accumulateThetaJ(values, thetaJ, rigMapping, additive);

    // 3. Apply to active joints
    const { jointOffsets, jointPreRotations } = rigMapping;
    const activeJoints = this._activeJointList;

    for (let i = 0; i < activeJoints.length; i++) {
      const j = activeJoints[i];
      const base = j * 7;
      const tx = thetaJ[base + 0];
      const ty = thetaJ[base + 1];
      const tz = thetaJ[base + 2];
      const rx = thetaJ[base + 3];
      const ry = thetaJ[base + 4];
      const rz = thetaJ[base + 5];
      const scaleVal = 2.0 ** thetaJ[base + 6];

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

      _boneScale.set(scaleVal, scaleVal, scaleVal);

      const bone = bones[boneIndexMap[j]];
      if (bone) {
        bone.position.copy(_bonePos);
        bone.quaternion.copy(_boneQPre);
        bone.scale.copy(_boneScale);
      }
    }
  }
}
