import * as THREE from "three";
import { scene } from "./scene.js";
import { crowdEyes } from "./crowdEyes.js";
import {
  DROP_BACK,
  DROP_DOWN,
  LOWER_GAP_REST,
  LOWER_VISIBLE_AT,
  MODULE_EYE_MID,
  TONGUE_REL_LOWER_ARCH,
  UPPER_ARCH_POS,
  createDentalArchMesh,
  createMouthCavity,
  createMouthFloor,
  createSkinnedTongue,
} from "./eyesOral.js";

// Inner mouths for LOD1 GPU agents: the same dental/tongue/cavity builders
// as actor #0, re-anchored under agent heads. LOD1-only by design (far mouths
// are invisible): ~12 draws per mouth, shared geometries/materials across
// agents (clones share both), so per-agent cost is transforms only.
//
// Frames mirror EyesOral.attach/update, minus jaw-bone motion (GPU agents
// have no bones): the upper assembly rides the head matrix rigidly while
// the lower assembly follows the agent's jawDrop weight. CPU agents are
// skipped (their jaw is shared with actor #0's mesh and cannot move alone).
const _headM = new THREE.Matrix4();
const _rootM = new THREE.Matrix4();
const _follow = new THREE.Matrix4();
const _lowerM = new THREE.Matrix4();
const _one = new THREE.Vector3(1, 1, 1);

export class CrowdMouth {
  constructor() {
    this._mouths = new Map();
    this._templates = null;
    this._rootLocal = null;
    this._lowerRest = null;
    this.count = 0;
  }

  _ensureTemplates() {
    if (this._templates) return;
    const upperArch = createDentalArchMesh(true);
    upperArch.group.position.copy(UPPER_ARCH_POS);
    const cavity = createMouthCavity();
    const lowerArch = createDentalArchMesh(false);
    const tongue = createSkinnedTongue();
    tongue.mesh.position.copy(TONGUE_REL_LOWER_ARCH);
    const floor = createMouthFloor();
    // Clones share geometries/materials; transforms bake in once here.
    this._templates = {
      upperArch: upperArch.group,
      cavity: cavity.group,
      lowerArch: lowerArch.group,
      tongue: tongue.mesh,
      floor,
    };
  }

  _ensureFrame() {
    if (this._rootLocal) return true;
    const frame = crowdEyes.mouthFrame();
    if (!frame) return false;
    // Module root in head space (mirrors upperRoot in EyesOral.attach).
    this._rootLocal = new THREE.Matrix4()
      .compose(frame.mid, frame.quat, _one)
      .multiply(new THREE.Matrix4().makeTranslation(
        -MODULE_EYE_MID.x, -MODULE_EYE_MID.y, -MODULE_EYE_MID.z,
      ));
    // Lower rest in module space: module Y maps to head-up by construction.
    this._lowerRest = new THREE.Matrix4().compose(
      new THREE.Vector3(
        UPPER_ARCH_POS.x, UPPER_ARCH_POS.y - LOWER_GAP_REST, UPPER_ARCH_POS.z,
      ),
      new THREE.Quaternion(),
      _one,
    );
    return true;
  }

  _create() {
    this._ensureTemplates();
    const t = this._templates;
    const root = new THREE.Group();
    root.add(t.upperArch.clone(true), t.cavity.clone(true));
    const lower = new THREE.Group();
    lower.add(t.lowerArch.clone(true));
    lower.add(t.tongue.clone());
    lower.add(t.floor.clone());
    root.add(lower);
    scene.add(root);
    return { root, lower };
  }

  update(agents) {
    const jawIdx = crowdEyes.jawDropIndex();
    if (!this._ensureFrame()) {
      this.clear();
      return;
    }
    const seen = new Set();
    for (const agent of agents) {
      if (!agent.isGpuAgent || agent.lod !== 1) continue;
      seen.add(agent.id);
      let mouth = this._mouths.get(agent.id);
      if (!mouth) {
        mouth = this._create();
        this._mouths.set(agent.id, mouth);
      }
      this._pose(mouth, agent, jawIdx);
    }
    for (const [id, mouth] of this._mouths) {
      if (seen.has(id)) continue;
      scene.remove(mouth.root);
      this._mouths.delete(id);
    }
    this.count = this._mouths.size;
  }

  _pose(mouth, agent, jawIdx) {
    if (!crowdEyes.headMatrixFor(agent, _headM)) {
      mouth.root.visible = false;
      return;
    }
    mouth.root.visible = Boolean(agent.mesh && agent.mesh.visible !== false);
    if (!mouth.root.visible) return;
    _rootM.multiplyMatrices(_headM, this._rootLocal);
    _rootM.decompose(mouth.root.position, mouth.root.quaternion, mouth.root.scale);
    const drop = jawIdx >= 0 && agent.expressionValues.length > jawIdx
      ? THREE.MathUtils.clamp(agent.expressionValues[jawIdx], 0, 1)
      : 0;
    _follow.makeTranslation(0, -drop * DROP_DOWN, -drop * DROP_BACK);
    _lowerM.multiplyMatrices(_follow, this._lowerRest);
    _lowerM.decompose(mouth.lower.position, mouth.lower.quaternion, mouth.lower.scale);
    mouth.lower.visible = drop > LOWER_VISIBLE_AT;
  }

  clear() {
    for (const mouth of this._mouths.values()) scene.remove(mouth.root);
    this._mouths.clear();
    this.count = 0;
  }
}

export const crowdMouth = new CrowdMouth();
