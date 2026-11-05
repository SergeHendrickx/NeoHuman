import * as THREE from "three";
import { Actor } from "../actor.js";
import { GaitController } from "./GaitController.js";

export class Agent extends Actor {
  /**
   * Constructs an independent autonomous agent with its own skeleton, physics, and locomotion state.
   */
  constructor(id, sourceMesh, sourceRootBone, sourceSkeleton, rigMapping, boneIndexMap, options = {}) {
    // 1. Deep clone the bone hierarchy so this agent has independent joints
    const rootBone = sourceRootBone.clone(true);
    const clonedBonesByName = new Map();
    rootBone.traverse((obj) => {
      if (obj.isBone) clonedBonesByName.set(obj.name, obj);
    });

    const clonedBones = sourceSkeleton.bones.map((bone) => {
      const cloned = clonedBonesByName.get(bone.name);
      if (!cloned) {
        throw new Error(`Agent #${id}: Missing cloned bone "${bone.name}".`);
      }
      return cloned;
    });

    const skeleton = new THREE.Skeleton(clonedBones, sourceSkeleton.boneInverses);

    // 2. Clone SkinnedMesh (shares geometry with main mesh for zero extra VRAM)
    const material = sourceMesh.material.clone();
    const goldenRatio = 0.618033988749895;
    const hue = ((id * goldenRatio) % 1.0);
    const color = options.color || new THREE.Color().setHSL(hue, 0.65, 0.60);
    material.color.copy(color);
    material.wireframe = Boolean(sourceMesh.material.wireframe);

    const mesh = new THREE.SkinnedMesh(sourceMesh.geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.bind(skeleton, sourceMesh.bindMatrix);

    // 3. Container group
    const group = new THREE.Group();
    group.name = `agent-${id}`;
    group.add(rootBone);
    group.add(mesh);

    const scale = options.scale || (0.94 + ((id * 37) % 13) / 100);
    const maxSpeed = options.maxSpeed || (1.1 + ((id * 17) % 5) * 0.1);

    super(id, `Actor #${id}`, {
      group,
      mesh,
      rootBone,
      skeleton,
      material,
      boneIndexMap,
      rigMapping,
      poseCount: rigMapping.tpCols,
      jointCount: rigMapping.jointCount,
      initialPosition: options.initialPosition,
      maxSpeed,
      maxForce: options.maxForce || 4.5,
      scale,
      behavior: options.behavior || "wander",
      color,
      cadence: 3.3,
      gaitPhase: (id * 1.61803) % (Math.PI * 2),
    });

    // Initialize all bones to neutral MHR rest pose
    GaitController.applyRestPoseToAllBones(this.skeleton.bones, this.boneIndexMap, this.rigMapping);
    this.rootBone.updateWorldMatrix(true, true);
    this.skeleton.update();
    this.updateWorldJointPositions();
  }
}
