import * as THREE from "three";
import { MeshStandardNodeMaterial } from "three/webgpu";
import * as TSL from "three/tsl";

// Basic procedural skin for every body mesh (actor #0, CPU agents, GPU slots).
//
// A standard PBR node material whose albedo/roughness get object-space
// noise detail: large soft mottling (blood variation) plus fine grain.
// The tint still lives in `material.color` via the `materialColor` node,
// so every existing call site (body-color picker, per-agent colors, LOD
// colors) and every `.clone()` keeps working with independent tints —
// clones share the node graph, but `materialColor` resolves per material.
//
// Known limitation: GPU agents render compute-posed positions, so their
// `positionLocal` is posed (not bind) space and the mottling drifts slowly
// over articulating limbs. Amplitudes are deliberately small (±5%/±2%) so
// this reads as life, not shimmer.

export function createSkinMaterial(tint) {
  const mat = new MeshStandardNodeMaterial({
    color: tint,
    roughness: 0.58,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });
  const p = TSL.positionLocal;
  const blotch = TSL.mx_fractal_noise_float(p.mul(8.0));
  const grain = TSL.mx_noise_float(p.mul(90.0));
  const mottle = TSL.mix(0.95, 1.05, blotch).mul(TSL.mix(0.98, 1.02, grain));
  mat.colorNode = TSL.materialColor.mul(mottle);
  mat.roughnessNode = TSL.materialRoughness.add(TSL.mix(-0.06, 0.06, grain));
  return mat;
}
