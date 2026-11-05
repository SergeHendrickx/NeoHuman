import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";

/**
 * MakeHuman fitted assets (teeth, tongue, brows, lashes).
 *
 * Asset space (MH .obj / Blender GLB exports): Y-up decimeters, +Z
 * face-forward, origin at the pelvis. The oral-cavity module space in
 * eyesOral.js is meters with the same handedness (X-left, Y-up,
 * Z-forward), so the conversion is a uniform 0.1 scale; X-sign is moot
 * because every fitted asset so far is left-right symmetric.
 */
export const MH_DM_TO_M = 0.1;

const _loader = new GLTFLoader();
const _objLoader = new OBJLoader();
const _texLoader = new THREE.TextureLoader();

export const mhAssetFileUrl = (id) => `/api/mhasset/file/${id}`;

/** Sibling-file id: same folder + stem, new suffix (".png", ".thumb"). */
export function siblingAssetId(id, suffix) {
  const slash = id.lastIndexOf("/");
  const dir = slash >= 0 ? id.slice(0, slash + 1) : "";
  const stem = id.slice(slash + 1).replace(/\.[^.]*$/, "");
  return `${dir}${stem}${suffix}`;
}

export async function fetchMhassetList() {
  const resp = await fetch("/api/mhasset/list");
  const data = await resp.json();
  return data.available ? (data.assets ?? []) : [];
}

/**
 * Load one asset mesh into oral-cavity module space. Bakes node
 * transforms, centers the geometry on its bbox centroid, scales dm->m.
 * Returns { mesh, mhCentroid } with mhCentroid in MH dm (kept so callers
 * can preserve authored inter-mesh offsets, e.g. the dental arch gap).
 */
export async function loadMhMesh(id) {
  const gltf = await _loader.loadAsync(`/api/mhasset/file/${id}`);
  let mesh = null;
  gltf.scene.updateMatrixWorld(true);
  gltf.scene.traverse((o) => {
    if (mesh || !o.isMesh) return;
    o.updateWorldMatrix(true, false);
    o.geometry.applyMatrix4(o.matrixWorld);
    o.position.set(0, 0, 0);
    o.quaternion.identity();
    o.scale.setScalar(MH_DM_TO_M);
    o.castShadow = false;
    o.receiveShadow = false;
    mesh = o;
  });
  if (!mesh) throw new Error(`No mesh in asset ${id}`);
  mesh.geometry.computeBoundingBox();
  const centroid = mesh.geometry.boundingBox.getCenter(new THREE.Vector3());
  mesh.geometry.translate(-centroid.x, -centroid.y, -centroid.z);
  return { mesh, mhCentroid: centroid };
}

/**
 * OBJ twin of loadMhMesh for the raw MakeHuman decoration sets (brows,
 * lashes, ...). Same contract: centered geometry, dm->m scale, vertex-mean
 * centroid in MH dm. Decoration OBJs ship UVs but no normals, so normals
 * are computed; UVs survive (cards need them). Material is the caller's
 * job (cards want an alpha texture, modelled strands a flat tint).
 */
export async function loadMhObjMesh(id) {
  const group = await _objLoader.loadAsync(mhAssetFileUrl(id));
  let mesh = null;
  group.updateMatrixWorld(true);
  group.traverse((o) => {
    if (mesh || !o.isMesh) return;
    o.updateWorldMatrix(true, false);
    o.geometry.applyMatrix4(o.matrixWorld);
    o.position.set(0, 0, 0);
    o.quaternion.identity();
    o.scale.setScalar(MH_DM_TO_M);
    o.castShadow = false;
    o.receiveShadow = false;
    mesh = o;
  });
  if (!mesh) throw new Error(`No mesh in asset ${id}`);
  if (!mesh.geometry.attributes.normal) mesh.geometry.computeVertexNormals();
  // Center on the vertex MEAN, not the bbox: decoration mass is skewed
  // (dense brow heads, sparse tails), and the placement refs in eyesOral
  // are means, so bbox-centering would sink styles by millimeters.
  const pos = mesh.geometry.attributes.position;
  const centroid = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    centroid.x += pos.getX(i);
    centroid.y += pos.getY(i);
    centroid.z += pos.getZ(i);
  }
  centroid.divideScalar(Math.max(pos.count, 1));
  mesh.geometry.translate(-centroid.x, -centroid.y, -centroid.z);
  return { mesh, mhCentroid: centroid };
}

/** Card alpha texture (sRGB) for a decoration style id. */
export async function loadMhCardTexture(id) {
  const tex = await _texLoader.loadAsync(mhAssetFileUrl(siblingAssetId(id, ".png")));
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

// Eye-decoration placement (module meters; node-importable for tests).
// Anchors are measured MHR anatomy (lid-rim mean, brow ridge); per-style
// placement preserves each style's authored offset from its category
// reference centroid (vertex means in MH dm, measured across all styles).
export const LASH_ANCHOR = new THREE.Vector3(0, 0.034, 0.067);
export const BROW_ANCHOR = new THREE.Vector3(0, 0.047, 0.076);
export const LASH_REF_DM = new THREE.Vector3(0, 7.284, 1.396);
export const BROW_REF_DM = new THREE.Vector3(0, 7.432, 1.414);
// Flat cards float proud of the skin (alphaTest, no z-fight); modelled
// hair clumps sink ~1mm by authorial design, so they sit at the anchor.
// Lash values keep every style clear of the corneal bulge (13.5mm ball
// + ~1mm bulge, gaze to +/-20deg): min +1.0mm neutral, +0.3mm extreme.
export const CARD_PROUD = 0.004;
export const LASH_CARD_PROUD = 0.005;
export const CARD_BROW_TILT = -30 * Math.PI / 180;
export const CARD_BROW_PUSH = -0.002;
export const STRAND_LASH_PUSH = 0.0045;
export const isCardDecoId = (id) => id.startsWith("eye_decoration_card/");

/**
 * Pure placement twin of the mesh positioning: module-space position for
 * a style centroid. eyesOral delegates; node tests assert ball clearance.
 */
export function placeDecoPosition(mhCentroid, anchor, refDm, proud = 0) {
  return anchor.clone()
    .addScaledVector(mhCentroid.clone().sub(refDm), MH_DM_TO_M)
    .add(new THREE.Vector3(0, 0, proud));
}
