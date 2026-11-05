import * as THREE from "three";
import { state } from "./state.js";
import { requestRender, scene } from "./scene.js";
import { actorManager } from "./actor.js";

// Procedural clothing prototype (actor #0 only): garments generated from the
// body's own mesh. No garments exist as assets, so each garment reuses
// anatomy: region vertices are copied, pushed out along their normals, and
// bound to the actor-#0 skeleton with the body's own skinning weights. Because
// the weights are identical, garments track the body exactly in every pose.
// Limits of the prototype: skin-tight fit only, jagged hems (weight
// boundaries), open edges rendered DoubleSide, and garments are built from
// the body's CURRENT shaped vertices — refit after changing identity (pose
// changes need no refit, skinning handles them).

const SHIRT_THICKNESS = 0.008;
const SHORTS_THICKNESS = 0.006;
// Shirt hem hangs this far below the root (pelvis) joint; shorts waistband
// sits above it so the shirt layers over with no skin gap (different
// offsets avoid z-fighting in the overlap).
const HEM_BELOW_ROOT = 0.1;
const WAIST_ABOVE_ROOT = 0.02;
// Sleeve covers this fraction of the shoulder->elbow axis. The negative
// floor matters: cap-top verts project behind the shoulder pivot plane,
// and cutting them at 0 punches a hole in the shoulder (seen 2026-11).
const SLEEVE_T = 0.45;
const SLEEVE_T_MIN = -0.25;
// Shorts leg covers this fraction of the hip->knee axis.
const SHORTS_T = 0.55;

function torsoJoint(name) {
  return name === "root"
    || name.startsWith("c_spine")
    || name.startsWith("l_clavicle")
    || name.startsWith("r_clavicle");
}

function upperArmJoint(side, name) {
  return name.startsWith(`${side}_uparm`);
}

function upperLegJoint(side, name) {
  return name.startsWith(`${side}_upleg`);
}

function axisT(v, origin, axis, len2) {
  return v.clone().sub(origin).dot(axis) / len2;
}

// Exact joint first (twist joints share the same prefix and sit along
// the bone, so they must not win as axis pivots).
function exactOrPrefix(names, exact) {
  const i = names.indexOf(exact);
  return i >= 0 ? i : names.findIndex((n) => n.startsWith(exact));
}

export class Garment {
  constructor(def) {
    this.def = def;
    this.enabled = false;
    this.thickness = def.thickness;
    this.color = new THREE.Color(def.color);
    this.mesh = null;
    this.material = null;
  }

  _joints() {
    const names = state.metadata?.jointNames;
    if (!names || !state.jointPositions) return null;
    const need = this.def.joints(names);
    if (!need) return null;
    const pos = (j) => new THREE.Vector3().fromArray(state.jointPositions, j * 3);
    return { names, pos, ...need };
  }

  refit() {
    this._clear();
    const body = state.skinnedMesh;
    if (!body || !state.skeleton || !state.boneIndexMap) return false;
    const J = this._joints();
    if (!J) return false;
    const posAttr = body.geometry.getAttribute("position");
    const nrmAttr = body.geometry.getAttribute("normal");
    const skiAttr = body.geometry.getAttribute("skinIndex");
    const skwAttr = body.geometry.getAttribute("skinWeight");
    const index = body.geometry.getIndex();
    if (!posAttr || !nrmAttr || !skiAttr || !skwAttr || !index) return false;

    // Bone order -> joint order (skinIndex addresses bones).
    const boneToJoint = new Int32Array(state.skeleton.bones.length).fill(-1);
    for (let j = 0; j < state.boneIndexMap.length; j++) {
      boneToJoint[state.boneIndexMap[j]] = j;
    }
    const jointOf = (bone) => (bone >= 0 && bone < boneToJoint.length ? boneToJoint[bone] : -1);

    const vCount = posAttr.count;
    const keep = new Uint8Array(vCount);
    const v = new THREE.Vector3();
    const names = state.metadata.jointNames;
    const ctx = this.def.context(J);
    for (let i = 0; i < vCount; i++) {
      // Dominant ANATOMICAL joint: stray *_null end-joint weights (e.g.
      // ring_null dominating hip verts) are skipped, since region
      // selection must follow anatomy. Skinning still copies all four
      // weights untouched, so deformation stays exact.
      let best = -1;
      let bestW = 0;
      for (let k = 0; k < 4; k++) {
        const w = skwAttr.array[i * 4 + k];
        if (w <= bestW) continue;
        const j = jointOf(skiAttr.array[i * 4 + k]);
        if (j < 0 || names[j].endsWith("_null")) continue;
        bestW = w;
        best = j;
      }
      if (best < 0) continue;
      v.fromBufferAttribute(posAttr, i);
      if (this.def.select(ctx, v, best)) keep[i] = 1;
    }

    // Faces fully inside the selection (avoids stretched partial tris).
    const srcIndex = index.array;
    const faces = [];
    for (let f = 0; f < srcIndex.length; f += 3) {
      if (keep[srcIndex[f]] && keep[srcIndex[f + 1]] && keep[srcIndex[f + 2]]) {
        faces.push(srcIndex[f], srcIndex[f + 1], srcIndex[f + 2]);
      }
    }
    if (faces.length === 0) return false;

    // Compact remap + offset along normals.
    const remap = new Int32Array(vCount).fill(-1);
    const positions = [];
    const normals = [];
    const skinIndex = [];
    const skinWeight = [];
    const n = new THREE.Vector3();
    const push = (src) => {
      let dst = remap[src];
      if (dst >= 0) return dst;
      dst = positions.length / 3;
      remap[src] = dst;
      v.fromBufferAttribute(posAttr, src);
      n.fromBufferAttribute(nrmAttr, src);
      positions.push(v.x + n.x * this.thickness, v.y + n.y * this.thickness, v.z + n.z * this.thickness);
      normals.push(n.x, n.y, n.z);
      for (let k = 0; k < 4; k++) {
        skinIndex.push(skiAttr.array[src * 4 + k]);
        skinWeight.push(skwAttr.array[src * 4 + k]);
      }
      return dst;
    };
    const outIndex = new Uint32Array(faces.length);
    for (let f = 0; f < faces.length; f++) outIndex[f] = push(faces[f]);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(positions), 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(new Float32Array(normals), 3));
    geo.setAttribute("skinIndex", new THREE.BufferAttribute(new Uint16Array(skinIndex), 4));
    geo.setAttribute("skinWeight", new THREE.BufferAttribute(new Float32Array(skinWeight), 4));
    geo.setIndex(new THREE.BufferAttribute(outIndex, 1));
    geo.computeBoundingSphere();

    if (!this.material) {
      this.material = new THREE.MeshStandardMaterial({
        color: this.color.clone(),
        roughness: 0.85,
        metalness: 0,
        side: THREE.DoubleSide,
      });
    }
    this.mesh = new THREE.SkinnedMesh(geo, this.material);
    // Identical bind to the body: same skeleton, same bind matrices, so
    // the garment transforms exactly like the skin underneath it. The
    // explicit bind matrix also keeps bind() from recalculating the
    // shared skeleton's inverses (which would corrupt body skinning).
    this.mesh.bind(state.skeleton, body.bindMatrix);
    this.mesh.bindMatrixInverse.copy(body.bindMatrixInverse);
    this.mesh.castShadow = false;
    this.mesh.receiveShadow = false;
    this.mesh.visible = this.enabled;
    this.mesh.frustumCulled = false;
    // Under the anchor group (same parent as the body), so actor Scale
    // carries garments with the skin. Falls back to scene pre-rig.
    const anchorGroup = actorManager.getActor(0)?.group;
    if (anchorGroup) anchorGroup.add(this.mesh);
    else scene.add(this.mesh);
    requestRender();
    return true;
  }

  _clear() {
    if (this.mesh) {
      this.mesh.parent?.remove(this.mesh);
      this.mesh.geometry.dispose();
      this.mesh.dispose?.();
      this.mesh = null;
    }
  }

  setEnabled(on) {
    this.enabled = Boolean(on);
    if (this.enabled && !this.mesh) {
      if (!this.refit()) {
        this.enabled = false;
        return false;
      }
    }
    if (this.mesh) this.mesh.visible = this.enabled;
    requestRender();
    return this.enabled;
  }

  setThickness(t) {
    this.thickness = t;
    if (this.mesh) this.refit();
  }

  setColor(hex) {
    this.color.set(hex);
    if (this.material) {
      this.material.color.copy(this.color);
      requestRender();
    }
  }
}

const shirtDef = {
  name: "shirt",
  thickness: SHIRT_THICKNESS,
  color: "#2f8f6f",
  joints(names) {
    const root = names.indexOf("root");
    const lS = exactOrPrefix(names, "l_uparm");
    const rS = exactOrPrefix(names, "r_uparm");
    const lE = exactOrPrefix(names, "l_lowarm");
    const rE = exactOrPrefix(names, "r_lowarm");
    if (root < 0 || lS < 0 || rS < 0 || lE < 0 || rE < 0) return null;
    return { root, lS, rS, lE, rE };
  },
  context(J) {
    const torso = new Set(
      J.names.map((n, i) => (torsoJoint(n) ? i : -1)).filter((i) => i >= 0),
    );
    const lSpos = J.pos(J.lS);
    const lAxis = J.pos(J.lE).sub(lSpos);
    const rSpos = J.pos(J.rS);
    const rAxis = J.pos(J.rE).sub(rSpos);
    return {
      torso,
      names: J.names,
      hemY: J.pos(J.root).y - HEM_BELOW_ROOT,
      lSpos,
      lAxis,
      lLen2: Math.max(lAxis.lengthSq(), 1e-8),
      rSpos,
      rAxis,
      rLen2: Math.max(rAxis.lengthSq(), 1e-8),
    };
  },
  select(ctx, v, best) {
    if (ctx.torso.has(best)) return v.y > ctx.hemY;
    const name = ctx.names[best];
    if (upperArmJoint("l", name)) {
      const t = axisT(v, ctx.lSpos, ctx.lAxis, ctx.lLen2);
      return t >= SLEEVE_T_MIN && t <= SLEEVE_T;
    }
    if (upperArmJoint("r", name)) {
      const t = axisT(v, ctx.rSpos, ctx.rAxis, ctx.rLen2);
      return t >= SLEEVE_T_MIN && t <= SLEEVE_T;
    }
    return false;
  },
};

const shortsDef = {
  name: "shorts",
  thickness: SHORTS_THICKNESS,
  color: "#3a4d8f",
  joints(names) {
    const root = names.indexOf("root");
    const lH = exactOrPrefix(names, "l_upleg");
    const rH = exactOrPrefix(names, "r_upleg");
    const lK = exactOrPrefix(names, "l_lowleg");
    const rK = exactOrPrefix(names, "r_lowleg");
    if (root < 0 || lH < 0 || rH < 0 || lK < 0 || rK < 0) return null;
    return { root, lH, rH, lK, rK };
  },
  context(J) {
    const lHpos = J.pos(J.lH);
    const lKpos = J.pos(J.lK);
    const lAxis = lKpos.clone().sub(lHpos);
    const rHpos = J.pos(J.rH);
    const rKpos = J.pos(J.rK);
    const rAxis = rKpos.clone().sub(rHpos);
    return {
      names: J.names,
      root: J.root,
      waistY: J.pos(J.root).y + WAIST_ABOVE_ROOT,
      kneeY: (lKpos.y + rKpos.y) / 2,
      lHpos,
      lAxis,
      lLen2: Math.max(lAxis.lengthSq(), 1e-8),
      rHpos,
      rAxis,
      rLen2: Math.max(rAxis.lengthSq(), 1e-8),
    };
  },
  select(ctx, v, best) {
    // Hips/pelvis/crotch via the root joint, thighs via the upper legs.
    if (best === ctx.root) return v.y < ctx.waistY && v.y > ctx.kneeY;
    const name = ctx.names[best];
    if (upperLegJoint("l", name)) {
      return axisT(v, ctx.lHpos, ctx.lAxis, ctx.lLen2) <= SHORTS_T;
    }
    if (upperLegJoint("r", name)) {
      return axisT(v, ctx.rHpos, ctx.rAxis, ctx.rLen2) <= SHORTS_T;
    }
    return false;
  },
};

export const shirt = new Garment(shirtDef);
export const shorts = new Garment(shortsDef);
