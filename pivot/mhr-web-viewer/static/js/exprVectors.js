import * as THREE from "three";
import { Line2NodeMaterial } from "three/webgpu";
import { LineSegments2 } from "three/addons/lines/webgpu/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { scene } from "./scene.js";
import { state } from "./state.js";
import { addBlendshapeUpdateListener } from "./rig.js";
import { mirrorPairs, strongestVertex } from "./exprVectorMath.js";

export { mirrorPairs, strongestVertex };

// Expression vector overview: one line per expression shape, drawn on the
// 3D face from the min extreme to the max extreme of its strongest vertex.
// Blendshapes are linear by construction (neutral + w * delta), so each
// line is exactly the travel of that shape's point of max effect.
//
// Lines are fat lines (WebGPU LineSegments2 + Line2NodeMaterial: the GLSL
// LineMaterial cannot run on WebGPURenderer). Width is in pixels, so lines
// stay readable at any zoom; resolution self-syncs from the viewport each
// render. Clicking a line selects it (transform tooltip + list sync).
//
// Mirrored shapes (FACS base_L/R pairs) are kept as pairs for HIGHLIGHT
// only: hovering either line (or list row) highlights both, but the shapes
// move independently — selecting or driving one never touches the other.
// Lines live in bind space parented to the head bone, so head turns carry
// them along; identity changes rebuild the endpoints from the current
// unposed base.
const LINE_WIDTH_PX = 3;
const PICK_THRESHOLD_PX = 3;
const COL_BASE = new THREE.Color(0x4a5570);
const COL_HOVER = new THREE.Color(0xffb020);
const COL_SELECTED = new THREE.Color(0x35d0ff);

class ExprVectors {
  constructor() {
    this.lines = null;
    this.segs = [];
    this.segToExpr = [];
    this.pairOf = [];
    this.hover = -1;
    this.selected = -1;
    this.visible = false;
    this._pos = null;
    this._col = null;
    this._posBuf = null;
    this._colBuf = null;
    this._ray = new THREE.Raycaster();
    this._ray.params.Line2 = { threshold: PICK_THRESHOLD_PX };
  }

  init() {
    if (this.lines) return true;
    const bs = state.blendshapes;
    const expr = state.metadata?.groups?.expression;
    const mesh = state.skinnedMesh;
    if (!bs || !expr || !mesh) {
      console.warn("[exprVectors] init deferred: model not ready yet.");
      return false;
    }
    const posAttr = mesh.geometry.getAttribute("position");
    if (!posAttr || posAttr.array.length !== bs.vCount * 3) {
      console.warn("[exprVectors] no lines: mesh vertex order differs from blendshape table.");
      return false;
    }
    // Delta layout follows the FILE totals, not the metadata group lengths.
    const totalBs = bs.totalBs;
    const idCount = bs.idCount;
    this.pairOf = mirrorPairs(expr.map((p) => p.name));
    this.segs = [];
    for (let j = 0; j < expr.length; j++) {
      if (idCount + j >= totalBs) break;
      const v = strongestVertex(bs.blendshapeDeltas, bs.vCount, totalBs, idCount + j);
      if (v < 0) continue;
      const span = expr[j].max - expr[j].min;
      const idx = (v * totalBs + idCount + j) * 4;
      const len = Math.hypot(
        bs.blendshapeDeltas[idx], bs.blendshapeDeltas[idx + 1], bs.blendshapeDeltas[idx + 2],
      ) * Math.abs(span);
      if (len < 1e-6) continue;
      this.segs.push({ expr: j, vertex: v, min: expr[j].min, max: expr[j].max });
    }
    if (this.segs.length === 0) {
      console.warn("[exprVectors] no lines: every expression scanned degenerate.");
      return false;
    }
    // setPositions/setColors wrap (not copy) Float32Arrays, so rebuild()
    // and _recolor() mutate these arrays in place + flag the buffers.
    this._pos = new Float32Array(this.segs.length * 6);
    this._col = new Float32Array(this.segs.length * 6);
    const geo = new LineSegmentsGeometry();
    geo.setPositions(this._pos);
    geo.setColors(this._col);
    this._posBuf = geo.getAttribute("instanceStart").data;
    this._colBuf = geo.getAttribute("instanceColorStart").data;
    this.lines = new LineSegments2(
      geo,
      // Instance colors multiply the material color: white keeps them exact.
      new Line2NodeMaterial({
        color: 0xffffff,
        linewidth: LINE_WIDTH_PX,
        vertexColors: true,
        transparent: true,
        opacity: 0.95,
      }),
    );
    this.lines.frustumCulled = false;
    this.lines.visible = false;
    this.segToExpr = this.segs.map((s) => s.expr);
    const parent = this._parent();
    parent.add(this.lines);
    addBlendshapeUpdateListener(() => this.rebuild());
    this.rebuild();
    const paired = this.pairOf.filter((p, i) => p !== i).length / 2;
    console.info(
      `[exprVectors] ${this.segs.length} lines, ${paired} mirror pairs, ` +
      `parent: ${parent === scene ? "scene" : "head bone"}.`,
    );
    return true;
  }

  _parent() {
    const mesh = state.skinnedMesh;
    const names = state.metadata?.jointNames;
    const map = state.boneIndexMap;
    if (mesh && names && map) {
      const head = names.indexOf("c_head");
      const bone = head >= 0 ? mesh.skeleton?.bones?.[map[head]] : null;
      const inverse = head >= 0 ? mesh.skeleton?.boneInverses?.[map[head]] : null;
      if (bone && inverse) {
        // Lines are defined in bind space: cancel the bone's bind offset so
        // headWorld * inverse * bind lands exactly on the skinned surface.
        this.lines.matrixAutoUpdate = false;
        this.lines.matrix.copy(inverse);
        return bone;
      }
    }
    return scene;
  }

  rebuild() {
    if (!this.lines || !state.blendshapes) return;
    const bs = state.blendshapes;
    const idCount = bs.idCount;
    const totalBs = bs.totalBs;
    const base = state.unposedBase;
    const baseStride = base ? 3 : 4;
    const src = base ?? bs.neutralVertices;
    if (!src) return;
    const pos = this._pos;
    this.segs.forEach((s, i) => {
      const idx = (s.vertex * totalBs + idCount + s.expr) * 4;
      const dx = bs.blendshapeDeltas[idx];
      const dy = bs.blendshapeDeltas[idx + 1];
      const dz = bs.blendshapeDeltas[idx + 2];
      const b = s.vertex * baseStride;
      const o = i * 6;
      pos[o] = src[b] + s.min * dx;
      pos[o + 1] = src[b + 1] + s.min * dy;
      pos[o + 2] = src[b + 2] + s.min * dz;
      pos[o + 3] = src[b] + s.max * dx;
      pos[o + 4] = src[b + 1] + s.max * dy;
      pos[o + 5] = src[b + 2] + s.max * dz;
      s.travel = Math.hypot(pos[o + 3] - pos[o], pos[o + 4] - pos[o + 1], pos[o + 5] - pos[o + 2]);
    });
    this._posBuf.needsUpdate = true;
    // Box FIRST: computeBoundingSphere reuses the box center without
    // recomputing it, and the box is stale (all-zero positions) from
    // setPositions time. Sphere-only leaves a floor-point box here that
    // rejects every face ray in the raycast early-out.
    this.lines.geometry.computeBoundingBox();
    this.lines.geometry.computeBoundingSphere();
    this._recolor();
  }

  _recolor() {
    if (!this.lines) return;
    const col = this._col;
    const pair = this.hover >= 0 ? [this.hover, this.pairOf[this.hover] ?? this.hover] : [];
    this.segs.forEach((s, i) => {
      const c = pair.includes(s.expr) ? COL_HOVER : s.expr === this.selected ? COL_SELECTED : COL_BASE;
      const o = i * 6;
      col[o] = c.r; col[o + 1] = c.g; col[o + 2] = c.b;
      col[o + 3] = c.r; col[o + 4] = c.g; col[o + 5] = c.b;
    });
    this._colBuf.needsUpdate = true;
  }

  // Transform summary for the selection tooltip: extremes, live travel,
  // strongest vertex, and the mirror name (highlight-only pairing).
  describe(exprIndex) {
    const i = this.segs.findIndex((s) => s.expr === exprIndex);
    const seg = i >= 0 ? this.segs[i] : null;
    const meta = state.metadata?.groups?.expression?.[exprIndex];
    if (!seg || !meta) return null;
    const pair = this.pairOf[exprIndex] ?? exprIndex;
    const mirror = pair !== exprIndex
      ? state.metadata.groups.expression[pair]?.name ?? null
      : null;
    const p = this._pos;
    const o = i * 6;
    return {
      name: meta.name,
      min: seg.min,
      max: seg.max,
      value: meta.value ?? null,
      vertex: seg.vertex,
      travelMm: (seg.travel ?? 0) * 1000,
      deltaMm: p
        ? [(p[o + 3] - p[o]) * 1000, (p[o + 4] - p[o + 1]) * 1000, (p[o + 5] - p[o + 2]) * 1000]
        : null,
      mirror,
    };
  }

  setVisible(on) {
    this.visible = Boolean(on);
    if (this.lines) this.lines.visible = this.visible;
  }

  setHover(exprIndex) {
    const next = exprIndex ?? -1;
    if (next === this.hover) return false;
    this.hover = next;
    this._recolor();
    return true;
  }

  setSelected(exprIndex) {
    const next = exprIndex ?? -1;
    if (next === this.selected) return false;
    this.selected = next;
    this._recolor();
    return true;
  }

  // World-space endpoints of one expression's travel segment, for
  // pointer-drag scrubbing (ray-to-segment projection in app.js).
  worldSegment(exprIndex) {
    const i = this.segs.findIndex((s) => s.expr === exprIndex);
    if (i < 0 || !this.lines) return null;
    this.lines.updateWorldMatrix(true, false);
    const o = i * 6;
    const p = this._pos;
    return {
      a: new THREE.Vector3(p[o], p[o + 1], p[o + 2]).applyMatrix4(this.lines.matrixWorld),
      b: new THREE.Vector3(p[o + 3], p[o + 4], p[o + 5]).applyMatrix4(this.lines.matrixWorld),
    };
  }

  pick(ndc, camera) {
    if (!this.visible || !this.lines) return null;
    // Fat-line raycast needs the camera for its screen-space math and
    // reports the segment directly in faceIndex (no triangle index / 2).
    this._ray.camera = camera;
    this._ray.setFromCamera(ndc, camera);
    const hits = this._ray.intersectObject(this.lines);
    if (hits.length === 0 || hits[0].faceIndex === undefined) return null;
    return this.segToExpr[hits[0].faceIndex] ?? null;
  }
}

export const exprVectors = new ExprVectors();
export const initExprVectors = () => exprVectors.init();
export const setExprVectorsVisible = (on) => exprVectors.setVisible(on);
export const setExprVectorHover = (i) => exprVectors.setHover(i);
export const setExprVectorSelected = (i) => exprVectors.setSelected(i);
export const pickExprVector = (ndc, camera) => exprVectors.pick(ndc, camera);
export const describeExprVector = (i) => exprVectors.describe(i);
export const exprVectorWorldSegment = (i) => exprVectors.worldSegment(i);
