import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import * as THREE from "three";
import {
  LASH_ANCHOR,
  LASH_CARD_PROUD,
  LASH_REF_DM,
  STRAND_LASH_PUSH,
  isCardDecoId,
  placeDecoPosition,
} from "../../static/js/mhAssets.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

// Ball model, module meters. MODULE_EYE_MID + eyeball geometry constants
// are parsed live out of eyesOral.js (browser-only import), so the test
// fails loudly on drift instead of silently passing against stale values.
const EYES_ORAL = readFileSync(join(ROOT, "static", "js", "eyesOral.js"), "utf8");
function parseConstVector(name) {
  const m = EYES_ORAL.match(
    new RegExp(`export const ${name} = new THREE\\.Vector3\\(([^)]+)\\)`)
  );
  assert.ok(m, `cannot parse ${name} from eyesOral.js`);
  return new THREE.Vector3(...m[1].split(",").map(Number));
}
function parseConstNumber(name) {
  const m = EYES_ORAL.match(new RegExp(`export const ${name} = ([0-9.e+-]+)`));
  assert.ok(m, `cannot parse ${name} from eyesOral.js`);
  return Number(m[1]);
}
const EYE_MID = parseConstVector("MODULE_EYE_MID");
const BALL_R = parseConstNumber("EYEBALL_RADIUS");
// MHR neutral l_eye/r_eye |x| in meters. Rig fact (model joints), the one
// value this test pins by hand: the module basis is orthonormal, so the
// module-space half eye distance equals the MHR one.
const HALF_EYE_M = 0.0321;
const BALLS = [
  new THREE.Vector3(HALF_EYE_M, EYE_MID.y, EYE_MID.z),
  new THREE.Vector3(-HALF_EYE_M, EYE_MID.y, EYE_MID.z),
];
// Corneal bulge mirrors createEyeballGeometry: verts past 0.84R shift +45%.
const BULGE_FROM = 0.84;
const BULGE_GAIN = 0.45;

function lashObjPaths() {
  // Asset ids are posix (server-side); fs paths stay native. The split
  // matters on Windows, where join() yields backslashes that would break
  // isCardDecoId and silently test the wrong proud value.
  const out = [];
  for (const dir of [
    "eye_decoration_card/eyelashes",
    "eye_decoration_modelled/eyelashes01_cc0/eyelashes",
  ]) {
    const base = join(ROOT, "data", "makehuman", dir);
    for (const style of readdirSync(base)) {
      for (const file of readdirSync(join(base, style))) {
        if (file.endsWith(".obj")) out.push(`${dir}/${style}/${file}`);
      }
    }
  }
  return out.sort();
}

function parseObjVerts(path) {
  const verts = [];
  for (const line of readFileSync(join(ROOT, "data", "makehuman", path), "utf8").split("\n")) {
    if (!line.startsWith("v ")) continue;
    const [, x, y, z] = line.split(/\s+/).map(Number);
    verts.push(new THREE.Vector3(x, y, z));
  }
  assert.ok(verts.length > 0, `no verts in ${path}`);
  return verts;
}

// Loader centers geometry on the vertex mean (loadMhObjMesh), then
// positions it via placeDecoPosition; mirror both with the REAL imports.
function placedModule(verts, id) {
  const mean = new THREE.Vector3();
  for (const v of verts) mean.add(v);
  mean.divideScalar(verts.length);
  const proud = isCardDecoId(id) ? LASH_CARD_PROUD : STRAND_LASH_PUSH;
  const pos = placeDecoPosition(mean, LASH_ANCHOR, LASH_REF_DM, proud);
  return { mean, pos };
}

function clearanceMm(p, yawDeg = 0) {
  // Un-rotate the query into eye space so the bulge follows gaze.
  const yw = THREE.MathUtils.degToRad(-yawDeg);
  const cy = Math.cos(yw);
  const sy = Math.sin(yw);
  let best = Infinity;
  for (const c of BALLS) {
    const dx = (p.x - c.x) * 1000;
    const dy = (p.y - c.y) * 1000;
    const dz = (p.z - c.z) * 1000;
    const radial = Math.hypot(dx, dy, dz) - BALL_R * 1000;
    const lz = Math.min(dx * sy + dz * cy, BALL_R * 1000);
    const bulge = Math.max(0, lz - BULGE_FROM * BALL_R * 1000) * BULGE_GAIN;
    best = Math.min(best, radial - bulge);
  }
  return best;
}

test("lash styles clear the corneal bulge at neutral gaze", () => {
  const ids = lashObjPaths();
  assert.equal(ids.length, 9);
  for (const id of ids) {
    const verts = parseObjVerts(id);
    const { mean, pos } = placedModule(verts, id);
    let min = Infinity;
    for (const v of verts) {
      const p = v.clone().sub(mean).multiplyScalar(0.1).add(pos);
      min = Math.min(min, clearanceMm(p));
    }
    assert.ok(
      min >= 1.0,
      `${id}: nearest lash ${min.toFixed(2)}mm from cornea, want >= +1.0mm`
    );
  }
});

test("lash styles survive gaze yaw +/-20deg", () => {
  for (const id of lashObjPaths()) {
    const verts = parseObjVerts(id);
    const { mean, pos } = placedModule(verts, id);
    let min = Infinity;
    for (const yaw of [-20, 20]) {
      for (const v of verts) {
        const p = v.clone().sub(mean).multiplyScalar(0.1).add(pos);
        min = Math.min(min, clearanceMm(p, yaw));
      }
    }
    assert.ok(
      min >= 0.3,
      `${id}: nearest lash ${min.toFixed(2)}mm at gaze extremes, want >= +0.3mm`
    );
  }
});
