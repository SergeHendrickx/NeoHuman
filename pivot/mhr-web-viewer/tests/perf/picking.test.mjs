import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as THREE from "three";

// AgentSimulation pulls browser-only modules, so load it with a stub module
// for every relative import (three stays real) and skip the singleton, which
// touches the DOM in its constructor.
const src = readFileSync(
  new URL("../../static/js/sim/AgentSimulation.js", import.meta.url), "utf8");
const names = [...src.matchAll(/^import\s+{([^}]+)}/gm)]
  .flatMap((m) => m[1].split(",").map((n) => n.trim()).filter(Boolean));
const stubUrl = `data:text/javascript,${encodeURIComponent(
  [...new Set(names)].map((n) => `export const ${n} = undefined;`).join("\n"),
)}`;
const threeUrl = import.meta.resolve("three");
const stubbed = src
  .replace(/from "\.\.?\/[^"]+"/g, `from "${stubUrl}"`)
  .replaceAll('from "three"', `from "${threeUrl}"`)
  .replace(
    "export const simulation = new AgentSimulation();",
    "export const simulation = null;",
  );
const sim = await import(`data:text/javascript,${encodeURIComponent(stubbed)}`);

const pos = new THREE.Vector3(0, 0, 0);

test("rayCapsuleHit hits a ray through the capsule", () => {
  const ray = new THREE.Ray(
    new THREE.Vector3(0, 1, -5), new THREE.Vector3(0, 0, 1));
  const t = sim.rayCapsuleHit(ray, pos, 1.9, 0.5);
  assert.ok(t !== null, "expected a hit");
  assert.ok(Math.abs(t - 5) < 1e-6, `along-ray distance ${t}`);
});

test("rayCapsuleHit misses a ray outside the radius", () => {
  const ray = new THREE.Ray(
    new THREE.Vector3(2, 1, -5), new THREE.Vector3(0, 0, 1));
  assert.equal(sim.rayCapsuleHit(ray, pos, 1.9, 0.5), null);
});

test("rayCapsuleHit misses a ray passing above the capsule", () => {
  const ray = new THREE.Ray(
    new THREE.Vector3(0, 3, -5), new THREE.Vector3(0, 0, 1));
  assert.equal(sim.rayCapsuleHit(ray, pos, 1.9, 0.5), null);
});
