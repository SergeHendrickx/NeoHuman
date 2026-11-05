import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

// bench.js imports browser-only modules (crowd/scene/state), so load it with
// a stub module satisfying the named imports. The pure summary math under
// test never touches them at import time.
const names = [
  "clearAgents", "isAgentSimulationActive", "setAgentLodPin",
  "setAutoLodEnabled", "simulation", "spawnAgents", "toggleAgentSimulation",
  "controls", "fitCamera", "isFollowing", "orthographicCamera",
  "perspectiveCamera", "readRenderExecMs", "renderer", "setFollow",
  "syncCameraInputs", "state",
];
const stubUrl = `data:text/javascript,${encodeURIComponent(
  names.map((n) => `export const ${n} = undefined;`).join("\n"),
)}`;
// bench.js only needs THREE.Vector3 (crowd framing); stub the namespace.
const threeUrl = `data:text/javascript,${encodeURIComponent(
  "export class Vector3 { constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; } }",
)}`;
const src = readFileSync(new URL("../../static/js/bench.js", import.meta.url), "utf8");
const stubbed = src
  .replaceAll('"./crowd.js"', `"${stubUrl}"`)
  .replaceAll('"./scene.js"', `"${stubUrl}"`)
  .replaceAll('"./state.js"', `"${stubUrl}"`)
  .replaceAll('"three"', `"${threeUrl}"`);
const bench = await import(`data:text/javascript,${encodeURIComponent(stubbed)}`);

function sample(frameMs, extra = {}) {
  return {
    frameMs, agentMs: 1, postMs: 2, uiMs: 0.5,
    calls: 100, tris: 50000, gpuExecMs: null, renderExecMs: null,
    ...extra,
  };
}

test("summarizeSamples computes fps from mean frame time", () => {
  const s = bench.summarizeSamples([sample(10), sample(20), sample(30)]);
  assert.equal(s.frames, 3);
  assert.ok(Math.abs(s.fpsAvg - 50) < 1e-9, `fps ${s.fpsAvg}`);
  assert.ok(Math.abs(s.frameAvg - 20) < 1e-9);
  assert.equal(s.frameP50, 20);
  assert.equal(s.frameP95, 30);
  assert.ok(Math.abs(s.tickAvg - 3.5) < 1e-9);
});

test("summarizeSamples averages only non-null gpu exec times", () => {
  const s = bench.summarizeSamples([
    sample(16, { gpuExecMs: 1.0 }),
    sample(16, { gpuExecMs: null }),
    sample(16, { gpuExecMs: 3.0 }),
  ]);
  assert.equal(s.gpuExecSamples, 2);
  assert.ok(Math.abs(s.gpuExecAvg - 2.0) < 1e-9);
  assert.equal(s.renderExecSamples, 0);
});

test("summarizeSamples handles empty input", () => {
  const s = bench.summarizeSamples([]);
  assert.equal(s.frames, 0);
  assert.equal(s.fpsAvg, 0);
  assert.equal(s.frameP95, 0);
});
