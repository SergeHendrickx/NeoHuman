import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

// gizmo.js pulls browser-only modules, so load it with a stub module for
// every relative import. `state` reads from a test-controlled global, which
// also lets tests flip groups between cases. three + addons stay real.
globalThis.__gizmoTestState = {
  currentGroup: "pose",
  metadata: {},
  rigMapping: {
    tpCols: 2,
    jointCount: 2,
    TpMatrix: new Float32Array(2 * 7 * 2),
  },
};
// Param 0 drives joint 1 hardest (row joint*7+channel, col 0); param 1 is dead.
globalThis.__gizmoTestState.rigMapping.TpMatrix[(1 * 7 + 3) * 2 + 0] = 5.0;
globalThis.__gizmoTestState.rigMapping.TpMatrix[(0 * 7 + 0) * 2 + 0] = 1.0;

const src = readFileSync(new URL("../../static/js/gizmo.js", import.meta.url), "utf8");
const names = [...src.matchAll(/^import\s+{([^}]+)}/gm)]
  .flatMap((m) => m[1].split(",").map((n) => n.trim()).filter(Boolean));
const stubSrc = [...new Set(names)]
  .map((n) => (n === "state"
    ? "export const state = globalThis.__gizmoTestState;"
    : `export const ${n} = undefined;`))
  .join("\n");
const stubUrl = `data:text/javascript,${encodeURIComponent(stubSrc)}`;
const threeUrl = import.meta.resolve("three");
const tcUrl = import.meta.resolve("three/addons/controls/TransformControls.js");
const stubbed = src
  .replace(/from "\.\.?\/[^"]+"/g, `from "${stubUrl}"`)
  .replaceAll('from "three"', `from "${threeUrl}"`)
  .replaceAll(
    '"three/addons/controls/TransformControls.js"',
    `"${tcUrl}"`,
  );
const gizmo = await import(`data:text/javascript,${encodeURIComponent(stubbed)}`);

test("strongestJointForParam finds the hardest-driven joint", () => {
  assert.equal(gizmo.strongestJointForParam(0), 1);
});

test("strongestJointForParam returns -1 for a dead column", () => {
  assert.equal(gizmo.strongestJointForParam(1), -1);
});

test("strongestJointForParam returns -1 out of range", () => {
  assert.equal(gizmo.strongestJointForParam(5), -1);
  assert.equal(gizmo.strongestJointForParam(-1), -1);
});

test("strongestJointForParam returns -1 outside the pose group", () => {
  globalThis.__gizmoTestState.currentGroup = "identity";
  assert.equal(gizmo.strongestJointForParam(0), -1);
  globalThis.__gizmoTestState.currentGroup = "pose";
});

test("axisThroughPoints connects two joints exactly", async () => {
  const { Vector3 } = await import("three");
  const dir = gizmo.axisThroughPoints([new Vector3(-1, 0, 0), new Vector3(1, 0, 0)]);
  assert.ok(dir !== null);
  assert.ok(Math.abs(dir.x - 1) < 1e-9 && Math.abs(dir.y) < 1e-9 && Math.abs(dir.z) < 1e-9);
  const rev = gizmo.axisThroughPoints([new Vector3(1, 0, 0), new Vector3(-1, 0, 0)]);
  assert.ok(Math.abs(rev.x + 1) < 1e-9, "sign follows first-to-last");
});

test("axisThroughPoints finds the principal axis of several joints", async () => {
  const { Vector3 } = await import("three");
  const pts = [new Vector3(0, 0, 0), new Vector3(2, 0, 0), new Vector3(1, 1, 0)];
  const dir = gizmo.axisThroughPoints(pts);
  assert.ok(dir !== null);
  assert.ok(Math.abs(dir.x - 1) < 1e-6, `x=${dir.x}`);
  assert.ok(Math.abs(dir.y) < 1e-6 && Math.abs(dir.z) < 1e-6);
});

test("axisThroughPoints rejects degenerate sets", async () => {
  const { Vector3 } = await import("three");
  assert.equal(gizmo.axisThroughPoints([]), null);
  assert.equal(gizmo.axisThroughPoints([new Vector3(0, 0, 0)]), null);
  assert.equal(
    gizmo.axisThroughPoints([new Vector3(1, 1, 1), new Vector3(1, 1, 1)]),
    null,
  );
});

test("paramJointMode maps the dominant channel to a gizmo mode", () => {
  const tp = globalThis.__gizmoTestState.rigMapping.TpMatrix;
  tp[(0 * 7 + 6) * 2 + 0] = 9.0;
  tp[(0 * 7 + 2) * 2 + 1] = 4.0;
  assert.equal(gizmo.paramJointMode(0, 1), "rotate");
  assert.equal(gizmo.paramJointMode(0, 0), "scale");
  assert.equal(gizmo.paramJointMode(1, 0), "move");
  assert.equal(gizmo.paramJointMode(1, 1), null);
  assert.equal(gizmo.paramJointMode(9, 0), null);
});
