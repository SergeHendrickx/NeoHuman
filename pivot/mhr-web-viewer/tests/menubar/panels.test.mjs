import test from "node:test";
import assert from "node:assert/strict";
import { homeRefFor } from "../../static/js/menubar.js";

const open = (...ids) => (id) => ids.includes(id);

test("home ref is an open member of the same group", () => {
  assert.equal(homeRefFor("actors", open("world", "crowd")), "world");
  assert.equal(homeRefFor("ai", open("pose", "eyes")), "pose");
  assert.equal(homeRefFor("crowdstats", open("telemetry")), "telemetry");
});

test("no ref when the group is closed, viewport, or unknown", () => {
  assert.equal(homeRefFor("actors", open()), null);
  assert.equal(homeRefFor("ai", open("world", "crowd")), null);
  assert.equal(homeRefFor("viewport", open("world")), null);
  assert.equal(homeRefFor("nope", open("world")), null);
});

test("never references the pane itself", () => {
  assert.equal(homeRefFor("world", open("world", "camera")), "camera");
  assert.equal(homeRefFor("world", open("world")), null);
});
