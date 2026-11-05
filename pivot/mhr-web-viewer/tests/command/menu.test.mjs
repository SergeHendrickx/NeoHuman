import test from "node:test";
import assert from "node:assert/strict";
import { buildEntries, filterEntries, formFieldsFor } from "../../static/js/commandMenu.js";

const ACTORS = [
  { id: 0, name: "Actor #0" },
  { id: 3, name: "Actor #3" },
];

test("entries cover every catalog action exactly once", async () => {
  const { TOOL_SCHEMAS } = await import("../../static/js/actions.js");
  const entries = buildEntries({ actors: ACTORS });
  for (const tool of TOOL_SCHEMAS) {
    const name = tool.function.name;
    assert.ok(entries.some((e) => e.name === name), `missing palette entry for ${name}`);
  }
});

test("actor entries are dynamic; anchor is never removable", () => {
  const entries = buildEntries({ actors: ACTORS });
  assert.ok(entries.some((e) => e.key === "select-0" && e.args.actorId === 0));
  assert.ok(entries.some((e) => e.key === "select-3" && e.args.actorId === 3));
  assert.ok(entries.some((e) => e.key === "remove-3" && e.destructive === true));
  assert.ok(!entries.some((e) => e.key === "remove-0"));
  assert.ok(entries.some((e) => e.key === "clear-agents" && e.destructive === true));
});

test("UI entries reflect live state", () => {
  const running = buildEntries({ actors: [], simRunning: true, headerHidden: false });
  assert.ok(running.some((e) => e.key === "ui-sim" && e.title === "Pause simulation"));
  assert.ok(running.some((e) => e.key === "ui-header" && e.title === "Hide header"));
  const paused = buildEntries({ actors: [], simRunning: false, headerHidden: true });
  assert.ok(paused.some((e) => e.key === "ui-sim" && e.title === "Resume simulation"));
  assert.ok(paused.some((e) => e.key === "ui-header" && e.title === "Show header"));
});

test("fuzzy filter ranks title matches first", () => {
  const entries = buildEntries({ actors: ACTORS });
  const lod = filterEntries(entries, "lod");
  assert.ok(lod.length > 0);
  assert.ok(lod[0].title.toLowerCase().includes("lod"), `top hit: ${lod[0].title}`);
  const spawn = filterEntries(entries, "spawn 6");
  assert.equal(spawn[0].key, "spawn-6");
  assert.deepEqual(filterEntries(entries, "zzz-no-such-command"), []);
  assert.equal(filterEntries(entries, "").length, 12);
});

test("form fields follow the action schema", () => {
  const param = formFieldsFor("setParam");
  assert.deepEqual(param.map((f) => f.key), ["group", "index", "value"]);
  assert.equal(param[0].type, "enum");
  assert.deepEqual(param[0].enum, ["pose", "identity", "expression"]);
  assert.equal(param[1].type, "integer");
  const bench = formFieldsFor("benchmark");
  assert.deepEqual(bench.map((f) => [f.key, f.type, f.def]), [
    ["agents", "integer", 60],
    ["seconds", "number", 5],
  ]);
  assert.deepEqual(formFieldsFor("clearAgents"), []);
  assert.deepEqual(formFieldsFor("nope"), []);
});
