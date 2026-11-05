import test from "node:test";
import assert from "node:assert/strict";
import { SpatialHash } from "../../static/js/sim/SpatialHash.js";

const agent = (x, z) => ({ position: { x, z } });

test("query returns the 3x3 neighborhood", () => {
  const grid = new SpatialHash(3.5);
  const agents = [agent(0, 0), agent(1, 1), agent(100, 100)];
  grid.build(agents);
  const found = grid.query(0, 0);
  assert.equal(found.length, 2);
  assert.ok(found.includes(agents[0]) && found.includes(agents[1]));
  assert.equal(grid.cellCount(), 2);
  assert.equal(grid.size, 3);
});

test("rebuild clears previous contents", () => {
  const grid = new SpatialHash(3.5);
  grid.build([agent(0, 0), agent(100, 100)]);
  grid.build([agent(0, 0)]);
  assert.equal(grid.size, 1);
  assert.equal(grid.cellCount(), 1);
  assert.equal(grid.query(100, 100).length, 0);
});

test("negative coordinates hash distinctly", () => {
  const grid = new SpatialHash(3.5);
  const a = agent(-0.1, -0.1);
  const b = agent(0.1, 0.1);
  grid.build([a, b]);
  assert.equal(grid.cellCount(), 2);
  assert.ok(grid.query(-0.1, -0.1).includes(a));
});

test("query never misses an in-radius agent (randomized vs brute force)", () => {
  let seed = 123456789;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let round = 0; round < 50; round++) {
    const grid = new SpatialHash(3.5);
    const agents = [];
    const n = 5 + Math.floor(rand() * 40);
    for (let i = 0; i < n; i++) {
      agents.push(agent((rand() - 0.5) * 22, (rand() - 0.5) * 22));
    }
    grid.build(agents);
    for (let q = 0; q < 10; q++) {
      const qx = (rand() - 0.5) * 22;
      const qz = (rand() - 0.5) * 22;
      const found = new Set(grid.query(qx, qz));
      for (const a of agents) {
        const d = Math.hypot(a.position.x - qx, a.position.z - qz);
        if (d <= 3.5) assert.ok(found.has(a), `missed agent at distance ${d}`);
      }
    }
  }
});
