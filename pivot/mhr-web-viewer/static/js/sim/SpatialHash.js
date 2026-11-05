// Uniform spatial hash grid for per-tick crowd queries.
//
// Dependency-free (reads only `.position.x/.z`), so Node tests it directly.
// The simulation rebuilds it once per tick; steering queries it today
// instead of scanning the whole population (auto-LOD banding and frustum
// visibility are future consumers of the same structure).
//
// Correctness contract: cellSize must be >= the largest query radius, and
// queries gather the full 3x3 neighborhood. Then every agent within `radius`
// of the query point is in the returned set (the block extends >= radius in
// every direction from any point of the center cell). Callers still apply
// their exact distance check — the grid returns a superset.
export class SpatialHash {
  constructor(cellSize = 3.5) {
    this.cellSize = cellSize;
    this.cells = new Map();
    this.size = 0;
  }

  static _key(cx, cz) {
    return ((cx & 0xffff) << 16) | (cz & 0xffff);
  }

  _cellOf(x) {
    return Math.floor(x / this.cellSize);
  }

  clear() {
    this.cells.clear();
    this.size = 0;
  }

  insert(agent) {
    const cx = this._cellOf(agent.position.x);
    const cz = this._cellOf(agent.position.z);
    const key = SpatialHash._key(cx, cz);
    let cell = this.cells.get(key);
    if (!cell) {
      cell = [];
      this.cells.set(key, cell);
    }
    cell.push(agent);
    this.size++;
  }

  build(agents) {
    this.clear();
    for (let i = 0; i < agents.length; i++) this.insert(agents[i]);
  }

  /**
   * Gathers every agent in the 3x3 cells around (x, z) into `out`
   * (allocated when omitted). Superset of the radius neighborhood when
   * radius <= cellSize; callers filter by exact distance.
   */
  query(x, z, out) {
    const result = out || [];
    const ccx = this._cellOf(x);
    const ccz = this._cellOf(z);
    for (let dx = -1; dx <= 1; dx++) {
      for (let dz = -1; dz <= 1; dz++) {
        const cell = this.cells.get(SpatialHash._key(ccx + dx, ccz + dz));
        if (cell) {
          for (let i = 0; i < cell.length; i++) result.push(cell[i]);
        }
      }
    }
    return result;
  }

  cellCount() {
    return this.cells.size;
  }
}
