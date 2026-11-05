// TSL compute kernels for the GPU crowd path (Stage C: batched).
//
// buildBatchedKernels(TSL, io) builds exactly TWO compute nodes per tick,
// no matter the population:
//   fkAll   (S*J threads): thetaJ -> per-slot bone matrices (row-major 4x4)
//   skinAll (S*V threads): LOD gather + blendshapes + LBS -> position + normal
//
// Thread t covers slot (t / N) + item (t % N), guarded by a uTotal uniform
// so edge workgroups stay idle. Slots come from the thread id: the per-agent
// uAgent/uShape uniforms of Stage A/B are gone (the shaped flag is now a
// per-slot word in a small shared buffer). The provider passes explicit
// [x, y, 1] dispatch sizes per node, so dispatch never depends on the
// backend's 1D-count overflow spillover.
//
// The TSL namespace is injected so Node tests can pass the real build.
// Math conventions mirror rig.js / Actor exactly: ZYX Euler, pre-rotation
// premultiply, exponential scale, cm->m on translation, row-major bone
// matrices composed as bone = chainWorld x boneInverse (mesh space; the
// agent group's model matrix applies placement, identical to three.js
// AttachedBindMode with identity bind).
import { GPU_MAX_CHAIN } from "./layouts.js";

export function buildBatchedKernels(TSL, io) {
  const { Fn, If, Loop, float, int, sin, cos, pow, sqrt, vec3, storage, globalId } = TSL;
  const { S, J, V, V1, B } = io.dims;
  const rig = io.dims.rig;
  const body = io.dims.body;
  const lod = io.dims.lod;

  const I = (n) => int(n);
  const F = (n) => float(n);

  const sRig = storage(io.rig, "float", io.rigCount);
  const sDeltas = storage(io.body, "float", io.bodyCount);
  const sTheta = storage(io.theta, "float", io.thetaCount);
  const sWeights = storage(io.weights, "float", io.weightCount);
  const sShape = storage(io.shape, "float", S);
  const sBones = storage(io.bones, "float", S * J * 16);
  const sLod = storage(io.lod, "float", io.lodCount);
  const sOutPos = storage(io.outPos, "vec3", S * V);
  const sOutNormal = storage(io.outNormal, "vec3", S * V);
  const uFkTotal = io.uFkTotal;
  const uSkinRowW = io.uSkinRowW;
  const uSkinTotal = io.uSkinTotal;

  // Emits TSL composing joint n's local TRS into scratch vars L.
  // L = { r: [9 float vars] (row-major 3x3), t: [3 float vars] }.
  function emitLocal(n, aBase, L) {
    const base = n.mul(I(7));
    const tOff = aBase.add(base);
    const tx = sTheta.element(tOff.add(I(0))).toVar();
    const ty = sTheta.element(tOff.add(I(1))).toVar();
    const tz = sTheta.element(tOff.add(I(2))).toVar();
    const rx = sTheta.element(tOff.add(I(3))).toVar();
    const ry = sTheta.element(tOff.add(I(4))).toVar();
    const rz = sTheta.element(tOff.add(I(5))).toVar();
    const sc = sTheta.element(tOff.add(I(6))).toVar();

    const oBase = n.mul(I(3)).add(I(rig.jointOffsets));
    L.t[0].assign(sRig.element(oBase.add(I(0))).add(tx).div(F(100)));
    L.t[1].assign(sRig.element(oBase.add(I(1))).add(ty).div(F(100)));
    L.t[2].assign(sRig.element(oBase.add(I(2))).add(tz).div(F(100)));

    // ZYX Euler -> quaternion (matches THREE.Quaternion.setFromEuler).
    const c1 = cos(rx.div(F(2))).toVar();
    const c2 = cos(ry.div(F(2))).toVar();
    const c3 = cos(rz.div(F(2))).toVar();
    const s1 = sin(rx.div(F(2))).toVar();
    const s2 = sin(ry.div(F(2))).toVar();
    const s3 = sin(rz.div(F(2))).toVar();
    const ex = s1.mul(c2).mul(c3).sub(c1.mul(s2).mul(s3)).toVar();
    const ey = c1.mul(s2).mul(c3).add(s1.mul(c2).mul(s3)).toVar();
    const ez = c1.mul(c2).mul(s3).sub(s1.mul(s2).mul(c3)).toVar();
    const ew = c1.mul(c2).mul(c3).add(s1.mul(s2).mul(s3)).toVar();

    // Premultiply the pre-rotation: q = pre * e.
    const pBase = n.mul(I(4)).add(I(rig.jointPreRots));
    const px = sRig.element(pBase.add(I(0))).toVar();
    const py = sRig.element(pBase.add(I(1))).toVar();
    const pz = sRig.element(pBase.add(I(2))).toVar();
    const pw = sRig.element(pBase.add(I(3))).toVar();
    const qx = px.mul(ew).add(pw.mul(ex)).add(py.mul(ez)).sub(pz.mul(ey)).toVar();
    const qy = py.mul(ew).add(pw.mul(ey)).add(pz.mul(ex)).sub(px.mul(ez)).toVar();
    const qz = pz.mul(ew).add(pw.mul(ez)).add(px.mul(ey)).sub(py.mul(ex)).toVar();
    const qw = pw.mul(ew).sub(px.mul(ex)).sub(py.mul(ey)).sub(pz.mul(ez)).toVar();

    // Quaternion -> scaled rotation matrix (row-major).
    const s = pow(F(2), sc).toVar();
    const x2 = qx.add(qx).toVar();
    const y2 = qy.add(qy).toVar();
    const z2 = qz.add(qz).toVar();
    const xx = qx.mul(x2).toVar();
    const yy = qy.mul(y2).toVar();
    const zz = qz.mul(z2).toVar();
    const xy = qx.mul(y2).toVar();
    const xz = qx.mul(z2).toVar();
    const yz = qy.mul(z2).toVar();
    const wx = qw.mul(x2).toVar();
    const wy = qw.mul(y2).toVar();
    const wz = qw.mul(z2).toVar();
    L.r[0].assign(F(1).sub(yy).sub(zz).mul(s));
    L.r[1].assign(xy.sub(wz).mul(s));
    L.r[2].assign(xz.add(wy).mul(s));
    L.r[3].assign(xy.add(wz).mul(s));
    L.r[4].assign(F(1).sub(xx).sub(zz).mul(s));
    L.r[5].assign(yz.sub(wx).mul(s));
    L.r[6].assign(xz.sub(wy).mul(s));
    L.r[7].assign(yz.add(wx).mul(s));
    L.r[8].assign(F(1).sub(xx).sub(yy).mul(s));
  }

  const fkAll = Fn(() => {
    const t = int(globalId.x).toVar();
    const fkTotal = int(uFkTotal).toVar();
    If(t.lessThan(fkTotal), () => {
      const slot = t.div(I(J)).toVar();
      const j = t.sub(slot.mul(I(J))).toVar();
      const aBase = slot.mul(I(J * 7));

      // Accumulators: world rotation (row-major 3x3) + translation.
      const W = {
        r: [
          F(1).toVar(), F(0).toVar(), F(0).toVar(),
          F(0).toVar(), F(1).toVar(), F(0).toVar(),
          F(0).toVar(), F(0).toVar(), F(1).toVar(),
        ],
        t: [F(0).toVar(), F(0).toVar(), F(0).toVar()],
      };
      const L = {
        r: Array.from({ length: 9 }, () => F(0).toVar()),
        t: [F(0).toVar(), F(0).toVar(), F(0).toVar()],
      };
      // World = L(j) * L(parent) * ... walking up; joint order is
      // topological but each thread recomputes its own chain (no cross-thread
      // reads within a dispatch).
      const accumulate = (n) => {
        emitLocal(n, aBase, L);
        const tR = Array.from({ length: 9 }, () => F(0).toVar());
        for (let r = 0; r < 3; r++) {
          for (let c = 0; c < 3; c++) {
            tR[r * 3 + c].assign(
              L.r[r * 3].mul(W.r[c]).add(L.r[r * 3 + 1].mul(W.r[3 + c])).add(L.r[r * 3 + 2].mul(W.r[6 + c])),
            );
          }
        }
        const tT = [F(0).toVar(), F(0).toVar(), F(0).toVar()];
        for (let r = 0; r < 3; r++) {
          tT[r].assign(
            L.r[r * 3].mul(W.t[0]).add(L.r[r * 3 + 1].mul(W.t[1])).add(L.r[r * 3 + 2].mul(W.t[2])).add(L.t[r]),
          );
        }
        for (let k = 0; k < 9; k++) W.r[k].assign(tR[k]);
        for (let k = 0; k < 3; k++) W.t[k].assign(tT[k]);
      };
      // First iteration uses j itself (avoids a parents[j] pre-read).
      accumulate(j);
      const n = int(sRig.element(I(rig.jointParents).add(j))).toVar();
      Loop(GPU_MAX_CHAIN, () => {
        If(n.greaterThanEqual(I(0)), () => {
          accumulate(n);
          n.assign(int(sRig.element(I(rig.jointParents).add(n))));
        });
      });

      // bone = world4x4 x inverse (inverse stored column-major, THREE layout).
      const invBase = I(rig.inverses).add(j.mul(I(16)));
      const mBase = slot.mul(I(J * 16)).add(j.mul(I(16)));
      for (let row = 0; row < 4; row++) {
        for (let col = 0; col < 4; col++) {
          let acc = null;
          for (let k = 0; k < 4; k++) {
            // world element (row, k): [R | t] for rows 0..2, [0 0 0 1] row 3.
            const wVal = row < 3 ? (k < 3 ? W.r[row * 3 + k] : W.t[row]) : k === 3 ? F(1) : F(0);
            // inverse element (k, col) in column-major storage.
            const iVal = sRig.element(invBase.add(I(col * 4 + k)));
            const term = wVal.mul(iVal);
            acc = acc === null ? term : acc.add(term);
          }
          sBones.element(mBase.add(I(row * 4 + col))).assign(acc);
        }
      }
    });
  })().computeKernel();

  const skinAll = Fn(() => {
    // 2D dispatch: row width (threads) arrives via uniform, set per tick.
    const gx = int(globalId.x).toVar();
    const gy = int(globalId.y).toVar();
    const rowW = int(uSkinRowW).toVar();
    const skinTotal = int(uSkinTotal).toVar();
    const idx = gy.mul(rowW).add(gx).toVar();
    If(idx.lessThan(skinTotal), () => {
      const slot = idx.div(I(V)).toVar();
      const v = idx.sub(slot.mul(I(V))).toVar();
      const v3 = v.mul(I(3));
      const au = slot;
      const i0 = int(sLod.element(I(lod.indices).add(v3))).toVar();
      const i1 = int(sLod.element(I(lod.indices).add(v3).add(I(1)))).toVar();
      const i2 = int(sLod.element(I(lod.indices).add(v3).add(I(2)))).toVar();
      const b0 = sLod.element(I(lod.barycentric).add(v3)).toVar();
      const b1 = sLod.element(I(lod.barycentric).add(v3).add(I(1))).toVar();
      const b2 = sLod.element(I(lod.barycentric).add(v3).add(I(2))).toVar();

      const nBase = I(body.neutral);
      const g = (buf, off, c) =>
        buf.element(off.add(i0.mul(I(3)).add(I(c)))).mul(b0)
          .add(buf.element(off.add(i1.mul(I(3)).add(I(c)))).mul(b1))
          .add(buf.element(off.add(i2.mul(I(3)).add(I(c)))).mul(b2));
      const bx = g(sDeltas, nBase, 0).toVar();
      const by = g(sDeltas, nBase, 1).toVar();
      const bz = g(sDeltas, nBase, 2).toVar();
      const rN = I(rig.neutralNrm);
      const nx = g(sRig, rN, 0).toVar();
      const ny = g(sRig, rN, 1).toVar();
      const nz = g(sRig, rN, 2).toVar();

      If(sShape.element(au).greaterThan(F(0)), () => {
        const wBase = au.mul(I(B));
        Loop(B, ({ i: b }) => {
          const w = sWeights.element(wBase.add(b)).toVar();
          If(w.notEqual(F(0)), () => {
            const dBase = I(body.deltas);
            for (let c = 0; c < 3; c++) {
              const d =
                sDeltas.element(dBase.add(i0.mul(I(B)).add(b).mul(I(3)).add(I(c)))).mul(b0)
                  .add(sDeltas.element(dBase.add(i1.mul(I(B)).add(b).mul(I(3)).add(I(c)))).mul(b1))
                  .add(sDeltas.element(dBase.add(i2.mul(I(B)).add(b).mul(I(3)).add(I(c)))).mul(b2));
              const tgt = c === 0 ? bx : c === 1 ? by : bz;
              tgt.addAssign(d.mul(w));
            }
          });
        });
      });

      // Linear blend skinning against this slot's bone matrices.
      const px = F(0).toVar();
      const py = F(0).toVar();
      const pz = F(0).toVar();
      const ox = F(0).toVar();
      const oy = F(0).toVar();
      const oz = F(0).toVar();
      const sBase = v.mul(I(4));
      const bonesBase = au.mul(I(J * 16));
      Loop(4, ({ i: k }) => {
        const jj = int(sRig.element(I(rig.skinIdx).add(sBase.add(k)))).toVar();
        const w = sRig.element(I(rig.skinW).add(sBase.add(k))).toVar();
        const mBase = bonesBase.add(jj.mul(I(16)));
        const m = (r, c) => sBones.element(mBase.add(I(r * 4 + c)));
        px.addAssign(m(0, 0).mul(bx).add(m(0, 1).mul(by)).add(m(0, 2).mul(bz)).add(m(0, 3)).mul(w));
        py.addAssign(m(1, 0).mul(bx).add(m(1, 1).mul(by)).add(m(1, 2).mul(bz)).add(m(1, 3)).mul(w));
        pz.addAssign(m(2, 0).mul(bx).add(m(2, 1).mul(by)).add(m(2, 2).mul(bz)).add(m(2, 3)).mul(w));
        ox.addAssign(m(0, 0).mul(nx).add(m(0, 1).mul(ny)).add(m(0, 2).mul(nz)).mul(w));
        oy.addAssign(m(1, 0).mul(nx).add(m(1, 1).mul(ny)).add(m(1, 2).mul(nz)).mul(w));
        oz.addAssign(m(2, 0).mul(nx).add(m(2, 1).mul(ny)).add(m(2, 2).mul(nz)).mul(w));
      });

      const il = F(1).div(sqrt(ox.mul(ox).add(oy.mul(oy)).add(oz.mul(oz)).add(F(1e-20))));
      const ov = au.mul(I(V)).add(v);
      sOutPos.element(ov).assign(vec3(px, py, pz));
      sOutNormal.element(ov).assign(vec3(ox.mul(il), oy.mul(il), oz.mul(il)));
    });
  })().computeKernel();

  return { fkAll, skinAll };
}
