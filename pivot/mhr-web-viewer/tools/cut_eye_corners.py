"""Delete leftover corner webs from the cut eye apertures (forward one-off).

The sealed eye strips were already cut, so deformation re-detection finds
nothing. This works forward from the CURRENT cut mesh instead: per LOD and
eye it takes the aperture rim (globally-once edges near the eye joint),
seeds faces attached to the rim's x-extreme ends, and grows strictly
outward along +/-x into faces that stretch under eyesClosed. The blunt
rim ends become pointed canthi; the dev bar stops growth at rigid skin.

Usage:
    python tools/cut_eye_corners.py --analyze [--plots DIR]
    python tools/cut_eye_corners.py --apply
"""

import argparse
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

VIEWER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VIEWER_ROOT))
sys.path.insert(0, str(VIEWER_ROOT / "tools"))

from server import DEFAULT_MODEL_PATH, LOD_TOPOLOGY_PATH, MHREngine  # noqa: E402
from remove_sealed_polys import (  # noqa: E402
    deformation,
    edge_adjacency,
    grow_axis,
)

ROI_RADIUS = 4.0  # cm around the eye joint
END_TOL = 0.25  # cm: rim verts within this of the x-extreme are an "end"
YBAND = 0.6  # cm: corner corridor half-height around the eye joint
ZBAND = 1.0  # cm: corridor depth around the eye joint
SEED_BAR = 1.05  # rim-end faces: positionally certain, rigid skin is exactly 1.0
DEV_BAR = 1.1  # chains bridge 10%+ stretch; rigid skin is exactly 1.0
LATERAL_RAIL = 1.60  # cm from eye center: absolute lateral canthus bound
MEDIAL_REACH = 0.35  # cm past the medial rim end (usually morph-silent)
CAP_PER_EYE = 40  # hard cap on picks per eye/LOD (anti-runaway)
MAX_NEW_XSPAN = 3.2  # cm; aperture must not grow past this after the cut


def load_engine():
    import torch

    engine = MHREngine(DEFAULT_MODEL_PATH)
    base = (
        engine.model.character_torch.blend_shape.base_shape.detach()
        .cpu().numpy().astype(np.float64)
    )
    fe = (
        engine.model.face_expressions_model.shape_vectors.detach()
        .cpu().numpy().astype(np.float64)
    )
    with torch.inference_mode():
        _, skeleton = engine.model(
            identity_coeffs=torch.zeros(1, engine.identity_count),
            model_parameters=torch.zeros(1, len(engine.pose_names)),
            face_expr_coeffs=torch.zeros(1, 72),
            apply_correctives=False,
        )
    joints = skeleton[0].detach().cpu().numpy()[:, :3]
    return engine, base, fe, joints


def global_boundary(faces):
    """Edges used exactly once in the whole mesh (vectorized)."""
    e = np.stack(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=1
    )
    e = np.sort(e, axis=2).reshape(-1, 2)
    u, c = np.unique(e, axis=0, return_counts=True)
    return u[c == 1]


def order_loops(bnd):
    """Greedy chains of boundary edges, largest first (mesh vert ids)."""
    adj = defaultdict(set)
    for a, b in bnd:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    loops, seen = [], set()
    for v in adj:
        if v in seen:
            continue
        chain, cur, prev = [v], v, None
        seen.add(v)
        while True:
            nxt = [w for w in adj[cur] if w != prev]
            if not nxt:
                break
            cur = nxt[0]
            if cur in seen:
                break
            seen.add(cur)
            chain.append(cur)
            prev = chain[-2]
        loops.append(chain)
    loops.sort(key=len, reverse=True)
    return loops


def detect_corners(faces, neutral, center, dev, neighbours):
    """Rim-end-anchored outward growth into stretching corner faces."""
    from matplotlib.path import Path

    centroids = neutral[faces].mean(axis=1)
    bnd = global_boundary(faces)
    mid = neutral[bnd].mean(axis=1)
    near = bnd[np.linalg.norm(mid - center, axis=1) < ROI_RADIUS]
    if len(near) == 0:
        return {"picks": np.array([], dtype=int), "error": "no-rim"}
    loops = order_loops(near)
    loop = np.array(loops[0])
    rim_xy = neutral[loop][:, :2]
    rim_xspan = (float(rim_xy[:, 0].min()), float(rim_xy[:, 0].max()))
    rim_yspan = (float(rim_xy[:, 1].min()), float(rim_xy[:, 1].max()))

    roi = np.flatnonzero(np.linalg.norm(centroids - center, axis=1) < ROI_RADIUS)
    in_loop = np.zeros(len(faces), dtype=bool)
    in_loop[roi] = Path(rim_xy).contains_points(centroids[roi][:, :2])
    raw_inside = int(in_loop.sum())

    lid_band = (
        (np.abs(centroids[:, 1] - center[1]) < YBAND)
        & (np.abs(centroids[:, 2] - center[2]) < ZBAND)
        & ~in_loop
    )
    # Coarse faces pass centroid tests while spilling far outside the
    # corridor, so eligibility also requires every VERT inside margins.
    fv = neutral[faces]
    v_y = (np.abs(fv[:, :, 1] - center[1]) < YBAND + 0.15).all(axis=1)
    v_z = (np.abs(fv[:, :, 2] - center[2]) < ZBAND + 0.3).all(axis=1)
    xs = neutral[loop][:, 0]
    # Lateral = away from the nose (outward sign per eye), medial = inward.
    outward = 1.0 if center[0] >= 0 else -1.0
    ends = {
        -1.0: set(loop[xs <= xs.min() + END_TOL].tolist()),
        1.0: set(loop[xs >= xs.max() - END_TOL].tolist()),
    }
    seeds, windows, dev_show = [], {}, {}
    for sign, end_verts in ends.items():
        edge_x = xs.min() if sign < 0 else xs.max()
        if sign == outward:
            rail = center[0] + outward * LATERAL_RAIL
            lo, hi = (edge_x - 0.1, rail) if outward > 0 else (rail, edge_x + 0.1)
            v_x = (
                (fv[:, :, 0] > lo - 0.05).all(axis=1)
                & (fv[:, :, 0] < hi + 0.15).all(axis=1)
            )
            x_ok = (centroids[:, 0] > lo) & (centroids[:, 0] < hi)
        else:
            if sign < 0:
                x_ok = (centroids[:, 0] < edge_x + 0.1) & (
                    centroids[:, 0] > edge_x - MEDIAL_REACH
                )
                v_x = (
                    (fv[:, :, 0] < edge_x + 0.15).all(axis=1)
                    & (fv[:, :, 0] > edge_x - MEDIAL_REACH - 0.15).all(axis=1)
                )
            else:
                x_ok = (centroids[:, 0] > edge_x - 0.1) & (
                    centroids[:, 0] < edge_x + MEDIAL_REACH
                )
                v_x = (
                    (fv[:, :, 0] > edge_x - 0.15).all(axis=1)
                    & (fv[:, :, 0] < edge_x + MEDIAL_REACH + 0.15).all(axis=1)
                )
        eligible = lid_band & v_y & v_z & v_x
        cand = np.flatnonzero(
            eligible
            & (
                np.isin(faces[:, 0], list(end_verts))
                | np.isin(faces[:, 1], list(end_verts))
                | np.isin(faces[:, 2], list(end_verts))
            )
        )
        for c in cand:
            seeds.append((int(c), sign))
        windows[sign] = np.flatnonzero(eligible & x_ok)
        wdev = sorted(dev[windows[sign]].tolist(), reverse=True)
        dev_show[sign] = [round(v, 2) for v in wdev[:12]]

    # Seeds are rim-end web faces themselves; pre-add the hot ones so no
    # fringe survives when a chain never steps back onto its own seed.
    # Below-bar seeds start no chain: their hot neighbour would otherwise
    # be cut while detached from the rim (floating hole in the lid).
    sel = {t for (t, s) in seeds if dev[t] > SEED_BAR}
    for sign in (-1.0, 1.0):
        side_seeds = [(t, s) for (t, s) in seeds if s == sign and dev[t] > SEED_BAR]
        if side_seeds:
            sel = grow_axis(sel, side_seeds, neighbours, centroids, dev,
                            DEV_BAR, windows[sign])
    picks = np.array(sorted(sel), dtype=int)
    new_xspan = rim_xspan
    if len(picks):
        px = neutral[faces[picks].ravel()][:, 0]
        new_xspan = (
            float(min(rim_xspan[0], px.min())),
            float(max(rim_xspan[1], px.max())),
        )
    return {
        "picks": picks,
        "seeds": [t for (t, s) in seeds],
        "seed_dev": sorted([round(float(dev[t]), 2) for (t, s) in seeds],
                           reverse=True),
        "outward": outward,
        "roi": roi,
        "loop": loop,
        "loop_sizes": sorted([len(l) for l in loops], reverse=True),
        "rim_xspan": rim_xspan,
        "rim_yspan": rim_yspan,
        "inside_raw": raw_inside,
        "new_xspan": new_xspan,
        "dev_show": dev_show,
    }


def plot_detection(outdir, lod, side, faces, neutral, center, det):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    matplotlib.use("Agg")
    tris = neutral[faces[det["roi"]]][:, :, :2]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_collection(PolyCollection(tris, facecolor="0.75", edgecolor="0.4",
                                     linewidths=0.3))
    if det["seeds"]:
        stris = neutral[faces[np.array(det["seeds"])]][:, :, :2]
        ax.add_collection(PolyCollection(stris, facecolor="cyan",
                                         edgecolor="blue", linewidths=0.8))
    if len(det["picks"]):
        ptris = neutral[faces[det["picks"]]][:, :, :2]
        ax.add_collection(PolyCollection(ptris, facecolor="orange",
                                         edgecolor="red", linewidths=0.8))
    loop_xy = neutral[np.append(det["loop"], det["loop"][0])][:, :2]
    ax.plot(loop_xy[:, 0], loop_xy[:, 1], "r-", lw=1.5)
    ax.plot([center[0]], [center[1]], "b+", ms=10, mew=2)
    ax.set_aspect("equal")
    ax.set_xlim(center[0] - 3, center[0] + 3)
    ax.set_ylim(center[0] - 3, center[0] + 3)
    ax.set_ylim(center[1] - 3, center[1] + 3)
    nx = det["new_xspan"]
    ax.set_title(
        f"LOD{lod} eye{side}: picks={len(det['picks'])} "
        f"rim_x=({det['rim_xspan'][0]:.2f},{det['rim_xspan'][1]:.2f}) "
        f"new_x=({nx[0]:.2f},{nx[1]:.2f})"
    )
    out = Path(outdir) / f"lod{lod}_eye{side}.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--plots", default=None)
    args = ap.parse_args()
    if not (args.analyze or args.apply):
        ap.error("pass --analyze or --apply")

    t0 = time.time()
    engine, base, fe, joints = load_engine()
    names = engine.joint_names
    expr = {n: i for i, n in enumerate(engine.expression_names)}
    print(f"engine ready in {time.time() - t0:.1f}s")

    lods = sorted(engine.lod_faces.keys())
    results = {}
    for lod in lods:
        faces = engine.lod_faces[lod]
        neutral = engine._render_lod(base, lod) if lod != 1 else base
        neighbours = edge_adjacency(faces)
        for side, jn, en in (("L", "l_eye", "eyesClosed_L"),
                             ("R", "r_eye", "eyesClosed_R")):
            center = joints[names.index(jn)]
            deformed = (
                engine._render_lod(base + fe[expr[en]], lod)
                if lod != 1 else base + fe[expr[en]]
            )
            dev = deformation(neutral, deformed, faces)
            det = detect_corners(faces, neutral, center, dev, neighbours)
            results[(lod, side)] = det
            if "error" in det:
                print(f"LOD{lod} eye{side}: {det['error']}")
                continue
            ok = (len(det["picks"]) <= CAP_PER_EYE
                  and det["new_xspan"][1] - det["new_xspan"][0] <= MAX_NEW_XSPAN)
            out = det["outward"]
            print(
                f"LOD{lod} eye{side}: rim_xw="
                f"{det['rim_xspan'][1] - det['rim_xspan'][0]:.2f} "
                f"seeds={len(det['seeds'])} seed_dev={det['seed_dev']} "
                f"picks={len(det['picks'])} "
                f"new_xw={det['new_xspan'][1] - det['new_xspan'][0]:.2f} "
                f"dev_out={det['dev_show'][out]} dev_in={det['dev_show'][-out]} "
                f"{'OK' if ok else 'SKIP'}"
            )
            if args.plots:
                out = plot_detection(args.plots, lod, side, faces, neutral,
                                     center, det)
                print(f"    plot: {out}")

    if not args.apply:
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = LOD_TOPOLOGY_PATH.with_suffix(f".npz.bak-{stamp}")
    shutil.copy2(LOD_TOPOLOGY_PATH, backup)
    print(f"backup: {backup} (existing .bak preserved)")
    topo = dict(np.load(LOD_TOPOLOGY_PATH))
    total = 0
    for lod in lods:
        drop = set()
        for side in ("L", "R"):
            det = results[(lod, side)]
            if "error" in det:
                continue
            if (len(det["picks"]) > CAP_PER_EYE
                    or det["new_xspan"][1] - det["new_xspan"][0] > MAX_NEW_XSPAN):
                print(f"LOD{lod} eye{side}: SKIP (cap exceeded)")
                continue
            drop.update(det["picks"].tolist())
        key = f"lod{lod}_faces"
        arr = topo[key]
        keep = np.ones(len(arr), dtype=bool)
        keep[sorted(drop)] = False
        topo[key] = arr[keep]
        total += len(drop)
        print(f"LOD{lod}: removed {len(drop)} faces ({len(arr)} -> {keep.sum()})")
    np.savez(LOD_TOPOLOGY_PATH, **topo)
    print(f"wrote {LOD_TOPOLOGY_PATH}, total removed: {total}")


if __name__ == "__main__":
    main()
