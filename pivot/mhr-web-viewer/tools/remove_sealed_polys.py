"""Open sealed eye/mouth strips by removing one row of faces per feature.

Eyes: the seal is the fan of long thin triangles spanning the open aperture
(tall in y, stretching under eyesClosed). Detected geometrically at LOD1,
transferred to other LODs by nearest-seal-centroid (each LOD tessellates the
aperture differently, so thresholds do not transfer but anatomy does).
Both paths then grow along the lid axis into the canthi, where corner
faces are too small to seed geometrically but still stretch; without
this the inner corners keep a web of leftover faces.
Mouth: the lip seal stretches far more than its surroundings under jawDrop.
Dense LODs tessellate the seal as several hot layers plus warm corners, so
the verified LOD1 row anchors the anatomy (lip line, x extent) and every
LOD takes all hot bands in that window, warm sideways neighbours, and an
axis-grown corner continuation. Every selection is verified per LOD
(physical width, border margin) before anything is written.

Writes `lod1_faces` (override for the immutable model faces) plus cut faces
for LODs 0, 2-6 into data/lod_topology.npz. Always backs up first.

Usage:
    python tools/remove_sealed_polys.py --analyze [--plots DIR]
    python tools/remove_sealed_polys.py --apply
"""

import argparse
import shutil
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

VIEWER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VIEWER_ROOT))

from server import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    LOD_TOPOLOGY_PATH,
    MHREngine,
    SUPPORTED_LODS,
)

FEATURES = ("eyeL", "eyeR", "mouth")


def neutral_joints(engine):
    import torch

    with torch.inference_mode():
        _, skeleton = engine.model(
            identity_coeffs=torch.zeros(1, engine.identity_count),
            model_parameters=torch.zeros(1, len(engine.pose_names)),
            face_expr_coeffs=torch.zeros(1, 72),
            apply_correctives=False,
        )
    return skeleton[0].detach().cpu().numpy()[:, :3]


def feature_config(engine, joints):
    names = engine.joint_names
    left_eye = joints[names.index("l_eye")]
    right_eye = joints[names.index("r_eye")]
    jaw = joints[names.index("c_jaw")]
    mid_eye = (left_eye + right_eye) / 2
    mouth = np.array(
        [0.0, mid_eye[1] + (jaw[1] - mid_eye[1]) * 0.7, mid_eye[2] + 2.0]
    )
    expr = {n: i for i, n in enumerate(engine.expression_names)}
    return {
        "eyeL": {
            "detect": expr["eyesClosed_L"],
            "center": left_eye,
            "radius": 4.0,
            "window": (1.6, 0.55),
            "xspan": (1.5, 4.5),
            "plot": (2.6, 1.6),
        },
        "eyeR": {
            "detect": expr["eyesClosed_R"],
            "center": right_eye,
            "radius": 4.0,
            "window": (1.6, 0.55),
            "xspan": (1.5, 4.5),
            "plot": (2.6, 1.6),
        },
        "mouth": {
            "detect": expr["jawDrop"],
            "center": mouth,
            "radius": 5.0,
            "xspan": (2.5, 6.0),
            "grow_bar": 3.5,
            "plot": (3.2, 2.2),
        },
    }


def deformation(neutral, deformed, faces):
    t0, t1 = neutral[faces], deformed[faces]
    e0 = np.stack(
        [t0[:, 1] - t0[:, 0], t0[:, 2] - t0[:, 1], t0[:, 0] - t0[:, 2]], axis=1
    )
    e1 = np.stack(
        [t1[:, 1] - t1[:, 0], t1[:, 2] - t1[:, 1], t1[:, 0] - t1[:, 2]], axis=1
    )
    n0 = np.linalg.norm(e0, axis=2)
    n1 = np.linalg.norm(e1, axis=2)
    n0[n0 < 1e-9] = 1e-9
    ratio = n1 / n0
    return np.maximum(ratio.max(axis=1), (1.0 / ratio).max(axis=1))


def edge_adjacency(faces):
    edge_to_faces = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for edge in (
            tuple(sorted((a, b))),
            tuple(sorted((b, c))),
            tuple(sorted((a, c))),
        ):
            edge_to_faces[edge].append(fi)
    neighbours = [set() for _ in range(len(faces))]
    for shared in edge_to_faces.values():
        for fi in shared:
            for fj in shared:
                if fi != fj:
                    neighbours[fi].add(fj)
    return neighbours


def connected_sets(seed, neighbours):
    remaining = set(int(s) for s in seed)
    comps = []
    while remaining:
        queue, comp = deque([remaining.pop()]), []
        while queue:
            face = queue.popleft()
            comp.append(face)
            for other in neighbours[face]:
                if other in remaining:
                    remaining.remove(other)
                    queue.append(other)
        comps.append(comp)
    return sorted(comps, key=len, reverse=True)


def strip_stats(component, neighbours):
    inset = set(component)
    counts = [len([n for n in neighbours[f] if n in inset]) for f in component]
    return {
        "n": len(component),
        "max_neighbours": max(counts),
        "endpoints": sum(1 for c in counts if c == 1),
    }


def order_strip(component, neighbours):
    """Order strip faces end to end; None when the shape is not a strip."""
    inset = set(component)
    ends = [f for f in component if sum(1 for n in neighbours[f] if n in inset) == 1]
    if len(ends) != 2:
        return None
    order, seen, current = [], set(), ends[0]
    while True:
        order.append(current)
        seen.add(current)
        nxt = [n for n in neighbours[current] if n in inset and n not in seen]
        if not nxt:
            break
        current = nxt[0]
    return order if len(order) == len(component) else None


def grow_row(order, neighbours, centroids, dev, grow_bar, max_turn_cos=0.3):
    """Extend a strip along its own axis into lower-deformation neighbours."""
    inset, path = set(order), list(order)
    for _ in range(80):
        grown = False
        for end in (0, -1):
            tip, prev = path[end], path[end + (1 if end == 0 else -1)]
            direction = centroids[tip] - centroids[prev]
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                continue
            direction /= norm
            best, best_cos = None, max_turn_cos
            for cand in neighbours[tip]:
                if cand in inset or dev[cand] <= grow_bar:
                    continue
                step = centroids[cand] - centroids[tip]
                step_norm = float(np.linalg.norm(step))
                if step_norm < 1e-9:
                    continue
                cos = float(step @ direction / step_norm)
                if cos > best_cos:
                    best, best_cos = cand, cos
            if best is not None:
                if end == 0:
                    path.insert(0, best)
                else:
                    path.append(best)
                inset.add(best)
                grown = True
        if not grown:
            break
    return path


def extent_xspan(faces, neutral, sel):
    xs = neutral[faces[sorted(sel)]][:, :, 0]
    return float(xs.max() - xs.min())


def grow_axis(sel, seeds, neighbours, centroids, dev, bar, window, cos_min=0.2):
    """Extend a band from seed tips along fixed ±x directions.

    Seeds are the core strip's ends: second layers would otherwise hijack
    the x-extreme start and the growth would never reach the corners.
    """
    sel = set(sel)
    inset = set(window.tolist())
    for tip, sign in seeds:
        for _ in range(80):
            best, best_cos = None, cos_min
            for cand in neighbours[tip]:
                if cand in sel or dev[cand] <= bar or cand not in inset:
                    continue
                step = centroids[cand] - centroids[tip]
                norm = float(np.linalg.norm(step))
                if norm < 1e-9:
                    continue
                cos = float(step[0] * sign / norm)
                if cos > best_cos:
                    best, best_cos = cand, cos
            if best is None:
                break
            sel.add(best)
            tip = best
    return sel


def sideways_completion(sel, neighbours, dev, bar, window):
    """Absorb hot edge-neighbours (second seal layers) to a fixpoint."""
    sel = set(sel)
    inset = set(window.tolist())
    frontier = list(sel)
    while frontier:
        tip = frontier.pop()
        for cand in neighbours[tip]:
            if cand in sel or dev[cand] <= bar or cand not in inset:
                continue
            sel.add(cand)
            frontier.append(cand)
    return sel


def lip_window(centroids, lip_y, x_extent=2.4, y_band=0.35):
    """Faces on the lip seam, including its wrap around the corners."""
    return np.flatnonzero(
        (np.abs(centroids[:, 0]) <= x_extent)
        & (np.abs(centroids[:, 1] - lip_y) <= y_band)
    )


def complete_mouth_band(core, faces, neutral, dev, neighbours, lip_y, extent,
                        side_bar):
    """All hot bands in the lip window + warm sides + corner continuation."""
    centroids = neutral[faces].mean(axis=1)
    sel = set(core)
    core_listed = sorted(core)
    core_seeds = [
        (
            core_listed[int(np.argmax(centroids[core_listed][:, 0] * sign))],
            sign,
        )
        for sign in (1.0, -1.0)
    ]
    # Second seal layers share only vertices with the core row, never edges,
    # so heat alone must find them; lip peaks at dev ~2.6, seal starts at 3.9.
    # The window covers the seam's wrap around the corners; cheek past |x|
    # 2.4 and folds off the seam stay untouched.
    wide = lip_window(centroids, lip_y)
    hot = wide[dev[wide] > 3.9]
    for comp in connected_sets(hot, neighbours):
        if len(comp) >= 2:
            sel.update(comp)
    sel = sideways_completion(sel, neighbours, dev, side_bar, wide)
    # Grow from the core ends (reaching around second layers) and from the
    # current extremes (catching stragglers past hot corners). Cold bridges
    # stay inside the lip extent; only heat reaches around the corners.
    sel_listed = sorted(sel)
    seeds = dict(
        core_seeds
        + [
            (
                sel_listed[
                    int(np.argmax(centroids[sel_listed][:, 0] * sign))
                ],
                sign,
            )
            for sign in (1.0, -1.0)
        ]
    )
    narrow = lip_window(centroids, lip_y, x_extent=extent)
    sel = grow_axis(
        sel, list(seeds.items()), neighbours, centroids, dev, 2.0, narrow
    )
    sel = sideways_completion(sel, neighbours, dev, side_bar, wide)
    # No size cap: the window, heat bar, margin check, and mandatory plot /
    # render review before --apply are the guards. LOD0's real seal incl.
    # its corner wrap is ~260 faces; any fixed multiple of the core would
    # false-trigger there while adding nothing elsewhere.
    return sel


def complete_eye_band(core, faces, neutral, dev, neighbours, cfg):
    """Core strip plus axis-grown corner continuation (both canthi)."""
    centroids = neutral[faces].mean(axis=1)
    sel = set(core)
    core_listed = sorted(core)
    seeds = [
        (
            core_listed[int(np.argmax(centroids[core_listed][:, 0] * sign))],
            sign,
        )
        for sign in (1.0, -1.0)
    ]
    wx, wy = cfg["window"]
    wide = np.flatnonzero(
        (np.abs(centroids[:, 0] - cfg["center"][0]) < wx + 0.6)
        & (np.abs(centroids[:, 1] - cfg["center"][1]) < wy + 0.3)
    )
    # Corners stretch less than the aperture midline; measured corner-web
    # faces go down to dev 1.14 while rigid skin is exactly 1.0, so the bar
    # sits at 1.1 and the window stops the nose/temple.
    return grow_axis(sel, seeds, neighbours, centroids, dev, 1.1, wide)


def border_margin(sel, dev, neighbours):
    inset = set(sel)
    border = set()
    for face in sel:
        for other in neighbours[face]:
            if other not in inset:
                border.add(other)
    if not border:
        return float("inf")
    return float(np.median(dev[sorted(sel)]) / max(np.median(dev[sorted(border)]), 1e-9))


def valid_strip(sel, faces, neutral, dev, neighbours, xspan, margin_min=1.5):
    """Single ordered row of physical width with a deformation border."""
    if order_strip(sel, neighbours) is None:
        return None
    return valid_band(sel, faces, neutral, dev, neighbours, xspan, margin_min)


def valid_band(sel, faces, neutral, dev, neighbours, xspan, margin_min=1.5):
    """Band of physical width with a deformation border (need not order)."""
    if len(sel) < 2:
        return None
    xw = extent_xspan(faces, neutral, sel)
    if not (xspan[0] <= xw <= xspan[1]):
        return None
    margin = border_margin(sel, dev, neighbours)
    if margin < margin_min:
        return None
    return {"xspan": xw, "border_margin": margin}


def detect_eye_lod1(faces, neutral, dev, cfg, neighbours):
    """Long thin triangles spanning the aperture (one row, full height)."""
    centroids = neutral[faces].mean(axis=1)
    roi = np.flatnonzero(
        np.linalg.norm(centroids - cfg["center"], axis=1) < cfg["radius"]
    )
    spans = neutral[faces[roi]][:, :, 1]
    median_h = float(np.median(spans.max(axis=1) - spans.min(axis=1)))
    heights = neutral[faces][:, :, 1]
    tall = heights.max(axis=1) - heights.min(axis=1) > 2.5 * median_h
    wx, wy = cfg["window"]
    seed = np.flatnonzero(
        (np.abs(centroids[:, 0] - cfg["center"][0]) < wx)
        & (np.abs(centroids[:, 1] - cfg["center"][1]) < wy)
        & tall
        & (dev > 1.5)
    )
    if len(seed) < 2:
        return None
    core = connected_sets(seed, neighbours)[0]
    anchor = valid_strip(core, faces, neutral, dev, neighbours, cfg["xspan"])
    if anchor is None:
        return None
    sel = complete_eye_band(core, faces, neutral, dev, neighbours, cfg)
    check = valid_band(sel, faces, neutral, dev, neighbours, (1.5, 5.5))
    if check is None:
        return None
    return {
        "faces": sorted(sel),
        "method": f"longtri+corners{len(sel) - len(core):+d}",
        "median_h": median_h,
        **check,
    }


def transfer_seal(faces, neutral, dev, cfg, neighbours, lod1):
    """Nearest-LOD1-seal-centroid transfer for non-LOD1 tessellations."""
    centroids = neutral[faces].mean(axis=1)
    wx, wy = cfg["window"]
    window = np.flatnonzero(
        (np.abs(centroids[:, 0] - cfg["center"][0]) < wx)
        & (np.abs(centroids[:, 1] - cfg["center"][1]) < wy)
    )
    if len(window) == 0:
        return None
    cand_xyz = lod1["centroids"][lod1["candidates"]]
    dist2 = ((centroids[window, None, :] - cand_xyz[None, :, :]) ** 2).sum(
        axis=2
    )
    nearest = lod1["candidates"][dist2.argmin(axis=1)]
    dist_seal = np.sqrt(
        ((centroids[window, None, :] - lod1["seal_xyz"][None, :, :]) ** 2).sum(
            axis=2
        )
    ).min(axis=1)
    sel = window[np.isin(nearest, lod1["seal"]) & (dist_seal < 0.3)]
    if len(sel) < 2:
        return None
    core = connected_sets(sel, neighbours)[0]
    anchor = valid_strip(core, faces, neutral, dev, neighbours, cfg["xspan"])
    if anchor is None:
        return None
    grown = complete_eye_band(core, faces, neutral, dev, neighbours, cfg)
    check = valid_band(grown, faces, neutral, dev, neighbours, (1.5, 5.5))
    if check is None:
        return None
    return {
        "faces": sorted(grown),
        "method": f"transfer+corners{len(grown) - len(core):+d}",
        **check,
    }


def detect_mouth(faces, neutral, dev, cfg, neighbours):
    """Largest high-stretch component grown along its own axis."""
    centroids = neutral[faces].mean(axis=1)
    roi = np.flatnonzero(
        np.linalg.norm(centroids - cfg["center"], axis=1) < cfg["radius"]
    )
    for core_bar in (7.0, 6.0, 5.0, 4.0, 3.0, 2.5):
        seed = roi[dev[roi] > core_bar]
        if len(seed) < 2:
            continue
        core = connected_sets(seed, neighbours)[0]
        stats = strip_stats(core, neighbours)
        order = order_strip(core, neighbours)
        if order is None:
            if stats["endpoints"] == 0 and stats["max_neighbours"] <= 2:
                check = valid_strip(
                    core, faces, neutral, dev, neighbours, cfg["xspan"]
                )
                if check is not None:
                    return {
                        "faces": sorted(core),
                        "method": f"loop>{core_bar}",
                        **check,
                    }
            continue
        grown = grow_row(order, neighbours, centroids, dev, cfg["grow_bar"])
        check = valid_strip(
            grown, faces, neutral, dev, neighbours, cfg["xspan"]
        )
        if check is not None:
            return {
                "faces": sorted(grown),
                "method": f"core>{core_bar}",
                **check,
            }
    return None


def lod_data(engine, base, fe, lod):
    faces = engine.lod_faces[lod]
    neutral = engine._render_lod(base, lod) if lod != 1 else base
    return faces, neutral, edge_adjacency(faces)


def detect_mouth_full(faces, neutral, dev, cfg, neighbours, lip_y, extent, lod):
    """Verified core plus hot bands, warm sides, and corner continuation."""
    core = detect_mouth(faces, neutral, dev, cfg, neighbours)
    if core is None:
        return None
    # Coarse faces span less, so LOD6 accepts a lower sideways bar; its lip
    # peaks at dev 1.3, far below either bar.
    side_bar = 3.0 if lod == 6 else 4.0
    sel = complete_mouth_band(
        core["faces"], faces, neutral, dev, neighbours, lip_y, extent,
        side_bar,
    )
    if sel is None:
        return None
    check = valid_band(sel, faces, neutral, dev, neighbours, cfg["xspan"])
    if check is None:
        return None
    ordered = order_strip(sel, neighbours) is not None
    return {
        "faces": sorted(sel),
        "method": f"{core['method']}+band{len(sel) - len(core['faces']):+d}"
        f"{'+ordered' if ordered else ''}",
        **check,
    }


def lod_dev(engine, base, fe, lod, neutral, faces, expr_idx):
    deformed = (
        engine._render_lod(base + fe[expr_idx], lod)
        if lod != 1
        else base + fe[expr_idx]
    )
    return deformation(neutral, deformed, faces)


def analyze(engine):
    base = (
        engine.model.character_torch.blend_shape.base_shape.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    fe = (
        engine.model.face_expressions_model.shape_vectors.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    config = feature_config(engine, neutral_joints(engine))
    report = {}
    lod1 = {}

    faces, neutral, neighbours = lod_data(engine, base, fe, 1)
    report[1] = {}
    for feature in ("eyeL", "eyeR"):
        cfg = config[feature]
        dev = lod_dev(engine, base, fe, 1, neutral, faces, cfg["detect"])
        result = detect_eye_lod1(faces, neutral, dev, cfg, neighbours)
        report[1][feature] = result
        if result is not None:
            centroids = neutral[faces].mean(axis=1)
            seal = np.array(result["faces"])
            wx, wy = cfg["window"]
            lod1[feature] = {
                "seal": seal,
                "seal_xyz": centroids[seal],
                "candidates": np.flatnonzero(
                    (np.abs(centroids[:, 0] - cfg["center"][0]) < wx + 0.6)
                    & (np.abs(centroids[:, 1] - cfg["center"][1]) < wy + 0.55)
                ),
                "centroids": centroids,
            }
    # The verified LOD1 mouth row anchors every LOD's lip geometry.
    cfg = config["mouth"]
    dev = lod_dev(engine, base, fe, 1, neutral, faces, cfg["detect"])
    mouth_core_1 = detect_mouth(faces, neutral, dev, cfg, neighbours)
    if mouth_core_1 is None:
        report[1]["mouth"] = None
        lip_y, extent = 155.75, 1.90
    else:
        seal_xyz = neutral[faces[np.array(mouth_core_1["faces"])]]
        lip_y = float(seal_xyz[:, :, 1].mean())
        extent = round(float(np.abs(seal_xyz[:, :, 0]).max()) + 0.01, 2)
        report[1]["mouth"] = detect_mouth_full(
            faces, neutral, dev, cfg, neighbours, lip_y, extent, 1
        )
    print(f"lip line: y={lip_y:.2f} extent=+/-{extent:.2f}")

    for lod in SUPPORTED_LODS:
        if lod == 1:
            continue
        faces, neutral, neighbours = lod_data(engine, base, fe, lod)
        report[lod] = {}
        for feature in ("eyeL", "eyeR"):
            cfg = config[feature]
            if feature not in lod1:
                report[lod][feature] = None
                continue
            dev = lod_dev(engine, base, fe, lod, neutral, faces, cfg["detect"])
            report[lod][feature] = transfer_seal(
                faces, neutral, dev, cfg, neighbours, lod1[feature]
            )
        cfg = config["mouth"]
        dev = lod_dev(engine, base, fe, lod, neutral, faces, cfg["detect"])
        report[lod]["mouth"] = detect_mouth_full(
            faces, neutral, dev, cfg, neighbours, lip_y, extent, lod
        )
    return report, base, config


def plot_report(report, base, config, engine, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fe = (
        engine.model.face_expressions_model.shape_vectors.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    for lod in SUPPORTED_LODS:
        faces = engine.lod_faces[lod]
        neutral = engine._render_lod(base, lod) if lod != 1 else base
        centroids = neutral[faces].mean(axis=1)
        proj = neutral[:, [0, 1]]
        for feature in FEATURES:
            result = report[lod][feature]
            if result is None:
                continue
            cfg = config[feature]
            sel = set(result["faces"])
            cx, cy = cfg["center"][0], cfg["center"][1]
            wx, wy = cfg["plot"]
            nearby = np.flatnonzero(
                (np.abs(centroids[:, 0] - cx) < wx)
                & (np.abs(centroids[:, 1] - cy) < wy)
            )
            rest = [f for f in nearby if f not in sel]
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.add_collection(
                PolyCollection(
                    proj[faces[rest]],
                    facecolors=["#d5dbe3"] * len(rest),
                    edgecolors="none",
                )
            )
            ax.add_collection(
                PolyCollection(
                    proj[faces[sorted(sel)]],
                    facecolors=["#d62728"] * len(sel),
                    edgecolors="k",
                    linewidths=0.3,
                )
            )
            ax.set_xlim(cx - wx, cx + wx)
            ax.set_ylim(cy - wy, cy + wy)
            ax.set_aspect("equal")
            ax.set_title(f"LOD{lod} {feature}: {len(sel)} faces ({result['method']})")
            fig.savefig(out_dir / f"cut_lod{lod}_{feature}.png", dpi=120)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--plots", default=None)
    args = parser.parse_args()
    if bool(args.analyze) == bool(args.apply):
        parser.error("pass exactly one of --analyze or --apply")

    engine = MHREngine(DEFAULT_MODEL_PATH)
    report, base, config = analyze(engine)
    failed = []
    for lod in SUPPORTED_LODS:
        for feature in FEATURES:
            result = report[lod][feature]
            if result is None:
                failed.append((lod, feature))
                print(f"LOD {lod} {feature}: NO VALID STRIP FOUND")
            else:
                print(
                    f"LOD {lod} {feature}: n={len(result['faces'])} "
                    f"{result['method']} xspan={result['xspan']:.2f} "
                    f"margin={result['border_margin']:.2f}"
                )
    if args.plots:
        plot_report(report, base, config, engine, args.plots)
        print(f"plots -> {args.plots}")
    if args.analyze:
        return 0 if not failed else 1
    if failed:
        print("refusing to apply with unresolved features:", failed)
        return 1
    backup = LOD_TOPOLOGY_PATH.with_suffix(".npz.bak")
    shutil.copy2(LOD_TOPOLOGY_PATH, backup)
    print(f"backup -> {backup}")
    with np.load(LOD_TOPOLOGY_PATH) as topology:
        arrays = {key: topology[key] for key in topology.files}
    for lod in SUPPORTED_LODS:
        if lod == 1:
            faces = engine.source_faces.astype(np.uint32)
        else:
            faces = arrays[f"lod{lod}_faces"].astype(np.uint32)
        drop = set()
        for feature in FEATURES:
            drop.update(report[lod][feature]["faces"])
        keep = np.array([i for i in range(len(faces)) if i not in drop])
        arrays[f"lod{lod}_faces"] = faces[keep]
    np.savez(LOD_TOPOLOGY_PATH, **arrays)
    print(f"wrote {LOD_TOPOLOGY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
