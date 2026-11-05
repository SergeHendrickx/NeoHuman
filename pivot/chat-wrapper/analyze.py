"""analyze.py — thin import-wrapper over sam3dbody-ingest/run_all.py stages.

No model code is duplicated here. Everything heavy lives in
``pivot/sam3dbody-ingest/run_all.py`` (+ ``joint_solve.py``); this module only:
  * imports those helpers (single source of truth),
  * keeps ONE model resident at a time (one-slot manager),
  * runs one photo through ingest→pose→maps→body→report,
  * reports via callbacks so chat_server.py can stream SSE.

Usage from chat_server.py:
    from analyze import run_photo
    run_photo(bgr, job_dir, hooks, opts)  # hooks: stage_start/stage_end/log/progress/artifact/warn + cancelled()
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INGEST = REPO / "pivot" / "sam3dbody-ingest"

sys.path.insert(0, str(INGEST))

STAGES = ["ingest", "pose", "maps", "body", "report"]
STAGE_WEIGHT = {"ingest": 2, "pose": 4, "maps": 6, "body": 80, "report": 8}


class StageError(Exception):
    def __init__(self, message, remedy=""):
        super().__init__(message)
        self.remedy = remedy


class Cancelled(StageError):
    def __init__(self):
        super().__init__("cancelled by user", remedy="resend the photo to retry")


class ModelSlot:
    """One resident model at a time. Evicts previous before loading next."""

    def __init__(self):
        self.name: str | None = None
        self.obj = None

    def ensure(self, name, loader, hooks=None):
        import torch

        if self.name == name and self.obj is not None:
            return self.obj
        if self.obj is not None:
            try:
                del self.obj
            except Exception:
                pass
            self.obj = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if hooks is not None:
                hooks.log("body", f"evicted model slot (was {self.name})")
        t0 = time.time()
        self.obj = loader()
        self.name = name
        if hooks is not None:
            hooks.log("body", f"loaded {name} in {time.time()-t0:.1f}s")
        return self.obj

    def evict(self, hooks=None):
        import torch

        if self.obj is not None:
            try:
                del self.obj
            except Exception:
                pass
            self.obj = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass


def _vram_mb():
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "now_mb": round(torch.cuda.memory_allocated() / 2**20, 1),
                "peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            }
    except Exception:
        pass
    return {"now_mb": 0.0, "peak_mb": 0.0}


def validate_bbox(bbox, W, H):
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False, "bbox unreadable"
    if not (x2 > x1 and y2 > y1):
        return False, "bbox has zero/negative area"
    if (x2 - x1) * (y2 - y1) < 16:
        return False, "bbox area < 16px^2"
    if not (0 <= x1 < W and 0 <= y1 < H and 0 < x2 <= W and 0 < y2 <= H):
        return True, "clamped"
    return True, "ok"


def _check_cancel(hooks):
    try:
        if hooks.cancelled():
            raise Cancelled()
    except Cancelled:
        raise
    except Exception:
        pass


def run_photo(bgr, job_dir: Path, hooks, opts: dict | None = None):
    """Run one photo, mirroring run_all.py per-image logic. Raises StageError."""
    import cv2
    import numpy as np

    import run_all as R

    try:
        from joint_solve import solve_joint as _solve_joint
    except Exception:
        _solve_joint = None

    opts = opts or {}
    full_maps = bool(opts.get("full_maps", False))
    t_all = time.time()
    timings: dict[str, float] = {}
    H, W = bgr.shape[:2]
    slot = ModelSlot()

    job_dir = Path(job_dir)
    for sub in ("input", "sapiens", "threedb", "sapiens/raw", "threedb/raw", "logs"):
        (job_dir / sub).mkdir(parents=True, exist_ok=True)

    def checkpoint(stage):
        _check_cancel(hooks)

    # ---- ingest ----
    hooks.stage_start("ingest")
    t0 = time.time()
    bbox = opts.get("bbox")
    if bbox is not None:
        ok, why = validate_bbox(bbox, W, H)
        if not ok:
            raise StageError(f"person box invalid ({why})", remedy="redraw the box or send no box for auto-detect")
        hooks.log("ingest", f"manual bbox {bbox} ({why})")
    else:
        bbox = [0, 0, W - 1, H - 1]
        hooks.log("ingest", f"full-frame bbox, auto person_box after pose ({W}x{H})")
    cv2.imwrite(str(job_dir / "input" / "01_original.png"), bgr)
    hooks.artifact("ingest", "original", job_dir / "input" / "01_original.png",
                   kind="image", label="01 original", provenance="live")
    timings["ingest"] = round(time.time() - t0, 2)
    hooks.stage_end("ingest", {"timings": timings.copy(), **_vram_mb()})
    checkpoint("ingest")
    if hooks.choice("ingest", f"Photo received ({W}x{H}). Find the person automatically?",
                    [{"id": "auto", "label": "Continue · auto-detect person box (Recommended)", "primary": True},
                     {"id": "cancel", "label": "Cancel run"}],
                    default="auto") == "cancel":
        raise Cancelled()

    # ---- pose (Sapiens2 308, two-pass like run_all) ----
    hooks.stage_start("pose")
    t0 = time.time()
    pose_model = slot.ensure("pose", R.load_pose, hooks)
    kp_any, _ = R.pose_run(pose_model, bgr)
    if list(bbox) == [0, 0, W - 1, H - 1]:
        bbox = R.person_box(kp_any, W, H)
        hooks.log("pose", f"auto person_box {[round(float(v),1) for v in bbox]}")
    ok, why = validate_bbox(bbox, W, H)
    if not ok:
        raise StageError(f"person box invalid after pose ({why})", remedy="try a different photo or manual box")
    kp, score = R.pose_run(pose_model, bgr, bbox)
    cv2.imwrite(str(job_dir / "sapiens" / "02_sapiens_keypoints.jpg"), R.draw_pose(bgr, kp))
    np.savez_compressed(job_dir / "sapiens" / "raw" / "sapiens_pose.npz",
                        kp=kp.astype(np.float32), score=np.asarray(score, np.float32),
                        bbox=np.asarray(bbox, np.float32))
    (job_dir / "sapiens" / "raw" / "person_box.json").write_text(json.dumps({
        "bbox_xyxy": [round(float(v), 3) for v in bbox],
        "pose_score_mean": round(float(np.mean(score)), 4),
        "pose_score_median": round(float(np.median(score)), 4)}, indent=2))
    hooks.artifact("pose", "keypoints", job_dir / "sapiens" / "02_sapiens_keypoints.jpg",
                   kind="image", label="02 Sapiens 308 keypoints", provenance="live",
                   note="green 70 body · magenta 238 face")
    timings["pose"] = round(time.time() - t0, 2)
    slot.evict()
    pose_med = round(float(np.median(score)), 4)
    hooks.stage_end("pose", {"timings": timings.copy(), "pose_median": pose_med,
                             "bbox": [round(float(v), 1) for v in bbox], **_vram_mb()})
    checkpoint("pose")
    if hooks.choice("pose", f"308 keypoints found (pose score {pose_med}). Continue to seg/normal/depth maps?",
                    [{"id": "maps", "label": "Continue · seg / normal / depth (Recommended)", "primary": True},
                     {"id": "cancel", "label": "Cancel run"}],
                    default="maps") == "cancel":
        raise Cancelled()

    # ---- maps (seg + normal + depth 0.4b; 1b behind full_maps) ----
    hooks.stage_start("maps")
    t0 = time.time()
    seg_model = slot.ensure("seg", lambda: R.load_dense(
        "seg", "seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
        "seg/sapiens2_0.4b_seg.safetensors"), hooks)
    try:
        from sapiens.dense.visualizers import SegVisualizer
        seg_vis = SegVisualizer(class_palette_type="dome29", with_labels=False)
    except Exception:
        seg_vis = None
    labels, overlay = R.seg_run(seg_model, seg_vis, bgr)
    skin = R.skin_mask_from_labels(labels)
    cv2.imwrite(str(job_dir / "sapiens" / "raw" / "sapiens_seg_labels.png"), labels)
    cv2.imwrite(str(job_dir / "sapiens" / "raw" / "skin_mask.png"), (skin.astype(np.uint8) * 255))
    cv2.imwrite(str(job_dir / "sapiens" / "04_sapiens_seg.jpg"), overlay)
    hooks.artifact("maps", "seg", job_dir / "sapiens" / "04_sapiens_seg.jpg",
                   kind="image", label="04 segmentation", provenance="live")
    hooks.progress("maps", pct=30)
    slot.evict()
    checkpoint("maps")

    normal_model = slot.ensure("normal", lambda: R.load_dense(
        "normal", "normal/metasim_render_people/sapiens2_0.4b_normal_metasim_render_people-1024x768.py",
        "normal/sapiens2_0.4b_normal.safetensors"), hooks)
    sn, vis_n = R.normal_run(normal_model, bgr, labels)
    sn_f32 = sn.astype(np.float32)
    np.save(job_dir / "sapiens" / "raw" / "sapiens_normals.npy", sn.astype(np.float16))
    cv2.imwrite(str(job_dir / "sapiens" / "05_sapiens_normal.jpg"), vis_n)
    hooks.artifact("maps", "normal", job_dir / "sapiens" / "05_sapiens_normal.jpg",
                   kind="image", label="05 normals", provenance="live")
    hooks.progress("maps", pct=60)
    slot.evict()
    checkpoint("maps")

    depth_model = slot.ensure("depth", lambda: R.load_dense(
        "pointmap", "pointmap/render_people/sapiens2_0.4b_pointmap_render_people-1024x768.py",
        "pointmap/sapiens2_0.4b_pointmap.safetensors"), hooks)
    depth, vis_d = R.depth_run(depth_model, bgr, labels)
    np.save(job_dir / "sapiens" / "raw" / "sapiens_depth_m.npy", depth.astype(np.float32))
    cv2.imwrite(str(job_dir / "sapiens" / "03_sapiens_depth.jpg"), vis_d)
    person = labels > 0
    depth_range = ([round(float(np.percentile(depth[person], 1)), 3),
                    round(float(np.percentile(depth[person], 99)), 3)] if person.any() else None)
    hooks.artifact("maps", "depth", job_dir / "sapiens" / "03_sapiens_depth.jpg",
                   kind="image", label="03 depth", provenance="live")
    slot.evict()
    checkpoint("maps")

    want_full = bool(full_maps)
    if not want_full:
        pick = hooks.choice("maps", f"Base maps done (depth {depth_range}). 3D fit next, or run the slow 1b maps first?",
                            [{"id": "body", "label": "Continue · 3D fit (Recommended)", "primary": True},
                             {"id": "full", "label": "Run full_maps first (1b pointmap + matting, slower)"},
                             {"id": "cancel", "label": "Cancel run"}],
                            default="body")
        if pick == "cancel":
            raise Cancelled()
        want_full = (pick == "full")
    full_maps = want_full

    matting_stats = depth1b_range = None
    if full_maps:
        hooks.log("maps", "full_maps=1: running 1b pointmap + 1b matting")
        pm1b = slot.ensure("pointmap1b", lambda: R.load_dense(
            "pointmap", "pointmap/render_people/sapiens2_1b_pointmap_render_people-1024x768.py",
            R.SAPIENS_1B_POINTMAP), hooks)
        depth1b, vis1b = R.depth_run(pm1b, bgr, labels)
        np.save(job_dir / "sapiens" / "raw" / "sapiens_pointmap_1b_depth_m.npy", depth1b.astype(np.float32))
        cv2.imwrite(str(job_dir / "sapiens" / "10_pointmap1b_depth.jpg"), vis1b)
        hooks.artifact("maps", "pointmap1b", job_dir / "sapiens" / "10_pointmap1b_depth.jpg",
                       kind="image", label="10 pointmap-1b depth", provenance="live")
        if person.any():
            depth1b_range = [round(float(np.percentile(depth1b[person], 1)), 3),
                             round(float(np.percentile(depth1b[person], 99)), 3)]
        slot.evict()
        checkpoint("maps")
        mat1b = slot.ensure("matting1b", lambda: R.load_dense(
            "matting", "matting/gss_p3m_metasim/sapiens2_1b_matting_gss_p3m_metasim-1024x768.py",
            R.SAPIENS_1B_MATTING), hooks)
        alpha = R.matting_run(mat1b, bgr)
        np.save(job_dir / "sapiens" / "raw" / "sapiens_matting_1b_alpha.npy", np.asarray(alpha))
        cv2.imwrite(str(job_dir / "sapiens" / "11_matting1b_alpha.png"),
                    (np.clip(np.asarray(alpha), 0, 1) * 255).astype(np.uint8))
        hooks.artifact("maps", "matting1b", job_dir / "sapiens" / "11_matting1b_alpha.png",
                       kind="image", label="11 matting-1b alpha", provenance="live")
        slot.evict()
    timings["maps"] = round(time.time() - t0, 2)
    hooks.stage_end("maps", {"timings": timings.copy(), "depth_m": depth_range,
                             "full_maps": full_maps, **_vram_mb()})
    checkpoint("maps")

    # ---- body (3DB fit + identity_refine + joint + 06/07/08/09) ----
    hooks.stage_start("body")
    t0 = time.time()
    hooks.log("body", "loading SAM 3D Body (body mode, pinned)")
    model, est, faces = slot.ensure("3db", R.load_3db, hooks)
    out, verts, f = R.fit_3db(est, bgr, np.asarray(bbox, np.float32))
    for k in ("pred_vertices", "shape_params", "mhr_model_params", "focal_length", "pred_cam_t",
              "pred_keypoints_2d"):
        if k not in out:
            raise StageError(f"3DB output missing key: {k}", remedy="checkpoint/model mismatch — re-check weights")
    if not bool(np.isfinite(np.asarray(out["pred_vertices"])).all()):
        raise StageError("3DB returned non-finite mesh (bad box/photo)", remedy="retry with auto box")
    import torch

    raw = job_dir / "threedb" / "raw"
    np.savez_compressed(raw / "3db_outputs.npz",
                        **{k: np.asarray(v) for k, v in out.items()
                           if isinstance(v, (np.ndarray, list)) and k != "faces"},
                        faces=np.asarray(faces, np.int32))
    np.save(raw / "3db_vertices_fit.npy", np.asarray(verts, np.float64))
    R._write_obj(job_dir / "threedb" / "mesh_fit.obj", np.asarray(verts, np.float64), faces,
                 "SAM 3D Body fit (camera frame, metres)")
    hooks.log("body", f"3DB fit |beta|={float(np.linalg.norm(out['shape_params'])):.2f} f={float(f):.0f}px")
    hooks.progress("body", pct=25)
    checkpoint("body")

    kp3_before = np.asarray(out["pred_keypoints_2d"], np.float64)
    kp3, cam_t, verts0 = kp3_before, np.asarray(out["pred_cam_t"], np.float64), verts
    beta_ref = np.asarray(out["shape_params"], np.float64)
    params_ref = np.asarray(out["mhr_model_params"], np.float64)
    used, joint, expr_ref = "raw", None, np.zeros(72, np.float64)
    verts_neutral = verts0

    # normal supervision for the joint solve (same as run_all)
    n_target = n_mask = None
    if "sn_f32" in dir() and skin is not None:
        try:
            sn_t = sn_f32 * np.array([1.0, -1.0, -1.0], np.float32)
            n_target = sn_t / np.maximum(np.linalg.norm(sn_t, axis=2, keepdims=True), 1e-8)
            n_mask = (skin & (np.linalg.norm(sn_f32, axis=2) > 0.5)
                      & ~R.feature_exclusion_mask(kp, W, H))
        except Exception as e:
            hooks.warn("body", f"normal supervision skipped: {e}")

    try:
        ref = R.identity_refine(model, out, kp, score, W, H, f, cam_t)
    except Exception as e:
        ref = {"applied": False, "error": f"{type(e).__name__}: {e}"}
        hooks.warn("body", f"identity refine failed, using 3DB fit: {e}")
    if ref.get("applied"):
        beta_ref = np.asarray(ref.get("beta", beta_ref), np.float64)
        hooks.log("body", "identity-only refine applied")
    else:
        hooks.warn("body", f"identity refine skipped: {ref.get('reason') or ref.get('error')}")
    hooks.progress("body", pct=45)
    checkpoint("body")

    if _solve_joint is not None:
        try:
            joint = _solve_joint(model, out, kp, score, W, H, f, cam_t,
                                 beta_init=beta_ref, identity_only=ref, iters=300,
                                 normal_target=n_target, normal_mask=n_mask,
                                 faces=faces, normal_weight=0.5)
        except Exception as exc:
            joint = {"applied": False, "error": f"{type(exc).__name__}: {exc}"}
            hooks.warn("body", f"joint solve crashed, using {used}: {exc}")
    else:
        joint = {"applied": False, "error": "joint_solve unavailable"}
    if joint is not None and joint.get("applied"):
        beta_ref = np.asarray(joint["shape_params"], np.float64)
        params_ref = np.asarray(joint["mhr_model_params"], np.float64)
        cam_t = np.asarray(joint["cam_t"], np.float64)
        expr_ref = np.asarray(joint.get("expr_params", np.zeros(72)), np.float64)
        verts_root, lm_r = R.mesh_and_landmarks(model, out, beta_ref, model_params=params_ref, expr=expr_ref)
        verts_r = verts_root + cam_t[None]
        verts_neutral_root, _ = R.mesh_and_landmarks(model, out, beta_ref, model_params=params_ref)
        verts_neutral = verts_neutral_root + cam_t[None]
        kp3 = R.project_cam(lm_r[:70], f, cam_t, W, H)
        used = "joint"
        R._write_obj(job_dir / "threedb" / "mesh_refined.obj", np.asarray(verts_r, np.float64), faces,
                     "refined mesh (camera frame, metres)")
        hooks.log("body", f"joint solve applied (|expr|={float(np.linalg.norm(expr_ref)):.2f})")
    elif ref.get("applied"):
        verts_root, lm_r = R.mesh_and_landmarks(model, out, beta_ref)
        cam_t = np.asarray(out["pred_cam_t"], np.float64)
        verts_r = verts_root + cam_t[None]
        kp3 = R.project_cam(lm_r[:70], f, cam_t, W, H)
        used = "identity"
        R._write_obj(job_dir / "threedb" / "mesh_refined.obj", np.asarray(verts_r, np.float64), faces,
                     "refined mesh (camera frame, metres)")
    else:
        verts_r = verts0
    hooks.progress("body", pct=65)
    checkpoint("body")

    cv2.imwrite(str(job_dir / "threedb" / "06_3db_keypoints.jpg"), R.draw_3db_match(bgr, kp3, kp))
    hooks.artifact("body", "match", job_dir / "threedb" / "06_3db_keypoints.jpg",
                   kind="image", label="06 3DB vs Sapiens match", provenance="live")
    try:
        vcol = R.texture_mesh(verts_r, faces, bgr, f, W, H)
        timg, tmask = R.render_rgb(verts_r, faces, f, W, H, vcol=vcol)
        overlay = R.overlay_photo(bgr, timg, tmask, kp3, alpha=0.62)
        front = cv2.cvtColor(timg, cv2.COLOR_RGB2BGR)
        tile = max(W, H)
        side_t = cv2.cvtColor(R.render_clay_view(verts_r, faces, f, tile, yaw=35.0, vcol=vcol), cv2.COLOR_RGB2BGR)
        side_c = cv2.cvtColor(R.render_clay_view(verts_r, faces, f, tile, yaw=35.0), cv2.COLOR_RGB2BGR)
        panel = R.compose_grid([overlay, front, side_t, side_c],
                               ["3DB mesh + kps over photo", "3DB render (photo colours)",
                                "same mesh, 3/4 view", "geometry only (clay)"], cols=2)
        cv2.imwrite(str(job_dir / "threedb" / "07_3db_mesh_keypoints.jpg"), panel)
        hooks.artifact("body", "sheet", job_dir / "threedb" / "07_3db_mesh_keypoints.jpg",
                       kind="image", label="07 mesh sheets (2x2)", provenance="live")
        face = kp[R.FACE_ROWS]
        x1, y1 = np.floor(face.min(0)).astype(int)
        x2, y2 = np.ceil(face.max(0)).astype(int)
        pad = int(0.2 * max(x2 - x1, y2 - y1))
        x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
        x2, y2 = min(x2 + pad, W), min(y2 + pad, H)
        crops = []
        for tag, vv, kk in (("before", verts0, kp3_before), ("after", verts_r, kp3)):
            img_b, mask_b = R.render_rgb(vv, faces, f, W, H, colour=R.MESH_COLOUR, ss=1.5)
            ov = R.overlay_photo(bgr, img_b, mask_b, kk, alpha=0.55)
            crops.append(ov[y1:y2, x1:x2].copy())
        cv2.imwrite(str(job_dir / "threedb" / "08_head_calibration.jpg"), R.compose_panels(
            crops, ["before: 3DB identity", f"after: {used} solve on Sapiens2 keypoints"]))
        hooks.artifact("body", "calibration", job_dir / "threedb" / "08_head_calibration.jpg",
                       kind="image", label="08 head calibration", provenance="live")
        photo = bgr.copy()
        for r in R.FACE_ROWS:
            x, y = kp[r]
            if np.isfinite([x, y]).all() and 0 <= x < W and 0 <= y < H:
                cv2.circle(photo, (int(x), int(y)), 2, (0, 255, 0), -1)
        img_n, _ = R.render_rgb(verts_neutral, faces, f, W, H, colour=R.MESH_COLOUR, ss=1.6)
        img_e, _ = R.render_rgb(verts_r, faces, f, W, H, colour=R.MESH_COLOUR, ss=1.6)

        def _crop_scale(im):
            c = im[y1:y2, x1:x2].copy()
            s = max(1.0, min(3.0, 820.0 / max(1, c.shape[0])))
            return cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)

        expr_l2 = (joint or {}).get("metrics", {}).get("expr_l2", float(np.linalg.norm(expr_ref)))
        cv2.imwrite(str(job_dir / "threedb" / "09_expression.jpg"), R.compose_panels(
            [_crop_scale(photo), _crop_scale(img_n), _crop_scale(img_e)],
            ["photo + Sapiens2 face kps", "joint solve, neutral jaw",
             f"joint + expression fit (|expr|={float(expr_l2):.2f})"]))
        hooks.artifact("body", "expression", job_dir / "threedb" / "09_expression.jpg",
                       kind="image", label="09 expression", provenance="live")
    except Exception as exc:
        hooks.warn("body", f"mesh render failed ({type(exc).__name__}), fallback 07: {exc}")
        cv2.imwrite(str(job_dir / "threedb" / "07_3db_mesh_keypoints.jpg"), R.draw_3db_match(bgr, kp3, kp))
        hooks.artifact("body", "sheet", job_dir / "threedb" / "07_3db_mesh_keypoints.jpg",
                       kind="image", label="07 fallback match", provenance="live",
                       note=f"render failed: {type(exc).__name__}")
    if not bool(np.isfinite(np.asarray(verts_r)).all()):
        raise StageError("3DB refined mesh non-finite", remedy="retry with auto box")

    (job_dir / "threedb" / "identity_joint.json").write_text(json.dumps({
        "format": "mhr-identity-joint/1",
        "original_shape_params": [round(float(v), 6) for v in np.asarray(out["shape_params"]).reshape(-1)],
        "original_mhr_model_params": [round(float(v), 6) for v in np.asarray(out["mhr_model_params"]).reshape(-1)],
        "original_cam_t": [round(float(v), 6) for v in np.asarray(out["pred_cam_t"]).reshape(-1)],
        "refined_shape_params": [round(float(v), 6) for v in np.asarray(beta_ref).reshape(-1)],
        "refined_mhr_model_params": [round(float(v), 6) for v in np.asarray(params_ref).reshape(-1)],
        "refined_cam_t": [round(float(v), 6) for v in np.asarray(cam_t).reshape(-1)],
        "refined_expr_params": [round(float(v), 6) for v in np.asarray(expr_ref).reshape(-1)],
        "metrics": R._jsonable({k: v for k, v in (joint or {}).items()
                    if k not in ("shape_params", "mhr_model_params", "cam_t")}),
        "refine_used": used}, indent=2))
    hooks.artifact("body", "identity", job_dir / "threedb" / "identity_joint.json",
                   kind="json", label="identity_joint.json", provenance="live")
    slot.evict()
    timings["body"] = round(time.time() - t0, 2)
    ident_l2 = round(float(np.linalg.norm(beta_ref)), 3)
    hooks.stage_end("body", {"timings": timings.copy(), "refine_used": used,
                             "identity_l2": ident_l2, **_vram_mb()})
    checkpoint("body")
    if hooks.choice("body", f"Fit done (identity {used}, |β| {ident_l2}). Write the report?",
                    [{"id": "report", "label": "Continue · write report (Recommended)", "primary": True},
                     {"id": "cancel", "label": "Cancel run"}],
                    default="report") == "cancel":
        raise Cancelled()

    # ---- report ----
    hooks.stage_start("report")
    stats = {"size_wh": [W, H], "bbox_xyxy": [round(float(v), 1) for v in np.asarray(bbox).reshape(-1)],
             "pose_score_median": round(float(np.median(score)), 4),
             "focal_px": round(float(f), 1),
             "cam_t_m": [round(float(v), 3) for v in np.asarray(cam_t).reshape(-1)],
             "identity_l2": round(float(np.linalg.norm(beta_ref)), 3), "refine_used": used,
             "joint_applied": bool((joint or {}).get("applied")),
             "expr_l2": round(float(np.linalg.norm(expr_ref)), 3),
             "depth_m": depth_range, "depth1b_m": depth1b_range if full_maps else None,
             "full_maps": full_maps, "timings_s": timings,
             "total_s": round(time.time() - t_all, 1)}
    (job_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    hooks.artifact("report", "stats", job_dir / "stats.json",
                   kind="json", label="stats.json", provenance="live")
    hooks.stage_end("report", {"stats": stats, **_vram_mb()})
    return stats
