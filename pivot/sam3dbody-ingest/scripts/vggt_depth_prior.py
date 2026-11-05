"""VGGT-Omega depth prior for one job: run VGGT on the photo, z-buffer our fitted mesh with the VGGT
camera, solve the 7-DoF (scale+translation) alignment to the 3DB metric frame and measure the depth gap."""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
sys.path.insert(0, str(REPO / "pivot" / "vggt-omega"))

import cv2
import numpy as np
import torch

import run_all as R

STEM = sys.argv[1] if len(sys.argv) > 1 else "tex"
JOB = BASE / "outputs" / STEM
PHOTO = next((BASE / "input-images").rglob(f"{STEM}.*"))
OUT = JOB / "threedb"
OUT.mkdir(parents=True, exist_ok=True)
VGGT_CKPT = REPO / "pivot" / "vggt-omega" / "checkpoints" / "vggt_omega_1b_512.pt"
RES = 512


def zbuffer_depth(verts, faces, f, W, H):
    """Perspective z-buffer of the mesh into a W x H depth map + mask."""
    z = verts[:, 2]
    keep = z > 1e-6
    u = f * verts[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = f * verts[:, 1] / np.maximum(z, 1e-6) + H / 2
    zbuf = np.full((H, W), np.inf, np.float32)
    for t in faces:
        if not keep[t].all():
            continue
        ax, ay, az = u[t[0]], v[t[0]], z[t[0]]
        bx, by, bz = u[t[1]], v[t[1]], z[t[1]]
        cx, cy, cz = u[t[2]], v[t[2]], z[t[2]]
        x1 = max(0, int(np.floor(min(ax, bx, cx)))); x2 = min(W - 1, int(np.ceil(max(ax, bx, cx))))
        y1 = max(0, int(np.floor(min(ay, by, cy)))); y2 = min(H - 1, int(np.ceil(max(ay, by, cy))))
        if x2 < x1 or y2 < y1:
            continue
        det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(det) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x1, x2 + 1) + 0.5, np.arange(y1, y2 + 1) + 0.5)
        w1 = ((gx - ax) * (cy - ay) - (cx - ax) * (gy - ay)) / det
        w2 = ((bx - ax) * (gy - ay) - (gx - ax) * (by - ay)) / det
        w0 = 1.0 - w1 - w2
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue
        rr, cc = np.nonzero(inside)
        zz = (w0[inside] * az + w1[inside] * bz + w2[inside] * cz).astype(np.float32)
        row = rr + y1; col = cc + x1
        upd = zz < zbuf[row, col]
        zbuf[row[upd], col[upd]] = zz[upd]
    return zbuf


def main() -> int:
    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    model3d, est, faces = R.load_3db()
    j = json.loads((OUT / "identity_joint.json").read_text(encoding="utf-8"))
    stats = json.loads((JOB / "stats.json").read_text(encoding="utf-8"))
    bgr = cv2.imread(str(PHOTO))
    Hp, Wp = bgr.shape[:2]
    f = float(stats["focal_px"])
    beta = np.asarray(j["refined_shape_params"], np.float64)
    params = np.asarray(j["refined_mhr_model_params"], np.float64)
    cam_t = np.asarray(j["refined_cam_t"], np.float64)
    expr = np.asarray(j.get("metrics", {}).get("expr_params", np.zeros(72)), np.float64)
    vroot, lm = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta,
                                     model_params=params, expr=expr)
    verts = vroot + cam_t[None]

    # --- VGGT on the photo
    images = load_and_preprocess_images([str(PHOTO)], image_resolution=RES)
    vmodel = VGGTOmega().eval()
    sd = torch.load(VGGT_CKPT, map_location="cpu")
    vmodel.load_state_dict(sd); del sd
    vmodel = vmodel.cuda()
    with torch.no_grad():
        preds = vmodel(images.cuda())
    depth_v = preds["depth"][0, 0, ..., 0].cpu().numpy().astype(np.float32)   # 1 view
    conf_v = preds["depth_conf"][0, 0].cpu().numpy().astype(np.float32)
    _, intr = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
    K = intr[0, 0].cpu().numpy().astype(np.float32)
    del vmodel; R.free({})
    Hv, Wv = depth_v.shape
    fv = float(K[0, 0]); cx = float(K[0, 2]); cy = float(K[1, 2])
    print(f"[vggt] views {depth_v.shape}, K fx={fv:.1f} cx={cx:.1f} cy={cy:.1f}, conf mean {conf_v.mean():.2f}")

    # --- our mesh into the VGGT camera (same pinhole convention, so only focal/centre differ)
    zv = verts[:, 2]
    uv = np.stack([fv * verts[:, 0] / np.maximum(zv, 1e-6) + cx,
                   fv * verts[:, 1] / np.maximum(zv, 1e-6) + cy], 1)
    Dm = zbuffer_depth(verts, faces, fv, Wv, Hv)
    valid = np.isfinite(Dm) & (depth_v > 1e-3)
    print(f"[vggt] overlap pixels: {valid.sum()} ({100*valid.mean():.1f}%)")

    # per-pixel correspondence: mesh depth at pixel vs VGGT depth
    zm = Dm[valid]; zvg = depth_v[valid]; cf = conf_v[valid]
    good = cf > np.percentile(cf, 30) if cf.size else np.ones_like(cf, bool)
    s = float(np.median(zm[good] / np.maximum(zvg[good], 1e-6)))
    # translation: after scaling VGGT depth, compare backprojected x/y/z offsets
    xs = (np.arange(Wv)[None, :] - cx) / fv
    ys = (np.arange(Hv)[:, None] - cy) / fv
    X = np.stack([xs * depth_v, ys * depth_v, depth_v], -1)   # VGGT points
    yy, xx = np.nonzero(valid)
    P_v = X[yy, xx]
    P_m = np.stack([(xx - cx) / fv * zm, (yy - cy) / fv * zm, zm], 1)
    # solve X_m ~ s*X_v + t (rotation identity, first camera)
    A = np.zeros((3 * len(P_v), 4)); b = np.zeros(3 * len(P_v))
    A[0::3, 0] = P_v[:, 0]; A[0::3, 1] = 1; b[0::3] = P_m[:, 0]
    A[1::3, 0] = P_v[:, 1]; A[1::3, 2] = 1; b[1::3] = P_m[:, 1]
    A[2::3, 0] = P_v[:, 2]; A[2::3, 3] = 1; b[2::3] = P_m[:, 2]
    fit = np.linalg.lstsq(A, b, rcond=None)[0]
    s_fit, tx, ty, tz = map(float, fit)
    resid = (P_m - (s_fit * P_v + np.array([tx, ty, tz]))) * 1000.0
    print(f"[vggt] scale median {s:.4f} ls {s_fit:.4f} | t mm [{tx*1000:.1f}, {ty*1000:.1f}, {tz*1000:.1f}]")
    print(f"[vggt] aligned depth residual: median {np.median(np.abs(resid[:,2])):.1f} mm, "
          f"p90 {np.percentile(np.abs(resid[:,2]),90):.1f} mm, mean {np.abs(resid[:,2]).mean():.1f} mm")

    # depth error map for visualization (in metres, signed)
    diff = np.zeros_like(depth_v)
    target = np.zeros_like(depth_v)
    target[valid] = s_fit * depth_v[valid] + tz
    diff[valid] = Dm[valid] - target[valid]
    vis = np.clip(diff / 0.05 * 127 + 127, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    vis[~valid] = 40
    cv2.imwrite(str(OUT / "vggt_depth_error.jpg"), vis)
    np.savez_compressed(OUT / "vggt_prior.npz",
                        depth=depth_v, conf=conf_v, K=K, scale=float(s_fit),
                        translation_m=np.array([tx, ty, tz], np.float64),
                        mesh_depth=Dm.astype(np.float32), valid=valid)
    (OUT / "vggt_prior.json").write_text(json.dumps({
        "photo": str(PHOTO), "vggt_res": [Wv, Hv], "K": K.round(3).tolist(),
        "scale": round(s_fit, 5), "translation_m": [round(tx, 5), round(ty, 5), round(tz, 5)],
        "depth_residual_mm": {"median": round(float(np.median(np.abs(resid[:, 2]))), 2),
                              "p90": round(float(np.percentile(np.abs(resid[:, 2]), 90)), 2),
                              "mean": round(float(np.abs(resid[:, 2]).mean()), 2)},
        "depth_vggt_m": {"p1": round(float(np.percentile(depth_v, 1)), 3),
                         "p99": round(float(np.percentile(depth_v, 99)), 3)},
    }, indent=2), encoding="utf-8")
    print("wrote", OUT / "vggt_prior.npz", OUT / "vggt_depth_error.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
