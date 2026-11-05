"""Scale-invariant 3D face proportions from VGGT vs our fitted mesh (same landmarks, same normalisation)."""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(REPO / "pivot" / "vggt-omega"))

import cv2
import numpy as np
import torch

import run_all as R

VGGT_CKPT = REPO / "pivot" / "vggt-omega" / "checkpoints" / "vggt_omega_1b_512.pt"
RES = 512

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def vggt_pointmap(photo: Path):
    images = load_and_preprocess_images([str(photo)], image_resolution=RES)
    model = VGGTOmega().eval()
    sd = torch.load(VGGT_CKPT, map_location="cpu")
    model.load_state_dict(sd); del sd
    model = model.cuda()
    with torch.no_grad():
        preds = model(images.cuda())
    depth = preds["depth"][0, 0, ..., 0].cpu().numpy().astype(np.float32)
    conf = preds["depth_conf"][0, 0].cpu().numpy().astype(np.float32)
    _, intr = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
    K = intr[0, 0].cpu().numpy().astype(np.float32)
    del model; R.free({})
    return depth, conf, K


def backproject(depth, K, uv):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (uv[:, 0] - cx) / fx * depth
    y = (uv[:, 1] - cy) / fy * depth
    return np.stack([x, y, depth], 1)


def sample_depth(depth, uv):
    H, W = depth.shape
    x = np.clip(uv[:, 0], 0, W - 1); y = np.clip(uv[:, 1], 0, H - 1)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    wx = x - x0; wy = y - y0
    return (depth[y0, x0] * (1 - wx) * (1 - wy) + depth[y0, x1] * wx * (1 - wy)
            + depth[y1, x0] * (1 - wx) * wy + depth[y1, x1] * wx * wy)


def ratios(P):
    """P: 6x3 points in the order eyes L/R, ears L/R, chin, nose."""
    eye = 0.5 * (P[0] + P[1]); chin = P[4]; nose = P[5]
    face_h = np.linalg.norm(chin - eye)
    ear_w = np.linalg.norm(P[2] - P[3])
    nose_depth = abs(float(nose[2] - eye[2]))
    return {"ear_w/face_h": round(float(ear_w / face_h), 3),
            "nose_dz/face_h": round(float(nose_depth / face_h), 3)}


def main() -> int:
    stems = sys.argv[1:] or ["linkedin-nieuwe-fot", "tex"]
    model3d, est, faces = R.load_3db()
    report = {}
    for stem in stems:
        job = BASE / "outputs" / stem
        stats = json.loads((job / "stats.json").read_text(encoding="utf-8"))
        photo = next((BASE / "input-images").rglob(f"{stem}.*"))
        j = json.loads((job / "threedb" / "identity_joint.json").read_text(encoding="utf-8"))
        kp = np.load(job / "sapiens" / "raw" / "sapiens_pose.npz")["kp"][:, :2]
        bgr = cv2.imread(str(photo)); Hp, Wp = bgr.shape[:2]

        depth, conf, K = vggt_pointmap(photo)
        Hv, Wv = depth.shape
        uv = kp.copy()
        uv[:, 0] *= Wv / Wp; uv[:, 1] *= Hv / Hp
        rows = np.array([1, 2, 3, 4, 77, 178])
        d = sample_depth(depth, uv[rows])
        ok = np.isfinite(d) & (d > 1e-3) & (d < 50)
        P = backproject(d, K, uv[rows])
        vggt_r = ratios(P) if ok.all() else {"error": "keypoint outside valid depth", "ok": ok.tolist()}

        beta = np.asarray(j["refined_shape_params"]); params = np.asarray(j["refined_mhr_model_params"])
        cam_t = np.asarray(j["refined_cam_t"])
        expr = np.asarray(j.get("metrics", {}).get("expr_params", np.zeros(72)))
        vroot, lm = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta,
                                         model_params=params, expr=expr)
        lm = lm + cam_t[None]
        mesh_r = ratios(lm[rows])

        # also 2D-landmark proportions for reference (pixels)
        e2 = 0.5 * (kp[1] + kp[2]); fh2 = np.linalg.norm(kp[77] - e2)
        kp_r = {"ear_w/face_h": round(float(np.linalg.norm(kp[3] - kp[4]) / fh2), 3),
                "nose_dz/face_h": round(float(np.linalg.norm(kp[178] - e2) / fh2), 3)}
        report[stem] = {"vggt_3d": vggt_r, "mesh_3d": mesh_r, "sapiens2d": kp_r,
                        "vggt_depth_conf_at_landmarks": [round(float(c), 2) for c in
                                                         sample_depth(conf, uv[rows])]}
        print(stem, json.dumps(report[stem]))

    out = BASE / "outputs" / "_global" / "vggt_proportions.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
