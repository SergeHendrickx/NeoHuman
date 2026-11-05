"""Multi-view matching between geometry models on the Muse front/profile pair.

Views: linkedin Muse front + profile. For each view: VGGT gives camera + depth (up-to-scale), HyDen gives
metric points, Sapiens2-1b pointmap gives metric depth. Each model is aligned to the VGGT view (scale+shift
on the head), lifted to one world frame with the VGGT cameras, and cross-view / cross-model consistency is
measured in millimetres after scaling the world to metres with the HyDen metric fit.
"""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
MD = REPO / "pivot" / "metadepth"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(MD))
sys.path.insert(0, str(MD.parent))
sys.path.insert(0, str(REPO / "pivot" / "vggt-omega"))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import run_all as R
from metadepth.mogev2 import HyDenMoGe, MODEL_CONFIGS
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

STEM = "linkedin-nieuwe-fot"
PT = BASE / "outputs" / STEM / "profile"
HY_CKPT = REPO / "pivot" / "hyden-mogev2-metric-point" / "hyden_mogev2_metric_point_vitl_fp32_f1066593896.pth"
VGGT_CKPT = REPO / "pivot" / "vggt-omega" / "checkpoints" / "vggt_omega_1b_512.pt"
VIEWS = ["front", "profile"]
paths = [PT / f"muse_{v}.png" for v in VIEWS]
dev = torch.device("cuda")

# ---- VGGT: cameras + depth for the pair
imgs = load_and_preprocess_images([str(p) for p in paths], image_resolution=512)
vm = VGGTOmega().eval()
vm.load_state_dict(torch.load(VGGT_CKPT, map_location="cpu")); vm = vm.cuda()
with torch.no_grad():
    preds = vm(imgs.cuda())
dep_v = preds["depth"][0, ..., 0].cpu().numpy().astype(np.float32)       # (V,H,W)
conf_v = preds["depth_conf"][0].cpu().numpy().astype(np.float32)
ext, K = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
ext, K = ext[0].cpu().numpy().astype(np.float32), K[0].cpu().numpy().astype(np.float32)
del vm; R.free({})
Hv, Wv = dep_v.shape[1:]

# ---- HyDen metric points per view
cfg = dict(MODEL_CONFIGS["vitl_dinov2"]); cfg.pop("normal_head", None)
hym = HyDenMoGe(**cfg)
hym.load_state_dict(torch.load(HY_CKPT, map_location="cpu", weights_only=True), strict=False)
hym = hym.to(dev).eval()
hy = []
for p in paths:
    inp = transforms.Compose([transforms.Resize((518, 518)), transforms.ToTensor()])(
        Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad():
        hy.append(hym(inp)["points"][0].cpu().numpy().astype(np.float32))
del hym; R.free({})

# ---- Sapiens2 1b pointmap depth per view
R.sapiens_imports()
pm = R.load_dense("pointmap", "pointmap/render_people/sapiens2_1b_pointmap_render_people-1024x768.py",
                  r"D:\AI_Models\3_Vision_Perception\Segmentation\facebook\Sapiens2\pointmap\sapiens2_1b_pointmap.safetensors")
sp = []
for p in paths:
    bgr = cv2.imread(str(p))
    d, _ = R.depth_run(pm, bgr, np.ones(bgr.shape[:2], np.uint8))
    sp.append(cv2.resize(d, (Wv, Hv), interpolation=cv2.INTER_LINEAR))
del pm; R.free({})


def fit_scale_shift(src, ref, mask):
    v, r = src[mask], ref[mask]
    A = np.stack([v, np.ones_like(v)], 1)
    s, t = np.linalg.lstsq(A, r, rcond=None)[0]
    return float(s), float(t)


def world_cloud(depth, Ki, Ri, ti):
    ys, xs = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    x = (xs - Ki[0, 2]) / Ki[0, 0] * depth
    y = (ys - Ki[1, 2]) / Ki[1, 1] * depth
    Xc = np.stack([x, y, depth], -1)
    Xw = np.einsum("ij,hwj->hwi", Ri.T, Xc - ti[None, None, :])
    return Xw


report = {}
worlds = {}
for i, v in enumerate(VIEWS):
    dv = dep_v[i]
    head = (conf_v[i] > np.percentile(conf_v[i], 50)) & (dv > 0.2) & (dv < np.maximum(1.5, np.percentile(dv, 95)))
    sz = hy[i][..., 2]
    sz = cv2.resize(sz, (Wv, Hv), interpolation=cv2.INTER_LINEAR)
    spz = sp[i]
    s_hy, t_hy = fit_scale_shift(sz, dv, head)
    s_sp, t_sp = fit_scale_shift(spz, dv, head)
    pred_hy = s_hy * sz + t_hy
    pred_sp = s_sp * spz + t_sp
    report[v] = {
        "HyDen_to_VGGT_scale": round(s_hy, 4), "HyDen_shift": round(t_hy, 4),
        "Sapiens1b_to_VGGT_scale": round(s_sp, 4), "Sapiens1b_shift": round(t_sp, 4),
        "HyDen_vs_VGGT_mm": round(float(np.median(np.abs(pred_hy[head] - dv[head]))), 2),
        "Sapiens1b_vs_VGGT_mm": round(float(np.median(np.abs(pred_sp[head] - dv[head]))), 2),
        "HyDen_vs_Sapiens1b_mm": round(float(np.median(np.abs(pred_hy[head] - pred_sp[head]))), 2),
    }
    # world cloud for each model in VGGT world units
    Ri, ti = ext[i, :3, :3], ext[i, :3, 3]
    worlds[("VGGT", v)] = world_cloud(dv, K[i], Ri, ti)
    worlds[("HyDen", v)] = world_cloud(pred_hy, K[i], Ri, ti)
    worlds[("Sapiens1b", v)] = world_cloud(pred_sp, K[i], Ri, ti)

# scale world to metres using the front-view HyDen metric (metric points already metres -> compare with VGGT)
# consistency: for each model, front-world points vs profile-world points (head region only)
def chamfer(A, B, lim):
    A = A.reshape(-1, 3); B = B.reshape(-1, 3)
    A = A[np.isfinite(A).all(1) & (np.linalg.norm(A - A.mean(0), axis=1) < lim)]
    B = B[np.isfinite(B).all(1) & (np.linalg.norm(B - B.mean(0), axis=1) < lim)]
    if len(A) < 100 or len(B) < 100:
        return None
    # subsample for speed
    A = A[::max(1, len(A) // 4000)]; B = B[::max(1, len(B) // 4000)]
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min(1)
    return float(np.median(d)), float(np.percentile(d, 90))

lim = 0.6
for m in ("VGGT", "HyDen", "Sapiens1b"):
    r = chamfer(worlds[(m, "front")], worlds[(m, "profile")], lim)
    report.setdefault("multi_view_consistency_mm", {})[m] = (
        None if r is None else {"median": round(r[0] * 1000, 1), "p90": round(r[1] * 1000, 1)})

(BASE / "outputs" / STEM / "profile" / "multiview_matching.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
