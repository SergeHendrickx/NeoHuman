"""Smoke test: VGGT-Omega on our Muse front/profile pair, check env + output shapes."""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"A:\dev\GNM-V")
VGGT = REPO / "pivot" / "vggt-omega"
sys.path.insert(0, str(VGGT))

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

PT = REPO / "pivot" / "sam3dbody-ingest" / "outputs" / "linkedin-nieuwe-fot" / "profile"
paths = [str(PT / "muse_front.png"), str(PT / "muse_profile.png")]
print("images:", [Path(p).name for p in paths])
images = load_and_preprocess_images(paths, image_resolution=512)
print("preprocessed:", tuple(images.shape), images.dtype, float(images.min()), float(images.max()))

ckpt = VGGT / "checkpoints" / "vggt_omega_1b_512.pt"
model = VGGTOmega().eval()
sd = torch.load(ckpt, map_location="cpu")
model.load_state_dict(sd)
del sd
model = model.cuda()
with torch.no_grad():
    preds = model(images.cuda())
print("output keys:", sorted(preds.keys()))
for k, v in preds.items():
    if torch.is_tensor(v):
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
depth = preds["depth"]
conf = preds.get("depth_conf")
pose_enc = preds["pose_enc"]
ext, intr = encoding_to_camera(pose_enc, preds["images"].shape[-2:])
print("depth", tuple(depth.shape), "range", float(depth.min()), float(depth.max()))
print("extrinsic", tuple(ext.shape), "intrinsic", tuple(intr.shape))
print("extrinsic", ext[0].cpu().numpy().round(3))
print("intrinsic", intr[0].cpu().numpy().round(3))
print("depth_conf", None if conf is None else (tuple(conf.shape), float(conf.mean())))
torch.cuda.empty_cache()
print("OK")
