"""Quick check: load the 1b pointmap + matting heads and run them on one image."""
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import cv2
import numpy as np
import torch
import run_all as R

PM1B = r"D:\AI_Models\3_Vision_Perception\Segmentation\facebook\Sapiens2\pointmap\sapiens2_1b_pointmap.safetensors"
MT1B = r"D:\AI_Models\3_Vision_Perception\Segmentation\facebook\Sapiens2\matting\sapiens2_1b_matting.safetensors"
IMG = BASE / "input-images" / "linkedin-nieuwe-fot.png"
bgr = cv2.imread(str(IMG))
print("image", bgr.shape)

R.sapiens_imports()
import torch.nn.functional as F

m = R.load_dense("pointmap", "pointmap/render_people/sapiens2_1b_pointmap_render_people-1024x768.py", PM1B)
with torch.no_grad():
    data = m.data_preprocessor(m.pipeline(dict(img=bgr)))
    pm, scale = m(data["inputs"])
    pm = pm / scale
pl, pr, pt, pb = data["data_samples"]["meta"]["padding_size"]
pm = pm[:, :, pt:data["inputs"].shape[2] - pb, pl:data["inputs"].shape[3] - pr]
pm = F.interpolate(pm, size=bgr.shape[:2], mode="bilinear", align_corners=False)
d = pm[0, 2].cpu().numpy()
print("pointmap1b depth", d.shape, "range", float(np.nanmin(d)), float(np.nanmax(d)),
      "median", float(np.nanmedian(d)))
del m; R.free({})

m = R.load_dense("matting", "matting/gss_p3m_metasim/sapiens2_1b_matting_gss_p3m_metasim-1024x768.py", MT1B)
with torch.no_grad():
    data = m.data_preprocessor(m.pipeline(dict(img=bgr)))
    print("matting data keys", list(data.keys()), "meta", list((data.get("data_samples") or {}).get("meta", {}).keys()) if isinstance(data.get("data_samples"), dict) else type(data.get("data_samples")))
    a = m(data["inputs"])
    print("matting raw out", tuple(a.shape), float(a.min()), float(a.max()))
    try:
        pl, pr, pt, pb = data["data_samples"]["meta"]["padding_size"]
        a = a[:, :, pt:data["inputs"].shape[2] - pb, pl:data["inputs"].shape[3] - pr]
    except Exception as exc:
        print("no padding crop:", exc)
    a = F.interpolate(a, size=bgr.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
print("alpha", a.shape, "range", float(a.min()), float(a.max()), "mean", float(a.mean()),
      "pixels>0.5", float((a > 0.5).mean()))
del m; R.free({})
print("OK")
