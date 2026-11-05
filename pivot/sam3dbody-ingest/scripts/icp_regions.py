"""Per-region ICP residual: cloud vs A-pose mesh (head vs clothed body)."""
import json
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from pathlib import Path

OUT = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\outputs\Emiru\threedb")
mesh = trimesh.load(str(OUT / "mesh_apose.ply"), process=False)
mv = np.asarray(mesh.vertices)
sample, _ = trimesh.sample.sample_surface(mesh, 120000)
tree = cKDTree(sample)
cloud = trimesh.load(str(OUT / "vggt_fullbody_cloud_icp.ply"), process=False)
pts = np.asarray(cloud.vertices)

y = mv[:, 1]
lo, hi = np.percentile(y, [0.5, 99.5])
head_c = mv[y < lo + 0.16 * (hi - lo)].mean(0)
head_r = 0.13
d_head = np.linalg.norm(pts - head_c, axis=1)
regions = {
    "head": d_head < head_r,
    "upper_body(clothed)": (pts[:, 1] > lo + 0.16 * (hi - lo)) & (pts[:, 1] < lo + 0.55 * (hi - lo)),
    "lower_body(clothed)": pts[:, 1] > lo + 0.55 * (hi - lo),
}
rep = {}
for name, sel in regions.items():
    if sel.sum() < 200:
        rep[name] = {"n": int(sel.sum())}
        continue
    d, _ = tree.query(pts[sel])
    rep[name] = {"n": int(sel.sum()),
                 "median_mm": round(float(np.median(d) * 1000), 1),
                 "p75_mm": round(float(np.percentile(d, 75) * 1000), 1),
                 "p90_mm": round(float(np.percentile(d, 90) * 1000), 1)}
    print(name, rep[name])
(OUT / "icp_region_residuals.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
