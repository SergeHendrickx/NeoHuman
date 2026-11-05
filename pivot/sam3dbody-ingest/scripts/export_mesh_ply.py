"""PLY exports for Emiru: fitted mesh, A-pose mesh, and (via vggt_fullbody_grid.py) the VGGT point clouds."""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import numpy as np
import trimesh

import run_all as R

JOB = BASE / "outputs" / "Emiru"
OUT = JOB / "threedb"
j = json.loads((OUT / "identity_joint.json").read_text(encoding="utf-8"))
model3d, est, faces = R.load_3db()

beta = np.asarray(j["refined_shape_params"])
params = np.asarray(j["refined_mhr_model_params"])
cam = np.asarray(j["refined_cam_t"])
expr = np.asarray(j["metrics"]["expr_params"])
vroot, _ = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params, expr=expr)
verts = vroot + cam[None]
m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
m.export(OUT / "mesh_fitted.ply")
print("wrote", OUT / "mesh_fitted.ply", len(verts), "verts")

params_c = params.copy()
params_c[:6] = 0.0
params_c[6:136] = 0.0
vroot_c, _ = R.mesh_and_landmarks(model3d, {"mhr_model_params": params_c}, beta, model_params=params_c, expr=np.zeros(72))
mc = trimesh.Trimesh(vertices=vroot_c - vroot_c.mean(0), faces=faces, process=False)
mc.export(OUT / "mesh_apose.ply")
print("wrote", OUT / "mesh_apose.ply", len(vroot_c), "verts")
del model3d; R.free({})
