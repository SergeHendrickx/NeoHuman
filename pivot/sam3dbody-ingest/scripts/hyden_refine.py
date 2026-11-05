"""Joint refine with a dense metric-depth term from Meta HyDen-MoGeV2 metric point map.

Optimises identity(45) + mhr_model_params(204) + cam_t(3) from the existing joint solution, adding a
Huber depth loss on face-surface vertices against the metric point map (aligned to our camera frame by a
robust 1-D scale+shift fit). Keypoint loss/weights/caps follow joint_solve conventions.
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

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import joint_solve as J
import run_all as R
from metadepth.mogev2 import HyDenMoGe, MODEL_CONFIGS

STEM = sys.argv[1] if len(sys.argv) > 1 else "tex"
DEPTH_W = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
JOB = BASE / "outputs" / STEM
CKPT = REPO / "pivot" / "hyden-mogev2-metric-point" / "hyden_mogev2_metric_point_vitl_fp32_f1066593896.pth"
DEV = torch.device("cuda")

model3d, est, faces_np = R.load_3db()
hp = model3d.head_pose
MAP = hp.keypoint_mapping.detach().float()
faces_t = torch.as_tensor(faces_np, dtype=torch.long, device=DEV)

j = json.loads((JOB / "threedb" / "identity_joint.json").read_text(encoding="utf-8"))
stats = json.loads((JOB / "stats.json").read_text(encoding="utf-8"))
kpz = np.load(JOB / "sapiens" / "raw" / "sapiens_pose.npz")
kp, score = kpz["kp"][:, :2], kpz["score"]
labels = cv2.imread(str(JOB / "sapiens" / "raw" / "sapiens_seg_labels.png"), cv2.IMREAD_GRAYSCALE)
skin = labels == 3
photo = next((BASE / "input-images").rglob(f"{STEM}.*"))
H, W = cv2.imread(str(photo)).shape[:2]
f = float(stats["focal_px"])
expr0 = torch.tensor(np.asarray(j["metrics"]["expr_params"], np.float32), device=DEV)

# ---- HyDen metric point map
cfg = dict(MODEL_CONFIGS["vitl_dinov2"]); cfg.pop("normal_head", None)
hym = HyDenMoGe(**cfg)
hym.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True), strict=False)
hym = hym.to(DEV).eval()
inp = transforms.Compose([transforms.Resize((518, 518)), transforms.ToTensor()])(
    Image.open(photo).convert("RGB")).unsqueeze(0).to(DEV)
with torch.no_grad():
    pts = hym(inp)["points"][0].cpu().numpy().astype(np.float32)
del hym; R.free({})
z_moge = cv2.resize(pts[..., 2], (W, H), interpolation=cv2.INTER_LINEAR)
np.save(JOB / "sapiens" / "raw" / "hyden_metric_points.npy", pts)
print("hyDen z range", round(float(np.nanmin(pts[..., 2])), 3), round(float(np.nanmax(pts[..., 2])), 3))

# ---- baseline
beta_b = torch.tensor(np.asarray(j["refined_shape_params"], np.float32), device=DEV)
params_b = torch.tensor(np.asarray(j["refined_mhr_model_params"], np.float32), device=DEV)
cam_b = torch.tensor(np.asarray(j["refined_cam_t"], np.float32), device=DEV)


def forward(beta, params, cam):
    v, sk = hp.mhr(beta[None], params[None], expr0[None])
    verts = (v[0] / 100.0) * torch.tensor([1.0, -1.0, -1.0], device=DEV)
    p3 = (MAP @ torch.cat([v[0], sk[0, :, :3]], 0)) / 100.0
    p3 = torch.stack([p3[:, 0], -p3[:, 1], -p3[:, 2]], 1) + cam[None]
    z = p3[:, 2].clamp(min=1e-3)
    uv = torch.stack([f * p3[:, 0] / z, f * p3[:, 1] / z], 1) + torch.tensor([W / 2, H / 2], device=DEV)
    vcam = verts + cam[None]                        # vertices in camera frame (metres)
    vz = vcam[:, 2]
    vuv = torch.stack([f * vcam[:, 0] / vz.clamp(min=1e-3),
                       f * vcam[:, 1] / vz.clamp(min=1e-3)], 1) + torch.tensor([W / 2, H / 2], device=DEV)
    return verts, p3, z, uv, vcam, vz, vuv


with torch.no_grad():
    verts_b, p3_b, z_b, uv_b, vcam_b, vz_b, vuv_b = forward(beta_b, params_b, cam_b)
mask_face = ((vuv_b[:, 0] > 0) & (vuv_b[:, 0] < W) & (vuv_b[:, 1] > 0) & (vuv_b[:, 1] < H)
             & (vz_b > 0.2) & (vz_b < 3.0))
ui = vuv_b[:, 0].clamp(0, W - 1).long(); vi = vuv_b[:, 1].clamp(0, H - 1).long()
mw = MAP[70:272].abs().sum(0).detach().cpu().numpy()
fv_np = np.where(mw > 0.05)[0]
fv_np = fv_np[fv_np < verts_b.shape[0]]          # drop the 127 joint entries of the mapping
face_v = torch.tensor(fv_np, device=DEV)
sup = face_v[mask_face[face_v] & torch.tensor(skin[vi[face_v].cpu().numpy(), ui[face_v].cpu().numpy()], device=DEV)]
sup_uv = vuv_b[sup].detach().cpu().numpy()
sx = np.clip(sup_uv[:, 0], 0, W - 1).astype(int); sy = np.clip(sup_uv[:, 1], 0, H - 1).astype(int)
z_target_raw = z_moge[sy, sx]                     # HyDen z at the supervision pixels
z_mesh = vz_b[sup].detach().cpu().numpy()
ok = np.isfinite(z_target_raw) & (z_target_raw > 0.3) & (z_target_raw < 3.0)
sup = sup[torch.tensor(ok, device=DEV)]
z_target_raw, z_mesh = z_target_raw[ok], z_mesh[ok]
A = np.stack([z_target_raw, np.ones_like(z_target_raw)], 1)
s_fit, t_fit = np.linalg.lstsq(A, z_mesh, rcond=None)[0]
z_tgt = torch.tensor(s_fit * z_target_raw + t_fit, dtype=torch.float32, device=DEV)
print(f"depth supervision: {len(sup)} face vertices | fit z_mesh = {s_fit:.3f}*z_moge + {t_fit:.3f} m")
r0 = (z_mesh - (s_fit * z_target_raw + t_fit)) * 1000
print(f"baseline depth residual: median {np.median(np.abs(r0)):.1f} mm, p90 {np.percentile(np.abs(r0), 90):.1f} mm")

# ---- keypoint loss setup (same conventions as joint_solve)
params0 = np.asarray(j["refined_mhr_model_params"], np.float64)
uv0_np = uv_b.detach().cpu().numpy()
masks = J._eval_masks(kp, score, uv0_np, p3_b.detach().cpu().numpy(), W, H)
sel = np.zeros(308, bool)
for v_ in masks.values():
    sel[v_] = True
w_np = J.BASE_WEIGHT * sel
w_kp = torch.tensor(w_np, dtype=torch.float32, device=DEV)
kp_t = torch.tensor(kp, dtype=torch.float32, device=DEV)


def huber(d, delta):
    a = d.abs()
    return torch.where(a < delta, 0.5 * d * d, delta * (a - 0.5 * delta))


def kp_loss(uv):
    d = torch.linalg.norm(uv - kp_t, dim=1)
    h = huber(d, J.HUBER_DELTA)
    return (h * w_kp).sum() / w_kp.sum().clamp(min=1e-6)


kpc = torch.tensor([W / 2, H / 2], dtype=torch.float32, device=DEV)
beta_v = beta_b.clone().requires_grad_(True)
params_v = params_b.clone()                       # pose/scale fixed: refine identity (+ small camera shift)
cam_v = cam_b.clone().requires_grad_(True)
opt = torch.optim.Adam([beta_v, cam_v], lr=0.005)

for it in range(300):
    opt.zero_grad()
    verts, p3, z, uv, vcam, vz, vuv = forward(beta_v, params_v, cam_v)
    lk = kp_loss(uv)
    dz = (vz[sup] - z_tgt) / 0.01                 # residual in centimetre units
    ld = huber(dz, 1.0).mean()
    prior = ((beta_v - beta_b) ** 2).sum() * 0.05 + ((cam_v - cam_b) ** 2).sum() * 1.0
    loss = lk + DEPTH_W * ld + prior
    loss.backward()
    opt.step()
    with torch.no_grad():
        beta_v.clamp_(-J.IDENTITY_ABS, J.IDENTITY_ABS)
        cam_v.clamp_(cam_b - J.TRANS_CAP_M, cam_b + J.TRANS_CAP_M)
    if it % 50 == 0:
        print(f"it {it:3d} kp {float(lk.detach()):.2f} depth {float(ld.detach()):.3f}")

with torch.no_grad():
    verts_a, p3_a, z_a, uv_a, vcam_a, vz_a, vuv_a = forward(beta_v, params_v, cam_v)
    m_before = J._metrics(uv_b.detach().cpu().numpy(), kp, masks)
    m_after = J._metrics(uv_a.detach().cpu().numpy(), kp, masks)
    r1 = (vz_a[sup].detach().cpu().numpy() - z_tgt.detach().cpu().numpy()) * 1000
    d_beta = float((beta_v - beta_b).norm())
    d_expr = float(expr0.norm())

report = {
    "image": STEM, "depth_weight": DEPTH_W,
    "keypoints_before": {k: m_before[k] for k in ("face_err_px", "ear_err_px", "ear_span_px")},
    "keypoints_after": {k: m_after[k] for k in ("face_err_px", "ear_err_px", "ear_span_px")},
    "depth_residual_mm": {
        "before_median": round(float(np.median(np.abs(r0))), 2),
        "before_p90": round(float(np.percentile(np.abs(r0), 90)), 2),
        "after_median": round(float(np.median(np.abs(r1))), 2),
        "after_p90": round(float(np.percentile(np.abs(r1), 90)), 2),
    },
    "n_supervised_vertices": int(len(sup)),
    "depth_fit": {"s": round(float(s_fit), 4), "t_m": round(float(t_fit), 4)},
    "delta_beta_norm": round(d_beta, 3), "expr_l2": round(d_expr, 3),
}
(JOB / "threedb" / "hyden_refine_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

# save refined state
out = {
    "format": "mhr-hyden-refined/1",
    "image": STEM,
    "refined_shape_params": beta_v.detach().cpu().numpy().tolist(),
    "refined_mhr_model_params": params_v.detach().cpu().numpy().tolist(),
    "refined_cam_t": cam_v.detach().cpu().numpy().tolist(),
    "expr_params": expr0.detach().cpu().numpy().tolist(),
    "metrics": report,
}
(JOB / "threedb" / "identity_joint_hyden.json").write_text(json.dumps(out), encoding="utf-8")

# depth-error maps before / after
for tag, zz in (("before", z_b), ("after", z_a)):
    pass
