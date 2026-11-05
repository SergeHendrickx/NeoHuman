from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\hyden_refine.py")
t = p.read_text(encoding="utf-8")

old = '''def forward(beta, params, cam):
    v, sk = hp.mhr(beta[None], params[None], expr0[None])
    verts = (v[0] / 100.0) * torch.tensor([1.0, -1.0, -1.0], device=DEV)
    p3 = (MAP @ torch.cat([v[0], sk[0, :, :3]], 0)) / 100.0
    p3 = torch.stack([p3[:, 0], -p3[:, 1], -p3[:, 2]], 1) + cam[None]
    z = p3[:, 2].clamp(min=1e-3)
    uv = torch.stack([f * p3[:, 0] / z, f * p3[:, 1] / z], 1) + torch.tensor([W / 2, H / 2], device=DEV)
    return verts, p3, z, uv'''
new = '''def forward(beta, params, cam):
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
    return verts, p3, z, uv, vcam, vz, vuv'''
assert old in t
t = t.replace(old, new)

old = '''with torch.no_grad():
    verts_b, p3_b, z_b, uv_b = forward(beta_b, params_b, cam_b)
mask_face = (uv_b[:, 0] > 0) & (uv_b[:, 0] < W) & (uv_b[:, 1] > 0) & (uv_b[:, 1] < H) & (z_b > 0.2) & (z_b < 3.0)
ui = uv_b[:, 0].clamp(0, W - 1).long(); vi = uv_b[:, 1].clamp(0, H - 1).long()'''
new = '''with torch.no_grad():
    verts_b, p3_b, z_b, uv_b, vcam_b, vz_b, vuv_b = forward(beta_b, params_b, cam_b)
mask_face = ((vuv_b[:, 0] > 0) & (vuv_b[:, 0] < W) & (vuv_b[:, 1] > 0) & (vuv_b[:, 1] < H)
             & (vz_b > 0.2) & (vz_b < 3.0))
ui = vuv_b[:, 0].clamp(0, W - 1).long(); vi = vuv_b[:, 1].clamp(0, H - 1).long()'''
assert old in t
t = t.replace(old, new)

old = '''z_mesh = z_b[sup].detach().cpu().numpy()'''
new = '''z_mesh = vz_b[sup].detach().cpu().numpy()'''
assert old in t
t = t.replace(old, new)

old = '''    verts, p3, z, uv = forward(beta_v, params_v, cam_v)
    lk = kp_loss(uv)
    dz = (z[sup] - z_tgt) / 0.01                  # residual in centimetre units'''
new = '''    verts, p3, z, uv, vcam, vz, vuv = forward(beta_v, params_v, cam_v)
    lk = kp_loss(uv)
    dz = (vz[sup] - z_tgt) / 0.01                 # residual in centimetre units'''
assert old in t
t = t.replace(old, new)

old = '''    verts_a, p3_a, z_a, uv_a = forward(beta_v, params_v, cam_v)'''
new = '''    verts_a, p3_a, z_a, uv_a, vcam_a, vz_a, vuv_a = forward(beta_v, params_v, cam_v)'''
assert old in t
t = t.replace(old, new)

old = '''    r1 = (z_a[sup].detach().cpu().numpy() - z_tgt.detach().cpu().numpy()) * 1000'''
new = '''    r1 = (vz_a[sup].detach().cpu().numpy() - z_tgt.detach().cpu().numpy()) * 1000'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("vertex projections fixed")
