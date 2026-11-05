from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\hyden_refine.py")
t = p.read_text(encoding="utf-8")

old = '''beta_v = beta_b.clone().requires_grad_(True)
params_v = params_b.clone().requires_grad_(True)
cam_v = cam_b.clone().requires_grad_(True)
opt = torch.optim.Adam([beta_v, params_v, cam_v], lr=0.01)

for it in range(300):
    opt.zero_grad()
    verts, p3, z, uv, vcam, vz, vuv = forward(beta_v, params_v, cam_v)
    lk = kp_loss(uv)
    dz = (vz[sup] - z_tgt) / 0.01                 # residual in centimetre units
    ld = huber(dz, 1.0).mean()
    prior = ((beta_v - beta_b) ** 2).sum() * 0.02 + ((params_v - params_b) ** 2).sum() * 1e-4
    loss = lk + DEPTH_W * ld + prior
    loss.backward()
    opt.step()
    with torch.no_grad():
        beta_v.clamp_(-J.IDENTITY_ABS, J.IDENTITY_ABS)
        params_v.clamp_(-10.0, 10.0)
        cam_v.clamp_(cam_b - J.TRANS_CAP_M, cam_b + J.TRANS_CAP_M)
    if it % 100 == 0:
        print(f"it {it:3d} kp {float(lk):.2f} depth {float(ld):.3f}")'''
new = '''beta_v = beta_b.clone().requires_grad_(True)
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
        print(f"it {it:3d} kp {float(lk.detach()):.2f} depth {float(ld.detach()):.3f}")'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("optimizer: identity + camera only")
