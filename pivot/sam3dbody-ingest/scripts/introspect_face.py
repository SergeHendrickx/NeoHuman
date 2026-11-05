import torch, numpy as np

MHR = r"A:\dev\GNM-V\pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\assets\mhr_model.pt"
CKPT = r"A:\dev\GNM-V\pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\model.ckpt"

mhr = torch.jit.load(MHR, map_location="cpu")
with torch.no_grad():
    verts, skel = mhr(torch.zeros(1, 45), torch.zeros(1, 204), torch.zeros(1, 72))
V = verts[0].numpy()  # (18439,3) cm, y up, origin at feet
print("VERTS", V.shape)

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
km = sd["head_pose.keypoint_mapping"]  # (308, 18439+127)
nv = V.shape[0]
names70 = ["nose", "left-eye", "right-eye", "left-ear", "right-ear"]
for i, n in enumerate(names70):
    row = km[i]
    vidx = torch.nonzero(row[:nv]).flatten()
    jidx = torch.nonzero(row[nv:]).flatten()
    print(f"KP{i:02d} {n:10s} n_vert={vidx.numel():3d} n_joint={jidx.numel():3d} "
          f"vertidx[{(int(vidx.min()) if vidx.numel() else -1)}..{(int(vidx.max()) if vidx.numel() else -1)}]")
    if vidx.numel():
        w = row[vidx]
        pos = (V[vidx.numpy()] * w.numpy()[:, None]).sum(0)
        print("    weighted pos cm:", [round(float(x), 2) for x in pos], " weights sum", round(float(w.sum()), 4))
    if jidx.numel():
        print("    joint idx:", jidx.tolist()[:5])

# crude face-region estimate: vertices within r of nose position
nose_row = km[0]
vidx = torch.nonzero(nose_row[:nv]).flatten().numpy()
w = nose_row[vidx].numpy()
p = (V[vidx] * w[:, None]).sum(0)
d = np.linalg.norm(V - p, axis=1)
for r in (5, 7, 9, 12):
    print(f"verts within {r} cm of nose: {int((d < r).sum())}")
# head region above neck: joint 69 'neck' is a keypoint; find joint coords
J = skel[0].numpy()[:, :3]  # (127,3) cm
print("skel first 3 joints cm:", [ [round(float(x),2) for x in J[i]] for i in range(3)])
print("skel y max/min:", round(float(J[:,1].max()),2), round(float(J[:,1].min()),2))
