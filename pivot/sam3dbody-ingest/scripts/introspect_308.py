import torch, numpy as np

MHR = r"A:\dev\GNM-V\pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\assets\mhr_model.pt"
CKPT = r"A:\dev\GNM-V\pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\model.ckpt"

mhr = torch.jit.load(MHR, map_location="cpu")
with torch.no_grad():
    verts, skel = mhr(torch.zeros(1, 45), torch.zeros(1, 204), torch.zeros(1, 72))
V = verts[0]                                  # (18439,3) cm
J = skel[0][:, :3]                            # (127,3) cm
M = torch.cat([V, J], 0)                      # (18566,3)

sd = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)
km = sd["head_pose.keypoint_mapping"]         # (308, 18566)
K = (km @ M).numpy()                          # (308,3) cm
print("KPSHAPE", K.shape)
for i in range(0, 308, 20):
    print(i, [round(float(x), 1) for x in K[i]])
head = [(i, K[i]) for i in range(308) if K[i][1] > 145]
print("n keypoints above y=145cm (head):", len(head), "rows:", [i for i, _ in head])
ys = np.array([k[1] for _, k in head])
if len(ys):
    print("head kp y range", round(float(ys.min()), 1), round(float(ys.max()), 1))
# how many rows 70+ are in the head?
rows70 = [i for i, k in head if i >= 70]
print("rows>=70 in head:", len(rows70), "first20", rows70[:20])
# row sparsity for rows 70+
nz = (km[70:] != 0).sum(1)
print("rows>=70 nnz min/max", int(nz.min()), int(nz.max()))
# closest-pair spread in head: face cluster tighter?
if len(rows70) > 10:
    P = np.array([K[i] for i in rows70])
    print("head kp centroid", [round(float(x), 1) for x in P.mean(0)])
    print("head kp extent", [round(float(x), 1) for x in (P.max(0) - P.min(0))])
