import json
import torch

CKPT = r"A:\dev\GNM-V\pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\model.ckpt"
MHR = r"A:\dev\GNM-V\pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\assets\mhr_model.pt"

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = ck.get("state_dict", ck)
print("TOPKEYS", list(ck.keys())[:10] if isinstance(ck, dict) else type(ck))
want = [
    "head_pose.faces",
    "head_pose.keypoint_mapping",
    "head_pose.scale_mean",
    "head_pose.scale_comps",
    "head_pose.joint_rotation",
    "head_pose.hand_pose_mean",
    "head_pose.hand_pose_comps",
    "head_pose.hand_joint_idxs_left",
    "head_pose.hand_joint_idxs_right",
    "head_pose.nonhand_param_idxs",
    "head_pose.right_wrist_coords",
    "head_pose.root_coords",
    "head_pose.local_to_world_wrist",
    "init_pose.weight",
    "init_camera.weight",
    "head_pose_hand.proj.layers.0.weight",
    "head_camera.proj.layers.0.weight",
    "head_pose.proj.layers.0.weight",
]
for k in want:
    if k in sd:
        t = sd[k]
        print(f"SHAPE {k}\t{tuple(t.shape)}\t{t.dtype}\tmin {float(t.min()):.4g} max {float(t.max()):.4g}" if t.dtype.is_floating_point else f"SHAPE {k}\t{tuple(t.shape)}\t{t.dtype}")
    else:
        print(f"MISSING {k}")
for pat in ["face", "expr", "teeth", "tongue", "eye", "iris", "jaw", "extra_joint"]:
    hits = [k for k in sd if pat.lower() in k.lower()]
    print(f"PATTERN {pat}: {len(hits)} -> {hits[:10]}")
print("NUMKEYS", len(sd))
hp = [k for k in sd if k.startswith("head_pose.")]
print("HEADPOSE_KEYS", len(hp))
for k in hp:
    v = sd[k]
    print("  ", k, tuple(v.shape) if hasattr(v, "shape") else type(v))

print("=== keypoint_mapping row sums (nonzero span per keypoint) ===")
km = sd["head_pose.keypoint_mapping"]
nz = (km != 0).sum(dim=1)
print("rows with 0 nnz:", int((nz == 0).sum()), "min", int(nz.min()), "max", int(nz.max()))
print("vertex part nnz span:", [(int(torch.nonzero(km[i, :18439])[0].min()), int(torch.nonzero(km[i, :18439])[0].max())) if nz[i] else None for i in range(0, 70, 5)])

print("=== MHR TorchScript ===")
try:
    mhr = torch.jit.load(MHR, map_location="cpu")
    print("MHR_IS_SCRIPT", type(mhr))
    print(mhr.code)
    for name, m in mhr.named_modules():
        if hasattr(m, "code") and m.code is not None:
            print("--- submodule", name, "---")
            print(m.code)
except Exception as e:
    print("MHR_LOAD_ERROR", type(e).__name__, e)

print("=== checkpoint hyper_parameters / keys with 'cfg' ===")
try:
    if isinstance(ck, dict):
        for k in ck:
            if k != "state_dict":
                v = ck[k]
                s = repr(v)
                print("CKPTKEY", k, s[:2000])
except Exception as e:
    print("ERR", e)
