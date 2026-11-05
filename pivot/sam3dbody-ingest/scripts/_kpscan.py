"""Read-only scan of 3DB checkpoint MHR buffers: keypoint_mapping (308, 18439+127)."""
import torch, json, sys, time

CKPT = r"A:\dev\GNM-V\pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\model.ckpt"
t0 = time.time()
sd = torch.load(CKPT, map_location="cpu", weights_only=True, mmap=True)
print("load_s=%.1f n_keys=%d" % (time.time() - t0, len(sd)))

km_key = [k for k in sd if "keypoint_mapping" in k]
print("keypoint_mapping keys:", km_key)
km = sd[km_key[0]]
print("shape", tuple(km.shape), "dtype", km.dtype)
km = km.float()
NV = 18439
NJ = 127

rowsum = km.sum(1)
nonzero_per_row = (km != 0).sum(1)
print("rowsum: min %.6f max %.6f  distinct(rounded6)=%d" % (rowsum.min(), rowsum.max(), len(set(rowsum.round(decimals=6).tolist()))))
print("nonzero_per_row histogram:", torch.bincount(nonzero_per_row).tolist())

# per-row argmax column, split verts/joints
amax = km.argmax(1)
print("\nrow  argmax_col  in_space  maxval  nnz  rowsum")
for r in list(range(0, 75)) + [75, 80, 100, 150, 200, 250, 300, 307]:
    c = int(amax[r]); v = float(km[r, c]); n = int(nonzero_per_row[r])
    space = "VERT" if c < NV else "JOINT"
    extra = ""
    if c >= NV:
        extra = " joint_id=%d" % (c - NV)
    print("%3d  %8d  %-7s %7.4f  %3d  %.6f%s" % (r, c, space, v, n, float(rowsum[r]), extra))

# how many rows are exactly one-hot
onehot = (nonzero_per_row == 1).sum().item()
print("\none-hot rows:", onehot, "/ 308")
# any row with weights on joints?
joint_touch = (km[:, NV:].abs().sum(1) > 0).sum().item()
print("rows touching joint columns:", joint_touch)
print("rows 0-69 touching joints:", int((km[:70, NV:].abs().sum(1) > 0).sum()))
print("rows 70-307 touching joints:", int((km[70:, NV:].abs().sum(1) > 0).sum()))

# duplicate target vertices among rows?
tgt = [(int(amax[r]), round(float(km[r, int(amax[r])]), 4)) for r in range(308)]
print("distinct argmax cols:", len(set(t[0] for t in tgt)))

# weight-value taxonomy
vals = km[km != 0]
print("nonzero weights: n=%d min %.4f max %.4f" % (vals.numel(), vals.min(), vals.max()))

# Faces / mesh buffers present
buf = sorted([k for k in sd if any(s in k for s in ("faces", "joint_rotation", "nonhand_param_idxs", "hand_joint_idxs"))])
print("\nbuffers:", buf)
for k in buf:
    print("  %-45s %s %s" % (k, tuple(sd[k].shape), sd[k].dtype))

# save the 308 argmax table for reference
out = {str(r): [int(amax[r]), float(km[r, int(amax[r])]), int(nonzero_per_row[r]), float(rowsum[r])] for r in range(308)}
with open(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\_kp308rows.json", "w") as f:
    json.dump(out, f)
print("\nwrote _kp308rows.json")
