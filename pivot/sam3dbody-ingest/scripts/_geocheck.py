"""Geometric validation: do checkpoint keypoint_mapping rows 70..307 land on the head,
and does the l/r naming from the Sapiens-308 format agree with the mesh's x axis?
Read-only: loads the MHR TorchScript and the 308 JSON dumped earlier."""
import json, torch

MHR = r"A:\dev\GNM-V\pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\assets\mhr_model.pt"
CKPT = r"A:\dev\GNM-V\pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\model.ckpt"
J = json.load(open(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\_sapiens308.json"))
face_names = J["face_names"]          # 238 names, rows 70..307
names70 = J["names"]                  # format id -> name

sd = torch.load(CKPT, map_location="cpu", weights_only=True, mmap=True)
M = sd["head_pose.keypoint_mapping"].float()

mhr = torch.jit.load(MHR, map_location="cpu")
with torch.no_grad():
    out = mhr(torch.zeros(1, 45), torch.zeros(1, 204), torch.zeros(1, 72))
verts, skel = out[0], out[1]
print("verts", tuple(verts.shape), "skel", tuple(skel.shape))
joint_xyz = skel[..., 0:3]
vj = torch.cat([verts, joint_xyz], dim=1)[0]           # 18566 x 3
kp = M @ vj                                    # (308,18566)@(18566,3) -> 308 x 3, native cm
print("kp0 (nose)   ", [round(float(x), 2) for x in kp[0]])
print("kp1 (l_eye)  ", [round(float(x), 2) for x in kp[1]])
print("kp2 (r_eye)  ", [round(float(x), 2) for x in kp[2]])
print("kp3 (l_ear)  ", [round(float(x), 2) for x in kp[3]])
print("kp4 (r_ear)  ", [round(float(x), 2) for x in kp[4]])
print("kp69 (neck)  ", [round(float(x), 2) for x in kp[69]])

nose = kp[0]
d = (kp - nose).norm(dim=1)
print("\nrows 0..69 : dist-to-nose median %.1f cm (min %.1f max %.1f)" %
      (d[:70].median(), d[:70].min(), d[:70].max()))
print("rows 70..307: dist-to-nose median %.1f cm (min %.1f max %.1f)  <= head-radius test" %
      (d[70:].median(), d[70:].min(), d[70:].max()))
print("rows 70..307 all within 15 cm of nose:", bool((d[70:] < 15).all()))

# x axis sign vs l_/r_ naming in the 238 face names
agree = tot = 0
amb = 0
for i, n in enumerate(face_names):
    row = 70 + i
    base = n[2:] if n.startswith(("l_", "r_")) else None
    if base is None:
        amb += 1
        continue
    s = 1 if n.startswith("r_") else -1
    tot += 1
    if (float(kp[row, 0]) * s) > 0:
        agree += 1
print("\nl/r prefix test on face rows: %d/%d agree with x-axis sign (%.1f%%); %d names lack l_/r_ prefix"
      % (agree, tot, 100.0 * agree / tot, amb))

# same test on the known body names rows 1..6
for row, expect in [(1, "left_eye"), (2, "right_eye"), (3, "left_ear"), (4, "right_ear"),
                    (5, "left_shoulder"), (6, "right_shoulder")]:
    print("  row %d %-14s x=%7.2f" % (row, expect, float(kp[row, 0])))

# rows of interest: mouth/lip/jaw/teeth/chin/nose
def show(pat):
    print("\n-- names containing %r --" % pat)
    for i, n in enumerate(face_names):
        if pat in n.lower():
            print("  row %3d  %s" % (70 + i, n))
for p in ("jaw", "mouth", "lip", "chin", "tongue", "tooth"):
    show(p)

# compact 4-per-line table of all 238 rows for the report
lines = []
cells = ["%d:%s" % (70 + i, n) for i, n in enumerate(face_names)]
for k in range(0, len(cells), 4):
    lines.append(" Â· ".join(cells[k:k + 4]))
open(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\_face238.txt", "w", encoding="utf-8").write("\n".join(lines))
print("\nwrote _face238.txt (%d rows in %d lines)" % (len(cells), len(lines)))

# extra: does keypoint_mapping have negative weights anywhere (extrapolation risk)?
print("min weight", float(M.min()), " max weight", float(M.max()))
print("rows with any negative weight:", int((M < 0).any(dim=1).sum()))


