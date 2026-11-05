"""Category -> row ranges for the 238 face rows; jaw/teeth audit over the full 344 format."""
import json, re
from collections import OrderedDict

J = json.load(open(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\_sapiens308.json"))
names = J["names"]
face = J["face_names"]
rows = {70 + i: n for i, n in enumerate(face)}

print("all 344 names containing jaw/tooth/tongue/temple/beard/hair:")
for i in sorted(names, key=int):
    n = names[i]
    if re.search(r"jaw|tooth|teeth|tongue|temple|beard|hair", n, re.I):
        print("   fmt_id %3s %s" % (i, n))

groups = OrderedDict()
groups["nose"] = lambda n: "nose" in n
groups["eyebrow"] = lambda n: "eyebrow" in n
groups["eye (lid/border/corner)"] = lambda n: ("eye" in n) and "eyebrow" not in n and "iris" not in n and "pupil" not in n
groups["iris"] = lambda n: "iris" in n
groups["pupil"] = lambda n: "pupil" in n
groups["mouth+lips"] = lambda n: ("lip" in n) or ("mouth" in n)
groups["chin"] = lambda n: "chin" in n
groups["ear"] = lambda n: "ear" in n and "eyebrow" not in n
used = set()
print("\ncategory -> rows (238 face keypoints)")
for g, f in groups.items():
    r = [row for row, n in rows.items() if f(n)]
    used |= set(r)
    print("  %-24s n=%3d  rows %s" % (g, len(r), "%d..%d" % (min(r), max(r)) if r else "-"))
rest = [row for row in rows if row not in used]
print("  %-24s n=%3d  rows %s" % ("other (contour etc.)", len(rest), str(rest)))

print("\nrows 70..100 with names:")
for row in range(70, 101):
    print("  %3d %s" % (row, rows[row]))
print("\nrows containing'corner_of_eye' / 'center_of_eye':")
for row, n in rows.items():
    if "corner_of_eye" in n or "center_of_eye" in n:
        print("  %3d %s" % (row, n))
