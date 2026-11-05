"""Parse the Sapiens 'Sociopticon' 308-keypoint format from keypoints308.py using ast
(no import/exec of third-party code) and dump names/ids/categories."""
import ast, json
from collections import Counter

KP308 = r"A:\dev\GNM-V\pivot\sapiens2\sapiens\pose\configs\_base_\keypoints308.py"
tree = ast.parse(open(KP308, encoding="utf-8").read())


def val(n):
    if isinstance(n, ast.Constant):
        return n.value
    if isinstance(n, (ast.List, ast.Tuple)):
        return [val(e) for e in n.elts]
    if isinstance(n, ast.Dict):
        return {val(k): val(v) for k, v in zip(n.keys, n.values)}
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "dict":
        d = {}
        for kw in n.keywords:
            d[kw.arg] = val(kw.value)
        return d
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return -val(n.operand)
    if isinstance(n, ast.Name):
        return "<name:%s>" % n.id
    raise ValueError(type(n).__name__)


for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "dataset_info" for t in node.targets):
        di = {}
        for kw in node.value.keywords:
            try:
                di[kw.arg] = val(kw.value)
            except Exception:
                di[kw.arg] = None

ki = di["keypoint_info"]
names = {int(i): v["name"] for i, v in ki.items()}
print("keypoints in format:", len(ki), "| ids %d..%d" % (min(names), max(names)))
cats = ["body_keypoint_names", "foot_keypoint_names", "left_hand_keypoint_names",
        "right_hand_keypoint_names", "extra_keypoint_names", "face_keypoint_names"]
name2id = {v: k for k, v in names.items()}
for c in cats:
    v = di.get(c) or []
    ids = [name2id[n] for n in v]
    print("%-26s %4d  ids %4d..%4d contiguous=%s" % (c, len(v), min(ids), max(ids),
                                                    ids == list(range(min(ids), max(ids) + 1))))

teeth = [i for i, n in names.items() if "tooth" in n.lower() or "teeth" in n.lower()]
after = [i for i in sorted(names) if i not in set(teeth)]
print("\nteeth-like: %d ids %s" % (len(teeth), teeth))
print("format minus teeth: %d  (== 308 -> %s)" % (len(after), len(after) == 308))

face_ids = [i for i in after if i >= 70]
face_names = [names[i] for i in face_ids]
print("\nkeypoints with format-id >=70 (after teeth removal): %d" % len(face_ids))
print("first 20 (format order):")
for i in range(0, 20):
    print("   fmt_id %3d -> row %3d  %s" % (face_ids[i], 70 + i, face_names[i]))
print("last 8:")
for i in range(len(face_ids) - 8, len(face_ids)):
    print("   fmt_id %3d -> row %3d  %s" % (face_ids[i], 70 + i, face_names[i]))


def cls(n):
    l = n.lower()
    for k in ("jaw", "mouth", "lip", "eye", "eyebrow", "brow", "nose", "ear", "cheek",
              "chin", "forehead", "temple", "iris", "pupil", "tooth", "tongue", "beard",
              "hair", "neck"):
        if k in l:
            return k
    return "other"


print("\nface class counts:", dict(Counter(cls(n) for n in face_names)))
print("first 70 (format order, after teeth removal) ==")
print([names[i] for i in after[:70]])

json.dump({"names": {str(k): v for k, v in sorted(names.items())},
           "after_teeth": after, "face_ids": face_ids, "face_names": face_names},
          open(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\_sapiens308.json", "w"))
print("\nwrote _sapiens308.json")

