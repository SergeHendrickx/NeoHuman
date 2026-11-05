import ast
import json
from pathlib import Path

BASE = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest")
FILES = {
    "neutral-all": BASE / "outputs/_global/meta-image-test/neutral-all/make_neutral_all.py",
    "bald-all": BASE / "outputs/_global/meta-image-test/bald-all/make_bald_all.py",
    "tex-hat_off/neutral": BASE / "outputs/tex/variants/make_variants.py",
    "emiru-face-grid+body-strip": BASE / "scripts/muse_layout_emiru.py",
    "body-strip-retry": BASE / "scripts/muse_pose_retry.py",
    "fullbody-grid-2x2": BASE / "scripts/muse_fullbody_grid.py",
    "turnaround-6panel": BASE / "scripts/muse_turnaround.py",
    "turnaround-all": BASE / "scripts/muse_turnaround_all.py",
    "head6-grid": BASE / "scripts/head6_matching.py",
    "guide-bald3": BASE / "outputs/_global/meta-image-test/guide-test/make_guide_and_send.py",
    "guide-face": BASE / "outputs/_global/meta-image-test/guide-test/make_face_guide.py",
}
want = ["PROMPT", "FACE_PROMPT", "BODY_PROMPT", "HAT", "NEUTRAL"]
out = {}
for tag, p in FILES.items():
    if not p.exists():
        out[tag] = f"MISSING {p}"
        continue
    src = p.read_text(encoding="utf-8")
    tree = ast.parse(src)
    d = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in want:
                try:
                    d[name] = ast.literal_eval(node.value)
                except Exception as e:
                    d[name] = f"<could not eval: {e}>"
    out[tag] = d
print(json.dumps(out, indent=1, ensure_ascii=False))
