import json
from pathlib import Path

OUT = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\outputs")
rows = []
for d in sorted(OUT.iterdir()):
    s = d / "stats.json"
    if not d.is_dir() or not s.exists():
        continue
    st = json.loads(s.read_text(encoding="utf-8"))
    W, H = st["size_wh"]
    bb = st["person_box_xyxy"]
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    rows.append((d.name, W, H, bw / H * 100, bw / max(1, bh)))
for r in sorted(rows, key=lambda x: -x[3]):
    print("%-34s img %5dx%-5d box_h %5.1f%%  aspect %.2f" % r)
