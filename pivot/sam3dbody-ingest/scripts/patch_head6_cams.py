from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\vggt_head6_known_cams.py")
t = p.read_text(encoding="utf-8")
old = 'report["frames"][p] = {'
new = 'report["frames"][p.name] = {'
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("json keys fixed")
