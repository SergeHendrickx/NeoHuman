from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\vggt_head6_known_cams.py")
t = p.read_text(encoding="utf-8")
old = 'print(p, json.dumps({k: report["frames"][p][k] for k in ("yaw", "scale", "n_points")}), flush=True)'
new = 'print(p.name, json.dumps({k: report["frames"][p.name][k] for k in ("yaw", "scale", "n_points")}), flush=True)'
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("print key fixed")
