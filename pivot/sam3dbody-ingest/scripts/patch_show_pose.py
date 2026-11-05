from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\show_pose_strip_test.py")
t = p.read_text(encoding="utf-8")
old = 'sys.path.insert(0, str(BASE / "outputs" / "meta-image-test" / "ortho-neutral"))'
new = 'sys.path.insert(0, str(BASE / "outputs" / "_global" / "meta-image-test" / "ortho-neutral"))'
assert old in t, "anchor not found"
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("ortho path fixed")
