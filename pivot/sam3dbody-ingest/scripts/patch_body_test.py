from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\show_body_test.py")
t = p.read_text(encoding="utf-8")
old1 = 'ref_lm = fits["front"]["lm"][BODY_ROWS]'
new1 = 'ref_lm = fits["front"]["lm"][70:272]'
old2 = 'lm = fits[n]["lm"][BODY_ROWS]'
new2 = 'lm = fits[n]["lm"][70:272]'
assert old1 in t and old2 in t
t = t.replace(old1, new1).replace(old2, new2)
p.write_text(t, encoding="utf-8")
print("body test now aligns on head landmarks")
