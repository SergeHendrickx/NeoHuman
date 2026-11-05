from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\finalize_texture.py")
t = p.read_text(encoding="utf-8")
old1 = 'cv2.imwrite(str(OUT / "mhr_face_texture_2048.png"), cv2.cvtColor(tex, cv2.COLOR_RGB2BGR))'
new1 = 'cv2.imwrite(str(OUT / "mhr_face_texture_2048.png"), tex)'
old2 = 'vc = sample(uvv).astype(np.float32) / 255.0'
new2 = 'vc = sample(uvv).astype(np.float32)[:, ::-1] / 255.0'
assert old1 in t and old2 in t
t = t.replace(old1, new1).replace(old2, new2)
p.write_text(t, encoding="utf-8")
print("channel handling fixed")
