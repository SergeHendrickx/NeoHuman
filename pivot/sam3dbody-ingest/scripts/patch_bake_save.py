from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\bake_face_texture.py")
t = p.read_text(encoding="utf-8")
old = '''vcol = np.zeros((V, 3), np.float64)
ok = wsum > 0
vcol[ok] = acc[ok] / wsum[ok, None]
print("vertices with colour:", int(ok.sum()), "/", V)'''
new = '''vcol = np.zeros((V, 3), np.float64)
ok = wsum > 0
vcol[ok] = acc[ok] / wsum[ok, None]
np.savez_compressed(OUT / "texture_bake_data.npz", vcol=vcol, texcoords=texcoords,
                    texcoord_faces=texcoord_faces, faces=faces, canon=canon)
print("vertices with colour:", int(ok.sum()), "/", V, "-> texture_bake_data.npz")'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("save bake data added")
