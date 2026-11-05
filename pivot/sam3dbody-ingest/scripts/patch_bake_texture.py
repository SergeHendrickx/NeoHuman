from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\bake_face_texture.py")
t = p.read_text(encoding="utf-8")

old = '''for tri in texcoord_faces:
    vt = uvp[tri]'''
new = '''for fi in range(len(texcoord_faces)):
    tri = texcoord_faces[fi]
    vt = uvp[tri]'''
assert old in t
t = t.replace(old, new)

old = '''    # triangle -> vertex mapping: uv faces correspond to mesh faces by index
    for vids in (tri,):
        pass
    # find matching mesh triangle by index position
    t_idx = np.where((texcoord_faces == tri).all(1))[0]
    if len(t_idx) == 0:
        continue
    face_v = faces[t_idx[0]]'''
new = '''    face_v = faces[fi]                      # uv faces and mesh faces share row order'''
assert old in t
t = t.replace(old, new)

old = '''texture = np.clip(tex * 255, 0, 255).astype(np.uint8)'''
new = '''texture = np.clip(tex, 0, 255).astype(np.uint8)'''
assert old in t
t = t.replace(old, new)

old = '''texture = texture[::-1]'''
new = '''texture = texture[::-1]  # image row order for UV (v up)'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("texture bake fixes applied")
