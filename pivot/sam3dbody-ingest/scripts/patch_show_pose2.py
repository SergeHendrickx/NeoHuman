from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\show_pose_strip_test.py")
t = p.read_text(encoding="utf-8")
old = """    renders[n] = RO.raster_ortho(v, faces, clay, scale, x0, y0)
    cv2.imwrite(str(OUT / f"aligned_{n}.png"), renders[n])"""
new = """    rgb = RO.raster_ortho(v, faces, clay, scale, x0, y0)
    renders[n] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT / f"aligned_{n}.png"), renders[n])"""
assert old in t, "anchor not found"
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("channel order fixed")
