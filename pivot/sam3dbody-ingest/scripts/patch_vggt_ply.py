from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\vggt_fullbody_grid.py")
t = p.read_text(encoding="utf-8")
old = '''cv2.imwrite(str(OUT / "vggt_fullbody_grid_rgb.png"), canvas)
(OUT / "vggt_fullbody_grid_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("wrote", OUT / "vggt_fullbody_grid_rgb.png")'''
new = '''cv2.imwrite(str(OUT / "vggt_fullbody_grid_rgb.png"), canvas)
(OUT / "vggt_fullbody_grid_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

# ---- PLY exports: fused multi-view point cloud (coloured by view) + per-view clouds
import trimesh
rgb = np.clip(cc * 255, 0, 255).astype(np.uint8)
trimesh.PointCloud(pts, colors=rgb).export(OUT / "vggt_fullbody_cloud.ply")
for n in names:
    c = np.array(cols[n])[None, :]
    trimesh.PointCloud(worlds[n], colors=np.clip(np.tile(c, (len(worlds[n]), 1)) * 255, 0, 255).astype(np.uint8)) \\
        .export(OUT / f"vggt_cloud_{n.replace('+', 'p').replace('-', 'm')}.ply")
print("wrote", OUT / "vggt_fullbody_cloud.ply")'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("ply export added to vggt grid script")
