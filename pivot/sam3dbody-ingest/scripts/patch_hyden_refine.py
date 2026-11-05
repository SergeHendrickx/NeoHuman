from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\hyden_refine.py")
t = p.read_text(encoding="utf-8")
old = '''mw = MAP[70:272].abs().sum(0).detach().cpu().numpy()
face_v = torch.tensor(np.where(mw > 0.05)[0], device=DEV)'''
new = '''mw = MAP[70:272].abs().sum(0).detach().cpu().numpy()
fv_np = np.where(mw > 0.05)[0]
fv_np = fv_np[fv_np < verts_b.shape[0]]          # drop the 127 joint entries of the mapping
face_v = torch.tensor(fv_np, device=DEV)'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("face vertex indices capped to mesh vertices")
