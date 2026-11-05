from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\vggt_proportions.py")
t = p.read_text(encoding="utf-8")
start = t.index("def ratios(P):")
end = t.index("def main() -> int:")
new = '''def ratios(P):
    """P: 6x3 points in the order eyes L/R, ears L/R, chin, nose."""
    eye = 0.5 * (P[0] + P[1]); chin = P[4]; nose = P[5]
    face_h = np.linalg.norm(chin - eye)
    ear_w = np.linalg.norm(P[2] - P[3])
    nose_depth = abs(float(nose[2] - eye[2]))
    return {"ear_w/face_h": round(float(ear_w / face_h), 3),
            "nose_dz/face_h": round(float(nose_depth / face_h), 3)}


'''
t = t[:start] + new + t[end:]
p.write_text(t, encoding="utf-8")
print("ratios replaced")
