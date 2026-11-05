"""Local contact sheet of the Muse A-pose turnarounds that already exist (no API calls)."""
import cv2
import numpy as np
from pathlib import Path

OUT = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\outputs")
files = sorted(OUT.glob("*/threedb/muse_turnaround_apose.png"))
thumbs = []
for f in files:
    im = cv2.imread(str(f))
    h = 260
    w = int(im.shape[1] * h / im.shape[0])
    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    lab = np.full((24, w, 3), 24, np.uint8)
    name = f.parent.parent.name
    cv2.putText(lab, name[:40], (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)
    thumbs.append(np.vstack([lab, im]))
W = max(t.shape[1] for t in thumbs)
thumbs = [np.hstack([t, np.full((t.shape[0], W - t.shape[1], 3), 24, np.uint8)]) for t in thumbs]
cols = 3
H = max(t.shape[0] for t in thumbs)
rows = (len(thumbs) + cols - 1) // cols
sheet = np.full((rows * H, cols * W, 3), 24, np.uint8)
for i, t in enumerate(thumbs):
    r, c = divmod(i, cols)
    sheet[r * H:r * H + t.shape[0], c * W:c * W + t.shape[1]] = t
cv2.imwrite(str(OUT / "_global" / "muse_turnarounds_existing.jpg"), sheet)
print("files:", len(files), [f.parent.parent.name for f in files])
print("wrote", OUT / "_global" / "muse_turnarounds_existing.jpg")
