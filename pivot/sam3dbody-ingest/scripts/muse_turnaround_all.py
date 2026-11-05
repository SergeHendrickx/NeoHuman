"""Batch: A-pose 6-view render + reference photo -> Muse photoreal turnaround, for every job."""
import importlib.util
import json
import shutil
import time
from pathlib import Path

import cv2

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
OUT = BASE / "outputs"

spec = importlib.util.spec_from_file_location("meta_image", REPO / "pipeline" / "avatar" / "meta_image.py")
mi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mi)

PROMPT = (
    "Image 1 is a flat-shaded SIX-PANEL FULL-BODY TURNAROUND render of a person in A-pose, in this order: "
    "-90 (left profile), -45, 0 (front), +45, +90 (right profile), 180 (back). Image 2 is the real person. "
    "Make image 1 photorealistic as the SAME PERSON as image 2: same identity and face, same body shape and "
    "proportions, same skin tone, same hairstyle and hair colour, same clothing and colours. Preserve image 1 "
    "EXACTLY: the same six panels in the same order and boundaries, the same A-pose (arms slightly away from "
    "the body, palms relaxed), the same full-body framing with head and feet inside every panel, the same "
    "scale and camera angles. FLAT, EVEN, SHADOWLESS LIGHTING: uniform diffuse frontal light, no hard shadows, "
    "no light gradient; plain light-grey seamless background, sharp focus, photographic, no text, no labels, "
    "no borders, no watermark."
)
mi.PROMPTS["head-render"] = PROMPT
mi.REFERENCE_LABELS[0] = ("Image 1 (flat-shaded 6-panel A-pose full-body turnaround: -90/-45/0/+45/+90/180; "
                          "preserve its layout, pose and framing):")
mi.REFERENCE_LABELS[1] = "Image 2 (the real person: identity, hair, clothing and body shape reference):"

rows = []
for d in sorted(OUT.iterdir()):
    if not d.is_dir() or d.name.startswith("_"):
        continue
    render = d / "threedb" / "turnaround_apose.jpg"
    if not render.exists():
        continue
    stem = d.name
    photo = next((BASE / "input-images").rglob(f"{stem}.*"), None)
    if photo is None:
        rows.append({"image": stem, "status": "no_photo"})
        continue
    if (d / "threedb" / "muse_turnaround_apose.png").exists():
        rows.append({"image": stem, "status": "cached"})
        continue
    t0 = time.time()
    try:
        rc = mi.main(["--mode", "head-render", "--reference", str(render), "--reference2", str(photo),
                      "--out", str(d / "threedb" / "muse_turnaround_apose_raw"), "--size", "1536x1024",
                      "--timeout", "300"])
        sheet = d / "threedb" / "muse_turnaround_apose_raw" / "sheet.png"
        if rc == 0 and sheet.exists():
            shutil.copy2(sheet, d / "threedb" / "muse_turnaround_apose.png")
            rows.append({"image": stem, "status": "ok", "seconds": round(time.time() - t0, 1)})
        else:
            rows.append({"image": stem, "status": f"rc={rc}", "seconds": round(time.time() - t0, 1)})
    except BaseException as exc:  # noqa: BLE001
        rows.append({"image": stem, "status": f"error:{type(exc).__name__}", "seconds": round(time.time() - t0, 1)})
    print(json.dumps(rows[-1]), flush=True)
    (OUT / "_global" / "muse_turnarounds_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

ok = sum(1 for r in rows if r.get("status") in ("ok", "cached"))
print(f"turnarounds: {ok}/{len(rows)}")

# contact sheet of results
files = [d / "threedb" / "muse_turnaround_apose.png" for d in sorted(OUT.iterdir())
         if (d / "threedb" / "muse_turnaround_apose.png").exists()]
if files:
    thumbs = []
    for f in files:
        im = cv2.imread(str(f))
        h = 240
        w = int(im.shape[1] * h / im.shape[0])
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        lab = np.full((22, w, 3), 24, np.uint8)
        cv2.putText(lab, f.parent.parent.name[:34], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
        thumbs.append(np.vstack([lab, im]))
    W = max(t.shape[1] for t in thumbs)
    thumbs = [np.hstack([t, np.full((t.shape[0], W - t.shape[1], 3), 24, np.uint8)]) for t in thumbs]
    cols = 3
    rows_n = (len(thumbs) + cols - 1) // cols
    H = max(t.shape[0] for t in thumbs)
    sheet = np.full((rows_n * H, cols * W, 3), 24, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * H:r * H + t.shape[0], c * W:c * W + t.shape[1]] = t
    cv2.imwrite(str(OUT / "_global" / "muse_turnarounds_contact.jpg"), sheet)
    print("wrote", OUT / "_global" / "muse_turnarounds_contact.jpg")
