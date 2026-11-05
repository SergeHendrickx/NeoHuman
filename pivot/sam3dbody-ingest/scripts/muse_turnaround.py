"""Send the A-pose 6-view render + the reference photo to Meta Image -> photoreal turnaround."""
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import cv2

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"

STEM = sys.argv[1] if len(sys.argv) > 1 else "Emiru"
JOB = BASE / "outputs" / STEM
RENDER = JOB / "threedb" / "turnaround_apose.jpg"
PHOTO = next((BASE / "input-images").rglob(f"{STEM}.*"))
OUT = JOB / "threedb"

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

t0 = time.time()
rc = mi.main(["--mode", "head-render", "--reference", str(RENDER), "--reference2", str(PHOTO),
              "--out", str(OUT / "muse_turnaround_apose_raw"), "--size", "1536x1024", "--timeout", "300"])
sheet = OUT / "muse_turnaround_apose_raw" / "sheet.png"
if rc == 0 and sheet.exists():
    dst = OUT / "muse_turnaround_apose.png"
    shutil.copy2(sheet, dst)
    print("saved", dst, "in", round(time.time() - t0, 1), "s")
    (OUT / "muse_turnaround_apose.json").write_text(json.dumps({
        "image": STEM, "render": str(RENDER), "photo": str(PHOTO), "seconds": round(time.time() - t0, 1),
        "size": "1536x1024", "prompt": PROMPT}, indent=2), encoding="utf-8")
else:
    raise SystemExit(f"Muse failed rc={rc}")
