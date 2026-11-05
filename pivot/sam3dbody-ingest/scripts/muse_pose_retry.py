"""Retry the 1x3 full-body pose strip for Emiru with friendlier wording."""
import importlib.util
import json
import shutil
import time
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
STEM = "Emiru"
JOB = BASE / "outputs" / STEM
PHOTO = next((BASE / "input-images").rglob(f"{STEM}.*"))
OUT = JOB / "threedb"
GUIDE = OUT / "guide_body_strip_1x3.png"

spec = importlib.util.spec_from_file_location("meta_image", REPO / "pipeline" / "avatar" / "meta_image.py")
mi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mi)

PROMPT = (
    "Image 1 is a flat-shaded row of three full-length views of a person standing relaxed with the arms "
    "slightly away from the sides: front view, left side view, back view. Image 2 is the reference photo of "
    "the person. Make image 1 photorealistic as the same person as image 2: same face and identity, same "
    "hairstyle and hair colour, wearing the same clothing and shoes as in image 2. Keep the row layout, the "
    "panel order (front, side, back), the standing pose, the full-length framing (head to feet in every "
    "panel) and the scale exactly as in image 1. Flat, even, soft studio lighting, plain light-grey "
    "background, sharp focus, photographic, no text, no labels, no numbers, no borders."
)
mi.PROMPTS["head-render"] = PROMPT
mi.REFERENCE_LABELS[0] = "Image 1 (flat-shaded guide; preserve its row layout, panel order, pose and framing):"
mi.REFERENCE_LABELS[1] = "Image 2 (the real person: identity, hair and clothing reference):"
t0 = time.time()
rc = mi.main(["--mode", "head-render", "--reference", str(GUIDE), "--reference2", str(PHOTO),
              "--out", str(OUT / "raw_body_strip_1x3_retry"), "--size", "1536x1024", "--timeout", "300"])
sheet = OUT / "raw_body_strip_1x3_retry" / "sheet.png"
if rc == 0 and sheet.exists():
    shutil.copy2(sheet, OUT / "muse_body_strip_1x3.png")
    print("saved", OUT / "muse_body_strip_1x3.png", "in", round(time.time() - t0, 1), "s")
else:
    print("failed rc", rc)
