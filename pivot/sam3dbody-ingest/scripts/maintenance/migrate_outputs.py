"""Migrate outputs/ to the per-job / per-step layout.

Before: outputs/<stem>/01..09, raw/, identity*.json, stats.json, log.txt (+ ad-hoc subfolders)
        outputs/<experiment>/ (joint-test, meta-image-test, ...)
After:  outputs/<stem>/input|sapiens|threedb|texture|variants|profile/{...}, stats.json, log.txt
        outputs/_global/{joint-test, meta-image-test, ...}
"""
import json
import shutil
from pathlib import Path

OUT = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\outputs")
GLOBAL = OUT / "_global"
GLOBAL.mkdir(exist_ok=True)

JOBS = [d for d in OUT.iterdir() if d.is_dir() and (d / "stats.json").exists()]
EXPERIMENTS = [d for d in OUT.iterdir() if d.is_dir() and d not in JOBS and d.name != "_global"]

# step folder -> filename prefixes it owns (inside the old flat job folder)
SAPIENS = ("02_", "03_", "04_", "05_", "10_", "11_")
THREEDB = ("06_", "07_", "08_", "09_", "mesh_", "identity_")
INPUT = ("01_",)
# raw/ split by owner
RAW_SAPIENS = ("sapiens_pose", "person_box", "sapiens_seg", "skin_mask", "seg_class_counts",
               "sapiens_normals", "sapiens_depth", "sapiens_pointmap_1b", "sapiens_matting_1b")
RAW_THREEDB = ("3db_",)

moved = []
for job in JOBS:
    for sub in ("input", "sapiens", "threedb", "texture", "profile", "variants",
                "sapiens/raw", "threedb/raw"):
        (job / sub).mkdir(parents=True, exist_ok=True)
    for f in list(job.iterdir()):
        if f.is_dir():
            continue
        name = f.name
        if name.startswith(INPUT):
            dst = job / "input" / name
        elif name.startswith(SAPIENS):
            dst = job / "sapiens" / name
        elif name.startswith(THREEDB):
            dst = job / "threedb" / name
        else:  # stats.json / log.txt stay at job root
            continue
        if dst.exists():
            dst.unlink()
        f.rename(dst)
    # old flat raw/
    oldraw = job / "raw"
    if oldraw.exists():
        for f in list(oldraw.iterdir()):
            if f.name.startswith(RAW_SAPIENS):
                dst = job / "sapiens" / "raw" / f.name
            elif f.name.startswith(RAW_THREEDB):
                dst = job / "threedb" / "raw" / f.name
            else:
                dst = job / "threedb" / "raw" / f.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(f), str(dst))
        oldraw.rmdir()
    # ad-hoc subfolders created later in the session
    renames = {"profile-test": "profile", "texture-project": "texture"}
    for old, new in renames.items():
        src = job / old
        if src.exists():
            for f in list(src.iterdir()):
                dst = job / new / f.name
                if dst.exists():
                    if dst.is_dir():
                        shutil.rmtree(dst)
                    else:
                        dst.unlink()
                shutil.move(str(f), str(dst))
            src.rmdir()
    moved.append(job.name)

for exp in EXPERIMENTS:
    dst = GLOBAL / exp.name
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    shutil.move(str(exp), str(dst))

for f in list(OUT.iterdir()):
    if f.is_file() and f.name in ("run_all.log", "make_readme.py"):
        shutil.move(str(f), str(GLOBAL / f.name))

print(f"migrated {len(moved)} jobs -> step folders; {len(EXPERIMENTS)} experiment folders -> _global/")
print("jobs:", ", ".join(sorted(moved)))
