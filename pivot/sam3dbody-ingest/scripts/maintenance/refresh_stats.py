"""Refresh stats.json file inventories after the layout migration."""
import json
from pathlib import Path

OUT = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\outputs")
n = 0
for job in sorted(OUT.iterdir()):
    s = job / "stats.json"
    if not job.is_dir() or job.name.startswith("_") or not s.exists():
        continue
    st = json.loads(s.read_text(encoding="utf-8"))
    st["files"] = sorted(q.relative_to(job).as_posix() for q in job.rglob("*") if q.is_file())
    st["raw_files"] = {
        "sapiens": sorted(q.name for q in (job / "sapiens" / "raw").iterdir()) if (job / "sapiens" / "raw").exists() else [],
        "threedb": sorted(q.name for q in (job / "threedb" / "raw").iterdir()) if (job / "threedb" / "raw").exists() else [],
    }
    s.write_text(json.dumps(st, indent=2), encoding="utf-8")
    n += 1
print("refreshed", n, "jobs")
