import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chat_server as S


class H:
    def stage_start(self, s): print("START", s, flush=True)
    def stage_end(self, s, d=None): print("END", s, flush=True)
    def log(self, s, line, channel="stdout"): print(f"[{s}]", str(line)[:140], flush=True)
    def warn(self, s, m): print(f"WARN[{s}]", str(m)[:200], flush=True)
    def message(self, s, role, text, source="live"): print(f"MSG", str(text)[:140], flush=True)
    def progress(self, s, **d): pass
    def artifact(self, s, aid, path, **k): print(f"ART[{s}]", aid, str(path), flush=True)
    def cancelled(self): return False
    def choice(self, s, p, o, default=None):
        print(f"GATE[{s}]", p[:100], "->", [x["id"] for x in o], flush=True)
        return default


class FakeRun:
    seq = 0
    def emit(self, t, stage=None, data=None):
        self.seq += 1
        if t in ("api.call", "api.return"):
            d = dict(data or {})
            if t == "api.call" and isinstance(d.get("post"), dict):
                d = dict(d); d["post_keys"] = sorted(d["post"].keys()); d.pop("post", None)
            if t == "api.return" and isinstance(d.get("response"), dict):
                r = d["response"]; d = dict(d)
                d["response_keys"] = sorted(r.keys())
                d["response_len"] = len(json.dumps(r))
                d.pop("response", None)
            print(f"EV[{t}/{stage}]", json.dumps(d)[:600], flush=True)
        return {}


img = HERE.parent / "sam3dbody-ingest" / "input-images" / "Emiru.webp"
raw = img.read_bytes()
scan = S.vision_scan(raw, "image/webp", FakeRun(), H())
print("SCAN-PEOPLE", len(scan.get("people-detected", [])), flush=True)
print("SCAN-DESC", scan.get("photo-description"), flush=True)
for p in scan.get("people-detected", []):
    print("PERSON", json.dumps(p), flush=True)
