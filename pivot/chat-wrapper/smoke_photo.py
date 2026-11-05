import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cv2

import analyze

IMG = HERE.parent / "sam3dbody-ingest" / "input-images" / "Emiru.webp"
OUT = HERE / "runs" / "smoke-emiru"


class H:
    def stage_start(self, s):
        print("START", s, flush=True)

    def stage_end(self, s, d=None):
        print("END", s, sorted((d or {}).keys()), flush=True)

    def log(self, s, line, channel="stdout"):
        print(f"[{s}]", str(line)[:160], flush=True)

    def warn(self, s, message):
        print(f"WARN[{s}]", str(message)[:200], flush=True)

    def message(self, s, role, text, source="live"):
        print(f"MSG[{s}/{role}]", str(text)[:160], flush=True)

    def progress(self, s, **d):
        print(f"PROG[{s}]", d, flush=True)

    def artifact(self, s, aid, path, **k):
        print(f"ART[{s}]", aid, str(path), flush=True)

    def cancelled(self):
        return False

    def choice(self, stage, prompt, options, default=None):
        pick = default or (options[0]["id"] if options else "continue")
        print(f"GATE[{stage}] auto -> {pick}", flush=True)
        return pick


def main():
    bgr = cv2.imread(str(IMG), cv2.IMREAD_COLOR)
    print("img", None if bgr is None else bgr.shape, flush=True)
    stats = analyze.run_photo(bgr, OUT, H(), {"auto_continue": True})
    print("STATS", stats, flush=True)


if __name__ == "__main__":
    main()
