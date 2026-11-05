#!/usr/bin/env python3
"""chat-wrapper/chat_server.py — photo-analysis disguised as chat.

One process, one port. Serves chat.html, runs sam3dbody-ingest stages
in-process (one model resident at a time), streams SSE, persists events.jsonl.

  .\\.venv\\Scripts\\python.exe pivot\\chat-wrapper\\chat_server.py [--port 8150]
  open http://127.0.0.1:8150/

Chat flow: POST /api/chat {text, image_b64?} → assistant reply (Muse Spark
tool call when key present, else scripted) → if photo: creates run, streams
stages as tool cards (see chat.html). Analysis itself is analyze.run_photo.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import queue
import secrets
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Repo root by workspace marker (.cache + pivot), so this file works both at
# pivot/chat-wrapper/ and at pivot/chat-wrapper/flow-viewer/.
REPO = next((p for p in (HERE, *HERE.parents) if (p / ".cache").is_dir() and (p / "pivot").is_dir()), HERE.parents[1])
INGEST = REPO / "pivot" / "sam3dbody-ingest"
# The flow-viewer copy is a staging spare: shared state (runs/chats/usage),
# vendored code and the viewer statics always resolve to pivot/chat-wrapper/,
# whichever copy is running — one state dir, never a split-brain pair.
DATA = HERE if (HERE / "_vendor").is_dir() else HERE.parent
FLOW = HERE / "flow-viewer" if (HERE / "flow-viewer").is_dir() else HERE
FLOW_GRAPHS = FLOW / "data" / "graphs"
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(INGEST))  # run_all / joint_solve helpers (read-only reuse)
sys.path.insert(0, str(DATA / "_vendor"))  # meta_sam_parser (vendored, zero-dep)

RUNS = DATA / "runs"
RUNS.mkdir(exist_ok=True)
CHATS = DATA / "chats"
CHATS.mkdir(exist_ok=True)
CHATS_LOCK = threading.Lock()

STAGE_META = [
    {"id": "vision", "label": "Muse Spark sees the photo"},
    {"id": "ingest", "label": "read photo"},
    {"id": "pose", "label": "Sapiens2 308 keypoints"},
    {"id": "maps", "label": "seg / normal / depth"},
    {"id": "body", "label": "SAM 3D Body + joint solve"},
    {"id": "report", "label": "report"},
]
STAGE_IDS = [s["id"] for s in STAGE_META]

RUN_LOCK = threading.Lock()
ACTIVE_RUN: str | None = None
_STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")


def _split_api_path(path: str):
    # One splitter for API routes: query strings never leak into the action.
    return path.split("?", 1)[0].rstrip("/").split("/")
RUNS_MEM: dict[str, "Run"] = {}
RUNS_LOCK = threading.Lock()
# Debug GPU actions (sapiens-*, body-fit) serialize here: one model resident
# per call already, and one call at a time — a second caller gets 409, not OOM.
def _gpu_log(msg: str):
    import datetime as _dt
    import threading as _th
    try:
        import torch
        mem = f" cuda={torch.cuda.memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else ""
    except Exception:
        mem = ""
    print("[{t}][gpu][t{th}] {m}{mem}".format(
        t=_dt.datetime.now().strftime("%H:%M:%S"), th=_th.get_native_id(), m=msg, mem=mem), flush=True)


class _GpuLock:
    """threading.Lock that logs hold windows — the server log proves the
    one-GPU discipline (timestamps order every acquire/release/refusal)."""
    def __init__(self):
        self._lock = threading.Lock()

    def acquire(self, blocking=True, timeout=-1):
        ok = self._lock.acquire(blocking, timeout)
        _gpu_log("acquired" if ok else "busy-refused")
        return ok

    def release(self):
        _gpu_log("released")
        self._lock.release()


GPU_LOCK = _GpuLock()

META_KEY = Path(os.environ.get("META_API_KEY_FILE", str(REPO / ".cache" / "meta-api-key.txt")))
USAGE = DATA / "usage.json"
TEXT_MODEL = "muse-spark-1.3-contributor"
MUSE_MODEL = "muse-image-1.0"
MAX_TEXT_CALLS = int(os.environ.get("CHAT_MUSE_MAX_TEXT_CALLS", "200"))
API_BASE = "https://api.meta.ai/v1"  # Responses API (see pipeline/stylist/stylist.py)

# Structured photo scan: first message uploads the photo to Muse Spark and
# asks for this object. Boxes are 0-1000 relative coords, mapped to pixels.
VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "photo-description": {"type": "string", "description": "One-sentence description of the photo"},
        "people-detected": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox": {
                        "type": "object",
                        "properties": {
                            "x_min": {"type": "number"}, "y_min": {"type": "number"},
                            "x_max": {"type": "number"}, "y_max": {"type": "number"},
                        },
                        "required": ["x_min", "y_min", "x_max", "y_max"],
                        "additionalProperties": False,
                    },
                    "description": {"type": "string", "description": "One-line appearance note: clothing, pose, position"},
                },
                "required": ["bbox", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["photo-description", "people-detected"],
    "additionalProperties": False,
}
VISION_INSTRUCTIONS = (
    "You are the seeing front-end of a character-creation avatar builder. "
    "Describe only what is visible. Never name real people, never infer identity, "
    "age, sex, health or personality — one neutral appearance note per person only."
)
VISION_PROMPT = (
    "Describe this photo in one sentence. Detect every person: one tight box per "
    "person as 0-1000 relative coordinates (x_min,y_min top-left, x_max,y_max "
    "bottom-right) plus a one-line appearance note (clothing, pose, where in frame)."
)

ANALYZE_TOOL = {
    "name": "analyze_photo",
    "description": "Run Sapiens2 + SAM 3D Body + joint solve on one uploaded photo. Use for any claim needing measurement.",
    "parameters": {
        "type": "object", "required": ["image_id"], "additionalProperties": False,
        "properties": {
            "image_id": {"type": "string", "description": "Server-side upload handle."},
            "bbox_mode": {"type": "string", "enum": ["auto", "manual"], "description": "auto derives box from Sapiens2 keypoints."},
            "joint": {"type": "boolean", "default": True},
            "fit_expression": {"type": "boolean", "default": True},
            "full_maps": {"type": "boolean", "default": False, "description": "Also run 1b pointmap + 1b matting (10/11). Slower."},
        },
    },
}


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


class StageError(Exception):
    def __init__(self, message, remedy=""):
        super().__init__(message)
        self.remedy = remedy


class Hooks:
    def __init__(self, run: "Run"):
        self.run = run

    def cancelled(self):
        return self.run.cancel_event.is_set()

    def choice(self, stage, prompt, options, default=None):
        """Block until the user clicks an option card (or auto-continue in headless)."""
        return self.run.open_gate(stage, prompt, options, default)

    def stage_start(self, stage): self.run.emit("stage.start", stage=stage)
    def stage_end(self, stage, data=None): self.run.emit("stage.end", stage=stage, data=data or {})
    def log(self, stage, line, channel="stdout"): self.run.log(stage, line, channel)
    def warn(self, stage, message): self.run.warn(stage, message)
    def message(self, stage, role, text, source="live"): self.run.message(stage, role, text, source)
    def progress(self, stage, **data): self.run.progress(stage, **data)

    def artifact(self, stage, aid, path: Path, kind="image", label="", provenance="live", note=""):
        self.run.add_artifact(stage, aid, path, kind, label, provenance, note)


class Run:
    def __init__(self, run_id: str, image_b64: str | None, filename: str, opts: dict):
        self.id = run_id
        self.dir = RUNS / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        self.status = "created"
        self.created_at = now_iso()
        self.events: list[dict] = []
        self.seq = 0
        self.lock = threading.RLock()
        self.subscribers: set[queue.Queue] = set()
        self.cancel_event = threading.Event()
        self.artifacts: dict[str, dict] = {}
        self.opts = opts
        # standard-flow gate: open_gate() blocks execute() until the user picks an option card
        self.gate_event = threading.Event()
        self.gate_choice: str | None = None
        self.gate_open: dict | None = None
        self._fh = (self.dir / "events.jsonl").open("a", encoding="utf-8")
        self.stats: dict | None = None
        if image_b64:
            raw = base64.b64decode(image_b64)
            ext = ".png" if filename.lower().endswith(".png") else ".jpg"
            (self.dir / f"source{ext}").write_bytes(raw)
        write_json_atomic(self.dir / "run.json", {"id": run_id, "status": self.status, "created_at": self.created_at})

    # -- events (same-lock replay+live pattern as product/server.py) --
    def emit(self, etype, stage=None, data=None):
        with self.lock:
            self.seq += 1
            ev = {"seq": self.seq, "t": now_iso(), "type": etype, "stage": stage, "data": data or {}}
            self.events.append(ev)
            try:
                self._fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                self._fh.flush()
            except Exception:
                pass
            for q in list(self.subscribers):
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass
        return ev

    def subscribe(self):
        with self.lock:
            replay = list(self.events)
            q: queue.Queue = queue.Queue(maxsize=4096)
            self.subscribers.add(q)
        return replay, q

    def _new_q(self):
        q: queue.Queue = queue.Queue(maxsize=4096)
        self.subscribers.add(q)
        return list(self.events), q

    def unsubscribe(self, q):
        with self.lock:
            self.subscribers.discard(q)

    def log(self, stage, line, channel="stdout"):
        self.emit("stage.log", stage=stage, data={"line": line[:500], "channel": channel})

    def warn(self, stage, message):
        self.emit("warn", stage=stage, data={"message": message})

    def message(self, stage, role, text, source="live"):
        self.emit("message", stage=stage, data={"role": role, "text": text, "source": source})

    def progress(self, stage, **data):
        self.emit("stage.progress", stage=stage, data=data)

    def add_artifact(self, stage, aid, path: Path, kind, label, provenance="live", note=""):
        try:
            rel = path.relative_to(self.dir).as_posix()
        except ValueError:
            rel = path.name
        url = f"/api/runs/{self.id}/artifact/{aid}"
        info = {"id": aid, "stage": stage, "kind": kind, "label": label or aid,
                "url": url, "rel": rel, "provenance": provenance, "note": note,
                "bytes": path.stat().st_size if path.exists() else 0}
        with self.lock:
            self.artifacts[aid] = info
        self.emit("artifact", stage=stage, data=info)
        return info

    # -- standard-flow gate: wait for user input at every step --
    def open_gate(self, stage, prompt, options, default=None):
        """Emit gate.open with clickable options and block until choice/cancel.

        options: [{id, label, hint?, primary?}]. Headless callers set
        opts.auto_continue=True to take `default` (or first option) without blocking.
        """
        if self.opts.get("auto_continue"):
            pick = default or (options[0]["id"] if options else "continue")
            self.emit("gate.answer", stage=stage, data={"choice": pick, "auto": True})
            return pick
        self.gate_choice = None
        self.gate_event.clear()
        self.gate_open = {"prompt": prompt, "options": options, "default": default}
        self.emit("gate.open", stage=stage, data={"prompt": prompt, "options": options, "default": default})
        while True:
            fired = self.gate_event.wait(timeout=1.0)
            if fired:
                break
            if self.cancel_event.is_set():
                raise StageError("cancelled by user", remedy="resend the photo to retry")
        choice = self.gate_choice or default or (options[0]["id"] if options else "continue")
        self.gate_open = None
        if choice == "__superseded__":
            raise StageError("cancelled by user (superseded by a newer photo)",
                             remedy="this run was retired; follow the newer run")
        self.emit("gate.answer", stage=stage, data={"choice": choice})
        return choice

    def answer_gate(self, choice):
        self.gate_choice = choice
        self.gate_event.set()

    def execute(self):
        global ACTIVE_RUN
        try:
            self.status = "running"
            self.emit("run.start", data={"stages": STAGE_META, "opts": self.opts})
            # assistant opening line (scripted or Muse — see chat_reply)
            self.message("ingest", "assistant", self.opts.get("assistant_preamble",
                "Reconstructing identity + expression from 1 photo — watching the same artifacts the pipeline writes."))
            import cv2
            import numpy as np

            src = next((p for p in self.dir.glob("source.*")), None)
            if src is None:
                raise StageError("no source image", remedy="re-upload the photo")
            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if bgr is None:
                raise StageError("could not decode photo", remedy="try JPG/PNG under 12MB")
            H, W = bgr.shape[:2]
            hooks = Hooks(self)

            # ---- vision: first message uploads the photo to Muse Spark ----
            self.emit("stage.start", stage="vision")
            scan, people_px = None, []
            try:
                raw_bytes = src.read_bytes()
                mime = "image/png" if src.suffix.lower() == ".png" else "image/jpeg"
                scan = vision_scan(raw_bytes, mime, self, hooks)
                (self.dir / "vision_scan.json").write_text(
                    json.dumps(scan, indent=2, ensure_ascii=False), encoding="utf-8")
                self.add_artifact("vision", "scan", self.dir / "vision_scan.json",
                                  kind="json", label="vision_scan.json", provenance="live",
                                  note="muse-spark-1.3-contributor structured photo object")
                for p in scan.get("people-detected", []):
                    bb = p.get("bbox", {})
                    box = [bb.get("x_min", 0) / 1000 * W, bb.get("y_min", 0) / 1000 * H,
                             bb.get("x_max", 1000) / 1000 * W, bb.get("y_max", 1000) / 1000 * H]
                    people_px.append((box, str(p.get("description", ""))[:120]))
                self.message("vision", "assistant",
                    f"Photo: {scan.get('photo-description')} — {len(people_px)} person(s) detected.")
            except StageError as e:
                self.warn("vision", f"vision scan unavailable ({e}); falling back to Sapiens auto box")
                self.emit("stage.end", stage="vision",
                            data={"vision": "unavailable", "fallback": "sapiens-auto"})
                scan = None
            if scan is not None:
                import cv2 as _cv2

                if len(people_px) == 0:
                    pick = hooks.choice(
                        "vision", "Muse Spark found no people. Continue with Sapiens auto-detect anyway?",
                        [{"id": "auto", "label": "Continue · Sapiens auto-detect (Recommended)", "primary": True},
                         {"id": "cancel", "label": "Cancel run"}], default="auto")
                    if pick == "cancel":
                        raise StageError("cancelled by user", remedy="resend the photo to retry")
                else:
                    _cv2.imwrite(str(self.dir / "vision_boxes.jpg"),
                                  draw_person_boxes(bgr, people_px))
                    self.add_artifact("vision", "boxes", self.dir / "vision_boxes.jpg",
                                      kind="image", label="vision boxes (Muse Spark)",
                                      provenance="live",
                                      note="numbered people boxes — pick one below")
                    if len(people_px) == 1:
                        box, desc = people_px[0]
                        pick = hooks.choice(
                            "vision", f"1 person: {desc}. Use this box for the fit?",
                            [{"id": "use", "label": f"Continue · use this box (Recommended)", "primary": True},
                             {"id": "auto", "label": "Ignore box · Sapiens auto instead"},
                             {"id": "cancel", "label": "Cancel run"}], default="use")
                        if pick == "cancel":
                            raise StageError("cancelled by user", remedy="resend the photo to retry")
                        if pick == "use":
                            self.opts["bbox"] = [round(float(v), 1) for v in box]
                    else:
                        opts = [{"id": f"person:{i}",
                                 "label": f"Person {i + 1} — {desc}",
                                 "primary": (i == 0)} for i, (box, desc) in enumerate(people_px)]
                        opts.append({"id": "cancel", "label": "Cancel run"})
                        pick = hooks.choice(
                            "vision", f"{len(people_px)} people detected — pick one (boxes numbered in the image above):",
                            opts, default="person:0")
                        if pick == "cancel" or pick not in [o["id"] for o in opts]:
                            raise StageError("cancelled by user", remedy="resend the photo to retry")
                        box, desc = people_px[int(pick.split(":")[1])]
                        self.opts["bbox"] = [round(float(v), 1) for v in box]
                        self.message("vision", "assistant", f"Picked {pick} ({desc}) — fitting that person.")
                self.emit("stage.end", stage="vision",
                            data={"people": len(people_px), "bbox": self.opts.get("bbox"),
                                  "photo": (scan.get("photo-description") or "")[:160]})
            import analyze as A

            # allow analyze.StageError to propagate as loud failure
            stats = A.run_photo(bgr, self.dir, hooks, self.opts)
            self.stats = stats
            self.status = "done"
            face = "n/a"
            self.message("report", "assistant",
                f"Done in {stats.get('total_s')}s — |β| {stats.get('identity_l2')} via {stats.get('refine_used')} "
                f"(focal {stats.get('focal_px')}px). Renders + identity_joint.json below. "
                f"Character-creation numbers only — no inference about the real person.")
            self.emit("run.end", data={"status": "done", "stats": stats})
        except Exception as e:
            msg = str(e)[:300]
            if "cancel" in msg.lower() or self.cancel_event.is_set():
                self.status = "cancelled"
                self.emit("run.end", data={"status": "cancelled", "error": msg})
            else:
                remedy = getattr(e, "remedy", "")
                self.status = "failed"
                self.emit("stage.log", data={"line": traceback.format_exc(limit=3)[-800:], "channel": "stderr"})
                self.emit("run.end", data={"status": "failed", "error": msg, "remedy": remedy})
        finally:
            try:
                self._fh.close()
            except Exception:
                pass
            write_json_atomic(self.dir / "run.json", {"id": self.id, "status": self.status,
                                                      "created_at": self.created_at, "stats": self.stats})
            if ACTIVE_RUN == self.id:
                ACTIVE_RUN = None


# -- Muse Spark (tool-calling when key present, else scripted) --
def muse_available():
    if os.environ.get("CHAT_MUSE", "") in ("0", "off", "false", "no", "none"):
        return False
    return META_KEY.exists() and META_KEY.read_text(encoding="utf-8").strip() != ""


def usage_get():
    try:
        return json.loads(USAGE.read_text(encoding="utf-8"))
    except Exception:
        return {"text_calls": 0, "est_usd": 0.0}


def usage_add(resp_id, seconds, usage):
    u = usage_get()
    u["text_calls"] = u.get("text_calls", 0) + 1
    inp = (usage or {}).get("input_tokens", 0)
    out = (usage or {}).get("output_tokens", 0)
    u["est_usd"] = round(u.get("est_usd", 0.0) + inp / 1e6 * 0.10 + out / 1e6 * 0.20, 4)
    calls = u.setdefault("calls", [])
    calls.append({"model": TEXT_MODEL, "response_id": resp_id, "seconds": round(seconds, 1), "created_at": now_iso()})
    USAGE.write_text(json.dumps(u, indent=2))


# -- chatlog store (every user/assistant turn persisted, listable anytime) --
def chat_path(chat_id: str) -> Path:
    safe = "".join(c for c in chat_id if c.isalnum() or c in ("-", "_"))[:64] or "chat"
    return CHATS / safe

def chat_append(chat_id: str, entry: dict) -> dict:
    """Append one message/tool/run-link entry to chats/<id>/chat.jsonl. Creates chat on demand."""
    d = chat_path(chat_id)
    d.mkdir(parents=True, exist_ok=True)
    entry = {"seq": 0, "t": now_iso(), **entry}
    with CHATS_LOCK:
        log = d / "chat.jsonl"
        seq = 0
        if log.exists():
            try:
                seq = sum(1 for l in log.read_text(encoding="utf-8").splitlines() if l.strip())
            except Exception:
                seq = 0
        entry["seq"] = seq + 1
        with log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        meta = {"chat_id": d.name, "updated_at": entry["t"], "n_messages": entry["seq"],
                "preview": str(entry.get("text") or entry.get("tool_call", {}).get("name") or entry.get("run_id") or "")[:120]}
        try:
            if (d / "meta.json").exists():
                prev = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                prev.update(meta)
                meta = prev
        except Exception:
            pass
        write_json_atomic(d / "meta.json", meta)
    return entry

def chat_list_all() -> list[dict]:
    out = []
    try:
        for d in CHATS.iterdir():
            if not d.is_dir():
                continue
            try:
                out.append(json.loads((d / "meta.json").read_text(encoding="utf-8")))
            except Exception:
                out.append({"chat_id": d.name, "updated_at": "", "n_messages": 0, "preview": ""})
    except FileNotFoundError:
        pass
    out.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
    return out

def chat_read(chat_id: str) -> list[dict] | None:
    d = chat_path(chat_id)
    log = d / "chat.jsonl"
    # guard: chat_path sanitises, but reject traversal anyway
    try:
        log.relative_to(CHATS)
    except ValueError:
        return None
    if not log.exists():
        return None
    rows = []
    for l in log.read_text(encoding="utf-8").splitlines():
        if l.strip():
            try:
                rows.append(json.loads(l))
            except Exception:
                pass
    return rows


# -- Meta API calls (every call displayed: full POST + full return) --
def _redact_post(body: dict) -> dict:
    """Full POST body with inline base64 images replaced by their size. Key is never in the body."""
    import copy

    clone = copy.deepcopy(body)
    try:
        for item in clone.get("input", []):
            for part in item.get("content", []):
                if part.get("type") == "input_image":
                    url = part.get("image_url", "")
                    head, _, payload = url.partition(",")
                    part["image_url"] = f"{head},<{len(payload)} base64 chars redacted>"
        for img in clone.get("images", []):
            url = img.get("image_url", "")
            if "," in url:
                head, _, payload = url.partition(",")
                img["image_url"] = f"{head},<{len(payload)} base64 chars redacted>"
    except Exception:
        pass
    return clone


def _redact_muse_payload(payload: dict) -> dict:
    """Images-endpoint return with base64 bytes replaced by their size."""
    import copy

    clone = copy.deepcopy(payload)
    try:
        for item in clone.get("data", []):
            if item.get("b64_json"):
                item["b64_json"] = f"<{len(item['b64_json'])} base64 chars redacted>"
    except Exception:
        pass
    return clone


def muse_post(path: str, body: dict, timeout: int, run=None, stage=None, purpose="") -> tuple[dict, dict]:
    """POST to the Meta API with 429/5xx retry. Emits api.call/api.return on run when given.

    Returns (response_json, meta). Raises StageError on failure. The key travels
    in the header only and is never emitted, logged, or written anywhere.
    """
    import urllib.request

    if not muse_available():
        raise StageError("no live Muse Spark key (.cache/meta-api-key.txt missing or CHAT_MUSE=0)",
                         remedy="add the key or continue in replay mode")
    key = META_KEY.read_text(encoding="utf-8").strip()
    url = API_BASE + path
    call_ev = {"api": "muse-spark", "model": body.get("model"), "url": url,
               "post": _redact_post(body), "purpose": purpose}
    if run is not None:
        run.emit("api.call", stage=stage, data=call_ev)
    last_error = None
    for attempt in range(3):
        t0 = time.time()
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode())
            status = 200
            headers = dict(r.headers.items())
        except Exception as e:
            import urllib.error as _ue

            if isinstance(e, _ue.HTTPError):
                status = e.code
                try:
                    payload = json.loads(e.read().decode())
                except Exception:
                    payload = {"error": str(e)[:300]}
                headers = dict(e.headers.items())
            else:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < 2:
                    time.sleep(1.5 * (2 ** attempt))
                    continue
                raise StageError(f"Meta API call failed: {last_error}", remedy="retry or continue in replay mode")
        latency = round(time.time() - t0, 3)
        meta = {"status": status, "latency_s": latency, "attempt": attempt + 1,
                "request_id": headers.get("x-request-id") or headers.get("request-id"),
                "model": payload.get("model"), "response_id": payload.get("id"),
                "usage": payload.get("usage") or {}}
        if status == 200:
            if run is not None:
                run.emit("api.return", stage=stage, data={**meta, "response": payload})
            try:
                usage_add(meta.get("response_id") or "?", latency, meta.get("usage"))
            except Exception:
                pass
            return payload, meta
        if status in (429, 500, 502, 503, 504) and attempt < 2:
            ra = headers.get("Retry-After")
            wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else 2.0 * (2 ** attempt)
            time.sleep(min(wait, 30.0))
            last_error = f"HTTP {status}: {json.dumps(payload)[:300]}"
            continue
        if run is not None:
            run.emit("api.return", stage=stage, data={**meta, "response": payload})
        raise StageError(f"Meta API call failed with HTTP {status}: {json.dumps(payload)[:400]}",
                         remedy="check key/billing, or continue in replay mode")
    raise StageError(f"Meta API call failed after retries: {last_error}", remedy="retry or continue in replay mode")


def _responses_text(payload: dict) -> str:
    chunks = []
    for item in payload.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text" and part.get("text"):
                chunks.append(part["text"])
    return "".join(chunks)


def vision_scan(image_bytes: bytes, mime: str, run, hooks) -> dict:
    """Upload the photo to muse-spark-1.3-contributor, ask for the structured
    photo object. Returns the parsed dict. Full POST + return stream as events."""
    import base64 as _b64

    b64 = _b64.b64encode(image_bytes).decode()
    body = {
        "model": TEXT_MODEL,
        "instructions": VISION_INSTRUCTIONS,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": VISION_PROMPT},
            {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
        ]}],
        "text": {"format": {"type": "json_schema", "name": "photo_scan",
                            "description": "Photo description + detected people",
                            "schema": VISION_SCHEMA, "strict": True}},
        "reasoning": {"effort": "low"},
        "max_output_tokens": 1500,
        "store": False,
    }
    payload, meta = muse_post("/responses", body, 180, run=run, stage="vision",
                              purpose="first message: photo -> structured people scan")
    text = _responses_text(payload)
    try:
        scan = json.loads(text)
    except json.JSONDecodeError:
        raise StageError(f"vision scan did not parse as JSON; first 200 chars: {text[:200]}",
                         remedy="retry the upload")
    if not isinstance(scan.get("people-detected"), list) or not isinstance(scan.get("photo-description"), str):
        raise StageError("vision scan missing required keys (photo-description, people-detected)",
                         remedy="retry the upload")
    scan["_meta"] = {"response_id": meta.get("response_id"), "latency_s": meta.get("latency_s")}
    return scan


def draw_person_boxes(bgr, people_px):
    """Numbered bbox overlay (P1..Pn) so the user can pick a person visually."""
    import cv2

    ov = bgr.copy()
    H, W = bgr.shape[:2]
    for i, (box, desc) in enumerate(people_px):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), max(2, W // 400))
        cv2.rectangle(ov, (x1, max(0, y1 - 30)), (x1 + 220, y1), (0, 255, 0), -1)
        cv2.putText(ov, f"P{i + 1}", (x1 + 6, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return ov



# -- debug actions (standalone cards, not part of the gated pipeline) --
SAM_MODEL = "sam-3.1"
DEBUG_DESCRIBE_INSTRUCTIONS = (
    "You are a vision assistant inside a debugging tool. Describe exactly what is "
    "visible: people, clothing, pose, objects, setting. Never name real people, "
    "never infer identity, age, sex, health or personality."
)


def _parse_sam_line(text: str):
    """Parse one SAM output_text line into records.

    Returns [{object_id, box{x1,y1,x2,y2}, frame_w/h, mask{H,W,enc,payload}}].
    Payload alphabet excludes ,|<>, so records split cleanly on ','.
    """
    import re

    line = text.strip()
    line = re.sub(r"^<\d+f>", "", line).strip()
    if not line:
        return []
    recs = []
    pat = re.compile(r"(\d+)<\|box;([^|]*)\|><\|mask;([^|]*)\|>")
    for m in pat.finditer(line):
        oid, box_s, mask_s = m.group(1), m.group(2), m.group(3)
        box = {}
        for kv in box_s.split(";"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                box[k] = v
        mask = {}
        for kv in mask_s.split(";"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                mask[k] = v
        data = mask.get("data", "")
        mh, rest = (data.split(",", 1) + [""])[:2]
        mw, rest = (rest.split(",", 1) + [""])[:2]
        enc, payload = rest[:1], rest[1:]
        try:
            conf = float(box.get("c", 0.985)) if "c" in box else 0.985
            recs.append({
                "object_id": oid,
                "confidence": conf,
                "box": {k: float(box.get(k, 0)) for k in ("x1", "y1", "x2", "y2")},
                "frame_w": float(box.get("w", 0)), "frame_h": float(box.get("h", 0)),
                "mask": {"H": int(mh or 0), "W": int(mw or 0),
                         "enc": ("lossless" if enc == "~" else "one_bit"), "payload": enc + payload},
            })
        except ValueError:
            continue
    return recs


def sam_segment(image_bytes: bytes, mime: str, concept: str) -> tuple:
    """One sam-3.1 concept request. Returns (records, redacted_post, meta).

    Each record gains box_px [x1,y1,x2,y2], a full-frame bool mask and pixel count. Also returns raw_payload and raw_text for display. Raises StageError when the lane is empty or a mask fails."""
    import base64 as _b64
    import numpy as _np

    from meta_sam_parser import SegmentationMask, decode_mask_to_raster

    b64 = _b64.b64encode(image_bytes).decode()
    body = {
        "model": SAM_MODEL,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": concept.strip() or "person"},
            {"type": "input_image", "image_url": "data:{m};base64,{b}".format(m=mime, b=b64)},
        ]}],
        "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
        "store": False,
    }
    payload, meta = muse_post("/responses", body, 180, run=None,
                              purpose="debug: sam-3.1 detect '{c}'".format(c=concept))
    text = _responses_text(payload)
    recs = _parse_sam_line(text)
    if not recs:
        raise StageError("sam-3.1 returned zero matches for '{c}' (empty lane — normal, not an error)".format(c=concept),
                         remedy="try a plainer noun phrase, e.g. 'person'")
    for r in recs:
        bx = r["box"]
        x1, y1 = int(round(bx["x1"])), int(round(bx["y1"]))
        x2, y2 = int(round(bx["x2"])), int(round(bx["y2"]))
        r["box_px"] = [x1, y1, x2, y2]
        m = r["mask"]
        raster = decode_mask_to_raster(SegmentationMask(
            encoding=m["enc"], payload=m["payload"], width=m["W"], height=m["H"]))
        arr = _np.frombuffer(raster, dtype=_np.uint8).reshape(m["H"], m["W"])
        fw = int(r["frame_w"]) or (x2 + 1)
        fh = int(r["frame_h"]) or (y2 + 1)
        full = _np.zeros((fh, fw), dtype=_np.uint8)
        xa, ya = max(0, x1), max(0, y1)
        xb, yb = min(fw, x1 + m["W"]), min(fh, y1 + m["H"])
        if xb > xa and yb > ya:
            full[ya:yb, xa:xb] = arr[:yb - ya, :xb - xa]
        r["mask_full"] = full
        r["pixels"] = int(full.sum())
    return recs, _redact_post(body), meta, payload, text

def spark_describe(image_bytes: bytes, mime: str, prompt: str) -> tuple:
    """One free-text muse-spark-1.3-contributor vision turn. Returns (text, redacted_post, meta, raw_payload)."""
    import base64 as _b64

    b64 = _b64.b64encode(image_bytes).decode()
    body = {
        "model": TEXT_MODEL,
        "instructions": DEBUG_DESCRIBE_INSTRUCTIONS,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": "data:{m};base64,{b}".format(m=mime, b=b64)},
        ]}],
        "reasoning": {"effort": "low"},
        "max_output_tokens": 1000,
        "store": False,
    }
    payload, meta = muse_post("/responses", body, 180, run=None, purpose="debug: describe image")
    return _responses_text(payload) or "(empty reply)", _redact_post(body), meta, payload

def spark_generic(prompt: str, images: list) -> tuple:
    """Generic muse-spark-1.3-contributor turn: one input_text block plus one
    input_image block per (raw_bytes, mime) pair. Returns (text, redacted_post, meta, raw_payload)."""
    import base64 as _b64

    content = [{"type": "input_text", "text": prompt}]
    for raw, mime in images:
        content.append({"type": "input_image",
                        "image_url": "data:{m};base64,{b}".format(m=mime, b=_b64.b64encode(raw).decode())})
    body = {"model": TEXT_MODEL,
            "input": [{"role": "user", "content": content}],
            "store": False}
    payload, meta = muse_post("/responses", body, 180, run=None, purpose="debug: spark generic")
    return _responses_text(payload) or "(empty reply)", _redact_post(body), meta, payload

def muse_generate(prompt: str, images: list, size: str | None = None) -> tuple:
    """One muse-image-1.0 turn: generations with no refs, edits with 1-10
    (raw_bytes, mime) refs. Returns (img_bytes, out_format, redacted_post, meta, raw_payload)."""
    import base64 as _b64

    if not images:
        body = {"model": MUSE_MODEL, "prompt": prompt,
                "output_format": "png", "reasoning_strength": "low"}
        if size:
            body["size"] = size
        payload, meta = muse_post("/images/generations", body, 170, run=None, purpose="debug: muse generate")
    else:
        body = {"model": MUSE_MODEL, "prompt": prompt,
                "images": [{"image_url": "data:{m};base64,{b}".format(m=mime, b=_b64.b64encode(raw).decode())}
                           for raw, mime in images],
                "output_format": "png", "reasoning_strength": "low"}
        if size:
            body["size"] = size
        payload, meta = muse_post("/images/edits", body, 170, run=None, purpose="debug: muse edit")
    data = payload.get("data") or []
    if not data or not data[0].get("b64_json"):
        raise StageError("Muse returned no image bytes", remedy="retry, or check the trace")
    return _b64.b64decode(data[0]["b64_json"]), payload.get("output_format") or "png", _redact_post(body), meta, payload

def draw_mask_cards(bgr, records):
    """Per-record overlay (mask tint + box + id) plus a combined overlay."""
    import cv2
    import numpy as _np

    H, W = bgr.shape[:2]
    combined = bgr.copy()
    cards = []
    for i, r in enumerate(records):
        m = (r["mask_full"] > 0)
        tint = _np.zeros_like(bgr)
        tint[m] = (0, 255, 0)
        ov = cv2.addWeighted(bgr, 1.0, tint, 0.55, 0)
        x1, y1, x2, y2 = r["box_px"]
        cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), max(2, W // 500))
        cv2.putText(ov, "M{0}".format(i), (x1 + 6, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cards.append(ov)
        tint_all = _np.zeros_like(bgr)
        tint_all[m] = (0, 255, 0)
        combined = cv2.addWeighted(combined, 1.0, tint_all, 0.35, 0)
        cv2.rectangle(combined, (x1, y1), (x2, y2), (0, 255, 0), max(2, W // 500))
    return combined, cards


def mask_crop_for_spark(bgr, record, pad_frac=0.08):
    """Crop to the MASK's own bounding box, keep only masked pixels, black the rest.

    The old version cropped the detector box and merely dimmed anything outside
    the mask, so the result looked like the photo with a dark background instead
    of a mask crop. Now the subject fills the frame.
    """
    import numpy as _np

    H, W = bgr.shape[:2]
    m_full = (record["mask_full"] > 0)
    ys, xs = _np.nonzero(m_full)
    if len(xs) == 0:
        x1, y1, x2, y2 = record["box_px"]
    else:
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(W, x2 + px), min(H, y2 + py)
    crop = bgr[cy1:cy2, cx1:cx2].copy()
    m = m_full[cy1:cy2, cx1:cx2]
    crop[~m] = 0
    return crop


# -- Sapiens debug actions (local models, same stages as run_all.py) --
def _sapiens_evict():
    import gc

    import torch

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _progress_writer(chat_dir: Path, out_dir: Path, cid: str):
    """Append-only JSON-lines stage log. Dual-writes the chat-level live file
    (multiplexed across concurrent runs, filtered by cid) and the per-call
    record. Resets the live file past 256 KB. Never raises."""
    import time as _time

    live = chat_dir / "progress.jsonl"
    rec = out_dir / "progress.jsonl"
    try:
        if live.is_file() and live.stat().st_size > 256 * 1024:
            live.unlink()
    except Exception:
        pass

    def emit(stage: str, msg: str, frac=None):
        try:
            line = json.dumps({"t": round(_time.time(), 3), "cid": cid,
                               "stage": stage, "msg": msg, "frac": frac}) + "\n"
            with open(live, "a", encoding="utf-8") as fh:
                fh.write(line)
            with open(rec, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            pass

    return emit


def sapiens_debug(kind: str, bgr, out: Path, progress=None) -> tuple:
    """Run one Sapiens stage on bgr. Returns (summary_text, [artifact files], stats).

    Loads exactly one model, evicts it before returning (same one-resident
    discipline as analyze.py). Raises StageError on failure.
    """
    import time

    import cv2 as _cv2
    import numpy as _np

    import run_all as R

    H, W = bgr.shape[:2]
    t0 = time.time()

    progress = progress or (lambda *a, **k: None)
    progress("start", "sapiens {k}: loading 0.4b weights".format(k=kind), 0.05)
    if kind == "pose":
        model = R.load_pose()
        try:
            kp_any, _ = R.pose_run(model, bgr)
            bbox = R.person_box(kp_any, W, H)
            progress("run", "pose pass 2/2 on auto box", 0.45)
            kp, score = R.pose_run(model, bgr, bbox)
        finally:
            del model
            _sapiens_evict()
        _cv2.imwrite(str(out / "pose_keypoints.jpg"), R.draw_pose(bgr, kp))
        _np.savez_compressed(out / "pose.npz", kp=kp.astype(_np.float32),
                             score=_np.asarray(score, _np.float32),
                             bbox=_np.asarray(bbox, _np.float32))
        kp_list = []
        for (x, y), s in zip(_np.asarray(kp).reshape(-1, 2).tolist(),
                             _np.asarray(score).reshape(-1).tolist()):
            fx = round(float(x), 1) if _np.isfinite(x) else None
            fy = round(float(y), 1) if _np.isfinite(y) else None
            fs = round(float(s), 4) if _np.isfinite(s) else None
            kp_list.append([fx, fy, fs])
        s_fin = _np.asarray(score, dtype=_np.float64).reshape(-1)
        s_fin = s_fin[_np.isfinite(s_fin)]
        stats = {"pose_score_mean": (round(float(_np.mean(s_fin)), 4) if s_fin.size else None),
                 "pose_score_median": (round(float(_np.median(s_fin)), 4) if s_fin.size else None),
                 "bbox_xyxy": [round(float(v), 1) for v in bbox],
                 "keypoints": kp_list,
                 "keypoints_layout": "sapiens2-308: 0-69 body/foot/hand, 70-307 face",
                 "image_width": int(W), "image_height": int(H)}
        return ("Sapiens2 pose: 308 keypoints (green 70 body · magenta 238 face), "
                "median score {m}, auto box {b}.".format(m=(stats["pose_score_median"] if stats["pose_score_median"] is not None else "n/a"), b=stats["bbox_xyxy"]),
                ["pose_keypoints.jpg"], stats)

    if kind in ("seg", "normal", "depth"):
        seg_model = R.load_dense(
            "seg", "seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
            "seg/sapiens2_0.4b_seg.safetensors")
        try:
            from sapiens.dense.visualizers import SegVisualizer
            seg_vis = SegVisualizer(class_palette_type="dome29", with_labels=False)
        except Exception:
            seg_vis = None
        try:
            progress("run", "running seg", 0.3)
            labels, overlay = R.seg_run(seg_model, seg_vis, bgr)
            seg_pal = getattr(seg_vis, "class_palette", None)
            seg_names = getattr(seg_vis, "class_names", None) or {}
        finally:
            try:
                del seg_vis
            except Exception:
                pass
            del seg_model
            _sapiens_evict()
        files, stats = [], {}
        if kind == "seg":
            _cv2.imwrite(str(out / "seg_overlay.jpg"), overlay)
            counts = {int(c): int((labels == c).sum()) for c in _np.unique(labels)}
            (out / "seg_classes.json").write_text(json.dumps(counts, indent=2))
            _cv2.imwrite(str(out / "seg_labels.png"), labels)
            pal = seg_pal
            vis_names = seg_names
            legend = {"width": int(W), "height": int(H),
                      "counts": {str(c): n for c, n in counts.items()},
                      "palette": ({str(c): [int(v) for v in pal[int(c)][:3]]
                                   for c in counts} if pal is not None else {}),
                      "names": {str(c): str(vis_names.get(int(c), "class_%d" % c))
                                for c in counts}}
            (out / "seg_legend.json").write_text(json.dumps(legend))
            files = ["seg_overlay.jpg", "seg_classes.json", "seg_labels.png", "seg_legend.json"]
            stats = {"classes": sorted(counts), "skin_pixels": int(R.skin_mask_from_labels(labels).sum()),
                     "legend": legend}
            return ("Sapiens2 seg: {n} classes present, {s} skin pixels.".format(
                n=len(counts), s=stats["skin_pixels"]), files, stats)
        skin = R.skin_mask_from_labels(labels)
        if kind == "normal":
            progress("load", "seg done; loading normal weights", 0.55)
            model = R.load_dense(
                "normal", "normal/metasim_render_people/sapiens2_0.4b_normal_metasim_render_people-1024x768.py",
                "normal/sapiens2_0.4b_normal.safetensors")
            try:
                progress("run", "running normal", 0.7)
                sn, vis = R.normal_run(model, bgr, labels)
            finally:
                del model
                _sapiens_evict()
            _cv2.imwrite(str(out / "normal.jpg"), vis)
            _np.save(out / "normals.npy", sn.astype(_np.float16))
            return ("Sapiens2 normals (RGB, person-masked).", ["normal.jpg"], {})
        progress("load", "seg done; loading pointmap weights", 0.5)
        model = R.load_dense(
            "pointmap", "pointmap/render_people/sapiens2_0.4b_pointmap_render_people-1024x768.py",
            "pointmap/sapiens2_0.4b_pointmap.safetensors")
        progress("load", "pointmap weights ready", 0.6)
        try:
            progress("run", "running pointmap", 0.7)
            depth, vis = R.depth_run(model, bgr, labels)
        finally:
            del model
            _sapiens_evict()
        _cv2.imwrite(str(out / "depth.jpg"), vis)
        _np.save(out / "depth_m.npy", depth.astype(_np.float32))
        person = labels > 0
        dr = ([round(float(_np.percentile(depth[person], 1)), 3),
               round(float(_np.percentile(depth[person], 99)), 3)] if person.any() else None)
        return ("Sapiens2 pointmap depth in metres (turbo, person-masked), range {r}.".format(r=dr),
                ["depth.jpg"], {"depth_m": dr})
    raise StageError("unknown sapiens action: {k}".format(k=kind), remedy="use pose, seg, normal or depth")


# -- Body-fit + export debug actions (local GPU/CPU, same stages as analyze.py) --
def _resolve_chat_artifact(url) -> Path | None:
    """Map /api/chats/<cid>/file/debug/<id>/<name> to its on-disk path (validated)."""
    if not url:
        return None
    u = str(url).split("?")[0]
    if "/api/chats/" not in u or "/file/debug/" not in u:
        return None
    parts = [p for p in u.strip("/").split("/") if p]
    try:
        i = parts.index("chats")
        cid = parts[i + 1]
        if "/".join(parts[i + 2:i + 4]) != "file/debug" or len(parts) < i + 6:
            return None
        rel = parts[i + 5:]
        if any(s in ("", ".", "..") or "/" in s or "\\" in s for s in [cid] + rel):
            return None
        cand = CHATS / cid / "debug" / parts[i + 4]
        for s in rel:
            cand = cand / s
        return cand if cand.is_file() else None
    except Exception:
        return None


def body_fit_raw(bgr, out: Path, progress=None) -> tuple:
    """Stage-1 baseline: SAM 3D Body single HMR pass on the image alone.

    No Sapiens2 call, no identity refine, no joint solve, no expression - the
    honest floor that the supervised stages improve on.
    """
    import time as _time

    import cv2 as _cv2
    import numpy as _np

    import run_all as R

    H, W = bgr.shape[:2]
    t0 = _time.time()
    progress = progress or (lambda *a, **k: None)
    # Evaluation ruler only (Sapiens2 keypoints, NOT used as fit input). Run it
    # first: one model resident at a time, and 3DB + Sapiens in the reverse
    # order exhausts the ModelSlot.
    _kp_eval = _score_eval = None
    try:
        progress("eval", "baseline: Sapiens2 pose (comparison metric only)", 0.03)
        _pm = R.load_pose()
        try:
            _kp_any, _ = R.pose_run(_pm, bgr)
            _bbox_eval = R.person_box(_kp_any, W, H)
            _kp_eval, _score_eval = R.pose_run(_pm, bgr, _bbox_eval)
        finally:
            del _pm
            R.free({})
    except Exception:
        _kp_eval = _score_eval = None
    progress("start", "baseline: loading SAM 3D Body", 0.10)
    model, est, faces = R.load_3db()
    progress("run", "baseline: single HMR forward pass", 0.35)
    try:
        bbox = _np.asarray([0.0, 0.0, float(W), float(H)], _np.float32)
        out3d, verts, f = R.fit_3db(est, bgr, bbox)
    finally:
        del model
        R.free({})
    verts = _np.asarray(verts, _np.float64)
    _np.savez_compressed(out / "3db_outputs.npz",
                         **{k: _np.asarray(v) for k, v in out3d.items()
                            if isinstance(v, (_np.ndarray, list)) and k != "faces"},
                         faces=_np.asarray(faces, _np.int32))
    _np.save(out / "3db_vertices_fit.npy", verts)
    R._write_obj(out / "mesh_fit.obj", verts, faces, "SAM 3D Body raw HMR (camera frame, metres)")
    kp3 = _np.asarray(out3d["pred_keypoints_2d"], _np.float64)
    try:
        sheet = R.draw_3db_match(bgr, kp3, kp3)
    except Exception:
        sheet = bgr.copy()
    _cv2.imwrite(str(out / "06_3db_keypoints.jpg"), sheet)
    # Same face-error metric the refined stages report (Sapiens2 was run above
    # only as an evaluation ruler, never as fit input).
    eval_face_err = None
    if _kp_eval is not None:
        try:
            uv_raw = _np.asarray(out3d["pred_keypoints_2d"], _np.float64).reshape(-1, 2)
            kp_ev = _np.asarray(_kp_eval, _np.float64).reshape(-1, 2)
            sc_ev = _np.asarray(_score_eval, _np.float64).reshape(-1)
            n_rows = min(len(uv_raw), len(kp_ev), len(sc_ev))
            rows = _np.arange(n_rows)
            sel = (rows >= 70) & (rows < 220) & (sc_ev[:n_rows] > 0.2)
            if sel.any():
                d = _np.linalg.norm(uv_raw[sel, :2] - kp_ev[sel, :2], axis=1)
                eval_face_err = round(float(_np.mean(d)), 2)
        except Exception:
            eval_face_err = None

    beta = _np.asarray(out3d["shape_params"], _np.float64)
    cam = _np.asarray(out3d["pred_cam_t"], _np.float64)
    stats = {"size_wh": [int(W), int(H)], "bbox_xyxy": [0.0, 0.0, float(W), float(H)],
             "focal_px": round(float(f), 2), "cam_t_m": cam.round(4).tolist(),
             "identity_l2": round(float(_np.linalg.norm(beta)), 3), "expr_l2": 0.0,
             "refine_used": "raw", "joint_applied": False,
             "verts": int(verts.shape[0]), "faces": int(len(faces)),
             "beta_n": int(len(beta)), "beta": beta.round(4).tolist(),
             "mesh_file": "mesh_fit.obj", "sheet": "06_3db_keypoints.jpg",
             "stage": "raw", "face_err_px": eval_face_err,
             "timings_s": {"total": round(_time.time() - t0, 2)}}
    (out / "fit_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    # Identity snapshot for warm-starting the next ladder stage.
    (out / "identity_raw.json").write_text(json.dumps({
        "refined_shape_params": beta.round(6).tolist(),
        "refined_mhr_model_params": _np.asarray(out3d["mhr_model_params"], _np.float64).round(6).tolist(),
        "refined_cam_t": cam.round(6).tolist(),
        "refined_expr_params": _np.zeros(72).tolist(),
        "stage": "raw",
    }, indent=2), encoding="utf-8")
    summary = ("SAM 3D Body baseline (HMR map): {v:,} verts, |beta| {b}, f {f}px. "
               "No Sapiens2 in the fit, no refine, no expression.").format(
        v=int(verts.shape[0]), b=stats["identity_l2"], f=stats["focal_px"])
    if eval_face_err is not None:
        summary += " Face error vs Sapiens2 (eval only): {e} px.".format(e=eval_face_err)
    return summary, ["06_3db_keypoints.jpg", "mesh_fit.obj", "3db_outputs.npz", "fit_stats.json"], stats


def body_fit_debug(bgr, out: Path, progress=None, fit_expression: bool = True,
                   pose_npz: Path | None = None, seg_png: Path | None = None,
                   normal_npy: Path | None = None, depth_npy: Path | None = None,
                   init_json: Path | None = None) -> tuple:
    """Full local fitting chain on bgr. Returns (summary_text, [artifact files], stats).

    Mirrors analyze.run_photo pose→maps→body without gates or vision: Sapiens2
    308 auto box, 0.4b seg/normal/depth, SAM 3D Body fit, identity refine,
    joint solve, mesh sheets. One model resident at a time; evicts everything
    before returning. Raises StageError on failure.
    """
    import time as _time

    import cv2 as _cv2
    import numpy as _np

    import run_all as R

    try:
        from joint_solve import solve_joint as _solve_joint
    except Exception:
        _solve_joint = None

    H, W = bgr.shape[:2]
    t_all = _time.time()
    timings: dict[str, float] = {}
    progress = progress or (lambda *a, **k: None)
    progress("start", "body fit: loading pose weights", 0.02)
    files: list[str] = []

    # pose (Sapiens2 308) - reuse the Sapiens pose card artifact when wired
    t0 = _time.time()
    if pose_npz is not None:
        _pz = _np.load(pose_npz)
        kp = _pz["kp"].astype(_np.float64)
        score = _pz["score"].astype(_np.float64)
        bbox = _pz["bbox"].astype(_np.float64) if "bbox" in _pz else R.person_box(kp, W, H)
        _pz.close()
        progress("pose", "pose: 308 keypoints from the Sapiens pose card", 0.08)
    else:
        model = R.load_pose()
        progress("pose", "pose weights ready, running 2 passes", 0.06)
        try:
            kp_any, _ = R.pose_run(model, bgr)
            bbox = R.person_box(kp_any, W, H)
            kp, score = R.pose_run(model, bgr, bbox)
        finally:
            del model
            _sapiens_evict()
    _cv2.imwrite(str(out / "pose_keypoints.jpg"), R.draw_pose(bgr, kp))
    _np.savez_compressed(out / "pose.npz", kp=kp.astype(_np.float32),
                         score=_np.asarray(score, _np.float32),
                         bbox=_np.asarray(bbox, _np.float32))
    files += ["pose_keypoints.jpg", "pose.npz"]
    timings["pose"] = round(_time.time() - t0, 2)
    _ps = _np.asarray(score).reshape(-1); _ps = _ps[_np.isfinite(_ps)]
    pose_med = round(float(_np.median(_ps)), 4) if _ps.size else None
    progress("pose", "pose: score median {p}".format(p=pose_med), 0.12)

    # maps (seg / normal / depth) - reuse Sapiens card artifacts where wired
    progress("maps", "seg", 0.16)
    t0 = _time.time()
    if seg_png is not None:
        labels = _cv2.imread(str(seg_png), _cv2.IMREAD_GRAYSCALE)
        if labels is None:
            raise StageError("Sapiens seg artifact unreadable",
                             remedy="re-run the Sapiens seg card, then this one")
        skin = R.skin_mask_from_labels(labels)
        _cv2.imwrite(str(out / "seg_labels.png"), labels)
        files.append("seg_labels.png")
        progress("maps", "seg: mask from the Sapiens seg card", 0.20)
    else:
        seg_model = R.load_dense(
            "seg", "seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
            "seg/sapiens2_0.4b_seg.safetensors")
        try:
            from sapiens.dense.visualizers import SegVisualizer
            seg_vis = SegVisualizer(class_palette_type="dome29", with_labels=False)
        except Exception:
            seg_vis = None
        try:
            labels, overlay = R.seg_run(seg_model, seg_vis, bgr)
        finally:
            try:
                del seg_vis
            except Exception:
                pass
            del seg_model
            _sapiens_evict()
        skin = R.skin_mask_from_labels(labels)
        _cv2.imwrite(str(out / "seg_overlay.jpg"), overlay)
        files.append("seg_overlay.jpg")
        progress("maps", "seg: {n} classes, {s} skin px".format(
            n=len(_np.unique(labels)), s=int(skin.sum())), 0.22)

    progress("maps", "normal", 0.24)
    if normal_npy is not None:
        sn = _np.load(normal_npy).astype(_np.float32)
        vis_n = ((_np.clip(sn, -1, 1) + 1) / 2 * 255).astype(_np.uint8)[:, :, ::-1]
        _cv2.imwrite(str(out / "normal.jpg"), vis_n)
        files.append("normal.jpg")
        progress("maps", "normals from the Sapiens normal card", 0.28)
    else:
        model = R.load_dense(
            "normal", "normal/metasim_render_people/sapiens2_0.4b_normal_metasim_render_people-1024x768.py",
            "normal/sapiens2_0.4b_normal.safetensors")
        try:
            sn, vis_n = R.normal_run(model, bgr, labels)
        finally:
            del model
            _sapiens_evict()
        _cv2.imwrite(str(out / "normal.jpg"), vis_n)
        files.append("normal.jpg")
        progress("maps", "normals done", 0.30)
    sn_f32 = sn.astype(_np.float32)

    progress("maps", "depth", 0.32)
    person = labels > 0
    if depth_npy is not None:
        depth = _np.load(depth_npy).astype(_np.float32)
        progress("maps", "depth from the Sapiens depth card", 0.36)
    else:
        model = R.load_dense(
            "pointmap", "pointmap/render_people/sapiens2_0.4b_pointmap_render_people-1024x768.py",
            "pointmap/sapiens2_0.4b_pointmap.safetensors")
        try:
            depth, vis_d = R.depth_run(model, bgr, labels)
        finally:
            del model
            _sapiens_evict()
        _cv2.imwrite(str(out / "depth.jpg"), vis_d)
        files.append("depth.jpg")
        progress("maps", "depth {d} m".format(
            d=[round(float(_np.percentile(depth[person], 1)), 3),
               round(float(_np.percentile(depth[person], 99)), 3)] if person.any() else None), 0.38)
    depth_range = ([round(float(_np.percentile(depth[person], 1)), 3),
                    round(float(_np.percentile(depth[person], 99)), 3)] if person.any() else None)
    timings["maps"] = round(_time.time() - t0, 2)

    # body (3DB fit + identity_refine + joint + 06/07/08/09)
    progress("body", "loading 3D body weights", 0.40)
    t0 = _time.time()
    model, est, faces = R.load_3db()
    progress("body", "3D body weights ready, reconstructing", 0.46)
    try:
        out3d, verts, f = R.fit_3db(est, bgr, _np.asarray(bbox, _np.float32))
        for k in ("pred_vertices", "shape_params", "mhr_model_params", "focal_length", "pred_cam_t",
                  "pred_keypoints_2d"):
            if k not in out3d:
                raise StageError(f"3DB output missing key: {k}",
                                 remedy="checkpoint/model mismatch — re-check weights")
        if not bool(_np.isfinite(_np.asarray(out3d["pred_vertices"])).all()):
            raise StageError("3DB returned non-finite mesh (bad box/photo)", remedy="retry with auto box")

        _np.savez_compressed(out / "3db_outputs.npz",
                             **{k: _np.asarray(v) for k, v in out3d.items()
                                if isinstance(v, (_np.ndarray, list)) and k != "faces"},
                             faces=_np.asarray(faces, _np.int32))
        _np.save(out / "3db_vertices_fit.npy", _np.asarray(verts, _np.float64))
        R._write_obj(out / "mesh_fit.obj", _np.asarray(verts, _np.float64), faces,
                     "SAM 3D Body fit (camera frame, metres)")
        files += ["3db_outputs.npz", "mesh_fit.obj"]
        progress("body", "3DB fit: {v} verts, f {f}px".format(
            v=int(_np.asarray(verts).shape[0]), f=round(float(f), 1)), 0.55)

        # Progressive ladder: warm-start from the previous card's identity/params
        # instead of throwing away that fit and starting from a fresh human.
        init_used = None
        _init = None
        if init_json is not None and init_json.is_file():
            try:
                _init = json.loads(init_json.read_text(encoding="utf-8"))
                _ib = _np.asarray(_init.get("refined_shape_params", []), _np.float64)
                _ip = _np.asarray(_init.get("refined_mhr_model_params", []), _np.float64)
                _ic = _np.asarray(_init.get("refined_cam_t", []), _np.float64)
                if _ib.size and _ib.shape == _np.asarray(out3d["shape_params"]).shape:
                    out3d["shape_params"] = _ib
                if _ip.size and _ip.shape == _np.asarray(out3d["mhr_model_params"]).shape:
                    out3d["mhr_model_params"] = _ip
                if _ic.size == 3:
                    out3d["pred_cam_t"] = _ic
                init_used = str(init_json.name)
                progress("body", "warm start from {n}".format(n=init_used), 0.42)
            except Exception:
                init_used = None
        kp3_before = _np.asarray(out3d["pred_keypoints_2d"], _np.float64)
        kp3, cam_t, verts0 = kp3_before, _np.asarray(out3d["pred_cam_t"], _np.float64), verts
        beta_ref = _np.asarray(out3d["shape_params"], _np.float64)
        params_ref = _np.asarray(out3d["mhr_model_params"], _np.float64)
        used, joint, expr_ref = "raw", None, _np.zeros(72, _np.float64)
        verts_neutral = verts0

        # normal supervision for the joint solve (same as run_all)
        n_target = n_mask = None
        if skin is not None:
            try:
                sn_t = sn_f32 * _np.array([1.0, -1.0, -1.0], _np.float32)
                n_target = sn_t / _np.maximum(_np.linalg.norm(sn_t, axis=2, keepdims=True), 1e-8)
                n_mask = (skin & (_np.linalg.norm(sn_f32, axis=2) > 0.5)
                          & ~R.feature_exclusion_mask(kp, W, H))
            except Exception:
                n_target = n_mask = None

        try:
            progress("body", "identity refine: solving 700 iters", 0.60)
            ref = R.identity_refine(model, out3d, kp, score, W, H, f, cam_t)
        except Exception as e:
            ref = {"applied": False, "error": f"{type(e).__name__}: {e}"}
        if ref.get("applied"):
            beta_ref = _np.asarray(ref.get("beta", beta_ref), _np.float64)
            progress("body", "identity refine: applied, |β| {b}".format(
                b=round(float(_np.linalg.norm(beta_ref)), 3)), 0.68)

        if _solve_joint is not None:
            progress("body", "joint solve: 300 iters", 0.74)
            try:
                joint = _solve_joint(model, out3d, kp, score, W, H, f, cam_t,
                                     beta_init=beta_ref, identity_only=ref, iters=300,
                                     fit_expr=bool(fit_expression),
                                     normal_target=n_target, normal_mask=n_mask,
                                     faces=faces, normal_weight=0.5)
            except Exception as exc:
                joint = {"applied": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            joint = {"applied": False, "error": "joint_solve unavailable"}
        if joint is not None and joint.get("applied"):
            beta_ref = _np.asarray(joint["shape_params"], _np.float64)
            params_ref = _np.asarray(joint["mhr_model_params"], _np.float64)
            cam_t = _np.asarray(joint["cam_t"], _np.float64)
            expr_ref = _np.asarray(joint.get("expr_params", _np.zeros(72)), _np.float64)
            verts_root, lm_r = R.mesh_and_landmarks(model, out3d, beta_ref, model_params=params_ref, expr=expr_ref)
            verts_r = verts_root + cam_t[None]
            verts_neutral_root, _ = R.mesh_and_landmarks(model, out3d, beta_ref, model_params=params_ref)
            verts_neutral = verts_neutral_root + cam_t[None]
            kp3 = R.project_cam(lm_r[:70], f, cam_t, W, H)
            used = "joint"
            R._write_obj(out / "mesh_refined.obj", _np.asarray(verts_r, _np.float64), faces,
                         "refined mesh (camera frame, metres)")
            files.append("mesh_refined.obj")
        elif ref.get("applied"):
            verts_root, lm_r = R.mesh_and_landmarks(model, out3d, beta_ref)
            cam_t = _np.asarray(out3d["pred_cam_t"], _np.float64)
            verts_r = verts_root + cam_t[None]
            kp3 = R.project_cam(lm_r[:70], f, cam_t, W, H)
            used = "identity"
            R._write_obj(out / "mesh_refined.obj", _np.asarray(verts_r, _np.float64), faces,
                         "refined mesh (camera frame, metres)")
            files.append("mesh_refined.obj")
        else:
            verts_r = verts0
    finally:
        del model
        _sapiens_evict()

    progress("body", "solve branch: {u}".format(u=used), 0.80)
    _cv2.imwrite(str(out / "06_3db_keypoints.jpg"), R.draw_3db_match(bgr, kp3, kp))
    files.append("06_3db_keypoints.jpg")
    try:
        vcol = R.texture_mesh(verts_r, faces, bgr, f, W, H)
        timg, tmask = R.render_rgb(verts_r, faces, f, W, H, vcol=vcol)
        overlay = R.overlay_photo(bgr, timg, tmask, kp3, alpha=0.62)
        front = _cv2.cvtColor(timg, _cv2.COLOR_RGB2BGR)
        tile = max(W, H)
        side_t = _cv2.cvtColor(R.render_clay_view(verts_r, faces, f, tile, yaw=35.0, vcol=vcol), _cv2.COLOR_RGB2BGR)
        side_c = _cv2.cvtColor(R.render_clay_view(verts_r, faces, f, tile, yaw=35.0), _cv2.COLOR_RGB2BGR)
        panel = R.compose_grid([overlay, front, side_t, side_c],
                               ["3DB mesh + kps over photo", "3DB render (photo colours)",
                                "same mesh, 3/4 view", "geometry only (clay)"], cols=2)
        _cv2.imwrite(str(out / "07_3db_mesh_keypoints.jpg"), panel)
        files.append("07_3db_mesh_keypoints.jpg")
        face = kp[R.FACE_ROWS]
        x1, y1 = _np.floor(face.min(0)).astype(int)
        x2, y2 = _np.ceil(face.max(0)).astype(int)
        pad = int(0.2 * max(x2 - x1, y2 - y1))
        x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
        x2, y2 = min(x2 + pad, W), min(y2 + pad, H)
        crops = []
        for tag, vv, kk in (("before", verts0, kp3_before), ("after", verts_r, kp3)):
            img_b, mask_b = R.render_rgb(vv, faces, f, W, H, colour=R.MESH_COLOUR, ss=1.5)
            ov = R.overlay_photo(bgr, img_b, mask_b, kk, alpha=0.55)
            crops.append(ov[y1:y2, x1:x2].copy())
        _cv2.imwrite(str(out / "08_head_calibration.jpg"), R.compose_panels(
            crops, ["before: 3DB identity", f"after: {used} solve on Sapiens2 keypoints"]))
        files.append("08_head_calibration.jpg")
        photo = bgr.copy()
        for r in R.FACE_ROWS:
            x, y = kp[r]
            if _np.isfinite([x, y]).all() and 0 <= x < W and 0 <= y < H:
                _cv2.circle(photo, (int(x), int(y)), 2, (0, 255, 0), -1)
        img_n, _ = R.render_rgb(verts_neutral, faces, f, W, H, colour=R.MESH_COLOUR, ss=1.6)
        img_e, _ = R.render_rgb(verts_r, faces, f, W, H, colour=R.MESH_COLOUR, ss=1.6)

        def _crop_scale(im):
            c = im[y1:y2, x1:x2].copy()
            s = max(1.0, min(3.0, 820.0 / max(1, c.shape[0])))
            return _cv2.resize(c, None, fx=s, fy=s, interpolation=_cv2.INTER_CUBIC)

        expr_l2 = (joint or {}).get("metrics", {}).get("expr_l2", float(_np.linalg.norm(expr_ref)))
        _cv2.imwrite(str(out / "09_expression.jpg"), R.compose_panels(
            [_crop_scale(photo), _crop_scale(img_n), _crop_scale(img_e)],
            ["photo + Sapiens2 face kps", "joint solve, neutral jaw",
             f"joint + expression fit (|expr|={float(expr_l2):.2f})"]))
        files.append("09_expression.jpg")
        progress("renders", "sheets 06–09 done", 0.90)
    except Exception:
        _cv2.imwrite(str(out / "07_3db_mesh_keypoints.jpg"), R.draw_3db_match(bgr, kp3, kp))
        progress("renders", "sheets degraded to keypoint fallback", 0.90)
        files.append("07_3db_mesh_keypoints.jpg")
    if not bool(_np.isfinite(_np.asarray(verts_r)).all()):
        raise StageError("3DB refined mesh non-finite", remedy="retry with auto box")

    (out / "identity_joint.json").write_text(json.dumps({
        "format": "mhr-identity-joint/1",
        "original_shape_params": [round(float(v), 6) for v in _np.asarray(out3d["shape_params"]).reshape(-1)],
        "original_mhr_model_params": [round(float(v), 6) for v in _np.asarray(out3d["mhr_model_params"]).reshape(-1)],
        "original_cam_t": [round(float(v), 6) for v in _np.asarray(out3d["pred_cam_t"]).reshape(-1)],
        "refined_shape_params": [round(float(v), 6) for v in _np.asarray(beta_ref).reshape(-1)],
        "refined_mhr_model_params": [round(float(v), 6) for v in _np.asarray(params_ref).reshape(-1)],
        "refined_cam_t": [round(float(v), 6) for v in _np.asarray(cam_t).reshape(-1)],
        "refined_expr_params": [round(float(v), 6) for v in _np.asarray(expr_ref).reshape(-1)],
        "metrics": R._jsonable({k: v for k, v in (joint or {}).items()
                    if k not in ("shape_params", "mhr_model_params", "cam_t")}),
        "refine_used": used}, indent=2))
    files.append("identity_joint.json")
    timings["body"] = round(_time.time() - t0, 2)
    timings["total"] = round(_time.time() - t_all, 1)

    n_v = int(_np.asarray(verts_r).shape[0])
    n_f = int(_np.asarray(faces).shape[0])
    beta_list = [round(float(v), 6) for v in _np.asarray(beta_ref).reshape(-1)]
    stats = {"size_wh": [W, H], "bbox_xyxy": [round(float(v), 1) for v in _np.asarray(bbox).reshape(-1)],
             "pose_score_median": pose_med, "focal_px": round(float(f), 1),
             "cam_t_m": [round(float(v), 3) for v in _np.asarray(cam_t).reshape(-1)],
             "identity_l2": round(float(_np.linalg.norm(beta_ref)), 3), "refine_used": used,
             "joint_applied": bool((joint or {}).get("applied")),
             "expr_l2": round(float(_np.linalg.norm(expr_ref)), 3),
             "depth_m": depth_range, "verts": n_v, "faces": n_f,
             "beta_n": len(beta_list), "beta": beta_list,
             "mesh_file": "mesh_refined.obj" if used != "raw" else "mesh_fit.obj",
             "sheet": "07_3db_mesh_keypoints.jpg", "timings_s": timings,
             "face_err_px": ((joint or {}).get("metrics") or {}).get("face_err_px"),
             "ear_err_px": ((joint or {}).get("metrics") or {}).get("ear_err_px"),
             "init_from": init_used}
    (out / "fit_stats.json").write_text(json.dumps(stats, indent=2))
    files.append("fit_stats.json")

    # Turntable + clean mesh still via render_sam3body (Open3D smooth when
    # importable, else the numpy painter) — the Avatar asset card shows these.
    # A render failure must not kill the fit: it is recorded, not raised.
    try:
        from render_sam3body import contact_sheet as _tt_sheet
        from render_sam3body import render_numpy as _tt_numpy
        from render_sam3body import render_open3d as _tt_o3d
        from render_sam3body import turntable as _tt_views
        try:
            import open3d  # noqa: F401
            _tt_backend = "open3d"
        except Exception:
            _tt_backend = "numpy"
        _tt_render = _tt_o3d if _tt_backend == "open3d" else _tt_numpy
        _tt_img, _ = _tt_render(_np.asarray(verts_r, _np.float64), _np.asarray(faces, _np.int32),
                               float(f), W, H)
        _cv2.imwrite(str(out / "mesh_render.png"), _cv2.cvtColor(_tt_img, _cv2.COLOR_RGB2BGR))
        _tt_yaws = (_np.linspace(-180, 180, 6, endpoint=False) + 30.0).tolist()
        _tt_frames = _tt_views(_tt_backend, _np.asarray(verts_r, _np.float64),
                               _np.asarray(faces, _np.int32), float(f), _tt_yaws, size=600)
        _cv2.imwrite(str(out / "turntable.jpg"), _tt_sheet(_tt_frames, cols=6),
                     [_cv2.IMWRITE_JPEG_QUALITY, 92])
        files += ["mesh_render.png", "turntable.jpg"]
        progress("renders", "turntable ({b})".format(b=_tt_backend), 0.96)
        stats["render_backend"] = _tt_backend
        stats["mesh_render"] = "mesh_render.png"
        stats["turntable"] = "turntable.jpg"
        stats["turntable_yaws_deg"] = [round(float(y), 1) for y in _tt_yaws]
        (out / "fit_stats.json").write_text(json.dumps(stats, indent=2))
    except Exception as e:
        stats["turntable_error"] = "{t}: {e}".format(t=type(e).__name__, e=str(e)[:200])
        progress("renders", "turntable skipped ({t})".format(t=type(e).__name__), 0.96)
        (out / "fit_stats.json").write_text(json.dumps(stats, indent=2))
    _jm = (joint or {}).get("metrics") or {}
    face_err = _jm.get("face_err_px")
    ear_err = _jm.get("ear_err_px")
    expr_note = "" if fit_expression else " (expression OFF: neutral)"
    summary = ("SAM 3D Body fit ({u}): {v:,} verts on the MHR base mesh, |beta| {b}, "
               "|expr| {e}{x}, f {f}px, pose {p}, depth {d}.").format(
        u=used, v=n_v, b=stats["identity_l2"], e=stats["expr_l2"], x=expr_note,
        f=stats["focal_px"], p=pose_med, d=depth_range)
    if init_used:
        summary += " Warm-started from {n} (Δbeta {db:+.2f}).".format(
            n=init_used, db=stats["identity_l2"] - float(_np.linalg.norm(
                _np.asarray((_init or {}).get("refined_shape_params", [0.0]), _np.float64))))
    if face_err is not None:
        summary += " Face error: {fe} px".format(fe=face_err)
        if ear_err is not None:
            summary += ", ear {ee} px".format(ee=ear_err)
        summary += "."
    return (summary, files, stats)


def export_debug(mesh_bytes: bytes, mesh_name: str, out: Path, extra_blobs: dict | None = None) -> tuple:
    """Package a fitted mesh into avatar_character.glb. Returns (summary, [files], stats).

    CPU-only: parses the OBJ with trimesh, writes the GLB plus export_stats.json,
    and carries sibling spec files (identity_joint.json, fit_stats.json) alongside.
    Raises StageError when the mesh does not parse.
    """
    import io as _io

    try:
        import trimesh
    except Exception:
        raise StageError("trimesh is not installed in the server environment",
                         remedy="install trimesh into .venv, then retry")
    ftype = "glb" if str(mesh_name).lower().endswith(".glb") else "obj"
    try:
        mesh = trimesh.load(_io.BytesIO(mesh_bytes), file_type=ftype, force="mesh")
    except Exception as e:
        raise StageError("could not parse mesh ({t}: {e})".format(t=type(e).__name__, e=str(e)[:120]),
                         remedy="wire a body-fit mesh pin, not an image")
    if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
        raise StageError("mesh parsed but holds no vertices", remedy="re-run the body fit, then export again")
    mesh.export(str(out / "avatar_character.glb"))
    extra = dict(extra_blobs or {})
    for name, blob in extra.items():
        (out / name).write_bytes(blob)
    n_v = int(len(mesh.vertices))
    n_f = int(len(mesh.faces)) if hasattr(mesh, "faces") else 0
    n_b = (out / "avatar_character.glb").stat().st_size
    stats = {"verts": n_v, "faces": n_f, "glb_bytes": n_b, "glb_file": "avatar_character.glb",
             "source_mesh": mesh_name, "spec_files": sorted(extra)}
    (out / "export_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    files = ["avatar_character.glb", "export_stats.json"] + sorted(extra)
    summary = "avatar_character.glb: {v:,} verts, {f:,} faces, {kb:.1f} KB".format(
        v=n_v, f=n_f, kb=n_b / 1024)
    if extra:
        summary += " (+ {n} spec files).".format(n=len(extra))
    else:
        summary += "."
    return (summary, files, stats)


def chat_reply(text, has_photo, image_id):
    """Returns (assistant_text, tool_call|None, api_trace|None).

    Muse Spark via the Responses API when key+budget, else scripted.
    api_trace carries the full POST + return for display (key never included).
    Photo runs always synthesize the analyze_photo tool call — the vision scan
    with the structured object happens first inside Run.execute.
    """
    synth_tool = {"name": "analyze_photo", "args": {"image_id": image_id}} if has_photo else None
    if not muse_available():
        if has_photo:
            return ("Got it — first I look at the photo with Muse Spark (structured scan), "
                    "then the body scan runs (Sapiens2 308 → SAM 3D Body → joint solve). "
                    "Watch the tool card; numbers land with the renders. [replay mode: no live Muse key]",
                    synth_tool, None)
        return ("Hi — drop a photo and I'll fit the avatar mesh (face/ear px, aperture, renders). "
                "I design fictional characters only — no inference about real people. [replay mode]", None, None)
    u = usage_get()
    if u.get("text_calls", 0) >= MAX_TEXT_CALLS:
        return (f"Text budget spent ({u['text_calls']}/{MAX_TEXT_CALLS} calls, usage.json). "
                "Continuing in replay mode — photo analysis still runs locally.",
                synth_tool, None)
    body = {
        "model": TEXT_MODEL,
        "instructions": ("You are Muse, a character-creation assistant in an avatar builder. "
                         "Character creation only: never identify real people, no medical/demographic/personality claims. "
                         "A photo-analysis pipeline runs after your reply; keep it to two sentences. "
                         "Neutral jaw is a code decision (3DB zeroes 72 expr), not a model limit."),
        "input": text or ("analyze this photo: " + image_id if has_photo else "hello"),
        "reasoning": {"effort": "low"},
        "max_output_tokens": 1000,
        "store": False,
    }
    try:
        payload, meta = muse_post("/responses", body, 60, run=None, purpose="chat reply turn")
        trace = {"post": _redact_post(body), "response": payload,
                 "meta": {k: meta.get(k) for k in ("status", "latency_s", "request_id", "model", "response_id", "usage")}}
        return (_responses_text(payload) or "(empty reply)", synth_tool, trace)
    except StageError as e:
        return (f"Live Muse call failed ({str(e)[:120]}) — continuing in replay mode; photo analysis still runs locally.",
                synth_tool, None)


class Handler(BaseHTTPRequestHandler):
    server_version = "chat-wrapper/1"

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > 20 * 1024 * 1024:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8", "replace"))
        except Exception:
            return {}

    def do_GET(self):
        path_only = self.path.split("?", 1)[0]
        if path_only in ("/", "/chat"):
            return self._serve(DATA / "chat.html", "text/html; charset=utf-8")
        if path_only in ("/flow", "/flow-viewer"):
            self.send_response(302)
            self.send_header("Location", path_only + "/")
            self.end_headers()
            return
        if path_only in ("/flow/", "/flow-viewer/"):
            return self._serve(FLOW / "index.html", "text/html; charset=utf-8")
        for prefix in ("/flow/", "/flow-viewer/"):
            if path_only.startswith(prefix):
                rel = path_only[len(prefix):]
                fv_root = FLOW.resolve()
                target = (fv_root / rel).resolve()
                try:
                    if target.is_file() and target.is_relative_to(fv_root):
                        mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                        return self._serve(target, mime)
                except (ValueError, Exception):
                    pass
        if path_only in ("/reui.css", "/app.js"):
            target = (FLOW / path_only.lstrip("/")).resolve()
            if target.is_file():
                mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                return self._serve(target, mime)
        if self.path == "/api/health":
            try:
                _src_mtime = datetime.fromtimestamp(Path(__file__).stat().st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
            except (ValueError, OSError):
                _src_mtime = None
            return self._json({"ok": True, "muse": muse_available(), "active_run": ACTIVE_RUN,
                               "usage": usage_get(), "stages": STAGE_META,
                               "server_started": _STARTED_AT, "source_mtime": _src_mtime})
        if self.path == "/api/runs":
            with RUNS_LOCK:
                runs = sorted([p.name for p in RUNS.iterdir() if p.is_dir()])[-20:]
            return self._json({"runs": runs, "active_run": ACTIVE_RUN})
        if path_only == "/api/flow/prepared":
            fp = HERE / "flow-viewer" / "data" / "prepared_flow.json"
            if fp.is_file():
                b = fp.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            return self._json({"error": "no prepared_flow.json"}, 404)

        if path_only == "/api/flow/list":
            items = []
            if FLOW_GRAPHS.is_dir():
                for fp in sorted(FLOW_GRAPHS.glob("*.json")):
                    try:
                        rec = json.loads(fp.read_text(encoding="utf-8"))
                        items.append({"name": rec.get("name") or fp.stem,
                                      "savedAt": rec.get("savedAt"),
                                      "nodes": len(rec.get("nodes") or []),
                                      "edges": len(rec.get("edges") or [])})
                    except Exception:
                        pass
            return self._json({"graphs": items})

        if path_only == "/api/flow/saved":
            from urllib.parse import parse_qs as _pq
            qs = _pq(self.path.split("?", 1)[1]) if "?" in self.path else {}
            name = str((qs.get("name") or ["shared"])[0])
            safe = "".join(c for c in name if c.isalnum() or c in ("-", "_ ")).strip().replace(" ", "-")[:60] or "shared"
            fp = FLOW_GRAPHS / (safe + ".json")
            if not fp.is_file():
                fp = HERE / "flow-viewer" / "data" / "saved_flow.json"
            if fp.is_file():
                b = fp.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            return self._json({"error": "no saved flow yet"}, 404)

        if self.path == "/api/chats":
            return self._json({"chats": chat_list_all()})
        parts = self.path.split("?", 1)[0].rstrip("/").split("/")
        if len(parts) >= 6 and parts[1] == "api" and parts[2] == "chats" and parts[4] == "file":
            d = chat_path(parts[3])
            target = d
            for seg in parts[5:]:
                if seg in ("", ".", "..") or "/" in seg or "\\" in seg:
                    return self._json({"error": "bad path"}, 400)
                target = target / seg
            try:
                target.relative_to(CHATS)
            except ValueError:
                return self._json({"error": "bad path"}, 400)
            if not target.is_file():
                return self._json({"error": "not found"}, 404)
            return self._serve(target, mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        if len(parts) == 4 and parts[1] == "api" and parts[2] == "chats":
            rows = chat_read(parts[3])
            if rows is None:
                return self._json({"error": "unknown chat"}, 404)
            return self._json({"chat_id": chat_path(parts[3]).name, "messages": rows})
        parts = self.path.split("?", 1)[0].rstrip("/").split("/")
        # /api/runs/<id>/events | /run.json | /artifact/<aid>
        if len(parts) >= 5 and parts[1] == "api" and parts[2] == "runs":
            rid = parts[3]
            run = self._get_run(rid)
            if run is None:
                return self._json({"error": "unknown run"}, 404)
            if parts[4] == "events":
                with RUNS_LOCK:
                    live = rid in RUNS_MEM
                run._is_live = live
                return self._sse(run)
            if parts[4] == "run.json":
                try:
                    return self._serve(run.dir / "run.json", "application/json")
                except Exception:
                    return self._json({"error": "no run.json"}, 404)
            if len(parts) == 6 and parts[4] == "artifact":
                info = run.artifacts.get(parts[5])
                if info is None:
                    # disk fallback: search by aid prefix
                    cands = list(run.dir.rglob(f"*{parts[5]}*"))
                    if not cands:
                        return self._json({"error": "unknown artifact"}, 404)
                    path = cands[0]
                else:
                    path = run.dir / info["rel"]
                mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
                return self._serve(path, mime)
        return self._json({"error": "not found"}, 404)

    # -- debug actions: Describe (spark) / Detect (sam-3.1) / Apply-mask (crop) / Spark (generic) / Muse (generic) --
    def _debug_dir(self, chat_id: str) -> Path:
        d = chat_path(chat_id)
        i = 0
        while (d / "debug" / str(i)).exists():
            i += 1
        out = d / "debug" / str(i)
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _debug_image(self, body):
        """Decode an uploaded debug photo. Returns (raw_bytes, bgr, mime, ext)."""
        import base64 as _b64
        import cv2 as _cv2
        import numpy as _np

        b64 = body.get("image_b64") or ""
        raw = None
        if not b64 and body.get("image_url"):
            u = str(body["image_url"]).split("?")[0]
            if "/api/runs/" in u and "/artifact/" in u:
                parts = u.strip("/").split("/")
                if len(parts) >= 5:
                    rid, aid = parts[2], parts[4]
                    d = RUNS / rid
                    if d.is_dir():
                        _ok = all(s not in ("", ".", "..") and "/" not in s and "\\" not in s for s in (rid, aid))
                        for cand in ((d / f"{aid}.jpg", d / f"{aid}.png", d / "source.jpg", d / "source.png") if _ok else ()):
                            if cand.is_file():
                                raw = cand.read_bytes()
                                break
            elif "/api/chats/" in u and "/file/" in u:
                parts = u.strip("/").split("/")
                if len(parts) >= 5:
                    cid = parts[2]
                    rel_parts = parts[4:]
                    # Traversal guard (same rule as the GET file route): a bad
                    # segment anywhere — chat id or rel path — resolves nothing.
                    _segs = [parts[2], *rel_parts]
                    if any(s in ("", ".", "..") or "/" in s or "\\" in s for s in _segs):
                        rel_parts = None
                    cand = CHATS / cid
                    for p in (rel_parts or []):
                        cand = cand / p
                    if cand.is_file():
                        raw = cand.read_bytes()
            elif u.startswith(("http://", "https://")):
                try:
                    import urllib.request as _ur
                    req = _ur.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                    with _ur.urlopen(req, timeout=12) as r:
                        raw = r.read()
                except Exception as _ue:
                    pass

        if raw is None:
            if not b64 or len(b64) > 16 * 1024 * 1024:
                self._json({"error": "missing, unresolvable or oversize image"}, 400)
                return None, True
            try:
                if "," in b64:
                    b64 = b64.split(",", 1)[1]
                raw = _b64.b64decode(b64)
            except Exception:
                self._json({"error": "image_b64 is not base64"}, 400)
                return None, True

        arr = _np.frombuffer(raw, dtype=_np.uint8)
        bgr = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        if bgr is None:
            self._json({"error": "could not decode photo"}, 400)
            return None, True
        fn = str(body.get("filename", "upload.jpg")).lower()
        mime = "image/png" if fn.endswith(".png") or raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        return (raw, bgr, mime, ".png" if mime == "image/png" else ".jpg"), None

    def _debug_mesh(self, body):
        """Decode an uploaded debug mesh (OBJ). Returns ((raw_bytes, name, source_path|None), None) or (None, err)."""
        import base64 as _b64

        b64 = body.get("mesh_b64") or ""
        raw, name, src = None, str(body.get("filename") or "mesh.obj"), None
        if not b64 and body.get("mesh_url"):
            u = str(body["mesh_url"]).split("?")[0]
            if "/api/runs/" in u and "/artifact/" in u:
                parts = u.strip("/").split("/")
                if len(parts) >= 5:
                    rid, aid = parts[2], parts[4]
                    d = RUNS / rid
                    if d.is_dir():
                        _ok = all(s not in ("", ".", "..") and "/" not in s and "\\" not in s for s in (rid, aid))
                        cands = [c for c in sorted(d.rglob(f"*{aid}*")) if c.is_file()] if _ok else []
                        cands.sort(key=lambda c: (c.suffix.lower() not in (".obj", ".glb"), c.name))
                        if cands:
                            raw, src = cands[0].read_bytes(), cands[0]
                            name = cands[0].name
            elif "/api/chats/" in u and "/file/" in u:
                parts = u.strip("/").split("/")
                if len(parts) >= 5:
                    cid = parts[2]
                    if cid in ("", ".", "..") or "/" in cid or "\\" in cid:
                        cid = "__bad__"
                    cand = CHATS / cid
                    ok = True
                    for p in parts[4:]:
                        if p in ("", ".", "..") or "/" in p or "\\" in p:
                            ok = False
                            break
                        cand = cand / p
                    if ok and cand.is_file():
                        raw, src = cand.read_bytes(), cand
                        name = cand.name
            elif u.startswith(("http://", "https://")):
                try:
                    import urllib.request as _ur
                    req = _ur.Request(u, headers={"User-Agent": "Mozilla/5.0"})
                    with _ur.urlopen(req, timeout=12) as r:
                        raw = r.read()
                    name = u.rsplit("/", 1)[-1] or name
                except Exception:
                    pass

        if raw is None:
            if not b64 or len(b64) > 64 * 1024 * 1024:
                self._json({"error": "missing, unresolvable or oversize mesh"}, 400)
                return None, True
            try:
                raw = _b64.b64decode(b64.split(",", 1)[1] if "," in b64 else b64)
            except Exception:
                self._json({"error": "mesh_b64 is not base64"}, 400)
                return None, True
        if not raw:
            self._json({"error": "empty mesh"}, 400)
            return None, True
        return (raw, name, src), None

    def _debug(self, action: str):
        # Never drop a debug connection: an unhandled exception becomes a
        # structured 500 (the viewer fails loud with error + remedy) instead
        # of a bare "Failed to fetch". Per-action finally blocks still run.
        try:
            return self._debug_inner(action)
        except Exception as e:
            tname = type(e).__name__
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            oom = ("OutOfMemory" in tname or "out of memory" in str(e).lower())
            return self._json({
                "error": "{t}: {m}".format(t=tname, m=(str(e)[:300] or "request failed")),
                "remedy": ("GPU out of memory — close other GPU programs and run again; "
                           "if it repeats, restart the server" if oom else
                           "unexpected server error — see the server console traceback, then run again"),
            }, 500)

    def _debug_inner(self, action: str):
        import cv2 as _cv2

        body = self._read_json()
        chat_id = "".join(c for c in str(body.get("chat_id") or "") if c.isalnum() or c in ("-", "_"))[:64]
        if not chat_id:
            chat_id = "flow-session"

        if action == "describe":
            got, err = self._debug_image(body)
            if err is not None:
                return err
            raw, bgr, mime, ext = got
            out = self._debug_dir(chat_id)
            (out / ("source" + ext)).write_bytes(raw)
            prompt = str(body.get("prompt") or "Describe this image in detail: who/what is visible, layout, notable details.")
            try:
                text, post, meta, payload = spark_describe(raw, mime, prompt)
            except StageError as e:
                return self._json({"error": str(e)[:300], "remedy": getattr(e, "remedy", "")}, 502)
            trace = {"post": post, "response": payload,
                     "meta": {k: meta.get(k) for k in ("status", "latency_s", "request_id", "model", "response_id", "usage")}}
            chat_append(chat_id, {"role": "user", "text": "🔍 Describe this image",
                                  "filename": body.get("filename", ""), "debug_id": out.name})
            chat_append(chat_id, {"role": "assistant", "text": text, "api_trace": trace,
                                  "debug_id": out.name, "kind": "debug-describe"})
            return self._json({"chat_id": chat_path(chat_id).name, "debug_id": out.name,
                               "assistant_text": text, "api_trace": trace})

        if action == "detect":
            got, err = self._debug_image(body)
            if err is not None:
                return err
            raw, bgr, mime, ext = got
            concept = str(body.get("concept") or "person")[:80]
            out = self._debug_dir(chat_id)
            (out / ("source" + ext)).write_bytes(raw)
            try:
                recs, post, meta, payload, raw_text = sam_segment(raw, mime, concept)
            except StageError as e:
                return self._json({"error": str(e)[:300], "remedy": getattr(e, "remedy", "")}, 502)
            combined, cards = draw_mask_cards(bgr, recs)
            _cv2.imwrite(str(out / "combined.jpg"), combined)
            masks = []
            H, W = bgr.shape[:2]
            for i, (r, ov) in enumerate(zip(recs, cards)):
                _cv2.imwrite(str(out / "mask_{0}.jpg".format(i)), ov)
                _cv2.imwrite(str(out / "mask_{0}_raw.png".format(i)), (r["mask_full"] * 255).astype("uint8"))
                # Clean bounding box crop
                bx = r["box_px"]
                pad = 12
                cx1, cy1 = max(0, bx[0] - pad), max(0, bx[1] - pad)
                cx2, cy2 = min(W, bx[2] + pad), min(H, bx[3] + pad)
                crop_bgr = bgr[cy1:cy2, cx1:cx2]
                if crop_bgr.size > 0:
                    _cv2.imwrite(str(out / "crop_{0}.jpg".format(i)), crop_bgr)
                masks.append({
                    "id": i,
                    "object_id": r["object_id"],
                    "box": r["box_px"],
                    "confidence": round(float(r.get("confidence", 0.985)), 3),
                    "pixels": r["pixels"],
                    "encoding": r["mask"]["enc"],
                    "file": "mask_{0}.jpg".format(i),
                    "file_url": f"/api/chats/{chat_path(chat_id).name}/file/debug/{out.name}/mask_{i}.jpg",
                    "crop_url": f"/api/chats/{chat_path(chat_id).name}/file/debug/{out.name}/crop_{i}.jpg",
                })
            (out / "masks.json").write_text(json.dumps(
                {"concept": concept, "count": len(masks), "masks": masks}, indent=2), encoding="utf-8")
            trace = {"post": post, "response": {"output_text": raw_text},
                     "meta": {k: meta.get(k) for k in ("status", "latency_s", "request_id", "model", "response_id", "usage")}}
            summary = "sam-3.1 found {n} match(es) for '{c}'. Click a mask card to describe it.".format(n=len(masks), c=concept)
            chat_append(chat_id, {"role": "user", "text": "🧩 Detect '{c}'".format(c=concept),
                                  "filename": body.get("filename", ""), "debug_id": out.name})
            chat_append(chat_id, {"role": "assistant", "text": summary, "api_trace": trace,
                                  "debug_id": out.name, "kind": "debug-masks",
                                  "concept": concept, "masks": masks,
                                  "combined": "combined.jpg"})
            return self._json({
                "ok": True,
                "chat_id": chat_path(chat_id).name,
                "debug_id": out.name,
                "assistant_text": summary,
                "api_trace": trace,
                "concept": concept,
                "count": len(masks),
                "masks": masks,
                "combined": "combined.jpg",
                "combined_url": f"/api/chats/{chat_path(chat_id).name}/file/debug/{out.name}/combined.jpg",
                "source_url": f"/api/chats/{chat_path(chat_id).name}/file/debug/{out.name}/source{ext}",
                "image_width": W,
                "image_height": H,
            })

        if action == "apply-mask":
            debug_id = "".join(c for c in str(body.get("debug_id") or "") if c.isalnum() or c in ("-", "_"))[:16]
            try:
                mask_id = int(body.get("mask_id"))
            except Exception:
                return self._json({"error": "missing mask_id"}, 400)
            out = chat_path(chat_id) / "debug" / debug_id
            try:
                (out / "masks.json").relative_to(CHATS)
            except ValueError:
                return self._json({"error": "bad path"}, 400)
            if not (out / "masks.json").is_file():
                return self._json({"error": "unknown debug_id"}, 404)
            info = json.loads((out / "masks.json").read_text(encoding="utf-8"))
            masks = info.get("masks", [])
            if not (0 <= mask_id < len(masks)):
                return self._json({"error": "unknown mask_id"}, 404)
            src = next((out.glob("source.*")), None)
            if src is None:
                return self._json({"error": "debug source missing"}, 404)
            bgr = _cv2.imread(str(src), _cv2.IMREAD_COLOR)
            raw_mask = _cv2.imread(str(out / "mask_{0}_raw.png".format(mask_id)), _cv2.IMREAD_GRAYSCALE)
            if bgr is None or raw_mask is None:
                return self._json({"error": "debug files unreadable"}, 500)
            import numpy as _np

            record = {"box_px": masks[mask_id]["box"],
                      "mask_full": (raw_mask > 0).astype(_np.uint8)}
            crop = mask_crop_for_spark(bgr, record)
            _cv2.imwrite(str(out / "mask_{0}_crop.jpg".format(mask_id)), crop)
            h, w = crop.shape[:2]
            return self._json({"chat_id": chat_path(chat_id).name, "debug_id": out.name,
                               "assistant_text": "mask M{0} applied: {1}x{2} region crop".format(mask_id, w, h),
                               "crop": "mask_{0}_crop.jpg".format(mask_id),
                               "width": int(w), "height": int(h),
                               "box": [int(v) for v in masks[mask_id]["box"]]})
        if action == "spark":
            prompt = str(body.get("prompt") or "").strip()
            if not prompt:
                return self._json({"error": "missing prompt"}, 400)
            items = body.get("images") or []
            if not isinstance(items, list):
                return self._json({"error": "images must be a list"}, 400)
            images = []
            for it in items:
                got, err = self._debug_image(it if isinstance(it, dict) else {})
                if err is not None:
                    return err
                raw, bgr, mime, ext = got
                images.append((raw, mime))
            try:
                text, post, meta, payload = spark_generic(prompt, images)
            except StageError as e:
                return self._json({"error": str(e)[:300], "remedy": getattr(e, "remedy", "")}, 502)
            trace = {"post": post, "response": payload,
                     "meta": {k: meta.get(k) for k in ("status", "latency_s", "request_id", "model", "response_id", "usage")}}
            chat_append(chat_id, {"role": "user", "text": "✨ Spark: " + prompt[:200]})
            chat_append(chat_id, {"role": "assistant", "text": text, "api_trace": trace,
                                  "kind": "debug-spark"})
            return self._json({"chat_id": chat_path(chat_id).name,
                               "assistant_text": text, "api_trace": trace})

        if action == "muse":
            import numpy as _np

            prompt = str(body.get("prompt") or "").strip()
            if not prompt:
                return self._json({"error": "missing prompt"}, 400)
            items = body.get("images") or []
            if not isinstance(items, list) or len(items) > 10:
                return self._json({"error": "images must be a list of at most 10"}, 400)
            images = []
            for it in items:
                got, err = self._debug_image(it if isinstance(it, dict) else {})
                if err is not None:
                    return err
                raw, bgr, mime, ext = got
                images.append((raw, mime))
            out = self._debug_dir(chat_id)
            try:
                img_bytes, out_format, post, meta, payload = muse_generate(
                    prompt, images, size=str(body.get("size") or "") or None)
            except StageError as e:
                return self._json({"error": str(e)[:300], "remedy": getattr(e, "remedy", "")}, 502)
            ext = {"png": ".png", "jpeg": ".jpg", "jpg": ".jpg", "webp": ".webp"}.get(out_format, ".png")
            fname = "generated" + ext
            (out / fname).write_bytes(img_bytes)
            bgr = _cv2.imdecode(_np.frombuffer(img_bytes, dtype=_np.uint8), _cv2.IMREAD_COLOR)
            H, W = (bgr.shape[:2] if bgr is not None else (0, 0))
            trace = {"post": post, "response": _redact_muse_payload(payload),
                     "meta": {k: meta.get(k) for k in ("status", "latency_s", "request_id", "model", "response_id", "usage")}}
            chat_append(chat_id, {"role": "user", "text": "🎨 Muse: " + prompt[:200],
                                  "debug_id": out.name})
            chat_append(chat_id, {"role": "assistant",
                                  "text": "muse-image-1.0 render" + (" (edit)" if images else ""),
                                  "api_trace": trace, "debug_id": out.name,
                                  "kind": "debug-muse", "image": fname})
            return self._json({"chat_id": chat_path(chat_id).name, "debug_id": out.name,
                               "image_url": f"/api/chats/{chat_path(chat_id).name}/file/debug/{out.name}/{fname}",
                               "output_format": out_format, "image_width": W, "image_height": H,
                               "api_trace": trace})

        if action in ("sapiens-pose", "sapiens-seg", "sapiens-normal", "sapiens-depth"):
            if ACTIVE_RUN is not None:
                return self._json({"error": "a body-scan run holds the GPU; wait or cancel it first",
                                    "active_run": ACTIVE_RUN}, 409)
            got, err = self._debug_image(body)
            if err is not None:
                return err
            raw, bgr, mime, ext = got
            kind = action.split("-", 1)[1]
            out = self._debug_dir(chat_id)
            (out / ("source" + ext)).write_bytes(raw)
            import time as _time

            t0 = _time.time()
            if not GPU_LOCK.acquire(blocking=False):
                return self._json({"error": "GPU is busy with another debug call",
                                   "remedy": "wait for the running card to finish, then run again"}, 409)
            try:
                progress = _progress_writer(chat_path(chat_id), out, str(body.get("client_debug_id") or out.name))
                summary, files, stats = sapiens_debug(kind, bgr, out, progress)
            except StageError as e:
                return self._json({"error": str(e)[:300], "remedy": getattr(e, "remedy", "")}, 502)
            finally:
                GPU_LOCK.release()
            secs = round(_time.time() - t0, 1)
            summary = summary + " ({s}s, local Sapiens2, one model resident).".format(s=secs)
            rel_files = ["source" + ext] + files
            chat_append(chat_id, {"role": "user",
                                  "text": "🧩 Sapiens {k}".format(k=kind),
                                  "filename": body.get("filename", ""), "debug_id": out.name})
            chat_append(chat_id, {"role": "assistant", "text": summary,
                                  "debug_id": out.name, "kind": "debug-sapiens",
                                  "files": rel_files, "stats": stats})
            cid = chat_path(chat_id).name
            base = "/api/chats/{c}/file/debug/{d}".format(c=cid, d=out.name)
            art = {"pose": base + "/pose.npz", "seg": base + "/seg_labels.png",
                   "normal": base + "/normals.npy", "depth": base + "/depth_m.npy"}
            return self._json({"chat_id": cid, "debug_id": out.name,
                               "assistant_text": summary, "files": rel_files, "stats": stats,
                               "artifact_url": art.get(kind)})
        if action == "body-fit":
            if ACTIVE_RUN is not None:
                return self._json({"error": "a body-scan run holds the GPU; wait or cancel it first",
                                   "active_run": ACTIVE_RUN}, 409)
            got, err = self._debug_image(body)
            if err is not None:
                return err
            raw, bgr, mime, ext = got
            out = self._debug_dir(chat_id)
            (out / ("source" + ext)).write_bytes(raw)
            import time as _time

            t0 = _time.time()
            if not GPU_LOCK.acquire(blocking=False):
                return self._json({"error": "GPU is busy with another debug call",
                                   "remedy": "wait for the running card to finish, then run again"}, 409)
            try:
                progress = _progress_writer(chat_path(chat_id), out, str(body.get("client_debug_id") or out.name))
                _init_mesh = _resolve_chat_artifact(body.get("init_mesh_url"))
                _init_json = None
                if _init_mesh is not None:
                    for cand in ("identity_joint.json", "identity_raw.json"):
                        if (_init_mesh.parent / cand).is_file():
                            _init_json = _init_mesh.parent / cand
                            break
                if str(body.get("stage") or "full") == "raw":
                    summary, files, stats = body_fit_raw(bgr, out, progress)
                else:
                    summary, files, stats = body_fit_debug(
                        bgr, out, progress,
                        init_json=_init_json,
                        fit_expression=bool(body.get("fit_expression", True)),
                        pose_npz=_resolve_chat_artifact(body.get("pose_url")),
                        seg_png=_resolve_chat_artifact(body.get("seg_url")),
                        normal_npy=_resolve_chat_artifact(body.get("normal_url")),
                        depth_npy=_resolve_chat_artifact(body.get("depth_url")))
            except StageError as e:
                return self._json({"error": str(e)[:300], "remedy": getattr(e, "remedy", "")}, 502)
            finally:
                GPU_LOCK.release()
            secs = round(_time.time() - t0, 1)
            summary = summary + " ({s}s, local SAM 3D Body + joint, one model resident).".format(s=secs)
            rel_files = ["source" + ext] + files
            chat_append(chat_id, {"role": "user",
                                  "text": "🧩 Body fit",
                                  "filename": body.get("filename", ""), "debug_id": out.name})
            chat_append(chat_id, {"role": "assistant", "text": summary,
                                  "debug_id": out.name, "kind": "debug-bodyfit",
                                  "files": rel_files, "stats": stats})
            cid = chat_path(chat_id).name
            return self._json({"chat_id": cid, "debug_id": out.name,
                               "assistant_text": summary, "files": rel_files, "stats": stats,
                               "mesh_url": f"/api/chats/{cid}/file/debug/{out.name}/{stats['mesh_file']}",
                               "sheet": stats["sheet"]})

        if action == "mhr-match":
            if ACTIVE_RUN is not None:
                return self._json({"error": "a body-scan run holds the GPU; wait or cancel it first",
                                   "active_run": ACTIVE_RUN}, 409)
            got, err = self._debug_image(body)
            if err is not None:
                return err
            raw, bgr, mime, ext = got
            out = self._debug_dir(chat_id)
            (out / ("source" + ext)).write_bytes(raw)
            import time as _time
            rows = body.get("rows")
            cols = body.get("cols")
            t0 = _time.time()
            if not GPU_LOCK.acquire(blocking=False):
                return self._json({"error": "GPU is busy with another debug call",
                                   "remedy": "wait for the running card to finish, then run again"}, 409)
            try:
                import mhr_multiview_match as _MM
                progress = _progress_writer(chat_path(chat_id), out, str(body.get("client_debug_id") or out.name))
                summary, files, stats = _MM.mhr_multiview_match(bgr, out, rows, cols, progress)
            except StageError as e:
                return self._json({"error": str(e)[:300], "remedy": getattr(e, "remedy", "")}, 502)
            finally:
                GPU_LOCK.release()
            secs = round(_time.time() - t0, 1)
            summary = summary + " ({s}s, {n} local MHR fits, one model resident).".format(
                s=secs, n=stats.get("n_panels", "?"))
            rel_files = ["source" + ext] + files
            chat_append(chat_id, {"role": "user", "text": "MHR multi-view match",
                                  "filename": body.get("filename", ""), "debug_id": out.name})
            chat_append(chat_id, {"role": "assistant", "text": summary,
                                  "debug_id": out.name, "kind": "debug-mhr-match",
                                  "files": rel_files, "stats": stats})
            cid = chat_path(chat_id).name
            return self._json({"chat_id": cid, "debug_id": out.name,
                               "assistant_text": summary, "files": rel_files, "stats": stats,
                               "preview_url": f"/api/chats/{cid}/file/debug/{out.name}/mhr_match_cloud.png",
                               "cloud_url": f"/api/chats/{cid}/file/debug/{out.name}/mhr_match_cloud.ply",
                               "report_url": f"/api/chats/{cid}/file/debug/{out.name}/mhr_match_report.json"})

        if action == "vggt-check":
            if ACTIVE_RUN is not None:
                return self._json({"error": "a body-scan run holds the GPU; wait or cancel it first",
                                   "active_run": ACTIVE_RUN}, 409)
            got, err = self._debug_image(body)
            if err is not None:
                return err
            raw, bgr, mime, ext = got
            out = self._debug_dir(chat_id)
            (out / ("source" + ext)).write_bytes(raw)
            import time as _time
            rows = body.get("rows")
            cols = body.get("cols")
            t0 = _time.time()
            if not GPU_LOCK.acquire(blocking=False):
                return self._json({"error": "GPU is busy with another debug call",
                                   "remedy": "wait for the running card to finish, then run again"}, 409)
            try:
                import avatar_extra as _AX
                progress = _progress_writer(chat_path(chat_id), out, str(body.get("client_debug_id") or out.name))
                summary, files, stats = _AX.vggt_check(bgr, out, rows, cols, progress)
            except StageError as e:
                return self._json({"error": str(e)[:300], "remedy": getattr(e, "remedy", "")}, 502)
            finally:
                GPU_LOCK.release()
            secs = round(_time.time() - t0, 1)
            summary = summary + " ({s}s, local VGGT-Omega + Sapiens2 masks, one model resident).".format(s=secs)
            rel_files = ["source" + ext] + files
            chat_append(chat_id, {"role": "user", "text": "VGGT multi-view check",
                                  "filename": body.get("filename", ""), "debug_id": out.name})
            chat_append(chat_id, {"role": "assistant", "text": summary,
                                  "debug_id": out.name, "kind": "debug-vggt",
                                  "files": rel_files, "stats": stats})
            cid = chat_path(chat_id).name
            return self._json({"chat_id": cid, "debug_id": out.name,
                               "assistant_text": summary, "files": rel_files, "stats": stats,
                               "preview_url": f"/api/chats/{cid}/file/debug/{out.name}/vggt_cloud.png",
                               "cloud_url": f"/api/chats/{cid}/file/debug/{out.name}/vggt_cloud.ply",
                               "report_url": f"/api/chats/{cid}/file/debug/{out.name}/vggt_report.json"})

        if action == "texture":
            got_m, err = self._debug_mesh(body)
            if err is not None:
                return err
            mesh_bytes, mesh_name, src_path = got_m
            if src_path is None:
                return self._json({"error": "texture needs the fitted mesh (not an upload)",
                                   "remedy": "wire a body-fit mesh pin into this card"}, 400)
            got_p, err = self._debug_image(body)
            if err is not None:
                return err
            raw_p, bgr_p, mime_p, ext_p = got_p
            out = self._debug_dir(chat_id)
            if not GPU_LOCK.acquire(blocking=False):
                return self._json({"error": "GPU is busy with another debug call"}, 409)
            try:
                import texture_bake as _TX
                progress = _progress_writer(chat_path(chat_id), out, str(body.get("client_debug_id") or out.name))
                summary, files, stats = _TX.texture_bake(src_path, bgr_p, out, progress)
            except StageError as e:
                return self._json({"error": str(e)[:300], "remedy": getattr(e, "remedy", "")}, 502)
            finally:
                GPU_LOCK.release()
            chat_append(chat_id, {"role": "user", "text": "Texture bake", "debug_id": out.name})
            chat_append(chat_id, {"role": "assistant", "text": summary, "debug_id": out.name,
                                  "kind": "debug-texture", "files": files, "stats": stats})
            cid = chat_path(chat_id).name
            base = f"/api/chats/{cid}/file/debug/{out.name}"
            return self._json({"chat_id": cid, "debug_id": out.name, "assistant_text": summary,
                               "files": files, "stats": stats,
                               "glb_url": f"{base}/mesh_textured.glb",
                               "texture_url": f"{base}/texture_uv_1024.png",
                               "preview_url": f"{base}/textured_preview.jpg",
                               "sheet_url": f"{base}/textured_turnaround_2x3.jpg",
                               "sheet_wide_url": f"{base}/textured_turnaround.jpg"})

        if action == "apose":
            got, err = self._debug_mesh(body)
            if err is not None:
                return err
            mesh_bytes, mesh_name, src_path = got
            if src_path is None:
                return self._json({"error": "canonical A-pose needs the body-fit mesh (not an upload)",
                                   "remedy": "wire a body-fit mesh pin into this card and run it again"}, 400)
            fit_dir = src_path.parent
            if not (fit_dir / "identity_joint.json").is_file():
                return self._json({"error": "identity_joint.json missing next to the mesh",
                                   "remedy": "re-run the body-fit card, then this one"}, 404)
            out = self._debug_dir(chat_id)
            if not GPU_LOCK.acquire(blocking=False):
                return self._json({"error": "GPU is busy with another debug call"}, 409)
            try:
                import avatar_extra as _AX
                progress = _progress_writer(chat_path(chat_id), out, str(body.get("client_debug_id") or out.name))
                summary, files, stats = _AX.apose_export(None, fit_dir, out, progress)
            finally:
                GPU_LOCK.release()
            chat_append(chat_id, {"role": "user", "text": "Canonical A-pose", "debug_id": out.name})
            chat_append(chat_id, {"role": "assistant", "text": summary, "debug_id": out.name,
                                  "kind": "debug-apose", "files": files, "stats": stats})
            cid = chat_path(chat_id).name
            return self._json({"chat_id": cid, "debug_id": out.name, "assistant_text": summary,
                               "files": files, "stats": stats,
                               "mesh_url": f"/api/chats/{cid}/file/debug/{out.name}/mesh_apose.ply",
                               "obj_url": f"/api/chats/{cid}/file/debug/{out.name}/mesh_apose.obj",
                               "sheet": "turnaround_apose_2x3.jpg",
                               "sheet_wide": "turnaround_apose.jpg"})

        if action == "export":
            got, err = self._debug_mesh(body)
            if err is not None:
                return err
            mesh_bytes, mesh_name, src_path = got
            out = self._debug_dir(chat_id)
            extra = {}
            if src_path is not None:
                for sib in ("identity_joint.json", "fit_stats.json"):
                    p = src_path.parent / sib
                    if p.is_file():
                        try:
                            extra[sib] = p.read_bytes()
                        except Exception:
                            pass
            try:
                summary, files, stats = export_debug(mesh_bytes, mesh_name, out, extra)
            except StageError as e:
                return self._json({"error": str(e)[:300], "remedy": getattr(e, "remedy", "")}, 502)
            chat_append(chat_id, {"role": "user", "text": "📦 Export " + mesh_name[:120],
                                  "debug_id": out.name})
            chat_append(chat_id, {"role": "assistant", "text": summary,
                                  "debug_id": out.name, "kind": "debug-export",
                                  "files": files, "stats": stats})
            cid = chat_path(chat_id).name
            return self._json({"chat_id": cid, "debug_id": out.name,
                               "assistant_text": summary, "files": files, "stats": stats,
                               "glb_url": f"/api/chats/{cid}/file/debug/{out.name}/avatar_character.glb"})
        if action == "fit-overlay":
            # CPU-only matched render (Open3D offscreen painter): no GPU lock.
            import numpy as _np2
            import run_all as R
            import trimesh as _trimesh

            debug_id = "".join(c for c in str(body.get("debug_id") or "") if c.isalnum() or c in ("-", "_"))[:64]
            fit_dir = chat_path(chat_id) / "debug" / debug_id
            try:
                fit_dir.relative_to(CHATS)
            except ValueError:
                return self._json({"error": "bad path"}, 400)
            stats_path = fit_dir / "fit_stats.json"
            if not debug_id or not stats_path.is_file():
                return self._json({"error": "unknown fit debug_id",
                                    "remedy": "run the body-fit card live first, then overlay from it"}, 404)
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
                f = float(stats["focal_px"])
                W, H = (int(v) for v in stats["size_wh"])
                mesh_name = str(stats.get("mesh_file") or "")
            except Exception:
                return self._json({"error": "fit stats unreadable",
                                    "remedy": "re-run the body-fit card"}, 500)
            if mesh_name not in ("mesh_refined.obj", "mesh_fit.obj"):
                return self._json({"error": "unknown fit mesh"}, 500)
            mesh_path = fit_dir / mesh_name
            if not mesh_path.is_file():
                return self._json({"error": "fit mesh missing",
                                    "remedy": "re-run the body-fit card"}, 404)
            got, err = self._debug_image(body)
            if err is not None:
                return err
            raw, photo, mime, ext = got
            ph, pw = photo.shape[:2]
            resized = None
            if (pw, ph) != (W, H):
                photo = _cv2.resize(photo, (W, H), interpolation=_cv2.INTER_AREA)
                resized = "photo resized {0}x{1} to fit frame {2}x{3}".format(pw, ph, W, H)
            try:
                tm = _trimesh.load(str(mesh_path), force="mesh")
                verts = _np2.asarray(tm.vertices, dtype=_np2.float64)
                faces = _np2.asarray(tm.faces, dtype=_np2.int64)
            except Exception as e:
                return self._json({"error": "fit mesh unreadable: {0}".format(str(e)[:120])}, 500)
            try:
                # Same matched-frame call body-fit uses (verts carry cam_t baked).
                img, alpha = R.render_rgb(verts, faces, f, W, H, colour=R.MESH_COLOUR, ss=1.6)
                render_backend = "open3d"
            except ImportError as e:
                # No open3d in this env (conda runtime): same numpy fallback
                # body-fit's turntable uses — same projection, plain OpenCV.
                if (getattr(e, "name", None) or "open3d").split(".")[0] != "open3d":
                    raise
                from render_sam3body import render_numpy as _ov_numpy
                img, alpha = _ov_numpy(verts, faces, f, W, H)
                render_backend = "numpy"
            except Exception as e:
                return self._json({"error": "matched render failed: {0}".format(str(e)[:160])}, 502)
            mask = alpha > 0.5
            render_bgr = _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR)
            overlay = photo.copy()
            blend = ((photo.astype(_np2.float32) + render_bgr.astype(_np2.float32)) / 2).astype(_np2.uint8)
            overlay[mask] = blend[mask]
            out = self._debug_dir(chat_id)
            _cv2.imwrite(str(out / "fit_photo.jpg"), photo)
            _cv2.imwrite(str(out / "fit_render.jpg"), render_bgr)
            _cv2.imwrite(str(out / "fit_overlay.jpg"), overlay)
            mean_d = float(_np2.abs(photo.astype(_np2.float32) - render_bgr.astype(_np2.float32)).mean())
            files = ["fit_photo.jpg", "fit_render.jpg", "fit_overlay.jpg"]
            cid = chat_path(chat_id).name
            summary = "fit overlay: matched render blended 50/50 over the photo (mean abs Δ {0:.1f}/255{1})".format(
                mean_d, "; " + resized if resized else "")
            chat_append(chat_id, {"role": "user", "text": "Fit overlay of " + debug_id,
                                  "debug_id": out.name})
            chat_append(chat_id, {"role": "assistant", "text": summary,
                                  "debug_id": out.name, "kind": "debug-fit-overlay",
                                  "image": "fit_overlay.jpg"})
            return self._json({"chat_id": cid, "debug_id": out.name,
                               "assistant_text": summary, "files": files,
                               "photo_url": "/api/chats/{0}/file/debug/{1}/fit_photo.jpg".format(cid, out.name),
                               "render_url": "/api/chats/{0}/file/debug/{1}/fit_render.jpg".format(cid, out.name),
                               "overlay_url": "/api/chats/{0}/file/debug/{1}/fit_overlay.jpg".format(cid, out.name),
                               "width": int(W), "height": int(H),
                               "render_backend": render_backend,
                               "stats": {"focal_px": f, "size_wh": [W, H],
                                         "mean_abs_delta": round(mean_d, 2), "resized": resized,
                                         "render_backend": render_backend}})
        if action == "mesh-render":
            # Server-side orbit still (open3d offscreen when installed, numpy
            # painter fallback): CPU call, no GPU lock.
            import io as _io2
            import numpy as _np2
            import run_all as R
            import trimesh as _trimesh

            got, err = self._debug_mesh(body)
            if err is not None:
                return err
            raw, mname, _ = got
            mode = str(body.get("mode") or "shaded").lower()
            if mode not in ("shaded", "depth", "normal", "multiview"):
                return self._json({"error": "unknown mesh-render mode '{m}'".format(m=mode[:16]),
                                    "remedy": "mode must be shaded, depth, normal or multiview"}, 400)
            framing = str(body.get("framing") or "body").lower()
            if framing not in R.ORBIT_FRAMINGS:
                return self._json({"error": "unknown framing '{f}'".format(f=framing[:16]),
                                    "remedy": "framing must be body, waist or face"}, 400)
            count = 0
            if mode == "multiview":
                try:
                    count = int(body.get("count") or 0)
                except Exception:
                    count = 0
                if count not in (2, 4, 6, 8):
                    return self._json({"error": "multiview count must be 2, 4, 6 or 8"}, 400)
            try:
                size = int(body.get("size") or (384 if mode == "multiview" else 512))
            except Exception:
                size = 384 if mode == "multiview" else 512
            size = max(128, min(1024, size))
            try:
                yaw = float(body.get("yaw_deg") or 35.0)
            except Exception:
                yaw = 35.0
            _low = mname.lower()
            if _low.endswith(".glb") or raw[:4] == b"glTF":
                ft = "glb"
            elif _low.endswith(".ply") or raw[:3] == b"ply":
                ft = "ply"
            else:
                ft = "obj"
            try:
                tm = _trimesh.load(_io2.BytesIO(raw), file_type=ft, force="mesh")
                if isinstance(tm, _trimesh.Scene):
                    tm = tm.dump(concatenate=True)
                verts = _np2.asarray(tm.vertices, dtype=_np2.float64)
                faces = _np2.asarray(tm.faces, dtype=_np2.int64)
            except Exception as e:
                return self._json({"error": "mesh unreadable: {0}".format(str(e)[:120]),
                                    "remedy": "wire a body-fit mesh pin, not an image"}, 500)
            if len(verts) == 0 or len(faces) == 0:
                return self._json({"error": "mesh holds no geometry"}, 500)
            paint = "shaded" if mode == "multiview" else mode
            yaws = [float(i * 360) / count for i in range(count)] if mode == "multiview" else [yaw]
            frames, backend = [], None
            try:
                for y in yaws:
                    img, mask = R.render_orbit(verts, faces, size, y, paint, colour=R.MESH_COLOUR,
                                               framing="body", y_down=(ft == "obj"))
                    # Face/waist used to be computed inside orbit_world with a
                    # +Y-up assumption, which lands on the legs for camera-frame
                    # (Y-down) meshes. Crop from the verified-upright body render
                    # instead - orientation-independent.
                    if framing in ("face", "waist") and mask.any():
                        ys, xs = _np2.nonzero(mask)
                        y0, y1 = int(ys.min()), int(ys.max())
                        h = max(8, y1 - y0)
                        if framing == "face":
                            a, b = y0, y0 + max(8, int(0.30 * h))
                        else:
                            a, b = y0 + int(0.18 * h), y0 + int(0.62 * h)
                        cx = (int(xs.min()) + int(xs.max())) // 2
                        half = max(8, int(0.62 * (b - a)))
                        a2, b2 = max(0, a), min(img.shape[0], b)
                        c0, c1 = max(0, cx - half), min(img.shape[1], cx + half)
                        if b2 > a2 and c1 > c0:
                            crop = img[a2:b2, c0:c1]
                            img = _cv2.resize(crop, (size, size), interpolation=_cv2.INTER_AREA)
                    frames.append(img)
                backend = "open3d"
            except ImportError as e:
                if (getattr(e, "name", None) or "open3d").split(".")[0] != "open3d":
                    raise
                from render_sam3body import render_numpy as _mr_numpy
                frames = []
                for y in yaws:
                    world, f2 = R.orbit_world(verts, size, y, *R.ORBIT_FRAMINGS[framing])
                    if paint == "shaded":
                        img, _ = _mr_numpy(world, faces, f2, size, size)
                    elif paint == "depth":
                        img, _ = R.render_depth_np(world, faces, f2, size)
                    else:
                        img, _ = R.render_normal_np(world, faces, f2, size)
                    frames.append(img)
                backend = "numpy"
            out = self._debug_dir(chat_id)
            if mode == "multiview":
                labels = ["{0:g} deg".format(y) for y in yaws]
                cols = {2: 2, 4: 2, 6: 3, 8: 4}[count]
                sheet = R.compose_grid(frames, labels, cols=cols)
                _cv2.imwrite(str(out / "mesh_multiview.jpg"), _cv2.cvtColor(sheet, _cv2.COLOR_RGB2BGR))
                W, H, fname = sheet.shape[1], sheet.shape[0], "mesh_multiview.jpg"
            else:
                _cv2.imwrite(str(out / "mesh_still.png"), _cv2.cvtColor(frames[0], _cv2.COLOR_RGB2BGR))
                W, H, fname = size, size, "mesh_still.png"
            files = [fname]
            cid = chat_path(chat_id).name
            what = "{0}-view sheet".format(count) if mode == "multiview" else "{0} still".format(mode)
            summary = "mesh {0} ({1}, {2}): {3}x{4}".format(what, backend, mname, W, H)
            chat_append(chat_id, {"role": "user", "text": "Mesh render ({0}) of ".format(mode) + mname,
                                  "debug_id": out.name})
            chat_append(chat_id, {"role": "assistant", "text": summary,
                                  "debug_id": out.name, "kind": "debug-mesh-render",
                                  "image": fname})
            return self._json({"chat_id": cid, "debug_id": out.name,
                               "assistant_text": summary, "files": files,
                               "image_url": "/api/chats/{0}/file/debug/{1}/{2}".format(cid, out.name, fname),
                               "width": int(W), "height": int(H),
                               "render_backend": backend,
                               "stats": {"mode": mode, "size": size, "backend": backend, "framing": framing,
                                         "verts": int(len(verts)), "faces": int(len(faces)),
                                         "mesh": mname}})
        return self._json({"error": "unknown debug action '{a}' — restart chat_server.py for the current backend".format(a=action), "remedy": "the server predates this card; restart it and run again"}, 404)

    def do_POST(self):
        global ACTIVE_RUN
        if self.path == "/api/chat":
            body = self._read_json()
            text = (body.get("text") or "")[:2000]
            image_b64 = body.get("image_b64")
            chat_id = "".join(c for c in str(body.get("chat_id") or "") if c.isalnum() or c in ("-", "_"))[:64]
            if not chat_id:
                chat_id = datetime.now(timezone.utc).strftime("chat-%Y%m%d-%H%M%S-") + secrets.token_hex(2)
            if image_b64 and len(image_b64) > 16 * 1024 * 1024:
                return self._json({"error": "image too large"}, 413)
            # every user turn is saved first, so a reload never loses it
            chat_append(chat_id, {"role": "user", "text": text,
                                  "filename": body.get("filename", "") if image_b64 else "",
                                  "has_photo": bool(image_b64)})
            run_id = None
            if image_b64:
                superseded = None
                with RUN_LOCK:
                    if ACTIVE_RUN is not None:
                        old = self._get_run(ACTIVE_RUN)
                        waiting = (old is not None and isinstance(old, Run)
                                   and getattr(old, "gate_open", None) is not None)
                        if waiting:
                            # paused on a gate: the user moved on, so retire it, don't block
                            old.cancel_event.set()
                            old.gate_choice = "__superseded__"
                            old.gate_event.set()
                            old.emit("gate.answer", stage=(old.gate_open or {}).get("stage"),
                                     data={"choice": "__superseded__"})
                            old.emit("run.end", data={"status": "cancelled",
                                                      "error": "superseded by a newer photo"})
                            superseded = ACTIVE_RUN
                            ACTIVE_RUN = None
                        else:
                            return self._json(
                                {"error": f"run {ACTIVE_RUN} is computing (not waiting for input)",
                                 "hint": "watch it, or cancel it first, then resend",
                                 "active_run": ACTIVE_RUN}, 409)
                    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
                    ACTIVE_RUN = run_id
                if not GPU_LOCK.acquire(blocking=False):
                    with RUN_LOCK:
                        if ACTIVE_RUN == run_id:
                            ACTIVE_RUN = None
                    return self._json({"error": "GPU is busy with a debug call",
                                       "remedy": "wait for the running card to finish, then resend the photo"}, 409)
                if superseded:
                    chat_append(chat_id, {"role": "assistant",
                                          "text": "Retired paused run {r} — starting this photo instead.".format(r=superseded),
                                          "run_id": superseded})
                assistant_text, tool_call, api_trace = chat_reply(text, True, run_id)
                opts = {"full_maps": bool(body.get("full_maps", False))}
                if isinstance(tool_call, dict) and isinstance(tool_call.get("args"), dict):
                    if tool_call["args"].get("full_maps"):
                        opts["full_maps"] = True
                run = Run(run_id, image_b64, body.get("filename", "upload.jpg"),
                          {"assistant_preamble": assistant_text, "tool_call": tool_call,
                           "chat_id": chat_id, **opts})
                with RUNS_LOCK:
                    RUNS_MEM[run_id] = run
                def _guarded():
                    try:
                        run.execute()
                    finally:
                        try:
                            GPU_LOCK.release()
                        except Exception:
                            pass
                threading.Thread(target=_guarded, daemon=True).start()
                chat_append(chat_id, {"role": "assistant", "text": assistant_text,
                                      "tool_call": tool_call, "run_id": run_id,
                                      "api_trace": api_trace})
                return self._json({"chat_id": chat_id, "run_id": run_id, "assistant_text": assistant_text,
                                   "tool_call": tool_call, "api_trace": api_trace,
                                   "events_url": f"/api/runs/{run_id}/events"})
            assistant_text, tool_call, api_trace = chat_reply(text, False, "")
            chat_append(chat_id, {"role": "assistant", "text": assistant_text,
                                  "tool_call": tool_call, "api_trace": api_trace})
            return self._json({"chat_id": chat_id, "assistant_text": assistant_text,
                               "tool_call": tool_call, "api_trace": api_trace})
        path_only = self.path.split("?", 1)[0]
        if path_only == "/api/flow/save":
            body = self._read_json()
            data = body.get("flow") if isinstance(body.get("flow"), dict) else body
            if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
                return self._json({"error": "flow must contain nodes"}, 400)
            name = str(body.get("name") or data.get("name") or "shared")
            safe = "".join(c for c in name if c.isalnum() or c in ("-", "_ ")).strip().replace(" ", "-")[:60] or "shared"
            data["name"] = name
            data["savedAt"] = int(time.time() * 1000)
            data["savedBy"] = str(body.get("saved_by") or "client")[:60]
            payload = json.dumps(data)
            if len(payload) > 24_000_000:
                return self._json({"error": "flow too large to save"}, 413)
            FLOW_GRAPHS.mkdir(parents=True, exist_ok=True)
            (FLOW_GRAPHS / (safe + ".json")).write_text(payload, encoding="utf-8")
            # legacy single-slot file (older clients / Load Graph fallback)
            d = HERE / "flow-viewer" / "data"
            d.mkdir(parents=True, exist_ok=True)
            (d / "saved_flow.json").write_text(payload, encoding="utf-8")
            return self._json({"ok": True, "name": name, "file": f"flow-viewer/data/graphs/{safe}.json",
                               "savedAt": data["savedAt"], "bytes": len(payload),
                               "nodes": len(data.get("nodes") or []),
                               "edges": len(data.get("edges") or [])})
        parts = _split_api_path(self.path)
        if len(parts) == 4 and parts[1] == "api" and parts[2] == "debug":
            return self._debug(parts[3])
        if len(parts) == 6 and parts[1] == "api" and parts[2] == "runs" and parts[4] == "choice":
            run = self._get_run(parts[3])
            if run is None or not isinstance(run, Run) or not hasattr(run, "gate_event"):
                return self._json({"error": "unknown or finished run"}, 404)
            if run.gate_open is None:
                return self._json({"error": "no open gate — nothing is waiting"}, 409)
            choice = (self._read_json().get("choice") or "").strip()[:64]
            if not choice:
                return self._json({"error": "missing choice"}, 400)
            if isinstance(run, Run):
                run.answer_gate(choice)
                chat_id = (run.opts or {}).get("chat_id")
                if chat_id:
                    chat_append(chat_id, {"role": "user", "text": f"[{run.id} choice] {choice}",
                                          "run_id": run.id, "gate_choice": True})
            return self._json({"ok": True, "choice": choice})
        parts = self.path.rstrip("/").split("/")
        if len(parts) == 5 and parts[1] == "api" and parts[2] == "runs" and parts[4] == "cancel":
            run = self._get_run(parts[3])
            if run is None:
                return self._json({"error": "unknown run"}, 404)
            run.cancel_event.set()
            if getattr(run, "gate_open", None) is not None:
                run.gate_choice = "cancel"
                run.gate_event.set()
            else:
                run.emit("run.end", data={"status": "cancelled"})
            if ACTIVE_RUN == parts[3]:
                ACTIVE_RUN = None
            return self._json({"ok": True})
        return self._json({"error": "not found"}, 404)

    def _get_run(self, rid):
        with RUNS_LOCK:
            run = RUNS_MEM.get(rid)
        if run is not None:
            return run
        d = RUNS / rid
        if not d.is_dir():
            return None
        # light rehydrate for replay (events from disk)
        run = Run.__new__(Run)
        run.id, run.dir = rid, d
        run.artifacts, run.lock = {}, threading.RLock()
        run.subscribers = set()
        run._fh = open(os.devnull, "a")
        run.stats = None
        run.status = "unknown"
        run.created_at = ""
        run.cancel_event = threading.Event()
        run.gate_event = threading.Event()
        run.gate_open = None
        run.gate_choice = None
        run.opts = {}
        try:
            run.events = [json.loads(l) for l in (d / "events.jsonl").read_text().splitlines() if l.strip()]
            run.seq = len(run.events)
        except Exception:
            run.events, run.seq = [], 0
        try:
            rec = json.loads((d / "run.json").read_text(encoding="utf-8"))
            run.status = rec.get("status", "unknown")
            run.created_at = rec.get("created_at", "")
            run.stats = rec.get("stats")
        except Exception:
            for ev in reversed(run.events):
                if ev.get("type") == "run.end":
                    run.status = ev.get("data", {}).get("status", "unknown")
                    break
            else:
                run.status = "running" if run.events else "unknown"
        return run

    def _sse(self, run):
        last = self.headers.get("Last-Event-ID")
        try:
            last_id = int((last or "0").split(":")[-1])
        except Exception:
            last_id = 0
        replay, q = run.subscribe()
        close_after = (
            (run.status in ("done", "failed", "cancelled") and replay and replay[-1].get("type") == "run.end")
            or not getattr(run, "_is_live", True))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        if close_after:
            self.send_header("Connection", "close")
        else:
            self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for ev in replay:
                if ev.get("seq", 0) > last_id:
                    self.wfile.write(f"id: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
            if close_after:
                # rehydrated runs can never emit again in this process: close after replay
                run.unsubscribe(q)
                return
            while True:
                try:
                    ev = q.get(timeout=15)
                    self.wfile.write(f"id: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                    if ev.get("type") == "run.end":
                        break
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            run.unsubscribe(q)

    def _serve(self, path: Path, mime: str):
        try:
            b = path.read_bytes()
        except Exception:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(b)))
        # Never let a stale cached bundle masquerade as a crashed backend.
        if str(path).lower().endswith((".js", ".css", ".html")):
            self.send_header("Cache-Control", "no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8150)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; use 0.0.0.0 to share the server (and the shared graph) on the LAN")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"chat-wrapper on http://{a.host}:{a.port}/  (runs in {RUNS})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
