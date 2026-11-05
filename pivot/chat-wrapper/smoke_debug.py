import base64
import io
import json
import urllib.request

BASE = "http://127.0.0.1:8153"

img = open(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\input-images\Emiru.webp", "rb").read()
b64 = base64.b64encode(img).decode()

# 1. make a chat
r = urllib.request.urlopen(urllib.request.Request(
    BASE + "/api/chat", data=json.dumps({"text": "debug session"}).encode(),
    headers={"Content-Type": "application/json"}))
chat = json.load(r)["chat_id"]
print("chat", chat, flush=True)


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


d = post("/api/debug/describe", {"chat_id": chat, "filename": "Emiru.webp", "image_b64": b64})
print("DESCRIBE:", d["assistant_text"][:300], flush=True)
print("TRACE-KEYS:", sorted(d["api_trace"].keys()), flush=True)

m = post("/api/debug/detect", {"chat_id": chat, "filename": "Emiru.webp",
                               "image_b64": b64, "concept": "person"})
print("DETECT:", m["assistant_text"], flush=True)
print("MASKS:", json.dumps(m["masks"], indent=1)[:800], flush=True)

# describe first mask
mk = post("/api/debug/describe-mask", {"chat_id": chat, "debug_id": m["debug_id"],
                                       "mask_id": 0})
print("MASK-DESC:", mk.get("assistant_text", mk)[:300], flush=True)
