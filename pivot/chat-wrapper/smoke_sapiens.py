import base64
import json
import urllib.request

BASE = "http://127.0.0.1:8154"

img = open(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\input-images\Emiru.webp", "rb").read()
b64 = base64.b64encode(img).decode()

r = urllib.request.urlopen(urllib.request.Request(
    BASE + "/api/chat", data=json.dumps({"text": "sapiens debug"}).encode(),
    headers={"Content-Type": "application/json"}))
chat = json.load(r)["chat_id"]
print("chat", chat, flush=True)


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


for kind in ("sapiens-pose", "sapiens-seg", "sapiens-normal", "sapiens-depth"):
    d = post("/api/debug/" + kind, {"chat_id": chat, "filename": "Emiru.webp", "image_b64": b64})
    print(kind, "->", d["assistant_text"], flush=True)
    print("  files:", d["files"], flush=True)
    print("  stats:", json.dumps(d.get("stats", {}))[:200], flush=True)

r = urllib.request.urlopen(BASE + "/api/chats/" + chat)
msgs = json.load(r)["messages"]
print("chat messages:", len(msgs), flush=True)
