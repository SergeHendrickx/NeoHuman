import json
import urllib.request

BASE = "http://127.0.0.1:8154"

r = urllib.request.urlopen(urllib.request.Request(
    BASE + "/api/chat", data=json.dumps({"text": "body-fit debug"}).encode(),
    headers={"Content-Type": "application/json"}))
chat = json.load(r)["chat_id"]
print("chat", chat, flush=True)


def post(path, body, timeout=1900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# Full local chain on the preset photo: pose -> maps -> 3DB -> refine -> joint.
d = post("/api/debug/body-fit", {
    "chat_id": chat, "filename": "dark_suit_pair.jpg",
    "image_url": "/api/runs/20261103-213035-3359/artifact/source"})
print("summary:", d["assistant_text"], flush=True)
print("stats:", json.dumps({k: v for k, v in d["stats"].items() if k != "beta"})[:400], flush=True)
print("beta_n:", d["stats"]["beta_n"], "verts:", d["stats"]["verts"], flush=True)
print("mesh_url:", d["mesh_url"], flush=True)

with urllib.request.urlopen(BASE + d["mesh_url"], timeout=60) as r:
    head = r.read(64)
print("mesh head:", head[:32], flush=True)

sheet = f"/api/chats/{d['chat_id']}/file/debug/{d['debug_id']}/{d['sheet']}"
with urllib.request.urlopen(BASE + sheet, timeout=60) as r:
    print("sheet:", r.status, r.headers.get("Content-Type"), flush=True)
