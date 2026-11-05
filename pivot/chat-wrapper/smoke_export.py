import json
import urllib.request

BASE = "http://127.0.0.1:8154"

r = urllib.request.urlopen(urllib.request.Request(
    BASE + "/api/chat", data=json.dumps({"text": "export debug"}).encode(),
    headers={"Content-Type": "application/json"}))
chat = json.load(r)["chat_id"]
print("chat", chat, flush=True)


def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# Package an existing fitted mesh (smoke-emiru run) into avatar_character.glb.
d = post("/api/debug/export", {
    "chat_id": chat,
    "mesh_url": "/api/runs/smoke-emiru/artifact/mesh"})
print("summary:", d["assistant_text"], flush=True)
print("files:", d["files"], flush=True)
print("stats:", json.dumps(d["stats"])[:300], flush=True)
print("glb_url:", d["glb_url"], flush=True)

with urllib.request.urlopen(BASE + d["glb_url"], timeout=60) as r:
    magic = r.read(4)
    print("glb magic:", magic, "status:", r.status, flush=True)
assert magic == b"glTF", "not a GLB file"
print("GLB OK", flush=True)
