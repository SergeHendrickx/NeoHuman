import sys

sys.path.insert(0, ".")
from server import create_app

app = create_app()
client = app.test_client()
for path in (
    "/vendor/dockview-core/dist/package/main.esm.mjs",
    "/vendor/dockview/dist/package/main.esm.mjs",
    "/vendor/dockview/dist/styles/dockview.css",
    "/vendor/three/build/three.module.js",
    "/",
):
    res = client.get(path)
    print(f"{res.status_code} {res.content_type} {path} ({len(res.data)} bytes)")
