"""Fetch the pinned three.js build the flow viewer needs (vendor/three).

The repo does not commit third-party JavaScript. Run this once after cloning:

    python pivot\\chat-wrapper\\flow-viewer\\fetch_vendor.py

Downloads three r186 (0.186.1) from jsDelivr into the paths the import map expects:
    vendor/three/three.module.js, three.core.js
    vendor/three/addons/controls/OrbitControls.js
    vendor/three/addons/loaders/{GLTFLoader,OBJLoader,PLYLoader}.js
    vendor/three/addons/environments/RoomEnvironment.js

If you are offline, copy an npm `three@0.186.1` package's build/ and examples/jsm/ here instead.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://cdn.jsdelivr.net/npm/three@0.186.1/"
FILES = {
    "build/three.module.js": "vendor/three/three.module.js",
    "build/three.core.js": "vendor/three/three.core.js",
    "examples/jsm/controls/OrbitControls.js": "vendor/three/addons/controls/OrbitControls.js",
    "examples/jsm/loaders/GLTFLoader.js": "vendor/three/addons/loaders/GLTFLoader.js",
    "examples/jsm/loaders/OBJLoader.js": "vendor/three/addons/loaders/OBJLoader.js",
    "examples/jsm/loaders/PLYLoader.js": "vendor/three/addons/loaders/PLYLoader.js",
    "examples/jsm/environments/RoomEnvironment.js": "vendor/three/addons/environments/RoomEnvironment.js",
}


def main() -> int:
    here = Path(__file__).resolve().parent
    for remote, local in FILES.items():
        dst = here / local
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > 0:
            print(f"keep  {local}")
            continue
        print(f"fetch {local} ...", end=" ", flush=True)
        try:
            with urllib.request.urlopen(BASE + remote, timeout=60) as r:
                dst.write_bytes(r.read())
            print("ok", dst.stat().st_size, "bytes")
        except Exception as e:
            print("FAILED:", e)
            return 1
    print("three.js r186 vendored into", here / "vendor" / "three")
    return 0


if __name__ == "__main__":
    sys.exit(main())
