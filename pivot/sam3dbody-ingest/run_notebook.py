"""Execute a notebook headlessly, writing everything inside this folder.

    . .\\scripts\\env.ps1
    $env:JUPYTER_PATH = "A:\\dev\\GNM-V\\.cache\\jupyter\\share\\jupyter"   # kernel "gnmvenv"
    .\\.venv\\Scripts\\python.exe pivot\\sam3dbody-ingest\\run_notebook.py [notebook]

Outputs: outputs/notebook/exec/<notebook>.ipynb (with outputs); the notebook writes outputs/notebook/artifacts/ itself.
Only outputs/ is written; the notebook itself writes outputs/notebook/artifacts/ by default.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NB = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "sam3d_sapiens2.ipynb"
EXEC_DIR = HERE / "outputs" / "notebook" / "exec"

# Jupyter's secure_write() calls SetFileSecurity on the connection file. Under a restricted token
# (and inside some sandboxes) that raises WinError 5, so make it a no-op here; the connection file
# still goes to a workspace-local runtime dir below.
import jupyter_core.paths as _jcp  # noqa: E402

_jcp.win32_restrict_file_to_user = lambda fname: None


def main() -> int:
    jup = HERE / ".jupyter"
    for sub in ("runtime", "data", "config"):
        (jup / sub).mkdir(parents=True, exist_ok=True)
    (HERE / ".ipython").mkdir(exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(jup / "runtime")
    os.environ["JUPYTER_DATA_DIR"] = str(jup / "data")
    os.environ["JUPYTER_CONFIG_DIR"] = str(jup / "config")
    os.environ["IPYTHONDIR"] = str(HERE / ".ipython")
    os.environ.setdefault("GNM_REPO", str(REPO))
    os.environ.setdefault("SAM3DBODY_NOTEBOOK_OUT", str(HERE / "outputs" / "notebook" / "artifacts"))
    kernel_path = REPO / ".cache" / "jupyter" / "share" / "jupyter"
    if kernel_path.is_dir():
        os.environ.setdefault("JUPYTER_PATH", str(kernel_path))
    else:
        print(f"WARNING: private kernelspec dir not found: {kernel_path}", file=sys.stderr)

    nb = nbformat.read(str(NB), as_version=4)
    client = NotebookClient(
        nb, timeout=600, kernel_name="gnmvenv",
        allow_errors=False, resources={"metadata": {"path": str(HERE)}}, record_timing=True,
    )
    client.execute()

    EXEC_DIR.mkdir(exist_ok=True)
    out = EXEC_DIR / NB.name
    nbformat.write(nb, str(out))
    errors = sum(1 for c in nb.cells if c.cell_type == "code"
                 for o in c.get("outputs", []) if o.get("output_type") == "error")
    print(f"\nEXECUTED -> {out}  (cells={len(nb.cells)}, errors={errors})")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
