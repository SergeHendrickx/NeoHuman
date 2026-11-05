import torch

MHR = r"A:\dev\GNM-V\pivot\sam-3d-body\sam-3d-body-dinov3\checkpoints\assets\mhr_model.pt"

mhr = torch.jit.load(MHR, map_location="cpu")
print("TYPE", type(mhr))
print("=== scripted code ===")
try:
    print(mhr.code)
except Exception as e:
    print("no code:", e)
print("=== submodules with code ===")
for name, m in mhr.named_modules():
    c = getattr(m, "code", None)
    if c is not None and name != "":
        print("---", name, "---")
        print(c)
print("=== schema ===")
for name, m in mhr.named_modules():
    if name == "":
        for meth in dir(m):
            if meth in ("forward", "forward_vertices", "forward_default"):
                f = getattr(m, meth, None)
                if hasattr(f, "schema"):
                    print(meth, ":", f.schema)

# units probe: zero pose, identity zero
N = 1
print("=== units probe ===")
try:
    out = mhr(torch.zeros(N, 45), torch.zeros(N, 204), torch.zeros(N, 72))
    print("n_return", len(out) if isinstance(out, (tuple, list)) else 1)
    for i, o in enumerate(out):
        print("out", i, type(o), tuple(o.shape) if hasattr(o, "shape") else o)
    verts = out[0]
    print("VVERTS", tuple(verts.shape))
    ext = verts.reshape(-1, 3).max(0).values - verts.reshape(-1, 3).min(0).values
    print("neutral_extent", [round(float(v), 3) for v in ext], "=> if ~170 the units are cm")
    print("min", [round(float(v), 2) for v in verts.reshape(-1,3).min(0).values],
          "max", [round(float(v), 2) for v in verts.reshape(-1,3).max(0).values])
except Exception as e:
    print("PROBE_ERROR", type(e).__name__, e)
    try:
        out = mhr(torch.zeros(N, 45), torch.zeros(N, 204), torch.zeros(N, 72), True)
        print("4-arg worked", type(out), len(out))
        print("VVERTS", tuple(out[0].shape))
    except Exception as e2:
        print("4ARG_ERROR", type(e2).__name__, e2)
