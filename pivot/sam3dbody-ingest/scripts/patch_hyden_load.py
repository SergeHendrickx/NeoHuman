from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\hyden_metric_test.py")
t = p.read_text(encoding="utf-8")
old = '''model = HyDenMoGe(**MODEL_CONFIGS["vitl_dinov2"])
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(sd); del sd
model = model.to("cuda").eval()'''
new = '''cfg = dict(MODEL_CONFIGS["vitl_dinov2"])
cfg.pop("normal_head", None)          # this checkpoint is the metric-point variant (no normal head)
model = HyDenMoGe(**cfg)
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
res = model.load_state_dict(sd, strict=False)
print("missing keys:", len(res.missing_keys), "unexpected keys:", len(res.unexpected_keys))
if res.missing_keys:
    print("  missing sample:", res.missing_keys[:5])
if res.unexpected_keys:
    print("  unexpected sample:", res.unexpected_keys[:5])
del sd
model = model.to("cuda").eval()'''
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("HyDen load patched")
