"""Slice 3: actor tab pane assembly. Anchor-based, assert-guarded, writes once."""
import re

P = "static/index.html"
lines = open(P, encoding="utf-8").read().splitlines(keepends=True)


def find(pat, start=0, end=None):
    rx = re.compile(pat)
    hits = [i for i in range(start, len(lines) if end is None else end) if rx.search(lines[i])]
    assert len(hits) == 1, f"expected 1 hit for {pat!r}, got {len(hits)}"
    return hits[0]


def first_after(pat, start):
    rx = re.compile(pat)
    return next(i for i in range(start, len(lines)) if rx.search(lines[i]))


def label_block(lid):
    li = find(f'id="{lid}"')
    lo = next(i for i in range(li, -1, -1) if "<label" in lines[i])
    lc = next(i for i in range(li, len(lines)) if lines[i].strip() == "</label>")
    return lo, lc


def div_depth_up(anchor_id, open_tag):
    li = find(f'id="{anchor_id}"')
    lo = next(i for i in range(li, -1, -1) if open_tag in lines[i])
    depth = 0
    for i in range(lo, len(lines)):
        depth += lines[i].count("<div") - lines[i].count("</div>")
        if depth == 0:
            return lo, i
    raise AssertionError(f"unbalanced div under {anchor_id}")


def single_line(lid):
    li = find(f'id="{lid}"')
    assert "</p>" in lines[li] or "/>" in lines[li] or "</div>" in lines[li], lines[li]
    return li, li


def grab(lo, lc):
    return lines[lo:lc + 1]


# --- world-pane toggles (moved into actor tabs) ---
sk_lo, sk_lc = label_block("skeleton-toggle")
skelBlock = grab(sk_lo, sk_lc)
ex_lo, ex_lc = label_block("expr-vectors-toggle")
exprBlock = grab(ex_lo, ex_lc)

# --- actor card regions ---
card_open = find(r'<section class="control-section" id="actor-card"')
params_summary = find(r"<summary>Parameters</summary>")
params_open = params_summary - 1
assert lines[params_open].strip() == "<details open>", lines[params_open]
model_open = find(r'aria-labelledby="model-settings-heading"')
model_open = None  # recomputed at splice time (Model section is unclosed; bound by </aside>)

# --- extract content blocks (indices valid until the big replace) ---
lod_lo, lod_lc = label_block("crowd-actor-lod")
lodBlock = grab(lod_lo, lod_lc)
scaleBlock = grab(*label_block("crowd-actor-scale"))
bodyBlock = grab(*label_block("body-color"))
wireBlock = grab(*label_block("wireframe-toggle"))
faceNote = grab(*single_line("actor-face-note"))
gazeYaw = grab(*label_block("gaze-yaw"))
gazePitch = grab(*label_block("gaze-pitch"))


def row_block(btn_id):
    li = find(f'id="{btn_id}"')
    lo = next(i for i in range(li, -1, -1) if '<div class="crowd-row"' in lines[i])
    lc = next(i for i in range(li, len(lines)) if lines[i].strip() == "</div>")
    return grab(lo, lc)


blinkRow = row_block("eyes-blink")
wanderRow = row_block("actor-gaze-wander")
autoBlock = grab(*div_depth_up("eyes-autoblink", '<div class="model-setting">'))
mouthBlock = grab(*label_block("mouth-open"))
teethBlocks, tongueBlocks = [], []
for lid in ("teeth-count", "teeth-crooked", "teeth-gap", "teeth-overbite", "teeth-fangs"):
    teethBlocks.extend(grab(*label_block(lid)))
for lid in ("tongue-out", "tongue-curl", "tongue-width", "tongue-roll"):
    tongueBlocks.extend(grab(*label_block(lid)))
bodyNote = grab(*single_line("actor-body-note"))
shirtBlock = grab(*div_depth_up("cloth-toggle", '<div class="model-setting">'))
thickBlock = grab(*label_block("cloth-thickness"))
fabricBlock = grab(*label_block("cloth-color"))
shortsBlock = grab(*div_depth_up("cloth-shorts-toggle", '<div class="model-setting">'))
shortsColor = grab(*label_block("cloth-shorts-color"))
refitRow = row_block("cloth-refit")
poseRow = row_block("actor-goto-pose")
behaviorBlock = grab(*label_block("crowd-actor-behavior"))
speedBlock = grab(*label_block("crowd-actor-speed"))
cadenceBlock = grab(*label_block("crowd-actor-cadence"))
stats_lo = find(r'<div class="crowd-actor-stats">')
stats_lc = next(i for i in range(stats_lo, len(lines)) if lines[i].strip() == "</div>")
statsBlock = grab(stats_lo, stats_lc)
mix_lo, mix_lc = div_depth_up("mixamo-play", '<div class="crowd-group crowd-group-mixamo">')
mixBlock = grab(mix_lo, mix_lc)
param_open = find(r'aria-labelledby="parameter-heading"')
param_open = next(i for i in range(param_open, -1, -1) if "<section" in lines[i])
param_close = next(i for i in range(param_open, len(lines)) if lines[i].strip() == "</section>")
paramBlock = grab(param_open, param_close)
assert 'id="param-component"' not in "".join(paramBlock)
paramBlock[0] = paramBlock[0].replace(
    '<section class="control-section" aria-labelledby="parameter-heading">',
    '<section class="control-section" id="param-component" aria-labelledby="parameter-heading">',
)
assert 'id="param-component"' in paramBlock[0]

I2, I3, I4, I5 = "    ", "      ", "        ", "          "
tabs = [("pose", "Pose", True, '<span id="pose-count"></span>'),
        ("identity", "Identity", False, '<span id="identity-count"></span>'),
        ("expression", "Expression", False, '<span id="expression-count"></span>'),
        ("eyes", "Eyes", False, ""), ("mouth", "Mouth", False, ""),
        ("materials", "Materials", False, ""), ("clothing", "Clothing", False, ""),
        ("animation", "Animation", False, ""), ("ai", "AI", False, "")]
new = []
new.append(f"{I2}<div class=\"tab-strip\" id=\"actor-tabs\" role=\"tablist\" aria-label=\"Actor sections\">\n")
for name, label, active, extra in tabs:
    cls = "tab is-active" if active else "tab"
    sel = "true" if active else "false"
    space = " " if extra else ""
    new.append(f"{I3}<button class=\"{cls}\" data-atab=\"{name}\" type=\"button\" role=\"tab\" aria-selected=\"{sel}\">{label}{space}{extra}</button>\n")
new.append(f"{I2}</div>\n")


def panel(name, blocks, active=False, mount=None):
    hid = "" if active else " hidden"
    new.append(f"{I2}<div class=\"tab-panel\" id=\"atab-{name}\" role=\"tabpanel\"{hid}>\n")
    new.append(f"{I3}<div class=\"details-body\">\n")
    if mount:
        new.append(f"{I4}<div class=\"param-mount\" id=\"param-mount-{mount}\"></div>\n")
    for b in blocks:
        new.extend(b)
    new.append(f"{I3}</div>\n")
    new.append(f"{I2}</div>\n")


panel("pose", [skelBlock], active=True, mount="pose")
panel("identity", [scaleBlock], mount="identity")
panel("expression", [exprBlock], mount="expression")
panel("eyes", [gazeYaw, gazePitch, blinkRow, wanderRow, autoBlock,
               grab(*label_block("eyes-pupil")), grab(*label_block("eyes-iris"))])
panel("mouth", [faceNote, mouthBlock, teethBlocks, tongueBlocks])
panel("materials", [bodyBlock, wireBlock])
panel("clothing", [bodyNote, shirtBlock, thickBlock, fabricBlock, shortsBlock, shortsColor, refitRow])
panel("animation", [mixBlock])
panel("ai", [poseRow, behaviorBlock, speedBlock, cadenceBlock, statsBlock])

# --- splice ---
out = list(lines)
# 1. delete world toggles (descending) + replace actor region + delete model section
for lo, lc in sorted([(sk_lo, sk_lc), (ex_lo, ex_lc)], reverse=True):
    del out[lo:lc + 1]
# recompute (world deletions shift actor indices)
card_open2 = next(i for i, l in enumerate(out) if 'id="actor-card"' in l)
psum2 = next(i for i, l in enumerate(out) if "<summary>Parameters</summary>" in l)
popen2 = psum2 - 1
assert out[popen2].strip() == "<details open>"
mopen2 = next(i for i, l in enumerate(out) if 'aria-labelledby="model-settings-heading"' in l)
mclose2 = next(i for i in range(mopen2, len(out)) if re.match(r"^    </aside>$", out[i]))
out[popen2:mclose2] = new
# drop the param component into the pose mount
pmount = next(i for i, l in enumerate(out) if 'id="param-mount-pose"' in l)
out[pmount:pmount + 1] = [out[pmount].replace(
    '<div class="param-mount" id="param-mount-pose"></div>',
    '<div class="param-mount" id="param-mount-pose">\n' + "".join(paramBlock) + I4 + "</div>",
)]
# LOD label into the Focus row (before its close)
focus_li = next(i for i, l in enumerate(out) if 'id="crowd-sim-focus"' in l)
row_close = next(i for i in range(focus_li, len(out)) if out[i].strip() == "</div>")
out[row_close:row_close] = lodBlock

final = "".join(out)
for probe in ['id="actor-tabs"', 'id="param-component"', 'id="atab-pose"', 'id="atab-ai"',
              'id="param-mount-pose"', 'id="param-mount-identity"', 'id="param-mount-expression"',
              'id="skeleton-toggle"', 'id="expr-vectors-toggle"', 'id="crowd-actor-lod"',
              'id="pose-count"', 'id="mixamo-play"', 'id="body-color"', 'id="wireframe-toggle"']:
    assert final.count(probe) == 1, (probe, final.count(probe))
for gone in ["<summary>Parameters</summary>", "<summary>Identity</summary>", "<summary>Face</summary>",
             "<summary>Body</summary>", "<summary>Motion</summary>", "<summary>Animation</summary>",
             "model-settings-heading", "group-tabs", 'data-group="pose"']:
    assert gone not in final, f"should be gone: {gone}"
open(P, "w", encoding="utf-8").write(final)
print("slice3 splice ok,", len(out), "lines")
