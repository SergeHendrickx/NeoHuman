from pathlib import Path

p = Path(r"A:\dev\GNM-V\pivot\sam3dbody-ingest\scripts\vggt_pose_strip.py")
t = p.read_text(encoding="utf-8")

old = '''V = len(PANELS)
worlds, masks = {}, {}
for i, n in enumerate(PANELS):
    H, W = depth[i].shape
    m = (conf[i] > np.percentile(conf[i], 40)) & np.isfinite(depth[i]) & (depth[i] > 0.05)
    ys, xs = np.nonzero(m)
    d = depth[i][ys, xs]
    Xc = np.stack([(xs - K[i, 0, 2]) / K[i, 0, 0] * d,
                   (ys - K[i, 1, 2]) / K[i, 1, 1] * d, d], 1)
    R, t = ext[i][:3, :3], ext[i][:3, 3]
    worlds[n] = (R.T @ (Xc - t[None]).T).T
    masks[n] = m
    print(n, "points", len(worlds[n]))'''
new = '''V = len(PANELS)
import run_all as R2

seg_model = R2.load_dense("seg", "seg/shutterstock_goliath/sapiens2_0.4b_seg_shutterstock_goliath-1024x768.py",
                          "seg/sapiens2_0.4b_seg.safetensors")
from sapiens.dense.visualizers import SegVisualizer
seg_vis = SegVisualizer(class_palette_type="dome29", with_labels=False)
pose_model = R2.load_pose()

worlds, masks, heads = {}, {}, {}
for i, n in enumerate(PANELS):
    bgr = cv2.imread(str(paths[i]))
    Hp, Wp = bgr.shape[:2]
    labels, _ = R2.seg_run(seg_model, seg_vis, bgr)
    person = cv2.resize((labels > 0).astype(np.uint8), (depth[i].shape[1], depth[i].shape[0]),
                        interpolation=cv2.INTER_NEAREST) > 0
    kp, _ = R2.pose_run(pose_model, bgr)
    face = kp[70:272][np.isfinite(kp[70:272]).all(1)]
    fc = face.mean(0)
    fr = 1.3 * max(np.ptp(face[:, 0]), np.ptp(face[:, 1]))
    ys_i, xs_i = np.mgrid[0:depth[i].shape[0], 0:depth[i].shape[1]]
    xs_p = xs_i * (Wp / depth[i].shape[1]); ys_p = ys_i * (Hp / depth[i].shape[0])
    head = person & (((xs_p - fc[0]) ** 2 + (ys_p - fc[1]) ** 2) < fr ** 2)
    m = person & (conf[i] > np.percentile(conf[i], 25)) & np.isfinite(depth[i]) & (depth[i] > 0.05)
    ys, xs = np.nonzero(m)
    d = depth[i][ys, xs]
    Xc = np.stack([(xs - K[i, 0, 2]) / K[i, 0, 0] * d,
                   (ys - K[i, 1, 2]) / K[i, 1, 1] * d, d], 1)
    Rm, tt = ext[i][:3, :3], ext[i][:3, 3]
    worlds[n] = (Rm.T @ (Xc - tt[None]).T).T
    masks[n] = m
    heads[n] = head[m]
    print(n, "person points", len(worlds[n]), "head points", int(heads[n].sum()))
del seg_model, seg_vis, pose_model; R2.free({})'''
assert old in t
t = t.replace(old, new)

old2 = '''allY = np.concatenate([worlds[n][:, 1] for n in PANELS])
lo, hi = np.percentile(allY, [1, 99])
head_sel = {}
for n in PANELS:
    y = worlds[n][:, 1]
    head_sel[n] = y < lo + 0.18 * (hi - lo)
'''
new2 = '''head_sel = heads
'''
assert old2 in t
t = t.replace(old2, new2)

p.write_text(t, encoding="utf-8")
print("vggt pose strip: person+head masks added")
