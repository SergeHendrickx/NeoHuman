"""Bake a face texture for the 6-view head set onto the MHR UV layout.

For each Muse panel: fit MHR, align head to view 0 (Procrustes), back-transform the canonical mesh into that
view's camera, sample image colour with visibility checks, blend by view angle, then rasterise into MHR UV.
"""
import json
import sys
from pathlib import Path

REPO = Path(r"A:\dev\GNM-V")
BASE = REPO / "pivot" / "sam3dbody-ingest"
sys.path.insert(0, str(BASE))

import cv2
import numpy as np
import trimesh
from trimesh.visual.material import SimpleMaterial
from trimesh.visual import TextureVisuals

import run_all as R

STEM = sys.argv[1] if len(sys.argv) > 1 else "teenage-woman-long-wavy-white-gray__01971-2947889296"
JOB = BASE / "outputs" / STEM
OUT = JOB / "threedb"
PANELS = ["muse_face_0.png", "muse_face_m45.png", "muse_face_45.png",
          "muse_face_m90.png", "muse_face_90.png", "muse_face_180.png"]
YAWS = [0.0, -45.0, 45.0, -90.0, 90.0, 180.0]
TEX = 2048

# ---------- UV data from the model ----------
model3d, est, faces = R.load_3db()
hp = model3d.head_pose
mesh_uv = hp.mhr.character_torch.mesh
texcoords = mesh_uv.texcoords.detach().cpu().numpy().astype(np.float32)
texcoord_faces = mesh_uv.texcoord_faces.detach().cpu().numpy().astype(np.int32)
print("texcoords", texcoords.shape, "uv faces", texcoord_faces.shape,
      "range", texcoords.min(0), texcoords.max(0))

# ---------- fit all panels ----------
pose_model = R.load_pose()


def procrustes(A, B):
    Ac, Bc = A.mean(0), B.mean(0)
    A0, B0 = A - Ac, B - Bc
    U, S, Vt = np.linalg.svd(B0.T @ A0)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    s = S.sum() / (A0 ** 2).sum()
    return s, Rm, Bc - s * (Rm @ Ac)


fits = {}
ref_lm = None
for name in PANELS:
    bgr = cv2.imread(str(OUT / name))
    h, w = bgr.shape[:2]
    kp_any, _ = R.pose_run(pose_model, bgr)
    bbox = R.person_box(kp_any, w, h)
    kp, score = R.pose_run(pose_model, bgr, bbox)
    out, verts, f = R.fit_3db(est, bgr, bbox)
    cam = np.asarray(out["pred_cam_t"], np.float64)
    ref = R.identity_refine(model3d, out, kp, score, w, h, f, cam)
    beta = np.asarray(ref.get("beta", out["shape_params"]), np.float64)
    joint = R._solve_joint(model3d, out, kp, score, w, h, f, cam, beta_init=beta, identity_only=ref,
                           iters=200, faces=faces, normal_weight=0.0)
    beta = np.asarray(joint.get("shape_params", beta))
    params = np.asarray(joint.get("mhr_model_params", out["mhr_model_params"]))
    expr = np.asarray(joint.get("expr_params", np.zeros(72)))
    camr = np.asarray(joint.get("cam_t", cam))
    vroot, lm = R.mesh_and_landmarks(model3d, {"mhr_model_params": params}, beta, model_params=params, expr=expr)
    verts_cam = vroot + camr[None]
    lm_cam = lm + camr[None]
    L = lm_cam[70:272]
    if ref_lm is None:
        ref_lm = L
        s, Rm, t = 1.0, np.eye(3), np.zeros(3)
    else:
        s, Rm, t = procrustes(L, ref_lm)
    fits[name] = {"verts": verts_cam, "f": float(f), "w": w, "h": h, "s": s, "R": Rm, "t": t,
                  "img": bgr}
    print("fit", name, "scale", round(float(s), 4), flush=True)
del pose_model, model3d; R.free({})

# canonical mesh = view-0 fitted mesh; UV belongs to this topology
canon = fits[PANELS[0]]["verts"]
V = len(canon)
normals = trimesh.Trimesh(vertices=canon, faces=faces, process=False).vertex_normals

acc = np.zeros((V, 3), np.float64)
wsum = np.zeros(V, np.float64)


def zbuffer(verts, faces_arr, f, W, H):
    z = verts[:, 2]
    ok = z > 1e-6
    u = f * verts[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = f * verts[:, 1] / np.maximum(z, 1e-6) + H / 2
    zb = np.full((H, W), np.inf, np.float32)
    for tri in faces_arr:
        if not ok[tri].all():
            continue
        ax, ay, az = u[tri[0]], v[tri[0]], z[tri[0]]
        bx, by, bz = u[tri[1]], v[tri[1]], z[tri[1]]
        cx, cy, cz = u[tri[2]], v[tri[2]], z[tri[2]]
        x1 = max(0, int(np.floor(min(ax, bx, cx)))); x2 = min(W - 1, int(np.ceil(max(ax, bx, cx))))
        y1 = max(0, int(np.floor(min(ay, by, cy)))); y2 = min(H - 1, int(np.ceil(max(ay, by, cy))))
        if x2 < x1 or y2 < y1:
            continue
        det = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if abs(det) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(x1, x2 + 1) + 0.5, np.arange(y1, y2 + 1) + 0.5)
        w1 = ((gx - ax) * (cy - ay) - (cx - ax) * (gy - ay)) / det
        w2 = ((bx - ax) * (gy - ay) - (gx - ax) * (by - ay)) / det
        w0 = 1.0 - w1 - w2
        ins = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not ins.any():
            continue
        rr, cc = np.nonzero(ins)
        zz = (w0[ins] * az + w1[ins] * bz + w2[ins] * cz).astype(np.float32)
        cur = zb[rr + y1, cc + x1]
        upd = zz < cur
        zb[rr[upd] + y1, cc[upd] + x1] = zz[upd]
    return zb


for name in PANELS:
    vw = fits[name]
    s, Rm, t = vw["s"], vw["R"], vw["t"]
    Vi = ((canon - t[None]) @ Rm) / s                    # canonical -> this view's camera frame
    f, W, H = vw["f"], vw["w"], vw["h"]
    zb = zbuffer(Vi, faces, f, W, H)
    z = Vi[:, 2]
    u = f * Vi[:, 0] / np.maximum(z, 1e-6) + W / 2
    v = f * Vi[:, 1] / np.maximum(z, 1e-6) + H / 2
    inside = (u > 1) & (u < W - 2) & (v > 1) & (v < H - 2)
    ui = np.clip(u.astype(int), 0, W - 1); vi_ = np.clip(v.astype(int), 0, H - 1)
    visible = inside & (np.abs(z - zb[vi_, ui]) < 0.01)
    # back-face culling in this view
    view_dir = -Vi / np.maximum(np.linalg.norm(Vi, axis=1, keepdims=True), 1e-9)
    facing = (np.einsum("ij,ij->i", normals, view_dir) > 0.15)
    sel = np.where(visible & facing)[0]
    # bilinear sample
    img = vw["img"].astype(np.float32)
    xs = u[sel]; ys = v[sel]
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    wx = (xs - x0)[:, None]; wy = (ys - y0)[:, None]
    c = (img[y0, x0] * (1 - wx) * (1 - wy) + img[y0, x1] * wx * (1 - wy)
         + img[y1, x0] * (1 - wx) * wy + img[y1, x1] * wx * wy)
    weight = np.clip(np.einsum("ij,ij->i", normals[sel], view_dir[sel]), 0.0, 1.0) ** 2 + 1e-3
    acc[sel] += c * weight[:, None]
    wsum[sel] += weight
    print(name, "visible verts", len(sel), flush=True)

vcol = np.zeros((V, 3), np.float64)
ok = wsum > 0
vcol[ok] = acc[ok] / wsum[ok, None]
np.savez_compressed(OUT / "texture_bake_data.npz", vcol=vcol, texcoords=texcoords,
                    texcoord_faces=texcoord_faces, faces=faces, canon=canon)
print("vertices with colour:", int(ok.sum()), "/", V, "-> texture_bake_data.npz")

# ---------- UV rasterisation ----------
tex = np.zeros((TEX, TEX, 3), np.float32)
mask = np.zeros((TEX, TEX), np.float32)
uvp = texcoords * (TEX - 1)                     # uv in [0,1]?
uvp = np.clip(uvp, 0, TEX - 1)
for fi in range(len(texcoord_faces)):
    tri = texcoord_faces[fi]
    vt = uvp[tri]
    x1 = max(0, int(np.floor(vt[:, 0].min()))); x2 = min(TEX - 1, int(np.ceil(vt[:, 0].max())))
    y1 = max(0, int(np.floor(vt[:, 1].min()))); y2 = min(TEX - 1, int(np.ceil(vt[:, 1].max())))
    if x2 < x1 or y2 < y1:
        continue
    det = (vt[1, 0] - vt[0, 0]) * (vt[2, 1] - vt[0, 1]) - (vt[2, 0] - vt[0, 0]) * (vt[1, 1] - vt[0, 1])
    if abs(det) < 1e-9:
        continue
    gx, gy = np.meshgrid(np.arange(x1, x2 + 1) + 0.5, np.arange(y1, y2 + 1) + 0.5)
    w1 = ((gx - vt[0, 0]) * (vt[2, 1] - vt[0, 1]) - (vt[2, 0] - vt[0, 0]) * (gy - vt[0, 1])) / det
    w2 = ((vt[1, 0] - vt[0, 0]) * (gy - vt[0, 1]) - (gx - vt[0, 0]) * (vt[1, 1] - vt[0, 1])) / det
    w0 = 1 - w1 - w2
    ins = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
    if not ins.any():
        continue
    face_v = faces[fi]                      # uv faces and mesh faces share row order
    cols = w0[ins, None] * vcol[face_v[0]][None] + w1[ins, None] * vcol[face_v[1]][None] + w2[ins, None] * vcol[face_v[2]][None]
    rr, cc = np.nonzero(ins)
    tex[rr + y1, cc + x1] = cols
    mask[rr + y1, cc + x1] = 1

texture = np.clip(tex, 0, 255).astype(np.uint8)
# flip y for image convention
texture = texture[::-1]  # image row order for UV (v up)
cv2.imwrite(str(OUT / "mhr_face_texture_2048.png"), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
print("wrote", OUT / "mhr_face_texture_2048.png", "coverage", float(mask.mean()))

# ---------- OBJ + MTL with proper UV indices ----------
obj_path = OUT / "mesh_face_textured.obj"
mtl_path = OUT / "mesh_face_textured.mtl"
with open(obj_path, "w", encoding="utf-8") as fh:
    fh.write("mtllib mesh_face_textured.mtl\n")
    fh.write("usemtl face_texture\n")
    for p in canon:
        fh.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    for uv in texcoords:
        fh.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
    for fi in range(len(faces)):
        a, b, c = faces[fi] + 1
        ta, tb, tc = texcoord_faces[fi] + 1
        fh.write(f"f {a}/{ta} {b}/{tb} {c}/{tc}\n")
with open(mtl_path, "w", encoding="utf-8") as fh:
    fh.write("newmtl face_texture\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nd 1\nillum 1\nmap_Kd mhr_face_texture_2048.png\n")
print("wrote", obj_path, mtl_path)
