"""Parity verification between client-side GPU/CPU algorithms and PyTorch server ground truth."""

import gzip
import struct
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial.transform import Rotation
import torch

from server import create_app


@pytest.fixture(scope="module")
def app_and_engine():
    app = create_app()
    engine = app.config["MHR_ENGINE"]
    return app, engine


def test_rig_mapping_joint_transforms_parity(app_and_engine):
    app, engine = app_and_engine
    client = app.test_client()

    mapping_res = client.get("/api/rig-mapping")
    assert mapping_res.status_code == 200
    data = mapping_res.data
    rows, cols, joints = struct.unpack_from("<III", data, 0)
    tp_len = rows * cols
    offsets_len = joints * 3
    prerot_len = joints * 4

    floats = np.frombuffer(data[12:], dtype=np.float32)
    tp = floats[:tp_len].reshape(rows, cols)

    # Set random pose
    np.random.seed(42)
    theta_p = np.random.uniform(-0.3, 0.3, size=cols).astype(np.float32)
    with engine.lock:
        engine.pose_values[:] = theta_p
        engine.snap_to_ground = False
        snap = engine._infer()

    theta_j = tp @ theta_p
    m = engine.model
    cat_params = torch.cat([torch.from_numpy(theta_p[np.newaxis, :]), torch.zeros(1, 45)], dim=1)
    jp_torch = m.character_torch.model_parameters_to_joint_parameters(cat_params).detach().cpu().numpy()[0]

    max_diff = float(np.abs(theta_j - jp_torch).max())
    assert max_diff < 1e-4, f"Joint parameters mismatch: {max_diff}"


def test_blendshape_deltas_parity(app_and_engine):
    app, engine = app_and_engine
    client = app.test_client()

    bs_res = client.get("/api/weights/blendshapes")
    assert bs_res.status_code == 200
    data = bs_res.data
    v_count, id_count, fe_count = struct.unpack_from("<III", data, 0)
    total_bs = id_count + fe_count
    floats = np.frombuffer(data[12:], dtype=np.float32)

    neutral_v4 = floats[: v_count * 4].reshape(v_count, 4)
    deltas_v4 = floats[v_count * 4 : v_count * 4 + v_count * total_bs * 4].reshape(
        v_count, total_bs, 4
    )

    np.random.seed(123)
    id_weights = np.random.uniform(-1.0, 1.0, size=id_count).astype(np.float32)
    fe_weights = np.random.uniform(-0.5, 0.5, size=fe_count).astype(np.float32)
    weights = np.concatenate([id_weights, fe_weights])

    with engine.lock:
        engine.identity_values[:] = id_weights
        engine.expression_values[:] = fe_weights
        server_unposed = engine.unposed_vertices()

    calc_unposed = neutral_v4[:, :3].copy()
    for b in range(total_bs):
        w = float(weights[b])
        if abs(w) > 1e-6:
            calc_unposed += deltas_v4[:, b, :3] * w

    max_diff = float(np.abs(server_unposed - calc_unposed).max())
    assert max_diff < 1e-5, f"Blendshape calculation mismatch: {max_diff}"


def test_pose_correctives_parity(app_and_engine):
    app, engine = app_and_engine
    client = app.test_client()

    corr_res = client.get("/api/weights/pose-correctives")
    assert corr_res.status_code == 200
    data = corr_res.data
    magic, l0_nnz, l0_in, l0_out, l2_rows, l2_cols, l2_nnz = struct.unpack_from(
        "<IIIIIII", data, 0
    )
    assert magic == 0x4350484D

    offset = 28
    l0_rows = np.frombuffer(data[offset : offset + l0_nnz * 2], dtype=np.uint16)
    offset += l0_nnz * 2
    l0_cols = np.frombuffer(data[offset : offset + l0_nnz * 2], dtype=np.uint16)
    offset += l0_nnz * 2
    l0_weights = np.frombuffer(data[offset : offset + l0_nnz * 4], dtype=np.float32)
    offset += l0_nnz * 4

    l2_indptr = np.frombuffer(data[offset : offset + (l2_rows + 1) * 4], dtype=np.uint32)
    offset += (l2_rows + 1) * 4
    l2_indices = np.frombuffer(data[offset : offset + l2_nnz * 2], dtype=np.uint16)
    offset += l2_nnz * 2
    l2_data = np.frombuffer(data[offset : offset + l2_nnz * 4], dtype=np.float32)

    # Test random joint parameters
    np.random.seed(999)
    jp = np.random.uniform(-0.4, 0.4, size=889).astype(np.float32)

    # 1. 6D features
    jp_mat = jp.reshape(127, 7)
    feats = np.zeros(750, dtype=np.float32)
    for j in range(2, 127):
        rx, ry, rz = float(jp_mat[j, 3]), float(jp_mat[j, 4]), float(jp_mat[j, 5])
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        off = (j - 2) * 6
        feats[off + 0] = cy * cz - 1.0
        feats[off + 1] = cy * sz
        feats[off + 2] = -sy
        feats[off + 3] = sx * sy * cz - cx * sz
        feats[off + 4] = sx * sy * sz + cx * cz - 1.0
        feats[off + 5] = sx * cy

    # 2. Layer 0 SparseLinear
    hidden = np.zeros(3000, dtype=np.float32)
    for k in range(l0_nnz):
        hidden[l0_rows[k]] += l0_weights[k] * feats[l0_cols[k]]

    # 3. Layer 1 ReLU
    hidden = np.maximum(hidden, 0.0)

    # 4. Layer 2 Linear CSR SpMV (in meters: divide by 100)
    deltas = np.zeros(l2_rows, dtype=np.float32)
    for r in range(l2_rows):
        start = l2_indptr[r]
        end = l2_indptr[r + 1]
        s = 0.0
        for idx in range(start, end):
            s += l2_data[idx] * hidden[l2_indices[idx]]
        deltas[r] = s / 100.0

    # Compare with Torch model
    m = engine.model
    with torch.inference_mode():
        torch_deltas = m.pose_correctives_model(torch.from_numpy(jp[np.newaxis, :])).numpy()[0] / 100.0

    max_diff = float(np.abs(deltas.reshape(18439, 3) - torch_deltas).max())
    assert max_diff < 1e-5, f"Pose correctives calculation mismatch: {max_diff}"


def test_client_bone_fk_parity(app_and_engine):
    """Client bone composition must match joint_parameters_to_skeleton_state.

    The client composes each local joint transform as
    T = Translate(offset + t) * Quat(pre) * EulerZYX(rx, ry, rz) * Scale(2**d)
    and chains it through the joint hierarchy. Any convention drift (Euler
    order, quaternion order, scale formula) shows up here as cm-level errors.
    """
    app, engine = app_and_engine
    model = engine.model
    skel = model.character_torch.skeleton
    offsets = skel.joint_translation_offsets.detach().cpu().numpy()
    prerot = skel.joint_prerotations.detach().cpu().numpy()
    parents = np.asarray(engine.joint_parents, dtype=int)
    tp = (
        model.get_parameter_transform()[: 127 * 7, : len(engine.pose_names)]
        .detach()
        .cpu()
        .numpy()
    )

    rng = np.random.RandomState(7)
    for _ in range(3):
        theta_p = rng.uniform(
            -0.5, 0.5, size=len(engine.pose_names)
        ).astype(np.float32)
        theta_j = (tp @ theta_p).reshape(127, 7)
        with torch.inference_mode():
            skeleton = model.character_torch.joint_parameters_to_skeleton_state(
                torch.from_numpy(theta_j.reshape(-1)[np.newaxis, :])
            ).detach().cpu().numpy()[0]

        local = np.zeros((127, 4, 4))
        for j in range(127):
            tx, ty, tz, rx, ry, rz, sd = theta_j[j]
            euler = (
                Rotation.from_euler("Z", rz).as_matrix()
                @ Rotation.from_euler("Y", ry).as_matrix()
                @ Rotation.from_euler("X", rx).as_matrix()
            )
            matrix = np.eye(4)
            matrix[:3, :3] = (
                Rotation.from_quat(prerot[j]).as_matrix() @ euler * (2.0**sd)
            )
            matrix[:3, 3] = offsets[j] + [tx, ty, tz]
            local[j] = matrix
        global_mats = np.zeros_like(local)
        for j in range(127):
            parent = parents[j]
            global_mats[j] = local[j] if parent < 0 else global_mats[parent] @ local[j]

        max_diff = float(np.abs(global_mats[:, :3, 3] - skeleton[:, :3]).max())
        assert max_diff < 1e-3, f"Client bone FK mismatch: {max_diff} cm"


def test_face_expression_name_ordering(app_and_engine):
    """FACE_EXPRESSION_NAMES[i] must match face_expr_coeffs[:, i].

    Checks a spread of action units by verifying each displacement field
    concentrates at its anatomical landmark. Guards the expression labels
    and region grouping against asset reorderings.
    """
    app, engine = app_and_engine
    assert len(engine.expression_names) == 72
    index_of = {name: i for i, name in enumerate(engine.expression_names)}

    with torch.inference_mode():
        _, skeleton = engine.model(
            identity_coeffs=torch.zeros(1, engine.identity_count),
            model_parameters=torch.zeros(1, len(engine.pose_names)),
            face_expr_coeffs=torch.zeros(1, 72),
            apply_correctives=False,
        )
    joints = skeleton[0].detach().cpu().numpy()[:, :3] / 100.0
    joint_index = {name: i for i, name in enumerate(engine.joint_names)}
    left_eye = joints[joint_index["l_eye"]]
    right_eye = joints[joint_index["r_eye"]]
    jaw = joints[joint_index["c_jaw"]]
    left_sign = float(np.sign(left_eye[0] - right_eye[0]))
    mouth = (left_eye + right_eye) / 2 + (jaw - (left_eye + right_eye) / 2) * 0.7

    fe = engine.model.face_expressions_model.shape_vectors.detach().cpu().numpy()
    base = (
        engine.model.character_torch.blend_shape.base_shape.detach()
        .cpu()
        .numpy()
        / 100.0
    )
    energy = (fe.astype(np.float64) ** 2).sum(axis=2)

    def centroid(k):
        weights = energy[k]
        top = np.argsort(weights)[::-1][: max(1, int(len(weights) * 0.05))]
        w = weights[top]
        return (w[:, None] * base[top]).sum(axis=0) / w.sum()

    def is_left(point):
        return np.sign(point[0]) == left_sign or abs(point[0]) < 0.005

    def near(point, target, radius):
        return float(np.linalg.norm(point - target)) < radius

    def near_mouth(point, dx, dy):
        return abs(point[0]) < dx and float(
            np.linalg.norm(point[:2] - mouth[:2])
        ) < dy

    checks = [
        ("browLowerer_L", lambda p: is_left(p) and p[1] > left_eye[1]),
        ("eyesClosed_L", lambda p: near(p, left_eye, 0.035)),
        ("eyesClosed_R", lambda p: near(p, right_eye, 0.035)),
        ("cheekRaiser_L", lambda p: is_left(p) and mouth[1] < p[1] < left_eye[1]),
        (
            "noseWrinkler_L",
            lambda p: abs(p[0]) < 0.03 and mouth[1] < p[1] < left_eye[1],
        ),
        ("lipCornerPuller_L", lambda p: p[0] * left_sign > 0.01),
        ("mouthLeft", lambda p: near_mouth(p, 0.02, 0.04)),
        ("jawDrop", lambda p: abs(p[0]) < 0.03 and p[1] < mouth[1] + 0.01),
    ]
    for want, predicate in checks:
        assert predicate(centroid(index_of[want])), f"ordering drift: {want}"
