from __future__ import annotations

from io import BytesIO
import gzip
import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest
from pxr import Sdf, Usd, UsdGeom, UsdSkel
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import server as server_module  # noqa: E402
from server import (
    DEFAULT_MODEL_PATH,
    EXPRESSION_ZONES,
    IDENTITY_ZONES,
    SUPPORTED_LODS,
    create_app,
    identity_zone_for_joint,
    joint_group_key,
)  # noqa: E402


def _parse_glb(payload: bytes) -> tuple[dict[str, object], bytes]:
    magic, version, total_length = struct.unpack_from("<4sII", payload, 0)
    assert magic == b"glTF"
    assert version == 2
    assert total_length == len(payload)
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    json_start = 20
    document = json.loads(
        payload[json_start : json_start + json_length].decode("utf-8")
    )
    binary_header = json_start + json_length
    binary_length, binary_type = struct.unpack_from(
        "<II",
        payload,
        binary_header,
    )
    assert binary_type == 0x004E4942
    binary_start = binary_header + 8
    return document, payload[binary_start : binary_start + binary_length]


def _glb_accessor(
    document: dict[str, object],
    binary: bytes,
    accessor_index: int,
) -> np.ndarray:
    accessor = document["accessors"][accessor_index]
    view = document["bufferViews"][accessor["bufferView"]]
    component_dtypes = {
        5121: np.dtype("<u1"),
        5123: np.dtype("<u2"),
        5125: np.dtype("<u4"),
        5126: np.dtype("<f4"),
    }
    component_counts = {
        "SCALAR": 1,
        "VEC2": 2,
        "VEC3": 3,
        "VEC4": 4,
        "MAT4": 16,
    }
    dtype = component_dtypes[accessor["componentType"]]
    components = component_counts[accessor["type"]]
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    values = np.frombuffer(
        binary,
        dtype=dtype,
        count=int(accessor["count"]) * components,
        offset=offset,
    ).reshape(int(accessor["count"]), components)
    if accessor["type"] == "MAT4":
        return values.reshape(-1, 4, 4).transpose(0, 2, 1)
    return values


@pytest.fixture(scope="module")
def client():
    app = create_app(DEFAULT_MODEL_PATH)
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def test_health_and_metadata(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json["status"] == "ok"
    assert health.json["vertices"] == 18439
    assert health.json["faces"] == 36750

    metadata = client.get("/api/metadata")
    assert metadata.status_code == 200
    assert len(metadata.json["groups"]["pose"]) == 204
    assert len(metadata.json["groups"]["identity"]) == 45
    assert len(metadata.json["groups"]["expression"]) == 72
    assert metadata.json["model"]["jointCount"] == 127
    assert metadata.json["model"]["exportFormats"] == ["obj", "glb", "usd"]
    assert metadata.json["model"]["riggedExportFormats"] == ["glb", "usd"]
    assert metadata.json["model"]["snapToGround"] is True
    root_ty = next(
        item for item in metadata.json["groups"]["pose"] if item["name"] == "root_ty"
    )
    assert root_ty["managed"] is True


def test_topology_and_geometry_payload_sizes(client):
    metadata = client.get("/api/metadata").json
    topology = client.get("/api/topology")
    geometry = client.get("/api/geometry")

    expected_topology_bytes = metadata["model"]["faceCount"] * 3 * 4
    expected_geometry_bytes = (
        (metadata["model"]["vertexCount"] + metadata["model"]["jointCount"]) * 3 * 4
    )

    assert topology.status_code == 200
    assert len(topology.data) == expected_topology_bytes
    assert geometry.status_code == 200
    assert len(geometry.data) == expected_geometry_bytes


def test_pose_deformation_changes_vertices(client):
    metadata = client.get("/api/metadata").json
    knee = next(
        item for item in metadata["groups"]["pose"] if item["name"] == "l_knee_bend"
    )
    neutral = np.frombuffer(client.get("/api/geometry").data, dtype="<f4").copy()

    response = client.post(
        "/api/deform",
        json={
            "action": "set",
            "group": "pose",
            "index": knee["index"],
            "value": 1.0,
        },
    )
    deformed = np.frombuffer(response.data, dtype="<f4")
    vertex_floats = metadata["model"]["vertexCount"] * 3

    assert response.status_code == 200
    assert float(np.abs(deformed[:vertex_floats] - neutral[:vertex_floats]).max()) > 0
    assert float(response.headers["X-MHR-Inference-Ms"]) >= 0


@pytest.mark.parametrize(
    ("group", "index", "value"),
    [
        ("identity", 0, 1.5),
        ("expression", 0, 0.5),
    ],
)
def test_identity_and_expression_deformation(client, group, index, value):
    response = client.post(
        "/api/deform",
        json={
            "action": "set",
            "group": group,
            "index": index,
            "value": value,
        },
    )
    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"


def test_reset_all_and_validation(client):
    reset = client.post(
        "/api/deform",
        json={"action": "reset_all", "group": "pose", "index": 0},
    )
    assert reset.status_code == 200

    invalid_group = client.post(
        "/api/deform",
        json={"action": "set", "group": "unknown", "index": 0, "value": 0},
    )
    assert invalid_group.status_code == 400

    invalid_index = client.post(
        "/api/deform",
        json={"action": "set", "group": "pose", "index": 999, "value": 0},
    )
    assert invalid_index.status_code == 400


def test_deform_set_group_writes_vector(client):
    metadata = client.get("/api/metadata").json
    count = len(metadata["groups"]["identity"])
    response = client.post(
        "/api/deform",
        json={
            "action": "set_group",
            "group": "identity",
            "index": 0,
            "values": [2.0] + [0.0] * (count - 1),
        },
    )
    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"
    zeroed = client.post(
        "/api/deform",
        json={
            "action": "set_group",
            "group": "identity",
            "index": 0,
            "values": [0.0] * count,
        },
    )
    reset = client.post(
        "/api/deform",
        json={"action": "reset_group", "group": "identity", "index": 0},
    )
    assert response.data != zeroed.data
    repeat = client.post(
        "/api/deform",
        json={
            "action": "set_group",
            "group": "identity",
            "index": 0,
            "values": [0.0] * count,
        },
    )
    a = np.frombuffer(repeat.data, dtype="<f4")
    b = np.frombuffer(zeroed.data, dtype="<f4")
    assert a.shape == b.shape and len(a) > 0
    # Snap-to-ground feeds back into pose state, so consecutive identical
    # writes converge instead of matching bit-for-bit.
    assert bool((abs(a - b) < 1e-3).all())
    bad = client.post(
        "/api/deform",
        json={"action": "set_group", "group": "identity", "index": 0, "values": [1.0]},
    )
    assert bad.status_code == 400


def test_pose_correctives_can_be_toggled(client):
    client.post(
        "/api/deform",
        json={"action": "reset_all", "group": "pose", "index": 0},
    )
    metadata = client.get("/api/metadata").json
    head_lean = next(
        item for item in metadata["groups"]["pose"] if item["name"] == "head_lean"
    )
    corrected = client.post(
        "/api/deform",
        json={
            "action": "set",
            "group": "pose",
            "index": head_lean["index"],
            "value": 0.2,
        },
    )
    uncorrected_config = client.post(
        "/api/configure",
        json={"applyCorrectives": False},
    )
    uncorrected = client.get("/api/geometry")

    corrected_values = np.frombuffer(corrected.data, dtype="<f4")
    uncorrected_values = np.frombuffer(uncorrected.data, dtype="<f4")
    assert uncorrected_config.status_code == 200
    assert uncorrected_config.json["model"]["applyCorrectives"] is False
    assert float(np.abs(corrected_values - uncorrected_values).max()) > 0

    restored = client.post(
        "/api/configure",
        json={"applyCorrectives": True},
    )
    assert restored.status_code == 200


def test_lod_selection_updates_topology_and_geometry(client):
    configured = client.post("/api/configure", json={"lod": 6})
    assert configured.status_code == 200
    assert configured.json["model"]["lod"] == 6
    assert configured.json["model"]["vertexCount"] == 595
    assert configured.json["model"]["faceCount"] == 1171

    topology = client.get("/api/topology")
    geometry = client.get("/api/geometry")
    expected_geometry_bytes = (
        (
            configured.json["model"]["vertexCount"]
            + configured.json["model"]["jointCount"]
        )
        * 3
        * 4
    )
    assert len(topology.data) == 1171 * 3 * 4
    assert len(geometry.data) == expected_geometry_bytes
    assert topology.headers["Cache-Control"] == "no-store"

    restored = client.post("/api/configure", json={"lod": 1})
    assert restored.status_code == 200
    assert restored.json["model"]["vertexCount"] == 18439


@pytest.mark.parametrize(
    "payload",
    [
        {"lod": -1},
        {"lod": 7},
        {"lod": "high"},
        {"applyCorrectives": "yes"},
        {"snapToGround": "yes"},
    ],
)
def test_model_configuration_validation(client, payload):
    response = client.post("/api/configure", json=payload)
    assert response.status_code == 400


def test_snap_to_ground_is_driven_by_root_ty(client):
    client.post(
        "/api/configure",
        json={"snapToGround": True},
    )
    client.post(
        "/api/deform",
        json={"action": "reset_all", "group": "pose", "index": 0},
    )
    metadata = client.get("/api/metadata").json
    upper_legs = next(
        item for item in metadata["groups"]["pose"] if item["name"] == "scale_uplegs"
    )
    lower_legs = next(
        item for item in metadata["groups"]["pose"] if item["name"] == "scale_lowlegs"
    )
    root_ty = next(
        item for item in metadata["groups"]["pose"] if item["name"] == "root_ty"
    )

    client.post(
        "/api/deform",
        json={
            "action": "set",
            "group": "pose",
            "index": upper_legs["index"],
            "value": upper_legs["max"],
        },
    )
    snapped = client.post(
        "/api/deform",
        json={
            "action": "set",
            "group": "pose",
            "index": lower_legs["index"],
            "value": lower_legs["max"],
        },
    )
    vertex_count = metadata["model"]["vertexCount"]
    snapped_vertices = np.frombuffer(
        snapped.data,
        dtype="<f4",
        count=vertex_count * 3,
    ).reshape(-1, 3)
    snapped_metadata = client.get("/api/metadata").json
    snapped_root_ty = next(
        item for item in snapped_metadata["groups"]["pose"] if item["name"] == "root_ty"
    )

    assert np.min(snapped_vertices[:, 1]) == pytest.approx(0.0, abs=1e-6)
    assert snapped_root_ty["value"] != pytest.approx(0.0, abs=1e-4)
    assert snapped_root_ty["managed"] is True
    assert float(snapped.headers["X-MHR-Root-TY"]) == pytest.approx(
        snapped_root_ty["value"],
        abs=1e-6,
    )

    managed_update = client.post(
        "/api/deform",
        json={
            "action": "set",
            "group": "pose",
            "index": root_ty["index"],
            "value": 1.0,
        },
    )
    assert managed_update.status_code == 400
    assert managed_update.json == {"error": "Invalid deformation request."}

    unsnapped = client.post(
        "/api/configure",
        json={"snapToGround": False},
    )
    unsnapped_geometry = np.frombuffer(
        client.get("/api/geometry").data,
        dtype="<f4",
        count=vertex_count * 3,
    ).reshape(-1, 3)
    unsnapped_root_ty = next(
        item for item in unsnapped.json["groups"]["pose"] if item["name"] == "root_ty"
    )
    assert unsnapped.json["model"]["snapToGround"] is False
    assert unsnapped_root_ty["managed"] is False
    assert unsnapped_root_ty["value"] == pytest.approx(0.0, abs=1e-6)
    assert abs(float(np.min(unsnapped_geometry[:, 1]))) > 1e-3

    client.post(
        "/api/deform",
        json={"action": "reset_all", "group": "pose", "index": 0},
    )
    restored = client.post(
        "/api/configure",
        json={"snapToGround": True},
    )
    assert restored.status_code == 200


def test_obj_export_matches_current_mesh(client):
    metadata = client.get("/api/metadata").json
    geometry = np.frombuffer(client.get("/api/geometry").data, dtype="<f4")
    expected_vertices = geometry[: metadata["model"]["vertexCount"] * 3].reshape(-1, 3)
    response = client.get("/api/export/obj")

    assert response.status_code == 200
    assert response.data.startswith(b"# MHR mesh export")
    assert response.headers["X-MHR-LOD"] == str(metadata["model"]["lod"])
    assert response.headers["Content-Disposition"].endswith('.obj"')
    # Parse directly: trimesh.load drops unreferenced vertices even with
    # process=False, and the seal cut orphans a few. The payload must
    # still carry every vertex in live order (skin-weight indexing).
    positions, face_lines = [], 0
    for line in response.data.decode("utf-8").splitlines():
        if line.startswith("v "):
            positions.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            face_lines += 1
    assert len(positions) == metadata["model"]["vertexCount"]
    assert face_lines == metadata["model"]["faceCount"]
    np.testing.assert_allclose(
        np.array(positions, dtype=np.float32), expected_vertices, atol=1e-6
    )


def test_glb_export_matches_current_mesh(client):
    metadata = client.get("/api/metadata").json
    geometry = np.frombuffer(client.get("/api/geometry").data, dtype="<f4")
    expected_vertices = geometry[: metadata["model"]["vertexCount"] * 3].reshape(-1, 3)
    response = client.get("/api/export/glb")
    exported = trimesh.load(
        BytesIO(response.data),
        file_type="glb",
        force="mesh",
        process=False,
    )

    assert response.status_code == 200
    assert response.data[:4] == b"glTF"
    assert response.mimetype == "model/gltf-binary"
    assert response.headers["Content-Disposition"].endswith('.glb"')
    assert len(exported.vertices) == metadata["model"]["vertexCount"]
    assert len(exported.faces) == metadata["model"]["faceCount"]
    np.testing.assert_allclose(exported.vertices, expected_vertices, atol=1e-6)


def test_usd_export_matches_current_mesh(client):
    metadata = client.get("/api/metadata").json
    geometry = np.frombuffer(client.get("/api/geometry").data, dtype="<f4")
    expected_vertices = geometry[: metadata["model"]["vertexCount"] * 3].reshape(-1, 3)
    response = client.get("/api/export/usd")
    layer = Sdf.Layer.CreateAnonymous("test.usda")
    assert layer.ImportFromString(response.data.decode("utf-8"))
    stage = Usd.Stage.Open(layer)
    exported = UsdGeom.Mesh(stage.GetPrimAtPath("/MHR/Mesh"))

    assert response.status_code == 200
    assert response.data.startswith(b"#usda")
    assert response.mimetype == "model/vnd.usd"
    assert response.headers["Content-Disposition"].endswith('.usd"')
    assert len(exported.GetPointsAttr().Get()) == metadata["model"]["vertexCount"]
    assert (
        len(exported.GetFaceVertexCountsAttr().Get()) == metadata["model"]["faceCount"]
    )
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
    np.testing.assert_allclose(
        np.asarray(exported.GetPointsAttr().Get()),
        expected_vertices,
        atol=1e-6,
    )


def test_export_validation(client):
    response = client.get("/api/export/fbx")
    assert response.status_code == 400
    assert "Unsupported export format" in response.json["error"]

    response = client.get("/api/export/obj?ground=yes")
    assert response.status_code == 400
    assert "has been removed" in response.json["error"]


def test_exception_details_are_not_exposed(client, monkeypatch):
    sensitive_detail = "/srv/private/mhr_model.pt: password=super-secret"

    def raise_value_error(*_args, **_kwargs):
        raise ValueError(sensitive_detail)

    engine = client.application.config["MHR_ENGINE"]
    monkeypatch.setattr(engine, "update", raise_value_error)
    monkeypatch.setattr(engine, "configure", raise_value_error)

    deform = client.post("/api/deform", json={})
    configure = client.post("/api/configure", json={})

    assert deform.status_code == 400
    assert deform.json == {"error": "Invalid deformation request."}
    assert configure.status_code == 400
    assert configure.json == {"error": "Invalid configuration request."}
    assert sensitive_detail not in deform.get_data(as_text=True)
    assert sensitive_detail not in configure.get_data(as_text=True)


def test_export_exception_details_are_not_exposed(client, monkeypatch):
    sensitive_detail = "/srv/private/mhr_model.pt: password=super-secret"

    def raise_value_error(*_args, **_kwargs):
        raise ValueError(sensitive_detail)

    def raise_runtime_error(*_args, **_kwargs):
        raise RuntimeError(sensitive_detail)

    monkeypatch.setattr(server_module, "normalize_export_format", raise_value_error)
    invalid_format = client.get("/api/export/obj")
    monkeypatch.setattr(server_module, "normalize_export_format", lambda value: value)
    monkeypatch.setattr(server_module, "export_mesh", raise_value_error)
    invalid_export = client.get("/api/export/obj")
    monkeypatch.setattr(server_module, "export_mesh", raise_runtime_error)
    unavailable_export = client.get("/api/export/obj")

    assert invalid_format.status_code == 400
    assert "Unsupported export format" in invalid_format.json["error"]
    assert invalid_export.status_code == 400
    assert invalid_export.json == {"error": "Invalid mesh export request."}
    assert unavailable_export.status_code == 503
    assert unavailable_export.json == {
        "error": "Mesh export is temporarily unavailable."
    }
    for response in (invalid_format, invalid_export, unavailable_export):
        assert sensitive_detail not in response.get_data(as_text=True)


def test_glb_export_contains_armature_and_lbs_weights(client):
    metadata = client.get("/api/metadata").json
    response = client.get("/api/export/glb")
    document, binary = _parse_glb(response.data)
    primitive = document["meshes"][0]["primitives"][0]
    skin = document["skins"][0]

    assert len(document["skins"]) == 1
    assert len(skin["joints"]) == metadata["model"]["jointCount"]
    assert "JOINTS_0" in primitive["attributes"]
    assert "WEIGHTS_0" in primitive["attributes"]

    joint_indices = _glb_accessor(
        document,
        binary,
        primitive["attributes"]["JOINTS_0"],
    )
    joint_weights = _glb_accessor(
        document,
        binary,
        primitive["attributes"]["WEIGHTS_0"],
    )
    inverse_binds = _glb_accessor(
        document,
        binary,
        skin["inverseBindMatrices"],
    )
    assert joint_indices.shape == (metadata["model"]["vertexCount"], 4)
    assert joint_weights.shape == (metadata["model"]["vertexCount"], 4)
    assert int(joint_indices.max()) < metadata["model"]["jointCount"]
    np.testing.assert_allclose(
        joint_weights.sum(axis=1),
        1.0,
        atol=1e-6,
    )
    local_transforms = np.asarray(
        [
            np.asarray(document["nodes"][node_index]["matrix"]).reshape(4, 4).T
            for node_index in skin["joints"]
        ]
    )
    global_transforms = local_transforms.copy()
    for index, parent in enumerate(metadata["jointParents"]):
        if parent >= 0:
            global_transforms[index] = (
                global_transforms[parent] @ local_transforms[index]
            )
    np.testing.assert_allclose(
        global_transforms @ inverse_binds,
        np.broadcast_to(
            np.eye(4),
            (metadata["model"]["jointCount"], 4, 4),
        ),
        atol=2e-5,
    )


def test_usd_export_contains_armature_and_lbs_weights(client):
    metadata = client.get("/api/metadata").json
    response = client.get("/api/export/usd")
    layer = Sdf.Layer.CreateAnonymous("rigged-test.usda")
    assert layer.ImportFromString(response.data.decode("utf-8"))
    stage = Usd.Stage.Open(layer)
    root = UsdSkel.Root(stage.GetPrimAtPath("/MHR"))
    skeleton = UsdSkel.Skeleton(stage.GetPrimAtPath("/MHR/Skeleton"))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/MHR/Mesh"))
    binding = UsdSkel.BindingAPI(mesh.GetPrim())

    assert root
    assert skeleton
    expected_joint_paths = []
    for index, name in enumerate(metadata["jointNames"]):
        parent = metadata["jointParents"][index]
        expected_joint_paths.append(
            name if parent < 0 else f"{expected_joint_paths[parent]}/{name}"
        )
    exported_joint_paths = [str(path) for path in skeleton.GetJointsAttr().Get()]
    assert exported_joint_paths == expected_joint_paths
    assert len(exported_joint_paths) == metadata["model"]["jointCount"]
    assert (
        len(skeleton.GetBindTransformsAttr().Get()) == metadata["model"]["jointCount"]
    )
    assert (
        len(skeleton.GetRestTransformsAttr().Get()) == metadata["model"]["jointCount"]
    )
    assert binding.GetSkeletonRel().GetTargets() == [skeleton.GetPath()]

    joint_indices_primvar = binding.GetJointIndicesPrimvar()
    joint_weights_primvar = binding.GetJointWeightsPrimvar()
    joint_indices = np.asarray(
        joint_indices_primvar.Get(),
        dtype=np.int32,
    ).reshape(-1, 4)
    joint_weights = np.asarray(
        joint_weights_primvar.Get(),
        dtype=np.float32,
    ).reshape(-1, 4)
    assert joint_indices_primvar.GetElementSize() == 4
    assert joint_weights_primvar.GetElementSize() == 4
    assert joint_indices.shape == (metadata["model"]["vertexCount"], 4)
    assert joint_weights.shape == (metadata["model"]["vertexCount"], 4)
    assert int(joint_indices.max()) < metadata["model"]["jointCount"]
    np.testing.assert_allclose(
        joint_weights.sum(axis=1),
        1.0,
        atol=1e-6,
    )
    cache = UsdSkel.Cache()
    assert cache.Populate(root, Usd.PrimDefaultPredicate)
    skeleton_query = cache.GetSkelQuery(skeleton)
    assert skeleton_query
    assert (
        len(skeleton_query.ComputeSkinningTransforms())
        == metadata["model"]["jointCount"]
    )


def test_all_lods_have_normalized_four_influence_skinning(client):
    engine = client.application.config["MHR_ENGINE"]
    metadata = client.get("/api/metadata").json
    expected_counts = {
        entry["lod"]: entry["vertexCount"]
        for entry in metadata["model"]["supportedLods"]
    }

    with engine.lock:
        for lod, vertex_count in expected_counts.items():
            joint_indices, joint_weights = engine._skinning_for_lod(lod)
            assert joint_indices.shape == (vertex_count, 4)
            assert joint_weights.shape == (vertex_count, 4)
            assert int(joint_indices.max()) < metadata["model"]["jointCount"]
            assert np.all(joint_weights >= 0)
            np.testing.assert_allclose(
                joint_weights.sum(axis=1),
                1.0,
                atol=1e-6,
            )


@pytest.mark.parametrize("file_format", ["obj", "glb", "usd"])
def test_snapped_export_uses_grounded_model_state(client, file_format):
    response = client.get(f"/api/export/{file_format}")

    assert response.status_code == 200
    assert response.headers["X-MHR-Snap-To-Ground"] == "true"
    assert "-grounded." in response.headers["Content-Disposition"]

    if file_format == "usd":
        layer = Sdf.Layer.CreateAnonymous("grounded-test.usda")
        assert layer.ImportFromString(response.data.decode("utf-8"))
        stage = Usd.Stage.Open(layer)
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/MHR/Mesh"))
        vertices = np.asarray(mesh.GetPointsAttr().Get())
    else:
        mesh = trimesh.load(
            BytesIO(response.data),
            file_type=file_format,
            force="mesh",
            process=False,
        )
        vertices = np.asarray(mesh.vertices)

    assert np.min(vertices[:, 1]) == pytest.approx(0.0, abs=1e-6)


def test_rig_mapping_endpoint(client):
    response = client.get("/api/rig-mapping")
    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"
    payload = response.data
    assert len(payload) >= 12
    rows, cols, joints = struct.unpack_from("<III", payload, 0)
    assert rows == 889
    assert cols == 204
    assert joints == 127
    expected_bytes = 12 + (rows * cols + joints * 3 + joints * 4) * 4
    assert len(payload) == expected_bytes


def test_neutral_rigged_glb_endpoint(client):
    response = client.get("/api/model/rigged-glb?lod=1")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "model/gltf-binary"
    assert response.headers["X-MHR-LOD"] == "1"
    doc, binary = _parse_glb(response.data)
    assert len(doc["nodes"]) == 128
    assert len(doc["skins"]) == 1
    assert doc["nodes"][127]["name"] == "MHR_Mesh"


def test_unposed_geometry_endpoint(client):
    response = client.get("/api/geometry?unposed=1")
    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"
    metadata = client.get("/api/metadata").json
    expected_vertex_bytes = metadata["model"]["vertexCount"] * 3 * 4
    assert len(response.data) == expected_vertex_bytes


def test_blendshape_weights_endpoint(client):
    # Test uncompressed
    response = client.get("/api/weights/blendshapes")
    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"
    payload = response.data
    assert len(payload) >= 12
    v_count, id_count, fe_count = struct.unpack_from("<III", payload, 0)
    assert v_count == 18439
    assert id_count == 45
    assert fe_count == 72
    expected_length = 12 + (v_count * 4 + v_count * (id_count + fe_count) * 4) * 4
    assert len(payload) == expected_length

    # Test gzipped
    response_gz = client.get(
        "/api/weights/blendshapes",
        headers={"Accept-Encoding": "gzip"},
    )
    assert response_gz.status_code == 200
    assert response_gz.headers.get("Content-Encoding") == "gzip"
    decompressed = gzip.decompress(response_gz.data)
    assert len(decompressed) == expected_length


def test_lod_mapping_endpoint(client):
    # LOD 1 has no mapping (header count 0)
    res_lod1 = client.get("/api/lod-mapping?lod=1")
    assert res_lod1.status_code == 200
    assert len(res_lod1.data) == 4
    assert struct.unpack("<I", res_lod1.data)[0] == 0

    # LOD 6 has 595 vertices
    res_lod6 = client.get("/api/lod-mapping?lod=6")
    assert res_lod6.status_code == 200
    v_count = struct.unpack_from("<I", res_lod6.data, 0)[0]
    assert v_count == 595
    expected_bytes = 4 + (595 * 3 * 4) + (595 * 3 * 4)
    assert len(res_lod6.data) == expected_bytes


def test_pose_corrective_weights_endpoint(client):
    # Test uncompressed
    response = client.get("/api/weights/pose-correctives")
    assert response.status_code == 200
    assert response.mimetype == "application/octet-stream"
    payload = response.data
    assert len(payload) >= 28
    magic, l0_nnz, l0_in, l0_out, l2_rows, l2_cols, l2_nnz = struct.unpack_from(
        "<IIIIIII", payload, 0
    )
    assert magic == 0x4350484D
    assert l0_nnz == 53136
    assert l0_in == 750
    assert l0_out == 3000
    assert l2_rows == 55317
    assert l2_cols == 3000
    assert l2_nnz == 4451976
    expected_length = 28 + (53136 * 2 * 2 + 53136 * 4) + ((55317 + 1) * 4 + 4451976 * 2 + 4451976 * 4)
    assert len(payload) == expected_length

    # Test gzipped
    response_gz = client.get(
        "/api/weights/pose-correctives",
        headers={"Accept-Encoding": "gzip"},
    )
    assert response_gz.status_code == 200
    assert response_gz.headers.get("Content-Encoding") == "gzip"
    decompressed = gzip.decompress(response_gz.data)
    assert len(decompressed) == expected_length


def test_neutral_rigged_glb_renders_requested_lod(client):
    client.post("/api/configure", json={"lod": 1})

    requested = client.get("/api/model/rigged-glb?lod=6")
    assert requested.status_code == 200
    assert requested.headers["X-MHR-LOD"] == "6"
    document, binary = _parse_glb(requested.data)
    primitive = document["meshes"][0]["primitives"][0]
    positions = _glb_accessor(
        document,
        binary,
        primitive["attributes"]["POSITION"],
    )
    assert positions.shape == (595, 3)

    current = client.get("/api/model/rigged-glb?lod=1")
    assert current.status_code == 200
    document, binary = _parse_glb(current.data)
    primitive = document["meshes"][0]["primitives"][0]
    positions = _glb_accessor(
        document,
        binary,
        primitive["attributes"]["POSITION"],
    )
    assert positions.shape == (18439, 3)


def test_deform_nogeometry_returns_headers_without_body(client):
    metadata = client.get("/api/metadata").json
    knee = next(
        item for item in metadata["groups"]["pose"] if item["name"] == "l_knee_bend"
    )
    response = client.post(
        "/api/deform?nogeometry=1",
        json={
            "action": "set",
            "group": "pose",
            "index": knee["index"],
            "value": 0.5,
        },
    )
    assert response.status_code == 204
    assert response.data == b""
    assert float(response.headers["X-MHR-Inference-Ms"]) >= 0
    assert "X-MHR-Revision" in response.headers
    assert "X-MHR-Root-TY" in response.headers

    updated = client.get("/api/metadata").json
    updated_knee = next(
        item for item in updated["groups"]["pose"] if item["name"] == "l_knee_bend"
    )
    assert updated_knee["value"] == pytest.approx(0.5)

    client.post(
        "/api/deform",
        json={"action": "reset_all", "group": "pose", "index": 0},
    )


def test_identity_regions_are_labeled_by_zone(client):
    metadata = client.get("/api/metadata").json
    assert metadata["identityRegions"] == list(IDENTITY_ZONES)
    identity = metadata["groups"]["identity"]
    assert len(identity) == 45
    for entry in identity:
        assert entry["region"] in IDENTITY_ZONES
        assert 0.0 < entry["regionShare"] <= 1.0
    by_name = {entry["name"]: entry for entry in identity}
    for number in range(21, 41):
        assert by_name[f"identity_{number:02d}"]["region"] == "Head"
    for number in range(41, 46):
        assert by_name[f"identity_{number:02d}"]["region"] == "Hands"
    assert by_name["identity_01"]["region"] == "Torso"


def test_identity_zone_covers_all_joints(client):
    engine = client.application.config["MHR_ENGINE"]
    for name in engine.joint_names:
        zone = identity_zone_for_joint(name)
        if zone is None:
            assert joint_group_key(name) == "zero_weight"
        else:
            assert zone in IDENTITY_ZONES


def test_expression_regions_are_labeled_by_zone(client):
    metadata = client.get("/api/metadata").json
    assert metadata["expressionRegions"] == list(EXPRESSION_ZONES)
    expression = metadata["groups"]["expression"]
    assert len(expression) == 72
    by_name = {entry["name"]: entry for entry in expression}
    assert by_name["browLowerer_R"]["region"] == "Brows"
    assert by_name["eyesClosed_L"]["region"] == "Eyes"
    assert by_name["noseWrinkler_L"]["region"] == "Nose"
    assert by_name["cheekRaiser_R"]["region"] == "Cheeks"
    assert by_name["lipCornerPuller_L"]["region"] == "Mouth"
    assert by_name["jawDrop"]["region"] == "Jaw"
    assert by_name["lipCornerPuller_L"]["label"] == "Lip Corner Puller (Left)"
    assert by_name["lipFunneler_RT"]["label"] == "Lip Funneler (Right Top)"
    assert by_name["jawDrop"]["label"] == "Jaw Drop"
    for entry in expression:
        assert entry["region"] in EXPRESSION_ZONES


def test_metadata_exposes_topology_version(client):
    version = client.get("/api/metadata").json["model"]["topologyVersion"]
    assert len(version) == 12 and int(version, 16) >= 0


def test_mouth_seal_has_no_hot_remnants(client):
    """Every gap-spanning lip face must be cut: kept faces on the lip
    line must not stretch under jawDrop (remnant spikes)."""
    from tools.remove_sealed_polys import deformation

    engine = client.application.config["MHR_ENGINE"]
    base = (
        engine.model.character_torch.blend_shape.base_shape.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    shape_vectors = (
        engine.model.face_expressions_model.shape_vectors.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    jaw = engine.expression_names.index("jawDrop")
    for lod in SUPPORTED_LODS:
        faces = engine.lod_faces[lod]
        neutral = engine._render_lod(base, lod) if lod != 1 else base
        opened = (
            engine._render_lod(base + shape_vectors[jaw], lod)
            if lod != 1
            else base + shape_vectors[jaw]
        )
        dev = deformation(neutral, opened, faces)
        centroids = neutral[faces].mean(axis=1)
        core = np.flatnonzero(
            (np.abs(centroids[:, 0]) < 1.9)
            & (np.abs(centroids[:, 1] - 155.75) < 0.4)
        )
        hot = core[dev[core] >= 4.0]
        assert len(hot) == 0, f"LOD {lod}: {len(hot)} hot remnants"


def test_mixamo_clips_lists_supported_files_only(client, tmp_path, monkeypatch):
    (tmp_path / "b.glb").write_bytes(b"fake-glb")
    (tmp_path / "a.fbx").write_bytes(b"fake-fbx")
    (tmp_path / "note.txt").write_text("not a clip")
    (tmp_path / "sub").mkdir()
    monkeypatch.setitem(client.application.config, "MIXAMO_ROOT", tmp_path)
    resp = client.get("/api/mixamo/clips")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["available"] is True
    assert [clip["name"] for clip in payload["clips"]] == ["a.fbx", "b.glb"]
    assert payload["clips"][0]["bytes"] == 8


def test_mixamo_clips_missing_root_is_empty(client, tmp_path, monkeypatch):
    monkeypatch.setitem(client.application.config, "MIXAMO_ROOT", tmp_path / "absent")
    resp = client.get("/api/mixamo/clips")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["available"] is False
    assert payload["clips"] == []


def test_mixamo_file_serves_bytes_and_rejects_unknown(client, tmp_path, monkeypatch):
    (tmp_path / "dance.fbx").write_bytes(b"fake-fbx-bytes")
    (tmp_path / "evil.txt").write_text("x")
    monkeypatch.setitem(client.application.config, "MIXAMO_ROOT", tmp_path)
    ok = client.get("/api/mixamo/file/dance.fbx")
    assert ok.status_code == 200
    assert ok.data == b"fake-fbx-bytes"
    assert client.get("/api/mixamo/file/missing.fbx").status_code == 404
    assert client.get("/api/mixamo/file/evil.txt").status_code == 404


def test_mixamo_resolve_rejects_traversal_and_subpaths(tmp_path):
    resolve = server_module.resolve_mixamo_file
    assert resolve(tmp_path, "../server.py") is None
    assert resolve(tmp_path, "sub/dance.fbx") is None
    assert resolve(tmp_path, "dance.txt") is None


def test_hdri_list_supported_files_only_and_missing_root(client, tmp_path, monkeypatch):
    (tmp_path / "studio.exr").write_bytes(b"fake-exr")
    (tmp_path / "sky.hdr").write_bytes(b"fake-hdr")
    (tmp_path / "note.txt").write_text("not an environment")
    (tmp_path / "sub").mkdir()
    monkeypatch.setitem(client.application.config, "HDRI_ROOT", tmp_path)
    resp = client.get("/api/hdri/list")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["available"] is True
    assert [item["name"] for item in payload["files"]] == ["sky.hdr", "studio.exr"]
    monkeypatch.setitem(client.application.config, "HDRI_ROOT", tmp_path / "absent")
    resp = client.get("/api/hdri/list")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["available"] is False
    assert payload["files"] == []


def test_hdri_file_serves_bytes_and_rejects_unknown(client, tmp_path, monkeypatch):
    (tmp_path / "cannon.exr").write_bytes(b"fake-exr-bytes")
    (tmp_path / "evil.txt").write_text("x")
    monkeypatch.setitem(client.application.config, "HDRI_ROOT", tmp_path)
    ok = client.get("/api/hdri/file/cannon.exr")
    assert ok.status_code == 200
    assert ok.data == b"fake-exr-bytes"
    assert client.get("/api/hdri/file/missing.exr").status_code == 404
    assert client.get("/api/hdri/file/evil.txt").status_code == 404


def test_hdri_thumb_rejects_unknown_and_undecodable(client, tmp_path, monkeypatch):
    (tmp_path / "broken.exr").write_bytes(b"not an exr")
    monkeypatch.setitem(client.application.config, "HDRI_ROOT", tmp_path)
    assert client.get("/api/hdri/thumb/missing.exr").status_code == 404
    assert client.get("/api/hdri/thumb/broken.exr").status_code == 503


def test_hdri_resolve_rejects_traversal_and_subpaths(tmp_path):
    resolve = server_module.resolve_hdri_file
    assert resolve(tmp_path, "../server.py") is None
    assert resolve(tmp_path, "sub/cannon.exr") is None
    assert resolve(tmp_path, "cannon.jpg") is None


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

def _write_pose(root, stem="kick"):
    pose_dir = root / "poses01" / "poses" / stem
    pose_dir.mkdir(parents=True)
    (pose_dir / f"{stem}.bvh").write_bytes(b"HIERARCHY\n")
    (pose_dir / f"{stem}.meta").write_text(
        "name Kick-pose\ndescription A kick\nlicense CC-BY\ncopyright (c) 2020 Someone\n"
    )
    (pose_dir / f"{stem}.thumb").write_bytes(PNG_MAGIC + b"fake-png-body")
    return f"poses01/poses/{stem}/{stem}.bvh"


def test_pose_list_nests_meta_and_thumb_flags(client, tmp_path, monkeypatch):
    pose_id = _write_pose(tmp_path)
    (tmp_path / "note.txt").write_text("not a pose")
    monkeypatch.setitem(client.application.config, "POSE_ROOT", tmp_path)
    resp = client.get("/api/pose/list")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["available"] is True
    assert len(payload["poses"]) == 1
    pose = payload["poses"][0]
    assert pose["id"] == pose_id
    assert pose["name"] == "Kick-pose"
    assert pose["set"] == "poses01"
    assert pose["license"] == "CC-BY"
    assert pose["copyright"] == "(c) 2020 Someone"
    assert pose["hasThumb"] is True
    monkeypatch.setitem(client.application.config, "POSE_ROOT", tmp_path / "absent")
    resp = client.get("/api/pose/list")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["available"] is False
    assert payload["poses"] == []


def test_pose_file_serves_bytes_and_rejects_unknown(client, tmp_path, monkeypatch):
    pose_id = _write_pose(tmp_path)
    (tmp_path / "evil.txt").write_text("x")
    monkeypatch.setitem(client.application.config, "POSE_ROOT", tmp_path)
    ok = client.get(f"/api/pose/file/{pose_id}")
    assert ok.status_code == 200
    assert ok.data == b"HIERARCHY\n"
    assert client.get("/api/pose/file/poses01/poses/kick/missing.bvh").status_code == 404
    assert client.get("/api/pose/file/evil.txt").status_code == 404


def test_pose_thumb_serves_png_and_rejects_unknown(client, tmp_path, monkeypatch):
    pose_id = _write_pose(tmp_path)
    monkeypatch.setitem(client.application.config, "POSE_ROOT", tmp_path)
    ok = client.get(f"/api/pose/thumb/{pose_id}")
    assert ok.status_code == 200
    assert ok.mimetype == "image/png"
    assert ok.data.startswith(PNG_MAGIC)
    assert client.get("/api/pose/thumb/poses01/poses/kick/missing.bvh").status_code == 404


def test_pose_thumb_rejects_missing_and_non_png(client, tmp_path, monkeypatch):
    (tmp_path / "lonely.bvh").write_text("HIERARCHY\n")
    bad_dir = tmp_path / "set" / "poses" / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "bad.bvh").write_text("HIERARCHY\n")
    (bad_dir / "bad.thumb").write_bytes(b"not a png")
    monkeypatch.setitem(client.application.config, "POSE_ROOT", tmp_path)
    assert client.get("/api/pose/thumb/lonely.bvh").status_code == 503
    assert client.get("/api/pose/thumb/set/poses/bad/bad.bvh").status_code == 503


def _write_mhasset(root: Path) -> str:
    asset_dir = root / "teeth" / "teeth_base"
    asset_dir.mkdir(parents=True)
    (asset_dir / "jaw_upper.glb").write_bytes(b"glTF-binary")
    (asset_dir / "jaw_upper.thumb").write_bytes(PNG_MAGIC + b"fake-png-body")
    return "teeth/teeth_base/jaw_upper.glb"


def test_mhasset_list_flags_thumbs_and_missing_root(client, tmp_path, monkeypatch):
    asset_id = _write_mhasset(tmp_path)
    (tmp_path / "note.txt").write_text("not an asset")
    monkeypatch.setitem(client.application.config, "MHASSET_ROOT", tmp_path)
    resp = client.get("/api/mhasset/list")
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["available"] is True
    assert len(payload["assets"]) == 1
    asset = payload["assets"][0]
    assert asset["id"] == asset_id
    assert asset["category"] == "teeth"
    assert asset["hasThumb"] is True
    monkeypatch.setitem(client.application.config, "MHASSET_ROOT", tmp_path / "absent")
    resp = client.get("/api/mhasset/list")
    assert resp.status_code == 200
    assert resp.get_json()["assets"] == []


def test_mhasset_file_serves_bytes_and_rejects_unknown(client, tmp_path, monkeypatch):
    asset_id = _write_mhasset(tmp_path)
    (tmp_path / "evil.txt").write_text("x")
    monkeypatch.setitem(client.application.config, "MHASSET_ROOT", tmp_path)
    ok = client.get(f"/api/mhasset/file/{asset_id}")
    assert ok.status_code == 200
    assert ok.data == b"glTF-binary"
    assert client.get("/api/mhasset/file/teeth/teeth_base/missing.glb").status_code == 404
    assert client.get("/api/mhasset/file/evil.txt").status_code == 404
    thumb = client.get("/api/mhasset/file/teeth/teeth_base/jaw_upper.thumb")
    assert thumb.status_code == 200
    assert thumb.mimetype == "image/png"


def test_mhasset_list_resolves_folder_named_thumbs(client, tmp_path, monkeypatch):
    style_dir = tmp_path / "eye_decoration_modelled" / "mindfront_eyebrows_01"
    style_dir.mkdir(parents=True)
    (style_dir / "mind_eyebrows_01.obj").write_bytes(b"obj-bytes")
    (style_dir / "mindfront_eyebrows_01.thumb").write_bytes(PNG_MAGIC + b"fake")
    monkeypatch.setitem(client.application.config, "MHASSET_ROOT", tmp_path)
    resp = client.get("/api/mhasset/list")
    assert resp.status_code == 200
    (asset,) = resp.get_json()["assets"]
    assert asset["hasThumb"] is True
    assert asset["thumb"] == (
        "eye_decoration_modelled/mindfront_eyebrows_01/mindfront_eyebrows_01.thumb"
    )


def test_mhasset_resolve_rejects_traversal_but_keeps_nesting(tmp_path):
    resolve = server_module.resolve_mhasset_file
    nested = tmp_path / "teeth" / "teeth_base"
    nested.mkdir(parents=True)
    (nested / "jaw_upper.glb").write_bytes(b"glTF-binary")
    assert resolve(tmp_path, "../server.py") is None
    assert resolve(tmp_path, "teeth/teeth_base/evil.txt") is None
    assert resolve(tmp_path, "teeth/teeth_base/jaw_upper.glb") == nested / "jaw_upper.glb"


def test_pose_resolve_rejects_traversal_but_keeps_nesting(tmp_path):
    resolve = server_module.resolve_pose_file
    nested = tmp_path / "poses01" / "poses" / "kick"
    nested.mkdir(parents=True)
    (nested / "kick.bvh").write_text("HIERARCHY\n")
    assert resolve(tmp_path, "../server.py") is None
    assert resolve(tmp_path, "poses01/poses/kick/evil.txt") is None
    assert resolve(tmp_path, "poses01/poses/kick/kick.bvh") == nested / "kick.bvh"
