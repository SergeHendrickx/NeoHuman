"""Local Three.js web viewer backend for the Momentum Human Rig model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import importlib.util
from pathlib import Path
import re
import struct
import threading
import time
import webbrowser

from flask import Flask, Response, current_app, jsonify, request, send_from_directory
import numpy as np
import scipy.sparse as sp
import torch

from exporters import (
    RIGGED_EXPORT_FORMATS,
    RigData,
    SUPPORTED_EXPORT_FORMATS,
    export_mesh,
    normalize_export_format,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT.parent / "assets" / "mhr_model.pt"
STATIC_ROOT = PROJECT_ROOT / "static"
THREE_ROOT = PROJECT_ROOT / "node_modules" / "three"
DOCKVIEW_ROOT = PROJECT_ROOT / "node_modules" / "dockview"
DOCKVIEW_CORE_ROOT = PROJECT_ROOT / "node_modules" / "dockview-core"
LOD_MAPPING_ROOT = PROJECT_ROOT.parent / "tools" / "mhr_LOD_conversion"
LOD_TOPOLOGY_PATH = PROJECT_ROOT / "data" / "lod_topology.npz"
DEFAULT_MIXAMO_ROOT = PROJECT_ROOT / "data" / "mixamo"
MIXAMO_EXTENSIONS = (".fbx", ".glb")
DEFAULT_HDRI_ROOT = PROJECT_ROOT / "data" / "360hdri"
DEFAULT_POSE_ROOT = PROJECT_ROOT / "data" / "makehuman"
DEFAULT_MHASSET_ROOT = PROJECT_ROOT / "data" / "makehuman"
HDRI_EXTENSIONS = (".exr", ".hdr")
POSE_EXTENSIONS = (".bvh",)
MHASSET_EXTENSIONS = (".glb", ".obj", ".png", ".mhmat", ".thumb")
SUPPORTED_LODS = tuple(range(7))
ROOT_TY_NAME = "root_ty"
ROOT_TY_SOURCE_UNITS_PER_VALUE = 10.0
IDENTITY_RANGE = (-3.0, 3.0)
EXPRESSION_RANGE = (-1.0, 1.0)


def _pose_category(name: str) -> str:
    if name.startswith("root_"):
        return "Root"
    if name.startswith("spine"):
        return "Spine"
    if name.startswith(("neck_", "head_")):
        return "Neck & head"
    if name.startswith(("r_clavicle", "r_uparm", "r_elbow", "r_lowarm", "r_wrist")):
        return "Right arm"
    if name.startswith(("l_clavicle", "l_uparm", "l_elbow", "l_lowarm", "l_wrist")):
        return "Left arm"
    if name.startswith(
        (
            "r_upleg",
            "r_knee",
            "r_lowleg",
            "r_foot",
            "r_ball",
            "r_subtalar",
            "r_talocrural",
        )
    ):
        return "Right leg"
    if name.startswith(
        (
            "l_upleg",
            "l_knee",
            "l_lowleg",
            "l_foot",
            "l_ball",
            "l_subtalar",
            "l_talocrural",
        )
    ):
        return "Left leg"
    if name.startswith(("r_thumb", "r_index", "r_middle", "r_ring", "r_pinky")):
        return "Right hand"
    if name.startswith(("l_thumb", "l_index", "l_middle", "l_ring", "l_pinky")):
        return "Left hand"
    if name.startswith("scale_"):
        return "Proportions"
    return "Flexible body"


IDENTITY_ZONES = ("Head", "Torso", "Arms", "Hands", "Legs", "Feet")

_ZERO_WEIGHT_PARTS = {
    "eye",
    "tongue",
    "teeth",
    "brow",
    "cheek",
    "lip",
    "ear",
    "nose",
    "body_world",
}
_HAND_PARTS = {"thumb", "index", "middle", "ring", "pinky", "wrist"}
_FOOT_PARTS = {"foot", "talocrural", "subtalar", "transversetarsal", "ball"}


def joint_group_key(name: str) -> str:
    """Group a joint into a body part.

    Ported from tools/mhr_create_segmentation/example.py so the viewer can
    label blendshape regions without the pymomentum dependency.
    """
    lower = name.lower()
    if any(part in lower for part in _ZERO_WEIGHT_PARTS):
        return "zero_weight"
    suffix = lower[2:]
    if any(part in suffix for part in _HAND_PARTS):
        return f"{lower[:2]}hand"
    if any(part in suffix for part in _FOOT_PARTS):
        return f"{lower[:2]}foot"
    key = re.sub(r"(_null|_twist\d*|\d+|_twist\d_proc)$", "", name).rstrip("_")
    return key


def identity_zone_for_joint(name: str) -> str | None:
    """Coarse anatomical zone for identity-component labeling.

    Returns None for joints that carry no skinning influence.
    """
    key = joint_group_key(name)
    if key in {"c_head", "c_jaw", "c_neck"}:
        return "Head"
    if key in {"c_spine", "root", "l_clavicle", "r_clavicle"}:
        return "Torso"
    if key in {"l_uparm", "r_uparm", "l_lowarm", "r_lowarm"}:
        return "Arms"
    if key in {"l_hand", "r_hand"}:
        return "Hands"
    if key in {"l_upleg", "r_upleg", "l_lowleg", "r_lowleg"}:
        return "Legs"
    if key in {"l_foot", "r_foot"}:
        return "Feet"
    return None


EXPRESSION_ZONES = ("Brows", "Eyes", "Nose", "Cheeks", "Mouth", "Jaw")


def _load_expression_names(count: int) -> tuple[str, ...]:
    """FACS action-unit names aligned with face_expr_coeffs columns.

    Loaded from mhr/face_expression.py by path so the viewer does not depend
    on the pymomentum-backed mhr package. Falls back to generic names when
    the file is missing or does not match this model coefficient count.
    """
    try:
        path = PROJECT_ROOT.parent / "mhr" / "face_expression.py"
        spec = importlib.util.spec_from_file_location(
            "mhr_face_expression_names", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        names = tuple(module.FACE_EXPRESSION_NAMES)
    except (OSError, AttributeError):
        return ()
    return names if len(names) == count else ()


def expression_zone_for_action(name: str) -> str:
    """Face zone for a FACS action-unit name."""
    lower = name.lower()
    if "brow" in lower:
        return "Brows"
    if lower.startswith("eyes") or "lid" in lower:
        return "Eyes"
    if "nose" in lower or "nostril" in lower:
        return "Nose"
    if "cheek" in lower or "nasolabial" in lower:
        return "Cheeks"
    if "jaw" in lower:
        return "Jaw"
    if "chin" in lower:
        return "Jaw"
    return "Mouth"


_EXPRESSION_SIDE_WORDS = {"L": "Left", "R": "Right", "B": "Bottom", "T": "Top"}


def prettify_expression_name(name: str) -> str:
    """Humanize a FACS action name: lipFunneler_LB -> Lip Funneler (Left Bottom)."""
    base, separator, suffix = name.partition("_")
    words = re.sub(r"(?<!^)(?=[A-Z])", " ", base).strip().title()
    if not separator:
        return words
    sides = " ".join(_EXPRESSION_SIDE_WORDS.get(char, char) for char in suffix)
    return f"{words} ({sides})" if sides else words


@dataclass(frozen=True)
class GeometrySnapshot:
    vertices: np.ndarray
    joints: np.ndarray
    skeleton_state: np.ndarray
    revision: int
    inference_ms: float
    root_ty: float

    def to_bytes(self) -> bytes:
        return (
            self.vertices.astype("<f4", copy=False).tobytes()
            + self.joints.astype("<f4", copy=False).tobytes()
        )


class MHREngine:
    """Thread-safe state and inference wrapper around the TorchScript model."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MHR model not found at {model_path}. "
                "Pass --model with the path to mhr_model.pt."
            )

        self.model_path = model_path
        self.lock = threading.RLock()
        self.model = torch.jit.load(str(model_path), map_location="cpu").eval()
        self.lod = 1
        self.apply_correctives = True
        self.identity_count = int(self.model.get_num_identity_blendshapes())

        all_names = list(self.model.get_parameter_names())
        self.pose_names = all_names[: -self.identity_count]
        all_limits = (
            self.model.get_parameter_limits().detach().cpu().numpy().astype(np.float32)
        )
        self.pose_limits = all_limits[: len(self.pose_names)]
        self.pose_values = np.zeros(len(self.pose_names), dtype=np.float32)
        self.identity_values = np.zeros(self.identity_count, dtype=np.float32)
        self.expression_values = np.zeros(72, dtype=np.float32)
        self.identity_limits = np.tile(
            np.asarray([IDENTITY_RANGE], dtype=np.float32),
            (self.identity_count, 1),
        )
        self.expression_limits = np.tile(
            np.asarray([EXPRESSION_RANGE], dtype=np.float32),
            (len(self.expression_values), 1),
        )
        self.root_ty_index = self.pose_names.index(ROOT_TY_NAME)
        self.unsnapped_root_ty = float(self.pose_values[self.root_ty_index])
        self.snap_to_ground = True

        self.source_faces = (
            self.model.character_torch.mesh.faces.detach()
            .cpu()
            .numpy()
            .astype(np.uint32)
        )
        source_joint_indices, source_joint_weights = self.model.get_lbsw()
        self.source_joint_indices = (
            source_joint_indices.detach().cpu().numpy().astype(np.uint16)
        )
        self.source_joint_weights = (
            source_joint_weights.detach().cpu().numpy().astype(np.float32)
        )
        self.source_vertex_count = int(self.source_joint_indices.shape[0])
        self.lod_faces, self.lod_mappings = self._load_lod_assets()
        self.faces = self.lod_faces[self.lod]
        self.joint_names = list(self.model.get_joint_names())
        self.joint_parents = [
            int(value) for value in self.model.character_torch.skeleton.joint_parents
        ]
        self.identity_regions, self.identity_region_shares = (
            self._compute_identity_regions()
        )
        self.expression_names = _load_expression_names(len(self.expression_values))
        self.lod_skinning: dict[int, tuple[np.ndarray, np.ndarray]] = {
            1: self._top_four_skinning(
                self.source_joint_indices,
                self.source_joint_weights,
            )
        }
        self.influence_mapping = self._build_influence_mapping()
        self.revision = 0
        self._rig_mapping_bytes: bytes | None = None
        self._neutral_glb_cache: dict[int, bytes] = {}
        self._blendshape_weights_raw: bytes | None = None
        self._blendshape_weights_gz: bytes | None = None
        self._lod_mapping_cache: dict[int, bytes] = {}
        self._pose_correctives_raw: bytes | None = None
        self._pose_correctives_gz: bytes | None = None
        self.snapshot = self._infer()

    def _load_lod_assets(
        self,
    ) -> tuple[
        dict[int, np.ndarray],
        dict[int, tuple[np.ndarray, np.ndarray]],
    ]:
        if not LOD_TOPOLOGY_PATH.is_file():
            raise FileNotFoundError(
                f"LOD topology data not found at {LOD_TOPOLOGY_PATH}. "
                "Run tools/build_lod_topology.py from the viewer folder."
            )

        lod_faces: dict[int, np.ndarray] = {}
        lod_mappings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        with np.load(LOD_TOPOLOGY_PATH) as topology:
            for lod in SUPPORTED_LODS:
                faces_key = f"lod{lod}_faces"
                if faces_key not in topology:
                    if lod == 1:
                        # Fresh topology without seal cuts: LOD1 is the model.
                        lod_faces[lod] = self.source_faces
                        continue
                    raise ValueError(f"LOD topology data is missing {faces_key}.")
                lod_faces[lod] = topology[faces_key].astype(np.uint32)
                if lod == 1:
                    # LOD1 is the mapping source; it needs no conversion.
                    continue

                mapping_path = LOD_MAPPING_ROOT / f"lod1_to_lod{lod}_mapping.npz"
                if not mapping_path.is_file():
                    raise FileNotFoundError(
                        f"LOD conversion mapping not found at {mapping_path}."
                    )
                with np.load(mapping_path) as mapping:
                    triangle_ids = mapping["triangle_ids"].astype(np.int64)
                    barycentric = mapping["baryc_coords"].astype(np.float32)
                source_indices = self.source_faces[triangle_ids].astype(np.int64)
                if int(lod_faces[lod].max()) >= len(source_indices):
                    raise ValueError(
                        f"LOD {lod} topology references a vertex outside its "
                        "conversion mapping."
                    )
                lod_mappings[lod] = (source_indices, barycentric)
        digest = hashlib.sha1(LOD_TOPOLOGY_PATH.read_bytes()).hexdigest()[:12]
        self.topology_version = digest
        return lod_faces, lod_mappings

    def _compute_identity_regions(self) -> tuple[list[str], list[float]]:
        """Label each identity component by where its displacement concentrates.

        Distributes every vertex squared displacement across anatomical zones
        using the LBS weights, then takes the argmax zone per component. Mirror
        sides are merged into one zone (a bilateral hand component labels as
        "Hands", not arbitrarily left or right).
        """
        zone_index = {zone: i for i, zone in enumerate(IDENTITY_ZONES)}
        joint_zone = np.full(len(self.joint_names), -1, dtype=np.int64)
        for joint, name in enumerate(self.joint_names):
            zone = identity_zone_for_joint(name)
            if zone is not None:
                joint_zone[joint] = zone_index[zone]
        vertex_count, influences = self.source_joint_indices.shape
        zone_weights = np.zeros((vertex_count, len(IDENTITY_ZONES)), dtype=np.float64)
        group_of_influence = joint_zone[self.source_joint_indices]
        valid = group_of_influence >= 0
        rows = np.broadcast_to(
            np.arange(vertex_count, dtype=np.int64)[:, None],
            (vertex_count, influences),
        )[valid]
        np.add.at(
            zone_weights,
            (rows, group_of_influence[valid]),
            self.source_joint_weights.astype(np.float64)[valid],
        )
        row_sums = zone_weights.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0.0] = 1.0
        zone_weights /= row_sums

        deltas = (
            self.model.character_torch.blend_shape.shape_vectors.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        energy = (deltas**2).sum(axis=2)
        zone_energy = energy @ zone_weights
        totals = zone_energy.sum(axis=1, keepdims=True)
        totals[totals <= 0.0] = 1.0
        shares = zone_energy / totals
        best = shares.argmax(axis=1)
        return (
            [IDENTITY_ZONES[i] for i in best],
            [float(shares[k, best[k]]) for k in range(len(best))],
        )

    def _build_influence_mapping(self) -> dict[str, list[str]]:
        num_joints = len(self.joint_names)
        influence_matrix = (
            self.model.get_parameter_transform().detach().cpu().numpy().astype(bool).T
        )
        result: dict[str, list[str]] = {}
        for name, joint_mask in zip(self.pose_names, influence_matrix, strict=False):
            influenced = joint_mask.reshape(num_joints, 7).any(axis=1)
            result[name] = [
                self.joint_names[index] for index in np.flatnonzero(influenced)
            ]
        return result

    @staticmethod
    def _top_four_skinning(
        joint_indices: np.ndarray,
        joint_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(joint_weights, axis=1)[:, -4:][:, ::-1]
        indices = np.take_along_axis(joint_indices, order, axis=1)
        weights = np.take_along_axis(joint_weights, order, axis=1)
        weight_sums = weights.sum(axis=1, keepdims=True)
        if np.any(weight_sums <= 1e-8):
            raise ValueError("Every MHR vertex must have a skin influence.")
        return (
            np.ascontiguousarray(indices, dtype=np.uint16),
            np.ascontiguousarray(weights / weight_sums, dtype=np.float32),
        )

    def _skinning_for_lod(
        self,
        lod: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        cached = self.lod_skinning.get(lod)
        if cached is not None:
            return cached

        source_indices, barycentric = self.lod_mappings[lod]
        vertex_count = len(source_indices)
        joint_count = len(self.joint_names)
        dense_weights = np.zeros(
            (vertex_count, joint_count),
            dtype=np.float32,
        )
        rows = np.arange(vertex_count, dtype=np.int64)[:, np.newaxis]
        for corner in range(3):
            source_vertices = source_indices[:, corner]
            corner_indices = self.source_joint_indices[source_vertices]
            corner_weights = (
                self.source_joint_weights[source_vertices]
                * barycentric[:, corner, np.newaxis]
            )
            np.add.at(
                dense_weights,
                (
                    np.broadcast_to(rows, corner_indices.shape).reshape(-1),
                    corner_indices.astype(np.int64, copy=False).reshape(-1),
                ),
                corner_weights.reshape(-1),
            )
        dense_weights = np.maximum(dense_weights, 0.0)
        dense_indices = np.broadcast_to(
            np.arange(joint_count, dtype=np.uint16),
            dense_weights.shape,
        )
        result = self._top_four_skinning(dense_indices, dense_weights)
        self.lod_skinning[lod] = result
        return result

    def _infer(self) -> GeometrySnapshot:
        started = time.perf_counter()
        source_vertices, skeleton = self._evaluate_model()
        render_vertices = self._render_lod(source_vertices)

        if self.snap_to_ground:
            lowest_y = float(np.min(render_vertices[:, 1]))
            correction = -lowest_y / ROOT_TY_SOURCE_UNITS_PER_VALUE
            if not np.isclose(correction, 0.0, atol=1e-7):
                minimum, maximum = self.pose_limits[self.root_ty_index]
                current = float(self.pose_values[self.root_ty_index])
                self.pose_values[self.root_ty_index] = np.clip(
                    current + correction,
                    minimum,
                    maximum,
                )
                source_vertices, skeleton = self._evaluate_model()
                render_vertices = self._render_lod(source_vertices)

        self.revision += 1
        export_skeleton = skeleton.copy()
        export_skeleton[:, :3] /= 100.0
        return GeometrySnapshot(
            vertices=render_vertices / 100.0,
            joints=export_skeleton[:, :3],
            skeleton_state=export_skeleton,
            revision=self.revision,
            inference_ms=(time.perf_counter() - started) * 1000.0,
            root_ty=float(self.pose_values[self.root_ty_index]),
        )

    def _evaluate_model(self) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            vertices, skeleton = self.model(
                identity_coeffs=torch.from_numpy(
                    self.identity_values[np.newaxis, :].copy()
                ),
                model_parameters=torch.from_numpy(
                    self.pose_values[np.newaxis, :].copy()
                ),
                face_expr_coeffs=torch.from_numpy(
                    self.expression_values[np.newaxis, :].copy()
                ),
                apply_correctives=self.apply_correctives,
            )
        return (
            vertices[0].detach().cpu().numpy().astype(np.float32),
            skeleton[0].detach().cpu().numpy().astype(np.float32),
        )

    def _render_lod(
        self,
        source_vertices: np.ndarray,
        lod: int | None = None,
    ) -> np.ndarray:
        target = self.lod if lod is None else lod
        if target == 1:
            return source_vertices
        source_indices, barycentric = self.lod_mappings[target]
        triangle_vertices = source_vertices[source_indices]
        return np.einsum("ijk,ij->ik", triangle_vertices, barycentric).astype(
            np.float32
        )

    def metadata(self) -> dict[str, object]:
        pose_parameters = []
        for index, name in enumerate(self.pose_names):
            minimum, maximum = self.pose_limits[index]
            managed = index == self.root_ty_index and self.snap_to_ground
            pose_parameters.append(
                {
                    "index": index,
                    "name": name,
                    "label": name,
                    "category": _pose_category(name),
                    "min": float(minimum),
                    "max": float(maximum),
                    "value": float(self.pose_values[index]),
                    "joints": self.influence_mapping.get(name, []),
                    "managed": managed,
                    "lockedReason": ("Managed by Snap to ground" if managed else None),
                }
            )

        return {
            "model": {
                "path": str(self.model_path),
                "topologyVersion": self.topology_version,
                "lod": self.lod,
                "supportedLods": [
                    {
                        "lod": lod,
                        "vertexCount": (
                            int(self.snapshot.vertices.shape[0])
                            if lod == self.lod
                            else (
                                self.source_vertex_count
                                if lod == 1
                                else int(self.lod_mappings[lod][0].shape[0])
                            )
                        ),
                        "faceCount": int(self.lod_faces[lod].shape[0]),
                    }
                    for lod in SUPPORTED_LODS
                ],
                "applyCorrectives": self.apply_correctives,
                "snapToGround": self.snap_to_ground,
                "exportFormats": list(SUPPORTED_EXPORT_FORMATS),
                "riggedExportFormats": list(RIGGED_EXPORT_FORMATS),
                "vertexCount": int(self.snapshot.vertices.shape[0]),
                "faceCount": int(self.faces.shape[0]),
                "jointCount": len(self.joint_names),
            },
            "groups": {
                "pose": pose_parameters,
                "identity": self._latent_parameters(
                    "identity",
                    "Identity",
                    self.identity_values,
                    self.identity_limits,
                    self.identity_regions,
                    self.identity_region_shares,
                ),
                "expression": self._expression_parameters(),
            },
            "jointNames": self.joint_names,
            "jointParents": self.joint_parents,
            "identityRegions": list(IDENTITY_ZONES),
            "expressionRegions": list(EXPRESSION_ZONES),
            "revision": self.revision,
        }

    def _expression_parameters(self) -> list[dict[str, object]]:
        entries = self._latent_parameters(
            "expression",
            "Expression",
            self.expression_values,
            self.expression_limits,
        )
        for index, entry in enumerate(entries):
            action = (
                self.expression_names[index]
                if index < len(self.expression_names)
                else None
            )
            if action is None:
                continue
            entry["name"] = action
            entry["label"] = prettify_expression_name(action)
            entry["region"] = expression_zone_for_action(action)
        return entries

    @staticmethod
    def _latent_parameters(
        name_prefix: str,
        label_prefix: str,
        values: np.ndarray,
        limits: np.ndarray,
        regions: list[str] | None = None,
        shares: list[float] | None = None,
    ) -> list[dict[str, object]]:
        parameters = []
        for index, value in enumerate(values):
            entry: dict[str, object] = {
                "index": index,
                "name": f"{name_prefix}_{index + 1:02d}",
                "label": f"{label_prefix} component {index + 1:02d}",
                "category": label_prefix,
                "min": float(limits[index, 0]),
                "max": float(limits[index, 1]),
                "value": float(value),
                "joints": [],
            }
            if regions is not None:
                entry["region"] = regions[index]
            if shares is not None:
                entry["regionShare"] = shares[index]
            parameters.append(entry)
        return parameters

    def rig_mapping_bytes(self) -> bytes:
        if self._rig_mapping_bytes is not None:
            return self._rig_mapping_bytes

        with self.lock:
            tp = (
                self.model.get_parameter_transform()[
                    : 127 * 7, : len(self.pose_names)
                ]
                .detach()
                .cpu()
                .numpy()
                .astype("<f4", copy=False)
            )
            skel = self.model.character_torch.skeleton
            offsets = (
                skel.joint_translation_offsets.detach()
                .cpu()
                .numpy()
                .astype("<f4", copy=False)
            )
            prerot = (
                skel.joint_prerotations.detach()
                .cpu()
                .numpy()
                .astype("<f4", copy=False)
            )
            header = struct.pack(
                "<III",
                tp.shape[0],
                tp.shape[1],
                len(self.joint_names),
            )
            self._rig_mapping_bytes = (
                header + tp.tobytes() + offsets.tobytes() + prerot.tobytes()
            )
            return self._rig_mapping_bytes

    def neutral_rigged_glb(self, lod: int) -> bytes:
        if lod not in SUPPORTED_LODS:
            raise ValueError(f"Unsupported LOD: {lod}")
        cached = self._neutral_glb_cache.get(lod)
        if cached is not None:
            return cached

        with torch.inference_mode():
            neutral_v, neutral_s = self.model(
                torch.zeros(1, self.identity_count),
                torch.zeros(1, len(self.pose_names)),
                torch.zeros(1, 72),
                apply_correctives=False,
            )
        v0 = neutral_v[0].detach().cpu().numpy().astype(np.float32)
        s0 = neutral_s[0].detach().cpu().numpy().astype(np.float32)
        s0[:, :3] /= 100.0

        render_v = self._render_lod(v0, lod) / 100.0
        faces = self.lod_faces[lod]
        joint_indices, joint_weights = self._skinning_for_lod(lod)
        rig = RigData(
            joint_names=tuple(self.joint_names),
            joint_parents=np.asarray(self.joint_parents, dtype=np.int32),
            skeleton_state=s0,
            joint_indices=joint_indices,
            joint_weights=joint_weights,
        )
        result = export_mesh(render_v, faces, "glb", rig)
        self._neutral_glb_cache[lod] = result.data
        return result.data

    def unposed_vertices(self) -> np.ndarray:
        with torch.inference_mode():
            v, _ = self.model(
                identity_coeffs=torch.from_numpy(
                    self.identity_values[np.newaxis, :].copy()
                ),
                model_parameters=torch.zeros(
                    1,
                    len(self.pose_names),
                    dtype=torch.float32,
                ),
                face_expr_coeffs=torch.from_numpy(
                    self.expression_values[np.newaxis, :].copy()
                ),
                apply_correctives=False,
            )
        v0 = v[0].detach().cpu().numpy().astype(np.float32)
        return self._render_lod(v0) / 100.0

    def blendshape_weights_bytes(self, compressed: bool = True) -> bytes:
        if self._blendshape_weights_raw is None:
            with self.lock:
                if self._blendshape_weights_raw is None:
                    base = (
                        self.model.character_torch.blend_shape.base_shape.detach()
                        .cpu()
                        .numpy()
                        / 100.0
                    )
                    bs = (
                        self.model.character_torch.blend_shape.shape_vectors.detach()
                        .cpu()
                        .numpy()
                        / 100.0
                    )
                    fe = (
                        self.model.face_expressions_model.shape_vectors.detach()
                        .cpu()
                        .numpy()
                        / 100.0
                    )
                    all_bs = np.concatenate([bs, fe], axis=0)
                    all_bs_v = np.transpose(all_bs, (1, 0, 2)).astype(np.float32)

                    v_count = int(base.shape[0])
                    neutral_v4 = np.zeros((v_count, 4), dtype="<f4")
                    neutral_v4[:, :3] = base

                    deltas_v4 = np.zeros(
                        (v_count, int(all_bs.shape[0]), 4),
                        dtype="<f4",
                    )
                    deltas_v4[:, :, :3] = all_bs_v

                    header = struct.pack(
                        "<III",
                        v_count,
                        int(bs.shape[0]),
                        int(fe.shape[0]),
                    )
                    raw = header + neutral_v4.tobytes() + deltas_v4.tobytes()
                    self._blendshape_weights_raw = raw
                    self._blendshape_weights_gz = gzip.compress(
                        raw,
                        compresslevel=6,
                    )

        return (
            self._blendshape_weights_gz
            if compressed
            else self._blendshape_weights_raw
        )

    def lod_mapping_bytes(self, lod: int) -> bytes:
        if lod not in SUPPORTED_LODS:
            raise ValueError(f"Unsupported LOD: {lod}")
        cached = self._lod_mapping_cache.get(lod)
        if cached is not None:
            return cached
        if lod == 1:
            res = struct.pack("<I", 0)
            self._lod_mapping_cache[lod] = res
            return res
        source_indices, barycentric = self.lod_mappings[lod]
        header = struct.pack("<I", int(len(source_indices)))
        res = (
            header
            + source_indices.astype("<u4", copy=False).tobytes()
            + barycentric.astype("<f4", copy=False).tobytes()
        )
        self._lod_mapping_cache[lod] = res
        return res

    def pose_corrective_weights_bytes(self, compressed: bool = True) -> bytes:
        if self._pose_correctives_raw is None:
            with self.lock:
                if self._pose_correctives_raw is None:
                    pc = self.model.pose_correctives_model
                    l0 = getattr(pc.pose_dirs_predictor, "0")
                    l0_indices = l0.sparse_indices.detach().cpu().numpy()
                    l0_weights = (
                        l0.sparse_weight.detach().cpu().numpy().astype(np.float32)
                    )

                    l2 = getattr(pc.pose_dirs_predictor, "2")
                    w2 = l2.weight.detach().cpu().numpy()
                    csr = sp.csr_matrix(w2)
                    indptr = csr.indptr.astype(np.uint32)
                    indices = csr.indices.astype(np.uint16)
                    data = csr.data.astype(np.float32)

                    header = struct.pack(
                        "<IIIIIII",
                        0x4350484D,
                        int(len(l0_weights)),
                        750,
                        3000,
                        int(w2.shape[0]),
                        int(w2.shape[1]),
                        int(len(data)),
                    )
                    raw = (
                        header
                        + l0_indices[0].astype("<u2", copy=False).tobytes()
                        + l0_indices[1].astype("<u2", copy=False).tobytes()
                        + l0_weights.astype("<f4", copy=False).tobytes()
                        + indptr.astype("<u4", copy=False).tobytes()
                        + indices.astype("<u2", copy=False).tobytes()
                        + data.astype("<f4", copy=False).tobytes()
                    )
                    self._pose_correctives_raw = raw
                    self._pose_correctives_gz = gzip.compress(
                        raw, compresslevel=6
                    )

        return (
            self._pose_correctives_gz
            if compressed
            else self._pose_correctives_raw
        )

    def configure(self, payload: dict[str, object]) -> GeometrySnapshot:
        requested_lod = payload.get("lod", self.lod)
        try:
            lod = int(requested_lod)
        except (TypeError, ValueError) as error:
            raise ValueError("LOD must be an integer from 0 through 6.") from error
        if lod not in SUPPORTED_LODS:
            raise ValueError("LOD must be an integer from 0 through 6.")

        apply_correctives = payload.get("applyCorrectives", self.apply_correctives)
        if not isinstance(apply_correctives, bool):
            raise ValueError("applyCorrectives must be true or false.")

        snap_to_ground = payload.get("snapToGround", self.snap_to_ground)
        if not isinstance(snap_to_ground, bool):
            raise ValueError("snapToGround must be true or false.")

        with self.lock:
            changed = (
                lod != self.lod
                or apply_correctives != self.apply_correctives
                or snap_to_ground != self.snap_to_ground
            )
            if snap_to_ground != self.snap_to_ground:
                if snap_to_ground:
                    self.unsnapped_root_ty = float(self.pose_values[self.root_ty_index])
                else:
                    self.pose_values[self.root_ty_index] = self.unsnapped_root_ty
            self.lod = lod
            self.faces = self.lod_faces[lod]
            self.apply_correctives = apply_correctives
            self.snap_to_ground = snap_to_ground
            if changed:
                self.snapshot = self._infer()
            return self.snapshot

    def update(self, payload: dict[str, object]) -> GeometrySnapshot:
        group = str(payload.get("group", "pose"))
        action = str(payload.get("action", "set"))
        index_value = payload.get("index", 0)

        try:
            index = int(index_value)
        except (TypeError, ValueError) as error:
            raise ValueError("Parameter index must be an integer.") from error

        with self.lock:
            values, limits = self._group_state(group)

            if action in {"set", "reset_current"}:
                if index < 0 or index >= len(values):
                    raise ValueError(f"Parameter index {index} is out of range.")
                if (
                    group == "pose"
                    and index == self.root_ty_index
                    and self.snap_to_ground
                ):
                    raise ValueError(
                        "root_ty is managed while Snap to ground is enabled."
                    )
                requested = 0.0
                if action == "set":
                    try:
                        requested = float(payload.get("value", 0.0))
                    except (TypeError, ValueError) as error:
                        raise ValueError("Parameter value must be numeric.") from error
                values[index] = np.clip(requested, *limits[index])
                if group == "pose" and index == self.root_ty_index:
                    self.unsnapped_root_ty = float(values[index])
            elif action == "reset_group":
                values.fill(0.0)
                if group == "pose":
                    self.unsnapped_root_ty = 0.0
            elif action == "set_group":
                try:
                    incoming = np.asarray(payload.get("values", []), dtype=np.float32)
                except (TypeError, ValueError) as error:
                    raise ValueError("Group values must be numeric.") from error
                if incoming.shape != values.shape:
                    raise ValueError(
                        f"Expected {len(values)} values for {group}."
                    )
                values[:] = np.clip(incoming, limits[:, 0], limits[:, 1])
                if group == "pose":
                    self.unsnapped_root_ty = float(values[self.root_ty_index])
            elif action == "reset_all":
                self.pose_values.fill(0.0)
                self.identity_values.fill(0.0)
                self.expression_values.fill(0.0)
                self.unsnapped_root_ty = 0.0
            else:
                raise ValueError(f"Unknown action: {action}")

            self.snapshot = self._infer()
            return self.snapshot

    def _group_state(self, group: str) -> tuple[np.ndarray, np.ndarray]:
        if group == "pose":
            return self.pose_values, self.pose_limits
        if group == "identity":
            return self.identity_values, self.identity_limits
        if group == "expression":
            return self.expression_values, self.expression_limits
        raise ValueError(f"Unknown parameter group: {group}")


def _geometry_response(snapshot: GeometrySnapshot) -> Response:
    response = Response(snapshot.to_bytes(), mimetype="application/octet-stream")
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-MHR-Revision"] = str(snapshot.revision)
    response.headers["X-MHR-Inference-Ms"] = f"{snapshot.inference_ms:.3f}"
    response.headers["X-MHR-Root-TY"] = f"{snapshot.root_ty:.8f}"
    return response


def _json_request() -> dict[str, object]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON request body.")
    return payload


def list_mixamo_clips(root: Path) -> list[dict[str, object]]:
    """List loadable animation clips in a folder (flat, no recursion)."""
    if not root.is_dir():
        return []
    clips = []
    for entry in root.iterdir():
        if not entry.is_file() or entry.suffix.lower() not in MIXAMO_EXTENSIONS:
            continue
        stat = entry.stat()
        clips.append({"name": entry.name, "bytes": stat.st_size, "mtime": stat.st_mtime})
    clips.sort(key=lambda clip: clip["name"])
    return clips


def resolve_mixamo_file(root: Path, name: str) -> Path | None:
    """Resolve a clip name to a file, or None when missing/forbidden.

    Flat namespace only: subpaths, traversal, and non-clip extensions are
    rejected even before the filesystem is touched.
    """
    if Path(name).name != name:
        return None
    if Path(name).suffix.lower() not in MIXAMO_EXTENSIONS:
        return None
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def list_hdri_files(root: Path) -> list[dict[str, object]]:
    """List loadable 360 environment files in a folder (flat, no recursion)."""
    if not root.is_dir():
        return []
    files = []
    for entry in root.iterdir():
        if not entry.is_file() or entry.suffix.lower() not in HDRI_EXTENSIONS:
            continue
        stat = entry.stat()
        files.append({"name": entry.name, "bytes": stat.st_size, "mtime": stat.st_mtime})
    files.sort(key=lambda item: item["name"])
    return files


def resolve_hdri_file(root: Path, name: str) -> Path | None:
    """Resolve an environment name to a file, or None when missing/forbidden.

    Flat namespace only: subpaths, traversal, and non-HDRI extensions are
    rejected even before the filesystem is touched.
    """
    if Path(name).name != name:
        return None
    if Path(name).suffix.lower() not in HDRI_EXTENSIONS:
        return None
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def parse_pose_meta(path: Path) -> dict[str, str]:
    """Parse a MakeHuman .meta sidecar (one `key value` pair per line)."""
    fields: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fields
    for line in text.splitlines():
        key, sep, value = line.strip().partition(" ")
        if sep and key and value.strip():
            fields[key] = value.strip()
    return fields


def list_pose_files(root: Path) -> list[dict[str, object]]:
    """List BVH pose presets below a folder (recursive, with .meta sidecars)."""
    if not root.is_dir():
        return []
    base = root.resolve()
    poses = []
    for entry in sorted(base.rglob("*.bvh")):
        if not entry.is_file():
            continue
        meta = parse_pose_meta(entry.with_suffix(".meta"))
        stat = entry.stat()
        poses.append({
            "id": entry.relative_to(base).as_posix(),
            "name": meta.get("name") or entry.stem,
            "set": entry.relative_to(base).parts[0],
            "description": meta.get("description", ""),
            "license": meta.get("license", ""),
            "copyright": meta.get("copyright", ""),
            "hasThumb": entry.with_suffix(".thumb").is_file(),
            "bytes": stat.st_size,
        })
    return poses


def resolve_pose_file(root: Path, name: str) -> Path | None:
    """Resolve a pose id to a file, or None when missing/forbidden.

    Nested ids (set/pose/file.bvh) are allowed; absolute paths, traversal
    above the root, and non-BVH extensions are rejected.
    """
    if Path(name).suffix.lower() not in POSE_EXTENSIONS:
        return None
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def list_mhasset_files(root: Path) -> list[dict[str, object]]:
    """List fitted-asset meshes below a folder (recursive .glb/.obj)."""
    if not root.is_dir():
        return []
    base = root.resolve()
    assets = []
    entries = sorted(base.rglob("*.glb")) + sorted(base.rglob("*.obj"))
    for entry in entries:
        if not entry.is_file():
            continue
        rel = entry.relative_to(base)
        stat = entry.stat()
        # Thumb: same-stem first, else any .thumb next to the mesh (the
        # modelled sets name thumbs after the folder, not the .obj stem).
        thumb_rel = None
        same_stem = entry.with_suffix(".thumb")
        if same_stem.is_file():
            thumb_rel = same_stem.relative_to(base).as_posix()
        else:
            others = sorted(entry.parent.glob("*.thumb"))
            if others:
                thumb_rel = others[0].relative_to(base).as_posix()
        assets.append({
            "id": rel.as_posix(),
            "name": entry.stem,
            "category": rel.parts[0] if len(rel.parts) > 1 else "",
            "hasThumb": thumb_rel is not None,
            "thumb": thumb_rel,
            "bytes": stat.st_size,
        })
    return assets


def resolve_mhasset_file(root: Path, name: str) -> Path | None:
    """Resolve an asset id to a file, or None when missing/forbidden.

    Nested ids are allowed; absolute paths, traversal above the root,
    and non-asset extensions are rejected.
    """
    if Path(name).suffix.lower() not in MHASSET_EXTENSIONS:
        return None
    candidate = (root / name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def hdri_thumbnail_bytes(source: Path) -> bytes | None:
    """Tonemapped JPEG thumbnail for an EXR/HDR pano, cached next to it.

    Returns None when OpenCV is unavailable or the file cannot be decoded;
    the client falls back to a styled label tile.
    """
    import os

    # Local trusted files: opt into OpenCV's EXR codec (off by default).
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2
    except ImportError:
        return None
    cache_dir = source.parent / ".thumbs"
    cached = cache_dir / f"{source.name}.jpg"
    try:
        if cached.is_file() and cached.stat().st_mtime >= source.stat().st_mtime:
            return cached.read_bytes()
        img = cv2.imread(str(source), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            return None
        import numpy as _np

        linear = img.astype(_np.float32)
        linear = _np.nan_to_num(linear, nan=0.0, posinf=64.0, neginf=0.0)
        linear = _np.clip(linear, 0.0, None)
        # Reinhard + gamma: cheap, stable across sun/sky and studio files.
        mapped = linear / (1.0 + linear)
        mapped = _np.clip(mapped, 0.0, 1.0) ** (1.0 / 2.2)
        small = cv2.resize(mapped, (320, 160), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", (small * 255).astype(_np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not ok:
            return None
        cache_dir.mkdir(exist_ok=True)
        cached.write_bytes(buf.tobytes())
        return buf.tobytes()
    except Exception:
        return None


def create_app(
    model_path: Path = DEFAULT_MODEL_PATH,
    mixamo_root: Path = DEFAULT_MIXAMO_ROOT,
    hdri_root: Path = DEFAULT_HDRI_ROOT,
    pose_root: Path = DEFAULT_POSE_ROOT,
    mhasset_root: Path = DEFAULT_MHASSET_ROOT,
) -> Flask:
    engine = MHREngine(model_path.resolve())
    app = Flask(__name__, static_folder=None)

    @app.get("/")
    def index() -> Response:
        return send_from_directory(STATIC_ROOT, "index.html")

    @app.get("/static/<path:filename>")
    def static_file(filename: str) -> Response:
        return send_from_directory(STATIC_ROOT, filename)

    @app.get("/vendor/three/<path:filename>")
    def three_vendor(filename: str) -> Response:
        if not THREE_ROOT.is_dir():
            return (
                jsonify(
                    {
                        "error": "Three.js is not installed. Run npm install in the viewer folder."
                    }
                ),
                503,
            )
        return send_from_directory(THREE_ROOT, filename)

    @app.get("/vendor/dockview/<path:filename>")
    def dockview_vendor(filename: str) -> Response:
        if not DOCKVIEW_ROOT.is_dir():
            return (
                jsonify(
                    {
                        "error": "dockview is not installed. Run npm install in the viewer folder."
                    }
                ),
                503,
            )
        # Python mimetypes doesn't know .mjs on all platforms; browsers refuse
        # module scripts served as anything but JavaScript, so pin it.
        mimetype = "text/javascript" if filename.endswith((".js", ".mjs")) else None
        return send_from_directory(DOCKVIEW_ROOT, filename, mimetype=mimetype)

    @app.get("/vendor/dockview-core/<path:filename>")
    def dockview_core_vendor(filename: str) -> Response:
        # dockview's ESM entry re-exports dockview-core (bare specifier).
        if not DOCKVIEW_CORE_ROOT.is_dir():
            return (
                jsonify(
                    {
                        "error": "dockview-core is not installed. Run npm install in the viewer folder."
                    }
                ),
                503,
            )
        mimetype = "text/javascript" if filename.endswith((".js", ".mjs")) else None
        return send_from_directory(DOCKVIEW_CORE_ROOT, filename, mimetype=mimetype)

    @app.get("/api/mixamo/clips")
    def mixamo_clips() -> Response:
        root = Path(current_app.config["MIXAMO_ROOT"])
        return jsonify(
            {
                "root": str(root),
                "available": root.is_dir(),
                "clips": list_mixamo_clips(root),
            }
        )

    @app.get("/api/mixamo/file/<path:name>")
    def mixamo_file(name: str) -> Response:
        root = Path(current_app.config["MIXAMO_ROOT"])
        resolved = resolve_mixamo_file(root, name)
        if resolved is None:
            return jsonify({"error": "Mixamo clip not found."}), 404
        return send_from_directory(root, resolved.name)

    @app.get("/api/hdri/list")
    def hdri_list() -> Response:
        root = Path(current_app.config["HDRI_ROOT"])
        return jsonify(
            {
                "root": str(root),
                "available": root.is_dir(),
                "files": list_hdri_files(root),
            }
        )

    @app.get("/api/hdri/file/<path:name>")
    def hdri_file(name: str) -> Response:
        root = Path(current_app.config["HDRI_ROOT"])
        resolved = resolve_hdri_file(root, name)
        if resolved is None:
            return jsonify({"error": "HDRI environment not found."}), 404
        return send_from_directory(root, resolved.name)

    @app.get("/api/hdri/thumb/<path:name>")
    def hdri_thumb(name: str) -> Response:
        root = Path(current_app.config["HDRI_ROOT"])
        resolved = resolve_hdri_file(root, name)
        if resolved is None:
            return jsonify({"error": "HDRI environment not found."}), 404
        data = hdri_thumbnail_bytes(resolved)
        if data is None:
            return jsonify({"error": "Thumbnail unavailable for this file."}), 503
        return Response(data, mimetype="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/pose/list")
    def pose_list() -> Response:
        root = Path(current_app.config["POSE_ROOT"])
        return jsonify(
            {
                "root": str(root),
                "available": root.is_dir(),
                "poses": list_pose_files(root),
            }
        )

    @app.get("/api/pose/file/<path:name>")
    def pose_file(name: str) -> Response:
        root = Path(current_app.config["POSE_ROOT"])
        resolved = resolve_pose_file(root, name)
        if resolved is None:
            return jsonify({"error": "Pose preset not found."}), 404
        return send_from_directory(resolved.parent, resolved.name)

    @app.get("/api/pose/thumb/<path:name>")
    def pose_thumb(name: str) -> Response:
        root = Path(current_app.config["POSE_ROOT"])
        resolved = resolve_pose_file(root, name)
        if resolved is None:
            return jsonify({"error": "Pose preset not found."}), 404
        thumb = resolved.with_suffix(".thumb")
        try:
            data = thumb.read_bytes()
        except OSError:
            data = b""
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            return jsonify({"error": "Thumbnail unavailable for this pose."}), 503
        return Response(data, mimetype="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/mhasset/list")
    def mhasset_list() -> Response:
        root = Path(current_app.config["MHASSET_ROOT"])
        return jsonify(
            {
                "root": str(root),
                "available": root.is_dir(),
                "assets": list_mhasset_files(root),
            }
        )

    @app.get("/api/mhasset/file/<path:name>")
    def mhasset_file(name: str) -> Response:
        root = Path(current_app.config["MHASSET_ROOT"])
        resolved = resolve_mhasset_file(root, name)
        if resolved is None:
            return jsonify({"error": "Asset file not found."}), 404
        if resolved.suffix.lower() == ".thumb":
            try:
                data = resolved.read_bytes()
            except OSError:
                data = b""
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                return jsonify({"error": "Thumbnail unavailable for this asset."}), 503
            return Response(data, mimetype="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})
        return send_from_directory(resolved.parent, resolved.name)

    @app.get("/api/metadata")
    def metadata() -> Response:
        return jsonify(engine.metadata())

    @app.get("/api/rig-mapping")
    def rig_mapping() -> Response:
        payload = engine.rig_mapping_bytes()
        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "public, max-age=86400"
        return response

    @app.get("/api/model/rigged-glb")
    def neutral_rigged_glb() -> tuple[Response, int] | Response:
        lod_arg = request.args.get("lod", engine.lod)
        try:
            lod = int(lod_arg)
            if lod not in SUPPORTED_LODS:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid LOD parameter."}), 400

        with engine.lock:
            payload = engine.neutral_rigged_glb(lod)
        response = Response(payload, mimetype="model/gltf-binary")
        response.headers["Cache-Control"] = "public, max-age=86400"
        response.headers["X-MHR-LOD"] = str(lod)
        return response

    @app.get("/api/weights/blendshapes")
    def blendshape_weights() -> Response:
        accept_encoding = request.headers.get("Accept-Encoding", "")
        supports_gzip = "gzip" in accept_encoding.lower()
        payload = engine.blendshape_weights_bytes(compressed=supports_gzip)
        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "public, max-age=86400"
        if supports_gzip:
            response.headers["Content-Encoding"] = "gzip"
        return response

    @app.get("/api/weights/pose-correctives")
    def pose_corrective_weights() -> Response:
        accept_encoding = request.headers.get("Accept-Encoding", "")
        supports_gzip = "gzip" in accept_encoding.lower()
        payload = engine.pose_corrective_weights_bytes(compressed=supports_gzip)
        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "public, max-age=86400"
        if supports_gzip:
            response.headers["Content-Encoding"] = "gzip"
        return response

    @app.get("/api/lod-mapping")
    def lod_mapping() -> tuple[Response, int] | Response:
        lod_arg = request.args.get("lod", engine.lod)
        try:
            lod = int(lod_arg)
            if lod not in SUPPORTED_LODS:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"error": "Invalid LOD parameter."}), 400

        with engine.lock:
            payload = engine.lod_mapping_bytes(lod)
        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "public, max-age=86400"
        response.headers["X-MHR-LOD"] = str(lod)
        return response

    @app.get("/api/topology")
    def topology() -> Response:
        with engine.lock:
            payload = engine.faces.astype("<u4", copy=False).tobytes()
        response = Response(payload, mimetype="application/octet-stream")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/geometry")
    def geometry() -> Response:
        if request.args.get("unposed") in ("1", "true", "yes"):
            with engine.lock:
                unposed_v = engine.unposed_vertices()
            response = Response(
                unposed_v.astype("<f4", copy=False).tobytes(),
                mimetype="application/octet-stream",
            )
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-MHR-Revision"] = str(engine.revision)
            response.headers["X-MHR-Inference-Ms"] = f"{engine.snapshot.inference_ms:.3f}"
            response.headers["X-MHR-Root-TY"] = f"{engine.snapshot.root_ty:.8f}"
            return response
        return _geometry_response(engine.snapshot)

    @app.post("/api/deform")
    def deform() -> tuple[Response, int] | Response:
        try:
            snapshot = engine.update(_json_request())
        except ValueError:
            return jsonify({"error": "Invalid deformation request."}), 400
        if request.args.get("nogeometry") in ("1", "true", "yes"):
            response = Response(status=204)
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-MHR-Revision"] = str(snapshot.revision)
            response.headers["X-MHR-Inference-Ms"] = (
                f"{snapshot.inference_ms:.3f}"
            )
            response.headers["X-MHR-Root-TY"] = f"{snapshot.root_ty:.8f}"
            return response
        if request.args.get("unposed") in ("1", "true", "yes"):
            with engine.lock:
                unposed_v = engine.unposed_vertices()
            response = Response(
                unposed_v.astype("<f4", copy=False).tobytes(),
                mimetype="application/octet-stream",
            )
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-MHR-Revision"] = str(snapshot.revision)
            response.headers["X-MHR-Inference-Ms"] = f"{snapshot.inference_ms:.3f}"
            response.headers["X-MHR-Root-TY"] = f"{snapshot.root_ty:.8f}"
            return response
        return _geometry_response(snapshot)

    @app.post("/api/configure")
    def configure() -> tuple[Response, int] | Response:
        try:
            engine.configure(_json_request())
        except ValueError:
            return jsonify({"error": "Invalid configuration request."}), 400
        return jsonify(engine.metadata())

    @app.get("/api/export/<file_format>")
    def export_current_mesh(file_format: str) -> tuple[Response, int] | Response:
        if "ground" in request.args:
            return (
                jsonify(
                    {
                        "error": (
                            "The ground query parameter has been removed. Configure "
                            "snapToGround before exporting."
                        )
                    }
                ),
                400,
            )

        try:
            normalized_format = normalize_export_format(file_format)
        except ValueError:
            return jsonify({"error": "Unsupported export format."}), 400

        with engine.lock:
            vertices = engine.snapshot.vertices.copy()
            faces = engine.faces.copy()
            lod = engine.lod
            revision = engine.revision
            snap_to_ground = engine.snap_to_ground
            rig = None
            if normalized_format in RIGGED_EXPORT_FORMATS:
                joint_indices, joint_weights = engine._skinning_for_lod(lod)
                rig = RigData(
                    joint_names=tuple(engine.joint_names),
                    joint_parents=np.asarray(
                        engine.joint_parents,
                        dtype=np.int32,
                    ),
                    skeleton_state=engine.snapshot.skeleton_state.copy(),
                    joint_indices=joint_indices.copy(),
                    joint_weights=joint_weights.copy(),
                )
        try:
            result = export_mesh(vertices, faces, normalized_format, rig)
        except ValueError:
            return jsonify({"error": "Invalid mesh export request."}), 400
        except RuntimeError:
            return jsonify({"error": "Mesh export is temporarily unavailable."}), 503

        ground_suffix = "-grounded" if snap_to_ground else ""
        filename = f"mhr-lod{lod}-r{revision}{ground_suffix}.{result.extension}"
        response = Response(result.data, mimetype=result.mimetype)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
        response.headers["X-MHR-LOD"] = str(lod)
        response.headers["X-MHR-Revision"] = str(revision)
        response.headers["X-MHR-Snap-To-Ground"] = str(snap_to_ground).lower()
        return response

    @app.get("/health")
    def health() -> Response:
        return jsonify(
            {
                "status": "ok",
                "model": str(engine.model_path),
                "vertices": int(engine.snapshot.vertices.shape[0]),
                "faces": int(engine.faces.shape[0]),
                "joints": int(engine.snapshot.joints.shape[0]),
                "lod": engine.lod,
                "applyCorrectives": engine.apply_correctives,
                "snapToGround": engine.snap_to_ground,
                "revision": engine.revision,
            }
        )

    app.config["MHR_ENGINE"] = engine
    app.config["MIXAMO_ROOT"] = mixamo_root
    app.config["HDRI_ROOT"] = hdri_root
    app.config["POSE_ROOT"] = pose_root
    app.config["MHASSET_ROOT"] = mhasset_root
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to mhr_model.pt (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--mixamo",
        type=Path,
        default=DEFAULT_MIXAMO_ROOT,
        help=f"Folder of Mixamo .fbx/.glb clips (default: {DEFAULT_MIXAMO_ROOT})",
    )
    parser.add_argument(
        "--hdri",
        type=Path,
        default=DEFAULT_HDRI_ROOT,
        help=f"Folder of 360 .exr/.hdr environments (default: {DEFAULT_HDRI_ROOT})",
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=DEFAULT_POSE_ROOT,
        help=f"Folder of BVH pose presets (default: {DEFAULT_POSE_ROOT})",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8891)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.model, args.mixamo, args.hdri, args.poses)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"MHR Three.js Viewer: {url}")
    print(f"Model: {args.model.resolve()}")
    print(f"Mixamo: {args.mixamo.resolve()}")
    print(f"HDRI: {args.hdri.resolve()}")
    print(f"Poses: {args.poses.resolve()}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
