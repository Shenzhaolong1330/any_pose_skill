from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .models import PositionEstimate


class TransformError(ValueError):
    """Raised when calibration transforms cannot be loaded or applied."""


@dataclass(frozen=True)
class RigidTransform:
    matrix: np.ndarray
    name: str

    @classmethod
    def from_spec(cls, name: str, spec: dict[str, Any]) -> "RigidTransform":
        if not isinstance(spec, dict):
            raise TransformError(f"{name} must be a YAML object")

        if "matrix" in spec:
            matrix = _matrix4(spec["matrix"], name)
        else:
            translation = _translation(spec.get("translation_m", [0.0, 0.0, 0.0]), name)
            rotation = _rotation(spec, name)
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3] = rotation
            matrix[:3, 3] = translation
        return cls(matrix=matrix, name=name)

    def apply_point(self, point_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        point = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=np.float64)
        transformed = self.matrix @ point
        return float(transformed[0]), float(transformed[1]), float(transformed[2])

    def __matmul__(self, other: "RigidTransform") -> "RigidTransform":
        return RigidTransform(matrix=self.matrix @ other.matrix, name=f"{self.name} * {other.name}")

    def inverse(self, name: str) -> "RigidTransform":
        return RigidTransform(matrix=np.linalg.inv(self.matrix), name=name)


def position_base_from_calibration(
    position: PositionEstimate,
    *,
    enabled: bool,
    calibration_file: str | None,
    active_camera: str,
    config_path: str | Path,
    position_anchor: str,
) -> dict[str, Any]:
    base = {
        "available": False,
        "enabled": bool(enabled),
        "active_camera": active_camera,
        "calibration_file": calibration_file,
        "position_anchor": position_anchor,
        "convention": "parent_T_child maps p_child to p_parent: p_parent = R * p_child + t",
    }
    if not enabled:
        return {**base, "reason": "calibration.enabled is false"}
    if not calibration_file:
        return {**base, "reason": "calibration.file is not set"}

    try:
        calibration_path = _resolve_relative(calibration_file, Path(config_path).parent)
        raw = _load_yaml(calibration_path)
        transform, metadata = base_T_camera_from_calibration(
            raw,
            active_camera=active_camera,
            calibration_path=calibration_path,
        )
        point_base = transform.apply_point((position.x_m, position.y_m, position.z_m))
    except TransformError as exc:
        return {**base, "reason": str(exc)}

    return {
        **base,
        "available": True,
        "reason": None,
        "frame": metadata["base_frame"],
        "camera_frame": metadata["camera_frame"],
        "x_m": point_base[0],
        "y_m": point_base[1],
        "z_m": point_base[2],
        "source_position_camera_m": {
            "x": position.x_m,
            "y": position.y_m,
            "z": position.z_m,
        },
        "transform_chain": metadata["transform_chain"],
        "calibration_file": str(metadata["calibration_file"]),
    }


def base_T_camera_from_calibration(
    raw: dict[str, Any],
    *,
    active_camera: str,
    calibration_path: Path,
) -> tuple[RigidTransform, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise TransformError(f"calibration file must contain a YAML object: {calibration_path}")

    cameras = raw.get("cameras")
    if not isinstance(cameras, dict):
        raise TransformError("calibration file must contain a cameras object")

    camera_name = active_camera or str(raw.get("active_camera", "head"))
    camera = cameras.get(camera_name)
    if not isinstance(camera, dict):
        raise TransformError(f"camera {camera_name!r} not found in calibration file")

    frames = raw.get("frames") if isinstance(raw.get("frames"), dict) else {}
    base_frame = str(camera.get("base_frame", frames.get("base", "base")))
    camera_frame = str(
        camera.get("camera_frame", frames.get("camera", "realsense_color_optical_frame"))
    )

    if "base_T_camera" in camera:
        transform = RigidTransform.from_spec("base_T_camera", camera["base_T_camera"])
        chain = ["base_T_camera"]
    elif "camera_T_base" in camera:
        transform = RigidTransform.from_spec(
            "camera_T_base", camera["camera_T_base"]
        ).inverse("base_T_camera")
        chain = ["inverse(camera_T_base)"]
    elif "flange_T_camera" in camera or "camera_T_flange" in camera:
        if "flange_T_camera" in camera:
            flange_T_camera = RigidTransform.from_spec(
                "flange_T_camera", camera["flange_T_camera"]
            )
            chain_tail = "flange_T_camera"
        else:
            flange_T_camera = RigidTransform.from_spec(
                "camera_T_flange", camera["camera_T_flange"]
            ).inverse("flange_T_camera")
            chain_tail = "inverse(camera_T_flange)"
        base_T_flange = _load_base_T_flange(camera.get("base_T_flange"), calibration_path)
        transform = base_T_flange @ flange_T_camera
        chain = ["base_T_flange", chain_tail]
    else:
        raise TransformError(
            f"camera {camera_name!r} must define base_T_camera or flange_T_camera"
        )

    return transform, {
        "active_camera": camera_name,
        "base_frame": base_frame,
        "camera_frame": camera_frame,
        "transform_chain": chain,
        "calibration_file": calibration_path,
    }


def _load_base_T_flange(value: Any, calibration_path: Path) -> RigidTransform:
    if value is None:
        raise TransformError(
            "wrist camera requires base_T_flange, either inline or as a file reference"
        )
    if isinstance(value, dict) and "file" in value:
        pose_path = _resolve_relative(str(value["file"]), calibration_path.parent)
        pose_raw = _load_yaml(pose_path)
        if isinstance(pose_raw, dict) and "base_T_flange" in pose_raw:
            pose_raw = pose_raw["base_T_flange"]
        elif isinstance(pose_raw, dict) and "flange_T_base" in pose_raw:
            return RigidTransform.from_spec(
                "flange_T_base", pose_raw["flange_T_base"]
            ).inverse("base_T_flange")
        return RigidTransform.from_spec("base_T_flange", pose_raw)
    if isinstance(value, dict) and "flange_T_base" in value:
        return RigidTransform.from_spec("flange_T_base", value["flange_T_base"]).inverse(
            "base_T_flange"
        )
    return RigidTransform.from_spec("base_T_flange", value)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise TransformError(f"calibration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TransformError(f"calibration file must contain a YAML object: {path}")
    return raw


def _resolve_relative(path: str, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _matrix4(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise TransformError(f"{name}.matrix must be 4x4 or length 16")
    return matrix


def _translation(value: Any, name: str) -> np.ndarray:
    translation = np.asarray(value, dtype=np.float64)
    if translation.shape != (3,):
        raise TransformError(f"{name}.translation_m must have exactly 3 values")
    return translation


def _rotation(spec: dict[str, Any], name: str) -> np.ndarray:
    if "rotation_matrix" in spec:
        rotation = np.asarray(spec["rotation_matrix"], dtype=np.float64)
        if rotation.shape != (3, 3):
            raise TransformError(f"{name}.rotation_matrix must be 3x3")
        return rotation
    if "rotation_quat_xyzw" in spec:
        return _quat_xyzw_to_matrix(spec["rotation_quat_xyzw"], name)
    if "rotation_rpy_deg" in spec:
        roll, pitch, yaw = [math.radians(value) for value in _triple(spec["rotation_rpy_deg"], name)]
        return _rpy_to_matrix(roll, pitch, yaw)
    if "rotation_rpy_rad" in spec:
        roll, pitch, yaw = _triple(spec["rotation_rpy_rad"], name)
        return _rpy_to_matrix(roll, pitch, yaw)
    return np.eye(3, dtype=np.float64)


def _triple(value: Any, name: str) -> tuple[float, float, float]:
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (3,):
        raise TransformError(f"{name} rotation RPY must have exactly 3 values")
    return float(values[0]), float(values[1]), float(values[2])


def _quat_xyzw_to_matrix(value: Any, name: str) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,):
        raise TransformError(f"{name}.rotation_quat_xyzw must have exactly 4 values")
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        raise TransformError(f"{name}.rotation_quat_xyzw must not be zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx
