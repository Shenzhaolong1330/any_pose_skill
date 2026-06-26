from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .models import BoundingBox, CameraIntrinsics, DetectionResult, OrientationEstimate, PixelPoint, PositionEstimate


class DepthEstimationError(ValueError):
    """Raised when a bounding box does not contain enough usable depth."""


@dataclass(frozen=True)
class DepthEstimatorConfig:
    min_depth_m: float = 0.05
    max_depth_m: float = 6.0
    inner_ratio: float = 0.7
    min_samples: int = 50
    fallback_min_samples: int = 5
    max_expand_ratio: float = 2.5
    expand_steps: int = 3
    strategy: str = "median"


@dataclass(frozen=True)
class OrientationEstimatorConfig:
    min_depth_m: float = 0.05
    max_depth_m: float = 6.0
    window_radius_px: int = 5
    min_samples: int = 3


def deproject_pixel(
    intrinsics: CameraIntrinsics,
    u_px: float,
    v_px: float,
    depth_m: float,
) -> tuple[float, float, float]:
    x_m = (u_px - intrinsics.ppx) / intrinsics.fx * depth_m
    y_m = (v_px - intrinsics.ppy) / intrinsics.fy * depth_m
    return x_m, y_m, depth_m


def estimate_position_from_depth(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    bbox: BoundingBox,
    config: DepthEstimatorConfig | None = None,
) -> PositionEstimate:
    if config is None:
        config = DepthEstimatorConfig()
    if depth_m.ndim != 2:
        raise DepthEstimationError("depth_m must be a 2D array in meters")

    image_height, image_width = depth_m.shape[:2]
    full_box = bbox.denormalized_if_needed(image_width, image_height).clamp(
        image_width, image_height
    )
    if not full_box.is_valid():
        raise DepthEstimationError(f"invalid bounding box: {full_box}")

    candidates = _depth_candidate_boxes(
        full_box,
        image_width,
        image_height,
        inner_ratio=config.inner_ratio,
        max_expand_ratio=config.max_expand_ratio,
        expand_steps=config.expand_steps,
    )
    best_fallback = None
    samples = np.array([], dtype=np.float32)
    valid_mask = np.zeros((0, 0), dtype=bool)
    roi_origin = (0, 0)
    box_used = full_box
    sampling_quality = "normal"

    for candidate in candidates:
        candidate_samples, candidate_mask, candidate_origin = _extract_valid_depths(
            depth_m, candidate, config.min_depth_m, config.max_depth_m
        )
        if candidate_samples.size >= config.min_samples:
            samples = candidate_samples
            valid_mask = candidate_mask
            roi_origin = candidate_origin
            box_used = candidate
            break
        if (
            best_fallback is None
            and candidate_samples.size >= max(1, config.fallback_min_samples)
        ):
            best_fallback = (candidate_samples, candidate_mask, candidate_origin, candidate)
    else:
        if best_fallback is not None:
            samples, valid_mask, roi_origin, box_used = best_fallback
            sampling_quality = "sparse"

    if samples.size < max(1, config.fallback_min_samples):
        raise DepthEstimationError(
            f"only {samples.size} valid depth samples near bbox; "
            f"need at least {max(1, config.fallback_min_samples)} "
            f"(target={config.min_samples})"
        )

    strategy = config.strategy.lower()
    if strategy == "center":
        u_px, v_px, z_m = _center_depth_or_median(
            depth_m, full_box, samples, config.min_depth_m, config.max_depth_m
        )
    elif strategy == "foreground":
        u_px, v_px, z_m = _foreground_cluster(valid_mask, roi_origin, depth_m, samples)
    elif strategy == "median":
        u_px, v_px = full_box.center
        z_m = float(np.median(samples))
    else:
        raise DepthEstimationError(
            f"unknown depth strategy {config.strategy!r}; use median, foreground, or center"
        )
    if sampling_quality != "normal":
        strategy = f"{strategy}_{sampling_quality}"

    x_m, y_m, z_m = deproject_pixel(intrinsics, u_px, v_px, z_m)
    _, _, x1, y1 = box_used.to_int_roi(image_width, image_height)
    x0, y0, _, _ = box_used.to_int_roi(image_width, image_height)
    roi_area = max(1, (x1 - x0) * (y1 - y0))

    return PositionEstimate(
        x_m=float(x_m),
        y_m=float(y_m),
        z_m=float(z_m),
        u_px=float(u_px),
        v_px=float(v_px),
        depth_m=float(z_m),
        sample_count=int(samples.size),
        valid_fraction=float(samples.size / roi_area),
        strategy=strategy,
        bbox_used=box_used,
    )


def estimate_orientation(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    detection: DetectionResult,
    config: OrientationEstimatorConfig | None = None,
) -> OrientationEstimate | None:
    if config is None:
        config = OrientationEstimatorConfig()
    if detection.head_px is None or detection.tail_px is None:
        return None

    image_height, image_width = depth_m.shape[:2]
    head_px = detection.head_px.denormalized_if_needed(image_width, image_height).clamp(
        image_width, image_height
    )
    tail_px = detection.tail_px.denormalized_if_needed(image_width, image_height).clamp(
        image_width, image_height
    )
    dx = head_px.x - tail_px.x
    dy = head_px.y - tail_px.y
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None

    head_depth, head_samples = _sample_depth_near_point(
        depth_m,
        head_px,
        config.min_depth_m,
        config.max_depth_m,
        config.window_radius_px,
    )
    tail_depth, tail_samples = _sample_depth_near_point(
        depth_m,
        tail_px,
        config.min_depth_m,
        config.max_depth_m,
        config.window_radius_px,
    )

    head_m = None
    tail_m = None
    vector_3d = None
    if (
        head_depth is not None
        and tail_depth is not None
        and head_samples >= config.min_samples
        and tail_samples >= config.min_samples
    ):
        head_m = deproject_pixel(intrinsics, head_px.x, head_px.y, head_depth)
        tail_m = deproject_pixel(intrinsics, tail_px.x, tail_px.y, tail_depth)
        vector = np.array(head_m, dtype=np.float32) - np.array(tail_m, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-6:
            normalized = vector / norm
            vector_3d = (float(normalized[0]), float(normalized[1]), float(normalized[2]))

    return OrientationEstimate(
        head_px=head_px,
        tail_px=tail_px,
        angle_deg_image=float(math.degrees(math.atan2(dy, dx))),
        vector_px_x=float(dx / length),
        vector_px_y=float(dy / length),
        head_depth_m=head_depth,
        tail_depth_m=tail_depth,
        head_m=head_m,
        tail_m=tail_m,
        vector_3d=vector_3d,
        head_sample_count=head_samples,
        tail_sample_count=tail_samples,
    )


def _sample_depth_near_point(
    depth_m: np.ndarray,
    point: PixelPoint,
    min_depth_m: float,
    max_depth_m: float,
    radius_px: int,
) -> tuple[float | None, int]:
    radius_px = max(0, int(radius_px))
    x = int(round(point.x))
    y = int(round(point.y))
    x0 = max(0, x - radius_px)
    y0 = max(0, y - radius_px)
    x1 = min(depth_m.shape[1], x + radius_px + 1)
    y1 = min(depth_m.shape[0], y + radius_px + 1)
    if x1 <= x0 or y1 <= y0:
        return None, 0

    roi = depth_m[y0:y1, x0:x1].astype(np.float32, copy=False)
    valid = roi[np.isfinite(roi) & (roi >= min_depth_m) & (roi <= max_depth_m)]
    if valid.size == 0:
        return None, 0
    return float(np.median(valid)), int(valid.size)


def _depth_candidate_boxes(
    full_box: BoundingBox,
    image_width: int,
    image_height: int,
    *,
    inner_ratio: float,
    max_expand_ratio: float,
    expand_steps: int,
) -> list[BoundingBox]:
    candidates = []
    if inner_ratio < 0.999:
        candidates.append(full_box.shrink(inner_ratio).clamp(image_width, image_height))
    candidates.append(full_box)

    max_expand_ratio = max(1.0, max_expand_ratio)
    expand_steps = max(0, expand_steps)
    if max_expand_ratio > 1.0 and expand_steps > 0:
        for ratio in np.linspace(1.0, max_expand_ratio, expand_steps + 1)[1:]:
            candidates.append(_scale_box(full_box, float(ratio)).clamp(image_width, image_height))

    unique = []
    seen = set()
    for box in candidates:
        key = tuple(round(value, 3) for value in (box.x_min, box.y_min, box.x_max, box.y_max))
        if key not in seen and box.is_valid():
            unique.append(box)
            seen.add(key)
    return unique


def _scale_box(box: BoundingBox, ratio: float) -> BoundingBox:
    ratio = max(0.05, ratio)
    center_x, center_y = box.center
    half_w = box.width * ratio / 2.0
    half_h = box.height * ratio / 2.0
    return BoundingBox(
        center_x - half_w,
        center_y - half_h,
        center_x + half_w,
        center_y + half_h,
    )


def _extract_valid_depths(
    depth_m: np.ndarray,
    box: BoundingBox,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    image_height, image_width = depth_m.shape[:2]
    x0, y0, x1, y1 = box.to_int_roi(image_width, image_height)
    if x1 <= x0 or y1 <= y0:
        return np.array([], dtype=np.float32), np.zeros((0, 0), dtype=bool), (x0, y0)

    roi = depth_m[y0:y1, x0:x1].astype(np.float32, copy=False)
    valid_mask = np.isfinite(roi) & (roi >= min_depth_m) & (roi <= max_depth_m)
    return roi[valid_mask], valid_mask, (x0, y0)


def _center_depth_or_median(
    depth_m: np.ndarray,
    box: BoundingBox,
    samples: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[float, float, float]:
    u_px, v_px = box.center
    u_i = int(round(u_px))
    v_i = int(round(v_px))
    if 0 <= v_i < depth_m.shape[0] and 0 <= u_i < depth_m.shape[1]:
        z_m = float(depth_m[v_i, u_i])
        if np.isfinite(z_m) and min_depth_m <= z_m <= max_depth_m:
            return u_px, v_px, z_m
    return u_px, v_px, float(np.median(samples))


def _foreground_cluster(
    valid_mask: np.ndarray,
    roi_origin: tuple[int, int],
    depth_m: np.ndarray,
    samples: np.ndarray,
) -> tuple[float, float, float]:
    if valid_mask.size == 0:
        raise DepthEstimationError("no valid depth mask available")

    near_depth = float(np.percentile(samples, 20.0))
    tolerance_m = max(0.03, near_depth * 0.08)
    x0, y0 = roi_origin
    roi = depth_m[y0 : y0 + valid_mask.shape[0], x0 : x0 + valid_mask.shape[1]]
    selected = valid_mask & (roi <= near_depth + tolerance_m)

    if int(np.count_nonzero(selected)) < 10:
        selected = valid_mask

    local_v, local_u = np.where(selected)
    selected_depths = roi[selected]
    u_px = float(np.median(local_u) + x0)
    v_px = float(np.median(local_v) + y0)
    z_m = float(np.median(selected_depths))
    return u_px, v_px, z_m
