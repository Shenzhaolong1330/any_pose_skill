from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float

    @classmethod
    def from_realsense(cls, intrinsics: Any) -> "CameraIntrinsics":
        return cls(
            width=int(intrinsics.width),
            height=int(intrinsics.height),
            fx=float(intrinsics.fx),
            fy=float(intrinsics.fy),
            ppx=float(intrinsics.ppx),
            ppy=float(intrinsics.ppy),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def is_valid(self, min_size_px: float = 2.0) -> bool:
        return self.width >= min_size_px and self.height >= min_size_px

    def denormalized_if_needed(self, image_width: int, image_height: int) -> "BoundingBox":
        values = [self.x_min, self.y_min, self.x_max, self.y_max]
        max_value = max(values)
        min_value = min(values)

        if 0.0 <= min_value and max_value <= 1.5:
            return BoundingBox(
                self.x_min * image_width,
                self.y_min * image_height,
                self.x_max * image_width,
                self.y_max * image_height,
            )

        if 0.0 <= min_value and max_value <= 1000.0 and (
            self.x_max > image_width or self.y_max > image_height
        ):
            return BoundingBox(
                self.x_min / 1000.0 * image_width,
                self.y_min / 1000.0 * image_height,
                self.x_max / 1000.0 * image_width,
                self.y_max / 1000.0 * image_height,
            )

        return self

    def clamp(self, image_width: int, image_height: int) -> "BoundingBox":
        x0 = min(max(self.x_min, 0.0), float(image_width))
        y0 = min(max(self.y_min, 0.0), float(image_height))
        x1 = min(max(self.x_max, 0.0), float(image_width))
        y1 = min(max(self.y_max, 0.0), float(image_height))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        return BoundingBox(x0, y0, x1, y1)

    def shrink(self, ratio: float) -> "BoundingBox":
        ratio = min(max(ratio, 0.05), 1.0)
        center_x, center_y = self.center
        half_w = self.width * ratio / 2.0
        half_h = self.height * ratio / 2.0
        return BoundingBox(
            center_x - half_w,
            center_y - half_h,
            center_x + half_w,
            center_y + half_h,
        )

    def to_int_roi(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        clamped = self.clamp(image_width, image_height)
        x0 = int(max(0, min(image_width, clamped.x_min)))
        y0 = int(max(0, min(image_height, clamped.y_min)))
        x1 = int(max(0, min(image_width, clamped.x_max + 0.999999)))
        y1 = int(max(0, min(image_height, clamped.y_max + 0.999999)))
        return x0, y0, x1, y1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PixelPoint:
    x: float
    y: float

    def is_valid(self) -> bool:
        return self.x > 0.0 or self.y > 0.0

    def denormalized_if_needed(self, image_width: int, image_height: int) -> "PixelPoint":
        if 0.0 <= self.x <= 1.5 and 0.0 <= self.y <= 1.5:
            return PixelPoint(self.x * image_width, self.y * image_height)
        if 0.0 <= self.x <= 1000.0 and 0.0 <= self.y <= 1000.0 and (
            self.x > image_width or self.y > image_height
        ):
            return PixelPoint(self.x / 1000.0 * image_width, self.y / 1000.0 * image_height)
        return self

    def clamp(self, image_width: int, image_height: int) -> "PixelPoint":
        return PixelPoint(
            min(max(self.x, 0.0), float(max(0, image_width - 1))),
            min(max(self.y, 0.0), float(max(0, image_height - 1))),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionResult:
    found: bool
    label: str
    confidence: float
    bbox: BoundingBox
    notes: str = ""
    source: str = "unknown"
    head_px: PixelPoint | None = None
    tail_px: PixelPoint | None = None
    mask: Any | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "notes": self.notes,
            "source": self.source,
            "head_px": self.head_px.to_dict() if self.head_px else None,
            "tail_px": self.tail_px.to_dict() if self.tail_px else None,
            "mask_area_px": int(self.mask.sum()) if self.mask is not None else None,
        }


@dataclass(frozen=True)
class OrientationEstimate:
    head_px: PixelPoint
    tail_px: PixelPoint
    angle_deg_image: float
    vector_px_x: float
    vector_px_y: float
    head_depth_m: float | None
    tail_depth_m: float | None
    head_m: tuple[float, float, float] | None
    tail_m: tuple[float, float, float] | None
    vector_3d: tuple[float, float, float] | None
    head_sample_count: int
    tail_sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "head_px": self.head_px.to_dict(),
            "tail_px": self.tail_px.to_dict(),
            "angle_deg_image": self.angle_deg_image,
            "vector_px": {"x": self.vector_px_x, "y": self.vector_px_y},
            "head_depth_m": self.head_depth_m,
            "tail_depth_m": self.tail_depth_m,
            "head_m": _tuple3_to_dict(self.head_m),
            "tail_m": _tuple3_to_dict(self.tail_m),
            "vector_3d": _tuple3_to_dict(self.vector_3d),
            "head_sample_count": self.head_sample_count,
            "tail_sample_count": self.tail_sample_count,
            "definition": "tail_px -> head_px; for sample bottles, this points toward the black cap/head",
        }


@dataclass(frozen=True)
class PositionEstimate:
    x_m: float
    y_m: float
    z_m: float
    u_px: float
    v_px: float
    depth_m: float
    sample_count: int
    valid_fraction: float
    strategy: str
    bbox_used: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox_used"] = self.bbox_used.to_dict()
        return data


def _tuple3_to_dict(value: tuple[float, float, float] | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"x": value[0], "y": value[1], "z": value[2]}
