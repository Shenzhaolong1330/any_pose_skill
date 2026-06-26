"""RealSense + VLM object localization."""

from .models import (
    BoundingBox,
    CameraIntrinsics,
    DetectionResult,
    OrientationEstimate,
    PixelPoint,
    PositionEstimate,
)

__all__ = [
    "BoundingBox",
    "CameraIntrinsics",
    "DetectionResult",
    "OrientationEstimate",
    "PixelPoint",
    "PositionEstimate",
]

__version__ = "0.1.0"
