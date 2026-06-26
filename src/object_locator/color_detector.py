from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from .models import BoundingBox, DetectionResult


SUPPORTED_COLORS = {
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "black",
    "white",
}

COLOR_ALIASES = {
    "red": "red",
    "reddish": "red",
    "红": "red",
    "红色": "red",
    "orange": "orange",
    "橙": "orange",
    "橙色": "orange",
    "yellow": "yellow",
    "黄": "yellow",
    "黄色": "yellow",
    "green": "green",
    "绿色": "green",
    "绿": "green",
    "blue": "blue",
    "蓝色": "blue",
    "蓝": "blue",
    "purple": "purple",
    "violet": "purple",
    "紫": "purple",
    "紫色": "purple",
    "pink": "pink",
    "magenta": "pink",
    "粉": "pink",
    "粉色": "pink",
    "black": "black",
    "黑": "black",
    "黑色": "black",
    "white": "white",
    "白": "white",
    "白色": "white",
}


@dataclass(frozen=True)
class ColorDetectorConfig:
    color: str = "auto"
    min_area_px: int = 150
    max_area_ratio: float = 0.2
    min_saturation: int = 60
    min_value: int = 40
    morph_kernel: int = 5


def resolve_target_color(target: str, configured_color: str = "auto") -> str | None:
    configured = configured_color.strip().lower()
    if configured and configured != "auto":
        return COLOR_ALIASES.get(configured, configured if configured in SUPPORTED_COLORS else None)

    target_lower = target.lower()
    tokens = re.findall(r"[a-zA-Z]+|[\u4e00-\u9fff]+", target_lower)
    for token in tokens:
        if token in COLOR_ALIASES:
            return COLOR_ALIASES[token]
        for alias, color in COLOR_ALIASES.items():
            if alias and alias in token:
                return color
    return None


def detect_colored_object(
    image_bgr: np.ndarray,
    target: str,
    config: ColorDetectorConfig | None = None,
) -> DetectionResult:
    if config is None:
        config = ColorDetectorConfig()

    color = resolve_target_color(target, config.color)
    if color is None:
        return DetectionResult(
            found=False,
            label=target,
            confidence=0.0,
            bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
            notes="no supported color word found in target",
            source="color",
        )

    mask = build_color_mask(
        image_bgr,
        color=color,
        min_saturation=config.min_saturation,
        min_value=config.min_value,
    )
    mask = _cleanup_mask(mask, config.morph_kernel)
    component = _largest_component(mask, config.min_area_px, config.max_area_ratio)
    if component is None:
        return DetectionResult(
            found=False,
            label=f"{color} object",
            confidence=0.0,
            bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
            notes=f"no {color} component above {config.min_area_px} px",
            source=f"color:{color}",
        )

    x, y, w, h, area = component
    bbox = _padded_bbox(BoundingBox(x, y, x + w, y + h), image_bgr.shape[1], image_bgr.shape[0])
    confidence = _confidence(area, image_bgr.shape[0] * image_bgr.shape[1])
    return DetectionResult(
        found=True,
        label=f"{color} object",
        confidence=confidence,
        bbox=bbox,
        notes=f"largest {color} color component area={area} px",
        source=f"color:{color}",
    )


def color_ratio_in_box(
    image_bgr: np.ndarray,
    box: BoundingBox,
    *,
    color: str,
    min_saturation: int = 60,
    min_value: int = 40,
    morph_kernel: int = 5,
) -> float:
    mask = build_color_mask(
        image_bgr,
        color=color,
        min_saturation=min_saturation,
        min_value=min_value,
    )
    mask = _cleanup_mask(mask, morph_kernel)
    x0, y0, x1, y1 = box.to_int_roi(image_bgr.shape[1], image_bgr.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = mask[y0:y1, x0:x1]
    return float(np.count_nonzero(roi) / max(roi.size, 1))


def build_color_mask(
    image_bgr: np.ndarray,
    *,
    color: str,
    min_saturation: int = 60,
    min_value: int = 40,
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    color = color.lower()

    if color == "red":
        lower_1 = np.array([0, min_saturation, min_value], dtype=np.uint8)
        upper_1 = np.array([15, 255, 255], dtype=np.uint8)
        lower_2 = np.array([165, min_saturation, min_value], dtype=np.uint8)
        upper_2 = np.array([179, 255, 255], dtype=np.uint8)
        return cv2.bitwise_or(cv2.inRange(hsv, lower_1, upper_1), cv2.inRange(hsv, lower_2, upper_2))

    ranges = {
        "orange": ((5, min_saturation, min_value), (25, 255, 255)),
        "yellow": ((20, min_saturation, min_value), (38, 255, 255)),
        "green": ((35, min_saturation, min_value), (90, 255, 255)),
        "blue": ((85, min_saturation, min_value), (135, 255, 255)),
        "purple": ((125, min_saturation, min_value), (165, 255, 255)),
        "pink": ((145, min_saturation, min_value), (179, 255, 255)),
        "black": ((0, 0, 0), (179, 255, 55)),
        "white": ((0, 0, 180), (179, 60, 255)),
    }
    if color not in ranges:
        raise ValueError(f"unsupported color {color!r}; supported colors: {sorted(SUPPORTED_COLORS)}")

    lower, upper = ranges[color]
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def _cleanup_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(1, int(kernel_size))
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def _largest_component(
    mask: np.ndarray,
    min_area_px: int,
    max_area_ratio: float,
) -> tuple[int, int, int, int, int] | None:
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None

    image_area = mask.shape[0] * mask.shape[1]
    max_area_px = max(1, int(image_area * max_area_ratio))
    best = None
    best_area = 0

    for label in range(1, num_labels):
        x, y, w, h, area = [int(value) for value in stats[label]]
        if area < min_area_px or area > max_area_px:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.15 or aspect > 6.0:
            continue
        if area > best_area:
            best = (x, y, w, h, area)
            best_area = area
    return best


def _padded_bbox(box: BoundingBox, image_width: int, image_height: int) -> BoundingBox:
    padding = max(3.0, min(box.width, box.height) * 0.15)
    return BoundingBox(
        box.x_min - padding,
        box.y_min - padding,
        box.x_max + padding,
        box.y_max + padding,
    ).clamp(image_width, image_height)


def _confidence(area: int, image_area: int) -> float:
    ratio = area / max(image_area, 1)
    return min(0.98, max(0.55, 0.65 + ratio * 40.0))
