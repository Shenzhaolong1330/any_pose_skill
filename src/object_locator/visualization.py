from __future__ import annotations

import cv2
import numpy as np

from .models import DetectionResult, OrientationEstimate, PixelPoint, PositionEstimate


BOX_COLOR = (60, 220, 60)
MISSING_COLOR = (80, 80, 255)
INNER_BOX_COLOR = (0, 220, 255)
POINT_COLOR = (255, 255, 255)


def draw_detection(
    image_bgr: np.ndarray,
    detection: DetectionResult,
    position: PositionEstimate | None = None,
    orientation: OrientationEstimate | None = None,
    title: str | None = None,
) -> np.ndarray:
    output = image_bgr.copy()
    _draw_mask_overlay(output, detection)
    box = detection.bbox
    x0, y0, x1, y1 = box.to_int_roi(output.shape[1], output.shape[0])
    color = BOX_COLOR if detection.found else MISSING_COLOR
    cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)

    lines = [f"{detection.label} {detection.confidence:.2f} [{detection.source}]"]
    if position is not None:
        lines.append(f"x={position.x_m:.3f} y={position.y_m:.3f} z={position.z_m:.3f} m")
        _draw_position_overlay(output, position)
    if orientation is not None:
        lines.append(f"angle={orientation.angle_deg_image:.1f} deg")
        _draw_orientation_overlay(output, orientation.tail_px, orientation.head_px)
    elif detection.head_px is not None and detection.tail_px is not None:
        _draw_orientation_overlay(output, detection.tail_px, detection.head_px)
    _draw_label(output, lines, x0, y0, color)
    if title:
        _draw_title(output, title)
    return output


def draw_depth_detection(
    depth_m: np.ndarray,
    detection: DetectionResult,
    position: PositionEstimate | None = None,
    orientation: OrientationEstimate | None = None,
    *,
    min_depth_m: float = 0.05,
    max_depth_m: float = 6.0,
    title: str | None = None,
) -> np.ndarray:
    output = colorize_depth(depth_m, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    _draw_mask_overlay(output, detection)
    box = detection.bbox
    x0, y0, x1, y1 = box.to_int_roi(output.shape[1], output.shape[0])
    color = BOX_COLOR if detection.found else MISSING_COLOR
    cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)

    lines = [f"{detection.label} {detection.confidence:.2f} [{detection.source}]"]
    if position is not None:
        lines.append(f"depth={position.depth_m:.3f} m samples={position.sample_count}")
        _draw_position_overlay(output, position)
    if orientation is not None:
        lines.append(f"angle={orientation.angle_deg_image:.1f} deg")
        _draw_orientation_overlay(output, orientation.tail_px, orientation.head_px)
    elif detection.head_px is not None and detection.tail_px is not None:
        _draw_orientation_overlay(output, detection.tail_px, detection.head_px)
    _draw_label(output, lines, x0, y0, color)
    _draw_depth_scale(output, min_depth_m, max_depth_m)
    if title:
        _draw_title(output, title)
    return output


def draw_debug_panel(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    detection: DetectionResult,
    position: PositionEstimate | None = None,
    orientation: OrientationEstimate | None = None,
    *,
    min_depth_m: float = 0.05,
    max_depth_m: float = 6.0,
) -> np.ndarray:
    rgb = draw_detection(color_bgr, detection, position, orientation, title="RGB")
    depth = draw_depth_detection(
        depth_m,
        detection,
        position,
        orientation,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        title="Depth",
    )
    if rgb.shape[:2] != depth.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.hconcat([rgb, depth])


def colorize_depth(
    depth_m: np.ndarray,
    *,
    min_depth_m: float = 0.05,
    max_depth_m: float = 6.0,
) -> np.ndarray:
    depth = depth_m.astype(np.float32, copy=False)
    valid = np.isfinite(depth) & (depth > 0.0)
    clipped = np.where(valid, np.clip(depth, min_depth_m, max_depth_m), min_depth_m)
    normalized = ((clipped - min_depth_m) / max(max_depth_m - min_depth_m, 1e-6) * 255.0).astype(
        np.uint8
    )
    normalized[~valid] = 0
    colorized = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colorized[~valid] = (0, 0, 0)
    return colorized


def _draw_position_overlay(image: np.ndarray, position: PositionEstimate) -> None:
    box = position.bbox_used
    x0, y0, x1, y1 = box.to_int_roi(image.shape[1], image.shape[0])
    cv2.rectangle(image, (x0, y0), (x1, y1), INNER_BOX_COLOR, 1)
    _draw_crosshair(image, int(round(position.u_px)), int(round(position.v_px)))


def _draw_mask_overlay(image: np.ndarray, detection: DetectionResult) -> None:
    if detection.mask is None:
        return
    mask = np.asarray(detection.mask).astype(bool)
    if mask.shape[:2] != image.shape[:2]:
        return
    overlay = image.copy()
    overlay[mask] = (0, 180, 255)
    cv2.addWeighted(overlay, 0.32, image, 0.68, 0, dst=image)


def _draw_orientation_overlay(image: np.ndarray, tail: PixelPoint, head: PixelPoint) -> None:
    tail_xy = (int(round(tail.x)), int(round(tail.y)))
    head_xy = (int(round(head.x)), int(round(head.y)))
    cv2.arrowedLine(image, tail_xy, head_xy, (255, 80, 255), 2, cv2.LINE_AA, tipLength=0.22)
    cv2.circle(image, tail_xy, 4, (255, 80, 255), -1, cv2.LINE_AA)
    cv2.circle(image, head_xy, 5, (0, 255, 255), -1, cv2.LINE_AA)


def _draw_label(image: np.ndarray, lines: list[str], x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1
    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
    width = max(size[0] for size in sizes) + 10
    height = len(lines) * 20 + 8
    y_top = max(0, y - height)
    x_left = min(max(0, x), max(0, image.shape[1] - width))
    cv2.rectangle(image, (x_left, y_top), (x_left + width, y_top + height), color, -1)
    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x_left + 5, y_top + 20 + idx * 20),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )


def _draw_crosshair(image: np.ndarray, x: int, y: int) -> None:
    if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
        return
    cv2.drawMarker(
        image,
        (x, y),
        POINT_COLOR,
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
        line_type=cv2.LINE_AA,
    )


def _draw_title(image: np.ndarray, title: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(title, font, scale, thickness)
    cv2.rectangle(image, (8, 8), (18 + text_w, 18 + text_h), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (13, 13 + text_h),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _draw_depth_scale(image: np.ndarray, min_depth_m: float, max_depth_m: float) -> None:
    text = f"near {min_depth_m:.2f}m  far {max_depth_m:.2f}m"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.46
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(8, image.shape[1] - text_w - 14)
    y = image.shape[0] - 12
    cv2.rectangle(image, (x - 5, y - text_h - 7), (x + text_w + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
