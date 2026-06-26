from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

import cv2

from .color_detector import (
    ColorDetectorConfig,
    color_ratio_in_box,
    detect_colored_object,
    resolve_target_color,
)
from .config import DEFAULT_CONFIG_PATH, DEFAULT_MODEL, load_config
from .geometry import (
    DepthEstimatorConfig,
    DepthEstimationError,
    OrientationEstimatorConfig,
    estimate_orientation,
    estimate_position_from_depth,
)
from .grounded_sam_detector import GroundedSamConfig, GroundedSamDetector
from .models import BoundingBox
from .openrouter_vlm import OpenRouterError, OpenRouterVLMClient
from .realsense_camera import RealSenseCamera, list_realsense_devices
from .transforms import position_base_from_calibration
from .visualization import draw_debug_panel, draw_depth_detection, draw_detection


def main() -> int:
    _load_dotenv()
    args = _parse_args()

    try:
        if args.list_devices:
            return _print_realsense_devices(args.json)
        app_config = load_config(args.config)
        runtime = _merge_args_with_config(args, app_config)
        result = locate_once(runtime)
        _maybe_archive_vlm_response(runtime, result)
        _maybe_save_result_json(runtime, result)
    except (DepthEstimationError, OpenRouterError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    if result["_runtime"]["json"]:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(_format_result(result))
    return 0


def locate_once(runtime: dict) -> dict:
    with RealSenseCamera(
        width=runtime["width"],
        height=runtime["height"],
        fps=runtime["fps"],
        serial_number=runtime["serial_number"],
        reset_on_start=runtime["reset_on_start"],
        reset_wait_s=runtime["reset_wait_s"],
    ) as camera:
        frame = camera.capture(
            warmup_frames=runtime["warmup_frames"],
            timeout_ms=runtime["frame_timeout_ms"],
            retries=runtime["capture_retries"],
        )

    detection = _detect_object(frame.color_bgr, runtime)
    if not detection.found:
        result = {
            "target": runtime["target"],
            "found": False,
            "run_id": runtime["run_id"],
            "detection": detection.to_dict(),
            "message": "object was not found by the configured detector",
            "debug_outputs": _maybe_save_debug(runtime, frame.color_bgr, frame.depth_m, detection, None),
            "_runtime": _runtime_metadata(runtime),
        }
        return result

    depth_config = DepthEstimatorConfig(
        min_depth_m=runtime["min_depth_m"],
        max_depth_m=runtime["max_depth_m"],
        inner_ratio=runtime["inner_ratio"],
        min_samples=runtime["min_samples"],
        fallback_min_samples=runtime["fallback_min_samples"],
        max_expand_ratio=runtime["max_expand_ratio"],
        expand_steps=runtime["expand_steps"],
        strategy=runtime["depth_strategy"],
    )
    position_box, position_anchor = _position_box_for_detection(detection, runtime)
    position = estimate_position_from_depth(
        frame.depth_m,
        frame.intrinsics,
        position_box,
        depth_config,
    )
    orientation = estimate_orientation(
        frame.depth_m,
        frame.intrinsics,
        detection,
        OrientationEstimatorConfig(
            min_depth_m=runtime["min_depth_m"],
            max_depth_m=runtime["max_depth_m"],
        ),
    )
    position_base = position_base_from_calibration(
        position,
        enabled=runtime["calibration_enabled"],
        calibration_file=runtime["calibration_file"],
        active_camera=runtime["calibration_active_camera"],
        config_path=runtime["config"],
        position_anchor=position_anchor,
    )
    debug_outputs = _maybe_save_debug(
        runtime, frame.color_bgr, frame.depth_m, detection, position, orientation
    )

    if runtime["save_depth"]:
        depth_path = Path(runtime["save_depth"])
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.save(depth_path, frame.depth_m)

    return {
        "target": runtime["target"],
        "found": True,
        "run_id": runtime["run_id"],
        "detection": detection.to_dict(),
        "position": position.to_dict(),
        "position_anchor": position_anchor,
        "position_base": position_base,
        "orientation": orientation.to_dict() if orientation else None,
        "debug_outputs": debug_outputs,
        "intrinsics": frame.intrinsics.to_dict(),
        "timestamp_ms": frame.timestamp_ms,
        "realsense": {
            "serial_number": runtime["serial_number"],
            "width": runtime["width"],
            "height": runtime["height"],
            "fps": runtime["fps"],
            "reset_on_start": runtime["reset_on_start"],
            "reset_wait_s": runtime["reset_wait_s"],
        },
        "coordinate_frame": {
            "name": "RealSense color optical frame",
            "units": "meters",
            "x": "right",
            "y": "down",
            "z": "forward from camera",
        },
        "_runtime": _runtime_metadata(runtime),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locate a named object relative to a RealSense D435i camera."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML config path. Defaults to config.yaml.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List connected RealSense devices and exit.",
    )
    parser.add_argument("--target", help="Override object name from config.")
    parser.add_argument(
        "--detector",
        choices=["auto", "color", "vlm", "grounded_sam"],
        help="Detection backend. auto uses color detection for colored targets, otherwise VLM.",
    )
    parser.add_argument("--color", help="Override color detector target color, e.g. red.")
    parser.add_argument(
        "--model",
        default=None,
        help="OpenRouter vision model slug.",
    )
    parser.add_argument("--serial-number", help="Override RealSense serial number from config.")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--warmup-frames", type=int)
    parser.add_argument("--frame-timeout-ms", type=int)
    parser.add_argument("--capture-retries", type=int)
    parser.add_argument(
        "--reset-realsense",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Hardware-reset the RealSense before opening streams.",
    )
    parser.add_argument("--reset-wait-s", type=float)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--jpeg-quality", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--no-json-schema", action="store_true")
    parser.add_argument(
        "--no-json-retry",
        action="store_true",
        help="Disable retry without response_format json_schema when VLM JSON parsing fails.",
    )
    parser.add_argument(
        "--require-parameters",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Ask OpenRouter to route only to providers supporting requested parameters.",
    )
    parser.add_argument(
        "--depth-strategy",
        choices=["median", "foreground", "center"],
        default=None,
        help="How to choose depth inside the detected bbox.",
    )
    parser.add_argument("--inner-ratio", type=float)
    parser.add_argument("--min-depth", type=float)
    parser.add_argument("--max-depth", type=float)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--fallback-min-samples", type=int)
    parser.add_argument("--max-expand-ratio", type=float)
    parser.add_argument("--expand-steps", type=int)
    parser.add_argument("--output", help="Optional combined RGB/depth debug panel path.")
    parser.add_argument("--output-rgb", help="Optional RGB bbox debug image path.")
    parser.add_argument("--output-depth", help="Optional depth bbox debug image path.")
    parser.add_argument(
        "--result-json",
        help="Optional final result JSON path. Supports {run_id}; use empty string to disable.",
    )
    parser.add_argument(
        "--history-dir",
        help=(
            "Directory for timestamped per-run archives. "
            "Use an empty string in config to disable."
        ),
    )
    parser.add_argument("--save-vlm-response", help="Optional raw VLM response JSON path.")
    parser.add_argument("--save-depth", help="Optional .npy path for aligned depth in meters.")
    parser.add_argument(
        "--json",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Print machine-readable JSON.",
    )
    return parser.parse_args()


def _print_realsense_devices(as_json: bool | None) -> int:
    devices = list_realsense_devices()
    if as_json:
        print(json.dumps([device.to_dict() for device in devices], ensure_ascii=False, indent=2))
        return 0

    if not devices:
        print("No RealSense devices found.")
        return 0

    for index, device in enumerate(devices, start=1):
        firmware = f", firmware={device.firmware_version}" if device.firmware_version else ""
        print(f"{index}. {device.name}: serial_number={device.serial_number}{firmware}")
    return 0


def _merge_args_with_config(args: argparse.Namespace, config) -> dict:
    model = args.model or config.openrouter.model or os.environ.get("OPENROUTER_MODEL")
    runtime = {
        "target": args.target or config.target.name,
        "target_description": config.target.description,
        "detector_mode": args.detector or config.detector.mode,
        "detector_color": args.color or config.detector.color,
        "color_min_area_px": config.detector.min_area_px,
        "color_max_area_ratio": config.detector.max_area_ratio,
        "color_min_saturation": config.detector.min_saturation,
        "color_min_value": config.detector.min_value,
        "color_morph_kernel": config.detector.morph_kernel,
        "validate_vlm_color": config.detector.validate_vlm_color,
        "min_vlm_color_ratio": config.detector.min_vlm_color_ratio,
        "fallback_to_color_on_vlm_mismatch": config.detector.fallback_to_color_on_vlm_mismatch,
        "grounding_model": config.grounded_sam.grounding_model,
        "sam_model": config.grounded_sam.sam_model,
        "grounded_sam_text_prompt": config.grounded_sam.text_prompt,
        "grounded_sam_selection": config.grounded_sam.selection,
        "grounded_sam_box_threshold": config.grounded_sam.box_threshold,
        "grounded_sam_text_threshold": config.grounded_sam.text_threshold,
        "grounded_sam_device": config.grounded_sam.device,
        "grounded_sam_use_sam": config.grounded_sam.use_sam,
        "grounded_sam_refine_bbox_with_mask": config.grounded_sam.refine_bbox_with_mask,
        "grounded_sam_min_box_area_px": config.grounded_sam.min_box_area_px,
        "grounded_sam_max_box_area_ratio": config.grounded_sam.max_box_area_ratio,
        "grounded_sam_min_mask_area_px": config.grounded_sam.min_mask_area_px,
        "grounded_sam_cap_dark_threshold": config.grounded_sam.cap_dark_threshold,
        "grounded_sam_cap_min_area_px": config.grounded_sam.cap_min_area_px,
        "model": model or DEFAULT_MODEL,
        "serial_number": args.serial_number or config.realsense.serial_number,
        "width": _coalesce(args.width, config.realsense.width),
        "height": _coalesce(args.height, config.realsense.height),
        "fps": _coalesce(args.fps, config.realsense.fps),
        "warmup_frames": _coalesce(args.warmup_frames, config.realsense.warmup_frames),
        "frame_timeout_ms": _coalesce(args.frame_timeout_ms, config.realsense.frame_timeout_ms),
        "capture_retries": _coalesce(args.capture_retries, config.realsense.capture_retries),
        "reset_on_start": (
            config.realsense.reset_on_start
            if args.reset_realsense is None
            else args.reset_realsense
        ),
        "reset_wait_s": _coalesce(args.reset_wait_s, config.realsense.reset_wait_s),
        "timeout_s": _coalesce(args.timeout, config.openrouter.timeout_s),
        "temperature": _coalesce(args.temperature, config.openrouter.temperature),
        "jpeg_quality": _coalesce(args.jpeg_quality, config.openrouter.jpeg_quality),
        "max_tokens": _coalesce(args.max_tokens, config.openrouter.max_tokens),
        "use_json_schema": config.openrouter.use_json_schema and not args.no_json_schema,
        "require_parameters": (
            config.openrouter.require_parameters
            if args.require_parameters is None
            else args.require_parameters
        ),
        "retry_without_json_schema": (
            config.openrouter.retry_without_json_schema and not args.no_json_retry
        ),
        "depth_strategy": _coalesce(args.depth_strategy, config.depth.strategy),
        "position_anchor": config.depth.position_anchor,
        "anchor_radius_px": config.depth.anchor_radius_px,
        "inner_ratio": _coalesce(args.inner_ratio, config.depth.inner_ratio),
        "min_depth_m": _coalesce(args.min_depth, config.depth.min_depth_m),
        "max_depth_m": _coalesce(args.max_depth, config.depth.max_depth_m),
        "min_samples": _coalesce(args.min_samples, config.depth.min_samples),
        "fallback_min_samples": _coalesce(
            args.fallback_min_samples, config.depth.fallback_min_samples
        ),
        "max_expand_ratio": _coalesce(args.max_expand_ratio, config.depth.max_expand_ratio),
        "expand_steps": _coalesce(args.expand_steps, config.depth.expand_steps),
        "calibration_enabled": config.calibration.enabled,
        "calibration_file": config.calibration.file,
        "calibration_active_camera": config.calibration.active_camera,
        "result_json": (
            args.result_json if args.result_json is not None else config.output.result_json
        ),
        "history_dir": (
            args.history_dir if args.history_dir is not None else config.output.history_dir
        ),
        "run_id": _new_run_id(),
        "debug_image": args.output if args.output is not None else config.output.debug_image,
        "debug_rgb_image": (
            args.output_rgb if args.output_rgb is not None else config.output.debug_rgb_image
        ),
        "debug_depth_image": (
            args.output_depth
            if args.output_depth is not None
            else config.output.debug_depth_image
        ),
        "vlm_response": (
            args.save_vlm_response
            if args.save_vlm_response is not None
            else config.output.vlm_response
        ),
        "save_depth": args.save_depth if args.save_depth is not None else config.output.save_depth,
        "json": config.output.json if args.json is None else args.json,
        "config": args.config,
    }
    if runtime["depth_strategy"] not in {"median", "foreground", "center"}:
        raise ValueError("depth.strategy must be one of: median, foreground, center")
    if runtime["position_anchor"] not in {"auto", "bbox", "head", "tail"}:
        raise ValueError("depth.position_anchor must be one of: auto, bbox, head, tail")
    if runtime["detector_mode"] not in {"auto", "color", "vlm", "grounded_sam"}:
        raise ValueError("detector.mode must be one of: auto, color, vlm, grounded_sam")
    if not runtime["target"]:
        raise ValueError("target.name must be set in config or passed with --target")
    return runtime


def _coalesce(value, fallback):
    return fallback if value is None else value


def _runtime_metadata(runtime: dict) -> dict:
    return {
        "config": runtime["config"],
        "detector_mode": runtime["detector_mode"],
        "detector_color": runtime["detector_color"],
        "model": runtime["model"],
        "json": runtime["json"],
        "debug_image": runtime["debug_image"],
        "debug_rgb_image": runtime["debug_rgb_image"],
        "debug_depth_image": runtime["debug_depth_image"],
        "vlm_response": runtime["vlm_response"],
        "save_depth": runtime["save_depth"],
        "result_json": runtime["result_json"],
        "history_dir": runtime["history_dir"],
        "run_id": runtime["run_id"],
        "frame_timeout_ms": runtime["frame_timeout_ms"],
        "capture_retries": runtime["capture_retries"],
        "reset_on_start": runtime["reset_on_start"],
        "reset_wait_s": runtime["reset_wait_s"],
        "calibration_enabled": runtime["calibration_enabled"],
        "calibration_file": runtime["calibration_file"],
        "calibration_active_camera": runtime["calibration_active_camera"],
    }


def _format_result(result: dict) -> str:
    if not result.get("found"):
        detection = result["detection"]
        return (
            f"Target: {result['target']}\n"
            f"Found: no\n"
            f"Detector: {detection['source']}\n"
            f"Label: {detection['label']} confidence={detection['confidence']:.2f}"
        )

    detection = result["detection"]
    position = result["position"]
    orientation = result.get("orientation")
    bbox = detection["bbox"]
    output = (
        f"Target: {result['target']}\n"
        f"Found: yes ({detection['label']}, detector={detection['source']}, "
        f"confidence={detection['confidence']:.2f})\n"
        f"BBox px: x_min={bbox['x_min']:.1f}, y_min={bbox['y_min']:.1f}, "
        f"x_max={bbox['x_max']:.1f}, y_max={bbox['y_max']:.1f}\n"
        f"Position in camera frame: x={position['x_m']:.3f} m, "
        f"y={position['y_m']:.3f} m, z={position['z_m']:.3f} m "
        f"(anchor={result.get('position_anchor', 'bbox')})\n"
        f"Depth samples: {position['sample_count']} "
        f"(valid fraction {position['valid_fraction']:.2%}, strategy={position['strategy']})"
    )
    position_base = result.get("position_base")
    if position_base and position_base.get("available"):
        output += (
            f"\nPosition in base frame: x={position_base['x_m']:.3f} m, "
            f"y={position_base['y_m']:.3f} m, z={position_base['z_m']:.3f} m "
            f"(camera={position_base['active_camera']}, "
            f"chain={'+'.join(position_base['transform_chain'])})"
        )
    elif position_base and position_base.get("enabled"):
        output += f"\nPosition in base frame: unavailable ({position_base.get('reason')})"
    if orientation:
        output += (
            f"\nOrientation tail->head: angle={orientation['angle_deg_image']:.1f} deg "
            f"(image coords, 0=right, 90=down)"
        )
        if orientation.get("vector_3d"):
            vector = orientation["vector_3d"]
            output += (
                f"\nOrientation 3D unit vector: x={vector['x']:.3f}, "
                f"y={vector['y']:.3f}, z={vector['z']:.3f}"
            )
        else:
            output += "\nOrientation 3D unit vector: unavailable (not enough valid depth at both endpoints)"
    return output


def _position_box_for_detection(detection, runtime: dict) -> tuple[BoundingBox, str]:
    anchor = runtime["position_anchor"]
    if anchor == "auto":
        anchor = "head" if detection.head_px is not None else "bbox"
    if anchor == "head" and detection.head_px is not None:
        return _point_anchor_box(detection.head_px, runtime["anchor_radius_px"]), "head"
    if anchor == "tail" and detection.tail_px is not None:
        return _point_anchor_box(detection.tail_px, runtime["anchor_radius_px"]), "tail"
    return detection.bbox, "bbox"


def _point_anchor_box(point, anchor_radius_px: int | float) -> BoundingBox:
    radius = max(2.0, float(anchor_radius_px))
    return BoundingBox(
        point.x - radius,
        point.y - radius,
        point.x + radius,
        point.y + radius,
    )


def _detect_object(color_bgr, runtime: dict):
    mode = runtime["detector_mode"]
    should_try_color = mode == "color" or (
        mode == "auto" and resolve_target_color(runtime["target"], runtime["detector_color"])
    )

    if should_try_color:
        detection = detect_colored_object(
            color_bgr,
            runtime["target"],
            ColorDetectorConfig(
                color=runtime["detector_color"],
                min_area_px=runtime["color_min_area_px"],
                max_area_ratio=runtime["color_max_area_ratio"],
                min_saturation=runtime["color_min_saturation"],
                min_value=runtime["color_min_value"],
                morph_kernel=runtime["color_morph_kernel"],
            ),
        )
        if detection.found or mode == "color":
            _write_detector_trace(runtime, detection, vlm_called=False)
            return detection

    if mode == "grounded_sam":
        detector = GroundedSamDetector(
            GroundedSamConfig(
                grounding_model=runtime["grounding_model"],
                sam_model=runtime["sam_model"],
                text_prompt=runtime["grounded_sam_text_prompt"],
                selection=runtime["grounded_sam_selection"],
                box_threshold=runtime["grounded_sam_box_threshold"],
                text_threshold=runtime["grounded_sam_text_threshold"],
                device=runtime["grounded_sam_device"],
                use_sam=runtime["grounded_sam_use_sam"],
                refine_bbox_with_mask=runtime["grounded_sam_refine_bbox_with_mask"],
                min_box_area_px=runtime["grounded_sam_min_box_area_px"],
                max_box_area_ratio=runtime["grounded_sam_max_box_area_ratio"],
                min_mask_area_px=runtime["grounded_sam_min_mask_area_px"],
                cap_dark_threshold=runtime["grounded_sam_cap_dark_threshold"],
                cap_min_area_px=runtime["grounded_sam_cap_min_area_px"],
            )
        )
        return detector.detect(color_bgr, runtime["target"])

    client = OpenRouterVLMClient.from_env(
        model=runtime["model"],
        timeout_s=runtime["timeout_s"],
    )
    client.require_parameters = runtime["require_parameters"]
    vlm_detection = client.detect_object(
        color_bgr,
        runtime["target"],
        target_description=runtime["target_description"],
        jpeg_quality=runtime["jpeg_quality"],
        use_json_schema=runtime["use_json_schema"],
        temperature=runtime["temperature"],
        max_tokens=runtime["max_tokens"],
        retry_without_json_schema=runtime["retry_without_json_schema"],
        raw_response_path=runtime["vlm_response"],
    )
    return _validate_or_replace_vlm_detection(color_bgr, runtime, vlm_detection)


def _validate_or_replace_vlm_detection(color_bgr, runtime: dict, vlm_detection):
    if not runtime["validate_vlm_color"] or not vlm_detection.found:
        return vlm_detection

    target_color = resolve_target_color(runtime["target"], runtime["detector_color"])
    if target_color is None:
        return vlm_detection

    ratio = color_ratio_in_box(
        color_bgr,
        vlm_detection.bbox,
        color=target_color,
        min_saturation=runtime["color_min_saturation"],
        min_value=runtime["color_min_value"],
        morph_kernel=runtime["color_morph_kernel"],
    )
    if ratio >= runtime["min_vlm_color_ratio"]:
        _append_detector_trace(
            runtime,
            {
                "vlm_color_guard": {
                    "passed": True,
                    "color": target_color,
                    "color_ratio": ratio,
                    "min_ratio": runtime["min_vlm_color_ratio"],
                }
            },
        )
        return vlm_detection

    replacement = detect_colored_object(
        color_bgr,
        runtime["target"],
        ColorDetectorConfig(
            color=target_color,
            min_area_px=runtime["color_min_area_px"],
            max_area_ratio=runtime["color_max_area_ratio"],
            min_saturation=runtime["color_min_saturation"],
            min_value=runtime["color_min_value"],
            morph_kernel=runtime["color_morph_kernel"],
        ),
    )
    trace = {
        "vlm_color_guard": {
            "passed": False,
            "color": target_color,
            "color_ratio": ratio,
            "min_ratio": runtime["min_vlm_color_ratio"],
            "vlm_detection": vlm_detection.to_dict(),
            "replacement_detection": replacement.to_dict(),
        }
    }
    _append_detector_trace(runtime, trace)
    if replacement.found and runtime["fallback_to_color_on_vlm_mismatch"]:
        return replacement
    return vlm_detection


def _write_detector_trace(runtime: dict, detection, *, vlm_called: bool) -> None:
    path = runtime.get("vlm_response")
    if not path or vlm_called:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "vlm_called": False,
                "reason": "local detector returned a result before VLM fallback",
                "detection": detection.to_dict(),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def _append_detector_trace(runtime: dict, trace: dict) -> None:
    path = runtime.get("vlm_response")
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            data = {}
    data.update(trace)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _maybe_save_debug(
    runtime: dict,
    color_bgr,
    depth_m,
    detection,
    position,
    orientation=None,
) -> dict[str, str]:
    outputs = {}
    if runtime["debug_rgb_image"]:
        rgb_path = Path(runtime["debug_rgb_image"])
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_debug = draw_detection(color_bgr, detection, position, orientation, title="RGB")
        cv2.imwrite(str(rgb_path), rgb_debug)
        outputs["rgb"] = str(rgb_path)
        history_rgb_path = _history_path(runtime, "rgb.jpg")
        if history_rgb_path is not None:
            cv2.imwrite(str(history_rgb_path), rgb_debug)
            outputs["rgb_history"] = str(history_rgb_path)

    if runtime["debug_depth_image"]:
        depth_path = Path(runtime["debug_depth_image"])
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        depth_debug = draw_depth_detection(
            depth_m,
            detection,
            position,
            orientation,
            min_depth_m=runtime["min_depth_m"],
            max_depth_m=runtime["max_depth_m"],
            title="Depth",
        )
        cv2.imwrite(str(depth_path), depth_debug)
        outputs["depth"] = str(depth_path)
        history_depth_path = _history_path(runtime, "depth.jpg")
        if history_depth_path is not None:
            cv2.imwrite(str(history_depth_path), depth_debug)
            outputs["depth_history"] = str(history_depth_path)

    if runtime["debug_image"]:
        panel_path = Path(runtime["debug_image"])
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel = draw_debug_panel(
            color_bgr,
            depth_m,
            detection,
            position,
            orientation,
            min_depth_m=runtime["min_depth_m"],
            max_depth_m=runtime["max_depth_m"],
        )
        cv2.imwrite(str(panel_path), panel)
        outputs["panel"] = str(panel_path)
        history_panel_path = _history_path(runtime, "panel.jpg")
        if history_panel_path is not None:
            cv2.imwrite(str(history_panel_path), panel)
            outputs["panel_history"] = str(history_panel_path)

    history_dir = _history_dir(runtime)
    if history_dir is not None:
        outputs["history_dir"] = str(history_dir)

    return outputs


def _maybe_save_result_json(runtime: dict, result: dict) -> None:
    outputs = result.setdefault("debug_outputs", {})
    result_path = _output_path(runtime.get("result_json"), runtime)
    history_path = _history_path(runtime, "result.json")

    if result_path is not None:
        outputs["result_json"] = str(result_path)
    if history_path is not None:
        outputs["result_json_history"] = str(history_path)

    written: set[Path] = set()
    for path in (result_path, history_path):
        if path is None or path in written:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        written.add(path)


def _maybe_archive_vlm_response(runtime: dict, result: dict) -> None:
    source_path = runtime.get("vlm_response")
    if not source_path:
        return
    source = Path(source_path)
    if not source.exists():
        return
    archive_path = _history_path(runtime, "detector_trace.json")
    if archive_path is None:
        return
    shutil.copy2(source, archive_path)
    result.setdefault("debug_outputs", {})["detector_trace_history"] = str(archive_path)


def _history_path(runtime: dict, filename: str) -> Path | None:
    history_dir = _history_dir(runtime)
    if history_dir is None:
        return None
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir / filename


def _output_path(path_template: str | None, runtime: dict) -> Path | None:
    if not path_template:
        return None
    try:
        return Path(str(path_template).format(run_id=runtime["run_id"]))
    except KeyError as exc:
        raise ValueError("output path templates support only {run_id}") from exc


def _history_dir(runtime: dict) -> Path | None:
    history_root = runtime.get("history_dir")
    if not history_root:
        return None
    return Path(history_root) / str(runtime["run_id"])


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


if __name__ == "__main__":
    raise SystemExit(main())
