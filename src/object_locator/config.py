from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"


@dataclass(frozen=True)
class TargetConfig:
    name: str = "red cup"
    description: str | None = None


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str = DEFAULT_MODEL
    timeout_s: float = 60.0
    temperature: float = 0.0
    jpeg_quality: int = 90
    max_tokens: int = 1024
    use_json_schema: bool = True
    require_parameters: bool = False
    retry_without_json_schema: bool = True


@dataclass(frozen=True)
class DetectorConfig:
    mode: str = "auto"
    color: str = "auto"
    min_area_px: int = 150
    max_area_ratio: float = 0.2
    min_saturation: int = 60
    min_value: int = 40
    morph_kernel: int = 5
    validate_vlm_color: bool = True
    min_vlm_color_ratio: float = 0.02
    fallback_to_color_on_vlm_mismatch: bool = True


@dataclass(frozen=True)
class GroundedSamConfig:
    grounding_model: str = "IDEA-Research/grounding-dino-tiny"
    sam_model: str = "facebook/sam-vit-base"
    text_prompt: str = "sample bottle"
    selection: str = "leftmost"
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    device: str = "auto"
    use_sam: bool = True
    refine_bbox_with_mask: bool = True
    min_box_area_px: int = 100
    max_box_area_ratio: float = 0.5
    min_mask_area_px: int = 100
    cap_dark_threshold: int = 80
    cap_min_area_px: int = 20


@dataclass(frozen=True)
class RealSenseConfig:
    serial_number: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    warmup_frames: int = 30


@dataclass(frozen=True)
class DepthConfig:
    strategy: str = "median"
    position_anchor: str = "auto"
    anchor_radius_px: int = 12
    inner_ratio: float = 0.7
    min_depth_m: float = 0.05
    max_depth_m: float = 6.0
    min_samples: int = 50
    fallback_min_samples: int = 5
    max_expand_ratio: float = 2.5
    expand_steps: int = 3


@dataclass(frozen=True)
class CalibrationConfig:
    enabled: bool = False
    file: str | None = "calibration/extrinsics.yaml"
    active_camera: str = "head"


@dataclass(frozen=True)
class OutputConfig:
    json: bool = False
    debug_image: str | None = "runs/latest_panel.jpg"
    debug_rgb_image: str | None = "runs/latest_rgb.jpg"
    debug_depth_image: str | None = "runs/latest_depth.jpg"
    vlm_response: str | None = "runs/latest_vlm_response.json"
    save_depth: str | None = None


@dataclass(frozen=True)
class AppConfig:
    target: TargetConfig = field(default_factory=TargetConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    grounded_sam: GroundedSamConfig = field(default_factory=GroundedSamConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    realsense: RealSenseConfig = field(default_factory=RealSenseConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"config file must contain a YAML object: {config_path}")
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> AppConfig:
    target = _section(raw, "target")
    detector = _section(raw, "detector")
    grounded_sam = _section(raw, "grounded_sam")
    openrouter = _section(raw, "openrouter")
    realsense = _section(raw, "realsense")
    depth = _section(raw, "depth")
    calibration = _section(raw, "calibration")
    output = _section(raw, "output")

    return AppConfig(
        target=TargetConfig(
            name=str(target.get("name", TargetConfig.name)),
            description=_optional_str(target.get("description")),
        ),
        detector=DetectorConfig(
            mode=str(detector.get("mode", "auto")),
            color=str(detector.get("color", "auto")),
            min_area_px=int(detector.get("min_area_px", 150)),
            max_area_ratio=float(detector.get("max_area_ratio", 0.2)),
            min_saturation=int(detector.get("min_saturation", 60)),
            min_value=int(detector.get("min_value", 40)),
            morph_kernel=int(detector.get("morph_kernel", 5)),
            validate_vlm_color=bool(detector.get("validate_vlm_color", True)),
            min_vlm_color_ratio=float(detector.get("min_vlm_color_ratio", 0.02)),
            fallback_to_color_on_vlm_mismatch=bool(
                detector.get("fallback_to_color_on_vlm_mismatch", True)
            ),
        ),
        grounded_sam=GroundedSamConfig(
            grounding_model=str(
                grounded_sam.get("grounding_model", "IDEA-Research/grounding-dino-tiny")
            ),
            sam_model=str(grounded_sam.get("sam_model", "facebook/sam-vit-base")),
            text_prompt=str(grounded_sam.get("text_prompt", "sample bottle")),
            selection=str(grounded_sam.get("selection", "leftmost")),
            box_threshold=float(grounded_sam.get("box_threshold", 0.25)),
            text_threshold=float(grounded_sam.get("text_threshold", 0.25)),
            device=str(grounded_sam.get("device", "auto")),
            use_sam=bool(grounded_sam.get("use_sam", True)),
            refine_bbox_with_mask=bool(grounded_sam.get("refine_bbox_with_mask", True)),
            min_box_area_px=int(grounded_sam.get("min_box_area_px", 100)),
            max_box_area_ratio=float(grounded_sam.get("max_box_area_ratio", 0.5)),
            min_mask_area_px=int(grounded_sam.get("min_mask_area_px", 100)),
            cap_dark_threshold=int(grounded_sam.get("cap_dark_threshold", 80)),
            cap_min_area_px=int(grounded_sam.get("cap_min_area_px", 20)),
        ),
        openrouter=OpenRouterConfig(
            model=str(openrouter.get("model", DEFAULT_MODEL)),
            timeout_s=float(openrouter.get("timeout_s", 60.0)),
            temperature=float(openrouter.get("temperature", 0.0)),
            jpeg_quality=int(openrouter.get("jpeg_quality", 90)),
            max_tokens=int(openrouter.get("max_tokens", 1024)),
            use_json_schema=bool(openrouter.get("use_json_schema", True)),
            require_parameters=bool(openrouter.get("require_parameters", False)),
            retry_without_json_schema=bool(openrouter.get("retry_without_json_schema", True)),
        ),
        realsense=RealSenseConfig(
            serial_number=_optional_str(realsense.get("serial_number")),
            width=int(realsense.get("width", 640)),
            height=int(realsense.get("height", 480)),
            fps=int(realsense.get("fps", 30)),
            warmup_frames=int(realsense.get("warmup_frames", 30)),
        ),
        depth=DepthConfig(
            strategy=str(depth.get("strategy", "median")),
            position_anchor=str(depth.get("position_anchor", "auto")),
            anchor_radius_px=int(depth.get("anchor_radius_px", 12)),
            inner_ratio=float(depth.get("inner_ratio", 0.7)),
            min_depth_m=float(depth.get("min_depth_m", 0.05)),
            max_depth_m=float(depth.get("max_depth_m", 6.0)),
            min_samples=int(depth.get("min_samples", 50)),
            fallback_min_samples=int(depth.get("fallback_min_samples", 5)),
            max_expand_ratio=float(depth.get("max_expand_ratio", 2.5)),
            expand_steps=int(depth.get("expand_steps", 3)),
        ),
        calibration=CalibrationConfig(
            enabled=bool(calibration.get("enabled", False)),
            file=_optional_str(calibration.get("file", "calibration/extrinsics.yaml")),
            active_camera=str(calibration.get("active_camera", "head")),
        ),
        output=OutputConfig(
            json=bool(output.get("json", False)),
            debug_image=_optional_str(output.get("debug_image", "runs/latest_panel.jpg")),
            debug_rgb_image=_optional_str(output.get("debug_rgb_image", "runs/latest_rgb.jpg")),
            debug_depth_image=_optional_str(
                output.get("debug_depth_image", "runs/latest_depth.jpg")
            ),
            vlm_response=_optional_str(output.get("vlm_response", "runs/latest_vlm_response.json")),
            save_depth=_optional_str(output.get("save_depth")),
        ),
    )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a YAML object")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
