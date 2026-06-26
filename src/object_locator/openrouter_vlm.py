from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import requests

from .models import BoundingBox, DetectionResult, PixelPoint
from .config import DEFAULT_MODEL


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter request or response cannot be used."""


class VLMResponseParseError(OpenRouterError):
    """Raised when the VLM response is not valid JSON."""


DETECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {
            "type": "boolean",
            "description": "Whether the requested object is visible in the image.",
        },
        "label": {
            "type": "string",
            "description": "The object name or short visual description.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence from 0 to 1.",
        },
        "box_2d": {
            "type": "object",
            "description": "Pixel coordinates in the original image.",
            "properties": {
                "x_min": {"type": "number"},
                "y_min": {"type": "number"},
                "x_max": {"type": "number"},
                "y_max": {"type": "number"},
            },
            "required": ["x_min", "y_min", "x_max", "y_max"],
            "additionalProperties": False,
        },
        "notes": {
            "type": "string",
            "description": "Short reason for the selected instance.",
        },
        "orientation": {
            "type": "object",
            "description": (
                "Optional object-axis keypoints in original image pixels. "
                "For bottles, head_px is the black cap/head center and tail_px is the opposite end."
            ),
            "properties": {
                "head_px": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
                "tail_px": {
                    "type": "object",
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
            },
            "required": ["head_px", "tail_px"],
            "additionalProperties": False,
        },
    },
    "required": ["found", "label", "confidence", "box_2d", "notes", "orientation"],
    "additionalProperties": False,
}


class OpenRouterVLMClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        endpoint: str = OPENROUTER_CHAT_COMPLETIONS_URL,
        http_referer: str | None = None,
        app_title: str | None = None,
        timeout_s: float = 60.0,
        require_parameters: bool = False,
    ) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is required")
        if "your" in api_key.lower() and "key" in api_key.lower():
            raise OpenRouterError(
                "OPENROUTER_API_KEY still looks like the placeholder from .env.example. "
                "Create a real OpenRouter API key and put it in .env."
            )
        if not api_key.startswith("sk-or-"):
            raise OpenRouterError(
                "OPENROUTER_API_KEY does not look like an OpenRouter key. "
                "It should usually start with 'sk-or-'."
            )
        if not model:
            raise OpenRouterError("OpenRouter model is required")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.http_referer = http_referer
        self.app_title = app_title
        self.timeout_s = timeout_s
        self.require_parameters = require_parameters

    @classmethod
    def from_env(cls, model: str | None = None, timeout_s: float = 60.0) -> "OpenRouterVLMClient":
        require_parameters = _env_bool("OPENROUTER_REQUIRE_PARAMETERS", default=False)
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            model=model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
            http_referer=os.environ.get("OPENROUTER_HTTP_REFERER") or None,
            app_title=os.environ.get("OPENROUTER_APP_TITLE") or "RealSense VLM Object Locator",
            timeout_s=timeout_s,
            require_parameters=require_parameters,
        )

    def detect_object(
        self,
        image_bgr: Any,
        target: str,
        *,
        target_description: str | None = None,
        jpeg_quality: int = 90,
        use_json_schema: bool = True,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retry_without_json_schema: bool = True,
        raw_response_path: str | None = None,
    ) -> DetectionResult:
        height, width = image_bgr.shape[:2]
        image_url = _encode_bgr_as_data_url(image_bgr, jpeg_quality)
        attempts = [(use_json_schema, "primary")]
        if use_json_schema and retry_without_json_schema:
            attempts.append((False, "retry_without_json_schema"))

        raw_attempts: list[dict[str, Any]] = []
        parse_errors: list[str] = []

        for attempt_index, (attempt_json_schema, attempt_name) in enumerate(attempts, start=1):
            payload = self._build_payload(
                target=target,
                target_description=target_description,
                image_url=image_url,
                width=width,
                height=height,
                use_json_schema=attempt_json_schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            data = self._post_payload(payload)
            content, finish_reason = _extract_message_content(data)
            raw_attempts.append(
                {
                    "attempt": attempt_index,
                    "name": attempt_name,
                    "model": self.model,
                    "use_json_schema": attempt_json_schema,
                    "finish_reason": finish_reason,
                    "content": content,
                    "usage": data.get("usage") if isinstance(data, dict) else None,
                }
            )

            try:
                parsed = _parse_json_content(content)
            except VLMResponseParseError as exc:
                parse_errors.append(
                    f"{attempt_name}: {exc}; finish_reason={finish_reason or 'unknown'}"
                )
                continue

            _write_raw_response(raw_response_path, raw_attempts)
            return _detection_from_json(parsed, width, height, fallback_label=target)

        _write_raw_response(raw_response_path, raw_attempts)
        details = " | ".join(parse_errors) if parse_errors else "no parse details"
        raise OpenRouterError(
            "VLM did not return complete JSON after retry. "
            f"Details: {details}. "
            "Try model='google/gemini-2.5-flash' or 'openai/gpt-4o-mini', "
            "or set openrouter.use_json_schema=false in config.yaml."
        )

    def _post_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(
                self.endpoint,
                headers=self._headers(),
                json=payload,
                timeout=self.timeout_s,
            )
        except requests.RequestException as exc:
            raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc

        if response.status_code >= 400:
            if response.status_code == 401:
                raise OpenRouterError(
                    "OpenRouter authentication failed with HTTP 401. "
                    "Check that OPENROUTER_API_KEY in .env is a real, current key "
                    "and that your shell is not overriding it with another value."
                )
            raise OpenRouterError(
                f"OpenRouter returned HTTP {response.status_code}: {response.text[:1000]}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise OpenRouterError(f"unexpected non-JSON OpenRouter response: {response.text[:1000]}") from exc

        if not isinstance(data, dict):
            raise OpenRouterError(f"unexpected OpenRouter response type: {type(data)!r}")
        return data

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        return headers

    def _build_payload(
        self,
        *,
        target: str,
        target_description: str | None,
        image_url: str,
        width: int,
        height: int,
        use_json_schema: bool,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        description = (
            f"\nExtra target description: {target_description.strip()}\n"
            if target_description
            else ""
        )
        prompt = (
            f"Find the object named: {target!r}.\n"
            f"{description}"
            f"The image size is width={width}, height={height}.\n"
            "Return the tightest bounding box around the single most relevant visible instance. "
            "If there are multiple matching instances and the target says leftmost/rightmost/topmost/bottommost, "
            "select the instance by the full object bounding box in image coordinates. "
            "The box must enclose only the visible pixels of the target object itself. "
            "Do not include nearby objects, shadows, background tabletop, robot parts, cables, mouse, labels, or empty space. "
            "For a small colored cube/block, first locate the colored pixels, then put the box tightly around that colored object. "
            "Before returning, verify that the center of the box lies on the target object and that each edge touches the target's visible extent. "
            "Coordinates must be pixel coordinates in the original image, with x_min/y_min as the "
            "top-left corner and x_max/y_max as the bottom-right corner. "
            "Also return orientation.head_px and orientation.tail_px. For a sample bottle, "
            "head_px is the center of the black cap/head; tail_px is the opposite end of the transparent bottle body. "
            "The direction tail_px -> head_px is the bottle pointing direction. "
            "If object-axis orientation is not meaningful or not visible, set both keypoints to {\"x\": 0, \"y\": 0}. "
            "If the object is not visible, set found=false, confidence=0, and all box coordinates to 0. "
            "Return exactly one complete JSON object. Do not include markdown, explanation, or trailing text."
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a precise visual object localization engine. "
                        "Return only JSON that matches the requested schema."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if use_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "object_bbox",
                    "strict": True,
                    "schema": DETECTION_SCHEMA,
                },
            }
        if self.require_parameters:
            payload["provider"] = {"require_parameters": True}
        return payload


def _encode_bgr_as_data_url(image_bgr: Any, jpeg_quality: int) -> str:
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise OpenRouterError("failed to encode camera image as JPEG")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_message_content(data: dict[str, Any]) -> tuple[Any, str | None]:
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError(f"unexpected OpenRouter response: {json.dumps(data)[:1000]}") from exc
    return content, str(finish_reason) if finish_reason is not None else None


def _parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                text_parts.append(part)
        content = "\n".join(text_parts)

    if not isinstance(content, str):
        raise OpenRouterError(f"message content is not text JSON: {type(content)!r}")

    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise VLMResponseParseError(f"could not find complete JSON in VLM response: {content[:1000]}")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise VLMResponseParseError(
                f"found JSON-looking text but could not parse it: {text[start : end + 1][:1000]}"
            ) from exc

    if not isinstance(parsed, dict):
        raise VLMResponseParseError("VLM response JSON must be an object")
    return parsed


def _write_raw_response(path: str | None, attempts: list[dict[str, Any]]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump({"attempts": attempts}, handle, ensure_ascii=False, indent=2)


def _detection_from_json(
    parsed: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    fallback_label: str,
) -> DetectionResult:
    box = parsed.get("box_2d") or parsed.get("bbox") or {}
    bbox = _bbox_from_value(box)
    bbox = bbox.denormalized_if_needed(image_width, image_height).clamp(
        image_width, image_height
    )
    confidence = _clamp_float(parsed.get("confidence", 0.75 if bbox.is_valid() else 0.0), 0.0, 1.0)
    found = bool(parsed.get("found", bbox.is_valid())) and bbox.is_valid()
    head_px, tail_px = _orientation_points_from_json(parsed, image_width, image_height)
    return DetectionResult(
        found=found,
        label=str(parsed.get("label") or fallback_label),
        confidence=confidence,
        bbox=bbox,
        notes=str(parsed.get("notes") or ""),
        source="vlm",
        head_px=head_px,
        tail_px=tail_px,
    )


def _bbox_from_value(value: Any) -> BoundingBox:
    if isinstance(value, dict):
        return BoundingBox(
            float(value.get("x_min", value.get("xmin", value.get("x1", 0.0)))),
            float(value.get("y_min", value.get("ymin", value.get("y1", 0.0)))),
            float(value.get("x_max", value.get("xmax", value.get("x2", 0.0)))),
            float(value.get("y_max", value.get("ymax", value.get("y2", 0.0)))),
        )

    if isinstance(value, (list, tuple)):
        box = value
        if len(box) == 1 and isinstance(box[0], (list, tuple, dict)):
            return _bbox_from_value(box[0])
        if len(box) >= 4:
            return BoundingBox(float(box[0]), float(box[1]), float(box[2]), float(box[3]))

    return BoundingBox(0.0, 0.0, 0.0, 0.0)


def _orientation_points_from_json(
    parsed: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[PixelPoint | None, PixelPoint | None]:
    orientation = parsed.get("orientation") or parsed.get("keypoints") or {}
    if not isinstance(orientation, dict):
        return None, None

    head = (
        orientation.get("head_px")
        or orientation.get("head")
        or orientation.get("cap_px")
        or orientation.get("cap")
    )
    tail = (
        orientation.get("tail_px")
        or orientation.get("tail")
        or orientation.get("bottom_px")
        or orientation.get("base_px")
    )
    head_px = _point_from_value(head, image_width, image_height)
    tail_px = _point_from_value(tail, image_width, image_height)
    if head_px is None or tail_px is None:
        return None, None
    if not head_px.is_valid() or not tail_px.is_valid():
        return None, None
    return head_px, tail_px


def _point_from_value(value: Any, image_width: int, image_height: int) -> PixelPoint | None:
    point = None
    if isinstance(value, dict):
        x_value = value.get("x", value.get("u", value.get("col")))
        y_value = value.get("y", value.get("v", value.get("row")))
        if x_value is not None and y_value is not None:
            point = PixelPoint(float(x_value), float(y_value))
    elif isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], (list, tuple, dict)):
            return _point_from_value(value[0], image_width, image_height)
        if len(value) >= 2:
            point = PixelPoint(float(value[0]), float(value[1]))

    if point is None:
        return None
    return point.denormalized_if_needed(image_width, image_height).clamp(image_width, image_height)


def _clamp_float(value: Any, low: float, high: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, numeric))


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}
