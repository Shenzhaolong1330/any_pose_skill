from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .models import BoundingBox, DetectionResult, PixelPoint


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
class GroundingCandidate:
    bbox: BoundingBox
    score: float
    label: str


class GroundedSamDetector:
    def __init__(self, config: GroundedSamConfig) -> None:
        self.config = config
        self._torch = None
        self._dino_processor = None
        self._dino_model = None
        self._sam_processor = None
        self._sam_model = None
        self._device = None

    def detect(self, image_bgr: np.ndarray, target: str) -> DetectionResult:
        torch = self._load_torch()
        pil_image = self._bgr_to_pil(image_bgr)
        candidate = self._detect_with_grounding_dino(pil_image, image_bgr.shape[:2], target)
        if candidate is None:
            return DetectionResult(
                found=False,
                label=target,
                confidence=0.0,
                bbox=BoundingBox(0.0, 0.0, 0.0, 0.0),
                notes="Grounding DINO found no candidate boxes",
                source="grounded_sam",
            )

        mask = None
        sam_score = None
        notes = [f"Grounding DINO selected {self.config.selection}: {candidate.label}"]
        if self.config.use_sam:
            try:
                mask, sam_score = self._segment_with_sam(pil_image, candidate.bbox)
                if mask is not None:
                    notes.append(f"SAM mask area={int(mask.sum())} px")
            except Exception as exc:  # pragma: no cover - depends on optional model runtime
                notes.append(f"SAM failed, using DINO box only: {exc}")

        bbox = candidate.bbox
        if (
            mask is not None
            and self.config.refine_bbox_with_mask
            and int(mask.sum()) >= self.config.min_mask_area_px
        ):
            mask_bbox = bbox_from_mask(mask)
            if mask_bbox is not None:
                bbox = mask_bbox

        head_px = None
        tail_px = None
        if mask is not None and int(mask.sum()) >= self.config.min_mask_area_px:
            head_px, tail_px = estimate_sample_bottle_keypoints(
                image_bgr,
                mask,
                bbox,
                cap_dark_threshold=self.config.cap_dark_threshold,
                cap_min_area_px=self.config.cap_min_area_px,
            )
            if head_px and tail_px:
                notes.append("orientation estimated from SAM mask principal axis and dark cap")
            else:
                notes.append("orientation unavailable: black cap or mask axis was not reliable")

        confidence = candidate.score
        if sam_score is not None:
            confidence = float(min(1.0, 0.7 * candidate.score + 0.3 * sam_score))

        return DetectionResult(
            found=True,
            label=candidate.label or target,
            confidence=confidence,
            bbox=bbox,
            notes="; ".join(notes),
            source="grounded_sam",
            head_px=head_px,
            tail_px=tail_px,
            mask=mask,
        )

    def _detect_with_grounding_dino(
        self,
        pil_image,
        image_shape_hw: tuple[int, int],
        target: str,
    ) -> GroundingCandidate | None:
        torch = self._load_torch()
        processor, model = self._load_grounding_dino()
        prompt = _normalize_grounding_prompt(self.config.text_prompt or target)
        inputs = build_grounding_dino_inputs(processor, pil_image, prompt)
        inputs = _move_batch_to_device(inputs, self._device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = [pil_image.size[::-1]]
        try:
            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                target_sizes=target_sizes,
            )
        except TypeError:
            results = processor.post_process_grounded_object_detection(
                outputs,
                threshold=self.config.box_threshold,
                text_threshold=self.config.text_threshold,
                target_sizes=target_sizes,
            )

        return select_candidate(
            results[0],
            image_shape_hw=image_shape_hw,
            selection=self.config.selection,
            min_box_area_px=self.config.min_box_area_px,
            max_box_area_ratio=self.config.max_box_area_ratio,
        )

    def _segment_with_sam(self, pil_image, bbox: BoundingBox) -> tuple[np.ndarray | None, float | None]:
        torch = self._load_torch()
        processor, model = self._load_sam()
        box = [float(bbox.x_min), float(bbox.y_min), float(bbox.x_max), float(bbox.y_max)]
        inputs = processor(pil_image, input_boxes=[[box]], return_tensors="pt")
        inputs = _move_batch_to_device(inputs, self._device)

        with torch.no_grad():
            outputs = model(**inputs)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
        )[0]
        mask_array = _to_numpy(masks)
        scores = _to_numpy(getattr(outputs, "iou_scores", None))
        return choose_sam_mask(mask_array, scores)

    def _load_torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Grounded-SAM mode requires optional dependencies. Install with "
                "`pip install -e '.[grounded-sam]'` and ensure model weights can be downloaded."
            ) from exc
        self._torch = torch
        self._device = _resolve_device(torch, self.config.device)
        return torch

    def _load_grounding_dino(self):
        if self._dino_processor is not None and self._dino_model is not None:
            return self._dino_processor, self._dino_model
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Grounded-SAM mode requires transformers. Install with "
                "`pip install -e '.[grounded-sam]'`."
            ) from exc
        self._dino_processor = AutoProcessor.from_pretrained(self.config.grounding_model)
        self._dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.config.grounding_model
        ).to(self._device)
        self._dino_model.eval()
        return self._dino_processor, self._dino_model

    def _load_sam(self):
        if self._sam_processor is not None and self._sam_model is not None:
            return self._sam_processor, self._sam_model
        try:
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Grounded-SAM mode requires transformers with SAM support. Install with "
                "`pip install -e '.[grounded-sam]'`."
            ) from exc
        self._sam_processor = SamProcessor.from_pretrained(self.config.sam_model)
        self._sam_model = SamModel.from_pretrained(self.config.sam_model).to(self._device)
        self._sam_model.eval()
        return self._sam_processor, self._sam_model

    def _bgr_to_pil(self, image_bgr: np.ndarray):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Grounded-SAM mode requires Pillow. Install with "
                "`pip install -e '.[grounded-sam]'`."
            ) from exc
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)


def build_grounding_dino_inputs(processor, pil_image, prompt: str):
    """Build HF Grounding DINO inputs with the text shape expected by tokenizer."""
    return processor(images=pil_image, text=prompt, return_tensors="pt")


def select_candidate(
    result: dict[str, Any],
    *,
    image_shape_hw: tuple[int, int],
    selection: str,
    min_box_area_px: int,
    max_box_area_ratio: float,
) -> GroundingCandidate | None:
    boxes = _to_numpy(result.get("boxes", []))
    scores = _to_numpy(result.get("scores", []))
    labels = result.get("labels", [])
    if boxes is None or len(boxes) == 0:
        return None

    image_area = image_shape_hw[0] * image_shape_hw[1]
    max_area_px = max(1, int(image_area * max_box_area_ratio))
    candidates = []
    for index, raw_box in enumerate(boxes):
        if len(raw_box) < 4:
            continue
        bbox = BoundingBox(float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3]))
        bbox = bbox.clamp(image_shape_hw[1], image_shape_hw[0])
        area = bbox.area
        if area < min_box_area_px or area > max_area_px or not bbox.is_valid():
            continue
        score = float(scores[index]) if scores is not None and index < len(scores) else 0.0
        label = str(labels[index]) if index < len(labels) else "grounded object"
        candidates.append(GroundingCandidate(bbox=bbox, score=score, label=label))

    if not candidates:
        return None

    selection = selection.lower()
    if selection == "leftmost":
        return min(candidates, key=lambda item: (item.bbox.x_min, -item.score))
    if selection == "rightmost":
        return max(candidates, key=lambda item: (item.bbox.x_max, item.score))
    if selection == "topmost":
        return min(candidates, key=lambda item: (item.bbox.y_min, -item.score))
    if selection == "bottommost":
        return max(candidates, key=lambda item: (item.bbox.y_max, item.score))
    if selection == "largest":
        return max(candidates, key=lambda item: (item.bbox.area, item.score))
    return max(candidates, key=lambda item: item.score)


def choose_sam_mask(mask_array: np.ndarray, scores: np.ndarray | None) -> tuple[np.ndarray | None, float | None]:
    masks = np.asarray(mask_array)
    masks = np.squeeze(masks)
    if masks.ndim == 2:
        return masks.astype(bool), _best_score(scores)
    if masks.ndim != 3 or masks.shape[0] == 0:
        return None, None

    if scores is not None:
        flat_scores = np.asarray(scores).reshape(-1)
        if flat_scores.size >= masks.shape[0]:
            index = int(np.argmax(flat_scores[: masks.shape[0]]))
            return masks[index].astype(bool), float(flat_scores[index])

    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    index = int(np.argmax(areas))
    return masks[index].astype(bool), None


def bbox_from_mask(mask: np.ndarray) -> BoundingBox | None:
    y_indices, x_indices = np.where(mask.astype(bool))
    if x_indices.size == 0:
        return None
    return BoundingBox(
        float(x_indices.min()),
        float(y_indices.min()),
        float(x_indices.max() + 1),
        float(y_indices.max() + 1),
    )


def estimate_sample_bottle_keypoints(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: BoundingBox,
    *,
    cap_dark_threshold: int,
    cap_min_area_px: int,
) -> tuple[PixelPoint | None, PixelPoint | None]:
    mask_bool = mask.astype(bool)
    if int(mask_bool.sum()) < 2:
        return None, None

    endpoints = _mask_axis_endpoints(mask_bool)
    if endpoints is None:
        return None, None

    cap_center = _dark_cap_center(
        image_bgr,
        mask_bool,
        bbox,
        cap_dark_threshold=cap_dark_threshold,
        cap_min_area_px=cap_min_area_px,
    )
    if cap_center is None:
        return None, None

    endpoint_a, endpoint_b = endpoints
    dist_a = _point_distance(cap_center, endpoint_a)
    dist_b = _point_distance(cap_center, endpoint_b)
    tail = endpoint_b if dist_a <= dist_b else endpoint_a
    return cap_center, tail


def _mask_axis_endpoints(mask: np.ndarray) -> tuple[PixelPoint, PixelPoint] | None:
    y_indices, x_indices = np.where(mask)
    if x_indices.size < 2:
        return None
    points = np.column_stack([x_indices.astype(np.float32), y_indices.astype(np.float32)])
    center = points.mean(axis=0)
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    projections = centered @ axis
    p_low = center + axis * np.percentile(projections, 2.0)
    p_high = center + axis * np.percentile(projections, 98.0)
    return PixelPoint(float(p_low[0]), float(p_low[1])), PixelPoint(float(p_high[0]), float(p_high[1]))


def _dark_cap_center(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: BoundingBox,
    *,
    cap_dark_threshold: int,
    cap_min_area_px: int,
) -> PixelPoint | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray <= int(cap_dark_threshold)) & mask
    x0, y0, x1, y1 = bbox.to_int_roi(image_bgr.shape[1], image_bgr.shape[0])
    dark_roi = dark[y0:y1, x0:x1].astype(np.uint8)
    if dark_roi.size == 0:
        return None
    kernel = np.ones((3, 3), dtype=np.uint8)
    dark_roi = cv2.morphologyEx(dark_roi, cv2.MORPH_OPEN, kernel)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(dark_roi, connectivity=8)
    best_label = None
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= cap_min_area_px and area > best_area:
            best_label = label
            best_area = area
    if best_label is None:
        return None
    cx, cy = centroids[best_label]
    return PixelPoint(float(cx + x0), float(cy + y0))


def _normalize_grounding_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    return prompt if prompt.endswith(".") else f"{prompt}."


def _resolve_device(torch, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _move_batch_to_device(batch, device: str):
    if hasattr(batch, "to"):
        return batch.to(device)
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _point_distance(a: PixelPoint, b: PixelPoint) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _best_score(scores: np.ndarray | None) -> float | None:
    if scores is None:
        return None
    flat = np.asarray(scores).reshape(-1)
    if flat.size == 0:
        return None
    return float(np.max(flat))
