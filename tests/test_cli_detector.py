import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_locator.cli import _position_box_for_detection, _validate_or_replace_vlm_detection  # noqa: E402
from object_locator.models import BoundingBox, DetectionResult, PixelPoint  # noqa: E402


class CliDetectorTests(unittest.TestCase):
    def test_vlm_color_guard_replaces_bbox_without_target_color(self):
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        image[45:75, 70:105] = (0, 0, 220)
        vlm_detection = DetectionResult(
            found=True,
            label="red cube",
            confidence=0.95,
            bbox=BoundingBox(10, 10, 40, 40),
            source="vlm",
        )
        runtime = {
            "target": "red cube",
            "detector_color": "auto",
            "validate_vlm_color": True,
            "min_vlm_color_ratio": 0.02,
            "fallback_to_color_on_vlm_mismatch": True,
            "color_min_area_px": 100,
            "color_max_area_ratio": 0.2,
            "color_min_saturation": 60,
            "color_min_value": 40,
            "color_morph_kernel": 5,
            "vlm_response": None,
        }

        detection = _validate_or_replace_vlm_detection(image, runtime, vlm_detection)

        self.assertEqual(detection.source, "color:red")
        self.assertLessEqual(detection.bbox.x_min, 70)
        self.assertGreaterEqual(detection.bbox.x_max, 105)

    def test_position_box_uses_head_anchor_when_available(self):
        detection = DetectionResult(
            found=True,
            label="sample bottle",
            confidence=0.8,
            bbox=BoundingBox(10, 20, 80, 90),
            head_px=PixelPoint(70, 30),
            tail_px=PixelPoint(20, 80),
        )

        box, anchor = _position_box_for_detection(
            detection,
            {"position_anchor": "auto", "anchor_radius_px": 5},
        )

        self.assertEqual(anchor, "head")
        self.assertEqual(box, BoundingBox(65, 25, 75, 35))

    def test_position_box_uses_tail_anchor_when_requested(self):
        detection = DetectionResult(
            found=True,
            label="sample bottle",
            confidence=0.8,
            bbox=BoundingBox(10, 20, 80, 90),
            head_px=PixelPoint(70, 30),
            tail_px=PixelPoint(20, 80),
        )

        box, anchor = _position_box_for_detection(
            detection,
            {"position_anchor": "tail", "anchor_radius_px": 6},
        )

        self.assertEqual(anchor, "tail")
        self.assertEqual(box, BoundingBox(14, 74, 26, 86))

    def test_position_box_falls_back_to_bbox_when_tail_unavailable(self):
        detection = DetectionResult(
            found=True,
            label="sample bottle",
            confidence=0.8,
            bbox=BoundingBox(10, 20, 80, 90),
            head_px=PixelPoint(70, 30),
        )

        box, anchor = _position_box_for_detection(
            detection,
            {"position_anchor": "tail", "anchor_radius_px": 6},
        )

        self.assertEqual(anchor, "bbox")
        self.assertEqual(box, detection.bbox)


if __name__ == "__main__":
    unittest.main()
