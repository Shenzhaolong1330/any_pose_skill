import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_locator.color_detector import (  # noqa: E402
    ColorDetectorConfig,
    color_ratio_in_box,
    detect_colored_object,
    resolve_target_color,
)
from object_locator.models import BoundingBox  # noqa: E402


class ColorDetectorTests(unittest.TestCase):
    def test_resolves_english_and_chinese_color_words(self):
        self.assertEqual(resolve_target_color("red cube"), "red")
        self.assertEqual(resolve_target_color("红色方块"), "red")
        self.assertEqual(resolve_target_color("blue block"), "blue")

    def test_detects_red_component(self):
        image = np.full((120, 160, 3), 255, dtype=np.uint8)
        image[45:75, 70:105] = (0, 0, 220)

        detection = detect_colored_object(
            image,
            "red cube",
            ColorDetectorConfig(min_area_px=100),
        )

        self.assertTrue(detection.found)
        self.assertEqual(detection.source, "color:red")
        self.assertLessEqual(detection.bbox.x_min, 70)
        self.assertLessEqual(detection.bbox.y_min, 45)
        self.assertGreaterEqual(detection.bbox.x_max, 105)
        self.assertGreaterEqual(detection.bbox.y_max, 75)

    def test_returns_not_found_without_color_word(self):
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        detection = detect_colored_object(image, "cube")
        self.assertFalse(detection.found)

    def test_color_ratio_in_box(self):
        image = np.full((50, 50, 3), 255, dtype=np.uint8)
        image[10:30, 10:30] = (0, 0, 220)

        red_box_ratio = color_ratio_in_box(image, BoundingBox(10, 10, 30, 30), color="red")
        empty_box_ratio = color_ratio_in_box(image, BoundingBox(35, 35, 45, 45), color="red")

        self.assertGreater(red_box_ratio, 0.9)
        self.assertEqual(empty_box_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
