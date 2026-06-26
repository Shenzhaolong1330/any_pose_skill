import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_locator.models import BoundingBox, DetectionResult, PositionEstimate  # noqa: E402
from object_locator.visualization import (  # noqa: E402
    colorize_depth,
    draw_debug_panel,
    draw_depth_detection,
)


class VisualizationTests(unittest.TestCase):
    def test_draws_depth_and_panel_with_position_overlay(self):
        color = np.zeros((48, 64, 3), dtype=np.uint8)
        depth = np.full((48, 64), 1.2, dtype=np.float32)
        detection = DetectionResult(
            found=True,
            label="cube",
            confidence=0.9,
            bbox=BoundingBox(20, 12, 40, 32),
        )
        position = PositionEstimate(
            x_m=0.1,
            y_m=0.0,
            z_m=1.2,
            u_px=30.0,
            v_px=22.0,
            depth_m=1.2,
            sample_count=200,
            valid_fraction=1.0,
            strategy="median",
            bbox_used=BoundingBox(23, 15, 37, 29),
        )

        depth_debug = draw_depth_detection(depth, detection, position)
        panel = draw_debug_panel(color, depth, detection, position)

        self.assertEqual(depth_debug.shape, (48, 64, 3))
        self.assertEqual(panel.shape, (48, 128, 3))
        self.assertGreater(int(panel.sum()), 0)

    def test_colorize_depth_handles_invalid_values(self):
        depth = np.array([[0.0, np.nan, 1.0, 2.0]], dtype=np.float32)
        colorized = colorize_depth(depth)
        self.assertEqual(colorized.shape, (1, 4, 3))
        self.assertTrue(np.all(colorized[0, 0] == 0))
        self.assertTrue(np.all(colorized[0, 1] == 0))


if __name__ == "__main__":
    unittest.main()
