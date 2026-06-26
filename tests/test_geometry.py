import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_locator.geometry import (  # noqa: E402
    DepthEstimatorConfig,
    DepthEstimationError,
    OrientationEstimatorConfig,
    deproject_pixel,
    estimate_orientation,
    estimate_position_from_depth,
)
from object_locator.models import BoundingBox, CameraIntrinsics, DetectionResult, PixelPoint  # noqa: E402


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.intrinsics = CameraIntrinsics(
            width=640,
            height=480,
            fx=100.0,
            fy=100.0,
            ppx=320.0,
            ppy=240.0,
        )

    def test_deproject_center(self):
        self.assertEqual(deproject_pixel(self.intrinsics, 320.0, 240.0, 2.0), (0.0, 0.0, 2.0))

    def test_estimate_position_uses_bbox_center_and_median_depth(self):
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        bbox = BoundingBox(400.0, 280.0, 440.0, 300.0)
        estimate = estimate_position_from_depth(
            depth,
            self.intrinsics,
            bbox,
            DepthEstimatorConfig(min_samples=5, strategy="median"),
        )
        self.assertAlmostEqual(estimate.x_m, 2.0)
        self.assertAlmostEqual(estimate.y_m, 1.0)
        self.assertAlmostEqual(estimate.z_m, 2.0)

    def test_normalized_bbox_is_supported(self):
        depth = np.full((480, 640), 1.0, dtype=np.float32)
        bbox = BoundingBox(0.45, 0.45, 0.55, 0.55)
        estimate = estimate_position_from_depth(
            depth,
            self.intrinsics,
            bbox,
            DepthEstimatorConfig(min_samples=5, strategy="median"),
        )
        self.assertAlmostEqual(estimate.x_m, 0.0, places=5)
        self.assertAlmostEqual(estimate.y_m, 0.0, places=5)
        self.assertAlmostEqual(estimate.z_m, 1.0, places=5)

    def test_invalid_depth_raises(self):
        depth = np.zeros((480, 640), dtype=np.float32)
        with self.assertRaises(DepthEstimationError):
            estimate_position_from_depth(
                depth,
                self.intrinsics,
                BoundingBox(100.0, 100.0, 200.0, 200.0),
                DepthEstimatorConfig(min_samples=5),
            )

    def test_sparse_bbox_depth_can_be_used_for_tiny_object(self):
        depth = np.zeros((480, 640), dtype=np.float32)
        depth[100, 100:106] = 1.2
        bbox = BoundingBox(96.0, 96.0, 112.0, 112.0)

        estimate = estimate_position_from_depth(
            depth,
            self.intrinsics,
            bbox,
            DepthEstimatorConfig(
                min_samples=50,
                fallback_min_samples=5,
                inner_ratio=1.0,
                strategy="median",
            ),
        )

        self.assertEqual(estimate.sample_count, 6)
        self.assertEqual(estimate.strategy, "median_sparse")
        self.assertAlmostEqual(estimate.z_m, 1.2, places=5)

    def test_depth_search_expands_around_bbox_when_needed(self):
        depth = np.zeros((480, 640), dtype=np.float32)
        depth[92:95, 92:95] = 0.8
        bbox = BoundingBox(100.0, 100.0, 110.0, 110.0)

        estimate = estimate_position_from_depth(
            depth,
            self.intrinsics,
            bbox,
            DepthEstimatorConfig(
                min_samples=5,
                fallback_min_samples=3,
                inner_ratio=1.0,
                max_expand_ratio=3.0,
                expand_steps=2,
                strategy="median",
            ),
        )

        self.assertEqual(estimate.sample_count, 9)
        self.assertGreater(estimate.bbox_used.width, bbox.width)
        self.assertAlmostEqual(estimate.z_m, 0.8, places=5)

    def test_estimate_orientation_tail_to_head(self):
        depth = np.full((480, 640), 1.0, dtype=np.float32)
        detection = DetectionResult(
            found=True,
            label="leftmost sample bottle",
            confidence=0.9,
            bbox=BoundingBox(100, 100, 200, 160),
            head_px=PixelPoint(200, 100),
            tail_px=PixelPoint(100, 100),
        )

        orientation = estimate_orientation(
            depth,
            self.intrinsics,
            detection,
            OrientationEstimatorConfig(window_radius_px=1, min_samples=1),
        )

        self.assertIsNotNone(orientation)
        self.assertAlmostEqual(orientation.angle_deg_image, 0.0)
        self.assertAlmostEqual(orientation.vector_px_x, 1.0)
        self.assertAlmostEqual(orientation.vector_px_y, 0.0)
        self.assertIsNotNone(orientation.vector_3d)


if __name__ == "__main__":
    unittest.main()
