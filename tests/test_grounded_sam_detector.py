import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_locator.grounded_sam_detector import (  # noqa: E402
    bbox_from_mask,
    build_grounding_dino_inputs,
    choose_sam_mask,
    estimate_sample_bottle_keypoints,
    select_candidate,
)
from object_locator.models import BoundingBox  # noqa: E402


class GroundedSamDetectorTests(unittest.TestCase):
    def test_build_grounding_dino_inputs_uses_plain_text_prompt(self):
        calls = {}

        class FakeProcessor:
            def __call__(self, **kwargs):
                calls.update(kwargs)
                return {"ok": True}

        result = build_grounding_dino_inputs(FakeProcessor(), object(), "sample bottle.")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(calls["text"], "sample bottle.")
        self.assertEqual(calls["return_tensors"], "pt")

    def test_selects_leftmost_candidate(self):
        result = {
            "boxes": np.array([[50, 10, 90, 40], [10, 20, 30, 50]], dtype=np.float32),
            "scores": np.array([0.9, 0.7], dtype=np.float32),
            "labels": ["sample bottle", "sample bottle"],
        }

        candidate = select_candidate(
            result,
            image_shape_hw=(100, 100),
            selection="leftmost",
            min_box_area_px=10,
            max_box_area_ratio=0.8,
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.bbox.x_min, 10)

    def test_choose_sam_mask_uses_highest_score(self):
        masks = np.zeros((1, 3, 12, 12), dtype=bool)
        masks[0, 0, 1:3, 1:3] = True
        masks[0, 1, 2:8, 2:8] = True
        masks[0, 2, 4:6, 4:6] = True
        scores = np.array([[[0.2, 0.9, 0.4]]], dtype=np.float32)

        mask, score = choose_sam_mask(masks, scores)

        self.assertEqual(score, np.float32(0.9))
        self.assertEqual(int(mask.sum()), 36)

    def test_bbox_from_mask(self):
        mask = np.zeros((20, 30), dtype=bool)
        mask[5:10, 7:15] = True

        bbox = bbox_from_mask(mask)

        self.assertEqual(bbox, BoundingBox(7.0, 5.0, 15.0, 10.0))

    def test_estimates_sample_bottle_keypoints_from_mask_and_black_cap(self):
        image = np.full((80, 160, 3), 220, dtype=np.uint8)
        mask = np.zeros((80, 160), dtype=bool)
        mask[35:45, 40:120] = True
        image[35:45, 105:120] = 10

        head, tail = estimate_sample_bottle_keypoints(
            image,
            mask,
            BoundingBox(40, 35, 120, 45),
            cap_dark_threshold=50,
            cap_min_area_px=10,
        )

        self.assertIsNotNone(head)
        self.assertIsNotNone(tail)
        self.assertGreater(head.x, tail.x)
        self.assertGreater(head.x, 105)
        self.assertLess(tail.x, 55)


if __name__ == "__main__":
    unittest.main()
