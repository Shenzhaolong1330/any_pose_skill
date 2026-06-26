import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_locator.openrouter_vlm import OpenRouterError, OpenRouterVLMClient  # noqa: E402


class OpenRouterVLMTests(unittest.TestCase):
    def test_rejects_placeholder_key(self):
        with self.assertRaisesRegex(OpenRouterError, "placeholder"):
            OpenRouterVLMClient(
                api_key="sk-or-v1-your-key-here",
                model="google/gemini-2.5-flash-lite",
            )

    def test_rejects_non_openrouter_key_shape(self):
        with self.assertRaisesRegex(OpenRouterError, "sk-or-"):
            OpenRouterVLMClient(
                api_key="not-an-openrouter-key",
                model="google/gemini-2.5-flash-lite",
            )

    def test_retries_without_json_schema_when_primary_json_is_incomplete(self):
        client = OpenRouterVLMClient(
            api_key="sk-or-v1-valid-looking-test-key",
            model="google/gemini-2.5-flash-lite",
        )
        incomplete = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": '{\n  "found":'},
                }
            ]
        }
        complete = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"found": true, "label": "cube", "confidence": 0.8, '
                            '"box_2d": {"x_min": 1, "y_min": 2, "x_max": 10, "y_max": 20}, '
                            '"notes": "single visible cube"}'
                        )
                    },
                }
            ]
        }

        with patch.object(client, "_post_payload", side_effect=[incomplete, complete]) as post:
            detection = client.detect_object(
                np.zeros((32, 32, 3), dtype=np.uint8),
                "cube",
                retry_without_json_schema=True,
            )

        self.assertEqual(post.call_count, 2)
        self.assertTrue(detection.found)
        self.assertEqual(detection.label, "cube")

    def test_accepts_list_bbox_and_missing_found_fields(self):
        client = OpenRouterVLMClient(
            api_key="sk-or-v1-valid-looking-test-key",
            model="google/gemini-2.5-flash-lite",
        )
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"box_2d": [2, 3, 12, 18], "label": "red cube"}'
                    },
                }
            ]
        }

        with patch.object(client, "_post_payload", return_value=response):
            detection = client.detect_object(
                np.zeros((32, 32, 3), dtype=np.uint8),
                "red cube",
                retry_without_json_schema=False,
            )

        self.assertTrue(detection.found)
        self.assertEqual(detection.confidence, 0.75)
        self.assertEqual(detection.bbox.x_min, 2)
        self.assertEqual(detection.bbox.y_max, 18)

    def test_accepts_nested_list_bbox(self):
        client = OpenRouterVLMClient(
            api_key="sk-or-v1-valid-looking-test-key",
            model="google/gemini-2.5-flash-lite",
        )
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": '{"box_2d": [[2, 3, 12, 18]], "label": "red cube"}'
                    },
                }
            ]
        }

        with patch.object(client, "_post_payload", return_value=response):
            detection = client.detect_object(
                np.zeros((32, 32, 3), dtype=np.uint8),
                "red cube",
                retry_without_json_schema=False,
            )

        self.assertTrue(detection.found)
        self.assertEqual(detection.bbox.x_max, 12)

    def test_parses_orientation_keypoints(self):
        client = OpenRouterVLMClient(
            api_key="sk-or-v1-valid-looking-test-key",
            model="google/gemini-2.5-flash-lite",
        )
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"found": true, "label": "leftmost sample bottle", '
                            '"confidence": 0.9, '
                            '"box_2d": {"x_min": 20, "y_min": 10, "x_max": 80, "y_max": 60}, '
                            '"orientation": {"head_px": {"x": 70, "y": 20}, '
                            '"tail_px": {"x": 25, "y": 55}}, '
                            '"notes": "leftmost visible bottle"}'
                        )
                    },
                }
            ]
        }

        with patch.object(client, "_post_payload", return_value=response):
            detection = client.detect_object(
                np.zeros((100, 100, 3), dtype=np.uint8),
                "leftmost sample bottle",
                retry_without_json_schema=False,
            )

        self.assertTrue(detection.found)
        self.assertIsNotNone(detection.head_px)
        self.assertIsNotNone(detection.tail_px)
        self.assertEqual(detection.head_px.x, 70)
        self.assertEqual(detection.tail_px.y, 55)


if __name__ == "__main__":
    unittest.main()
