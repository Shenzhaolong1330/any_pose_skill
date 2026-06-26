import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_locator.config import DEFAULT_MODEL, config_from_dict, load_config  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_default_config_when_file_missing(self):
        config = load_config("/tmp/object-locator-missing-config.yaml")
        self.assertEqual(config.openrouter.model, DEFAULT_MODEL)
        self.assertEqual(config.realsense.width, 640)
        self.assertFalse(config.realsense.reset_on_start)
        self.assertEqual(config.detector.mode, "auto")

    def test_config_from_dict_reads_nested_values(self):
        config = config_from_dict(
            {
                "target": {"name": "blue cube"},
                "detector": {"mode": "color", "color": "blue", "min_area_px": 99},
                "grounded_sam": {
                    "text_prompt": "sample bottle",
                    "selection": "leftmost",
                    "box_threshold": 0.3,
                    "cap_dark_threshold": 70,
                },
                "openrouter": {
                    "model": "openai/gpt-4o-mini",
                    "max_tokens": 2048,
                    "require_parameters": True,
                    "retry_without_json_schema": False,
                },
                "realsense": {
                    "serial_number": "123",
                    "width": 848,
                    "height": 480,
                    "frame_timeout_ms": 20000,
                    "capture_retries": 5,
                    "reset_on_start": True,
                    "reset_wait_s": 7.5,
                },
                "depth": {
                    "strategy": "foreground",
                    "position_anchor": "tail",
                    "anchor_radius_px": 9,
                    "max_depth_m": 3.5,
                    "fallback_min_samples": 4,
                    "max_expand_ratio": 3.0,
                    "expand_steps": 2,
                },
                "calibration": {
                    "enabled": True,
                    "file": "calibration/extrinsics.yaml",
                    "active_camera": "wrist",
                },
                "output": {
                    "json": True,
                    "result_json": "runs/results/{run_id}.json",
                    "history_dir": "runs/archive",
                    "debug_image": None,
                    "vlm_response": "runs/raw.json",
                },
            }
        )
        self.assertEqual(config.target.name, "blue cube")
        self.assertEqual(config.detector.mode, "color")
        self.assertEqual(config.detector.color, "blue")
        self.assertEqual(config.detector.min_area_px, 99)
        self.assertTrue(config.detector.validate_vlm_color)
        self.assertTrue(config.detector.fallback_to_color_on_vlm_mismatch)
        self.assertEqual(config.grounded_sam.text_prompt, "sample bottle")
        self.assertEqual(config.grounded_sam.selection, "leftmost")
        self.assertEqual(config.grounded_sam.box_threshold, 0.3)
        self.assertEqual(config.grounded_sam.cap_dark_threshold, 70)
        self.assertEqual(config.openrouter.model, "openai/gpt-4o-mini")
        self.assertEqual(config.openrouter.max_tokens, 2048)
        self.assertTrue(config.openrouter.require_parameters)
        self.assertFalse(config.openrouter.retry_without_json_schema)
        self.assertEqual(config.realsense.serial_number, "123")
        self.assertEqual(config.realsense.width, 848)
        self.assertEqual(config.realsense.frame_timeout_ms, 20000)
        self.assertEqual(config.realsense.capture_retries, 5)
        self.assertTrue(config.realsense.reset_on_start)
        self.assertEqual(config.realsense.reset_wait_s, 7.5)
        self.assertEqual(config.depth.strategy, "foreground")
        self.assertEqual(config.depth.position_anchor, "tail")
        self.assertEqual(config.depth.anchor_radius_px, 9)
        self.assertEqual(config.depth.max_depth_m, 3.5)
        self.assertEqual(config.depth.fallback_min_samples, 4)
        self.assertEqual(config.depth.max_expand_ratio, 3.0)
        self.assertEqual(config.depth.expand_steps, 2)
        self.assertTrue(config.calibration.enabled)
        self.assertEqual(config.calibration.file, "calibration/extrinsics.yaml")
        self.assertEqual(config.calibration.active_camera, "wrist")
        self.assertTrue(config.output.json)
        self.assertEqual(config.output.result_json, "runs/results/{run_id}.json")
        self.assertEqual(config.output.history_dir, "runs/archive")
        self.assertIsNone(config.output.debug_image)
        self.assertEqual(config.output.debug_rgb_image, "runs/latest_rgb.jpg")
        self.assertEqual(config.output.debug_depth_image, "runs/latest_depth.jpg")
        self.assertEqual(config.output.vlm_response, "runs/raw.json")

    def test_load_config_from_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("target:\n  name: banana\n", encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.target.name, "banana")


if __name__ == "__main__":
    unittest.main()
