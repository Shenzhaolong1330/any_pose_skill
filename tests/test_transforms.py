import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_locator.models import BoundingBox, PositionEstimate  # noqa: E402
from object_locator.transforms import (  # noqa: E402
    RigidTransform,
    base_T_camera_from_calibration,
    position_base_from_calibration,
)


class TransformTests(unittest.TestCase):
    def test_rigid_transform_applies_translation(self):
        transform = RigidTransform.from_spec(
            "base_T_camera",
            {"translation_m": [1.0, 2.0, 3.0], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
        )

        self.assertEqual(transform.apply_point((0.1, 0.2, 0.3)), (1.1, 2.2, 3.3))

    def test_head_camera_uses_base_T_camera(self):
        transform, metadata = base_T_camera_from_calibration(
            {
                "cameras": {
                    "head": {
                        "base_T_camera": {
                            "translation_m": [1.0, 0.0, 0.0],
                            "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                        }
                    }
                }
            },
            active_camera="head",
            calibration_path=Path("calibration/extrinsics.yaml"),
        )

        self.assertEqual(metadata["transform_chain"], ["base_T_camera"])
        self.assertEqual(transform.apply_point((0.2, 0.0, 0.0)), (1.2, 0.0, 0.0))

    def test_wrist_camera_composes_base_flange_and_flange_camera(self):
        transform, metadata = base_T_camera_from_calibration(
            {
                "cameras": {
                    "wrist": {
                        "flange_T_camera": {
                            "translation_m": [0.0, 2.0, 0.0],
                            "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                        "base_T_flange": {
                            "translation_m": [1.0, 0.0, 0.0],
                            "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                }
            },
            active_camera="wrist",
            calibration_path=Path("calibration/extrinsics.yaml"),
        )

        self.assertEqual(metadata["transform_chain"], ["base_T_flange", "flange_T_camera"])
        self.assertEqual(transform.apply_point((0.0, 0.0, 3.0)), (1.0, 2.0, 3.0))

    def test_inverse_camera_transform_names_are_supported(self):
        transform, metadata = base_T_camera_from_calibration(
            {
                "cameras": {
                    "head": {
                        "camera_T_base": {
                            "translation_m": [-1.0, 0.0, 0.0],
                            "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                        }
                    }
                }
            },
            active_camera="head",
            calibration_path=Path("calibration/extrinsics.yaml"),
        )

        self.assertEqual(metadata["transform_chain"], ["inverse(camera_T_base)"])
        self.assertEqual(transform.apply_point((0.2, 0.0, 0.0)), (1.2, 0.0, 0.0))

    def test_position_base_from_calibration_outputs_json_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "extrinsics.yaml"
            calibration.write_text(
                """
cameras:
  head:
    base_T_camera:
      translation_m: [1.0, 2.0, 3.0]
      rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
""",
                encoding="utf-8",
            )
            config = root / "config.yaml"
            config.write_text("calibration:\n  file: extrinsics.yaml\n", encoding="utf-8")
            position = PositionEstimate(
                x_m=0.1,
                y_m=0.2,
                z_m=0.3,
                u_px=10.0,
                v_px=20.0,
                depth_m=0.3,
                sample_count=5,
                valid_fraction=0.5,
                strategy="median",
                bbox_used=BoundingBox(0, 0, 10, 10),
            )

            result = position_base_from_calibration(
                position,
                enabled=True,
                calibration_file="extrinsics.yaml",
                active_camera="head",
                config_path=config,
                position_anchor="tail",
            )

        self.assertTrue(result["available"])
        self.assertEqual(result["position_anchor"], "tail")
        self.assertAlmostEqual(result["x_m"], 1.1)
        self.assertAlmostEqual(result["y_m"], 2.2)
        self.assertAlmostEqual(result["z_m"], 3.3)
        self.assertEqual(result["source_position_camera_m"], {"x": 0.1, "y": 0.2, "z": 0.3})


if __name__ == "__main__":
    unittest.main()
