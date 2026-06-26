# Output Format Reference

Use `object-locator --config config.yaml --json` or `output.json: true` to print final JSON to stdout.

```bash
object-locator --config config.yaml --json
```

The final JSON is also saved to `output.result_json`, normally `runs/results/{run_id}.json`, and to `runs/history/{run_id}/result.json` when history is enabled. Avoid shell redirection to `runs/latest_result.json` for repeated measurements because that fixed path will be overwritten by the shell.

`runs/latest_vlm_response.json` is detector trace or raw VLM response, not the final localization result.

## Top-Level Fields

```json
{
  "target": "leftmost sample bottle",
  "found": true,
  "run_id": "20260626_153012_123456",
  "detection": {},
  "position": {},
  "position_anchor": "head",
  "position_base": {},
  "orientation": {},
  "debug_outputs": {},
  "intrinsics": {},
  "realsense": {},
  "coordinate_frame": {}
}
```

- `found`: Whether the configured detector found the target.
- `run_id`: Timestamp-like identifier used for `runs/history/<run_id>/`.
- `detection`: 2D detection metadata.
- `position`: 3D position estimate for the selected anchor point.
- `position_anchor`: Which part was used for position, usually `head`, `tail`, or `bbox`.
- `position_base`: Optional transformed position in robot base frame.
- `orientation`: Optional direction estimate.
- `debug_outputs`: Paths to generated debug images.

When `output.result_json` or `output.history_dir` is enabled, `debug_outputs` includes paths such as `result_json`, `result_json_history`, `rgb_history`, `depth_history`, `panel_history`, and `detector_trace_history`.

## RealSense

```json
{
  "serial_number": "123456789012",
  "width": 1280,
  "height": 720,
  "fps": 30,
  "reset_on_start": false,
  "reset_wait_s": 5.0
}
```

- `width`, `height`, `fps`: Active stream profile requested from RealSense.
- `reset_on_start`: Whether this run attempted a hardware reset before opening streams.
- `reset_wait_s`: Wait time after hardware reset.

## Detection

```json
{
  "found": true,
  "label": "sample bottle",
  "confidence": 0.82,
  "bbox": {
    "x_min": 120.5,
    "y_min": 300.2,
    "x_max": 210.8,
    "y_max": 390.1
  },
  "notes": "...",
  "source": "grounded_sam",
  "head_px": {"x": 180.2, "y": 320.5},
  "tail_px": {"x": 130.1, "y": 370.2},
  "mask_area_px": 2450
}
```

- `bbox`: Pixel coordinates in the color image.
- `source`: `color`, `vlm`, or `grounded_sam`.
- `head_px`: For sample bottles, black cap/head center.
- `tail_px`: Opposite end of transparent body.
- `mask_area_px`: Number of pixels in SAM mask, when available.

## Position

```json
{
  "x_m": -0.123,
  "y_m": 0.045,
  "z_m": 0.682,
  "u_px": 180.2,
  "v_px": 320.5,
  "depth_m": 0.682,
  "sample_count": 87,
  "valid_fraction": 0.24,
  "strategy": "median",
  "bbox_used": {
    "x_min": 168.2,
    "y_min": 308.5,
    "x_max": 192.2,
    "y_max": 332.5
  }
}
```

- `x_m`, `y_m`, `z_m`: Position in meters relative to the RealSense color optical frame.
- `u_px`, `v_px`: Pixel coordinate used for deprojection.
- `depth_m`: Depth used at that pixel or sampling region.
- `sample_count`: Count of valid depth samples.
- `valid_fraction`: Valid depth ratio inside `bbox_used`.
- `strategy`: Depth strategy; suffix `_sparse` means fallback sparse depth was used.
- `bbox_used`: Actual depth sampling region, which may be a small box around `head_px` or `tail_px`.

## Orientation

```json
{
  "head_px": {"x": 180.2, "y": 320.5},
  "tail_px": {"x": 130.1, "y": 370.2},
  "angle_deg_image": -38.4,
  "vector_px": {"x": 0.79, "y": -0.61},
  "head_depth_m": 0.68,
  "tail_depth_m": null,
  "head_m": {"x": -0.12, "y": 0.04, "z": 0.68},
  "tail_m": null,
  "vector_3d": null,
  "definition": "tail_px -> head_px; for sample bottles, this points toward the black cap/head"
}
```

- `angle_deg_image`: 2D image-plane angle. In image coordinates, 0 degrees points right and 90 degrees points down.
- `vector_px`: Unit vector in pixel coordinates from tail to head.
- `vector_3d`: 3D unit vector when both endpoints have enough valid depth.
- `vector_3d: null` is common for transparent bottles because the tail often lacks depth.

## Base Position

When `calibration.enabled: true`, the result includes a base-frame projection:

```json
{
  "available": true,
  "enabled": true,
  "active_camera": "wrist",
  "frame": "base",
  "camera_frame": "wrist_realsense_color_optical_frame",
  "x_m": 0.45,
  "y_m": -0.18,
  "z_m": 0.32,
  "source_position_camera_m": {
    "x": -0.12,
    "y": 0.04,
    "z": 0.68
  },
  "position_anchor": "tail",
  "transform_chain": ["base_T_flange", "flange_T_camera"],
  "calibration_file": "calibration/extrinsics.yaml",
  "convention": "parent_T_child maps p_child to p_parent: p_parent = R * p_child + t"
}
```

If unavailable, the shape is still explicit:

```json
{
  "available": false,
  "enabled": true,
  "active_camera": "wrist",
  "calibration_file": "calibration/extrinsics.yaml",
  "position_anchor": "tail",
  "reason": "wrist camera requires base_T_flange, either inline or as a file reference"
}
```

Keep using `position` for camera-frame values. Use `position_base.x_m/y_m/z_m` only when `position_base.available` is true.

## Coordinate Frame

The coordinate frame is the RealSense color optical frame:

- Units: meters.
- `x`: right in the image.
- `y`: down in the image.
- `z`: forward from camera into the scene.
