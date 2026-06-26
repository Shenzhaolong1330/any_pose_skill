# Configuration Reference

Use `config.yaml` as the primary runtime contract. Values can be overridden by CLI flags where available.

## Target

```yaml
target:
  name: "leftmost sample bottle"
  description: "..."
```

- `name`: User-facing object name. It is passed to VLM detectors and shown in output.
- `description`: Extra detector guidance. For VLM use, include distractors and exact keypoint definitions. For Grounding DINO plus SAM, it is still useful as project documentation, but `grounded_sam.text_prompt` controls DINO.

## Detector

```yaml
detector:
  mode: "grounded_sam"
  color: "auto"
  validate_vlm_color: false
```

- `mode: "auto"`: Use HSV color detection for colored targets, otherwise VLM.
- `mode: "color"`: Force local HSV color detection.
- `mode: "vlm"`: Force OpenRouter VLM bbox detection.
- `mode: "grounded_sam"`: Use Grounding DINO for open-vocabulary boxes, select an instance locally, then optionally refine with SAM.
- `color`: Use `"auto"` unless forcing a color such as `"red"`, `"green"`, or `"blue"`.
- `validate_vlm_color`: Keep true for colored blocks; set false for transparent or black/clear objects.

## Grounded SAM

```yaml
grounded_sam:
  grounding_model: "IDEA-Research/grounding-dino-tiny"
  text_prompt: "sample bottle"
  selection: "leftmost"
  box_threshold: 0.25
  text_threshold: 0.25
  device: "auto"
  sam_model: "facebook/sam-vit-base"
  use_sam: true
  refine_bbox_with_mask: true
  cap_dark_threshold: 80
```

- `text_prompt`: Grounding DINO prompt. Keep it short and noun-like, for example `"sample bottle"`.
- `selection`: Supported values include `leftmost`, `rightmost`, `topmost`, `bottommost`, `largest`, and score-based default behavior.
- `box_threshold`: Lower to `0.15` if DINO misses objects; raise if it returns too many false positives.
- `text_threshold`: Lower if labels are weak; raise if text grounding is noisy.
- `device: "auto"`: Use CUDA if available, else CPU.
- `use_sam`: Set false to use DINO boxes only.
- `refine_bbox_with_mask`: Set false if SAM over-crops or selects the wrong region.
- `cap_dark_threshold`: Raise if the black cap is not detected; lower if shadows are being treated as cap.

## OpenRouter

```yaml
openrouter:
  model: "google/gemini-3.1-flash-image"
  timeout_s: 60.0
  temperature: 0.0
  use_json_schema: true
  require_parameters: true
  retry_without_json_schema: true
```

Use `.env` for `OPENROUTER_API_KEY`; never commit API keys. If a provider rejects `response_format`, keep `retry_without_json_schema: true`.

## RealSense

```yaml
realsense:
  serial_number: null
  width: 1280
  height: 720
  fps: 30
  warmup_frames: 30
  frame_timeout_ms: 20000
  capture_retries: 5
  reset_on_start: false
  reset_wait_s: 5.0
```

- `serial_number`: Set to a camera serial string or number when multiple RealSense devices are connected. Use `object-locator --list-devices` to inspect devices.
- `width`, `height`, `fps`: Must be supported by the device stream profile. `1280x720@30` is the highest common D435i color/depth profile used by this project; `848x480@30` is a more stable fallback.
- `warmup_frames`: Keep nonzero so auto exposure and depth stabilize.
- `frame_timeout_ms`: How long to wait for each RealSense frame before retrying. Increase if startup is slow.
- `capture_retries`: Number of frame wait attempts before failing.
- `reset_on_start`: Hardware-reset the RealSense before opening streams. Enable only when the camera was left in a bad USB/firmware state or use `--reset-realsense` for one run.
- `reset_wait_s`: Seconds to wait after hardware reset before creating the pipeline.

## Depth

```yaml
depth:
  strategy: "median"
  position_anchor: "auto"
  anchor_radius_px: 12
  inner_ratio: 0.7
  min_depth_m: 0.05
  max_depth_m: 6.0
  min_samples: 50
  fallback_min_samples: 5
  max_expand_ratio: 2.5
  expand_steps: 3
```

- `strategy: "median"`: Use median valid depth in the selected sampling region.
- `strategy: "foreground"`: Bias toward nearer depth clusters.
- `strategy: "center"`: Try center depth first, fall back to median.
- `position_anchor: "auto"`: Use `head` if keypoints exist; otherwise use `bbox`.
- `position_anchor: "head"`: Force black cap/head keypoint for bottles.
- `position_anchor: "tail"`: Force transparent body end keypoint for bottles.
- `position_anchor: "bbox"`: Use whole-object bbox center.
- `anchor_radius_px`: Sampling radius around the selected keypoint when using `head` or `tail`.
- `fallback_min_samples`: Allows sparse depth to produce a result. Keep small for tiny targets, but treat sparse outputs as less reliable.

## Output

```yaml
output:
  json: true
  result_json: "runs/results/{run_id}.json"
  history_dir: "runs/history"
  debug_image: "runs/latest_panel.jpg"
  debug_rgb_image: "runs/latest_rgb.jpg"
  debug_depth_image: "runs/latest_depth.jpg"
  vlm_response: "runs/latest_vlm_response.json"
  save_depth: null
```

`output.json: true` prints JSON to stdout.
`result_json` writes the final JSON result to a file. Use `{run_id}` in the path to prevent overwriting repeated measurements.
`history_dir` also writes each run to `runs/history/<run_id>/` with `result.json` and debug images, so repeated measurements do not overwrite earlier runs. Set it to `null` or an empty string to disable history archives.

## Calibration

```yaml
calibration:
  enabled: false
  active_camera: "head"
  file: "calibration/extrinsics.yaml"
```

- `enabled`: Set true only after real extrinsics are written.
- `active_camera`: Use `"head"` for the fixed head camera or `"wrist"` for the wrist camera.
- `file`: Path to the calibration YAML. Relative paths are resolved relative to `config.yaml`.

Calibration files use `parent_T_child`, meaning `p_parent = R * p_child + t`.
Prefer `base_T_camera`, `flange_T_camera`, and `base_T_flange`. If a calibration tool exports the opposite direction, `camera_T_base`, `camera_T_flange`, and `flange_T_base` are also accepted and inverted.

For the head camera, provide `base_T_camera`:

```yaml
cameras:
  head:
    base_T_camera:
      translation_m: [0.0, 0.0, 0.0]
      rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
```

For the wrist camera, provide `flange_T_camera` and a current `base_T_flange`:

```yaml
cameras:
  wrist:
    flange_T_camera:
      translation_m: [0.0, 0.0, 0.0]
      rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
    base_T_flange:
      file: "base_T_flange.yaml"
```

The code computes `base_T_camera = base_T_flange * flange_T_camera` for the wrist camera.
