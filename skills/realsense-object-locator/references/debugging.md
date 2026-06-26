# Debugging Reference

Start from the latest command, `config.yaml`, and the generated files in `runs/`.

## Visual Debug Images

- `runs/latest_rgb.jpg`: Color frame with detector bbox, SAM mask, position point, and orientation arrow.
- `runs/latest_depth.jpg`: Aligned depth visualization with the same overlays.
- `runs/latest_panel.jpg`: RGB and depth side by side.
- `runs/latest_vlm_response.json`: Raw VLM output or local detector trace. It is not the final result JSON.

Overlay meanings:

- Green box: detector bbox.
- Orange translucent region: SAM mask.
- Yellow thin box: actual region used for depth sampling.
- White cross: final pixel used for 3D deprojection.
- Purple arrow: `tail_px -> head_px`.
- Yellow point: head/cap keypoint.
- Purple point: tail keypoint.

## Common Problems

### Grounding DINO misses the object

Check `grounded_sam.text_prompt`. Prefer short nouns like `"sample bottle"` instead of long instructions. Lower `grounded_sam.box_threshold` to `0.15` for recall. If many false positives appear, raise it toward `0.35`.

### SAM mask is wrong

Set `grounded_sam.refine_bbox_with_mask: false` to keep the DINO bbox while still allowing SAM-related keypoint logic to be inspected. If SAM consistently fails, set `grounded_sam.use_sam: false` and rely on DINO bbox only.

### Leftmost bottle selection is wrong

The local selector uses bbox geometry after DINO returns candidates. Inspect `latest_rgb.jpg` to see whether DINO found all bottles. If it found the wrong object, improve `text_prompt`; if it found all objects but selection is wrong, inspect `select_candidate` in `src/object_locator/grounded_sam_detector.py`.

### Bottle direction is wrong

For sample bottles, direction is `tail_px -> head_px`, where `head_px` is the dark cap. Tune:

```yaml
grounded_sam:
  cap_dark_threshold: 80
  cap_min_area_px: 20
```

Raise `cap_dark_threshold` if the cap is not detected. Lower it if shadows or black table regions are being mistaken for cap.

### Depth samples are too sparse

Transparent bottles often return invalid RealSense depth. Prefer:

```yaml
depth:
  position_anchor: "head"
  fallback_min_samples: 5
  max_expand_ratio: 2.5
```

If the output strategy ends with `_sparse`, treat the 3D estimate as lower confidence. Use the debug depth image to confirm whether the depth region is on the cap or on transparent plastic.

### Position is on the wrong part of the object

Set `depth.position_anchor`:

- `"head"` for sample bottle cap/head.
- `"tail"` for the transparent tube/body end.
- `"bbox"` for whole-object bbox center.
- `"auto"` to use head when available, otherwise bbox.

The project currently supports `auto`, `bbox`, `head`, and `tail`. To add `mask_centroid`, update `DepthConfig` in `src/object_locator/config.py`, validation in `src/object_locator/cli.py`, and `_position_box_for_detection`.

### Transparent tail position is unavailable

When `position_anchor: "tail"` is selected, the code samples depth near `tail_px`. D435i depth on transparent material can be invalid. If the command reports too few samples, try increasing `anchor_radius_px`, lowering `fallback_min_samples`, improving lighting/background contrast, or accept that the tail may need to be estimated from geometry rather than directly measured by depth.

### OpenRouter returns HTTP 401

Check `.env` and shell environment. `OPENROUTER_API_KEY` must not be the placeholder from `.env.example`. If a shell export exists, it may override the `.env` value.

### VLM JSON parsing fails

Inspect `runs/latest_vlm_response.json`. Keep `openrouter.retry_without_json_schema: true`. If the provider cannot honor structured output, try a stronger image-capable model or a provider that supports requested parameters.

### RealSense device is not found

Run:

```bash
object-locator --list-devices
```

Then set `realsense.serial_number` to the matching device. Also check librealsense installation, udev rules, USB bandwidth, and whether another process is holding the camera.

### Python venv cannot be created

On Ubuntu or Debian, install venv support:

```bash
sudo apt update
sudo apt install python3.10-venv python3-pip
```

Then recreate `.venv`.

### Base-frame output is unavailable

Check `calibration.enabled`, `calibration.file`, and `calibration.active_camera` in `config.yaml`. For head camera mode, `calibration/extrinsics.yaml` must define `cameras.head.base_T_camera`. For wrist camera mode, it must define `cameras.wrist.flange_T_camera` plus a current `base_T_flange`, either inline or through `calibration/base_T_flange.yaml`.

Remember the transform convention: `parent_T_child` maps child-frame points to parent-frame points. If the calibration tool exports `camera_T_base`, `camera_T_flange`, or `flange_T_base`, the project can read those inverse names and invert them automatically.

## Validation After Code Changes

Run:

```bash
PYTHONPATH=src python3 -m unittest discover
python3 -m compileall src tests setup.py
```

Do not require live camera access for ordinary unit tests. Only run `object-locator --config config.yaml` when RealSense hardware and model dependencies are available.
