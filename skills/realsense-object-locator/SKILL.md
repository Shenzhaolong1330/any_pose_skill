---
name: realsense-object-locator
description: Use when configuring, running, debugging, or extending the RealSense D435i RGB-D object localization project in this repository, including config.yaml, RealSense serial selection, OpenRouter VLM detection, Grounding DINO plus SAM segmentation, depth-to-3D projection, sample bottle position and orientation, debug images, and JSON result interpretation.
---

# RealSense Object Locator

## Overview

Use this skill to operate the `any_pose_skill` project that estimates an object's 3D position relative to a RealSense D435i color optical frame. The project combines RGB detection, depth sampling, and camera intrinsics; it supports local color detection, OpenRouter VLM bounding boxes, and Grounding DINO plus SAM for sample bottles and other open-vocabulary targets.

If this skill is stored inside the repository at `skills/realsense-object-locator`, the project root is normally two directories above the skill directory. If the repository has been moved, locate the root by finding `pyproject.toml`, `config.yaml`, and `src/object_locator/`.

## Core Workflow

1. Inspect `config.yaml` before running or changing behavior.
2. Use `object-locator --config config.yaml` for human-readable output.
3. Use `object-locator --config config.yaml --json` for machine-readable output.
4. Use `output.result_json: "runs/results/{run_id}.json"` and `output.history_dir` for non-overwriting saved results; avoid redirecting repeated runs to `runs/latest_result.json`.
5. Inspect `runs/latest_panel.jpg`, `runs/latest_rgb.jpg`, `runs/latest_depth.jpg`, and `runs/latest_vlm_response.json` when localization looks wrong.
6. Run lightweight validation after code changes: `PYTHONPATH=src python3 -m unittest discover` and `python3 -m compileall src tests setup.py`.

Do not store API keys, `.env`, downloaded model weights, `.venv`, or `runs/` artifacts in the skill. Use `.env.example` and project installation docs for environment setup.

## Task Guide

For configuration questions, read `references/config-schema.md`.

For incorrect detections, sparse depth, SAM mask problems, OpenRouter errors, or RealSense setup problems, read `references/debugging.md`.

For JSON fields, coordinate frame definitions, and result interpretation, read `references/output-format.md`.

For a portable run helper, inspect or execute `scripts/run_once.sh`. It runs from the repository root inferred from the skill location and can optionally write JSON to a file.

## Project-Specific Rules

- Treat `position.x_m`, `position.y_m`, and `position.z_m` as meters in the RealSense color optical frame: x right, y down, z forward from the camera.
- Treat `position_base.x_m`, `position_base.y_m`, and `position_base.z_m` as meters in the robot base frame only when `position_base.available` is true.
- Use `calibration/extrinsics.yaml` for static extrinsics. Head cameras use `base_T_camera`; wrist cameras use `base_T_flange * flange_T_camera`.
- Treat `detection.bbox` as pixel coordinates in the aligned color image.
- For sample bottles, treat `head_px` as the black cap/head center and `tail_px` as the opposite transparent body end. The direction is `tail_px -> head_px`.
- Use `depth.position_anchor: "tail"` when the requested position is the transparent tube/body end. Warn that RealSense depth can be missing on transparent material.
- Prefer `depth.position_anchor: "head"` or `"auto"` when reliability matters more than locating the transparent end, because the black cap usually has more reliable depth than the transparent body.
- Use `detector.mode: "grounded_sam"` for multiple sample bottles or open-vocabulary local segmentation.
- Use `detector.mode: "color"` or `"auto"` for simple colored blocks, especially red/green/blue cubes.
- Use `detector.mode: "vlm"` when local detectors cannot describe the target well, and keep `openrouter.use_json_schema: true` unless the provider rejects structured output.

## Migration

To make this skill available on another computer after copying the repository, copy the skill folder into Codex's skill directory:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -a skills/realsense-object-locator "${CODEX_HOME:-$HOME/.codex}/skills/"
```

Then start a new Codex session so the skill metadata is discovered.
