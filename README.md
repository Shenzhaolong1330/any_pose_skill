# RealSense RGB-D 物体/部位定位

这个项目用 Intel RealSense D435i 的 RGB + 深度图，估计 RGB 画面中某个指定物体相对相机的位置。

项目当前支持三类检测方式：

- `grounded_sam`: Grounding DINO 按文本找目标，SAM 分割 mask，适合样本瓶、试管、小物体实例选择。
- `color`: 本地 HSV 颜色检测，适合红/绿/蓝方块这类颜色明确的目标。
- `vlm`: OpenRouter VLM 按自然语言返回 bbox/keypoint，适合复杂语义目标或临时指定部位。

## 工作原理

整体链路是：

1. RealSense 同时采集 RGB 和 depth，并把 depth 对齐到 color。
2. 在 RGB 图里检测目标，得到 bbox、mask 或关键点。
3. 根据 `depth.position_anchor` 选择要定位的部位，例如整个 bbox、黑色头部 `head`、透明尾端 `tail`。
4. 在对应像素附近取有效 depth，通常取中位数来抑制噪声。
5. 用 RealSense color camera 内参反投影成 3D 坐标。
6. 如果启用外参标定，再把相机坐标变换到机器人 base 坐标。

输出坐标是相机光学坐标系下的米制坐标：`x` 向右，`y` 向下，`z` 从相机向前。

当前 `config.yaml` 默认面向“桌面上散落的样本瓶/试管”：

```yaml
detector:
  mode: "grounded_sam"

grounded_sam:
  text_prompt: "sample bottle"
  selection: "leftmost"

depth:
  position_anchor: "tail"
```

这表示：先找所有样本瓶，选择图像最左侧的一个，用 SAM 得到瓶身 mask，再估计 `head_px` 和 `tail_px`，最后输出透明尾端 `tail` 附近的 3D 坐标。

## 适用范围

比较适合：

- 桌面或工作台上静止/低速移动的目标定位。
- 多个相似物体中选择最左、最右、最大、最上等实例。
- 样本瓶、试管、小瓶等有稳定几何外形的目标。
- 获取固定语义部位的位置，例如样本瓶黑色头部 `head` 或透明尾端 `tail`。
- 机器人抓取前的粗定位或中等精度定位。
- 已标定头部相机 `base_T_camera` 或腕部相机 `flange_T_camera` 后输出 base 坐标。

不太适合：

- 亚毫米级精密测量或装配级定位。
- 快速运动目标。
- 严重遮挡、堆叠、贴在一起的物体。
- RGB 中目标和背景几乎不可区分的场景。
- 任意开放语义部位的高鲁棒定位，例如“杯子把手最左端”这类未专门建模的细部。
- 透明、强反光、黑亮材质的直接深度测量，尤其是 D435i 经常量不到透明表面。

透明试管/样本瓶尾端是当前项目支持的一个专门场景：RGB 上会估计 `tail_px`，但 3D 位置仍依赖 `tail_px` 附近是否有有效 depth。如果输出 `median_sparse` 或报 depth samples 不足，需要查看 `runs/latest_depth.jpg` 判断 RealSense 是否真的量到了深度。

## 使用方式概览

常用模式选择：

```yaml
# 样本瓶/试管，推荐
detector:
  mode: "grounded_sam"

# 红色方块等颜色明确目标
detector:
  mode: "color"

# 复杂语义目标或临时自然语言部位
detector:
  mode: "vlm"
```

常用位置锚点：

```yaml
depth:
  position_anchor: "bbox"  # 整个检测框中心
  position_anchor: "head"  # 样本瓶黑色头部/瓶盖
  position_anchor: "tail"  # 样本瓶透明尾端
  position_anchor: "auto"  # 有 head 用 head，否则用 bbox
```

运行一次：

```bash
object-locator --config config.yaml --json
```

最终 JSON 会自动保存到 `output.result_json` 指定的位置，默认是 `runs/results/<run_id>.json`，不会覆盖。不要再用 `> runs/latest_result.json` 做重复实验；那个固定文件名一定会被 shell 覆盖。

最终 JSON 中最重要的字段：

- `position`: 相机坐标系下的位置，单位米。
- `position_anchor`: 本次定位的是 `bbox`、`head`、`tail` 还是自动选择结果。
- `detection.bbox`: RGB 图上的目标框，单位像素。
- `detection.head_px` / `detection.tail_px`: 样本瓶方向关键点。
- `position_base`: 启用标定后，机器人 base 坐标系下的位置。
- `debug_outputs`: RGB/depth/panel 调试图路径。

如果经常遇到：

```text
Frame didn't arrive within ...
```

这通常是 RealSense 没及时吐帧，不是 DINO/SAM 或 VLM 的问题。正常通过 `object-locator` 退出会调用 `pipeline.stop()`；但如果进程被强杀、USB/固件卡住，或者 `realsense-viewer`/其他程序占着设备，下一次启动可能仍然拿不到帧。最高公共 RGB/depth 档是 `1280x720@30`，如果它频繁超时，先退到 `848x480@30` 判断是不是 USB 带宽/稳定性问题。

可以临时硬重置相机：

```bash
object-locator --config config.yaml --json --reset-realsense
```

也可以在配置里加大等待并启用重置：

```yaml
realsense:
  width: 1280
  height: 720
  frame_timeout_ms: 20000
  capture_retries: 5
  warmup_frames: 30
  reset_on_start: true
  reset_wait_s: 5.0
```

## 安装

Ubuntu/Debian 如果创建 venv 时报 `ensurepip is not available`，先安装 venv 包：

```bash
sudo apt update
sudo apt install python3.10-venv python3-pip
```

然后重新创建虚拟环境：

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
cp .env.example .env
```

如果要用 Grounding DINO + SAM 本地检测/分割：

```bash
pip install -e ".[dev,grounded-sam]"
```

第一次运行会从 Hugging Face 下载模型权重，建议有 CUDA GPU；CPU 也能跑但会慢很多。

如果使用 `detector.mode: "vlm"` 或 `auto` 需要回退到 VLM，再编辑 `.env`：

```bash
OPENROUTER_API_KEY=<your-openrouter-api-key>
OPENROUTER_MODEL=google/gemini-2.5-flash-lite
```

如果只使用 `detector.mode: "grounded_sam"` 或 `color`，不需要 OpenRouter API key。

如果运行时报：

```text
OpenRouter returned HTTP 401
```

优先检查 `.env` 里的 `OPENROUTER_API_KEY`。不要保留 `.env.example` 里的 `<your-openrouter-api-key>` 占位符；如果你在 shell 里手动 `export OPENROUTER_API_KEY=...` 过，也确认它没有覆盖 `.env` 中的新 key。

如果 `pyrealsense2` 安装或相机权限有问题，先确认 librealsense / udev rules 已配置，并且普通用户能访问 D435i。

## VLM 模型建议

VLM 只在 `detector.mode: "vlm"`，或者 `auto` 需要调用 OpenRouter 时使用。默认用 `google/gemini-2.5-flash-lite`：这个项目只需要 VLM 做一次目标定位并返回 JSON，Flash-Lite 通常在速度和成本上更适合作为原型默认值。

如果 bbox 不够稳，可以在 `config.yaml` 里切换：

- `google/gemini-2.5-flash`: 更推荐的质量优先选项。
- `openai/gpt-4o-mini`: 成本和质量都比较均衡的 OpenAI 路线。
- `openai/gpt-4o`: 更强但更贵，适合复杂场景验证。

选模型时优先确认它支持图像输入和结构化输出。这个项目会发送 base64 图像，并默认使用 `response_format: json_schema`。

## 配置

主要参数都在 `config.yaml`：

```yaml
target:
  name: "red cup"
  description: null

detector:
  mode: "auto"
  color: "auto"
  min_area_px: 150
  max_area_ratio: 0.2
  min_saturation: 60
  min_value: 40
  morph_kernel: 5
  validate_vlm_color: true
  min_vlm_color_ratio: 0.02
  fallback_to_color_on_vlm_mismatch: true

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
  min_box_area_px: 100
  max_box_area_ratio: 0.5
  min_mask_area_px: 100
  cap_dark_threshold: 80
  cap_min_area_px: 20

openrouter:
  model: "google/gemini-2.5-flash-lite"
  max_tokens: 1024
  use_json_schema: true
  require_parameters: false
  retry_without_json_schema: true

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

calibration:
  enabled: false
  active_camera: "head"
  file: "calibration/extrinsics.yaml"
```

`detector.mode` 有四种：

- `auto`: 默认。目标名里有 `red`、`blue`、`green`、`红色` 等颜色词时，用本地 HSV 颜色检测；否则调用 VLM。
- `color`: 强制用本地颜色检测，适合 `red cube`、`blue block` 这类颜色明确的目标。
- `vlm`: 强制用 OpenRouter VLM，适合鼠标、杯子、工具等非纯颜色目标。
- `grounded_sam`: 用 Grounding DINO 检测开放词汇目标，再用 SAM 对选中的 bbox 分割 mask，适合多个样本瓶这类实例选择任务。

对红色方块，建议保持：

```yaml
target:
  name: "red cube"

detector:
  mode: "auto"
  color: "auto"
```

如果你希望强制用 VLM 做精细 bbox 标注，可以切到更强的视觉模型：

```yaml
target:
  name: "red cube"
  description: "the small red cube/block on the white tabletop; do not mark the black mouse, robot arm, wires, power strip, green marker, shadows, or background"

detector:
  mode: "vlm"

openrouter:
  model: "google/gemini-3-pro-image"
  use_json_schema: true
  require_parameters: true
```

速度/成本优先可以试 `google/gemini-3.1-flash-image`；如果当前模型在你的 OpenRouter 账户不可用，再退回 `google/gemini-2.5-flash`。

对带颜色词的目标，即使强制 `mode: "vlm"`，默认也会做颜色校验：如果 VLM 返回的 bbox 内目标颜色比例低于 `min_vlm_color_ratio`，程序会把这个框判为不可信，并用本地颜色候选框替换。这个校验会写进 `runs/latest_vlm_response.json`。

样本瓶这类“黑色头部 + 透明瓶身”的目标建议用 VLM，并让模型返回方向关键点：

```yaml
target:
  name: "leftmost sample bottle"
  description: "There are several sample bottles scattered on the tabletop. Each sample bottle has a black cap/head and a transparent body. Select the leftmost sample bottle in the image by the full bottle bounding box. Return a tight box around the whole selected bottle. For orientation, head_px is the center of the black cap/head and tail_px is the opposite transparent body end; the pointing direction is tail_px -> head_px."

detector:
  mode: "vlm"
  validate_vlm_color: false

openrouter:
  model: "google/gemini-3.1-flash-image"
  use_json_schema: true
  require_parameters: true
```

方向定义为 `tail_px -> head_px`。对样本瓶来说，就是从透明瓶身尾端指向黑色头部/瓶盖。终端会输出 2D 图像角度；如果两个端点附近都有有效深度，还会输出 3D 单位方向向量。透明瓶身经常没有可靠深度，所以 3D 方向可能显示 unavailable，这是 D435i 对透明材质的物理限制，不是程序崩了。

更鲁棒的样本瓶方案是 Grounding DINO + SAM：

```yaml
target:
  name: "leftmost sample bottle"
  description: "There are several sample bottles scattered on the tabletop. Each sample bottle has a black cap/head and a transparent body. Select the leftmost sample bottle in the image by the full bottle bounding box."

detector:
  mode: "grounded_sam"
  validate_vlm_color: false

grounded_sam:
  text_prompt: "sample bottle"
  selection: "leftmost"
  box_threshold: 0.25
  text_threshold: 0.25
  use_sam: true
  cap_dark_threshold: 80

depth:
  position_anchor: "auto"
```

这个流程会先检测所有 `sample bottle`，本地按 `x_min` 选择最左侧 bbox，再用 SAM mask 的主轴和黑色 cap 区域估计 `tail -> head` 指向。`position_anchor: auto` 会在有 head keypoint 时优先用黑色 cap 附近深度估计位置，比透明瓶身整体 bbox 更稳。

如果你需要透明瓶身尾端的位置，把 `depth.position_anchor` 设为 `tail`：

```yaml
depth:
  position_anchor: "tail"
```

这时输出的 `position.x_m/y_m/z_m` 对应 `tail_px` 附近的小区域。注意 D435i 对透明材质深度经常缺失；如果尾端没有足够有效深度，程序会报 depth samples 不足，或者输出 `strategy=median_sparse` 这类稀疏深度结果。

## Base 坐标系输出

相机默认输出的是 RealSense color optical frame 下的位置。如果你已经完成外参标定，可以启用 base 坐标输出：

```yaml
calibration:
  enabled: true
  active_camera: "head"   # head 或 wrist
  file: "calibration/extrinsics.yaml"
```

标定文件预留在：

```text
calibration/extrinsics.yaml
calibration/base_T_flange.yaml
```

外参命名约定是 `parent_T_child`，表示把 child 坐标系里的点变换到 parent 坐标系：

```text
p_parent = R_parent_child * p_child + t_parent_child
```

建议优先填写 `base_T_camera`、`flange_T_camera`、`base_T_flange`。如果你的标定工具导出的是反方向，程序也接受 `camera_T_base`、`camera_T_flange` 或 `flange_T_base`，会自动求逆。

头部相机，也就是相机相对 base 固定时，填写：

```yaml
cameras:
  head:
    base_T_camera:
      translation_m: [0.0, 0.0, 0.0]
      rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
```

腕部相机，也就是相机相对法兰固定时，填写 `flange_T_camera`，并在运行时提供当前 `base_T_flange`：

```yaml
cameras:
  wrist:
    flange_T_camera:
      translation_m: [0.0, 0.0, 0.0]
      rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
    base_T_flange:
      file: "base_T_flange.yaml"
```

程序会计算：

```text
head:  p_base = base_T_camera * p_camera
wrist: p_base = base_T_flange * flange_T_camera * p_camera
```

JSON 输出会保留原始相机坐标 `position`，并额外增加 `position_base`：

```json
{
  "position": {
    "x_m": -0.12,
    "y_m": 0.04,
    "z_m": 0.68
  },
  "position_anchor": "tail",
  "position_base": {
    "available": true,
    "enabled": true,
    "active_camera": "head",
    "frame": "base",
    "camera_frame": "head_realsense_color_optical_frame",
    "x_m": 0.45,
    "y_m": -0.18,
    "z_m": 0.32,
    "source_position_camera_m": {
      "x": -0.12,
      "y": 0.04,
      "z": 0.68
    },
    "position_anchor": "tail",
    "transform_chain": ["base_T_camera"],
    "calibration_file": "calibration/extrinsics.yaml",
    "convention": "parent_T_child maps p_child to p_parent: p_parent = R * p_child + t"
  }
}
```

如果标定未启用或外参缺失，`position_base.available` 会是 `false`，并在 `reason` 里说明原因；相机坐标下的 `position` 仍然正常输出。

如果接了多台 RealSense，把 `realsense.serial_number` 改成对应序列号字符串即可，例如：

```yaml
realsense:
  serial_number: "123456789012"
```

查看当前相机序列号：

```bash
object-locator --list-devices
```

调试输出也在 `config.yaml` 的 `output` 里：

```yaml
output:
  json: false
  result_json: "runs/results/{run_id}.json"
  history_dir: "runs/history"
  debug_image: "runs/latest_panel.jpg"
  debug_rgb_image: "runs/latest_rgb.jpg"
  debug_depth_image: "runs/latest_depth.jpg"
  vlm_response: "runs/latest_vlm_response.json"
  save_depth: null
```

每次运行后会保存三张最新检查图，同时在 `runs/history/<run_id>/` 下保留一份不会覆盖的归档：

- `result_json`: 最终 JSON 结果路径。支持 `{run_id}` 模板，默认每次生成新文件。
- `debug_rgb_image`: RGB 图，画 detector 返回的 bbox。
- `debug_depth_image`: 对齐后的深度伪彩色图，画同一个 bbox。
- `debug_image`: 左右拼接图，左边 RGB，右边 depth。
- `vlm_response`: OpenRouter 原始文本响应，用来排查 JSON 解析失败。
- `history_dir`: 每次运行的历史目录，包含 `result.json`、`rgb.jpg`、`depth.jpg`、`panel.jpg` 和 `detector_trace.json`。

做 z 轴重复性实验时，可以移动桌面上的目标后反复运行：

```bash
object-locator --config config.yaml --json
```

每次输出都会有不同的 `run_id`，归档路径类似：

```text
runs/results/20260626_153012_123456.json
runs/history/20260626_153012_123456/result.json
runs/history/20260626_153012_123456/panel.jpg
```

`runs/latest_*.jpg` 仍然会被覆盖，方便快速查看最新结果；`runs/results/<run_id>.json` 和 `runs/history/<run_id>/` 不会覆盖。要比较 z 是否稳定，重点看每次对应的 `result.json` 里的：

```json
{
  "run_id": "...",
  "position": {
    "z_m": 0.682,
    "sample_count": 87,
    "valid_fraction": 0.24,
    "strategy": "median"
  }
}
```

绿色框是 detector bbox，橙色半透明区域是 SAM mask，黄色细框是实际用于取深度的内部区域，白色十字是最终用于反投影的像素点。紫色箭头表示 `tail_px -> head_px` 指向，黄色点是 head，紫色点是 tail。

## 运行

```bash
object-locator --config config.yaml
```

或者不用安装入口，直接跑模块：

```bash
python -m object_locator.cli --config config.yaml
```

命令行参数可以临时覆盖配置：

```bash
object-locator \
  --target "banana" \
  --detector vlm \
  --model "google/gemini-2.5-flash" \
  --serial-number "123456789012" \
  --depth-strategy median \
  --inner-ratio 0.7 \
  --max-depth 4.0 \
  --output runs/banana_panel.jpg \
  --output-rgb runs/banana_rgb.jpg \
  --output-depth runs/banana_depth.jpg \
  --save-depth runs/banana_depth.npy \
  --json
```

红色方块也可以强制不用 VLM：

```bash
object-locator --config config.yaml --detector color --color red
```

`--depth-strategy` 有三种：

- `median`: 默认，在 bbox 中心区域取深度中位数，通常最稳。
- `foreground`: 偏向使用 bbox 内较近的一簇深度，适合物体前景明显、背景占比较大的情况。
- `center`: 优先使用 bbox 中心点深度，中心点无效时退回中位数。

小目标上 D435i 深度经常是稀疏的。`min_samples` 是理想采样数，`fallback_min_samples` 是最低可接受采样数；如果 tight bbox 深度点太少，程序会按 `max_expand_ratio` 和 `expand_steps` 在 bbox 周围搜索有效深度。若输出里的 strategy 变成 `median_sparse`，说明这次用了少量深度点，结果可用但置信度要手动看调试图。

如果某个模型不支持 `response_format: json_schema`，加：

```bash
object-locator --target "red cup" --no-json-schema
```

如果遇到 `could not find complete JSON in VLM response`，通常是模型返回了半截 JSON 或当前 provider 对结构化输出支持不稳。默认会自动关闭 `json_schema` 重试一次，并把原始响应写入 `runs/latest_vlm_response.json`。仍然失败时可以切换模型：

```yaml
openrouter:
  model: "google/gemini-2.5-flash"
  use_json_schema: true
  max_tokens: 1024
```

也可以直接关掉结构化输出参数：

```yaml
openrouter:
  use_json_schema: false
```

如果你想强制 OpenRouter 只路由到支持结构化输出参数的 provider：

```bash
object-locator --target "red cup" --require-parameters
```

## 输出示例

```json
{
  "target": "leftmost sample bottle",
  "found": true,
  "detection": {
    "source": "grounded_sam",
    "label": "sample bottle",
    "bbox": {
      "x_min": 120.5,
      "y_min": 300.2,
      "x_max": 210.8,
      "y_max": 390.1
    },
    "head_px": {"x": 180.2, "y": 320.5},
    "tail_px": {"x": 130.1, "y": 370.2}
  },
  "position_anchor": "tail",
  "position": {
    "x_m": -0.123,
    "y_m": 0.045,
    "z_m": 0.682,
    "strategy": "median",
    "sample_count": 87
  },
  "position_base": {
    "available": false,
    "enabled": false,
    "reason": "calibration.enabled is false"
  },
  "coordinate_frame": {
    "x": "right",
    "y": "down",
    "z": "forward from camera"
  }
}
```

## 精度和局限

这套方案适合机器人操作前的快速感知和中等精度定位，但不是严格实时/高精度测量系统。主要误差来源包括：

- RGB 检测误差：Grounding DINO/SAM 或 VLM 可能把 bbox、mask、keypoint 放偏，复杂遮挡时更明显。
- 深度误差：D435i 对透明、反光、黑色吸光材质不稳定，透明瓶身尾端尤其容易没有有效 depth。
- 采样误差：`position_anchor` 附近如果只有少量有效点，输出会变成 `median_sparse`，需要手动看调试图确认。
- 外参误差：启用 `position_base` 后，还会叠加 `base_T_camera`、`flange_T_camera`、`base_T_flange` 的标定误差。
- 时间同步误差：腕部相机必须使用拍照时刻的 `base_T_flange`，机器人运动时不同步会导致 base 坐标偏移。

经验上，不透明且有纹理的目标会比透明/反光目标稳定很多。样本瓶尾端这种透明部位，RGB 关键点通常可见，但 3D 坐标是否可靠取决于 RealSense 是否在该区域量到深度；必要时应引入桌面平面约束、已知试管几何或机器人二次观测来补偿。

## 测试

几何计算不依赖相机，可以直接测：

```bash
PYTHONPATH=src python -m unittest discover
```
