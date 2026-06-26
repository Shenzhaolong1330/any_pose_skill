from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CameraIntrinsics


@dataclass(frozen=True)
class FrameBundle:
    color_bgr: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp_ms: float


@dataclass(frozen=True)
class RealSenseDeviceInfo:
    name: str
    serial_number: str
    firmware_version: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "serial_number": self.serial_number,
            "firmware_version": self.firmware_version,
        }


def list_realsense_devices() -> list[RealSenseDeviceInfo]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError(
            "pyrealsense2 is not installed. Install dependencies with `pip install -e .`."
        ) from exc

    devices = []
    context = rs.context()
    for device in context.query_devices():
        name = _device_info(device, rs.camera_info.name) or "Unknown RealSense"
        serial = _device_info(device, rs.camera_info.serial_number) or ""
        firmware = _device_info(device, rs.camera_info.firmware_version)
        devices.append(
            RealSenseDeviceInfo(
                name=name,
                serial_number=serial,
                firmware_version=firmware,
            )
        )
    return devices


def _device_info(device, key) -> str | None:
    try:
        if device.supports(key):
            return str(device.get_info(key))
    except RuntimeError:
        return None
    return None


class RealSenseCamera:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial_number: str | None = None,
    ) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed. Install dependencies with `pip install -e .`."
            ) from exc

        self.rs = rs
        self.width = width
        self.height = height
        self.fps = fps
        self.serial_number = serial_number
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale = None
        self.started = False

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self.started:
            return
        rs = self.rs
        if self.serial_number:
            self.config.enable_device(self.serial_number)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        self.started = True

    def stop(self) -> None:
        if self.started:
            self.pipeline.stop()
            self.started = False

    def warmup(self, frames: int = 30) -> None:
        for _ in range(max(0, frames)):
            self.pipeline.wait_for_frames()

    def capture(self, *, warmup_frames: int = 0) -> FrameBundle:
        if not self.started:
            self.start()
        if warmup_frames:
            self.warmup(warmup_frames)

        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("failed to capture synchronized color/depth frames")

        depth_raw = np.asanyarray(depth_frame.get_data())
        color_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_m = depth_raw.astype(np.float32) * float(self.depth_scale)
        intrinsics = CameraIntrinsics.from_realsense(
            color_frame.profile.as_video_stream_profile().intrinsics
        )

        return FrameBundle(
            color_bgr=color_bgr,
            depth_m=depth_m,
            intrinsics=intrinsics,
            timestamp_ms=float(color_frame.get_timestamp()),
        )
