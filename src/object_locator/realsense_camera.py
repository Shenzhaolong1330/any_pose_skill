from __future__ import annotations

from dataclasses import dataclass
import time

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
        reset_on_start: bool = False,
        reset_wait_s: float = 5.0,
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
        self.reset_on_start = reset_on_start
        self.reset_wait_s = reset_wait_s
        self._make_pipeline_objects()
        self.profile = None
        self.depth_scale = None
        self.started = False

    def _make_pipeline_objects(self) -> None:
        rs = self.rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = rs.align(rs.stream.color)

    def __enter__(self) -> "RealSenseCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self.started:
            return
        rs = self.rs
        if self.reset_on_start:
            self._hardware_reset()
            self._make_pipeline_objects()
        if self.serial_number:
            self.config.enable_device(self.serial_number)
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
        )
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
        )
        self.profile = self.pipeline.start(self.config)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        self.started = True

    def stop(self) -> None:
        if self.started:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            finally:
                self.started = False

    def warmup(self, frames: int = 30, *, timeout_ms: int = 15000, retries: int = 3) -> None:
        for _ in range(max(0, frames)):
            self._wait_for_frames(timeout_ms=timeout_ms, retries=retries)

    def capture(
        self,
        *,
        warmup_frames: int = 0,
        timeout_ms: int = 15000,
        retries: int = 3,
    ) -> FrameBundle:
        if not self.started:
            self.start()
        if warmup_frames:
            self.warmup(warmup_frames, timeout_ms=timeout_ms, retries=retries)

        frames = self._wait_for_frames(timeout_ms=timeout_ms, retries=retries)
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

    def _wait_for_frames(self, *, timeout_ms: int, retries: int):
        timeout_ms = max(1000, int(timeout_ms))
        retries = max(1, int(retries))
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                return self.pipeline.wait_for_frames(timeout_ms)
            except RuntimeError as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.25)
        raise RuntimeError(
            f"RealSense frame did not arrive within {timeout_ms} ms "
            f"after {retries} attempt(s): {last_error}"
        ) from last_error

    def _hardware_reset(self) -> None:
        device = self._find_reset_device()
        serial = _device_info(device, self.rs.camera_info.serial_number)
        try:
            device.hardware_reset()
        except RuntimeError as exc:
            target = f" serial_number={serial}" if serial else ""
            raise RuntimeError(f"failed to hardware-reset RealSense{target}: {exc}") from exc

        wait_s = max(0.0, float(self.reset_wait_s))
        if wait_s <= 0:
            return

        # D435i disappears from USB briefly after hardware_reset(). Give it time
        # to reconnect before creating a fresh pipeline.
        time.sleep(min(1.0, wait_s))
        if not serial:
            time.sleep(max(0.0, wait_s - 1.0))
            return

        deadline = time.monotonic() + max(0.0, wait_s - 1.0)
        while time.monotonic() < deadline:
            if self._serial_is_present(serial):
                return
            time.sleep(0.25)

    def _find_reset_device(self):
        devices = list(self.rs.context().query_devices())
        if not devices:
            raise RuntimeError("no RealSense device found to hardware-reset")
        if not self.serial_number:
            return devices[0]

        for device in devices:
            serial = _device_info(device, self.rs.camera_info.serial_number)
            if serial == self.serial_number:
                return device
        raise RuntimeError(
            f"RealSense serial_number={self.serial_number} was not found for hardware reset"
        )

    def _serial_is_present(self, serial_number: str) -> bool:
        try:
            devices = self.rs.context().query_devices()
        except RuntimeError:
            return False
        for device in devices:
            if _device_info(device, self.rs.camera_info.serial_number) == serial_number:
                return True
        return False
