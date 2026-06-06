"""
sensors/camera.py — Camera Module
==================================
Supports:
  • Raspberry Pi Camera Module 3 (via picamera2)
  • USB webcam (via OpenCV)
  • Mock mode (returns static test frames for dev on PC)

Provides:
  • async stream() → AsyncIterator[np.ndarray]  — continuous frame generator
  • async get_frame() → np.ndarray              — single-shot capture
  • async initialise() / cleanup()
"""

import asyncio
import time
import numpy as np
import cv2
from pathlib import Path
from typing import AsyncIterator, Optional

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)

# Mock test image path (used in dev mode)
_MOCK_IMAGE_PATH = Path(__file__).parent.parent / "data" / "mock_frame.jpg"


class CameraModule:
    """
    Thread-safe camera abstraction.
    Runs the blocking capture in an executor thread to keep asyncio clean.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._picam2 = None
        self._lock = asyncio.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        """Open camera device."""
        hw = self.cfg.hardware
        sys_cfg = self.cfg.system

        if sys_cfg.MOCK_CAMERA:
            log.info("Camera: MOCK mode — using static test image.")
            self._latest_frame = self._load_mock_frame()
            self._running = True
            return

        loop = asyncio.get_event_loop()
        if hw.USE_PICAMERA2:
            await loop.run_in_executor(None, self._init_picamera2)
        else:
            await loop.run_in_executor(None, self._init_opencv)

        self._running = True
        log.info(f"Camera initialised ({hw.CAMERA_WIDTH}×{hw.CAMERA_HEIGHT} @ {hw.CAMERA_FPS}fps)")

    def _init_picamera2(self):
        """Initialise Pi Camera Module 3 via picamera2."""
        try:
            from picamera2 import Picamera2
            hw = self.cfg.hardware
            self._picam2 = Picamera2()
            config = self._picam2.create_preview_configuration(
                main={"size": (hw.CAMERA_WIDTH, hw.CAMERA_HEIGHT), "format": "RGB888"}
            )
            self._picam2.configure(config)
            self._picam2.set_controls({"FrameRate": hw.CAMERA_FPS})
            self._picam2.start()
            log.info("picamera2 started.")
        except ImportError:
            log.warning("picamera2 not available, falling back to OpenCV.")
            self._init_opencv()

    def _init_opencv(self):
        """Initialise USB webcam via OpenCV VideoCapture."""
        hw = self.cfg.hardware
        self._cap = cv2.VideoCapture(hw.CAMERA_DEVICE_INDEX)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, hw.CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hw.CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, hw.CAMERA_FPS)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {hw.CAMERA_DEVICE_INDEX}")
        log.info(f"OpenCV VideoCapture opened (device {hw.CAMERA_DEVICE_INDEX}).")

    async def cleanup(self):
        """Release camera resources."""
        self._running = False
        loop = asyncio.get_event_loop()
        if self._picam2:
            await loop.run_in_executor(None, self._picam2.stop)
            self._picam2 = None
        if self._cap:
            await loop.run_in_executor(None, self._cap.release)
            self._cap = None
        log.info("Camera released.")

    # ── Frame capture ─────────────────────────────────────────────────────────
    def _capture_frame_sync(self) -> Optional[np.ndarray]:
        """Blocking frame capture — called from executor thread."""
        if self._picam2:
            frame_rgb = self._picam2.capture_array()
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        elif self._cap:
            ret, frame = self._cap.read()
            return frame if ret else None
        return None

    async def get_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame asynchronously."""
        if self.cfg.system.MOCK_CAMERA:
            return self._load_mock_frame()
        loop = asyncio.get_event_loop()
        async with self._lock:
            frame = await loop.run_in_executor(None, self._capture_frame_sync)
        return frame

    async def stream(self) -> AsyncIterator[np.ndarray]:
        """
        Async generator yielding frames at the configured FPS.
        Runs the blocking capture in an executor thread.
        """
        target_interval = 1.0 / self.cfg.hardware.CAMERA_FPS

        while self._running:
            t0 = time.perf_counter()
            frame = await self.get_frame()
            if frame is not None:
                self._latest_frame = frame
                yield frame
            elapsed = time.perf_counter() - t0
            sleep_time = max(0.0, target_interval - elapsed)
            await asyncio.sleep(sleep_time)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recent captured frame (may be None before first capture)."""
        return self._latest_frame

    @staticmethod
    def _load_mock_frame() -> np.ndarray:
        """Return a test frame for mock mode."""
        if _MOCK_IMAGE_PATH.exists():
            return cv2.imread(str(_MOCK_IMAGE_PATH))
        # Generate a blank grey frame with text if no test image found
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        cv2.putText(
            frame, "MOCK CAMERA FRAME", (120, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2
        )
        return frame

    def is_ready(self) -> bool:
        return self._running
