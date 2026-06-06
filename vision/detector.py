"""
vision/detector.py — YOLO-World Object Detector
=================================================
Real-time object detection using ultralytics YOLO-World or YOLOv8.
Optimised for Raspberry Pi 5 with:
  • 320×320 input (fastest inference)
  • 8 FPS target (adjustable)
  • Thread-safe latest-result sharing

Provides:
  • async process_frame(frame)           — detect objects in frame
  • async get_latest_detections()        → list[Detection]
  • async initialise() / cleanup()
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# NOTE: ultralytics is imported lazily inside _load_model() to avoid a
# circular import in torchvision.transforms on Python 3.10.

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Detection:
    """A single detected object."""
    class_name: str
    confidence: float
    x1: int; y1: int; x2: int; y2: int  # bounding box (pixels)
    frame_width: int = 640
    frame_height: int = 480

    @property
    def centre(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def position_relative(self) -> str:
        """Return 'left', 'center', or 'right' depending on bounding box center."""
        cx = self.centre[0]
        left_bound = self.frame_width * 0.35
        right_bound = self.frame_width * 0.65
        if cx < left_bound:
            return "left"
        elif cx > right_bound:
            return "right"
        else:
            return "center"

    @property
    def estimated_distance_m(self) -> float:
        """Estimate distance using physical height heuristics of common objects."""
        h_frac = self.height / self.frame_height
        heights = {
            "person": 1.7,
            "car": 1.5,
            "truck": 2.5,
            "bus": 3.0,
            "bicycle": 1.0,
            "motorcycle": 1.0,
            "traffic light": 1.0,
            "stop sign": 0.8,
            "chair": 0.8,
            "door": 2.0,
            "stairs": 1.5,
            "dog": 0.6,
            "cat": 0.3,
        }
        ref_h = heights.get(self.class_name, 1.0)
        return round((ref_h * 1.5) / (h_frac + 0.01), 1)

    def is_in_path(self, frame_width: int, path_fraction: float = 0.4) -> bool:
        """True if detection centre is in the middle N% of the frame."""
        left  = frame_width * (0.5 - path_fraction / 2)
        right = frame_width * (0.5 + path_fraction / 2)
        cx = self.centre[0]
        return left <= cx <= right

    def __str__(self) -> str:
        return f"{self.class_name} ({self.confidence:.0%}) @ [{self.x1},{self.y1},{self.x2},{self.y2}], position={self.position_relative}, dist={self.estimated_distance_m}m"


class ObjectDetector:
    """
    YOLO-World / YOLOv8 real-time detector running in asyncio executor.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = None
        self._latest: list[Detection] = []
        self._lock = asyncio.Lock()
        self._frame_count = 0
        self._last_inference = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        ai = self.cfg.ai
        loop = asyncio.get_event_loop()

        try:
            self._model = await loop.run_in_executor(
                None, self._load_model
            )
            log.info(f"YOLO model loaded: {ai.YOLO_MODEL_PATH}")
        except (ImportError, AttributeError, Exception) as e:
            log.warning(f"YOLO load failed ({e}) — vision detection disabled.")
            self._model = None

    def _load_model(self):
        """Lazy import of ultralytics to avoid torchvision circular import on Python 3.10."""
        import os
        # Workaround: force-import torchvision.transforms fully before ultralytics
        try:
            import torchvision.transforms as _tv_transforms
            from torchvision.transforms import InterpolationMode  # noqa: F401
        except Exception:
            pass  # If torchvision isn't installed, continue — YOLO can still work

        from ultralytics import YOLO
        model_path = self.cfg.ai.YOLO_MODEL_PATH
        # If the specific model file doesn't exist, fall back to downloading yolov8n
        if not os.path.exists(model_path):
            log.warning(f"Model not found at {model_path}, downloading yolov8n…")
            model_path = "yolov8n.pt"
        return YOLO(model_path)

    async def cleanup(self):
        self._model = None

    # ── Detection ─────────────────────────────────────────────────────────────
    def _detect_sync(self, frame: np.ndarray) -> list[Detection]:
        """Blocking YOLO inference — call from executor thread."""
        if self._model is None:
            return []

        ai = self.cfg.ai
        input_size = ai.YOLO_INPUT_SIZE
        resized = cv2.resize(frame, (input_size, input_size))

        results = self._model.predict(
            resized,
            conf=ai.YOLO_CONFIDENCE,
            iou=ai.YOLO_IOU_THRESHOLD,
            verbose=False,
            device="cpu",
        )

        detections = []
        h, w = frame.shape[:2]
        scale_x = w / input_size
        scale_y = h / input_size

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                name   = result.names[cls_id]

                # Filter to classes of interest
                if name not in ai.YOLO_CLASSES_OF_INTEREST:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    class_name=name,
                    confidence=conf,
                    x1=int(x1 * scale_x), y1=int(y1 * scale_y),
                    x2=int(x2 * scale_x), y2=int(y2 * scale_y),
                    frame_width=w,
                    frame_height=h,
                ))

        return detections

    async def process_frame(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect objects in the given frame.
        Rate-limited to YOLO_FPS_TARGET — skips frames in between.
        """
        ai = self.cfg.ai
        now = time.perf_counter()
        min_interval = 1.0 / ai.YOLO_FPS_TARGET

        if now - self._last_inference < min_interval:
            return self._latest   # return cached result

        self._last_inference = now
        loop = asyncio.get_event_loop()
        detections = await loop.run_in_executor(
            None, lambda: self._detect_sync(frame)
        )

        async with self._lock:
            self._latest = detections

        if self.cfg.system.DEBUG and detections:
            log.debug(f"Detected: {[str(d) for d in detections]}")

        return detections

    async def get_latest_detections(self) -> list[Detection]:
        """Return the most recent detection results."""
        async with self._lock:
            return list(self._latest)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def get_hazardous(self, detections: list[Detection]) -> list[Detection]:
        """Filter detections to objects that pose navigation hazard."""
        hazardous = {
            "person", "car", "truck", "bus", "motorcycle", "bicycle",
            "dog", "fire hydrant", "stairs",
        }
        return [d for d in detections if d.class_name in hazardous]

    def get_in_path(self, detections: list[Detection], frame_width: int) -> list[Detection]:
        """Filter to objects directly in the user's walking path."""
        return [d for d in detections if d.is_in_path(frame_width)]

    def annotate_frame(self, frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        """Draw bounding boxes on frame (for debug display)."""
        annotated = frame.copy()
        for det in detections:
            cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
            label = f"{det.class_name} {det.confidence:.0%}"
            cv2.putText(annotated, label, (det.x1, det.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return annotated

    def is_ready(self) -> bool:
        return self._model is not None
