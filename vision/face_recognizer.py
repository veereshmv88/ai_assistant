"""
vision/face_recognizer.py — Face Recognition
=============================================
Identifies known people from a library of enrolled face images.
Uses the `face_recognition` library (dlib-based, CPU-efficient).

Enrollment:
  • Place photos named "PersonName.jpg" in config/known_faces/
  • Or use voice command "Remember this person as [Name]" at runtime

Provides:
  • async process_frame(frame)         → list[FaceResult]
  • async enroll_face(name, frame)     → bool
  • async initialise() / cleanup()
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class FaceResult:
    name: str
    confidence: float
    x1: int; y1: int; x2: int; y2: int
    is_known: bool = False
    last_seen: float = field(default_factory=time.time)

    def __str__(self):
        tag = "known" if self.is_known else "unknown"
        return f"Face({self.name}, {self.confidence:.0%}, {tag})"


class FaceRecognizer:
    """
    Async face recognition using face_recognition + dlib.
    Runs inference in executor thread. Caches embeddings to avoid re-loading.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._known_encodings: list[np.ndarray] = []
        self._known_names: list[str] = []
        self._registry_path: Path = cfg.ai.FACE_KNOWN_DIR / "registry.json"
        self._last_results: list[FaceResult] = []
        self._last_inference = 0.0
        self._face_rec = None   # the face_recognition module

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        if not self.cfg.system.ENABLE_FACE_RECOGNITION:
            log.info("Face recognition: disabled.")
            return

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, self._load_library_and_faces)
            log.info(f"Face recognition ready. Known faces: {len(self._known_names)}")
        except ImportError:
            log.warning("face_recognition not installed — face recognition disabled.")
        except Exception as e:
            log.warning(f"Face recognition init error: {e}")

    def _load_library_and_faces(self):
        """Blocking: import library and load all known face encodings."""
        import face_recognition
        self._face_rec = face_recognition
        self._load_known_faces()

    def _load_known_faces(self):
        """Load face encodings from known_faces directory."""
        known_dir = self.cfg.ai.FACE_KNOWN_DIR
        known_dir.mkdir(parents=True, exist_ok=True)

        self._known_encodings.clear()
        self._known_names.clear()

        fr = self._face_rec
        for img_path in known_dir.glob("*.jpg"):
            name = img_path.stem.replace("_", " ").title()
            try:
                img = fr.load_image_file(str(img_path))
                encs = fr.face_encodings(img)
                if encs:
                    self._known_encodings.append(encs[0])
                    self._known_names.append(name)
                    log.debug(f"Enrolled face: {name}")
            except Exception as e:
                log.warning(f"Failed to load face {img_path}: {e}")

    async def cleanup(self):
        self._known_encodings.clear()
        self._known_names.clear()

    # ── Recognition ───────────────────────────────────────────────────────────
    def _recognise_sync(self, frame: np.ndarray) -> list[FaceResult]:
        if self._face_rec is None:
            return []

        fr = self._face_rec
        cfg = self.cfg.ai

        # Resize to speed up face location
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = fr.face_locations(rgb_small, model=cfg.FACE_MODEL)
        if not locations:
            return []

        encodings = fr.face_encodings(rgb_small, locations)
        results = []

        for enc, (top, right, bottom, left) in zip(encodings, locations):
            name = "Unknown"
            confidence = 0.0
            is_known = False

            if self._known_encodings:
                distances = fr.face_distance(self._known_encodings, enc)
                best_idx = int(np.argmin(distances))
                best_dist = distances[best_idx]
                if best_dist <= cfg.FACE_TOLERANCE:
                    name = self._known_names[best_idx]
                    confidence = 1.0 - float(best_dist)
                    is_known = True

            # Scale coordinates back to original frame size
            results.append(FaceResult(
                name=name,
                confidence=confidence,
                x1=left * 2, y1=top * 2,
                x2=right * 2, y2=bottom * 2,
                is_known=is_known,
            ))

        return results

    async def process_frame(self, frame: np.ndarray) -> list[FaceResult]:
        """Recognise faces in frame, rate-limited to 2 FPS."""
        if not self.cfg.system.ENABLE_FACE_RECOGNITION or self._face_rec is None:
            return []

        now = time.perf_counter()
        if now - self._last_inference < 0.5:   # max 2 FPS
            return self._last_results

        self._last_inference = now
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, lambda: self._recognise_sync(frame)
        )
        self._last_results = results

        for r in results:
            log.debug(str(r))

        return results

    # ── Enrollment ────────────────────────────────────────────────────────────
    async def enroll_face(self, name: str, frame: np.ndarray) -> bool:
        """
        Enroll a new face from the current frame.
        Saves image to known_faces/ and reloads embeddings.
        """
        if self._face_rec is None:
            return False

        fr = self._face_rec
        loop = asyncio.get_event_loop()

        def _save_and_enroll():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locations = fr.face_locations(rgb)
            if not locations:
                return False
            # Save the largest face
            top, right, bottom, left = sorted(locations, key=lambda l: (l[2]-l[0])*(l[1]-l[3]))[0]
            safe_name = name.strip().replace(" ", "_").lower()
            save_path = self.cfg.ai.FACE_KNOWN_DIR / f"{safe_name}.jpg"
            face_img = frame[top:bottom, left:right]
            cv2.imwrite(str(save_path), face_img)
            self._load_known_faces()
            return True

        success = await loop.run_in_executor(None, _save_and_enroll)
        if success:
            log.info(f"Enrolled new face: {name}")
        return success

    def is_ready(self) -> bool:
        return self._face_rec is not None or not self.cfg.system.ENABLE_FACE_RECOGNITION
