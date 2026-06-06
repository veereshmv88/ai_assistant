"""
ocr/text_reader.py — OCR Text Extraction Engine
=================================================
Reads text from images using:
  Primary:  EasyOCR — multilingual, no internet needed
  Fallback: pytesseract — simpler but less accurate

Pre-processing pipeline:
  1. Resize to optimal OCR resolution
  2. Grayscale conversion
  3. Adaptive threshold / CLAHE contrast enhancement
  4. Deskew (correct tilted text)

Provides:
  • async read_frame(frame)          → str  (full OCR text)
  • async read_region(frame, bbox)   → str  (crop + OCR)
  • async initialise() / cleanup()
"""

import asyncio
import re
from typing import Optional

import cv2
import numpy as np

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


class TextReader:
    """
    EasyOCR-based text reader with image pre-processing and pytesseract fallback.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._reader = None
        self._initialised = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        if self._initialised:
            return
        loop = asyncio.get_event_loop()
        try:
            self._reader = await loop.run_in_executor(
                None, self._load_easyocr
            )
            log.info(f"EasyOCR loaded. Languages: {self.cfg.ai.OCR_LANGUAGES}")
        except Exception as e:
            log.warning(f"EasyOCR failed ({e}) — trying pytesseract.")
            self._reader = None
        self._initialised = True

    def _load_easyocr(self):
        import easyocr
        return easyocr.Reader(
            self.cfg.ai.OCR_LANGUAGES,
            gpu=self.cfg.ai.OCR_GPU,
            verbose=False,
        )

    async def cleanup(self):
        self._reader = None
        self._initialised = False

    # ── Public API ────────────────────────────────────────────────────────────
    async def read_frame(self, frame: np.ndarray) -> str:
        """Extract all text from a camera frame."""
        processed = self._preprocess(frame)
        return await self._ocr(processed)

    async def read_region(self, frame: np.ndarray, bbox: tuple) -> str:
        """Extract text from a specific region (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        region = frame[y1:y2, x1:x2]
        processed = self._preprocess(region)
        return await self._ocr(processed)

    # ── Pre-processing ────────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(frame: np.ndarray) -> np.ndarray:
        """
        Image pre-processing pipeline for best OCR accuracy.
        """
        # 1. Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # 2. Resize: ensure min 300 DPI equivalent (min width 1024px)
        h, w = gray.shape
        if w < 800:
            scale = 800 / w
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)

        # 3. CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 4. Adaptive threshold for clean binary image
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )

        # 5. Deskew
        deskewed = TextReader._deskew(binary)

        return deskewed

    @staticmethod
    def _deskew(image: np.ndarray) -> np.ndarray:
        """Correct image skew using moments-based method."""
        try:
            coords = np.column_stack(np.where(image > 0))
            if len(coords) < 10:
                return image
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            (h, w) = image.shape
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            return cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except Exception:
            return image

    # ── OCR ───────────────────────────────────────────────────────────────────
    async def _ocr(self, image: np.ndarray) -> str:
        """Run OCR on pre-processed image."""
        if self._reader:
            return await self._easyocr_read(image)
        return await self._tesseract_read(image)

    async def _easyocr_read(self, image: np.ndarray) -> str:
        """EasyOCR inference in executor."""
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: self._reader.readtext(
                    image,
                    detail=1,
                    paragraph=True,
                )
            )
            texts = [
                text
                for (_, text, conf) in results
                if conf >= self.cfg.ai.OCR_MIN_CONFIDENCE
            ]
            combined = " ".join(texts).strip()
            log.debug(f"OCR result: '{combined[:80]}'")
            return self._clean_text(combined)
        except Exception as e:
            log.error(f"EasyOCR error: {e}")
            return ""

    async def _tesseract_read(self, image: np.ndarray) -> str:
        """pytesseract fallback OCR."""
        loop = asyncio.get_event_loop()
        try:
            import pytesseract
            text = await loop.run_in_executor(
                None,
                lambda: pytesseract.image_to_string(image, lang="eng")
            )
            return self._clean_text(text)
        except Exception as e:
            log.warning(f"pytesseract error: {e}")
            return ""

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove noise characters from OCR output."""
        # Remove non-printable characters except newlines
        cleaned = re.sub(r"[^\x20-\x7E\n]", "", text)
        # Collapse multiple spaces/newlines
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

    def is_ready(self) -> bool:
        return self._initialised
