"""
vision/currency_detector.py — Currency Note Identification
===========================================================
Detects and identifies currency notes using:
  1. Template matching (fast, offline)
  2. Ollama LLaVA vision fallback (for unknown/worn notes)

Supports: INR (Indian Rupee) denominations by default.
Easily extensible to USD, EUR, etc. via config.

Provides:
  • async detect(frame)   → CurrencyResult | None
  • async initialise()
"""

import asyncio
import base64
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)

# INR denomination descriptions for LLM prompt
_INR_DENOMINATIONS = ["10", "20", "50", "100", "200", "500", "2000"]


@dataclass
class CurrencyResult:
    currency: str      # e.g. "INR", "USD"
    denomination: str  # e.g. "500", "20"
    confidence: float
    method: str        # "template" | "llm"

    def __str__(self):
        return (
            f"{self.denomination} {self.currency} note "
            f"({self.confidence:.0%} confidence via {self.method})"
        )


class CurrencyDetector:
    """
    Currency note detector combining template matching and LLM vision.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._templates: dict[str, np.ndarray] = {}
        self._ollama_client = None
        self._last_detect_time = 0.0

    async def initialise(self):
        if not self.cfg.system.ENABLE_CURRENCY_DETECTION:
            return

        # Load templates from data/currency_templates/
        await self._load_templates()

        # Ollama client as fallback
        try:
            import ollama
            self._ollama_client = ollama
        except ImportError:
            log.warning("Ollama not available for currency detection fallback.")

        log.info(
            f"Currency detector ready. Templates: {len(self._templates)}, "
            f"Country: {self.cfg.ai.CURRENCY_COUNTRY}"
        )

    async def _load_templates(self):
        """Load reference currency images from data/currency_templates/."""
        from pathlib import Path
        template_dir = self.cfg.system.DATA_DIR / "currency_templates"
        template_dir.mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_event_loop()

        def _load():
            loaded = {}
            for img_path in template_dir.glob("*.jpg"):
                denom = img_path.stem  # e.g. "100", "500"
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    loaded[denom] = img
                    log.debug(f"Loaded currency template: {denom}")
            return loaded

        self._templates = await loop.run_in_executor(None, _load)

    async def detect(self, frame: np.ndarray) -> Optional[CurrencyResult]:
        """
        Detect currency note in frame.
        Returns CurrencyResult or None if no note found.
        Rate-limited to 1 detection per 2 seconds.
        """
        now = time.time()
        if now - self._last_detect_time < 2.0:
            return None
        self._last_detect_time = now

        # Try template matching first (fast)
        result = await self._template_match(frame)
        if result and result.confidence > 0.7:
            return result

        # Fallback: LLM vision
        if self._ollama_client:
            return await self._llm_detect(frame)

        return result   # return low-confidence template result if any

    async def _template_match(self, frame: np.ndarray) -> Optional[CurrencyResult]:
        """Template-matching based currency detection."""
        if not self._templates:
            return None

        loop = asyncio.get_event_loop()

        def _match():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            best_denom = None
            best_score = 0.0

            for denom, template in self._templates.items():
                # Scale template to ~60% of frame width
                h, w = template.shape
                scale = min(frame.shape[1] * 0.6 / w, 1.0)
                scaled = cv2.resize(template, (int(w * scale), int(h * scale)))

                if scaled.shape[0] > gray.shape[0] or scaled.shape[1] > gray.shape[1]:
                    continue

                result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    best_denom = denom

            return best_denom, best_score

        denom, score = await loop.run_in_executor(None, _match)
        if denom:
            return CurrencyResult(
                currency=self.cfg.ai.CURRENCY_COUNTRY,
                denomination=denom,
                confidence=score,
                method="template",
            )
        return None

    async def _llm_detect(self, frame: np.ndarray) -> Optional[CurrencyResult]:
        """Use Ollama LLaVA to identify currency note."""
        _, jpg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_b64 = base64.b64encode(jpg_buf.tobytes()).decode()

        country = self.cfg.ai.CURRENCY_COUNTRY
        denoms = ", ".join(_INR_DENOMINATIONS if country == "INR" else ["unknown"])
        prompt = (
            f"Is there a currency note in this image? "
            f"If yes, what is the denomination? "
            f"Possible {country} denominations: {denoms}. "
            "Reply with ONLY the denomination number, or 'none' if no note visible."
        )

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._ollama_client.chat(
                    model=self.cfg.ai.OLLAMA_VISION_MODEL,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }],
                    options={"num_predict": 20},
                )
            )
            text = response["message"]["content"].strip()
            if text.lower() == "none" or not text:
                return None

            # Extract denomination number from response
            import re
            match = re.search(r"\d+", text)
            denom = match.group() if match else text[:10]

            return CurrencyResult(
                currency=country,
                denomination=denom,
                confidence=0.8,
                method="llm",
            )
        except Exception as e:
            log.warning(f"LLM currency detection error: {e}")
            return None

    def is_ready(self) -> bool:
        return True
