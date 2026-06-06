"""
vision/scene_analyzer.py — Multimodal Scene Understanding
===========================================================
Combines two analysis backends:
  1. Florence-2 (Microsoft) — local transformer, structured scene understanding
     (object detection, caption, OCR, depth hints)
  2. Ollama LLaVA — local multimodal LLM for natural-language scene description

Strategy:
  • YOLO detections first (fast, always on)
  • Florence-2 for structured captions (medium latency)
  • LLaVA only on explicit user queries or significant scene changes
    (slow, conserves Pi resources)

Provides:
  • async analyse(frame, query)     → SceneAnalysis
  • async describe_scene(frame)     → str
  • async initialise() / cleanup()
"""

import asyncio
import base64
import io
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SceneAnalysis:
    """Structured output of scene understanding."""
    caption: str = ""
    objects: list[str] = field(default_factory=list)
    hazards: list[str] = field(default_factory=list)
    navigation_hint: str = ""
    ocr_text: str = ""
    raw_llm_response: str = ""
    timestamp: float = field(default_factory=time.time)
    source: str = "yolo"   # "yolo" | "florence2" | "ollama"


class SceneAnalyzer:
    """
    Multimodal scene understanding engine.
    Tries Florence-2 first; falls back gracefully to Ollama LLaVA or YOLO-only.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._florence = None        # Florence-2 model
        self._florence_processor = None
        self._ollama_client = None
        self._last_frame_hash = ""
        self._last_analysis_time = 0.0
        self._analysis_cooldown = 4.0   # seconds between auto-analyses
        self._using_florence = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        if not self.cfg.system.ENABLE_OLLAMA_VISION:
            log.info("Scene analyzer: Ollama vision disabled by config.")
            return

        loop = asyncio.get_event_loop()

        # Try Florence-2 first (requires numpy <2.0 and compatible transformers)
        try:
            # Quick numpy compatibility pre-check before expensive model load
            import numpy as np
            np_ver = tuple(int(x) for x in np.__version__.split(".")[:2])
            if np_ver >= (2, 0):
                log.warning(
                    f"Florence-2 requires numpy <2.0 (installed: {np.__version__}). "
                    "Skipping — using Ollama LLaVA. Fix: pip install 'numpy<2.0'"
                )
            else:
                await loop.run_in_executor(None, self._load_florence)
                self._using_florence = True
                log.info("Florence-2 loaded for scene understanding.")
        except Exception as e:
            log.warning(f"Florence-2 not available ({e}) — using Ollama LLaVA.")

        # Set up Ollama client (used for LLaVA and as Florence fallback)
        try:
            import ollama
            self._ollama_client = ollama
            log.info(f"Ollama client ready (vision model: {self.cfg.ai.OLLAMA_VISION_MODEL})")
        except ImportError:
            log.warning("Ollama Python library not installed.")

    def _load_florence(self):
        """Load Florence-2 model and processor (blocking)."""
        # Workaround: pre-import torchvision fully to avoid circular import on Python 3.10
        try:
            import torchvision.transforms as _tv_t
            from torchvision.transforms import InterpolationMode  # noqa: F401
        except Exception:
            pass

        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch

        model_name = "microsoft/Florence-2-base-ft"
        self._florence_processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._florence = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        self._florence.eval()

    async def cleanup(self):
        self._florence = None
        self._florence_processor = None

    # ── Analysis ──────────────────────────────────────────────────────────────
    async def analyse(
        self,
        frame: np.ndarray,
        query: Optional[str] = None,
        detections: Optional[list] = None,
    ) -> SceneAnalysis:
        """
        Analyse a camera frame.
        If query is provided → always run full LLM analysis.
        Otherwise → rate-limited auto-analysis.
        """
        if query:
            return await self._deep_analyse(frame, query, detections)

        # Rate limit auto-analysis
        now = time.time()
        if now - self._last_analysis_time < self._analysis_cooldown:
            # Return lightweight detection-based summary
            return self._detections_to_analysis(detections or [])

        self._last_analysis_time = now
        return await self._deep_analyse(frame, None, detections)

    async def _deep_analyse(
        self,
        frame: np.ndarray,
        query: Optional[str],
        detections: Optional[list],
    ) -> SceneAnalysis:
        """Full LLM-based scene analysis."""
        if self._using_florence and self._florence is not None:
            try:
                result = await self._florence_analyse(frame, query)
                if result:
                    return result
            except Exception as e:
                log.warning(f"Florence-2 error ({e}) — trying Ollama.")

        if self._ollama_client:
            try:
                return await self._ollama_analyse(frame, query, detections)
            except Exception as e:
                log.warning(f"Ollama vision error: {e}")

        # Pure YOLO fallback
        return self._detections_to_analysis(detections or [])

    async def _florence_analyse(
        self, frame: np.ndarray, query: Optional[str]
    ) -> Optional[SceneAnalysis]:
        """Run Florence-2 inference in executor."""
        import torch

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        loop = asyncio.get_event_loop()

        def _infer():
            task = "<DETAILED_CAPTION>" if not query else "<CAPTION_TO_PHRASE_GROUNDING>"
            inputs = self._florence_processor(
                text=task, images=pil_img, return_tensors="pt"
            )
            with torch.no_grad():
                ids = self._florence.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=200,
                    do_sample=False,
                )
            return self._florence_processor.batch_decode(ids, skip_special_tokens=True)[0]

        raw = await loop.run_in_executor(None, _infer)
        return SceneAnalysis(
            caption=raw.strip(),
            source="florence2",
            navigation_hint=self._extract_nav_hint(raw),
        )

    async def _ollama_analyse(
        self,
        frame: np.ndarray,
        query: Optional[str],
        detections: Optional[list],
    ) -> SceneAnalysis:
        """Run Ollama LLaVA vision model for scene description."""
        # Encode frame as JPEG
        _, jpg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        img_b64 = base64.b64encode(jpg_buf.tobytes()).decode()

        det_list = ", ".join(d.class_name for d in (detections or []))
        if query:
            prompt = (
                f"A visually impaired person is asking: '{query}'. "
                f"Describe what you see in 2-3 short, clear sentences. "
                f"Focus on navigation safety. "
                f"Known nearby objects: {det_list or 'none'}."
            )
        else:
            prompt = (
                "Describe this scene for a blind person in 2-3 short, clear sentences. "
                "Mention: obstacles, people, vehicles, signage, or hazards. "
                f"Objects already detected: {det_list or 'none'}. "
                "Be concise and navigation-focused."
            )

        ai = self.cfg.ai
        loop = asyncio.get_event_loop()

        def _call_ollama():
            return self._ollama_client.chat(
                model=ai.OLLAMA_VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }],
                options={"num_predict": self.cfg.system.RESPONSE_MAX_TOKENS},
            )

        response = await loop.run_in_executor(None, _call_ollama)
        text = response["message"]["content"].strip()

        return SceneAnalysis(
            caption=text,
            raw_llm_response=text,
            navigation_hint=self._extract_nav_hint(text),
            source="ollama",
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _detections_to_analysis(detections: list) -> SceneAnalysis:
        """Build a lightweight SceneAnalysis from YOLO detections only."""
        if not detections:
            return SceneAnalysis(caption="Path appears clear.", source="yolo")

        names = [d.class_name for d in detections]
        hazards = [n for n in names if n in {"person", "car", "truck", "bus", "motorcycle"}]
        caption = f"Detected: {', '.join(set(names))}."
        hint = f"Caution, {', '.join(set(hazards))} ahead." if hazards else "Path appears clear."
        return SceneAnalysis(
            caption=caption,
            objects=names,
            hazards=hazards,
            navigation_hint=hint,
            source="yolo",
        )

    @staticmethod
    def _extract_nav_hint(text: str) -> str:
        """Heuristically extract navigation instruction from free text."""
        nav_keywords = ["walk", "turn", "stop", "avoid", "cross", "step", "move", "go"]
        for sentence in text.split("."):
            if any(kw in sentence.lower() for kw in nav_keywords):
                return sentence.strip() + "."
        return ""

    async def describe_scene(self, frame: np.ndarray) -> str:
        """Convenience method: return scene caption string."""
        result = await self.analyse(frame)
        return result.caption

    def is_ready(self) -> bool:
        return True  # always ready (falls back gracefully)
