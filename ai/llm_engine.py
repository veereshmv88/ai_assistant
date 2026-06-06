"""
ai/llm_engine.py — Ollama Local LLM Reasoning Engine
======================================================
Wraps the Ollama Python client with:
  • Conversation context management (rolling window)
  • Two model pools: vision (LLaVA) and text-only (llama3.2:3b)
  • Structured prompt templates per intent type
  • Async/executor-based calls (non-blocking)
  • Automatic retry on transient failures

Provides:
  • async chat(prompt, use_vision, image_b64)   → str
  • async query(prompt)                          → str
  • async initialise() / cleanup()
"""

import asyncio
import base64
import time
from collections import deque
from typing import Optional

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


# ── System prompt defining assistant persona ──────────────────────────────────
_SYSTEM_PROMPT = """You are an AI assistant helping a visually impaired person navigate the world.
Always be:
- CONCISE: 1-3 sentences maximum per response
- CLEAR: use simple, direct language
- ACTIONABLE: give specific instructions ("turn left", "stop", "walk forward 5 steps")
- SAFE: always prioritise the user's safety above all else
Never say "I can see" or "in the image" — speak as if describing reality directly to the user."""


class LLMEngine:
    """
    Ollama LLM client with dual-model support and rolling conversation memory.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._ollama = None
        self._context: deque[dict] = deque(maxlen=cfg.ai.OLLAMA_CONTEXT_WINDOW * 2)
        self._available = False
        self._lock = asyncio.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        try:
            import ollama
            self._ollama = ollama
            # Test connection
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: ollama.list())
            self._available = True
            log.info(
                f"Ollama connected. Models: vision={self.cfg.ai.OLLAMA_VISION_MODEL}, "
                f"text={self.cfg.ai.OLLAMA_TEXT_MODEL}"
            )
        except ImportError:
            log.warning("ollama Python package not installed.")
        except Exception as e:
            log.warning(f"Ollama not reachable ({e}) — LLM disabled.")

    async def cleanup(self):
        self._context.clear()

    # ── Core API ──────────────────────────────────────────────────────────────
    async def chat(
        self,
        prompt: str,
        use_vision: bool = False,
        image_b64: Optional[str] = None,
    ) -> str:
        """
        Send a message and get an LLM response.
        Maintains conversation context across calls.
        """
        if not self._available:
            return self._fallback_response(prompt)

        ai = self.cfg.ai
        model = ai.OLLAMA_VISION_MODEL if use_vision else ai.OLLAMA_TEXT_MODEL

        user_msg: dict = {"role": "user", "content": prompt}
        if image_b64 and use_vision:
            user_msg["images"] = [image_b64]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *list(self._context),
            user_msg,
        ]

        async with self._lock:
            response = await self._call_ollama(model, messages)

        if response:
            # Store turn in context
            self._context.append(user_msg)
            self._context.append({"role": "assistant", "content": response})
            return response

        return self._fallback_response(prompt)

    async def query(self, prompt: str) -> str:
        """Single-shot query (no context). Faster — uses text-only model."""
        if not self._available:
            return self._fallback_response(prompt)

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return await self._call_ollama(self.cfg.ai.OLLAMA_TEXT_MODEL, messages) or ""

    async def vision_query(self, prompt: str, frame_bytes: bytes) -> str:
        """Query LLaVA with a camera frame."""
        img_b64 = base64.b64encode(frame_bytes).decode()
        return await self.chat(prompt, use_vision=True, image_b64=img_b64)

    # ── Internal ──────────────────────────────────────────────────────────────
    async def _call_ollama(
        self,
        model: str,
        messages: list[dict],
        retries: int = 2,
    ) -> Optional[str]:
        """Execute Ollama API call in executor with retry logic."""
        ai = self.cfg.ai
        loop = asyncio.get_event_loop()

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self._ollama.chat(
                            model=model,
                            messages=messages,
                            options={"num_predict": self.cfg.system.RESPONSE_MAX_TOKENS},
                        )
                    ),
                    timeout=ai.OLLAMA_TIMEOUT,
                )
                text = response["message"]["content"].strip()
                log.debug(f"LLM ({model}): '{text[:80]}'")
                return text
            except asyncio.TimeoutError:
                log.warning(f"Ollama timeout (attempt {attempt + 1}/{retries + 1})")
                if attempt == retries:
                    return None
                await asyncio.sleep(1.0)
            except Exception as e:
                log.error(f"Ollama error: {e}")
                return None

        return None

    def clear_context(self):
        """Clear conversation history (e.g., after SOS or topic change)."""
        self._context.clear()

    @staticmethod
    def _fallback_response(prompt: str) -> str:
        """Rule-based fallback when LLM is unavailable."""
        prompt_lower = prompt.lower()
        if any(w in prompt_lower for w in ["front", "ahead", "see"]):
            return "I cannot analyse the scene right now. Please use your cane for safety."
        if any(w in prompt_lower for w in ["cross", "road", "traffic"]):
            return "I cannot verify road safety. Please wait for assistance before crossing."
        if "help" in prompt_lower or "sos" in prompt_lower:
            return "Emergency mode activated. Calling for help."
        return "I am having trouble processing that request. Please try again."

    def is_ready(self) -> bool:
        return self._available
