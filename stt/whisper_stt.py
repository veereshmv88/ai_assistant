"""
stt/whisper_stt.py — Speech-to-Text Engine
============================================
Two-tier STT architecture:
  1. Vosk (streaming, low-latency) — hotword & short commands
  2. faster-whisper (accurate, quantized) — full sentence transcription

Flow:
  raw audio chunks → Vosk partial result (hotword detect)
                   → if hotword: buffer 3s → Whisper transcribe → return text

Provides:
  • async transcribe(audio_chunk) → Optional[str]
  • async initialise() / cleanup()
"""

import asyncio
import io
import wave
import time
import struct
from collections import deque
from typing import Optional

import numpy as np

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)

# How many audio chunks to collect before transcribing (~3 seconds at 16kHz/1024)
_WHISPER_BUFFER_CHUNKS = 47


class WhisperSTT:
    """
    Dual-engine STT: Vosk for fast hotword detection,
    faster-whisper for accurate full-sentence transcription.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._vosk_model = None
        self._vosk_rec   = None
        self._whisper    = None
        self._activated  = False      # True when hotword has been detected
        self._buffer: deque[bytes] = deque(maxlen=_WHISPER_BUFFER_CHUNKS)
        self._buffer_count = 0
        self._last_activity = 0.0
        self._lock = asyncio.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        ai = self.cfg.ai
        loop = asyncio.get_event_loop()

        # Load Vosk (fast streaming)
        try:
            import vosk
            vosk.SetLogLevel(-1)   # suppress verbose logs
            model_path = str(ai.VOSK_MODEL_PATH)
            from pathlib import Path
            if Path(model_path).exists():
                self._vosk_model = await loop.run_in_executor(
                    None, lambda: vosk.Model(model_path)
                )
                self._vosk_rec = vosk.KaldiRecognizer(
                    self._vosk_model, ai.VOSK_SAMPLE_RATE
                )
                log.info(f"Vosk loaded: {model_path}")
            else:
                log.warning(f"Vosk model not found at {model_path} — skipping Vosk.")
        except ImportError:
            log.warning("Vosk not installed — hotword detection unavailable.")
        except Exception as e:
            log.warning(f"Vosk init error: {e}")

        # Load faster-whisper (accurate transcription)
        try:
            from faster_whisper import WhisperModel
            self._whisper = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    ai.WHISPER_MODEL_SIZE,
                    device=ai.WHISPER_DEVICE,
                    compute_type=ai.WHISPER_COMPUTE_TYPE,
                )
            )
            log.info(
                f"faster-whisper loaded: {ai.WHISPER_MODEL_SIZE} "
                f"({ai.WHISPER_COMPUTE_TYPE})"
            )
        except ImportError:
            log.warning("faster-whisper not installed — trying openai-whisper.")
            await self._load_openai_whisper(loop)
        except Exception as e:
            log.warning(f"faster-whisper init error: {e}")
            await self._load_openai_whisper(loop)

    async def _load_openai_whisper(self, loop):
        """Fallback to openai-whisper if faster-whisper unavailable."""
        try:
            import whisper
            ai = self.cfg.ai
            self._whisper = await loop.run_in_executor(
                None, lambda: whisper.load_model(ai.WHISPER_MODEL_SIZE)
            )
            self._whisper_backend = "openai"
            log.info(f"openai-whisper loaded: {ai.WHISPER_MODEL_SIZE}")
        except Exception as e:
            log.error(f"All Whisper backends failed: {e}")

        # Enable continuous listening if Vosk is not available
        if not self._vosk_rec:
            log.warning("Vosk not available — enabling continuous Whisper speech capture fallback.")
            self._activated = True

    async def cleanup(self):
        self._vosk_model = None
        self._vosk_rec   = None
        self._whisper    = None

    # ── Transcription ─────────────────────────────────────────────────────────
    async def transcribe(self, audio_chunk: bytes) -> Optional[str]:
        """
        Process one audio chunk.
        Returns transcribed text if a complete utterance was captured, else None.
        """
        if self.cfg.system.MOCK_MIC:
            # In mock mode, return None (voice loop should inject test queries externally)
            return None

        # Step 1: Vosk partial/final result for hotword
        if self._vosk_rec:
            text_so_far = await self._vosk_partial(audio_chunk)
            hotword = self.cfg.system.HOTWORD.lower()
            if hotword in (text_so_far or "").lower():
                self._activated = True
                self._buffer.clear()
                self._buffer_count = 0
                log.debug(f"Hotword '{hotword}' detected — collecting utterance…")

        # Step 2: Buffer audio if activated
        if self._activated:
            self._buffer.append(audio_chunk)
            self._buffer_count += 1

            # Transcribe after ~3 seconds of audio
            if self._buffer_count >= _WHISPER_BUFFER_CHUNKS:
                self._buffer_count = 0
                audio_data = b"".join(self._buffer)
                self._buffer.clear()
                # Keep active if Vosk is missing, else deactivate until next hotword
                if not self._vosk_rec:
                    self._activated = True
                else:
                    self._activated = False
                return await self._whisper_transcribe(audio_data)

        return None

    async def _vosk_partial(self, chunk: bytes) -> Optional[str]:
        """Feed chunk to Vosk and return current partial text."""
        try:
            import json
            loop = asyncio.get_event_loop()
            accepted = await loop.run_in_executor(
                None, lambda: self._vosk_rec.AcceptWaveform(chunk)
            )
            if accepted:
                result = json.loads(self._vosk_rec.Result())
                return result.get("text", "")
            partial = json.loads(self._vosk_rec.PartialResult())
            return partial.get("partial", "")
        except Exception:
            return None

    async def _whisper_transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """Convert buffered audio bytes to text using Whisper."""
        try:
            loop = asyncio.get_event_loop()
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if hasattr(self, "_whisper_backend") and self._whisper_backend == "openai":
                # openai-whisper API
                result = await loop.run_in_executor(
                    None,
                    lambda: self._whisper.transcribe(
                        audio_np, language=self.cfg.ai.WHISPER_LANGUAGE
                    )
                )
                text = result.get("text", "").strip()
            else:
                # faster-whisper API
                segments, _ = await loop.run_in_executor(
                    None,
                    lambda: self._whisper.transcribe(
                        audio_np,
                        language=self.cfg.ai.WHISPER_LANGUAGE,
                        beam_size=3,
                        vad_filter=True,
                    )
                )
                text = " ".join(seg.text for seg in segments).strip()

            log.info(f"Whisper transcribed: '{text}'")
            return text if text else None
        except Exception as e:
            log.error(f"Whisper transcription error: {e}")
            return None

    def force_activate(self):
        """Manually activate utterance collection (for testing)."""
        self._activated = True
        self._buffer.clear()
        self._buffer_count = 0

    def is_ready(self) -> bool:
        return self._whisper is not None
