"""
audio/speaker.py — Audio Output Manager
=========================================
Manages the TTS output pipeline:
  • Priority queue (EMERGENCY > NAVIGATION > INFO > IDLE)
  • Interruptible playback — high-priority messages cancel current speech
  • Delegates actual synthesis to tts/piper_tts.py

Provides:
  • async speak(text, priority) — queue a message for playback
  • async stop()               — interrupt current playback
"""

import asyncio
from enum import IntEnum
from typing import Optional

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


class Priority(IntEnum):
    IDLE        = 0
    INFO        = 1
    NAVIGATION  = 2
    HIGH        = 3
    EMERGENCY   = 4


class SpeakerOutput:
    """
    Priority-aware async audio output manager.
    Wraps PiperTTS and adds queuing + interruption logic.
    """

    def __init__(self, cfg: Config, tts):
        self.cfg = cfg
        self._tts = tts

    async def speak(self, text: str, priority: str = "INFO"):
        """Queue a message for speech output."""
        pri = Priority[priority.upper()] if isinstance(priority, str) else priority
        await self._tts.speak(text, priority=priority)

    async def speak_emergency(self, text: str):
        """Speak immediately, interrupting any current output."""
        await self._tts.speak(text, priority="EMERGENCY")

    async def stop(self):
        """Stop all pending and current speech."""
        await self._tts.stop()
