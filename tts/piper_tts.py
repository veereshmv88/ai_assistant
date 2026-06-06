"""
tts/piper_tts.py — Text-to-Speech Engine (Piper + Windows fallback)
=====================================================================
Primary TTS: Piper (piper-tts) — natural voice, <100ms latency on Pi 5.
Fallback TTS: eSpeak-ng — robotic but instant, used for EMERGENCY alerts
              when Piper is busy or unavailable.

Priority queue levels:
  EMERGENCY  (4) — preempts all, speaks via eSpeak immediately
  HIGH       (3) — next in line
  NAVIGATION (2) — navigation instructions
  INFO       (1) — general information
  IDLE       (0) — background announcements

Provides:
  • async speak(text, priority)    — queue a message
  • async stop()                   — clear queue and kill current speech
  • async run_output_loop(event)   — drain queue loop (run as asyncio task)
  • async initialise() / cleanup()
"""

import asyncio
import subprocess
import os
import platform
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
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


@dataclass(order=True)
class _TTSItem:
    """Priority queue item. Lower sort_key → higher priority (PriorityQueue is min-heap)."""
    sort_key: int = field(compare=True)
    text:     str = field(compare=False)
    priority: Priority = field(compare=False)

    @classmethod
    def make(cls, text: str, priority: Priority) -> "_TTSItem":
        # Invert so EMERGENCY (4) has lowest sort_key → spoken first
        return cls(sort_key=-priority.value, text=text, priority=priority)


class PiperTTS:
    """
    Piper TTS engine with eSpeak fallback and priority queue.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._queue: asyncio.PriorityQueue[_TTSItem] = asyncio.PriorityQueue()
        self._current_proc: Optional[asyncio.subprocess.Process] = None
        self._piper_available = False
        self._espeak_available = False
        self._speaking = False
        self._dedup_cache: dict[str, float] = {}   # text → last spoken time
        self._dedup_cooldown = cfg.system.DEDUP_COOLDOWN_SECONDS

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        # Check piper binary
        self._piper_available = self._check_piper()
        # Check espeak-ng
        self._espeak_available = self._check_espeak()
        # Check pyttsx3 (Windows SAPI / macOS say fallback)
        self._pyttsx3_available = self._check_pyttsx3()

        if self._piper_available:
            log.info(f"Piper TTS ready: {self.cfg.ai.PIPER_MODEL_PATH}")
        elif self._espeak_available:
            log.warning("Piper not found — using eSpeak fallback (robotic voice).")
        elif self._pyttsx3_available:
            log.info("Using pyttsx3 (Windows SAPI) for TTS — dev/test mode.")
        else:
            log.warning("No TTS engine available! Speech output disabled.")

        # Standalone diagnostic test
        log.info("Running TTS startup diagnostic check. Speaking 'Boss, TTS is working'...")
        try:
            if self._pyttsx3_available:
                await self._pyttsx3_speak("Boss, T T S is working.")
            elif self._piper_available:
                await self._piper_speak("Boss, T T S is working.")
            elif self._espeak_available:
                await self._espeak_speak("Boss, T T S is working.")
            else:
                log.warning("Diagnostic check skipped: no physical TTS engine available.")
        except Exception as e:
            log.error(f"TTS diagnostic test error: {e}")

    def _check_piper(self) -> bool:
        piper_exec = self.cfg.ai.PIPER_EXECUTABLE
        model_path = Path(self.cfg.ai.PIPER_MODEL_PATH)
        try:
            subprocess.run(
                [piper_exec, "--help"],
                capture_output=True, timeout=5
            )
            return model_path.exists()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _check_espeak(self) -> bool:
        try:
            subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True, timeout=3
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            try:
                subprocess.run(["espeak", "--version"], capture_output=True, timeout=3)
                return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return False

    def _check_pyttsx3(self) -> bool:
        """Check if pyttsx3 is available (Windows SAPI / macOS say fallback)."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.stop()
            return True
        except Exception:
            return False

    async def cleanup(self):
        await self.stop()
        log.info("TTS cleaned up.")

    # ── Public API ────────────────────────────────────────────────────────────
    async def speak(self, text: str, priority: str = "INFO"):
        """Queue text for speech output."""
        if not text or not text.strip():
            return

        pri = Priority[priority.upper()] if isinstance(priority, str) else priority

        # Emergency skips dedup
        if pri < Priority.EMERGENCY:
            now = time.time()
            last = self._dedup_cache.get(text.lower(), 0)
            if now - last < self._dedup_cooldown:
                log.debug(f"TTS dedup suppressed: '{text[:40]}'")
                return
            self._dedup_cache[text.lower()] = now

        item = _TTSItem.make(text, pri)
        await self._queue.put(item)
        log.debug(f"TTS queued [{priority}]: '{text[:60]}'")

    async def stop(self):
        """Interrupt current speech and clear queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        if self._current_proc:
            try:
                self._current_proc.terminate()
                await asyncio.sleep(0.1)
            except Exception:
                pass
            self._current_proc = None

    # ── Output loop (run as asyncio task) ────────────────────────────────────
    async def run_output_loop(self, shutdown: asyncio.Event):
        """
        Drain the TTS queue and speak items in priority order.
        Runs forever until shutdown is set.
        """
        while not shutdown.is_set():
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            await self._synthesise(item.text, item.priority)
            self._queue.task_done()

    # ── Synthesis ─────────────────────────────────────────────────────────────
    async def _synthesise(self, text: str, priority: Priority):
        """Synthesise and play audio for the given text."""
        log.info(f"TTS synthesis started for priority {priority.name}: '{text[:60]}'")
        if self.cfg.system.MOCK_TTS:
            # Dev mode: print to console instead of speaking
            log.info(f"[TTS MOCK {priority.name}] {text}")
            print(f"\n🔊 [{priority.name}] {text}\n")
            await asyncio.sleep(len(text) * 0.05)  # simulate speech duration
            log.info(f"TTS mock playback complete: '{text[:40]}'")
            return

        self._speaking = True
        try:
            if priority == Priority.EMERGENCY and self._espeak_available:
                log.info("Speaking via eSpeak (EMERGENCY)...")
                await self._espeak_speak(text)
            elif self._piper_available:
                log.info("Speaking via Piper...")
                await self._piper_speak(text)
            elif self._espeak_available:
                log.info("Speaking via eSpeak fallback...")
                await self._espeak_speak(text)
            elif self._pyttsx3_available:
                log.info("Speaking via pyttsx3 (SAPI)...")
                await self._pyttsx3_speak(text)
            else:
                log.warning("No physical TTS backend found! Printing to console as last resort.")
                print(f"\n🔊 [{priority.name}] {text}\n")
            log.info(f"TTS playback completed successfully for text: '{text[:40]}'")
        except asyncio.CancelledError:
            log.info("TTS playback was cancelled.")
        except Exception as e:
            log.error(f"TTS synthesis error: {e}")
        finally:
            self._speaking = False

    async def _piper_speak(self, text: str):
        """Synthesise with Piper and pipe to aplay/afplay."""
        ai = self.cfg.ai
        player = self._get_audio_player()

        cmd_piper = [
            ai.PIPER_EXECUTABLE,
            "--model", ai.PIPER_MODEL_PATH,
            "--output-raw",
        ]
        cmd_play = player + ["--rate", "22050", "--channels", "1", "--format", "S16_LE", "-"]

        try:
            # Piper → raw PCM → aplay pipeline
            piper_proc = await asyncio.create_subprocess_exec(
                *cmd_piper,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._current_proc = piper_proc

            play_proc = await asyncio.create_subprocess_exec(
                *cmd_play,
                stdin=piper_proc.stdout,
                stderr=asyncio.subprocess.DEVNULL,
            )

            # Feed text to piper stdin
            await piper_proc.communicate(text.encode("utf-8"))
            await play_proc.wait()
        except FileNotFoundError:
            log.warning("Piper not found — falling back to eSpeak.")
            await self._espeak_speak(text)
        finally:
            self._current_proc = None

    async def _espeak_speak(self, text: str):
        """Synthesise with eSpeak-ng (fallback — robotic voice)."""
        cmd = ["espeak-ng", "-s", "165", "-a", "180", text]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            self._current_proc = proc
            await proc.wait()
        except FileNotFoundError:
            try:
                cmd[0] = "espeak"
                proc = await asyncio.create_subprocess_exec(*cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            except FileNotFoundError:
                log.error("Neither espeak-ng nor espeak found.")
        finally:
            self._current_proc = None

    async def _pyttsx3_speak(self, text: str):
        """Speak using pyttsx3 (Windows SAPI / macOS NSSpeechSynthesizer)."""
        loop = asyncio.get_event_loop()
        def _speak_sync():
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)   # words per minute
            engine.setProperty("volume", 0.95)
            # Prefer a female voice if available (easier to understand)
            voices = engine.getProperty("voices")
            if voices:
                female = next((v for v in voices if "female" in v.name.lower()), None)
                if female:
                    engine.setProperty("voice", female.id)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        try:
            await loop.run_in_executor(None, _speak_sync)
        except Exception as e:
            log.error(f"pyttsx3 error: {e}")
            print(f"\n🔊 {text}\n")

    @staticmethod
    def _get_audio_player() -> list[str]:
        """Return appropriate audio player command for OS."""
        sys_name = platform.system()
        if sys_name == "Linux":
            return ["aplay"]
        elif sys_name == "Darwin":
            return ["afplay"]
        else:
            # Windows: use SoundPlayer via PowerShell
            return ["powershell", "-c", "(New-Object Media.SoundPlayer '%s').PlaySync()"]

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def is_ready(self) -> bool:
        return (
            self._piper_available
            or self._espeak_available
            or self._pyttsx3_available
            or self.cfg.system.MOCK_SENSORS  # mock always ready
        )
