"""
audio/mic_stream.py — Microphone Input Stream
===============================================
Continuously captures audio from the default microphone using PyAudio.
Buffers chunks into an asyncio.Queue for downstream STT consumption.
Supports hotword-gated mode: streams only when the hotword is detected.

Provides:
  • async stream() → AsyncIterator[bytes]   — raw PCM audio chunks
  • async initialise() / cleanup()
"""

import asyncio
import threading
from typing import AsyncIterator, Optional

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


class MicrophoneStream:
    """
    Non-blocking microphone stream using PyAudio with a background thread.
    Produces audio chunks into an asyncio.Queue accessible from the event loop.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._pyaudio = None
        self._stream = None
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=50)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        hw = self.cfg.hardware

        if self.cfg.system.MOCK_MIC:
            log.info("Microphone: MOCK mode — no audio capture.")
            self._running = True
            return

        try:
            import pyaudio
            self._loop = asyncio.get_event_loop()
            self._pyaudio = pyaudio.PyAudio()

            device_index = hw.AUDIO_INPUT_DEVICE if hw.AUDIO_INPUT_DEVICE >= 0 else None

            self._stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=hw.AUDIO_CHANNELS,
                rate=hw.AUDIO_SAMPLE_RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=hw.AUDIO_CHUNK_SIZE,
                stream_callback=self._pyaudio_callback,
            )
            self._stream.start_stream()
            self._running = True
            log.info(
                f"Microphone open: {hw.AUDIO_SAMPLE_RATE}Hz, "
                f"chunk={hw.AUDIO_CHUNK_SIZE}, device={device_index or 'default'}"
            )
        except Exception as e:
            log.warning(f"Microphone init failed ({e}) — MOCK mode.")
            self.cfg.system.MOCK_MIC = True
            self._running = True

    def _pyaudio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback: runs in C thread, safely put chunk on asyncio queue."""
        import pyaudio
        if self._loop and self._running:
            asyncio.run_coroutine_threadsafe(
                self._put_chunk(in_data), self._loop
            )
        return (None, pyaudio.paContinue)

    async def _put_chunk(self, chunk: bytes):
        """Put a chunk into the queue, dropping oldest if full."""
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()  # drop oldest
                self._queue.put_nowait(chunk)
            except asyncio.QueueEmpty:
                pass

    async def cleanup(self):
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pyaudio:
            self._pyaudio.terminate()
        log.info("Microphone closed.")

    # ── Stream ────────────────────────────────────────────────────────────────
    async def stream(self) -> AsyncIterator[bytes]:
        """
        Async generator yielding raw PCM audio chunks (16-bit, 16 kHz, mono).
        In mock mode, yields silence chunks to keep the pipeline alive.
        """
        if self.cfg.system.MOCK_MIC:
            silence = b"\x00" * self.cfg.hardware.AUDIO_CHUNK_SIZE * 2
            while self._running:
                await asyncio.sleep(0.064)   # ~16 chunks/s at 1024 frames
                yield silence
            return

        while self._running:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk

    def is_ready(self) -> bool:
        return self._running

    def list_devices(self) -> list:
        """List available audio input devices (useful for debugging)."""
        if not self._pyaudio:
            try:
                import pyaudio
                pa = pyaudio.PyAudio()
                devices = [
                    {
                        "index": i,
                        "name": pa.get_device_info_by_index(i)["name"],
                        "channels": pa.get_device_info_by_index(i)["maxInputChannels"],
                    }
                    for i in range(pa.get_device_count())
                    if pa.get_device_info_by_index(i)["maxInputChannels"] > 0
                ]
                pa.terminate()
                return devices
            except Exception:
                return []
        return []
