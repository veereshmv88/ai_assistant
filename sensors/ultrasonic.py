"""
sensors/ultrasonic.py — HC-SR04 Ultrasonic Distance Sensor
============================================================
Measures distance using a HC-SR04 sensor via RPi.GPIO.
On non-Pi hardware (MOCK mode) returns a configurable simulated distance.

Priority levels:
  CRITICAL  < 30 cm  → "Stop! Obstacle very close"
  WARNING   < 100 cm → "Caution, obstacle ahead"
  CLEAR     >= 100cm → safe

Provides:
  • async stream() → AsyncIterator[float]  — distance in cm
  • async get_distance() → float           — single reading
  • async initialise() / cleanup()
"""

import asyncio
import time
import collections
from typing import AsyncIterator, Optional

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)

# Distance thresholds (exported for use in decision engine)
CRITICAL_CM = 30.0
WARNING_CM  = 100.0


class UltrasonicSensor:
    """
    Async HC-SR04 driver with moving-average noise filtering.
    Falls back to mock mode when RPi.GPIO is unavailable.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._gpio = None
        self._running = False
        self._samples = collections.deque(
            maxlen=cfg.hardware.ULTRASONIC_SAMPLES
        )
        # Mock state
        self._mock_distance_cm: float = 200.0   # safe distance in mock mode

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        hw = self.cfg.hardware
        if self.cfg.system.MOCK_SENSORS:
            log.info("Ultrasonic: MOCK mode — simulated distances.")
            self._running = True
            return
        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(hw.ULTRASONIC_TRIGGER_PIN, GPIO.OUT)
            GPIO.setup(hw.ULTRASONIC_ECHO_PIN, GPIO.IN)
            GPIO.output(hw.ULTRASONIC_TRIGGER_PIN, False)
            await asyncio.sleep(0.3)   # sensor warm-up
            self._running = True
            log.info(
                f"HC-SR04 ready (TRIG=GPIO{hw.ULTRASONIC_TRIGGER_PIN}, "
                f"ECHO=GPIO{hw.ULTRASONIC_ECHO_PIN})"
            )
        except ImportError:
            log.warning("RPi.GPIO not available — switching to MOCK mode.")
            self.cfg.system.MOCK_SENSORS = True
            self._running = True

    async def cleanup(self):
        self._running = False
        if self._gpio:
            self._gpio.cleanup()
            self._gpio = None
        log.info("Ultrasonic sensor cleaned up.")

    # ── Measurement ───────────────────────────────────────────────────────────
    def _measure_sync(self) -> Optional[float]:
        """
        Blocking HC-SR04 measurement.
        Returns distance in cm or None on timeout.
        """
        hw = self.cfg.hardware
        GPIO = self._gpio

        trig = hw.ULTRASONIC_TRIGGER_PIN
        echo = hw.ULTRASONIC_ECHO_PIN

        # Send 10µs trigger pulse
        GPIO.output(trig, True)
        time.sleep(0.00001)
        GPIO.output(trig, False)

        # Wait for echo to go HIGH (timeout 50ms)
        t_start = time.time()
        pulse_start = t_start
        while GPIO.input(echo) == 0:
            pulse_start = time.time()
            if pulse_start - t_start > 0.05:
                return None  # timeout

        # Wait for echo to go LOW
        t_start = time.time()
        pulse_end = t_start
        while GPIO.input(echo) == 1:
            pulse_end = time.time()
            if pulse_end - t_start > 0.05:
                return None  # timeout

        duration = pulse_end - pulse_start
        distance_cm = (duration * 34300) / 2   # speed of sound = 343 m/s
        return round(distance_cm, 1)

    async def get_distance(self) -> float:
        """Return a single smoothed distance reading in cm."""
        if self.cfg.system.MOCK_SENSORS:
            return self._mock_distance_cm

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, self._measure_sync)

        if raw is None or raw < 2 or raw > 400:
            # Out-of-range or timeout — return last known value
            return self._samples[-1] if self._samples else 300.0

        self._samples.append(raw)
        return round(sum(self._samples) / len(self._samples), 1)

    async def stream(self) -> AsyncIterator[float]:
        """
        Async generator yielding distance readings at ULTRASONIC_POLL_HZ.
        """
        interval = 1.0 / self.cfg.hardware.ULTRASONIC_POLL_HZ

        while self._running:
            t0 = time.perf_counter()
            distance = await self.get_distance()
            yield distance

            elapsed = time.perf_counter() - t0
            await asyncio.sleep(max(0.0, interval - elapsed))

    # ── Mock helpers (for testing) ────────────────────────────────────────────
    def set_mock_distance(self, cm: float):
        """Manually set simulated distance (dev/test use)."""
        self._mock_distance_cm = cm

    def get_priority(self, distance_cm: float) -> str:
        """Classify distance into priority label."""
        if distance_cm < CRITICAL_CM:
            return "CRITICAL"
        elif distance_cm < WARNING_CM:
            return "WARNING"
        return "CLEAR"

    def is_ready(self) -> bool:
        return self._running
