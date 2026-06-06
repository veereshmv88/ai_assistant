"""
sensors/gps_sensor.py — GPS Module (NEO-6M / NEO-M8N)
=======================================================
Reads NMEA sentences from GPS module via UART serial.
Parses position, speed, bearing, satellite count.
Mock mode provides a configurable simulated position.

Provides:
  • async stream()      → AsyncIterator[GPSFix]
  • async get_fix()     → GPSFix
  • async initialise() / cleanup()
"""

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class GPSFix:
    """Represents a single GPS position fix."""
    latitude:   float = 0.0          # decimal degrees (positive = N)
    longitude:  float = 0.0          # decimal degrees (positive = E)
    altitude_m: float = 0.0          # metres above sea level
    speed_kmh:  float = 0.0          # speed over ground
    bearing_deg: float = 0.0         # true heading 0–360°
    satellites: int   = 0            # satellites in use
    hdop:       float = 99.9         # horizontal dilution of precision
    timestamp:  float = field(default_factory=time.time)
    is_valid:   bool  = False        # True if fix quality is acceptable

    @property
    def has_fix(self) -> bool:
        return self.is_valid and self.satellites >= 3

    def distance_to(self, other: "GPSFix") -> float:
        """Haversine distance in metres between two fixes."""
        R = 6_371_000  # Earth radius in metres
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def bearing_to(self, other: "GPSFix") -> float:
        """Compass bearing in degrees from this fix to another."""
        lat1 = math.radians(self.latitude)
        lat2 = math.radians(other.latitude)
        dlon = math.radians(other.longitude - self.longitude)
        x = math.sin(dlon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    def __str__(self) -> str:
        return (
            f"GPS({self.latitude:.6f}°N, {self.longitude:.6f}°E | "
            f"speed={self.speed_kmh:.1f}km/h | sats={self.satellites} | "
            f"{'FIXED' if self.has_fix else 'NO FIX'})"
        )


class GPSModule:
    """
    Async GPS reader using pyserial + pynmea2 for NMEA sentence parsing.
    No gpsd daemon required — direct serial access.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._serial = None
        self._running = False
        self._latest_fix = GPSFix()

        # Mock GPS location: defaults to Bengaluru city centre
        self._mock_fix = GPSFix(
            latitude=12.9716,
            longitude=77.5946,
            altitude_m=920.0,
            speed_kmh=0.0,
            bearing_deg=90.0,
            satellites=8,
            hdop=1.2,
            is_valid=True,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        if not self.cfg.system.ENABLE_GPS:
            log.info("GPS: disabled by config.")
            return

        async def resolve_ip_location():
            def fetch_ip_location_sync():
                import urllib.request
                import json
                try:
                    req = urllib.request.Request(
                        "http://ip-api.com/json/",
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    with urllib.request.urlopen(req, timeout=5) as response:
                        data = json.loads(response.read().decode())
                        if data.get("status") == "success":
                            return (
                                float(data.get("lat")),
                                float(data.get("lon")),
                                data.get("city", "Unknown"),
                                data.get("country", "Unknown")
                            )
                except Exception as e:
                    log.debug(f"IP Geolocation fetch failed: {e}")
                return None

            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(None, fetch_ip_location_sync)
            if res:
                lat, lon, city, country = res
                log.info(f"GPS: Successfully determined location via IP: {city}, {country} ({lat}, {lon})")
                self._mock_fix.latitude = lat
                self._mock_fix.longitude = lon
            else:
                log.info("GPS: IP Geolocation failed, using default simulated position (Bengaluru).")

        if self.cfg.system.MOCK_GPS:
            log.info("GPS: MOCK mode active. Attempting to fetch real coordinates via software IP geolocation...")
            await resolve_ip_location()
            self._latest_fix = self._mock_fix
            self._running = True
            return

        try:
            import serial
            hw = self.cfg.hardware
            loop = asyncio.get_event_loop()
            self._serial = await loop.run_in_executor(
                None,
                lambda: serial.Serial(
                    hw.GPS_SERIAL_PORT,
                    hw.GPS_BAUD_RATE,
                    timeout=hw.GPS_TIMEOUT,
                )
            )
            self._running = True
            log.info(
                f"GPS serial opened: {hw.GPS_SERIAL_PORT} @ {hw.GPS_BAUD_RATE} baud"
            )
        except Exception as e:
            log.warning(f"GPS hardware init failed ({e}) — switching to software IP geolocation mock mode.")
            self.cfg.system.MOCK_GPS = True
            await resolve_ip_location()
            self._latest_fix = self._mock_fix
            self._running = True

    async def cleanup(self):
        self._running = False
        if self._serial and self._serial.is_open:
            self._serial.close()
        log.info("GPS serial closed.")

    # ── Parsing ───────────────────────────────────────────────────────────────
    def _parse_line(self, line: str) -> Optional[GPSFix]:
        """Parse one NMEA sentence and update the current fix."""
        try:
            import pynmea2
            msg = pynmea2.parse(line.strip())
        except Exception:
            return None

        fix = GPSFix(timestamp=time.time())

        if hasattr(msg, "latitude") and msg.latitude:
            fix.latitude  = float(msg.latitude)
            fix.longitude = float(msg.longitude)
            fix.is_valid  = True

        # GPRMC carries speed + bearing
        if msg.sentence_type == "RMC" and hasattr(msg, "spd_over_grnd"):
            try:
                fix.speed_kmh = float(msg.spd_over_grnd or 0) * 1.852  # knots→km/h
                fix.bearing_deg = float(msg.true_course or 0)
                fix.is_valid = (msg.status == "A")
            except (ValueError, TypeError):
                pass

        # GPGGA carries altitude + satellites
        if msg.sentence_type == "GGA" and hasattr(msg, "altitude"):
            try:
                fix.altitude_m = float(msg.altitude or 0)
                fix.satellites = int(msg.num_sats or 0)
                fix.hdop       = float(msg.horizontal_dil or 99.9)
                fix.is_valid   = (int(msg.gps_qual or 0) > 0)
            except (ValueError, TypeError):
                pass

        # Merge with running fix so all fields stay populated
        if fix.is_valid:
            if fix.latitude:
                self._latest_fix.latitude   = fix.latitude
                self._latest_fix.longitude  = fix.longitude
            if fix.altitude_m:
                self._latest_fix.altitude_m = fix.altitude_m
            if fix.satellites:
                self._latest_fix.satellites = fix.satellites
            if fix.speed_kmh is not None:
                self._latest_fix.speed_kmh  = fix.speed_kmh
            if fix.bearing_deg:
                self._latest_fix.bearing_deg = fix.bearing_deg
            self._latest_fix.is_valid = True
            self._latest_fix.timestamp = fix.timestamp
            return self._latest_fix

        return None

    def _read_line_sync(self) -> Optional[str]:
        """Blocking readline from serial port."""
        try:
            raw = self._serial.readline()
            return raw.decode("ascii", errors="replace")
        except Exception:
            return None

    async def get_fix(self) -> GPSFix:
        """Return the most recent GPS fix."""
        return self._latest_fix

    async def stream(self) -> AsyncIterator[GPSFix]:
        """
        Async generator yielding new GPS fixes as they arrive (~1 Hz from NEO-6M).
        """
        if self.cfg.system.MOCK_GPS:
            while self._running:
                await asyncio.sleep(1.0)
                yield self._mock_fix
            return

        loop = asyncio.get_event_loop()
        while self._running:
            line = await loop.run_in_executor(None, self._read_line_sync)
            if line and line.startswith("$GP"):
                fix = self._parse_line(line)
                if fix:
                    yield fix
            else:
                await asyncio.sleep(0.01)

    # ── Mock helpers ──────────────────────────────────────────────────────────
    def set_mock_position(self, lat: float, lon: float):
        self._mock_fix.latitude  = lat
        self._mock_fix.longitude = lon

    def is_ready(self) -> bool:
        return self._running
