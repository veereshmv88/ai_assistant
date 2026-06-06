"""
emergency/sos_handler.py — Emergency SOS Alert System
======================================================
Triggered by:
  • Voice command: "SOS", "Help", "Emergency", "Call for help"
  • Hardware button: Long-press (≥3 seconds) on GPIO pin 17
  • Programmatic: sos_handler.trigger(message, gps_fix)

Actions (in sequence):
  1. Speak: "Sending emergency alert. Stay calm."
  2. Get current GPS coordinates
  3. Build SOS message with location link (Google Maps)
  4. Send SMS via Twilio (requires internet)
  5. Send email via SMTP (fallback)
  6. Log event to local SQLite database
  7. Repeat voice reassurance every 30 seconds

Provides:
  • async trigger(message, gps_fix)    — programmatic SOS trigger
  • async monitor_button(shutdown)     — GPIO button monitor loop
  • async initialise() / cleanup()
"""

import asyncio
import smtplib
import sqlite3
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional

from config import Config
from sensors.gps_sensor import GPSFix
from utils.logger import get_logger

log = get_logger(__name__)

_SOS_LOG_DB = None   # will be set in initialise()


class SOSHandler:
    """
    Multi-channel emergency alert handler with GPIO button monitoring.
    """

    def __init__(self, cfg: Config, gps_module, tts):
        self.cfg = cfg
        self._gps = gps_module
        self._tts = tts
        self._last_sos_time = 0.0
        self._gpio = None
        self._sos_db: Optional[sqlite3.Connection] = None
        self._button_pressed_time: Optional[float] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    async def initialise(self):
        if not self.cfg.system.ENABLE_SOS:
            log.info("SOS: disabled.")
            return

        # Initialise SOS log database
        db_path = self.cfg.system.DATA_DIR / "sos_log.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._sos_db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._sos_db.execute("""
            CREATE TABLE IF NOT EXISTS sos_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                message   TEXT,
                latitude  REAL,
                longitude REAL,
                sms_sent  INTEGER DEFAULT 0,
                email_sent INTEGER DEFAULT 0
            )
        """)
        self._sos_db.commit()

        # Set up GPIO button (Pi only)
        if not self.cfg.system.MOCK_SENSORS:
            try:
                import RPi.GPIO as GPIO
                self._gpio = GPIO
                pin = self.cfg.hardware.SOS_BUTTON_PIN
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
                log.info(f"SOS button configured on GPIO{pin}.")
            except ImportError:
                log.info("RPi.GPIO not available — GPIO SOS button disabled.")

        log.info("SOS handler ready.")

    async def cleanup(self):
        if self._sos_db:
            self._sos_db.close()
        if self._gpio:
            self._gpio.cleanup()

    # ── GPIO button monitor ───────────────────────────────────────────────────
    async def monitor_button(self, shutdown: asyncio.Event):
        """
        Monitor GPIO button for long-press SOS trigger.
        Checks button state every 100ms.
        """
        if self._gpio is None:
            # No GPIO available — just sleep
            await shutdown.wait()
            return

        pin = self.cfg.hardware.SOS_BUTTON_PIN
        hold_threshold = self.cfg.hardware.SOS_BUTTON_HOLD_SECONDS
        loop = asyncio.get_event_loop()

        while not shutdown.is_set():
            pressed = await loop.run_in_executor(
                None, lambda: not self._gpio.input(pin)   # active-low with pull-up
            )
            now = time.time()

            if pressed:
                if self._button_pressed_time is None:
                    self._button_pressed_time = now
                elif now - self._button_pressed_time >= hold_threshold:
                    self._button_pressed_time = None
                    await self.trigger("GPIO button long-press")
            else:
                self._button_pressed_time = None

            await asyncio.sleep(0.1)

    # ── SOS trigger ───────────────────────────────────────────────────────────
    async def trigger(
        self,
        message: str = "User requested emergency assistance.",
        gps_fix: Optional[GPSFix] = None,
    ):
        """
        Execute full SOS protocol.
        Rate-limited by SOS_COOLDOWN_SECONDS to prevent duplicate sends.
        """
        now = time.time()
        cooldown = self.cfg.emergency.SOS_COOLDOWN_SECONDS
        if now - self._last_sos_time < cooldown:
            log.warning("SOS cooldown active — ignoring duplicate trigger.")
            return

        self._last_sos_time = now
        log.warning(f"SOS TRIGGERED: {message}")

        # Step 1: Immediate voice feedback
        await self._tts.speak(
            "Emergency alert activated. Sending your location to your emergency contact. "
            "Stay calm and stay where you are.",
            priority="EMERGENCY",
        )

        # Step 2: Get GPS fix
        if gps_fix is None and self._gps:
            gps_fix = await self._gps.get_fix()

        # Step 3: Build SOS message
        lat = gps_fix.latitude if gps_fix and gps_fix.has_fix else None
        lon = gps_fix.longitude if gps_fix and gps_fix.has_fix else None
        sos_text = self._build_message(message, lat, lon)

        # Step 4–5: Send alerts concurrently
        sms_ok, email_ok = await asyncio.gather(
            self._send_sms(sos_text),
            self._send_email(sos_text),
            return_exceptions=True,
        )

        # Step 6: Log event
        self._log_event(message, lat, lon, bool(sms_ok), bool(email_ok))

        # Step 7: Confirm to user
        channels = []
        if sms_ok:
            channels.append("SMS")
        if email_ok:
            channels.append("email")
        channel_str = " and ".join(channels) if channels else "locally"
        await self._tts.speak(
            f"Emergency alert sent via {channel_str}. Help is on the way.",
            priority="EMERGENCY",
        )

        # Periodic reassurance (runs concurrently with rest of system)
        asyncio.create_task(self._periodic_reassurance())

    @staticmethod
    def _build_message(user_msg: str, lat: Optional[float], lon: Optional[float]) -> str:
        """Build the SOS message text with Google Maps link."""
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
        msg = (
            f"🆘 EMERGENCY ALERT — AI Blind Assistant\n"
            f"Time: {timestamp_str}\n"
            f"Message: {user_msg}\n"
        )
        if lat and lon:
            maps_link = f"https://maps.google.com/?q={lat:.6f},{lon:.6f}"
            msg += (
                f"GPS Location: {lat:.6f}°N, {lon:.6f}°E\n"
                f"Google Maps: {maps_link}\n"
            )
        else:
            msg += "GPS location: unavailable (no fix)\n"

        msg += "\nPlease respond immediately or contact emergency services."
        return msg

    async def _send_sms(self, message: str) -> bool:
        """Send SOS SMS via Twilio API."""
        em = self.cfg.emergency
        if not all([em.TWILIO_ACCOUNT_SID, em.TWILIO_AUTH_TOKEN,
                    em.TWILIO_FROM_NUMBER, em.EMERGENCY_PHONE]):
            log.info("Twilio not configured — SMS skipped.")
            return False

        loop = asyncio.get_event_loop()
        try:
            def _send():
                from twilio.rest import Client
                client = Client(em.TWILIO_ACCOUNT_SID, em.TWILIO_AUTH_TOKEN)
                client.messages.create(
                    body=message,
                    from_=em.TWILIO_FROM_NUMBER,
                    to=em.EMERGENCY_PHONE,
                )
            await asyncio.wait_for(
                loop.run_in_executor(None, _send),
                timeout=15.0,
            )
            log.info(f"SOS SMS sent to {em.EMERGENCY_PHONE}")
            return True
        except asyncio.TimeoutError:
            log.error("SMS send timed out.")
        except Exception as e:
            log.error(f"SMS send failed: {e}")
        return False

    async def _send_email(self, message: str) -> bool:
        """Send SOS email via SMTP."""
        em = self.cfg.emergency
        if not all([em.SMTP_USER, em.SMTP_PASS, em.EMERGENCY_EMAIL]):
            log.info("Email not configured — email alert skipped.")
            return False

        loop = asyncio.get_event_loop()
        try:
            def _send():
                msg = MIMEMultipart()
                msg["Subject"] = "🆘 EMERGENCY: Blind Assistant SOS Alert"
                msg["From"]    = em.SMTP_USER
                msg["To"]      = em.EMERGENCY_EMAIL
                msg.attach(MIMEText(message, "plain"))

                with smtplib.SMTP(em.SMTP_HOST, em.SMTP_PORT, timeout=10) as server:
                    server.starttls()
                    server.login(em.SMTP_USER, em.SMTP_PASS)
                    server.sendmail(em.SMTP_USER, em.EMERGENCY_EMAIL, msg.as_string())

            await asyncio.wait_for(
                loop.run_in_executor(None, _send),
                timeout=20.0,
            )
            log.info(f"SOS email sent to {em.EMERGENCY_EMAIL}")
            return True
        except Exception as e:
            log.error(f"Email send failed: {e}")
        return False

    async def _periodic_reassurance(self):
        """Speak reassuring messages every 30 seconds after SOS."""
        messages = [
            "Help has been alerted. Please stay where you are.",
            "Emergency services have been notified. Stay calm.",
            "Your emergency contact has your location. Help is coming.",
        ]
        for msg in messages:
            await asyncio.sleep(30)
            await self._tts.speak(msg, priority="HIGH")

    def _log_event(self, message, lat, lon, sms_ok, email_ok):
        """Log SOS event to local database."""
        if self._sos_db:
            try:
                self._sos_db.execute(
                    "INSERT INTO sos_events "
                    "(timestamp, message, latitude, longitude, sms_sent, email_sent) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (time.time(), message, lat, lon, int(sms_ok), int(email_ok)),
                )
                self._sos_db.commit()
            except Exception as e:
                log.error(f"Failed to log SOS event: {e}")

    def is_ready(self) -> bool:
        return True
