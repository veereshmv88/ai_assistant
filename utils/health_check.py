"""
utils/health_check.py — System Health Diagnostics
===================================================
Checks all subsystems and reports status via TTS and console.
Run with: python main.py --health-check

Checks:
  ✓ Camera availability
  ✓ Microphone availability
  ✓ Ollama server reachability + model availability
  ✓ Vosk model file exists
  ✓ Piper model file exists
  ✓ GPS serial port (Pi only)
  ✓ RPi.GPIO (Pi only)
  ✓ EasyOCR import
  ✓ face_recognition import
  ✓ Data directory writable
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import NamedTuple

from config import Config
from utils.logger import get_logger

log = get_logger(__name__)


class CheckResult(NamedTuple):
    name: str
    ok: bool
    message: str


def safe_print(text: str):
    try:
        print(text)
    except UnicodeEncodeError:
        text = text.replace("✅", "[OK]").replace("❌", "[ERROR]")
        try:
            print(text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8"))
        except Exception:
            print(text.encode("ascii", errors="replace").decode("ascii"))


async def run_health_check(cfg: Config) -> bool:
    """
    Run all health checks and print a report.
    Returns True if all critical checks pass.
    """
    safe_print("\n" + "=" * 60)
    safe_print("  AI Blind Assistant — System Health Check")
    safe_print("=" * 60)

    checks: list[CheckResult] = []

    # Run all checks concurrently
    results = await asyncio.gather(
        _check_camera(cfg),
        _check_microphone(cfg),
        _check_ollama(cfg),
        _check_vosk(cfg),
        _check_piper(cfg),
        _check_gps_serial(cfg),
        _check_gpio(cfg),
        _check_easyocr(),
        _check_face_recognition(cfg),
        _check_data_dir(cfg),
        return_exceptions=True,
    )

    for r in results:
        if isinstance(r, Exception):
            checks.append(CheckResult("unknown", False, str(r)))
        else:
            checks.append(r)

    # Print report
    all_ok = True
    for check in checks:
        icon = "✅" if check.ok else "❌"
        safe_print(f"  {icon}  {check.name:<30}  {check.message}")
        if not check.ok and _is_critical(check.name):
            all_ok = False

    safe_print("=" * 60)
    if all_ok:
        safe_print("  ✅  All critical systems OPERATIONAL")
    else:
        safe_print("  ❌  Some critical systems have issues — see above")
    safe_print("=" * 60 + "\n")

    return all_ok


def _is_critical(name: str) -> bool:
    critical = {"Camera", "Microphone", "Data directory"}
    return name in critical


# ── Individual checks ─────────────────────────────────────────────────────────
async def _check_camera(cfg: Config) -> CheckResult:
    if cfg.system.MOCK_CAMERA:
        return CheckResult("Camera", True, "MOCK mode")
    try:
        import cv2
        cap = cv2.VideoCapture(cfg.hardware.CAMERA_DEVICE_INDEX)
        ok = cap.isOpened()
        cap.release()
        return CheckResult("Camera", ok, "OpenCV device opened" if ok else "Device not found")
    except Exception as e:
        return CheckResult("Camera", False, str(e))


async def _check_microphone(cfg: Config) -> CheckResult:
    if cfg.system.MOCK_SENSORS:
        return CheckResult("Microphone", True, "MOCK mode")
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        count = pa.get_device_count()
        pa.terminate()
        return CheckResult("Microphone", count > 0, f"{count} audio device(s) found")
    except Exception as e:
        return CheckResult("Microphone", False, str(e))


async def _check_ollama(cfg: Config) -> CheckResult:
    try:
        import ollama
        loop = asyncio.get_event_loop()
        models = await asyncio.wait_for(
            loop.run_in_executor(None, ollama.list),
            timeout=5.0,
        )
        model_names = [m["name"] for m in models.get("models", [])]
        vision_ok = any(cfg.ai.OLLAMA_VISION_MODEL.split(":")[0] in n for n in model_names)
        text_ok   = any(cfg.ai.OLLAMA_TEXT_MODEL.split(":")[0] in n for n in model_names)
        msg = f"Models available: {', '.join(model_names[:5]) or 'none'}"
        ok = vision_ok or text_ok
        return CheckResult("Ollama", ok, msg if ok else "Required models not found")
    except asyncio.TimeoutError:
        return CheckResult("Ollama", False, "Server not responding (timeout)")
    except Exception as e:
        return CheckResult("Ollama", False, f"Not reachable: {e}")


async def _check_vosk(cfg: Config) -> CheckResult:
    path = Path(cfg.ai.VOSK_MODEL_PATH)
    ok = path.exists() and path.is_dir()
    return CheckResult(
        "Vosk model",
        ok,
        f"Found: {path}" if ok else f"Not found: {path}",
    )


async def _check_piper(cfg: Config) -> CheckResult:
    import subprocess
    model_path = Path(cfg.ai.PIPER_MODEL_PATH)
    model_ok = model_path.exists()
    try:
        subprocess.run([cfg.ai.PIPER_EXECUTABLE, "--help"],
                      capture_output=True, timeout=3)
        binary_ok = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        binary_ok = False

    ok = binary_ok and model_ok
    if ok:
        msg = "Binary and model found"
    elif binary_ok:
        msg = f"Binary OK but model not found: {model_path}"
    elif model_ok:
        msg = "Model found but piper binary not in PATH"
    else:
        msg = "Binary and model both missing"
    return CheckResult("Piper TTS", ok, msg)


async def _check_gps_serial(cfg: Config) -> CheckResult:
    if cfg.system.MOCK_GPS or not cfg.system.ENABLE_GPS:
        return CheckResult("GPS serial", True, "MOCK mode or disabled")
    port = cfg.hardware.GPS_SERIAL_PORT
    exists = Path(port).exists()
    return CheckResult("GPS serial", exists, f"Port {port} {'found' if exists else 'not found'}")


async def _check_gpio(cfg: Config) -> CheckResult:
    if cfg.system.MOCK_SENSORS:
        return CheckResult("RPi.GPIO", True, "MOCK mode")
    try:
        import RPi.GPIO as GPIO
        return CheckResult("RPi.GPIO", True, "Available")
    except ImportError:
        return CheckResult("RPi.GPIO", False, "Not installed (OK on non-Pi)")


async def _check_easyocr() -> CheckResult:
    try:
        import easyocr
        return CheckResult("EasyOCR", True, "Installed")
    except ImportError:
        return CheckResult("EasyOCR", False, "pip install easyocr")


async def _check_face_recognition(cfg: Config) -> CheckResult:
    if not cfg.system.ENABLE_FACE_RECOGNITION:
        return CheckResult("Face recognition", True, "Disabled by config")
    try:
        import face_recognition
        known_dir = cfg.ai.FACE_KNOWN_DIR
        count = len(list(known_dir.glob("*.jpg"))) if known_dir.exists() else 0
        return CheckResult("Face recognition", True, f"Installed, {count} known face(s)")
    except ImportError:
        return CheckResult("Face recognition", False, "pip install face-recognition")


async def _check_data_dir(cfg: Config) -> CheckResult:
    data_dir = cfg.system.DATA_DIR
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        return CheckResult("Data directory", True, str(data_dir))
    except Exception as e:
        return CheckResult("Data directory", False, str(e))
