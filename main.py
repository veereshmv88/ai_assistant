"""
main.py — AI Blind Assistant Entry Point
=========================================
Master asyncio orchestrator that starts all subsystems concurrently:
  • Camera capture loop
  • Ultrasonic distance monitoring
  • GPS location tracking
  • Microphone / STT listening
  • YOLO object detection
  • Ollama LLM reasoning
  • TTS audio output
  • Decision engine (sensor fusion)

Usage:
    python main.py                  # production mode (Pi hardware)
    python main.py --mock           # dev mode (Windows / macOS)
    python main.py --mock --debug   # dev + verbose logging
    python main.py --no-vision      # skip AI vision (faster startup)
"""

import asyncio
import argparse
import signal
import sys
from pathlib import Path

# ── Internal modules ──────────────────────────────────────────────────────────
from config import get_config
from utils.logger import get_logger
from utils.health_check import run_health_check

from sensors.camera import CameraModule
from sensors.ultrasonic import UltrasonicSensor
from sensors.gps_sensor import GPSModule

from audio.mic_stream import MicrophoneStream
from audio.speaker import SpeakerOutput

from stt.whisper_stt import WhisperSTT
from tts.piper_tts import PiperTTS

from vision.detector import ObjectDetector
from vision.scene_analyzer import SceneAnalyzer
from vision.face_recognizer import FaceRecognizer
from vision.currency_detector import CurrencyDetector

from ai.llm_engine import LLMEngine
from ai.intent_parser import IntentParser
from ai.decision_engine import DecisionEngine

from navigation.gps_navigator import GPSNavigator
from navigation.scene_memory import SceneMemory

from emergency.sos_handler import SOSHandler

log = get_logger(__name__)

# ─── Graceful shutdown ────────────────────────────────────────────────────────
_shutdown_event = asyncio.Event()


def _handle_signal(sig, frame):
    log.warning(f"Received signal {sig}, initiating graceful shutdown…")
    _shutdown_event.set()


# ─── CLI arguments ────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Blind Assistant — Raspberry Pi Voice Navigation System"
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Enable mock sensor mode for development (no Pi hardware needed)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--no-vision", action="store_true",
        help="Disable AI vision pipeline (YOLO + LLaVA) for faster startup"
    )
    parser.add_argument(
        "--no-gps", action="store_true",
        help="Disable GPS module"
    )
    parser.add_argument(
        "--health-check", action="store_true",
        help="Run system health check and exit"
    )
    return parser.parse_args()


# ─── Module initialisation ────────────────────────────────────────────────────
async def initialise_modules(cfg) -> dict:
    """
    Initialise all hardware and AI modules concurrently.
    Returns a dict of module name → instance.
    """
    log.info("Initialising all subsystems…")

    # TTS is created first so it can speak startup messages
    tts = PiperTTS(cfg)
    await tts.initialise()

    await tts.speak("Initialising AI Blind Assistant. Please wait.", priority="HIGH")

    # Initialise everything else concurrently
    camera       = CameraModule(cfg)
    ultrasonic   = UltrasonicSensor(cfg)
    gps          = GPSModule(cfg)
    mic          = MicrophoneStream(cfg)
    speaker      = SpeakerOutput(cfg, tts)
    stt          = WhisperSTT(cfg)
    detector     = ObjectDetector(cfg)
    scene_ana    = SceneAnalyzer(cfg)
    face_rec     = FaceRecognizer(cfg)
    currency     = CurrencyDetector(cfg)
    llm          = LLMEngine(cfg)
    intent       = IntentParser(cfg, llm)
    memory       = SceneMemory(cfg)
    navigator    = GPSNavigator(cfg, gps)
    sos          = SOSHandler(cfg, gps, tts)
    decision     = DecisionEngine(
        cfg=cfg,
        tts=tts,
        llm=llm,
        intent=intent,
        detector=detector,
        scene_analyzer=scene_ana,
        face_recognizer=face_rec,
        currency_detector=currency,
        navigator=navigator,
        memory=memory,
        sos=sos,
    )

    await asyncio.gather(
        camera.initialise(),
        ultrasonic.initialise(),
        gps.initialise(),
        mic.initialise(),
        stt.initialise(),
        detector.initialise(),
        scene_ana.initialise(),
        face_rec.initialise(),
        currency.initialise(),
        llm.initialise(),
        memory.initialise(),
        navigator.initialise(),
        sos.initialise(),
        decision.initialise(),
    )

    log.info("All subsystems ready.")
    await tts.speak(
        "System ready. I am your AI guide. Say 'Guide' followed by your question.",
        priority="HIGH",
    )

    return {
        "camera": camera,
        "ultrasonic": ultrasonic,
        "gps": gps,
        "mic": mic,
        "speaker": speaker,
        "stt": stt,
        "tts": tts,
        "detector": detector,
        "scene_analyzer": scene_ana,
        "face_recognizer": face_rec,
        "currency_detector": currency,
        "llm": llm,
        "intent": intent,
        "memory": memory,
        "navigator": navigator,
        "sos": sos,
        "decision": decision,
    }


# ─── Core async tasks ─────────────────────────────────────────────────────────
async def camera_loop(modules: dict, shutdown: asyncio.Event):
    """Continuously capture frames and push to shared queue."""
    camera   = modules["camera"]
    detector = modules["detector"]
    decision = modules["decision"]
    face_rec = modules["face_recognizer"]

    log.info("Camera loop started.")
    async for frame in camera.stream():
        if shutdown.is_set():
            break
        # Push frame to detector and decision engine concurrently
        await asyncio.gather(
            detector.process_frame(frame),
            face_rec.process_frame(frame),
            decision.on_camera_frame(frame),
        )


async def ultrasonic_loop(modules: dict, shutdown: asyncio.Event):
    """Poll ultrasonic sensor and push distance to decision engine."""
    sensor   = modules["ultrasonic"]
    decision = modules["decision"]
    tts      = modules["tts"]

    log.info("Ultrasonic loop started.")
    async for distance_cm in sensor.stream():
        if shutdown.is_set():
            break
        await decision.on_distance(distance_cm)


async def gps_loop(modules: dict, shutdown: asyncio.Event):
    """Stream GPS fixes and push to navigator + decision engine."""
    gps       = modules["gps"]
    navigator = modules["navigator"]
    decision  = modules["decision"]

    log.info("GPS loop started.")
    async for fix in gps.stream():
        if shutdown.is_set():
            break
        await navigator.on_gps_fix(fix)
        await decision.on_gps_fix(fix)


async def voice_loop(modules: dict, shutdown: asyncio.Event):
    """
    Microphone → STT → Intent → LLM → TTS pipeline.
    Listens continuously for hotword activation, then processes query.
    """
    mic      = modules["mic"]
    stt      = modules["stt"]
    decision = modules["decision"]

    log.info("Voice loop started.")
    async for audio_chunk in mic.stream():
        if shutdown.is_set():
            break
        text = await stt.transcribe(audio_chunk)
        if text:
            log.debug(f"STT: '{text}'")
            await decision.on_voice_input(text)


async def tts_output_loop(modules: dict, shutdown: asyncio.Event):
    """Drain the TTS output queue and speak queued messages."""
    tts = modules["tts"]
    log.info("TTS output loop started.")
    await tts.run_output_loop(shutdown)


async def sos_button_loop(modules: dict, shutdown: asyncio.Event):
    """Monitor hardware SOS button (GPIO) for long-press trigger."""
    sos = modules["sos"]
    log.info("SOS button monitor started.")
    await sos.monitor_button(shutdown)


async def mock_cli_loop(modules: dict, shutdown: asyncio.Event):
    """
    Non-blocking developer console loop. Runs only in mock mode.
    Enables manual input of voice queries or simulated sensor events.
    """
    decision = modules["decision"]
    ultrasonic = modules["ultrasonic"]
    gps = modules["gps"]
    log.info("Mock CLI Loop started. You can type commands directly into this console!")

    loop = asyncio.get_event_loop()
    def _read_line():
        import sys
        return sys.stdin.readline()

    # Wait a bit for startup logs to settle
    await asyncio.sleep(2.0)

    while not shutdown.is_set():
        try:
            line = await loop.run_in_executor(None, _read_line)
            if not line:
                await asyncio.sleep(0.5)
                continue
            line = line.strip()
            if not line:
                continue

            if line.startswith("distance "):
                try:
                    val = float(line.split()[1])
                    log.info(f"[Mock CLI] Simulating ultrasonic distance: {val} cm")
                    ultrasonic.set_mock_distance(val)
                except (IndexError, ValueError):
                    print("Usage: distance <number_cm>")
            elif line.startswith("gps "):
                try:
                    parts = line.split()
                    lat = float(parts[1])
                    lon = float(parts[2])
                    log.info(f"[Mock CLI] Simulating GPS position: {lat}, {lon}")
                    gps.set_mock_position(lat, lon)
                except (IndexError, ValueError):
                    print("Usage: gps <lat> <lon>")
            elif line.lower() in ("exit", "quit"):
                shutdown.set()
                break
            else:
                log.info(f"[Mock CLI] Processing text voice command: '{line}'")
                asyncio.create_task(decision.on_voice_input(line))
        except Exception as e:
            log.error(f"Error in Mock CLI loop: {e}")
            await asyncio.sleep(1.0)


# ─── Main orchestrator ────────────────────────────────────────────────────────
async def run(args: argparse.Namespace):
    """Set up config, initialise modules, run all tasks concurrently."""

    # 1. Configuration
    cfg = get_config(mock=args.mock)
    if args.debug:
        cfg.system.DEBUG = True
        cfg.system.LOG_LEVEL = "DEBUG"
    if args.no_vision:
        cfg.system.ENABLE_OLLAMA_VISION = False
    if args.no_gps:
        cfg.system.ENABLE_GPS = False

    log.info(f"Starting AI Blind Assistant (mock={args.mock}, debug={args.debug})")

    # 2. Health check
    if args.health_check:
        await run_health_check(cfg)
        return

    # 3. Initialise all modules
    modules = await initialise_modules(cfg)

    # 4. Register OS signal handlers for clean shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _shutdown_event.set)
        except NotImplementedError:
            # Windows does not support add_signal_handler for SIGINT
            signal.signal(sig, _handle_signal)

    # 5. Launch all concurrent tasks
    tasks = [
        asyncio.create_task(camera_loop(modules, _shutdown_event),    name="camera"),
        asyncio.create_task(ultrasonic_loop(modules, _shutdown_event), name="ultrasonic"),
        asyncio.create_task(voice_loop(modules, _shutdown_event),      name="voice"),
        asyncio.create_task(tts_output_loop(modules, _shutdown_event), name="tts_output"),
        asyncio.create_task(sos_button_loop(modules, _shutdown_event), name="sos_button"),
    ]

    if cfg.system.ENABLE_GPS:
        tasks.append(asyncio.create_task(gps_loop(modules, _shutdown_event), name="gps"))

    if args.mock:
        tasks.append(asyncio.create_task(mock_cli_loop(modules, _shutdown_event), name="mock_cli"))

    log.info(f"All {len(tasks)} async tasks running. Press Ctrl+C to stop.")

    # 6. Wait until shutdown is signalled
    await _shutdown_event.wait()

    # 7. Graceful teardown
    log.info("Shutting down all tasks…")
    tts = modules["tts"]
    await tts.speak("Shutting down. Goodbye.", priority="HIGH")
    await asyncio.sleep(2)

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Cleanup hardware
    for name, mod in modules.items():
        if hasattr(mod, "cleanup"):
            try:
                await mod.cleanup()
            except Exception as e:
                log.warning(f"Cleanup error for {name}: {e}")

    log.info("AI Blind Assistant stopped cleanly.")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n[Interrupted] AI Blind Assistant stopped.")
    except Exception as e:
        print(f"[FATAL] Unhandled error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
