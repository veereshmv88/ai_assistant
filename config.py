"""
config.py — Central Configuration for AI Blind Assistant
=========================================================
All system settings, hardware pins, model paths, thresholds, and feature
flags are consolidated here. Modify this file to tune the assistant for
your specific Raspberry Pi hardware setup.

Usage:
    from config import Config
    cfg = Config()
"""

from dataclasses import dataclass, field
from pathlib import Path
import os

# ─── Project Root ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()


@dataclass
class HardwareConfig:
    """GPIO pin assignments and hardware parameters."""

    # ── Ultrasonic Sensor (HC-SR04) ──────────────────────────────────────────
    ULTRASONIC_TRIGGER_PIN: int = 23       # BCM GPIO 23 (Pin 16)
    ULTRASONIC_ECHO_PIN: int = 24          # BCM GPIO 24 (Pin 18)
    ULTRASONIC_CRITICAL_CM: float = 30.0  # < 30 cm → STOP immediately
    ULTRASONIC_WARNING_CM: float = 100.0  # < 100 cm → caution alert
    ULTRASONIC_POLL_HZ: float = 20.0      # readings per second
    ULTRASONIC_SAMPLES: int = 5           # moving-average window

    # ── Camera ───────────────────────────────────────────────────────────────
    CAMERA_DEVICE_INDEX: int = 0          # 0 = default; -1 = picamera2
    CAMERA_WIDTH: int = 640
    CAMERA_HEIGHT: int = 480
    CAMERA_FPS: int = 30
    USE_PICAMERA2: bool = True            # set False for USB webcam

    # ── GPS (NEO-6M via UART) ─────────────────────────────────────────────────
    GPS_SERIAL_PORT: str = "/dev/serial0"
    GPS_BAUD_RATE: int = 9600
    GPS_TIMEOUT: float = 1.0
    GPS_MIN_SATELLITES: int = 4          # minimum fix quality

    # ── Audio ─────────────────────────────────────────────────────────────────
    AUDIO_SAMPLE_RATE: int = 16000       # Hz (Vosk/Whisper require 16kHz)
    AUDIO_CHANNELS: int = 1             # mono
    AUDIO_CHUNK_SIZE: int = 1024        # frames per buffer
    AUDIO_INPUT_DEVICE: int = -1        # -1 = system default
    AUDIO_OUTPUT_DEVICE: int = -1       # -1 = system default

    # ── SOS Button (optional GPIO button) ────────────────────────────────────
    SOS_BUTTON_PIN: int = 17            # BCM GPIO 17 (Pin 11)
    SOS_BUTTON_HOLD_SECONDS: float = 3.0  # hold duration to trigger SOS


@dataclass
class AIConfig:
    """AI model paths and inference parameters."""

    # ── Ollama ────────────────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_VISION_MODEL: str = "llava:7b"     # multimodal vision+text
    OLLAMA_TEXT_MODEL: str = "gemma4:e4b"    # fast text-only reasoning
    OLLAMA_TIMEOUT: int = 60                   # seconds
    OLLAMA_CONTEXT_WINDOW: int = 10            # last N turns to keep

    # ── YOLO-World / YOLOv8 Object Detection ─────────────────────────────────
    YOLO_MODEL_PATH: str = str(BASE_DIR / "models" / "yolov8n.pt")
    YOLO_CONFIDENCE: float = 0.45
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_INPUT_SIZE: int = 320             # 320px for Pi speed
    YOLO_FPS_TARGET: int = 8              # inference FPS goal
    YOLO_CLASSES_OF_INTEREST: list = field(default_factory=lambda: [
        "person", "car", "truck", "bus", "bicycle", "motorcycle",
        "traffic light", "stop sign", "fire hydrant", "bench", "chair",
        "bottle", "cup", "door", "stairs", "dog", "cat",
    ])

    # ── Whisper STT ───────────────────────────────────────────────────────────
    WHISPER_MODEL_SIZE: str = "base"      # tiny/base/small — base best for Pi5
    WHISPER_LANGUAGE: str = "en"
    WHISPER_DEVICE: str = "cpu"           # Pi has no CUDA
    WHISPER_COMPUTE_TYPE: str = "int8"    # quantized for CPU speed

    # ── Vosk (fast streaming STT) ─────────────────────────────────────────────
    VOSK_MODEL_PATH: str = str(BASE_DIR / "models" / "vosk-model-small-en-us")
    VOSK_SAMPLE_RATE: int = 16000

    # ── Piper TTS ─────────────────────────────────────────────────────────────
    PIPER_EXECUTABLE: str = "piper"       # must be in PATH or full path
    PIPER_MODEL_PATH: str = str(BASE_DIR / "models" / "en_US-lessac-medium.onnx")
    PIPER_CONFIG_PATH: str = str(BASE_DIR / "models" / "en_US-lessac-medium.onnx.json")
    PIPER_SPEAKER_ID: int = 0

    # ── EasyOCR ───────────────────────────────────────────────────────────────
    OCR_LANGUAGES: list = field(default_factory=lambda: ["en"])
    OCR_GPU: bool = False                 # Pi has no GPU
    OCR_MIN_CONFIDENCE: float = 0.4

    # ── Face Recognition ──────────────────────────────────────────────────────
    FACE_KNOWN_DIR: Path = BASE_DIR / "data" / "known_faces"
    FACE_TOLERANCE: float = 0.55          # lower = stricter match
    FACE_MODEL: str = "small"             # 'small' is faster on Pi

    # ── Currency Detection ────────────────────────────────────────────────────
    CURRENCY_COUNTRY: str = "INR"         # INR, USD, EUR


@dataclass
class NavigationConfig:
    """GPS navigation and map settings."""

    NOMINATIM_USER_AGENT: str = "blind_assistant_v1"
    OSM_CACHE_DIR: Path = BASE_DIR / "data" / "map_cache"
    NAV_ANNOUNCE_DISTANCE_M: float = 20.0  # announce turn X metres ahead
    NAV_RECALCULATE_INTERVAL: int = 10     # seconds between route checks
    SCENE_MEMORY_DB: Path = BASE_DIR / "data" / "scene_memory.db"


@dataclass
class EmergencyConfig:
    """SOS and emergency contact settings."""

    # Fill these with your credentials; can also be set via env vars
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_FROM_NUMBER: str = os.getenv("TWILIO_FROM_NUMBER", "")
    EMERGENCY_PHONE: str = os.getenv("EMERGENCY_PHONE", "")

    # Email fallback
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = 587
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASS: str = os.getenv("SMTP_PASS", "")
    EMERGENCY_EMAIL: str = os.getenv("EMERGENCY_EMAIL", "")

    SOS_COOLDOWN_SECONDS: int = 60       # prevent duplicate SOS spam


@dataclass
class SystemConfig:
    """Runtime flags and system-level settings."""

    # ── Mock / Dev mode ───────────────────────────────────────────────────────
    MOCK_SENSORS: bool = False            # True = run on Windows/dev PC
    MOCK_GPS: bool = False
    MOCK_CAMERA: bool = False
    MOCK_MIC: bool = False                # False = use real microphone if available
    MOCK_TTS: bool = False                # False = speak physically using pyttsx3/SAPI on Windows
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"              # DEBUG / INFO / WARNING / ERROR
    LOG_FILE: Path = BASE_DIR / "logs" / "assistant.log"

    # ── Feature flags ─────────────────────────────────────────────────────────
    ENABLE_FACE_RECOGNITION: bool = True
    ENABLE_CURRENCY_DETECTION: bool = True
    ENABLE_GPS: bool = False
    ENABLE_ULTRASONIC: bool = True
    ENABLE_SCENE_MEMORY: bool = True
    ENABLE_SOS: bool = True
    ENABLE_OLLAMA_VISION: bool = True     # set False to use YOLO-only (faster)

    # ── Conversation settings ─────────────────────────────────────────────────
    HOTWORD: str = "guide"               # voice hotword to activate
    RESPONSE_MAX_TOKENS: int = 150       # keep TTS short and snappy
    DEDUP_COOLDOWN_SECONDS: float = 8.0  # min seconds between same alert
    IDLE_DESCRIPTION_INTERVAL: float = 10.0  # auto scene describe every Xs
    PROACTIVE_SAFETY_COOLDOWN: float = 6.0   # min seconds between proactive alerts

    # ── Paths ─────────────────────────────────────────────────────────────────
    MODELS_DIR: Path = BASE_DIR / "models"
    DATA_DIR: Path = BASE_DIR / "data"


@dataclass
class Config:
    """Master configuration — instantiate once and share across modules."""

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    navigation: NavigationConfig = field(default_factory=NavigationConfig)
    emergency: EmergencyConfig = field(default_factory=EmergencyConfig)
    system: SystemConfig = field(default_factory=SystemConfig)

    def __post_init__(self):
        """Create required directories on first run."""
        dirs = [
            self.system.MODELS_DIR,
            self.system.DATA_DIR,
            self.ai.FACE_KNOWN_DIR,
            self.navigation.OSM_CACHE_DIR,
            self.navigation.SCENE_MEMORY_DB.parent,
            self.system.LOG_FILE.parent,
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

    def apply_mock_mode(self):
        """Enable all mock flags for development on non-Pi hardware."""
        self.system.MOCK_SENSORS = True
        self.system.MOCK_GPS = True
        self.system.MOCK_CAMERA = True
        self.system.MOCK_MIC = False
        self.system.MOCK_TTS = False
        self.hardware.USE_PICAMERA2 = False
        self.system.ENABLE_ULTRASONIC = True   # ultrasonic mock still runs
        self.system.DEBUG = True
        self.system.LOG_LEVEL = "DEBUG"


# ─── Singleton accessor ───────────────────────────────────────────────────────
_config_instance: Config | None = None


def get_config(mock: bool = False) -> Config:
    """Return the global Config singleton, creating it if needed."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
        if mock:
            _config_instance.apply_mock_mode()
    return _config_instance
