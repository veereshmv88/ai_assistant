# AI Blind Assistant

> **A fully autonomous, offline-first AI navigation guide for visually impaired users.**  
> Runs entirely on a Raspberry Pi 5. No cloud required. No internet needed (except SOS).

---

## What It Does

The assistant acts as an AI-powered replacement for visual perception — continuously monitoring the environment through camera, ultrasonic sensor, GPS, and microphone, then providing real-time spoken guidance through an earpiece.

| Capability | How |
|---|---|
| 🚶 Navigation guidance | "Walk forward", "Turn left in 20 metres", "Stop — obstacle 25cm ahead" |
| 👁️ Scene description | "Describe what's in front of me" → Florence-2 + LLaVA |
| 🔤 Text/sign reading | "Read this sign" → EasyOCR |
| 👤 Person recognition | Recognises enrolled faces by name |
| 🗺️ GPS navigation | "Guide me to the bus stop" → OSM routing |
| 💰 Currency detection | "What note is this?" → Template match + LLaVA |
| 🆘 Emergency SOS | "SOS" → SMS + email with GPS coordinates |
| 🧠 Memory | Remembers scenes, people, and signs seen earlier |
| 🎙️ Natural conversation | Full voice conversation via Whisper + Ollama |

---

## Quick Start (Development Mode — no Pi needed)

```bash
# Clone project
git clone https://github.com/your-org/blind-assistant
cd blind-assistant

# Install dependencies
pip install -r requirements.txt

# Run in mock mode (uses simulated sensors)
python main.py --mock --debug

# Run health check
python main.py --health-check
```

---

## Voice Commands

| Say... | Action |
|---|---|
| `"Guide, what is in front of me?"` | Full scene description |
| `"Guide, am I safe to walk?"` | Path safety check |
| `"Guide me to the hospital"` | Start GPS navigation |
| `"Read this sign"` | OCR + read aloud |
| `"Who is in front of me?"` | Face recognition |
| `"What note is this?"` | Currency identification |
| `"SOS"` / `"Call for help"` | Emergency alert |
| `"Remember this person as Mom"` | Enroll face |
| `"System status"` | Diagnostics report |
| `"Stop navigation"` | Cancel active route |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  AI BLIND ASSISTANT                      │
│                                                          │
│  Sensors → Fusion → Decision Engine → TTS → Speaker     │
│                                                          │
│  Camera ──────────┐                                      │
│  Ultrasonic ──────┼──► Decision Engine ──► Piper TTS    │
│  GPS ─────────────┤   (asyncio-based)    ──► eSpeak     │
│  Microphone ──────┘        │                            │
│       ↑                    ↓                            │
│  Vosk STT           YOLO Object Det.                    │
│  Whisper STT  ◄───► Florence-2 Scene                   │
│  Intent Parser      Ollama LLaVA/LLM                   │
│                     EasyOCR                             │
│                     Face Recognition                    │
│                     GPS Navigator (OSM)                 │
│                     SOS Handler (Twilio)                │
└─────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Vision (real-time) | YOLOv8-nano | Object detection @8 FPS |
| Vision (reasoning) | Florence-2 + Ollama LLaVA | Scene understanding |
| LLM | Ollama llama3.2:3b | Intent + conversational AI |
| STT | Vosk + faster-whisper | Speech recognition |
| TTS | Piper TTS + eSpeak | Natural voice output |
| OCR | EasyOCR | Sign/document reading |
| Face ID | face_recognition (dlib) | Person identification |
| GPS | pyserial + pynmea2 | Position tracking |
| Navigation | osmnx + geopy | Route calculation |
| Memory | SQLite | Scene history |
| SOS | Twilio + SMTP | Emergency alerts |
| Runtime | Python 3.11 asyncio | Concurrent event loops |
| Service | systemd | Auto-start on Pi boot |

---

## Project Structure

```
assistant/
├── main.py              # Entry point & asyncio orchestrator
├── config.py            # Central configuration
├── requirements.txt     # Python dependencies
│
├── sensors/
│   ├── camera.py        # Pi Camera / OpenCV / mock
│   ├── ultrasonic.py    # HC-SR04 distance sensor
│   └── gps_sensor.py    # NEO-6M GPS via serial
│
├── vision/
│   ├── detector.py      # YOLO-World object detection
│   ├── scene_analyzer.py  # Florence-2 + Ollama LLaVA
│   ├── face_recognizer.py # Face recognition
│   └── currency_detector.py # Currency identification
│
├── ai/
│   ├── llm_engine.py    # Ollama LLM client
│   ├── intent_parser.py # Voice intent classification
│   └── decision_engine.py # Sensor fusion + dispatch
│
├── stt/
│   └── whisper_stt.py   # Vosk + faster-whisper STT
│
├── tts/
│   └── piper_tts.py     # Piper TTS + eSpeak fallback
│
├── ocr/
│   └── text_reader.py   # EasyOCR text extraction
│
├── audio/
│   ├── mic_stream.py    # PyAudio microphone stream
│   └── speaker.py       # Priority audio output
│
├── navigation/
│   ├── gps_navigator.py # GPS routing (OSMnx)
│   └── scene_memory.py  # SQLite scene history
│
├── emergency/
│   └── sos_handler.py   # SOS alerts (Twilio + email)
│
├── utils/
│   ├── logger.py        # Rich console + file logger
│   ├── helpers.py       # Shared utilities
│   └── health_check.py  # System diagnostics
│
├── scripts/
│   ├── setup.sh         # One-command Pi setup
│   └── install_models.sh # Ollama model downloader
│
├── systemd/
│   └── blind_assistant.service # Auto-start on boot
│
└── docs/
    ├── hardware_wiring.md        # GPIO pin reference
    └── raspberry_pi_deployment.md # Full deployment guide
```

---

## Configuration

Edit `config.py` to customise:

```python
# Swap to faster/lighter models on Pi 4:
OLLAMA_VISION_MODEL = "moondream"    # instead of "llava:7b"
OLLAMA_TEXT_MODEL   = "llama3.2:1b"

# Disable features to save resources:
ENABLE_FACE_RECOGNITION  = False
ENABLE_CURRENCY_DETECTION = False
ENABLE_OLLAMA_VISION     = False  # YOLO-only mode

# Adjust alert distances:
ULTRASONIC_CRITICAL_CM = 30.0  # STOP threshold
ULTRASONIC_WARNING_CM  = 100.0 # CAUTION threshold

# Emergency contacts (or use .env file):
EMERGENCY_PHONE = "+919876543210"
EMERGENCY_EMAIL = "guardian@example.com"
```

---

## Hardware Required

- Raspberry Pi 5 (8GB) + 64GB SD card
- Pi Camera Module 3
- HC-SR04 Ultrasonic Sensor
- NEO-6M GPS Module
- USB Microphone
- USB Speaker / Earphones
- 20,000 mAh power bank (USB-C PD)
- Optional: SOS push button

See [docs/hardware_wiring.md](docs/hardware_wiring.md) for wiring details.  
See [docs/raspberry_pi_deployment.md](docs/raspberry_pi_deployment.md) for full setup guide.

---

## License

MIT License — Free for personal and educational use.

---

## Contributing

Pull requests welcome. Please open an issue first for major changes.
