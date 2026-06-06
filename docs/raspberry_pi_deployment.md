# Raspberry Pi Deployment Guide — AI Blind Assistant

Complete step-by-step guide to deploy the AI Blind Assistant on a Raspberry Pi 5.

---

## Hardware Bill of Materials

| Component | Model | Est. Cost (INR) |
|-----------|-------|----------------|
| Single Board Computer | Raspberry Pi 5 (8GB RAM) | ₹9,500 |
| Storage | 64GB Class 10 microSD (or NVMe SSD) | ₹800 |
| Camera | Raspberry Pi Camera Module 3 (Wide) | ₹3,500 |
| Ultrasonic Sensor | HC-SR04 | ₹50 |
| GPS Module | NEO-6M with antenna | ₹350 |
| Microphone | USB MEMS microphone | ₹400 |
| Speaker | Portable USB speaker (5W) | ₹600 |
| SOS Button | Latching push button + enclosure | ₹100 |
| Power Bank | 20,000 mAh (USB-C PD, 27W) | ₹1,500 |
| Enclosure | Custom 3D printed / project box | ₹300 |
| Resistors | 1kΩ, 2kΩ (for voltage divider) | ₹20 |
| **Total** | | **~₹17,100** |

---

## OS Installation

### 1. Flash Raspberry Pi OS
```bash
# Download Raspberry Pi Imager from: https://www.raspberrypi.com/software/
# Select: Raspberry Pi OS (64-bit Bookworm)
# Enable SSH and set username/password in imager settings
```

### 2. First Boot Configuration
```bash
# SSH into the Pi:
ssh pi@raspberrypi.local

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Run raspi-config
sudo raspi-config
```

**Required raspi-config settings:**
- `Interface Options > Camera` → Enable
- `Interface Options > Serial Port` → No (login shell) → Yes (hardware enabled)
- `Interface Options > I2C` → Enable (if using I2S mic)
- `Performance Options > GPU Memory` → Set to 128 (for camera)
- Optionally: expand filesystem if using SD card

### 3. Enable USB Audio (if using USB speaker/mic)
```bash
# Check audio devices
aplay -l   # list playback devices
arecord -l # list recording devices

# Set USB audio as default (edit /etc/asound.conf):
sudo nano /etc/asound.conf
```
```
defaults.pcm.card 1
defaults.ctl.card 1
```

---

## Project Installation

### 4. Clone / Transfer Project
```bash
# Option A: Copy from development machine
scp -r /path/to/assistant/ pi@raspberrypi.local:/opt/blind_assistant/

# Option B: Git clone (if hosted on GitHub)
git clone https://github.com/your-org/blind-assistant /opt/blind_assistant
```

### 5. Run Setup Script
```bash
cd /opt/blind_assistant
sudo chmod +x scripts/setup.sh
sudo ./scripts/setup.sh
```

The setup script takes 20–60 minutes depending on internet speed.

### 6. Configure Secrets
```bash
# Create environment file for API credentials
nano /opt/blind_assistant/.env
```
```bash
# Twilio (for SMS SOS alerts)
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_FROM_NUMBER=+1234567890
EMERGENCY_PHONE=+919876543210

# Email fallback
SMTP_USER=youremail@gmail.com
SMTP_PASS=your_app_password     # Use Gmail App Password, not account password
EMERGENCY_EMAIL=guardian@email.com
```

---

## Running the Assistant

### Development / Test Mode
```bash
cd /opt/blind_assistant
source venv/bin/activate

# Health check first
python main.py --health-check

# Mock mode (no hardware needed — for testing on PC)
python main.py --mock --debug

# Production mode (Pi with all hardware)
python main.py
```

### Production Auto-Start (systemd)
```bash
# The setup script installs and enables the service automatically.
# Manual control:
sudo systemctl start blind_assistant     # start now
sudo systemctl stop blind_assistant      # stop
sudo systemctl status blind_assistant    # check status
sudo systemctl restart blind_assistant   # restart

# View live logs:
sudo journalctl -u blind_assistant -f
```

---

## Performance Tuning

### Raspberry Pi 5 Optimisations
```bash
# Enable performance CPU governor
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Add to /etc/rc.local for persistence:
echo 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor' | sudo tee -a /etc/rc.local

# Overclock (optional, with active cooling only!)
# Add to /boot/firmware/config.txt:
# arm_freq=2800
# over_voltage=6
```

### Reduce Ollama Model Memory Usage
```bash
# In config.py, switch to lighter models for faster response:
OLLAMA_VISION_MODEL = "moondream"    # 1.8GB instead of llava:7b (5GB)
OLLAMA_TEXT_MODEL   = "llama3.2:1b" # 1GB instead of llama3.2:3b (2GB)
```

### Camera Optimisation
```bash
# Reduce resolution in config.py for faster YOLO inference:
CAMERA_WIDTH  = 320   # instead of 640
CAMERA_HEIGHT = 240
YOLO_INPUT_SIZE = 160  # instead of 320
```

---

## Wearable Packaging

### Recommended Configuration
The system can be worn as:

1. **Vest configuration**: Pi + battery in vest pocket, camera on collar clip
2. **Backpack configuration**: All components in padded backpack
3. **Cane attachment**: Compact box clipped to white cane

### 3D Printable Enclosure
See `docs/enclosure/` for STL files (Raspberry Pi 5 + components box, ~10cm × 7cm × 4cm).

---

## Troubleshooting

### No audio output
```bash
# Check audio routing
aplay -l
amixer cset numid=3 1     # force 3.5mm
amixer cset numid=3 2     # force HDMI
amixer cset numid=3 0     # auto
```

### GPS no fix
```bash
# Check serial data coming in
sudo screen /dev/serial0 9600
# Should see NMEA sentences like: $GPGGA,... $GPRMC,...
# Allow 1-5 minutes outdoors for cold start fix
```

### Ollama not responding
```bash
sudo systemctl status ollama
sudo systemctl restart ollama
# Check model list:
ollama list
# Re-pull if corrupted:
ollama pull llava:7b
```

### Camera not detected
```bash
libcamera-hello --list-cameras
vcgencmd get_camera
# Should show: supported=1 detected=1
```

### Face recognition fails to install
```bash
# dlib requires cmake and build tools
sudo apt-get install -y cmake build-essential libopenblas-dev liblapack-dev
pip install dlib face-recognition
```

---

## Monitoring & Maintenance

```bash
# Check system temperature (should stay < 80°C)
vcgencmd measure_temp

# Monitor CPU/RAM usage
htop

# Check disk space
df -h

# View recent assistant logs
tail -f /opt/blind_assistant/logs/assistant.log

# Run diagnostics
python main.py --health-check
```
