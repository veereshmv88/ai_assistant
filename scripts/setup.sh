#!/bin/bash
# =============================================================================
# scripts/setup.sh — One-Command Raspberry Pi Setup
# =============================================================================
# Run this ONCE on a fresh Raspberry Pi OS (64-bit Bookworm) installation.
# Usage: chmod +x scripts/setup.sh && sudo ./scripts/setup.sh
#
# What it does:
#   1. Updates system packages
#   2. Installs system-level dependencies (libportaudio, espeak-ng, cmake, etc.)
#   3. Creates Python virtual environment
#   4. Installs Python packages (requirements.txt)
#   5. Installs Ollama + pulls required models
#   6. Downloads Vosk speech model
#   7. Downloads Piper TTS model
#   8. Enables UART for GPS module
#   9. Configures gpsd (optional)
#  10. Installs systemd service for auto-start on boot
# =============================================================================

set -euo pipefail   # exit on error, unset variable, pipe failure

# ── Colour output helpers ─────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Check we're on Raspberry Pi ───────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
info "Project directory: $PROJECT_DIR"

if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    warn "Not running on Raspberry Pi — some steps will be skipped."
    IS_PI=false
else
    IS_PI=true
    info "Raspberry Pi detected."
fi

# ── 1. System packages ────────────────────────────────────────────────────────
info "Updating system packages…"
apt-get update -qq
apt-get upgrade -y -qq

info "Installing system dependencies…"
apt-get install -y -qq \
    python3 python3-pip python3-venv python3-dev \
    build-essential cmake pkg-config \
    libportaudio2 portaudio19-dev \
    libopencv-dev \
    espeak-ng \
    libdlib-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libatlas-base-dev \
    libgstreamer1.0-dev \
    gpsd gpsd-clients \
    git curl wget unzip \
    i2c-tools \
    alsa-utils pulseaudio

success "System packages installed."

# ── 2. Python virtual environment ────────────────────────────────────────────
info "Creating Python virtual environment…"
cd "$PROJECT_DIR"
python3 -m venv venv --system-site-packages
source venv/bin/activate

info "Upgrading pip…"
pip install --upgrade pip setuptools wheel -q

# ── 3. Python packages ────────────────────────────────────────────────────────
info "Installing Python dependencies (this may take 10–20 minutes)…"

# Install PyTorch CPU-only first (avoid pulling GPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q

# Install remaining requirements
pip install -r requirements.txt -q

# Pi-specific packages
if [ "$IS_PI" = true ]; then
    pip install RPi.GPIO picamera2 -q || warn "RPi.GPIO/picamera2 install failed (may need manual install)"
fi

success "Python packages installed."

# ── 4. Ollama ─────────────────────────────────────────────────────────────────
info "Installing Ollama…"
if ! command -v ollama &>/dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    success "Ollama installed."
else
    success "Ollama already installed."
fi

# Start Ollama service
systemctl enable ollama --now 2>/dev/null || ollama serve &
sleep 3

# Pull models (see install_models.sh for details)
info "Pulling AI models (may take 20–60 minutes depending on internet speed)…"
bash "$SCRIPT_DIR/install_models.sh"

# ── 5. Vosk speech model ──────────────────────────────────────────────────────
info "Downloading Vosk English model (small, ~40 MB)…"
MODELS_DIR="$PROJECT_DIR/models"
mkdir -p "$MODELS_DIR"

VOSK_URL="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
VOSK_ZIP="$MODELS_DIR/vosk-model-small-en-us.zip"
VOSK_DIR="$MODELS_DIR/vosk-model-small-en-us"

if [ ! -d "$VOSK_DIR" ]; then
    wget -q --show-progress "$VOSK_URL" -O "$VOSK_ZIP"
    unzip -q "$VOSK_ZIP" -d "$MODELS_DIR"
    mv "$MODELS_DIR/vosk-model-small-en-us-0.15" "$VOSK_DIR" 2>/dev/null || true
    rm -f "$VOSK_ZIP"
    success "Vosk model downloaded."
else
    success "Vosk model already present."
fi

# ── 6. Piper TTS model ────────────────────────────────────────────────────────
info "Downloading Piper TTS binary and voice model…"
PIPER_VERSION="2023.11.14-2"
PIPER_ARCH="aarch64"   # Raspberry Pi 64-bit
PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_${PIPER_ARCH}.tar.gz"

if [ "$IS_PI" = false ]; then
    PIPER_ARCH="amd64"
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_${PIPER_ARCH}.tar.gz"
fi

if ! command -v piper &>/dev/null; then
    PIPER_TGZ="$MODELS_DIR/piper.tar.gz"
    wget -q --show-progress "$PIPER_URL" -O "$PIPER_TGZ"
    tar -xzf "$PIPER_TGZ" -C /usr/local/
    chmod +x /usr/local/piper/piper
    ln -sf /usr/local/piper/piper /usr/local/bin/piper
    rm -f "$PIPER_TGZ"
    success "Piper binary installed."
fi

# Download Piper voice model (en-US Lessac medium)
PIPER_MODEL_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
PIPER_CONFIG_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
PIPER_MODEL="$MODELS_DIR/en_US-lessac-medium.onnx"
PIPER_CONFIG="$MODELS_DIR/en_US-lessac-medium.onnx.json"

if [ ! -f "$PIPER_MODEL" ]; then
    wget -q --show-progress "$PIPER_MODEL_URL" -O "$PIPER_MODEL"
    wget -q --show-progress "$PIPER_CONFIG_URL" -O "$PIPER_CONFIG"
    success "Piper voice model downloaded."
else
    success "Piper voice model already present."
fi

# ── 7. UART for GPS ───────────────────────────────────────────────────────────
if [ "$IS_PI" = true ]; then
    info "Configuring UART for GPS module…"
    # Enable UART in /boot/firmware/config.txt
    CONFIG_FILE="/boot/firmware/config.txt"
    if ! grep -q "enable_uart=1" "$CONFIG_FILE"; then
        echo "enable_uart=1" >> "$CONFIG_FILE"
        success "UART enabled in $CONFIG_FILE"
    fi
    # Disable serial console (so GPS can use /dev/serial0)
    if grep -q "console=serial0" /boot/firmware/cmdline.txt; then
        sed -i 's/console=serial0,[0-9]* //' /boot/firmware/cmdline.txt
        success "Serial console disabled (freed for GPS)."
    fi
fi

# ── 8. Data directories ───────────────────────────────────────────────────────
info "Creating data directories…"
mkdir -p \
    "$PROJECT_DIR/data/known_faces" \
    "$PROJECT_DIR/data/map_cache" \
    "$PROJECT_DIR/data/currency_templates" \
    "$PROJECT_DIR/logs"

# Create mock test frame
python3 -c "
import cv2, numpy as np
frame = np.full((480, 640, 3), 100, dtype=np.uint8)
cv2.putText(frame, 'MOCK CAMERA', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
cv2.imwrite('$PROJECT_DIR/data/mock_frame.jpg', frame)
" 2>/dev/null || true

success "Data directories created."

# ── 9. Systemd service ────────────────────────────────────────────────────────
if [ "$IS_PI" = true ]; then
    info "Installing systemd service…"
    cp "$PROJECT_DIR/systemd/blind_assistant.service" /etc/systemd/system/
    # Update paths in service file
    sed -i "s|/opt/blind_assistant|$PROJECT_DIR|g" /etc/systemd/system/blind_assistant.service
    systemctl daemon-reload
    systemctl enable blind_assistant.service
    success "Systemd service installed and enabled (starts on boot)."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  ✅  Setup Complete!                              ║${NC}"
echo -e "${GREEN}║                                                   ║${NC}"
echo -e "${GREEN}║  Run health check:                                ║${NC}"
echo -e "${GREEN}║    source venv/bin/activate                       ║${NC}"
echo -e "${GREEN}║    python main.py --health-check                  ║${NC}"
echo -e "${GREEN}║                                                   ║${NC}"
echo -e "${GREEN}║  Start assistant:                                 ║${NC}"
echo -e "${GREEN}║    python main.py                                 ║${NC}"
echo -e "${GREEN}║                                                   ║${NC}"
echo -e "${GREEN}║  Dev mode (Windows/macOS):                        ║${NC}"
echo -e "${GREEN}║    python main.py --mock --debug                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$IS_PI" = true ]; then
    warn "REBOOT REQUIRED to apply UART and serial changes."
    warn "Run: sudo reboot"
fi
