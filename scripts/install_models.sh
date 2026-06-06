#!/bin/bash
# =============================================================================
# scripts/install_models.sh — Download AI Models for Ollama
# =============================================================================
# Run after Ollama is installed and running.
# Usage: bash scripts/install_models.sh
#
# Downloads:
#   • llava:7b      — multimodal vision+text (5 GB) — PRIMARY vision model
#   • moondream     — lightweight vision (1.8 GB)    — FAST vision fallback
#   • llama3.2:3b   — text-only reasoning (2 GB)     — intent + reasoning
# =============================================================================

set -euo pipefail
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# Check Ollama is running
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
    warn "Ollama server not responding. Starting it…"
    ollama serve &
    sleep 5
fi

# ── Model definitions ─────────────────────────────────────────────────────────
# Format: "model_name|description|size_gb|priority"
MODELS=(
    "llama3.2:3b|Fast text reasoning (intent, QA)|2GB|REQUIRED"
    "llava:7b|Multimodal vision + text (primary)|5GB|RECOMMENDED"
    "moondream|Lightweight vision model (fast fallback)|1.8GB|OPTIONAL"
)

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  AI Blind Assistant — Ollama Model Downloader"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Models to install:"
for model_spec in "${MODELS[@]}"; do
    IFS="|" read -r name desc size priority <<< "$model_spec"
    echo "  • $name ($size) — $desc [$priority]"
done
echo ""

# ── Download each model ───────────────────────────────────────────────────────
for model_spec in "${MODELS[@]}"; do
    IFS="|" read -r name desc size priority <<< "$model_spec"

    info "Pulling $name ($desc, ~$size)…"
    if ollama pull "$name"; then
        success "$name downloaded."
    else
        warn "Failed to download $name — skipping."
    fi
    echo ""
done

# ── Verify models ─────────────────────────────────────────────────────────────
echo "Installed models:"
ollama list

echo ""
success "Model installation complete!"
echo ""
echo "Quick test:"
echo "  ollama run llama3.2:3b 'Say hello in one sentence'"
echo "  ollama run llava:7b 'Describe this image' /path/to/image.jpg"
