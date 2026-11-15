#!/usr/bin/env bash
# =============================================================================
# setup.sh — AI Style Advisor: Full Setup from Scratch
# =============================================================================
# Usage:
#   chmod +x setup.sh
#   ./setup.sh [--skip-train] [--ollama]
#
# Flags:
#   --skip-train    Skip model fine-tuning (use if models are already trained)
#   --ollama        Use Ollama backend for LLM instead of local transformers
# =============================================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BLUE}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════${RESET}"; \
             echo -e "${BOLD}${BLUE}  $*${RESET}"; \
             echo -e "${BOLD}${BLUE}══════════════════════════════════════${RESET}\n"; }

# ── Argument parsing ─────────────────────────────────────────────────────────
SKIP_TRAIN=false
USE_OLLAMA=false

for arg in "$@"; do
  case $arg in
    --skip-train) SKIP_TRAIN=true ;;
    --ollama)     USE_OLLAMA=true  ;;
    *) error "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ── Pre-flight checks ─────────────────────────────────────────────────────────
header "Pre-flight Checks"

check_cmd() {
  if command -v "$1" &>/dev/null; then
    success "$1 found: $(command -v "$1")"
  else
    error "$1 is required but not installed."
    exit 1
  fi
}

check_cmd python3
check_cmd pip
check_cmd docker
check_cmd docker

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_MINOR=10

if python3 -c "import sys; exit(0 if sys.version_info >= (3, $REQUIRED_MINOR) else 1)"; then
  success "Python version: $PYTHON_VERSION"
else
  error "Python 3.$REQUIRED_MINOR+ required, found $PYTHON_VERSION"
  exit 1
fi

# ── Virtual environment ───────────────────────────────────────────────────────
header "Virtual Environment"

if [ ! -d ".venv" ]; then
  info "Creating virtual environment …"
  python3 -m venv .venv
  success "Virtual environment created at .venv/"
else
  info "Virtual environment already exists."
fi

# Activate
source .venv/bin/activate
success "Activated .venv"

# ── Install Python dependencies ───────────────────────────────────────────────
header "Installing Python Dependencies"

pip install --upgrade pip --quiet
pip install -r requirements.txt
success "Dependencies installed."

# ── Qdrant (Docker) ───────────────────────────────────────────────────────────
header "Starting Qdrant Vector Database"

cd docker
docker compose up -d
cd ..

# Wait for Qdrant to be healthy
info "Waiting for Qdrant to be ready …"
MAX_WAIT=30
WAITED=0
until curl -sf http://localhost:6333/healthz > /dev/null 2>&1; do
  sleep 2
  WAITED=$((WAITED + 2))
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    error "Qdrant did not start within ${MAX_WAIT}s."
    docker compose -f docker/docker-compose.yml logs qdrant
    exit 1
  fi
  info "  … still waiting (${WAITED}s)"
done
success "Qdrant is running at http://localhost:6333"

# ── Ingest sample trend documents ─────────────────────────────────────────────
header "Ingesting Fashion Trend Documents"

python3 pipeline/retriever.py bootstrap
success "Sample trend documents ingested."

# ── Generate seed training pairs ──────────────────────────────────────────────
header "Generating LLM Training Data"

python3 data/generate_seed_pairs.py
success "Seed pairs saved to data/training_pairs/seed_pairs.jsonl"

# ── Model training (optional) ─────────────────────────────────────────────────
if [ "$SKIP_TRAIN" = false ]; then

  # ── Vision Encoder ──
  header "Training Vision Encoder (DINOv2 + LoRA)"

  if [ ! -f "data/labeled/train.jsonl" ]; then
    warn "data/labeled/train.jsonl not found."
    warn "Skipping vision encoder training. See README.md Section 4 to label data."
  else
    python3 models/vision/train.py \
      --backbone facebook/dinov2-base \
      --train_jsonl data/labeled/train.jsonl \
      --val_jsonl   data/labeled/val.jsonl \
      --epochs 10 \
      --batch_size 32 \
      --checkpoint_dir models/vision/checkpoints
    success "Vision encoder trained. Checkpoint: models/vision/checkpoints/best.pt"
  fi

  # ── LLM Fine-tuning ──
  header "Fine-tuning LLM (Mistral-7B + QLoRA)"

  python3 models/llm/train.py
  success "LLM fine-tuned. Merged model at: models/llm/merged/"

else
  warn "--skip-train flag set. Skipping model training."
  warn "Ensure models exist at:"
  warn "  models/vision/checkpoints/best.pt"
  warn "  models/llm/merged/"
fi

# ── Ollama setup (optional) ───────────────────────────────────────────────────
if [ "$USE_OLLAMA" = true ]; then
  header "Setting Up Ollama Backend"

  if command -v ollama &>/dev/null; then
    info "Pulling Mistral model via Ollama …"
    ollama pull mistral
    success "Ollama Mistral model ready."
    export LLM_BACKEND=ollama
  else
    error "Ollama not installed. Install from https://ollama.ai then re-run with --ollama."
    exit 1
  fi
else
  export LLM_BACKEND=local
fi

# ── Start API server ──────────────────────────────────────────────────────────
header "Starting API Server"

export VISION_CHECKPOINT="models/vision/checkpoints/best.pt"
export LLM_MODEL_PATH="models/llm/merged"
export TOP_K_TRENDS=5

info "Starting FastAPI on http://localhost:8000 …"
info "Frontend UI available at: http://localhost:8000/docs (API docs)"
info "Serving frontend separately: cd frontend && python3 -m http.server 3000"
info ""
info "Press Ctrl+C to stop."
echo ""

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
