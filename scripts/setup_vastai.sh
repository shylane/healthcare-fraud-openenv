#!/usr/bin/env bash
# =============================================================================
# Vast.ai Instance Setup — Healthcare Fraud GSPO Training
# =============================================================================
# Usage: bash scripts/setup_vastai.sh
#
# Tested on: RTX 3090 24GB, Ubuntu 22.04
# Expected runtime: 5-10 minutes (mostly downloading weights)
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Clone repo if not already present (run this from /workspace)
# ---------------------------------------------------------------------------
REPO_URL="https://github.com/shylane/healthcare-fraud-openenv.git"
if [ ! -d ".git" ]; then
    echo "[0/6] Cloning repo into current directory..."
    git clone "$REPO_URL" .
else
    echo "[0/6] Repo already present — pulling latest..."
    git pull --ff-only || echo "  (pull skipped — local changes present)"
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "=== Healthcare Fraud GSPO Setup ==="
echo "Project root: $PROJECT_ROOT"
echo "Date: $(date)"

# ---------------------------------------------------------------------------
# 1. Install uv (fast Python package manager)
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
    echo "[1/6] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/6] uv already installed: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# 2. Create venv with Python 3.12
# ---------------------------------------------------------------------------
VENV="$PROJECT_ROOT/.venv"
if [ ! -d "$VENV" ]; then
    echo "[2/6] Creating Python 3.12 venv at $VENV..."
    uv venv --python 3.12 "$VENV"
else
    echo "[2/6] Venv already exists at $VENV"
fi
source "$VENV/bin/activate"

# ---------------------------------------------------------------------------
# 3. Detect CUDA capability and install all training deps in one uv pass
# ---------------------------------------------------------------------------
echo "[3/6] Detecting CUDA driver and selecting wheel index..."

# Map driver major version → PyTorch CUDA wheel index
#   Driver >= 570  →  cu128  (CUDA 12.8+)
#   Driver >= 545  →  cu124  (CUDA 12.4-12.7)
#   Driver >= 525  →  cu121  (CUDA 12.1-12.3)
#   fallback       →  cu121
DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
    | head -1 | cut -d'.' -f1 || echo "0")
if   [ "${DRIVER_MAJOR:-0}" -ge 570 ]; then TORCH_INDEX="https://download.pytorch.org/whl/cu128"
elif [ "${DRIVER_MAJOR:-0}" -ge 545 ]; then TORCH_INDEX="https://download.pytorch.org/whl/cu124"
else                                         TORCH_INDEX="https://download.pytorch.org/whl/cu121"
fi
echo "  Driver major: ${DRIVER_MAJOR:-unknown}  →  wheel index: $TORCH_INDEX"

echo "[4/6] Installing Unsloth + TRL 0.29.1 + training deps..."
# Only hard pin: trl==0.29.1 (needs importance_sampling_level field).
# Everything else (torch, transformers, torchao, unsloth_zoo, peft, accelerate)
# resolved by uv from the detected torch wheel index — no manual version juggling.
uv pip install \
    "trl==0.29.1" \
    "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "datasets>=2.20.0" \
    "scipy>=1.13.0" \
    "numpy>=1.26.0" \
    --extra-index-url "$TORCH_INDEX"

# ---------------------------------------------------------------------------
# 5. Verify installation
# ---------------------------------------------------------------------------
echo "[5/6] Verifying installation..."
python - <<'PYEOF'
import sys
checks = []

try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    checks.append(f"torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {gpu}")
except Exception as e:
    checks.append(f"torch FAILED: {e}")

try:
    import unsloth
    checks.append(f"unsloth {unsloth.__version__}")
except Exception as e:
    checks.append(f"unsloth FAILED: {e}")

try:
    import trl
    from trl.trainer.grpo_config import GRPOConfig
    has_iss = "importance_sampling_level" in GRPOConfig.__dataclass_fields__
    checks.append(f"trl {trl.__version__} | importance_sampling_level: {has_iss}")
except Exception as e:
    checks.append(f"trl FAILED: {e}")

try:
    from transformers import AutoTokenizer
    import transformers
    checks.append(f"transformers {transformers.__version__} OK")
except Exception as e:
    checks.append(f"transformers FAILED: {e}")

for c in checks:
    print(f"  {c}")

fails = [c for c in checks if "FAILED" in c]
sys.exit(1 if fails else 0)
PYEOF

# ---------------------------------------------------------------------------
# 6. VRAM check
# ---------------------------------------------------------------------------
echo "[6/6] VRAM check..."
python -c "
import torch
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  VRAM: {vram:.1f} GB')
    if vram < 20:
        print('  WARNING: Less than 20 GB VRAM — reduce num_generations or batch size')
    else:
        print('  VRAM: sufficient for 3090 defaults')
else:
    print('  WARNING: No CUDA GPU detected!')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Run probe (< 15 min, < \$0.10):"
echo "     bash scripts/early_probe.sh"
echo ""
echo "  2. If probe passes, run full Cycle 1:"
echo "     python training/harvest_episodes.py --episodes 50 --seed-start 0 --output data/train_cycle1.jsonl"
echo "     python training/train_gspo_v2.py --cycle 1 --data data/train_cycle1.jsonl"
echo ""
echo "  3. Or use the orchestrator for all cycles:"
echo "     python training/orchestrate_cycles_v2.py --total-cycles 3"
echo ""
