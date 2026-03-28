#!/usr/bin/env bash
# =============================================================================
# Vast.ai Early Probe — validate environment before committing to full run
# =============================================================================
# Budget: < $0.10, < 15 minutes
# Exit 0 = all checks pass; Exit 1 = failures detected
# =============================================================================
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
LOG="$PROJECT_ROOT/probe.log"
PASS=0; FAIL=0

cd "$PROJECT_ROOT"
source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

check() {
    local name="$1"; shift
    echo -n "  [$name] "
    if "$@" >> "$LOG" 2>&1; then
        echo "PASS"; PASS=$((PASS+1))
    else
        echo "FAIL"; FAIL=$((FAIL+1))
    fi
}

echo "====================================================="
echo "  Healthcare Fraud GSPO Probe  |  $(date)"
echo "====================================================="
echo "  Project : $PROJECT_ROOT"
echo "  Log     : $LOG"
echo ""
> "$LOG"

# ----- Probe 1: Python imports -----
check "imports" python -c "
import torch, unsloth, trl, transformers, datasets, peft, accelerate
from trl.trainer.grpo_config import GRPOConfig
assert hasattr(GRPOConfig, '__dataclass_fields__') and 'importance_sampling_level' in GRPOConfig.__dataclass_fields__, 'No importance_sampling_level'
print('All imports OK')
"

# ----- Probe 2: CUDA + VRAM >= 20 GB -----
check "cuda_vram" python -c "
import torch
assert torch.cuda.is_available(), 'No CUDA'
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
assert vram >= 20, f'Only {vram:.1f} GB VRAM (need >= 20)'
print(f'CUDA OK: {torch.cuda.get_device_name(0)}, {vram:.1f} GB')
"

# ----- Probe 3: Environment runs 3 episodes -----
check "environment" python -c "
import sys; sys.path.insert(0, '.')
try:
    from environment.server.environment import ClaimsFraudEnvironment, EnvironmentConfig
except ImportError:
    from src.envs.healthcare_claims.server.environment import ClaimsFraudEnvironment, EnvironmentConfig
for seed in range(3):
    env = ClaimsFraudEnvironment(EnvironmentConfig(claims_per_episode=5, seed=seed))
    obs = env.reset()
    assert obs.prompt, 'Empty prompt'
    while not obs.done:
        from environment.models import ClaimAction
        obs = env.step(ClaimAction(response_text='Decision: APPROVE\nRationale: test\nEvidence: none'))
print('3 episodes OK')
"

# ----- Probe 4: Harvest 5 episodes to JSONL -----
check "harvest" python training/harvest_episodes.py \
    --episodes 5 --seed-start 999 \
    --output /tmp/probe_harvest.jsonl

check "harvest_format" python -c "
import json
lines = open('/tmp/probe_harvest.jsonl').readlines()
assert len(lines) > 0, 'Empty harvest output'
rec = json.loads(lines[0])
assert 'prompt' in rec and 'ground_truth' in rec, f'Bad record: {list(rec.keys())}'
print(f'Harvest OK: {len(lines)} records')
"

# ----- Probe 5: Model 4-bit load -----
check "model_load" python -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    'unsloth/Qwen2.5-1.5B-Instruct',
    max_seq_length=512, load_in_4bit=True, dtype=None,
)
print('4-bit load OK')
del model
import torch; torch.cuda.empty_cache()
"

# ----- Probe 6: GRPO 20-step training -----
# Use probe-safe memory budget: group-size 8, completion 256, batch 1.
# Full-scale defaults (group-size 12, completion 384, batch 2) are used in
# the actual training runs; probe only needs to confirm GRPO executes at all.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
check "grpo_probe" python training/train_gspo_v2.py \
    --cycle 1 \
    --data /tmp/probe_harvest.jsonl \
    --output-dir /tmp/probe_ckpt \
    --max-steps 20 \
    --group-size 8 \
    --max-completion-length 256 \
    --per-device-batch 1

check "rewards_logged" python -c "
import json, os
log = '/tmp/probe_ckpt/rewards_log.jsonl'
assert os.path.exists(log), f'Missing {log}'
lines = open(log).readlines()
assert len(lines) > 0, 'Empty rewards_log'
rec = json.loads(lines[0])
assert any(k for k in rec if 'reward' in k.lower()), f'No reward keys: {list(rec.keys())}'
print(f'Rewards log OK: {len(lines)} entries, keys={list(rec.keys())[:4]}')
"

echo ""
echo "====================================================="
echo "  Probe results: $PASS passed, $FAIL failed"
echo "====================================================="

if [ $FAIL -gt 0 ]; then
    echo ""
    echo "  Failures detected. Check $LOG for details."
    echo "  Do NOT proceed to full training run."
    exit 1
else
    echo ""
    echo "  All probes passed!"
    echo "  Estimated cost: < \$0.05"
    echo "  Safe to proceed with full training run."
    exit 0
fi
