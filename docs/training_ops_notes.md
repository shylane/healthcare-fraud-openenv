# Training Operations Notes

Chronological log of dependency issues, bugs, configuration decisions, and
operational lessons encountered while running GRPO/GSPO training on vast.ai
GPU instances. Companion to `upstream_issues.md` (library-level bugs) and
`reward_hacking_lessons.md`.

---

## Environment

- **GPU**: NVIDIA RTX 3090 (24 GB VRAM) on vast.ai
- **Model**: `unsloth/Qwen3-1.7B` (4-bit NF4 quantisation via unsloth)
- **Framework**: TRL 0.29.1, Unsloth, HuggingFace Transformers
- **Training algorithm**: GRPO with `importance_sampling_level="sequence"` (GSPO variant)
- **SSH**: `ssh -i ~/.ssh/id_ed25519_vast -p <port> root@ssh8.vast.ai`

---

## Issue Log

### 1. `num_mini_batches` does not exist in TRL 0.29.1

**Symptom**: `TypeError: GRPOConfig.__init__() got an unexpected keyword argument 'num_mini_batches'`

**Cause**: TRL 0.29.1 replaced `num_mini_batches` with `num_iterations`. The parameter
controls how many gradient updates are performed per generation batch.

**Fix**: Use `num_iterations=N` in `GRPOConfig`. Verified by inspecting
`GRPOConfig` signature directly.

---

### 2. `max_seq_length` truncation with long prompts

**Symptom**: `Input IDs of shape [2, 2291] > model's max sequence length of 2048`
Completions were being silently truncated. `clipped_ratio` appeared at 83%
when it should have been ~6%.

**Cause**: Default `max_seq_length=2048` in `FastLanguageModel.from_pretrained`.
Prompts are ~1380 tokens; with `max_completion_length=1024`, total =
2404 > 2048.

**Fix**: Added `--max-seq-length` CLI arg to `train_gspo_v2.py` with dynamic default:
```python
max(2048, args.max_completion_length + 1600)
```
Standard launch uses `--max-seq-length 4096`.

---

### 3. SyntaxError from mid-statement patch injection

**Symptom**: `SyntaxError: invalid syntax. Perhaps you forgot a comma?`
Training script failed to import.

**Cause**: Heredoc-based SSH patch inserted new CLI arguments (`--lora-rank`,
`--num-iterations`) inside an existing `add_argument()` call definition,
breaking the statement mid-token.

**Fix**: Used a Python script (`/tmp/fix_train.py`) uploaded via SCP that
performed string replacement using regex with `re.DOTALL`, targeting the
complete argument block rather than inserting inline.

**Lesson**: Never patch Python source files via bash heredoc string injection.
Always use a dedicated Python patch script uploaded via SCP.

---

### 4. `AssertionError: RewardLogger.on_log not found` in patch script

**Symptom**: Patch script asserted the target string was present but failed.

**Cause**: Heredoc in the SSH command preserved `\n` as a literal backslash-n
rather than a newline, so the search string didn't match the actual source.

**Fix**: Switched patch script to use `re.compile(..., re.DOTALL)` regex matching
instead of literal string search. Uploaded via SCP rather than embedded heredoc.

---

### 5. nohup process dies silently (SSH session detachment)

**Symptom**: `nohup python ... > orchestrate.log 2>&1 &` runs but `orchestrate.log`
stays 0 bytes. Process shows alive for 5–10 seconds then disappears.

**Cause**: Race condition between SSH session teardown and nohup detachment.
When the SSH command completes too quickly, SIGHUP is delivered before nohup
can protect the child process.

**Fix 1**: Use `setsid nohup ... < /dev/null &` — creates a new session, making
the process immune to SIGHUP regardless of SSH teardown timing.

**Fix 2**: Write process to a launch script file, then execute it:
```bash
cat > /workspace/launch_train.sh << 'SCRIPT'
#!/bin/bash
export VARS...
exec python training/orchestrate_cycles_v2.py ...
SCRIPT
chmod +x /workspace/launch_train.sh
nohup /workspace/launch_train.sh < /dev/null > /workspace/orchestrate.log 2>&1 &
```
The `exec` replaces bash with python, ensuring clean signal handling.

**Note**: `crontab` is not available in the vast.ai minimal container. Use a
`while true; sleep 300; done` background daemon instead.

---

### 6. Orchestrator 6-hour timeout kills long cycles

**Symptom**: `[TIMEOUT] GSPO Train (cycle 1) exceeded 6.0h — killing.`
Training killed at step 1053/1500 despite being healthy at that point.

**Cause**: `run_command()` in `orchestrate_cycles_v2.py` had a hardcoded
`timeout_hours=6.0` for train phases. With `num_iterations=2` and
`max_steps=1500`, cycle 1 takes ~14–16h wall-clock.

**Fix**: Bumped `timeout_hours` for train phases from 6.0 → 16.0 in
`orchestrate_cycles_v2.py`. Both cycle-1 and cycle-N train calls updated.

**Rule of thumb**: Set timeout to 1.3× estimated wall-clock. Estimate:
`max_steps × avg_step_time_seconds / 3600` hours.

---

### 7. VRAM appears underutilised (~37% used)

**Observed**: 8.8 GB / 24 GB used on RTX 3090 with 1.7B model.

**Cause**: 4-bit NF4 quantisation dramatically reduces model weight memory
(~0.85 GB vs ~3.4 GB bf16). LoRA r=32 optimizer states are small (~400 MB).
KV cache at group_size=16, 1024 tokens is ~1.85 GB. 24 GB is overkill for 1.7B.

**Optimisation options (not yet applied)**:
- Disable `gradient_checkpointing`: saves ~1h/cycle, +2-3 GB VRAM (no quality change)
- Switch to full bf16: +30% gen speed, better gradient quality, ~17 GB VRAM required
- Increase `group_size` 16→32: better advantage estimation, ~+2 GB VRAM, neutral wall-clock

---

### 8. `beta=0.001` — near-zero KL penalty

**Symptom**: KL divergence hit **27.16** at step 275 of the first run.
Policy drifted far from reference with no resistance.

**Cause**: Someone set `beta=0.001` in `GRPOConfig` (vs TRL default of 0.04).
At reward scale ±1–5, `beta=0.001` contributes essentially nothing to the loss.

**Fix**: Restored to `beta=0.04`. Also reduced `learning_rate` 2e-5 → 5e-6
and `num_iterations` 4 → 2 to reduce per-batch policy shift.

---

### 9. `learning_rate=2e-5` default — too high for RL

**Cause**: Default `--lr 2e-5` in `train_gspo_v2.py` is appropriate for
supervised fine-tuning but too aggressive for RL. Combined with `num_iterations=4`,
effective per-batch update was ~8× larger than appropriate.

**Fix**: Reduced to `5e-6` and added `--lr` CLI arg to orchestrator.
Passed explicitly: `--lr 5e-6` in `launch_train.sh`.

---

## Hyperparameter Decisions Log

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lora_rank` | 32 | Increased from default 16 for more model capacity |
| `lora_alpha` | 64 | Standard 2× rank scaling |
| `beta` | 0.04 | TRL default; at reward scale ±5 gives meaningful KL penalty |
| `learning_rate` | 5e-6 | Conservative for RL; avoids KL explosion |
| `num_iterations` | 2 | Balanced: 2× more unique rollouts vs 4, freshter gradients |
| `group_size` | 16 | 16 completions per prompt for advantage estimation |
| `max_completion_length` | 1024 | Qwen3 thinking model needs 540–810 tokens naturally |
| `max_seq_length` | 4096 | Prompt (~1380) + completion (1024) + margin |
| `max_steps` | 1500 | ~20 dataset passes; enough for meaningful RL signal |
| `epsilon` | 0.2 | Standard GRPO clip ratio |
| `epsilon_high` | 0.28 | Asymmetric clip for GSPO |
| `warmup_ratio` | 0.1 | 150-step warmup for 1500 total |
| `save_steps` | 25 | Recovery point every ~14 min on spot instances |
| `gradient_checkpointing` | True | VRAM headroom exists to disable, not done yet |

---

## Monitoring Setup

### Server-side daemon (survives session close)
- Script: `/workspace/monitor_daemon_v2.sh`
- Interval: every 5 minutes
- Output: `/workspace/logs/monitor.log` (running log), `/workspace/logs/ALERT.log` (collapse/crash events)
- Push alerts: ntfy.sh channel `rlenv-training-daf26e6f`
  - Install ntfy app → subscribe to `rlenv-training-daf26e6f` for phone alerts
  - Fires: urgent push on collapse (clip=100%), crash (no python + GPU dead), hourly progress ping

### Collapse detection criteria
- `completions/clipped_ratio = 1.0` for 2+ consecutive logged steps → model outputting at max length, generation loop broken
- `kl > 5.0` → policy has drifted far from reference
- `grad_norm > 10` sustained → gradient explosion
- `rewards/parse_reward/mean < −1.0` sustained → output format broken

### Claude Code local cron
- Hourly check at :07 past the hour (session-only, backup to server daemon)
- See CronCreate job `c3ae85e0`

---

## Instance Management

```bash
# Check instance is alive
ssh -i ~/.ssh/id_ed25519_vast -o StrictHostKeyChecking=no -p 13566 root@ssh8.vast.ai "echo ok"

# Check training status
ssh ... "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; tail -1 /workspace/checkpoints_v2/cycle_*/rewards_log.jsonl"

# Restart training from scratch (wipe collapsed checkpoint, keep data)
ssh ... "pkill -9 -f 'train_gspo_v2|orchestrate_cycles_v2' 2>/dev/null"
ssh ... "rm -rf /workspace/checkpoints_v2/cycle_1; rm -f /workspace/logs/train_cycle1.log"
ssh ... "nohup /workspace/launch_train.sh < /dev/null > /workspace/orchestrate.log 2>&1 &"

# Pull updated training files to local
scp -i ~/.ssh/id_ed25519_vast -P 13566 \
  "root@ssh8.vast.ai:/workspace/training/train_gspo_v2.py" \
  "root@ssh8.vast.ai:/workspace/training/orchestrate_cycles_v2.py" \
  D:/Code/RLenv/training/
```
