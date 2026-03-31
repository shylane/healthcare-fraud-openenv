"""GSPO training script v2 — Interruption-safe, auto-resume, per-reward logging.

Targets: NVIDIA RTX 3090 (24 GB VRAM) on Vast.ai / RunPod spot instances.

New in v2 vs v1 (train_gspo.py):
  1. SIGTERM/SIGINT handler — saves checkpoint immediately before the process dies.
     Vital on Vast.ai where spot instances are killed with ~30s notice.
  2. Auto-resume — scans output_dir for the latest HuggingFace Trainer checkpoint
     (checkpoint-N folders) and resumes from there automatically.  No more manual
     --resume pointing; the script figures it out.
  3. Per-reward JSONL log — every logging_steps interval, individual reward
     function scores are written to rewards_log.jsonl so you can diagnose which
     reward is dominating or collapsing without needing W&B.
  4. Probe mode health checks — --max-steps 20 now prints a structured PASS/FAIL
     diagnostic after the probe, including reward NaN/zero detection.
  5. 3090-tuned defaults — num_generations 8→12, max_completion_length 256→384,
     save_steps 100→25, gradient_accumulation 16→8 (effective batch unchanged),
     per_device_train_batch_size 1→2.
  6. Optional W&B — pass --wandb to enable; disabled by default for first runs.
  7. Merge-before-train — when resuming from a PREVIOUS CYCLE's LoRA checkpoint
     (--base-cycle), the adapters are merged into base weights before adding a
     fresh LoRA.  Prevents ever-growing nested adapter stacks.

--- Revision History (this file) ---
  v2.0  Initial Vast.ai-hardened version.

Usage:
  # Probe run (30 steps, health check, exit):
  python scripts/train_gspo_v2.py --cycle 1 --data data/train_cycle1.jsonl --max-steps 30

  # Full Cycle 1 (fresh):
  python scripts/train_gspo_v2.py --cycle 1 --data data/train_cycle1.jsonl

  # Cycle 2 (merges Cycle 1 LoRA into base, adds fresh LoRA):
  python scripts/train_gspo_v2.py --cycle 2 --data data/train_cycle2.jsonl \\
      --base-cycle checkpoints_v2/cycle_1

  # Resume interrupted run (auto-detects last checkpoint inside output_dir):
  python scripts/train_gspo_v2.py --cycle 1 --data data/train_cycle1.jsonl --auto-resume

  # W&B enabled:
  python scripts/train_gspo_v2.py --cycle 1 --data data/train_cycle1.jsonl --wandb
"""

import argparse
import json
import os
import re
import signal
import sys
from datetime import datetime
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Patch torch.compile BEFORE any other import that might trigger it.
# Required for Python 3.13 / PyTorch 2.6 + Unsloth on 3090 (same as v1).
# ---------------------------------------------------------------------------
def _no_op_compile(model=None, *args, **kwargs):
    if model is not None:
        return model
    def decorator(fn):
        return fn
    return decorator

torch.compile = _no_op_compile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# CRITICAL: unsloth MUST be imported before trl so it monkey-patches GRPOTrainer.
from unsloth import FastLanguageModel  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint  # auto-resume helper


# ---------------------------------------------------------------------------
# Global trainer reference — set in main() so signal handler can reach it.
# ---------------------------------------------------------------------------
_TRAINER: "GRPOTrainer | None" = None
_INTERRUPTED = False


def _signal_handler(signum, frame):
    """On SIGTERM/SIGINT: save checkpoint immediately and exit cleanly."""
    global _INTERRUPTED
    if _INTERRUPTED:
        # Second signal — hard exit
        print("\n[SIGNAL] Second interrupt received — forcing exit.", flush=True)
        sys.exit(1)
    _INTERRUPTED = True
    sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n[SIGNAL] {sig_name} received — saving emergency checkpoint...", flush=True)
    if _TRAINER is not None:
        try:
            _TRAINER.save_model()
            _TRAINER._save_checkpoint(_TRAINER.model, trial=None)
            print("[SIGNAL] Emergency checkpoint saved.", flush=True)
        except Exception as e:
            print(f"[SIGNAL] Could not save checkpoint: {e}", flush=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Decision extraction helpers (identical to v1 for compatibility)
# ---------------------------------------------------------------------------
_DECISION_RE = re.compile(
    r"DECISION:\s*(INVESTIGATE|FLAG_REVIEW|DENY|APPROVE|REQUEST_INFO)",
    re.IGNORECASE,
)
_DECISION_KEYWORDS = ["INVESTIGATE", "FLAG_REVIEW", "DENY", "APPROVE", "REQUEST_INFO"]


def _to_text(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, list):
        return " ".join(msg.get("content", "") for msg in item if isinstance(msg, dict))
    return str(item)


def _extract_decision(text) -> str | None:
    text_str = _to_text(text)
    match = _DECISION_RE.search(text_str)
    if match:
        return match.group(1).upper()
    upper = text_str.upper()
    for d in _DECISION_KEYWORDS:
        if d in upper:
            return d
    return None


_NEGATION_RE = re.compile(
    r"\b(no|not|none|never|without|absent|negative|deny|denies|undetected|unlikely|"
    r"doesn't|don't|isn't|wasn't|weren't|hasn't|haven't|hadn't|"
    r"no\s+evidence|no\s+sign|no\s+indication)\b",
    re.IGNORECASE,
)


def _is_affirmative(text: str, pattern: str) -> bool:
    """Return True only if pattern is found WITHOUT a preceding negation.

    Splits on sentence boundaries and checks a ±60-char window around each
    match for negation words.  This prevents the model from farming bingo
    rewards by mentioning indicators in a denial context
    ("no prior fraud detected" should NOT count as hitting prior_fraud).
    """
    for sentence in re.split(r'[.!?;]|\n', text):
        m = re.search(pattern, sentence, re.IGNORECASE)
        if m:
            start = max(0, m.start() - 60)
            end = min(len(sentence), m.end() + 60)
            window = sentence[start:end]
            if not _NEGATION_RE.search(window):
                return True
    return False


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def correctness_reward(completions, **kwargs):
    """Asymmetric reward. Range: [-10, +5]. Legit-approve baseline: -0.3."""
    ground_truths = kwargs.get("ground_truth", [{}] * len(completions))
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        is_fraud = gt.get("is_fraud", False)
        decision = _extract_decision(completion)
        if decision is None:
            rewards.append(-5.0)
            continue
        if is_fraud:
            rewards.append(5.0 if decision in ("INVESTIGATE", "FLAG_REVIEW", "DENY") else -10.0)
        else:
            if decision == "APPROVE":
                rewards.append(-0.3)
            elif decision in ("FLAG_REVIEW", "INVESTIGATE"):
                rewards.append(-1.0)
            else:
                rewards.append(-2.0)
    return rewards


_FIELD_RE = re.compile(
    r"\b([A-Z]\d{2,}\.?\d*|CPT\s*\d{5}|ICD-?\d+|\d+\.?\d*\s*%|\$\s*[\d,]+\.?\d*)\b",
    re.IGNORECASE,
)


def reasoning_reward(completions, **kwargs):
    """Conditional on correctness. Range: [-1, +1].

    Scores evidence of real analysis:
      +0.3  structural RATIONALE:/EVIDENCE: present
      +0.3  provider_id referenced (specific, not gameable)
      +0.4  at least one CPT code, ICD code, or percentage/dollar figure cited
            (replaces raw amount echo which was trivially gameable)
      -1.0  completion too short (<20 words)
    """
    ground_truths = kwargs.get("ground_truth", [{}] * len(completions))
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        is_fraud = gt.get("is_fraud", False)
        decision = _extract_decision(completion)
        text = _to_text(completion)
        upper_text = text.upper()
        if decision is None:
            rewards.append(-1.0)
            continue
        is_correct = (
            (is_fraud and decision in ("INVESTIGATE", "FLAG_REVIEW", "DENY"))
            or (not is_fraud and decision == "APPROVE")
        )
        if not is_correct:
            rewards.append(0.0)
            continue
        score = 0.0
        if "RATIONALE:" in upper_text and "EVIDENCE:" in upper_text:
            score += 0.3
        provider_id = str(gt.get("provider_id", "")).upper()
        if provider_id and provider_id in upper_text:
            score += 0.3
        # Field reference: CPT/ICD code, percentage, or dollar figure
        # Much harder to game than echoing the raw claim_amount scalar.
        if _FIELD_RE.search(text):
            score += 0.4
        if len(text.split()) < 20:
            score -= 1.0
        rewards.append(max(-1.0, min(1.0, score)))
    return rewards


def budget_reward(completions, **kwargs):
    """Investigation efficiency. Range: [-0.5, +1]."""
    ground_truths = kwargs.get("ground_truth", [{}] * len(completions))
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        is_fraud = gt.get("is_fraud", False)
        decision = _extract_decision(completion)
        if decision == "INVESTIGATE":
            rewards.append(1.0 if is_fraud else -0.5)
        else:
            rewards.append(0.0)
    return rewards


def memory_reward(completions, **kwargs):
    """Provider memory utilization. Range: [-0.5, +0.5]."""
    prompts = kwargs.get("prompts", [""] * len(completions))
    rewards = []
    for completion, prompt in zip(completions, prompts):
        prompt_upper = _to_text(prompt).upper()
        decision = _extract_decision(completion)
        if "KNOWN PROVIDER" in prompt_upper:
            if decision == "INVESTIGATE":
                rewards.append(-0.5)
            elif decision in ("DENY", "FLAG_REVIEW"):
                rewards.append(0.5)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def conciseness_reward(completions, **kwargs):
    """Penalise bloat only -- no short-output reward (prevents short-APPROVE hack).
    Range: [-1.5, +0.1].

    Closure bonus (0.1) requires >50 think words — prevents gaming by closing an
    empty or trivial <think> block purely for the bonus token.
    Tightened output thresholds: 150/300/500 (was 200/400) to discourage
    over-verbose responses now that think_quality_reward rewards substantive reasoning.
    """
    rewards = []
    for completion in completions:
        text = _to_text(completion)
        decision = _extract_decision(completion)
        if decision is None:
            rewards.append(0.0)
            continue
        if "</think>" in text:
            think_section, output_section = text.split("</think>", 1)
            output_section = output_section.strip()
            # Closure bonus only when think section is substantive
            think_word_count = len(think_section.split())
            closure_bonus = 0.1 if think_word_count > 50 else 0.0
        else:
            output_section = text
            closure_bonus = 0.0
        n = len(output_section.split())
        if n <= 150:
            base = 0.0    # tight acceptable range
        elif n <= 300:
            base = -0.5
        elif n <= 500:
            base = -1.0
        else:
            base = -1.5
        rewards.append(base + closure_bonus)
    return rewards


def parse_reward(completions, **kwargs):
    """Format gate.  Range: [-5, +0.5].

    +0.5 for correct DECISION:<action> format — gives a positive pull so the
    model keeps the format even when correctness_reward is negative.
    0.0 is no longer emitted; the reward is always +0.5 or -5.0.
    Empirical data: parse_reward was always 0.0 during early healthy training,
    then collapsed to -3.6 as format was lost — the +0.5 incentive should
    maintain the format signal during periods of reward uncertainty.
    """
    rewards = []
    for c in completions:
        decision = _extract_decision(c)
        if decision is not None:
            rewards.append(0.5)   # rewarded for having any valid structured output
        else:
            rewards.append(-5.0)
    return rewards


def warmup_investigation_reward(completions, **kwargs):
    """Cold-start incentive for steps 0-14. Range: [-0.3, +0.8]."""
    ground_truths = kwargs.get("ground_truth", [{}] * len(completions))
    rewards = []
    for completion, gt in zip(completions, ground_truths):
        step = gt.get("step_number", 99)
        decision = _extract_decision(completion)
        if step < 15:
            if decision == "INVESTIGATE":
                rewards.append(0.8)
            elif decision == "APPROVE":
                rewards.append(-0.3)
            else:
                rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


def investigation_bonus_reward(completions, **kwargs):
    """Anti-approve-all at all steps. Range: [0, +0.5].

    DENY reduced to +0.2 (was +0.5): DENY is a terminal action with no
    further information gain; over-incentivising it pushes the model toward
    hard denials rather than nuanced FLAG_REVIEW/INVESTIGATE escalations.
    """
    rewards = []
    for completion in completions:
        decision = _extract_decision(completion)
        if decision == "INVESTIGATE":
            rewards.append(0.5)
        elif decision in ("FLAG_REVIEW", "REQUEST_INFO", "DENY"):
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    return rewards


_SPECIFICITY_RE = re.compile(
    r"(\d+\.?\d*\s*%|\$\s*[\d,]+\.?\d*"
    r"|above\s+\w+\s+threshold"
    r"|compared\s+to\s+(?:average|baseline|normal|typical|expected)"
    r"|(\d+\.?\d*)\s*(?:times|x)\s+(?:higher|lower|more|less)"
    r"|exceed[s]?\s+(?:benchmark|threshold|limit|average)"
    r"|(?:significantly|substantially)\s+(?:higher|lower|above|below))",
    re.IGNORECASE,
)
_FRAUD_SIGNAL_RE = re.compile(
    r"\b(suspicious|anomalous|unusual|irregular|inflated|fabricated|"
    r"duplicate|phantom|upcoded|unbundled|kickback|mismatched|inconsistent|"
    r"concerning|red.?flag|fraud(?:ulent)?|billing.?error)\b",
    re.IGNORECASE,
)
_LEGIT_SIGNAL_RE = re.compile(
    r"\b(normal|routine|consistent|within.?range|expected|typical|"
    r"appropriate|legitimate|standard|reasonable|compliant|no.?issue)\b",
    re.IGNORECASE,
)
_INVEST_INTENT_RE = re.compile(
    r"\b(need\s+to\s+(?:verify|check|confirm|investigate|audit)|"
    r"require\s+(?:further|additional)\s+(?:review|investigation|documentation)|"
    r"unclear|cannot\s+confirm|insufficient\s+(?:evidence|information))\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "this", "that", "these",
    "those", "it", "its", "i", "we", "you", "he", "she", "they", "their",
    "our", "my", "your", "his", "her", "which", "who", "what", "when",
    "where", "how", "why", "not", "no", "if", "so", "as", "than",
})


def think_quality_reward(completions, **kwargs):
    """Quality of reasoning inside <think> tags.  Range: [-0.3, +0.3].

    Four components:
      Trigram anti-repetition (content-words only, stopword-filtered)
        < 0.5 unique-trigram ratio → -0.3 (heavy repetition penalty)
        0.5–0.7 → -0.1 (mild)
        ≥ 0.7 → neutral (good diversity carries no bonus — absence of penalty suffices)
      Specificity: ≥2 quantitative/comparative phrases → +0.1; exactly 1 → +0.05
      Coherence: signal direction (fraud/legit keywords) must match decision
        aligned → +0.1; strongly misaligned (>2 opposing signals) → -0.15
      Investigation alignment: "need to verify" intent in think → INVESTIGATE/
        REQUEST_INFO/FLAG_REVIEW → +0.08; same intent + APPROVE → -0.08

    Think sections with <20 words are skipped (score = 0.0).
    """
    rewards = []
    for completion in completions:
        text = _to_text(completion)
        decision = _extract_decision(text)
        if decision is None:
            rewards.append(0.0)
            continue

        # Extract think content
        if "</think>" in text:
            think_content = text.split("</think>", 1)[0]
            if "<think>" in think_content:
                think_content = think_content.split("<think>", 1)[1]
        else:
            think_content = ""

        think_words = think_content.split()
        if len(think_words) < 20:
            rewards.append(0.0)
            continue

        score = 0.0

        # 1. Trigram anti-repetition (content-words only)
        content_words = [
            w.lower().strip(".,!?;:\"'") for w in think_words
            if w.lower().strip(".,!?;:\"'") not in _STOPWORDS and len(w) > 2
        ]
        if len(content_words) >= 3:
            trigrams = [
                (content_words[i], content_words[i + 1], content_words[i + 2])
                for i in range(len(content_words) - 2)
            ]
            ratio = len(set(trigrams)) / len(trigrams)
            if ratio < 0.5:
                score -= 0.3
            elif ratio < 0.7:
                score -= 0.1

        # 2. Specificity: quantitative / comparative phrases
        spec_count = len(_SPECIFICITY_RE.findall(think_content))
        if spec_count >= 2:
            score += 0.1
        elif spec_count == 1:
            score += 0.05

        # 3. Coherence: keyword direction vs decision
        fraud_n = len(_FRAUD_SIGNAL_RE.findall(think_content))
        legit_n = len(_LEGIT_SIGNAL_RE.findall(think_content))
        if decision in ("INVESTIGATE", "FLAG_REVIEW", "DENY"):
            if fraud_n > legit_n:
                score += 0.1
            elif legit_n > fraud_n + 2:
                score -= 0.15
        elif decision == "APPROVE":
            if legit_n > fraud_n:
                score += 0.1
            elif fraud_n > legit_n + 2:
                score -= 0.15

        # 4. Investigation alignment
        invest_intent = bool(_INVEST_INTENT_RE.search(think_content))
        if invest_intent and decision in ("INVESTIGATE", "REQUEST_INFO", "FLAG_REVIEW"):
            score += 0.08
        elif invest_intent and decision == "APPROVE":
            score -= 0.08

        rewards.append(max(-0.3, min(0.3, score)))
    return rewards


_BINGO_PATTERNS = {
    "prior_fraud":          r"\b(prior\s+fraud|previous\s+fraud|fraud\s+history|past\s+fraud|historical\s+fraud)\b",
    "high_denial_rate":     r"\b(denial\s+rate|claim\s+denial|high\s+denial|rejection\s+rate)\b",
    "weekend_claims":       r"\b(weekend\s+claim|weekend\s+billing|weekend\s+service|off.?hour)\b",
    "high_cost_ratio":      r"\b(high.?cost|cost\s+outlier|expensive\s+claim|inflated\s+billing)\b",
    "elevated_risk_score":  r"\b(risk\s+score|elevated\s+risk|high\s+risk\s+score|risk\s+indicator)\b",
    "amount_outlier":       r"\b(amount\s+(?:anomaly|outlier|unusual|z.?score)|billing\s+(?:outlier|anomaly)|unusual\s+amount)\b",
    "fast_filing":          r"\b(fast\s+filing|same.?day\s+(?:filing|submission|billing)|rapid\s+(?:filing|billing)|filing\s+speed)\b",
    "specialty_mismatch":   r"\b(specialty\s+mismatch|procedure\s+mismatch|specialty.?procedure\s+(?:conflict|inconsistency))\b",
    "high_member_risk":     r"\b(member\s+risk|patient\s+risk|beneficiary\s+risk|high.?risk\s+member|member\s+fraud\s+risk)\b",
}


def _parse_active_bingo_indicators(prompt_text: str) -> set:
    """Extract which bingo indicators are ACTIVE in this claim from the prompt markdown."""
    pt = _to_text(prompt_text)
    active = set()

    m = re.search(r"prior.?fraud.?(?:rate|pct|percentage)?[:\s]+(\d+\.?\d*)\s*%", pt, re.IGNORECASE)
    if m and float(m.group(1)) > 5.0:
        active.add("prior_fraud")

    m = re.search(r"denial.?rate[:\s]+(\d+\.?\d*)\s*%", pt, re.IGNORECASE)
    if m and float(m.group(1)) > 15.0:
        active.add("high_denial_rate")

    m = re.search(r"weekend.?claims?[:\s]+(\d+\.?\d*)\s*%", pt, re.IGNORECASE)
    if m and float(m.group(1)) > 25.0:
        active.add("weekend_claims")

    m = re.search(r"high.?cost.?(?:claims?|pct|percentage)?[:\s]+(\d+\.?\d*)\s*%", pt, re.IGNORECASE)
    if m and float(m.group(1)) > 30.0:
        active.add("high_cost_ratio")

    m = re.search(r"risk.?score[:\s]+(\d+\.?\d*)", pt, re.IGNORECASE)
    if m and float(m.group(1)) > 0.35:
        active.add("elevated_risk_score")

    m = re.search(r"(?:amount|billing).?z.?score[:\s]+(\d+\.?\d*)", pt, re.IGNORECASE)
    if m and float(m.group(1)) > 1.5:
        active.add("amount_outlier")
    m = re.search(r"(?:amount|billing).?z.?score[:\s]+(\d+\.?\d*)\s*%", pt, re.IGNORECASE)
    if m and float(m.group(1)) > 150.0:
        active.add("amount_outlier")

    m = re.search(r"(?:filing.?(?:speed|delay|lag)|days?.?to.?file)[:\s]+(\d+\.?\d*)\s*day", pt, re.IGNORECASE)
    if m and float(m.group(1)) <= 1.0:
        active.add("fast_filing")
    if re.search(r"same.?day\s+(?:filing|submission|billed)", pt, re.IGNORECASE):
        active.add("fast_filing")

    if re.search(r"specialty.?mismatch|procedure.?specialty.?(?:conflict|mismatch|inconsistency)", pt, re.IGNORECASE):
        active.add("specialty_mismatch")

    m = re.search(r"member.?(?:risk|fraud.?risk)[:\s]+(\d+\.?\d*)\s*%", pt, re.IGNORECASE)
    if m and float(m.group(1)) > 15.0:
        active.add("high_member_risk")

    return active


def bingo_reward(completions, prompts=None, **kwargs):
    """Healthcare-specific Risk Factor Bingo.  Range: [-0.20, +0.30].

    Parses 9 structural fraud indicators from the prompt markdown, then rewards
    the model for correctly identifying ACTIVE indicators in its response.
    Uses _is_affirmative() to prevent negation farming
    (e.g. "no prior fraud" should NOT earn the prior_fraud hit).

    Scoring:
      +0.08  per active indicator mentioned affirmatively  (correct identification)
      -0.05  per INACTIVE indicator mentioned affirmatively  (hallucination)
      Active but not mentioned: no penalty (model may prefer other evidence paths)
      Cap: +0.30 / -0.20

    Clean claims (zero active indicators) always return 0.0 — this reward is
    only meaningful when the prompt contains real risk signals.
    """
    if prompts is None:
        prompts = kwargs.get("prompts", [""] * len(completions))

    rewards = []
    for completion, prompt in zip(completions, prompts):
        text = _to_text(completion)
        active = _parse_active_bingo_indicators(prompt)

        if not active:
            rewards.append(0.0)
            continue

        score = 0.0
        for indicator, pattern in _BINGO_PATTERNS.items():
            is_active = indicator in active
            mentioned_pos = _is_affirmative(text, pattern)
            if is_active and mentioned_pos:
                score += 0.08
            elif not is_active and mentioned_pos:
                score -= 0.05  # hallucinated indicator

        rewards.append(max(-0.20, min(0.30, score)))
    return rewards


# ---------------------------------------------------------------------------
# Per-reward logging callback
# ---------------------------------------------------------------------------

class RewardLogger(TrainerCallback):
    """Writes per-reward-function breakdown to a JSONL file every logging_steps.

    The GRPOTrainer logs individual reward scalars as
    `rewards/{fn.__name__}` in its metrics dict.  This callback intercepts
    those and writes them to a side-car file for offline analysis without
    needing W&B.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Capture ALL TRL metrics: rewards, completion stats, timing, KL, clip ratios, etc.
        _SKIP = {"total_flos", "train_loss", "train_runtime",
                 "train_samples_per_second", "train_steps_per_second"}
        entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            "ts": datetime.utcnow().isoformat(),
        }
        for k, v in logs.items():
            if k not in _SKIP:
                try:
                    entry[k] = float(v) if not isinstance(v, (int, float, str, bool)) else v
                except (TypeError, ValueError):
                    entry[k] = str(v)
        has_reward = any(k.startswith(("rewards/", "reward", "completion", "kl", "clip", "step_time", "num_token"))
                        for k in entry)
        if has_reward or "loss" in entry:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

class CollapseEarlyStop(TrainerCallback):
    """Stop training early when the policy has collapsed to save compute.

    Three collapse modes detected:
    - Length explosion: clipped_ratio >= 0.99 for ``patience`` consecutive logs
      (all completions hitting max_completion_length — zero group variance)
    - Reward freeze: frac_reward_zero_std >= 1.0 for ``patience`` consecutive logs
      (every completion in each group has identical reward — nothing to learn)
    - KL explosion: kl >= kl_threshold for ``kl_patience`` consecutive logs
      (policy diverged irreversibly from reference — IS ratios numerically unstable)
      Cycle 2 hit KL=781 using dapo+seq IS; with luspo this should not occur,
      but this acts as a hard safety net.  Default threshold=4.0 catches drift
      before IS ratios become catastrophically large.

    When triggered, sets control.should_training_stop = True so the trainer
    exits cleanly (exit 0), preserving the last saved checkpoint.
    """

    def __init__(self, patience: int = 20, kl_threshold: float = 4.0, kl_patience: int = 5):
        # patience in log steps (logging_steps=5, so 20 logs = 100 training steps)
        self.patience = patience
        self.kl_threshold = kl_threshold
        self.kl_patience = kl_patience
        self._clip_streak = 0
        self._zero_std_streak = 0
        self._kl_streak = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        clip = logs.get("completions/clipped_ratio", 0.0)
        zero_std = logs.get("frac_reward_zero_std", 0.0)
        kl = logs.get("kl", 0.0) or 0.0

        # KL streak is independent of the other two
        if kl >= self.kl_threshold:
            self._kl_streak += 1
        else:
            self._kl_streak = 0

        if clip >= 0.99:
            self._clip_streak += 1
            self._zero_std_streak = 0
        elif zero_std >= 1.0:
            self._zero_std_streak += 1
            self._clip_streak = 0
        else:
            self._clip_streak = 0
            self._zero_std_streak = 0

        if self._clip_streak >= self.patience:
            print(f"\n[EARLY STOP] clipped_ratio=100% for {self._clip_streak} consecutive "
                  f"log steps (step {state.global_step}) — length explosion detected, stopping.")
            control.should_training_stop = True

        if self._zero_std_streak >= self.patience:
            print(f"\n[EARLY STOP] frac_reward_zero_std=1.0 for {self._zero_std_streak} "
                  f"consecutive log steps (step {state.global_step}) — zero learning signal, stopping.")
            control.should_training_stop = True

        if self._kl_streak >= self.kl_patience:
            print(f"\n[EARLY STOP] KL={kl:.2f} >= {self.kl_threshold} for {self._kl_streak} "
                  f"consecutive log steps (step {state.global_step}) — KL explosion, stopping.")
            control.should_training_stop = True


# ---------------------------------------------------------------------------
# Probe health check
# ---------------------------------------------------------------------------

def _run_probe_health_check(log_path: Path, max_steps: int) -> bool:
    """Reads the rewards_log.jsonl after a probe run and prints PASS/FAIL.

    Returns True if healthy, False if issues detected.
    """
    print("\n" + "=" * 55)
    print("  PROBE HEALTH CHECK")
    print("=" * 55)

    if not log_path.exists():
        print("  FAIL: rewards_log.jsonl not written — trainer never logged.")
        return False

    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not entries:
        print("  FAIL: rewards_log.jsonl is empty.")
        return False

    last = entries[-1]
    reward_keys = [k for k in last if k.startswith("rewards/")]

    checks = []

    # 1. At least one reward entry
    checks.append(("Reward log has entries", len(entries) > 0))

    # 2. Total reward is not NaN
    total_reward = last.get("reward", float("nan"))
    checks.append(("Total reward is finite", not (total_reward != total_reward)))  # NaN check

    # 3. Total reward is not exactly 0.0 (would mean all functions returned 0)
    checks.append(("Total reward is non-zero", total_reward != 0.0))

    # 4. Loss is present and finite
    loss = last.get("loss", float("nan"))
    checks.append(("Loss is finite", loss == loss and loss != float("inf")))  # GRPO loss is typically negative

    # 5. Individual reward functions all fired (non-NaN)
    fn_names = [
        "correctness_reward", "reasoning_reward", "budget_reward",
        "conciseness_reward", "parse_reward",
        "warmup_investigation_reward", "investigation_bonus_reward",
        "think_quality_reward", "bingo_reward",
    ]
    for fn in fn_names:
        # rewards are logged with /mean suffix in rewards_log.jsonl
        key = f"rewards/{fn}/mean"
        if key in last:
            val = last[key]
            checks.append((f"{fn} fired", val == val))  # NaN check

    # 6. correctness_reward not stuck at extreme
    corr_key = "rewards/correctness_reward/mean"
    if corr_key in last:
        corr = last[corr_key]
        # Stuck at -10 = always approving fraud. Stuck at 0 = no gradient.
        checks.append(("correctness_reward not stuck at -10", corr > -9.5))
        checks.append(("correctness_reward not stuck at 0", corr != 0.0))

    # 7. KL not exploding
    kl = last.get("kl", 0.0)
    checks.append(("KL divergence < 5.0", kl < 5.0))

    all_pass = True
    for name, ok in checks:
        status = "  PASS" if ok else "  FAIL"
        print(f"  {status}  {name}")
        if not ok:
            all_pass = False

    print("-" * 55)
    print(f"  Last step metrics:")
    for k in ["loss", "reward", "reward_std", "kl", "grad_norm"]:
        if k in last:
            print(f"    {k}: {last[k]:.4f}")
    for k in reward_keys:
        print(f"    {k}: {last[k]:.4f}")

    print("=" * 55)
    verdict = "ALL CHECKS PASSED — safe to run full cycle." if all_pass \
              else "FAILURES DETECTED — fix before spending GPU time."
    print(f"\n  VERDICT: {verdict}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Dataset loading (identical to v1 Rev 4)
# ---------------------------------------------------------------------------

def load_dataset(jsonl_path: str):
    from datasets import Dataset
    prompts, ground_truths = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            prompts.append([
                {
                    "role": "system",
                    "content": (
                        "You are a healthcare fraud detection agent. "
                        "Think step by step using <think>...</think> tags before giving your answer. "
                        "Keep your reasoning focused on specific claim evidence. "
                        "After </think>, output ONLY the structured decision block:\n"
                        "Decision: <ACTION>\nRationale: <1-2 sentences>\nEvidence: <data points>"
                    ),
                },
                {"role": "user", "content": record["prompt"]},
            ])
            gt = dict(record["ground_truth"])
            gt["step_number"] = record.get("step_number", 99)
            ground_truths.append(gt)
    return Dataset.from_dict({"prompt": prompts, "ground_truth": ground_truths})


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

_MODEL_CONFIGS: dict[str, dict] = {
    # ── Tier 1: Primary (proven on GRPO) ──────────────────────────────────

    # Current baseline — battle-tested on this codebase (Rev 1-4).
    # 4-bit, 5 GB VRAM for GRPO on 3090. Two EOS tokens required.
    "unsloth/Qwen2.5-1.5B-Instruct": {
        "load_in_4bit": True,
        "eos_token_ids": [151645, 151643],          # <|im_end|> + <|endoftext|>
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    },

    # ── Tier 1: Recommended upgrade ───────────────────────────────────────

    # Qwen3-1.7B (May 2025, DENSE transformer — NOT the same as Qwen3.5).
    # Drop-in replacement for Qwen2.5-1.5B: same tokenizer family, same
    # Unsloth GRPO code path, same LoRA targets.  Base model as capable as
    # Qwen2.5-3B.  Native <think>...</think> thinking mode gives GRPO a strong
    # chain-of-thought scaffold — conciseness_reward already handles the split.
    # Use --max-completion-length 768 and --group-size 8 with thinking enabled.
    # IMPORTANT: use temperature >= 0.6 (greedy causes repetition loops in think).
    # EOS tokens: 151645 (<|im_end|>), 151643 (<|endoftext|>), 151668 (</think>).
    # Full bf16 — faster inference (no dequant), better gradient quality.
    # Uses ~3.4 GB vs 0.85 GB for 4-bit; needs ~16 GB VRAM with group_size=24.
    "Qwen/Qwen3-1.7B": {
        "load_in_4bit": False,
        "dtype": "bfloat16",
        "eos_token_ids": [151645, 151643, 151668],
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    },
    "unsloth/Qwen3-1.7B": {
        "load_in_4bit": False,       # full precision — avoids 4bit quantization quality loss
        "dtype": None,               # auto: Unsloth picks bf16 on Ampere+, fp16 elsewhere
        "eos_token_ids": [151645, 151643, 151668],  # add </think> as EOS
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    },
    # Instruct alias (same weights, different tag format)
    "unsloth/Qwen3-1.7B-Instruct": {
        "load_in_4bit": True,
        "eos_token_ids": [151645, 151643, 151668],
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    },

    # ── Tier 2: Experimental probe ────────────────────────────────────────

    # LFM2.5-1.2B-Instruct (Liquid AI, non-transformer hybrid conv+attention).
    # Exceptional IFEval (86.23) but risky for GRPO: non-transformer gradient
    # flow is unvalidated for policy optimisation. Use probe-only (30 steps).
    # Needs bf16 (no 4-bit). Pass --max-completion-length 256 --group-size 6.
    "unsloth/LFM2.5-1.2B-Instruct": {
        "load_in_4bit": False,                      # needs bf16 — don't change
        "eos_token_ids": None,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj",
                         "in_proj", "w1", "w2", "w3"],
    },
    # Thinking variant of LFM2.5 (adds <think> traces, ~same VRAM)
    "LiquidAI/LFM2.5-1.2B-Thinking": {
        "load_in_4bit": False,
        "eos_token_ids": None,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj",
                         "in_proj", "w1", "w2", "w3"],
    },

    # ── DO NOT USE for GRPO (documented reasons) ──────────────────────────

    # Qwen3.5-2B (March 2026) — hybrid Gated DeltaNet + sparse MoE.
    # vLLM NOT supported (breaks TRL fast-inference GRPO), Gated DeltaNet
    # gradient flow unvalidated for RL.  Revisit mid-2026.
    # "unsloth/Qwen3.5-2B": { ... }  # blocked — vLLM issue #35391

    # SmolLM2-1.7B — 8K context hard-blocks long-rollout GRPO. Skip.
    # "unsloth/SmolLM2-1.7B-Instruct": { ... }
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _TRAINER

    # Ensure bf16 autocast to match Unsloth bf16 model weights —
    # default ACCELERATE_MIXED_PRECISION=fp16 causes Half/BFloat16 mismatch in LoRA matmul.
    import os as _os
    if _os.environ.get("ACCELERATE_MIXED_PRECISION", "fp16") == "fp16":
        _os.environ["ACCELERATE_MIXED_PRECISION"] = "bf16"

    parser = argparse.ArgumentParser(description="GSPO Training v2 — Vast.ai hardened")
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--base-cycle", type=str, default=None,
                        help="Path to PREVIOUS cycle LoRA checkpoint to merge before "
                             "adding a fresh LoRA for this cycle.  Use when starting a "
                             "new cycle on top of a completed one.")
    parser.add_argument("--output-dir", type=str, default="checkpoints_v2")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Auto-detect and resume from latest checkpoint in output_dir.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=12,
                        help="Completions per prompt (3090 default=12, was 8 on 4060).")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="0=full run. 30 for probe. 100 for smoke test.")
    parser.add_argument("--max-completion-length", type=int, default=512,
                        help="3090 default=512 (was 384). "
                             "Use 768 for thinking models (Qwen3).")
    parser.add_argument("--model-id", type=str,
                        default="unsloth/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--per-device-batch", type=int, default=2,
                        help="Per-device batch size (3090 default=2).")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (3090 default=8).")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (default 16; use 32 to double capacity).")
    parser.add_argument("--num-iterations", type=int, default=2,
                        help="GRPO num_iterations: gradient updates per generation (default 2).")
    parser.add_argument("--max-seq-length", type=int, default=0,
                        help="Model max sequence length (0 = auto: completion+1600, min 2048).")
    args = parser.parse_args()

    model_cfg = _MODEL_CONFIGS.get(args.model_id, {
        "load_in_4bit": True, "eos_token_ids": None, "lora_targets": "all-linear",
    })

    output_dir = f"{args.output_dir}/cycle_{args.cycle}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(output_dir) / "rewards_log.jsonl"

    # ── Print banner ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GSPO Training v2  |  Cycle {args.cycle}")
    print(f"{'='*60}")
    print(f"  Model            : {args.model_id}")
    print(f"  Data             : {args.data}")
    print(f"  Output           : {output_dir}")
    print(f"  Base cycle       : {args.base_cycle or 'None (fresh start)'}")
    print(f"  Auto-resume      : {args.auto_resume}")
    print(f"  Group size       : {args.group_size}")
    print(f"  Max completion   : {args.max_completion_length} tokens")
    print(f"  Batch            : {args.per_device_batch} x {args.grad_accum} "
          f"= {args.per_device_batch * args.grad_accum} effective")
    print(f"  W&B              : {'enabled' if args.wandb else 'disabled'}")
    if args.max_steps > 0:
        print(f"  *** PROBE MODE: stopping after {args.max_steps} steps ***")
    print(f"{'='*60}\n")

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = load_dataset(args.data)
    print(f"Loaded {len(dataset)} training samples.")

    # ── Model loading ─────────────────────────────────────────────────────
    # Determine whether we are:
    #   (A) fresh start from base model
    #   (B) resuming an interrupted run (auto-resume from checkpoint-N folder)
    #   (C) starting a new cycle on top of a previous cycle's LoRA

    resume_from_checkpoint = None

    if args.auto_resume:
        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt:
            print(f"[AUTO-RESUME] Found checkpoint: {last_ckpt}")
            resume_from_checkpoint = last_ckpt
        else:
            print("[AUTO-RESUME] No checkpoint found — starting fresh.")

    if args.base_cycle and not resume_from_checkpoint:
        # Case (C): Merge previous cycle LoRA into base weights, add fresh LoRA.
        # This prevents nested adapter stacking across cycles.
        print(f"[CYCLE INIT] Merging base cycle adapters from: {args.base_cycle}")
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_id = args.model_id
        print(f"  Loading base model: {base_id}")
        _base = AutoModelForCausalLM.from_pretrained(
            base_id, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        _base = PeftModel.from_pretrained(_base, args.base_cycle)
        print("  Merging LoRA into base weights...")
        _base = _base.merge_and_unload()

        # Save merged weights to a temp dir so FastLanguageModel can load them
        merged_dir = Path(output_dir) / "_merged_base"
        merged_dir.mkdir(exist_ok=True)
        _base.save_pretrained(str(merged_dir))
        AutoTokenizer.from_pretrained(base_id).save_pretrained(str(merged_dir))
        del _base
        torch.cuda.empty_cache()
        model_source = str(merged_dir)
        print(f"  Merged base saved to: {model_source}")
    else:
        model_source = args.model_id

    print(f"Loading model from: {model_source}")
    _dtype_str = model_cfg.get("dtype", None)
    _dtype = getattr(torch, _dtype_str) if isinstance(_dtype_str, str) else _dtype_str
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_source,
        load_in_4bit=model_cfg["load_in_4bit"],
        dtype=_dtype,
        max_seq_length=args.max_seq_length if args.max_seq_length > 0 else max(2048, args.max_completion_length + 1600),
    )

    # If resuming an interrupted run, do NOT add a new LoRA — the checkpoint
    # already has the LoRA state embedded.
    if not resume_from_checkpoint:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=model_cfg["lora_targets"],
            lora_dropout=0,
            bias="none",
        )
        print(f"Added fresh LoRA adapters (r={args.lora_rank}, alpha={args.lora_rank * 2}).")
    else:
        print("Resuming with existing LoRA from checkpoint.")

    # ── Training config ──────────────────────────────────────────────────
    # temperature=0.7 + top_p=0.95 is required for Qwen3 thinking mode —
    # greedy decoding causes repetition loops inside <think> blocks.
    # For Qwen2.5 (no thinking) these values are harmless but keep diverse rollouts.
    gen_kwargs: dict = {
        "repetition_penalty": 1.15,
        "temperature": 0.7,
        "top_p": 0.95,
        "do_sample": True,
    }
    # Build EOS list dynamically: start from config, add tokenizer's eos_token_id
    # and </answer> token if present to ensure generation stops cleanly.
    _eos_ids = list(model_cfg["eos_token_ids"] or [])
    if tokenizer.eos_token_id and tokenizer.eos_token_id not in _eos_ids:
        _eos_ids.append(tokenizer.eos_token_id)
    _ans_id = tokenizer.convert_tokens_to_ids("</answer>")
    if _ans_id != tokenizer.unk_token_id:
        _eos_ids.append(_ans_id)
    if _eos_ids:
        gen_kwargs["eos_token_id"] = _eos_ids

    training_config = GRPOConfig(
        output_dir=output_dir,
        importance_sampling_level="sequence",  # GSPO: sequence-level IS ratio
        # loss_type="luspo": NOT supported by Unsloth compiled trainer (raises ValueError).
        # Unsloth overrides compute_loss with its own implementation that supports:
        # grpo/bnpo/dr_grpo/dapo/cispo/sapo.  "dapo" is TRL's default and is fine here —
        # the key stability fix is best-checkpoint merge (not luspo vs dapo).
        loss_type="dapo",
        mask_truncated_completions=True,       # exclude clipped completions from gradient (reduces collapse noise)
        num_iterations=args.num_iterations,
        beta=0.04,  # was 0.001, near-zero KL caused policy diverge to KL=27
        epsilon=0.2,
        epsilon_high=0.28,
        num_generations=args.group_size,
        max_completion_length=args.max_completion_length,
        generation_kwargs=gen_kwargs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.epochs if args.max_steps == 0 else 1,
        # Save every 25 steps — critical for spot-instance survival.
        # On 3090 with ~94 total steps/cycle this gives ~4 recovery points.
        logging_steps=5,
        save_steps=25,
        save_total_limit=4,       # keep last 4 (covers ~100 steps of fallback)
        bf16=True,
        gradient_checkpointing=True,  # re-enabled: bf16 activations larger, need the ~2GB VRAM savings
        max_grad_norm=0.5,            # tightened from 1.0 for KL stability (cycle 1 restart)
        torch_compile=False,
        report_to="wandb" if args.wandb else "none",
        run_name=f"gspo-cycle{args.cycle}-{args.model_id.split('/')[-1]}",
    )

    if args.max_steps > 0:
        training_config.max_steps = args.max_steps
        training_config.save_steps = max(5, args.max_steps // 3)
        print(f"  Probe mode: save_steps set to {training_config.save_steps}")

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        reward_funcs=[
            correctness_reward,   # [-10, +5]   — primary signal
            reasoning_reward,     # [-1, +1]    — evidence quality
            budget_reward,        # [-0.5, +1]  — investigation efficiency
            # memory_reward removed: always 0.0 — "KNOWN PROVIDER" never in prompts
            conciseness_reward,   # [-1.5, +0.1] — output length gate
            parse_reward,         # [-5, +0.5]  — format gate (now has +0.5 positive pull)
            warmup_investigation_reward,  # [-0.3, +0.8] — cold-start INVESTIGATE
            investigation_bonus_reward,   # [0, +0.5]   — anti-approve-all
            think_quality_reward, # [-0.3, +0.3] — reasoning quality in <think>
            bingo_reward,         # [-0.20, +0.30] — healthcare fraud indicators
        ],
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[RewardLogger(log_path), CollapseEarlyStop(patience=20)],
    )

    # Store globally for signal handler access
    _TRAINER = trainer

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\nStarting training...  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"  Reward log      : {log_path}")
    print(f"  Checkpoint dir  : {output_dir}")
    print(f"  Emergency save  : SIGTERM/SIGINT will trigger save before exit\n")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # ── Save final LoRA adapters ──────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n[DONE] Cycle {args.cycle} complete.  Adapters saved to: {output_dir}")
    print(f"       Reward log: {log_path}")

    # ── Probe health check ────────────────────────────────────────────────
    if args.max_steps > 0:
        healthy = _run_probe_health_check(log_path, args.max_steps)
        sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
