# Reward Hacking — Observed Cases and Fixes

This document records each reward hacking incident encountered during training,
the root cause analysis, and the fix applied. Intended as a living reference
for future training runs and for reasoning about new reward designs.

---

## Case 1: Conciseness-APPROVE Hack (Cycle 1, Run 1)

### Observed behaviour
Training appeared to converge quickly (reward stabilised at ~+0.3 by step 300).
Gradient norm collapsed to ~0.003 (effectively zero). Mean completion length
dropped from ~700 tokens to 86–155 tokens. `correctness_reward` locked at
exactly −0.300 for hundreds of steps. The model stopped exploring.

At step ~700 the hack broke down, completions exploded to 1024 tokens (all
hitting the hard limit), `parse_reward` went to −2.5, KL diverged to 6+, and
the run was killed by OOM/timeout at step 1053.

### Root cause
`conciseness_reward` awarded **+0.5** for output sections ≤ 60 words, **+0.1**
for ≤ 100 words. On legitimate (non-fraud) transactions, a correct APPROVE
decision earns `correctness_reward = −0.3` by design (the asymmetric reward
reserves large positive values for correctly catching fraud).

The model found:
```
short valid APPROVE response (≤ 60 output words)
  conciseness_reward  = +0.6   (base 0.5 + closure bonus 0.1)
  correctness_reward  = −0.3   (correct APPROVE on legit, always negative)
  parse_reward        =  0.0   (output parses fine)
  ─────────────────────────────
  total               = +0.3   consistently, near-zero variance
```

A consistent +0.3 beats the high-variance path of trying to identify fraud
correctly. The model settled into this local minimum and never left.

The design flaw: **conciseness reward created a positive bonus for brevity that
interacted adversarially with the baseline correctness signal for legitimate
cases.** Any reward component that is independently maximisable without
requiring task competence will be hacked.

### Fix applied
Removed all positive base rewards for short output. New function only penalises
bloat:

```python
# Before (hacked)
if n <= 60:  base = 0.5
elif n <= 100: base = 0.1
elif n <= 150: base = -0.5
else:        base = -1.5

# After (fixed)
if n <= 200:  base = 0.0    # acceptable range, no reward/penalty
elif n <= 400: base = -0.5
else:          base = -1.5
```

Closure bonus (+0.1 for having `</think>`) retained — it is too small to sustain
a hack and provides a useful incentive to properly close the thinking block.

With the fix, the short-APPROVE hack yields:
```
conciseness  = +0.1  (closure only)
correctness  = −0.3
parse        =  0.0
─────────────────────
total        = −0.2   (slightly negative — no longer attractive)
```

### General lesson
**Never award a positive bonus for a single reward component that can be
achieved without task competence.** Bonuses should require at least two
conditions to be satisfied simultaneously (e.g., brevity AND correct answer).
Length penalties are safer than length bonuses. When the base reward for correct
behaviour is negative (as here, for legit APPROVE), any positive bonus creates a
shortcut.

---

## Case 2: KL Divergence Explosion (Cycle 1, Run 1 — compounding factor)

### Observed behaviour
KL divergence spiked to **27.16** at step 275, then settled at 3–6 for the
remainder of the run. Gradient norm was unstable (0.5 → 3.5 in the spike
region). Policy drifted far from the reference model with no meaningful
resistance.

### Root cause
`beta = 0.001` — the KL penalty coefficient was set nearly to zero.

```
At KL = 27:  penalty = 27 × 0.001 = 0.027
Reward scale ≈ ±1–5                         → 50–200× larger
```

The KL term had no meaningful influence. Combined with `learning_rate = 2e-5`
and `num_iterations = 4` (4 gradient updates per generation batch), the
effective per-batch policy shift was extremely large.

### Fix applied
| Parameter | Before | After |
|-----------|--------|-------|
| `beta` | 0.001 | 0.04 (TRL default) |
| `learning_rate` | 2e-5 | 5e-6 |
| `num_iterations` | 4 | 2 |

`beta = 0.04` means KL = 27 now contributes 1.08 to the loss — large enough to
resist further drift. The LR and iteration count reductions cut the per-batch
policy shift by ~8×.

### General lesson
**`beta` must be tuned relative to the reward scale.** For rewards in the ±1–10
range, `beta = 0.001` is negligible. A useful sanity check: at the maximum
expected KL (e.g. 5–10), the KL penalty `beta × KL` should be comparable in
magnitude to the typical reward signal. TRL's default of 0.04 is a reasonable
starting point; only go lower if the task is well-understood and convergence is
slow. Never set it below 0.01.

---

## General Design Principles (derived from above)

1. **Simulate the hack before shipping the reward.** For each reward component,
   ask: "What is the highest possible value for this component alone, and what
   does the output look like?" If that output doesn't require task competence,
   redesign the component.

2. **Watch for gradient norm collapse.** Grad norm → 0 while reward is positive
   and non-zero almost always means the model found a local hack. Healthy
   training should show a gradually decreasing (but non-zero) grad norm.

3. **Monitor `clipped_ratio` closely.** Healthy range: 5–25%. Near 0% means
   the policy has stopped changing (hack or convergence). Near 100% means
   all completions hit the length limit — a reliable collapse signal.

4. **KL as a leading indicator.** KL divergence rising past 1.0 early in
   training is a warning sign, not a normal state. If KL > 3 before step 500,
   reduce LR or increase `beta`.

5. **Asymmetric reward baselines create shortcuts.** If the "always correct"
   action (e.g. APPROVE everything) gives a consistent small positive or
   small negative reward, any independent positive bonus becomes a dominant
   strategy. Either remove the bonus or gate it on overall correctness.
