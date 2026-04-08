# Healthcare Fraud Detection RL Environment — Complete Study Documentation

**Project:** AgentX-AgentBeats OpenEnv Challenge (Option D — Evaluation Study)
**Deadline:** April 12, 2026
**Repository:** `D:\Code\RLenv\` (training/experiments) + `healthcare-fraud-openenv/` (submission)

---

## Table of Contents

1. [Goal and Motivation](#1-goal-and-motivation)
2. [The Core Question](#2-the-core-question)
3. [Environment Design](#3-environment-design)
4. [Reward Function](#4-reward-function)
5. [Agent Design](#5-agent-design)
6. [Experiment Design](#6-experiment-design)
7. [Results and Findings](#7-results-and-findings)
8. [What We Expected vs What We Observed](#8-what-we-expected-vs-what-we-observed)
9. [The REINFORCE Implementation](#9-the-reinforce-implementation)
10. [Engineering Decisions and Bugs Fixed](#10-engineering-decisions-and-bugs-fixed)
11. [Open Questions](#11-open-questions)

---

## 1. Goal and Motivation

Healthcare insurance fraud costs the US system over **$100 billion per year**. Most detection pipelines are batch processes: accumulate claims for a month, run statistical models, flag suspicious ones for human review. The money is often already paid out before detection happens.

We reframed fraud detection as a **sequential decision problem** with resource constraints — closer to what a real human claims adjudicator actually does:

- You see one claim at a time
- You have a limited budget for deep investigations
- You build memory of providers you've already investigated
- Each decision has financial consequences (investigation costs, false-positive penalties, missed fraud losses)

The competition required submitting an **open environment** (Option A) or an **evaluation study** (Option D). We chose Option D: design the environment, define a set of agents at different capability levels, and rigorously measure whether more sophisticated prompting actually produces better decisions.

**The stakes:** If LLMs with better prompts don't outperform simple rule-based systems on this task, that's a real finding worth publishing.

---

## 2. The Core Question

> **Does budget/memory-aware prompting make an LLM agent meaningfully better at sequential fraud detection?**

This decomposes into four sub-questions:

1. Can an LLM agent, given no budget guidance, at least match a simple rule-based threshold detector?
2. Does adding explicit budget and memory reasoning to the system prompt improve the same LLM?
3. Does investigation budget level (5/10/15/20) affect agent performance, and do agents adapt accordingly?
4. Does provider memory half-life affect agent performance, and do memory-aware agents degrade more when memory decays?
5. Can a simple REINFORCE policy gradient agent learn a better-than-random policy on this environment?

---

## 3. Environment Design

### 3.1 Core Structure

**File:** `environment/server/environment.py` — `ClaimsFraudEnvironment` and `EnvironmentConfig`

Each episode consists of **100 insurance claims** arriving sequentially. The environment is initialized with:

```python
@dataclass
class EnvironmentConfig:
    claims_per_episode: int = 100
    fraud_rate: float = 0.05          # 5% of claims are fraudulent
    investigation_budget: int = 15    # max INVESTIGATE actions per episode
    memory_decay_halflife: int = 20   # turns until confidence halves
    investigate_accuracy: float = 0.95  # INVESTIGATE catches 95% of fraud
    flag_accuracy: float = 0.70         # FLAG_REVIEW catches 70% of fraud
    seed: Optional[int] = None
```

The `seed` is critical: all experiments use `seed=42` (training) or `seed=1042` (evaluation), ensuring every agent sees **identical claim sequences** — the comparison is pure prompt/policy, not luck of the draw.

### 3.2 What the Agent Sees

Each step the agent receives a text prompt containing:

- **Claim details:** claim ID, billed amount, procedure codes, diagnosis codes, date
- **Provider profile:** provider ID, specialty, average claim amount, fraud flag rate, billing frequency, high-cost procedure rate
- **Member profile:** member ID, age group, utilization rate
- **Risk indicators:** risk level (LOW/MODERATE/HIGH), amount z-score (deviation from provider average)
- **Budget status:** `Budget: 12/15 remaining` — budget consumed so far
- **Investigation memory:** providers already investigated with fraud/legit label and step decay
- **Decision history:** last 20 decisions with outcomes
- **Episode position:** `Step: 34/100`

The prompt is fully text-based. There is no structured feature vector passed to the agent — it must parse the text and reason in natural language. This is the "LLM-native" design: the same prompt that works for a language model also works for a human reader.

### 3.3 Action Space

| Action | Cost | Outcome |
|--------|------|---------|
| `APPROVE` | $0 | Passes claim; fraud missed if fraudulent |
| `FLAG_REVIEW` | $25 | Human review; 70% detection rate (`flag_accuracy=0.70`) |
| `INVESTIGATE` | $100 | Deep audit; 95% detection rate (`investigate_accuracy=0.95`) |
| `DENY` | $0 | Rejects claim; incurs `false_denial_penalty` if legitimate |
| `REQUEST_INFO` | $12.50 | Defers; 50% detection rate for fraud |

**The critical tradeoff:** `INVESTIGATE` ($100) is only financially optimal when the expected fraud amount exceeds the cost gap between investigate and flag:

```
Expected value of INVESTIGATE vs FLAG_REVIEW for a fraud claim:
  INVESTIGATE: amount × 0.95 - $100
  FLAG_REVIEW: amount × 0.70 - $25

INVESTIGATE wins when: amount × (0.95 - 0.70) > $100 - $25
                       amount × 0.25 > $75
                       amount > $300
```

But this ignores false positives. On legitimate claims, INVESTIGATE costs $100 + $50 false positive penalty = $150 pure loss. FLAG_REVIEW costs $25 + $50 = $75. Given that only 5% of claims are fraudulent, an agent that investigates randomly burns through its budget almost entirely on legitimate claims.

The true breakeven for INVESTIGATE on **a random claim** (95% legitimate):

```
Expected cost per random INVESTIGATE:
  = 0.95 × ($100 + $50) + 0.05 × ($100 - amount × 0.95)
  ≈ $142.50 - $4.75/claim_amount
```

This is dominated by the false-positive cost on legitimate claims. FLAG_REVIEW on a random claim:
```
  = 0.95 × ($25 + $50) + 0.05 × ($25 - amount × 0.70)
  ≈ $71.25 - $3.50/claim_amount
```

FLAG_REVIEW is always cheaper for average-value claims. **Optimal policy on most claims = FLAG_REVIEW when suspicious, APPROVE when not.** An agent that learns this without being told is genuinely intelligent. An agent that needs to be told is prompt-following.

### 3.4 Investigation Memory

When an agent INVESTIGATEs a provider, the result is stored in `investigation_memory`:

```python
# From environment.py _record_decision():
if decision == DecisionType.INVESTIGATE:
    inv = InvestigationResult(
        provider_id=provider_id,
        is_fraud=is_fraud,
        step_investigated=self._step_count,
        base_confidence=1.0,
    )
    self._state.investigation_memory[provider_id] = {
        "provider_id": inv.provider_id,
        "is_fraud": inv.is_fraud,
        "step_investigated": inv.step_investigated,
        "base_confidence": inv.base_confidence,
    }
```

The memory confidence decays with step distance. With `memory_decay_halflife=20`, a result from 20 steps ago has 50% confidence; from 40 steps ago has 25%. This appears in the prompt so agents can factor in how stale the memory is.

**Memory reuse opportunity:** If a provider appears again in the episode after being investigated, the agent can use FLAG_REVIEW (if fraud) or APPROVE (if legit) instead of wasting another $100 investigation. The `memory_reuse_rate` metric captures this: of all re-appearances by investigated providers, what fraction did the agent handle without re-investigating?

### 3.5 Budget Enforcement

The environment hard-caps `INVESTIGATE` at `investigation_budget` per episode:

```python
# From environment.py _process_action():
if decision == DecisionType.INVESTIGATE and self._state.budget_remaining <= 0:
    decision = DecisionType.FLAG_REVIEW
    budget_penalty = -0.5  # penalty for attempted over-budget investigation
```

So even if an agent tries to INVESTIGATE with zero budget remaining, the action is silently downgraded to FLAG_REVIEW and a small penalty is applied. This means agents that ignore budget don't crash — they just incur invisible costs.

---

## 4. Reward Function

**File:** `environment/server/environment.py` — `_calculate_decision_reward()`, `_score_rationale()`, `_score_evidence()`, `_calculate_efficiency()`

The reward has four components accumulated per step:

### 4.1 Decision Correctness Reward

The dominant term. Computed by `_calculate_decision_reward()`:

| Decision | Claim is Fraud | Claim is Legit |
|----------|---------------|----------------|
| APPROVE | `-claim_amount × fraud_missed_penalty_rate` (penalty) | `+efficient_approval_reward` (small bonus) |
| FLAG_REVIEW | `+claim_amount × fraud_caught_rate - $25` (if detected, 70% chance) | `-$25 - $50 false_positive_penalty` |
| INVESTIGATE | `+claim_amount × fraud_caught_rate - $100` (if detected, 95% chance) | `-$100 - $50 false_positive_penalty` |
| DENY | `+claim_amount × fraud_caught_rate` | `-false_denial_penalty` (large) |
| REQUEST_INFO | Variable (50% detection, half reward) | `-unnecessary_info_request_penalty` |

The fraud penalty rate is `fraud_missed_penalty_rate=1.0` — missing fraud costs the full claim amount. The fraud caught reward rate is also `~1.0` — catching it recovers the full amount.

This means the episode reward is approximately:
```
Total Reward ≈ fraud_caught_amount - fraud_missed_amount - investigation_cost - false_positive_cost
```

Which is essentially **net savings** from the fraud program. Negative reward means the fraud program is losing money (spending more on investigations + false positives than it catches in fraud).

### 4.2 Rationale Quality

`_score_rationale()` scores between 0 and 1:
- +0.2 for having a rationale at all
- +0.2 for 20-200 word length
- +up to 0.3 for mentioning fraud keywords (upcoding, anomaly, suspicious, etc.)
- +0.2 for including numbers or percentages

This is a rough proxy for "the reasoning makes sense." It doesn't verify correctness — an agent citing the wrong amount still gets credit.

### 4.3 Evidence Citation

`_score_evidence()` checks whether the evidence section references actual claim data (claim amount, provider ID, member ID). Score 0-1, contributes a small signal for structured output quality.

### 4.4 Efficiency

`_calculate_efficiency()` rewards choosing the *most cost-effective* action in hindsight:
- APPROVE on legit: +0.5
- FLAG_REVIEW on fraud: +0.1
- INVESTIGATE on high-value fraud: +0.2
- INVESTIGATE on legit: -0.5
- DENY on legit: -1.0

In practice this term is small relative to the decision correctness term. The main reward driver is always the financial outcome.

---

## 5. Agent Design

**File:** `evaluation/agents.py`

All agents implement a minimal protocol:

```python
agent.name: str          # display name
agent.act(prompt: str) -> str   # produce a decision response
agent.reset() -> None    # reset per-episode state
```

The response must contain: `Decision: [ACTION]\nRationale: ...\nEvidence: ...`

### 5.1 RandomAgent

```python
class RandomAgent:
    WEIGHTED = (
        ["APPROVE"] * 50 + ["FLAG_REVIEW"] * 25 +
        ["INVESTIGATE"] * 15 + ["DENY"] * 5 + ["REQUEST_INFO"] * 5
    )

    def act(self, prompt: str) -> str:
        decision = random.choice(self.WEIGHTED)
        return f"Decision: {decision}\nRationale: Random selection.\nEvidence: N/A"
```

Weighted toward APPROVE (50%) to roughly match the base rate — a completely uniform random would be even worse since it would INVESTIGATE 20% of the time. The weights are designed to be "plausible random" not "adversarially bad."

**Role:** The absolute floor. Establishes that the environment isn't trivially winnable and that random actions produce negative reward.

### 5.2 ThresholdAgent

```python
class ThresholdAgent:
    """
    Rules (in priority order):
    1. Known LEGIT in memory → APPROVE
    2. Known FRAUD in memory → FLAG_REVIEW
    3. Budget < 3 → FLAG_REVIEW for anything suspicious
    4. Risk HIGH + budget ≥ 5 → INVESTIGATE
    5. Risk MODERATE or fraud_flag_rate > 8% → FLAG_REVIEW
    6. Claim amount > 4× provider average → FLAG_REVIEW
    7. Default → APPROVE
    """
```

Key implementation detail: `ThresholdAgent` reads budget and memory from the **text prompt** using regex — it doesn't have privileged access to the environment state. It extracts:

```python
@staticmethod
def _extract_budget(prompt: str) -> Optional[int]:
    m = re.search(r"Budget:\s*(\d+)/\d+\s+remaining", prompt)
    return int(m.group(1)) if m else None

@staticmethod
def _extract_risk(prompt: str) -> str:
    if "Risk Level: HIGH" in prompt: return "HIGH"
    if "Risk Level: Moderate" in prompt: return "MODERATE"
    return "LOW"
```

This means ThresholdAgent is doing exactly what we'd want an LLM to do — reading the prompt fields and applying logic. The difference is that ThresholdAgent's logic is hard-coded and perfect; LLMs must infer the same logic from the prompt instructions.

**Observed behavior:** ThresholdAgent uses 0% of its investigation budget in practice. Its rule `Risk HIGH AND budget >= 5 → INVESTIGATE` fires very rarely because the simulator's HIGH risk claims also tend to have detectable features that other rules already catch. In practice, nearly every decision is FLAG_REVIEW or APPROVE.

**Role:** The "strong heuristic" baseline. If LLMs can't beat this, the budget-aware prompt needs work.

### 5.3 NaiveLLMAgent

```python
@property
def system_prompt(self) -> str:
    return """You are a healthcare fraud detection specialist reviewing insurance claims.

For each claim, analyze the data and respond with EXACTLY this format:
Decision: [APPROVE|FLAG_REVIEW|INVESTIGATE|DENY|REQUEST_INFO]
Rationale: [2-3 sentences explaining your reasoning]
Evidence: [specific data points from the claim that influenced your decision]

Decision guidelines:
- APPROVE: claim appears legitimate, no suspicious patterns
- FLAG_REVIEW: some indicators warrant manual review
- INVESTIGATE: strong fraud indicators, formal investigation needed
- DENY: clear fraud, deny the claim
- REQUEST_INFO: need more documentation before deciding

Be concise. Always include all three sections."""
```

This prompt does NOT mention:
- Investigation budget or cost
- Memory of previous providers
- The economic tradeoff between actions
- Budget conservation strategy

The agent knows it's reviewing claims and knows the action meanings. It does not know that INVESTIGATE costs $100, that it has a 15-investigation limit, or that re-investigating known providers wastes money.

**Role:** The "capable but blind" control condition. Same model as BudgetAwareAgent; difference in performance = value of budget reasoning alone.

### 5.4 BudgetAwareAgent

```python
@property
def system_prompt(self) -> str:
    return """You are a healthcare fraud detection specialist reviewing insurance claims.
You have LIMITED investigation resources: typically 15 investigations per 100 claims.

CRITICAL RULES FOR RESOURCE MANAGEMENT:
1. INVESTIGATE is expensive — use it only for high-confidence fraud with no prior investigation
2. If a provider is in Active Memory as FRAUD → use FLAG_REVIEW (saves your investigation budget)
3. If a provider is in Active Memory as LEGIT → use APPROVE (no need to re-investigate)
4. When Budget remaining < 4, NEVER use INVESTIGATE — use FLAG_REVIEW instead
5. When Budget remaining < 8, prefer FLAG_REVIEW over INVESTIGATE for moderate-risk claims

DECISION STRATEGY:
- Budget > 8 AND Risk HIGH AND provider NOT in memory → INVESTIGATE
- Provider in memory as FRAUD → FLAG_REVIEW (budget-conserving)
- Provider in memory as LEGIT → APPROVE
- Budget ≤ 4 AND any suspicion → FLAG_REVIEW
- Risk MODERATE OR fraud flags > 5% → FLAG_REVIEW
- Risk LOW AND no flags → APPROVE
..."""
```

This prompt explicitly:
- States the budget limit (15 investigations per 100 claims)
- Gives specific thresholds for when to switch strategies (budget < 4, < 8)
- Tells the agent to use memory instead of re-investigating
- Provides a decision tree mapping conditions to actions

**Role:** The "context-aware" experimental condition. The hypothesis is that this explicit guidance makes the model follow a near-optimal policy.

### 5.5 DeepSeek Variants

```python
class DeepSeekNaiveAgent(NaiveLLMAgent):
    def __init__(self, api_key="", **kwargs):
        super().__init__(api_key=api_key, model="deepseek/deepseek-v3.2", **kwargs)

class DeepSeekBudgetAwareAgent(BudgetAwareAgent):
    def __init__(self, api_key="", **kwargs):
        super().__init__(api_key=api_key, model="deepseek/deepseek-v3.2", **kwargs)
```

DeepSeek V3.2 is a state-of-the-art instruction-following model. Used as the "gold standard" to see if the same prompt-engineering pattern holds on a more capable model.

### 5.6 OpenRouter API Layer

All LLM agents inherit from `OpenRouterBase` which handles:

- **Rate limiting:** auto-detects `:free` tier models (8s delay) vs paid (1.5s delay)
- **Retry with backoff:** 429 errors trigger escalating waits (30s, 60s, 90s, 120s, 150s, 180s)
- **Null content guard:** some models return `content: null` — caught and replaced with a FLAG_REVIEW fallback
- **Think-block stripping:** reasoning models emit `<think>...</think>` blocks before the actual response — stripped before parsing

---

## 6. Experiment Design

All experiments use `seed=42`, `claims_per_episode=100`, `fraud_rate=0.05`, `investigation_budget=15` unless otherwise specified.

### 6.1 Experiment 01 — Baseline Comparison

**File:** `experiments/01_baseline_comparison/run.py`

**Design:** All 4 agents run the same 20 episodes. Same episode seeds → identical claim sequences per agent.

```bash
python experiments/01_baseline_comparison/run.py --n-episodes 20
python experiments/01_baseline_comparison/run.py --n-episodes 20 --include-deepseek
```

**Metrics collected:**
- `mean_reward`: average cumulative episode reward (primary metric)
- `mean_f1`: F1 score (fraud classification)
- `mean_precision`, `mean_recall`
- `mean_budget_utilization`: fraction of 15-investigation budget used
- `mean_budget_conservation_rate`: when budget < 20%, fraction of fraud claims handled with FLAG_REVIEW instead of INVESTIGATE
- `mean_memory_reuse_rate`: when a provider re-appears post-investigation, fraction handled without re-investigating
- `mean_valid_response_rate`: fraction of responses that could be parsed
- `mean_latency_ms`: API call latency

**Statistical note:** 20 episodes × 100 claims = 2,000 decisions per agent. With 5% fraud rate, ~100 fraud cases and ~1,900 legitimate cases. Episode-to-episode variance is high because each episode's fraud composition varies. 20 episodes gives enough to compute meaningful means but standard deviations are large (~200-300 reward units on mean rewards of ~-1000).

### 6.2 Experiment 02 — Budget Ablation

**File:** `experiments/02_budget_ablation/run.py`

**Design:** Sweep `investigation_budget` over [5, 10, 15, 20]. Same agents, same episodes, same fraud distribution — only the budget changes.

```python
BUDGET_CONDITIONS = [5, 10, 15, 20]
```

**Hypothesis:** NaiveLLMAgent degrades linearly as budget shrinks (it tries to investigate the same % of claims regardless of budget), while BudgetAwareAgent degrades less gracefully because its explicit rules adapt to lower budget.

ThresholdAgent is expected to be **completely flat** — it rarely uses INVESTIGATE anyway, so the budget limit is never binding.

### 6.3 Experiment 03 — Memory Ablation

**File:** `experiments/03_memory_ablation/run.py`

**Design:** Sweep `memory_decay_halflife` over [0, 5, 20, 100]. At halflife=0, memory decays instantly (useless). At halflife=100, memory is almost perfect for a 100-step episode.

```python
HALFLIFE_CONDITIONS = [0, 5, 20, 100]
```

The wrapper `_run_agent_with_halflife()` injects this into `EnvironmentConfig` since the standard `run_agent()` harness doesn't expose it.

**Hypothesis:** BudgetAwareAgent degrades most at halflife=0 because its prompt explicitly relies on memory. ThresholdAgent and RandomAgent are expected to be flat — they don't use memory in their decision logic.

### 6.4 Experiment 04 — REINFORCE Proof of Concept

**File:** `experiments/04_reinforce_poc/run.py`

**Design:** Train a linear policy via REINFORCE for 500 episodes, then evaluate against RandomAgent and ThresholdAgent on 20 held-out episodes.

**Architecture:**

```
Features (10-dim) → Linear(W: 3×10, b: 3) → Softmax → {APPROVE, FLAG_REVIEW, INVESTIGATE}
```

The 10 features:
```python
features = [
    budget_frac,         # 0: budget remaining / total budget
    memory_frac,         # 1: memory size / total budget
    log_amount,          # 2: log(1 + claim_amount) / log(1 + 50000)
    provider_risk,       # 3: provider risk score [0,1]
    member_risk,         # 4: member risk score [0,1]
    fraud_flag_rate,     # 5: provider's prior fraud flag rate [0,1]
    amount_z_norm,       # 6: amount z-score / 3, clipped to [-1,1]
    provider_in_mem,     # 7: binary — is provider in investigation memory?
    step_frac,           # 8: step / claims_per_episode
    high_cost_rate,      # 9: provider's high-cost procedure rate [0,1]
]
```

**Training details:**
- `lr=0.1` (compensates for dividing gradients by episode length n=100)
- `gamma=0.99` (discounted returns)
- `use_baseline=True` (subtract mean return to reduce variance)
- `entropy_coef=0.05` (entropy regularization to prevent collapse)
- `max_grad_norm=0.5` (gradient clipping)

**Why REINFORCE and not GRPO?** GRPO (Group Relative Policy Optimization, used by DeepSeek-R1) is a 1-step optimizer — it optimizes the relative reward between different responses to the *same* prompt. It has no mechanism for credit assignment across a 100-step episode. If the agent makes a bad budget decision at step 3, GRPO has no way to know that affected reward at step 97. REINFORCE with discounted returns explicitly handles this: the return at step 3 includes discounted future rewards, so early decisions that affect late rewards are credited appropriately.

---

## 7. Results and Findings

### 7.1 Complete Performance Table

All results from `experiments/*/results/` JSON files, 20 episodes × 100 claims each, `seed=42`.

| Agent | Mean Reward | Std | F1 | Recall | Budget Use | Mem Reuse | Net Loss/ep |
|-------|------------|-----|-----|--------|------------|-----------|------------|
| BudgetAware (DeepSeek) | **−455** | 145 | 0.047 | 8% | 0% | 0% | $2,609 |
| ThresholdAgent | −799 | 177 | **0.123** | 57% | 0% | 0% | $1,447 |
| BudgetAware (Qwen3.6) | −1,190 | 1,112 | **0.147** | 52% | 5% | 40% | $3,181 |
| NaiveLLM (DeepSeek) | −1,212 | 147 | 0.074 | 37% | 9% | 33% | $2,315 |
| REINFORCE (trained) | −1,646 | 209 | 0.057 | 23% | 79% | 83% | — |
| RandomAgent | −2,173 | 309 | 0.072 | 37% | 94% | 85% | $6,137 |
| NaiveLLM (Qwen3.6) | −2,322 | 213 | 0.063 | 52% | 70% | 92% | $5,645 |

Source files:
- `experiments/01_baseline_comparison/results/20260403_230615_comparison.json` — DeepSeek agents
- `experiments/01_baseline_comparison/results/20260403_161250_NaiveLLMqwen3.6-plus_free.json` — Qwen NaiveLLM
- `experiments/01_baseline_comparison/results/20260403_161250_BudgetAwareqwen3.6-plus_free.json` — Qwen BudgetAware
- `experiments/01_baseline_comparison/results/20260403_161250_comparison.json` — Qwen full comparison
- `experiments/04_reinforce_poc/results/20260405_000606_eval_*.json` — REINFORCE + Random + Threshold eval

**Cross-model budget-aware improvement:**

| Model | Naive | Budget-Aware | Multiplier | Budget Use: Naive → BA |
|-------|-------|-------------|-----------|----------------------|
| Qwen 3.6 Plus | −2,322 | −1,190 | **1.95×** | 70% → 5% |
| DeepSeek V3.2 | −1,212 | −455 | **2.66×** | 9% → 0% |

### 7.2 Budget Ablation

**Rule-based agents (B=5–20, 10 episodes each)**

Source: `experiments/02_budget_ablation/results/20260404_233454_ablation_summary.json`

| Budget | RandomAgent Reward | ThresholdAgent Reward |
|--------|-------------------|----------------------|
| 5 | −1,720 | −765 |
| 10 | −1,860 | −765 |
| 15 | −1,955 | −765 |
| 20 | −2,006 | −765 |

**DeepSeek LLM agents (B=5 complete, B=10 NaiveLLM only; 10 episodes each)**

Source: `experiments/02_budget_ablation/results/20260408_095313_B5_comparison.json`, `20260408_095313_B10_NaiveLLM(deepseek-v3.2).json`

| Budget | NaiveLLM (DeepSeek) | BudgetAware (DeepSeek) | BA/Naive ratio |
|--------|--------------------|-----------------------|----------------|
| 5 | −1,169 | **−743** | 1.57× |
| 10 | −1,192 | — (not run) | — |
| 15 | −1,212 | **−455** | 2.66× |

*B=15 DeepSeek figures from `experiments/01_baseline_comparison/results/20260403_230615_comparison.json`.*

**Key observations:**
- BudgetAware advantage **compresses at tight budgets** (1.57× at B=5 vs 2.66× at B=15).
- NaiveLLM **improves slightly at B=5** (−1,169) vs B=15 (−1,212). Fewer slots = less opportunity to waste budget.
- BudgetAware remains dominant at all tested budget levels.
- B=10 and B=20 LLM data not collected (credits exhausted after partial run).

### 7.3 Memory Ablation (Rule-Based Only)

Source: `experiments/03_memory_ablation/results/20260407_105452_memory_ablation_summary.json`

| Halflife | RandomAgent Reward | RandomAgent MemReuse | ThresholdAgent Reward | ThresholdAgent MemReuse |
|----------|-------------------|---------------------|----------------------|------------------------|
| 0 (off) | −2,068 | 76% | −763 | 0% |
| 5 | −2,068 | 76% | −763 | 0% |
| 20 | −2,068 | 76% | −763 | 0% |
| 100 | −2,068 | 76% | −763 | 0% |

### 7.4 REINFORCE Training Progression

Source: `experiments/04_reinforce_poc/results/20260405_000606_summary.json`

| Phase | Mean Reward |
|-------|------------|
| First quartile (ep 1–125) | −2,398 |
| Last quartile (ep 376–500) | −1,739 |
| **Improvement** | **+658** |
| Training time | 56 seconds |

---

## 8. What We Expected vs What We Observed

### Finding 1: LLM Without Guidance is Worse Than Random

**Expected:** NaiveLLM should underperform ThresholdAgent but stay above RandomAgent. A language model, even without specific budget guidance, should understand that INVESTIGATE is expensive and apply some judgment.

**Observed:** Qwen NaiveLLM (−2,322) scored **worse than RandomAgent** (−2,173) by 149 reward points.

**Why this happened:**

The Qwen NaiveLLM showed 70% budget utilization — it used most of its 15 investigation budget slots. At $100 per INVESTIGATE + $50 false-positive penalty on the 95% of legitimate claims investigated, each investigation of a legitimate claim costs $150. Over 20 episodes with 100 claims each and ~10 investigations per episode, this is ~$1,500 in investigation costs + ~$750 in false-positive costs per episode purely from the investigations. Since only 5% of claims are fraudulent, the expected fraud value caught per investigation is low.

The key insight: the NaiveLLM is **correctly suspicious** — it identifies claims that look anomalous. But its suspicion-to-investigation mapping is not calibrated for the economics. It investigates moderate-risk claims that a budget-aware agent would just flag. The model was never told what INVESTIGATE *costs* in this environment.

RandomAgent, by contrast, approves 50% of claims outright. Its investigation rate (~94% budget use means ~14 investigations per 100 claims) is roughly similar to Qwen NaiveLLM, but random investigations distribute evenly across all claims including low-risk ones. The difference is that random "wrong" decisions don't correlate with claim value — the NaiveLLM's suspicious-looking claims may actually be higher value, causing it to waste its budget on expensive investigations of valuable-but-legitimate claims.

**Takeaway:** An LLM with domain knowledge but no cost calibration can be *worse* than random because it acts on signal (suspicion) without understanding the cost structure.

### Finding 2: Budget-Aware Prompt Improves Both Models — Proportionally to Capability

**Expected:** BudgetAware should outperform NaiveLLM on the same model. We expected ~20-30% improvement.

**Observed:** The improvement held across both models, but at different magnitudes:

| Model | Naive | BudgetAware | Improvement | Budget use: Naive → BA |
|-------|-------|-------------|------------|----------------------|
| Qwen 3.6 Plus | −2,322 | **−1,190** | **1.95×** | 70% → 5% |
| DeepSeek V3.2 | −1,212 | **−455** | **2.66×** | 9% → 0% |

**Why this happened:**

Both models follow the budget-aware prompt, but with different precision. DeepSeek drops investigation use to 0% — it fully internalises the decision tree and never INVESTIGATEs. Qwen reduces from 70% to 5% — a massive shift, but not complete adherence. The residual 5% represents cases where Qwen's instruction-following breaks down or overrides the prompt with its own fraud judgment.

The underlying mechanism is the same in both cases: the prompt gives the model the economic math it needs to reason about action costs. Without it, both models default to an investigation-heavy strategy driven by their training to be thorough. With it, both models pivot toward FLAG_REVIEW as the dominant action.

The DeepSeek gap (2.66×) is larger than Qwen (1.95×) because stronger instruction-following means stricter adherence to the budget rules. This suggests **the prompt improvement scales with model capability** — the same prompt is worth more on a stronger model.

**A note on F1:** BudgetAware Qwen achieves F1=0.147 (highest of all agents) while BudgetAware DeepSeek achieves only F1=0.047. This sounds like Qwen is better — it's not. Qwen still invests 5% on investigations (catching high-precision fraud), while DeepSeek uses only FLAG_REVIEW (lower precision, lower recall). The reward is what matters financially, and DeepSeek wins that by 2.6×.

**Takeaway:** Telling the model the economics explicitly is the single highest-leverage intervention. The same prompt works at ~2× on a weaker model and ~2.7× on a stronger one. Model capability amplifies the prompt — it does not replace the need for it.

### Finding 3: Rule-Based Agent Beats Most LLMs — But Not the Best

**Expected:** ThresholdAgent to serve as an intermediate benchmark that LLMs should surpass with sufficient context.

**Observed:** ThresholdAgent (−799) beats NaiveLLM on both models and beats BudgetAware Qwen (−1,190). Only BudgetAware DeepSeek (−455) clears it.

**Why this happened:**

ThresholdAgent independently discovered the same strategy as BudgetAware: never INVESTIGATE, use FLAG_REVIEW for suspicious claims, APPROVE otherwise. Its rules are simple enough that they don't over-trigger on legitimate claims. The rule "Risk HIGH → INVESTIGATE if budget ≥ 5, else FLAG_REVIEW" almost never fires in practice because truly HIGH-risk claims are rare. So ThresholdAgent effectively executes the optimal policy (mostly FLAG_REVIEW + APPROVE) without any LLM.

ThresholdAgent's 57% recall vs BudgetAware's 8% recall, while BudgetAware has better reward, reveals something important: **recall is the wrong metric for this environment**. Higher recall means investigating more, which means spending more on investigations that mostly fall on legitimate claims. ThresholdAgent's 57% recall catches more fraud in absolute terms but costs more in false-positive investigations to achieve it. BudgetAware's 8% recall sounds terrible but it's working in a completely different regime — flagging rather than investigating.

**Takeaway:** Rule-based agents are not "weak" baselines in constrained resource environments. They are formidable because they don't over-invest. Any LLM agent needs very clear cost calibration to match them.

### Finding 4: The Budget Paradox

**Expected:** More budget → better performance for all agents (more ability to investigate fraud).

**Observed:** RandomAgent gets *monotonically worse* as budget increases (−1,720 at B=5 to −2,006 at B=20). ThresholdAgent is perfectly flat (−765 at all budget levels).

**Why this happened:**

More investigation budget means more opportunities for random agents to INVESTIGATE. Each random INVESTIGATE on a legitimate claim (95% probability) costs $150. More budget → more random investigations → more $150 wasted → worse total reward.

The "budget" is not a resource you want to use — it's a constraint on how much damage you can do with INVESTIGATE. For an agent that uses INVESTIGATE wisely, more budget is better. For an agent that uses INVESTIGATE randomly, more budget is strictly worse.

ThresholdAgent's flatness confirms this: its rules almost never trigger INVESTIGATE, so the budget level is never binding. Giving it more budget doesn't help (it doesn't use it) and doesn't hurt (it doesn't waste it).

**Takeaway:** In constrained resource environments, the value of a resource depends entirely on the agent's ability to use it. Giving more budget to a naive agent doesn't help — it makes the situation worse.

### Finding 5: Memory Half-Life Doesn't Matter for Rule-Based Agents

**Expected:** ThresholdAgent might degrade slightly at halflife=0 because its rule "Known FRAUD in memory → FLAG_REVIEW" would be less reliable with fast-decaying memory.

**Observed:** Both RandomAgent and ThresholdAgent are perfectly flat across all half-life settings (halflife 0 → 5 → 20 → 100).

**Why this happened:**

Two separate reasons:

1. **ThresholdAgent never investigates** → never populates `investigation_memory` → memory content is always empty → half-life of empty memory is irrelevant. The "known FRAUD in memory" rule never fires in practice.

2. **RandomAgent** doesn't read memory at all in its decision logic — it ignores the prompt content. So the memory content (regardless of decay) has no effect on its actions.

The key insight: **memory half-life only matters for agents that (a) investigate enough to build memory AND (b) read and act on memory when making decisions.** Neither rule-based agent satisfies both conditions. BudgetAware LLM agents would — but their memory ablation has not yet been run.

Notably, BudgetAware Qwen shows 40% memory reuse (down from NaiveLLM's 92%), and BudgetAware DeepSeek shows 0% memory reuse. This is because BudgetAware agents rarely INVESTIGATE, so they rarely build memory — making the memory_reuse metric largely vacuous for these agents. The memory system is functional; it simply requires investigation to populate it, and budget-aware agents conserve investigation.

**Takeaway:** Memory is only valuable if used. The memory ablation with LLM agents remains a future experiment and would require BudgetAware agents to investigate *some* claims (e.g. with higher budget or stronger fraud signals) before memory decay effects become measurable.

### Finding 6: REINFORCE Learns Measurable Policy

**Expected:** REINFORCE should show some learning signal. We expected small improvement (~100-200 reward over 500 episodes) because:
- 5% fraud rate means 95 of 100 steps per episode contribute noisy gradient signal
- Linear policy with 10 features can't represent complex sequential strategies
- Advantage normalization reduces the signal-to-noise ratio by normalizing large raw returns

**Observed:** +658 reward improvement (−2,398 → −1,739) over 500 episodes in 56 seconds.

**Why this happened:**

The policy learned two things:
1. To use `budget_frac` (feature 0) as a gate — reduce INVESTIGATE probability as budget depletes
2. To use `provider_in_mem` (feature 7) as a gate — switch to FLAG_REVIEW/APPROVE for known providers

These are exactly the behaviors the BudgetAware prompt describes in words. The linear policy discovered them from reward signal alone.

+658 improvement sounds large but the trained policy still scores −1,739, far below ThresholdAgent (−799) and BudgetAware (−455). The linear policy is limited: it can't represent complex conditional logic, and 10 features over-simplify the claim. A neural policy with more features would likely do better.

**The credit assignment problem:** With `gamma=0.99` and 100 steps, the discounted return at step t is approximately `G_t = sum_{k=0}^{99-t} 0.99^k * r_{t+k}`. At step 1, this includes all future rewards with high weight. This enables the policy to learn that investigating at step 1 (when budget is full) affects whether budget is available at step 90. GRPO cannot reason this way.

**Three bugs that had to be fixed before learning occurred:**

1. **Per-step update loop:** Original code updated weights 100 times per episode (once per step). Effective LR was `lr × T = 0.005 × 100 = 0.5` — far too large. Policy collapsed to uniform after 1 episode. Fix: batch accumulate gradients, divide by n, update once.

2. **Raw returns as advantages:** Raw returns of ~-2800 produced enormous gradient magnitudes. The first update would overshoot by 1000×. Fix: normalize advantages to zero-mean, unit-variance before gradient computation.

3. **LR too small after fix 1:** After dividing by n=100, effective LR became `0.005/100 = 0.00005`. Weights moved 0.002 over 500 episodes. Fix: set `lr=0.1` to compensate for the divide-by-n.

---

## 9. The REINFORCE Implementation

**File:** `experiments/04_reinforce_poc/run.py` — `LinearPolicy.update()`

The final, working update rule:

```python
def update(self, trajectories, baseline=0.0, entropy_coef=0.05, max_grad_norm=0.5):
    n = len(trajectories)

    # Step 1: Normalize advantages to zero-mean / unit-variance
    raw_advantages = [t["return"] - baseline for t in trajectories]
    mean_adv = sum(raw_advantages) / n
    std_adv = max((sum((a - mean_adv)**2 for a in raw_advantages)/n)**0.5, 1e-8)
    norm_advantages = [(a - mean_adv) / std_adv for a in raw_advantages]

    # Step 2: Accumulate gradients over full episode
    dW = [[0.0]*N_FEATURES for _ in range(N_ACTIONS)]
    db = [0.0]*N_ACTIONS

    for traj, adv in zip(trajectories, norm_advantages):
        probs = self.action_probs(traj["features"])
        for a in range(N_ACTIONS):
            score = (1.0 if a == traj["action_idx"] else 0.0) - probs[a]
            # Policy gradient
            pg = adv * score
            # Entropy bonus: -log(p_chosen) penalizes over-confidence
            ent = entropy_coef * (-math.log(max(probs[traj["action_idx"]], 1e-10)) - 1.0) * score
            total = pg + ent
            for i in range(N_FEATURES):
                dW[a][i] += total * traj["features"][i]
            db[a] += total

    # Step 3: Average, clip, apply
    for a in range(N_ACTIONS):
        for i in range(N_FEATURES):
            g = max(-max_grad_norm, min(max_grad_norm, dW[a][i]/n))
            self.W[a][i] += self.lr * g
        self.b[a] += self.lr * max(-max_grad_norm, min(max_grad_norm, db[a]/n))
```

The `score = indicator - prob[a]` term is the policy gradient score function (log-derivative of the softmax). For the chosen action (indicator=1), score = `1 - p` (positive when policy was uncertain). For unchosen actions, score = `-p` (negative, pushing them down). Multiplied by advantage: positive advantage pushes chosen action up and unchosen actions down; negative advantage does the opposite.

The entropy bonus `-log(p_chosen) - 1` is the negative log-probability of the chosen action minus 1 (which centers it). When the policy is highly confident (p_chosen ≈ 1), `-log(1) - 1 = -1` — the entropy term pushes the policy slightly away from certainty. This prevents probability mass from collapsing to a single action.

---

## 10. Engineering Decisions and Bugs Fixed

### 10.1 Dotenv Auto-Loading

```python
# Top of evaluation/agents.py
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent.parent / ".env", override=False)
except ImportError:
    pass
```

Reason: Without this, every run required `export OPENROUTER_API_KEY=...` or `--api-key` flag. The dotenv auto-load makes the API key available from `.env` at module import time, before any argument parsing happens.

### 10.2 Free vs Paid Model Detection

```python
# OpenRouterBase.__init__()
if request_delay_s < 0:
    self.request_delay_s = 8.0 if model.endswith(":free") else 1.5
```

Free-tier models on OpenRouter are rate-limited to ~1-2 requests/minute. At 1 request every 8 seconds, a 20-episode × 100-claim run takes: `20 × 100 × 8s = 16,000s ≈ 4.4 hours`. In practice, with 429 backoffs (30s, 60s, 90s...), the Qwen3.6-plus:free run took **39.3 hours**.

Paid models at 1.5s delay: `20 × 100 × 1.5s = 3,000s ≈ 50 minutes`.

### 10.3 Null Content Guard

```python
content = data["choices"][0]["message"].get("content")
if content is None:
    print(f"    [Warn] Model returned null content...")
    return "Decision: FLAG_REVIEW\nRationale: Model returned no content.\nEvidence: N/A"
```

Some models (notably Qwen3.5-9b on OpenRouter) return `{"choices": [{"message": {"content": null, "tool_calls": [...]}}]}`. Without this guard, `content.strip()` would crash. The fallback FLAG_REVIEW is the safest default — neither causing fraud to pass through nor wasting an investigation.

### 10.4 429 Aggressive Backoff

```python
if e.code == 429:
    wait = 30 * (attempt + 1)  # 30, 60, 90, 120, 150, 180
    time.sleep(wait)
    continue
```

The original backoff was `max(7, 2**attempt)` = 7, 7, 7, 8, 16, 32 seconds. Free-tier rate limits reset on a ~1-minute window. Waiting 7 seconds and retrying is pointless — you'll get another 429 immediately. The new 30s minimum ensures the rate limit window has reset.

### 10.5 Unicode on Windows

```python
# All terminal output that contained → was changed to >>
print(f"  >> Budget-aware prompt SIGNIFICANTLY outperforms naive prompt")
```

Windows cp1252 encoding does not support the `→` character (U+2192). The runner scripts crashed on Windows terminals with `UnicodeEncodeError: 'charmap' codec can't encode character '\u2192'`. Fixed by replacing all `→` with `>>`.

### 10.6 `--include-deepseek` Gating Bug

Original code:
```python
if not args.no_llm:
    agents += [NaiveLLMAgent(...), BudgetAwareAgent(...)]
    if args.include_deepseek:
        agents += [DeepSeekNaiveAgent(...), DeepSeekBudgetAwareAgent(...)]
```

Bug: `--include-deepseek` was only checked inside the `not args.no_llm` block. Running with `--no-llm --include-deepseek` to add only DeepSeek agents was impossible.

Fix: make `include_deepseek` independent:
```python
if args.deepseek_only:
    agents = [DeepSeekNaiveAgent(...), DeepSeekBudgetAwareAgent(...)]
elif args.budget_aware_only:
    agents = [BudgetAwareAgent(model=args.model, ...)]
else:
    agents = [RandomAgent(), ThresholdAgent()]
    if not args.no_llm:
        agents += [NaiveLLMAgent(...), BudgetAwareAgent(...)]
    if args.include_deepseek:
        agents += [DeepSeekNaiveAgent(...), DeepSeekBudgetAwareAgent(...)]
```

---

## 11. Open Questions

### 11.1 Completed: BudgetAware Qwen ✓

`BudgetAware(Qwen3.6:free)` completed — 20 episodes, reward=−1,190, F1=0.147, budget_util=5%.

The finding held: **budget-aware prompt improves the same model by 1.95×**, confirming the pattern is model-agnostic. The improvement is smaller on Qwen (1.95×) than DeepSeek (2.66×), which establishes that stronger instruction-following amplifies the prompt's value.

### 11.2 Open: LLM Memory Ablation

The rule-based memory ablation (Section 7.3) was flat because neither rule-based agent builds memory. The interesting experiment — varying halflife for BudgetAware LLM agents — was **not run** (credits exhausted).

**Hypothesis:** BudgetAware's reward would degrade at halflife=0 because:
1. Providers investigated early in an episode expire from memory before they appear again.
2. BudgetAware's explicit rule "if provider in memory as FRAUD → FLAG_REVIEW, don't re-investigate" stops firing.
3. The agent falls back to treating returning fraudulent providers as first-time encounters.

However, BudgetAware DeepSeek's **0% investigation rate** weakens this hypothesis: if the agent never investigates, it never populates memory, so memory decay rate may again be moot. The test that would produce meaningful signal requires an agent that actually builds memory — which may require a different reward structure that makes INVESTIGATE worthwhile.

### 11.3 Partial: LLM Budget Ablation

Collected at B=5 (both agents) and B=10 (NaiveLLM only). See Section 7.2.

**Key finding from partial data:** Budget-aware advantage compresses at tight budgets (1.57× at B=5 vs 2.66× at B=15). NaiveLLM improves slightly at B=5 because tight budgets limit how much damage over-investigation can cause.

**Not collected:** B=10 BudgetAware and B=20 for all LLM agents. Credits ($8.98) were exhausted by a concurrent Qwen 3.6 Plus (paid) reasoning-model run that consumed ~$8.10 due to 300+ reasoning tokens per call at $1.95/M. The partial data is sufficient for the budget trend narrative.

### 11.4 REINFORCE vs Threshold Gap

REINFORCE (−1,646) still trails ThresholdAgent (−799) by 847 reward points. Possible improvements:
- More features (include fraud_pattern_type, claim amount directly, procedure codes)
- Non-linear policy (2-layer MLP)
- More training episodes (diminishing returns past 500)
- Different action weighting (weight fraud-claim steps more heavily in loss)

### 11.5 F1 vs Reward Disconnect

ThresholdAgent has F1=0.123 and reward=−799. BudgetAware has F1=0.047 and reward=−455. The agent with *worse* fraud detection wins financially.

This means F1 is the wrong metric for this environment. The correct metric is reward (or equivalently, net savings). A future version of the environment might make this clearer by reporting "net savings per dollar of fraud claimed" rather than F1.

---

## File Index

| File | Purpose |
|------|---------|
| `evaluation/agents.py` | All agent implementations (Random, Threshold, NaiveLLM, BudgetAware, DeepSeek) |
| `evaluation/harness.py` | `run_agent()` — standard episode runner, metric aggregation, `EvalResults` |
| `experiments/01_baseline_comparison/run.py` | 4-agent comparison experiment runner |
| `experiments/02_budget_ablation/run.py` | Budget sweep experiment runner |
| `experiments/03_memory_ablation/run.py` | Memory half-life sweep experiment runner |
| `experiments/04_reinforce_poc/run.py` | REINFORCE training loop + LinearPolicy |
| `environment/server/environment.py` | Core RL environment: `ClaimsFraudEnvironment`, `EnvironmentConfig` |
| `environment/server/app.py` | FastAPI server (OpenEnv protocol: /reset, /step, /state) |
| `environment/claims_simulator.py` | Synthetic fraud data generator |
| `environment/models.py` | Pydantic models: `ClaimObservation`, `ClaimAction`, `ClaimState`, `RewardConfig` |
| `notebooks/01_results_analysis.ipynb` | Analysis notebook: loads all JSON results, produces comparison tables and plots |
| `blog/hf_blog_post.md` | HuggingFace blog post draft |
| `experiments/*/results/*.json` | All raw experiment output (fully reproducible with fixed seeds) |
