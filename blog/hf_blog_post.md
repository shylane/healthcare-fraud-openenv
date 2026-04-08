# When LLMs Lose to Coin Flips: Building a Healthcare Fraud Detection RL Environment

*A rigorous evaluation study — across two models, six agent types, and 14,000 claim decisions — showing that prompt engineering beats model capability, and an open environment so you can prove us wrong.*

---

Healthcare insurance fraud costs the US system over **$100 billion per year**. Most detection systems are batch processes: flag suspicious claims at the end of the month, review manually, claw back if possible. By then the money is gone.

What if we treated fraud detection as a **sequential decision problem** under real constraints? An agent reviewing one claim at a time, with a limited investigation budget, building memory of provider history across the episode — just like a real claims adjudicator.

We built exactly this environment, then ran a complete evaluation across six agents and two LLM models. The results were sharper than we expected.

**The short version:**
- A naive LLM performs *worse than random*
- A budget-aware prompt improves the same model by up to **2.7×**
- A rule-based heuristic beats naive LLMs without a single API call
- The same prompt works on both a weak and a strong model — but the stronger model executes it more precisely
- A trained REINFORCE policy discovers the same strategy from reward signal alone

---

## The Environment

Each episode consists of **100 insurance claims** arriving sequentially. The agent sees:

- Claim amount, procedure codes, provider billing history
- Current investigation budget remaining (starts at 15 per episode)
- Memory of previously investigated providers (confidence decays over time)
- Episode step counter and risk indicators

For each claim, the agent picks one of 5 actions:

| Action | Cost | Detection Rate |
|--------|------|---------------|
| `APPROVE` | $0 | 0% — fraud passes through |
| `FLAG_REVIEW` | $25 | 70% — human review |
| `INVESTIGATE` | $100 | 95% — deep audit |
| `DENY` | $0 | 100% — but penalises false denials |
| `REQUEST_INFO` | $12.50 | 50% — deferred review |

The **critical tension**: `INVESTIGATE` at $100 is only financially optimal for fraud claims above ~$3,000. For a typical $500 fraudulent claim, `FLAG_REVIEW` at $25 with 70% detection saves more money net. Most fraud is not high-value. Most agents don't figure this out.

Fraud rate is **5%** — 5 out of every 100 claims. The investigation budget is **15** — meaning even a perfect agent can only deep-audit 15% of claims. The rest must be handled via cheaper signals.

---

## Six Agents, One Leaderboard

We evaluated six agents across 20 episodes each (2,000 claim decisions per agent, same seeds throughout):

**RandomAgent** — weighted random decisions, no reasoning, no API calls. The absolute floor.

**ThresholdAgent** — pure rule-based logic using regex over the prompt text. Flags claims above anomaly thresholds, approves the rest. Never uses INVESTIGATE. No LLM, zero latency.

**NaiveLLMAgent** — sends each claim to an LLM with a minimal prompt: *"Review this claim and decide."* No mention of budget, investigation costs, or memory.

**BudgetAwareAgent** — same LLM, system prompt explicitly states the economics: INVESTIGATE costs $100, FLAG_REVIEW costs $25, budget limit is 15, switch strategy when budget falls below 20%.

For LLM experiments: **DeepSeek V3.2** (state-of-the-art instruction-following, $0.26/M tokens) and **Qwen 3.6 Plus** (current-gen Alibaba model, run on free tier). Each naive/budget-aware pair runs on the same model — the only difference is the system prompt.

**ReinforceAgent** — a linear policy trained for 500 episodes via REINFORCE policy gradient on 10 hand-crafted features.

---

## Complete Results

All results from 20 episodes × 100 claims each, fixed seeds (`seed=42`).

| Agent | Mean Reward | Std | F1 | Recall | Budget Use | Net Loss/ep |
|-------|------------|-----|-----|--------|------------|------------|
| BudgetAware (DeepSeek) | **−455** | 145 | 0.047 | 8% | **0%** | $2,609 |
| ThresholdAgent | −799 | 177 | **0.123** | 57% | 0% | $1,447 |
| BudgetAware (Qwen3.6) | −1,190 | 1,112 | **0.147** | 52% | 5% | $3,181 |
| NaiveLLM (DeepSeek) | −1,212 | 147 | 0.074 | 37% | 9% | $2,315 |
| REINFORCE (trained) | −1,646 | 209 | 0.057 | 23% | 79% | — |
| RandomAgent | −2,173 | 309 | 0.072 | 37% | 94% | $6,137 |
| NaiveLLM (Qwen3.6) | −2,322 | 213 | 0.063 | 52% | 70% | $5,645 |

*Lower reward = more net losses. Higher is better.*

---

## Finding 1: A Naive LLM Is Worse Than Random

Qwen NaiveLLM scored **−2,322** — worse than RandomAgent at **−2,173**, by 149 reward points.

This is not a model capability problem. The NaiveLLM *worked hard*:
- 70% budget utilization (used most of its 15 investigation slots)
- 92% memory reuse rate (correctly avoided re-investigating known providers)
- 52% recall (caught over half the fraud)

The problem is *what it did with that work*. It investigated frequently, burning $100 per slot on moderate-risk claims. On legitimate claims — which make up 95% of the episode — each investigation costs $100 + $50 false-positive penalty = $150 in pure waste. Over a 100-claim episode with ~10–14 investigations, that's $1,000–$2,100 in investigation overhead before catching a single dollar of fraud.

```
NaiveLLM (Qwen) episode sample:
  investigations_used:  14 out of 15 budget
  investigation_cost:   $1,825
  false_positive_cost:  $1,900   ← investigating legitimate claims
  fraud_caught_amount:  $149     ← barely worth it
  total_reward:        −1,950
```

A random agent, by contrast, has no intelligence to act on. It randomly approves many things, randomly investigates others. Its investigations are spread across low-risk and high-risk claims alike — so the expected false-positive rate per investigation is lower than the LLM's targeted-but-miscalibrated investigations.

**The lesson:** An LLM with domain knowledge but no cost calibration is *worse* than random. It acts on signal (suspicion) without understanding the cost of acting on that signal.

---

## Finding 2: Budget-Aware Prompting Improves Both Models — Proportionally to Their Capability

We ran the budget-aware prompt on both models:

| Model | Naive Reward | Budget-Aware Reward | Improvement | Budget Use: Naive → BA |
|-------|-------------|--------------------|-----------|-----------------------|
| Qwen 3.6 Plus | −2,322 | −1,190 | **1.95×** | 70% → 5% |
| DeepSeek V3.2 | −1,212 | −455 | **2.66×** | 9% → 0% |

The budget-aware system prompt adds three things:

1. **Economics:** *"INVESTIGATE costs $100. FLAG_REVIEW costs $25. Only INVESTIGATE when you have high confidence AND the claim is large."*
2. **Thresholds:** *"When budget remaining < 4, switch entirely to FLAG_REVIEW."*
3. **Memory rules:** *"If a provider is in memory as FRAUD, FLAG_REVIEW — don't re-investigate. If LEGIT, APPROVE."*

Both models respond. But with meaningfully different precision:

- **DeepSeek** drops investigation use to **0%** — it fully internalises the economics and never INVESTIGATEs
- **Qwen** drops to **5%** — a 14× reduction from 70%, but not complete

This reveals a second-order finding: **the value of a budget-aware prompt scales with the model's instruction-following capability**. A stronger model executes the decision tree exactly. A weaker model partially follows it, capturing most but not all of the benefit.

Prompt engineering is not a silver bullet — it requires a model that can actually follow the prompt.

---

## Finding 3: Rule-Based Wins (Until a Strong Model Is Told the Rules)

ThresholdAgent (−799) beats NaiveLLM on both models, and beats BudgetAware Qwen (−1,190). Only BudgetAware DeepSeek (−455) clears it.

ThresholdAgent independently discovered the optimal strategy: never INVESTIGATE, use FLAG_REVIEW for anomalies, APPROVE otherwise. It executes this via 10 lines of regex and if-statements with zero API calls, zero latency, and zero cost.

```python
# ThresholdAgent's effective policy (simplified):
if risk_level == "HIGH" and budget_remaining >= 5:
    return INVESTIGATE   # almost never fires in practice
elif risk_level == "MODERATE" or fraud_flag_rate > 8%:
    return FLAG_REVIEW
else:
    return APPROVE
```

The rule for INVESTIGATE barely fires because truly HIGH-risk claims are rare. In practice, ThresholdAgent is an optimised FLAG_REVIEW machine — which turns out to be nearly optimal.

**The gap between ThresholdAgent (−799) and BudgetAware DeepSeek (−455) is meaningful:** 344 reward points. The LLM can reason about *which specific claims* warrant flagging with more nuance than a fixed threshold — when explicitly told what it's optimising for. That gap is the value that strong LLMs add over hand-coded rules: context-sensitive discrimination, not brute force.

---

## Finding 4: High F1 Is the Wrong Goal

Notice something counterintuitive in the results table: BudgetAware DeepSeek has the *lowest* F1 (0.047) and *lowest* recall (8%) — yet the best reward.

Meanwhile, NaiveLLM Qwen has the highest recall (52%) and still loses to RandomAgent.

The environment rewards **financially efficient** fraud detection, not raw classification. An agent that catches 100% of fraud by investigating every single claim would have perfect recall — and would be catastrophically expensive (100 × $150 false-positive cost on legitimate claims = $14,250/episode in investigation waste alone).

BudgetAware DeepSeek achieved −$2,609 net loss per episode by catching only the fraud it can detect cheaply via FLAG_REVIEW. ThresholdAgent achieved −$1,447 net loss by catching more fraud (F1=0.123) with the same zero-INVESTIGATE strategy. The better F1 there reflects better anomaly detection in the rules, not more investigation.

**F1 only measures fraud detection. Reward measures fraud detection efficiency.** These diverge sharply when investigation is expensive and fraud is rare.

---

## Finding 5: The Budget Paradox — More Resources, Worse Results

Budget ablation across investigation budgets of 5, 10, 15, and 20 (rule-based agents, 10 episodes each):

| Budget | RandomAgent | ThresholdAgent |
|--------|-------------|---------------|
| 5 | −1,720 | −765 |
| 10 | −1,860 | −765 |
| 15 | −1,955 | −765 |
| 20 | −2,006 | −765 |

**Giving RandomAgent more budget makes it worse.** More investigation slots → more random INVESTIGATE calls → more $150 false-positive costs on legitimate claims → deeper negative reward.

ThresholdAgent is **perfectly flat** at −765 across all budget levels because it never uses INVESTIGATE — the budget is simply never binding.

This result has a clean interpretation: **investigation budget is only valuable to agents that can spend it wisely**. For agents that can't discriminate when to investigate, more budget is strictly harmful. The resource amplifies whatever decision-making quality (or lack thereof) the agent already has.

We ran a partial budget ablation with DeepSeek at B=5 and B=10 (10 episodes each):

| Budget | NaiveLLM (DeepSeek) | BudgetAware (DeepSeek) | BA / Naive ratio |
|--------|--------------------|-----------------------|-----------------|
| 5 | −1,169 | **−743** | 1.57× |
| 10 | −1,192 | — | — |
| 15 | −1,212 | **−455** | 2.66× |

Two things stand out. First, NaiveLLM actually *improves* slightly from B=15 (−1,212) to B=5 (−1,169) — tight budgets are self-correcting. When the agent only has 5 investigation slots, it can't burn 14 of them even when it wants to. The budget constraint acts as an accidental guardrail on its over-investigation tendency.

Second, the budget-aware advantage **compresses as budgets tighten**: 2.66× at B=15 drops to 1.57× at B=5. The gap closes because both agents end up in similar territory: BudgetAware explicitly rations its 5 slots, but NaiveLLM is also forced into rationing by scarcity. At very low budgets, the marginal value of knowing the rules decreases because resource exhaustion enforces the same behaviour anyway.

---

## Finding 6: RL Learns the Same Strategy From Scratch

The toughest challenge: 5% fraud rate means 95 legitimate-claim steps generate noisy gradient that drowns out signal from 5 fraud steps.

Training setup: linear policy (10 features → 3 actions), 500 episodes, REINFORCE with batch advantage normalisation, entropy regularisation, gradient clipping.

```
First quartile (ep 1–125):   mean reward = −2,398
Last quartile (ep 376–500):  mean reward = −1,739
Improvement: +658 reward   ✓ Policy LEARNED
Training time: 56 seconds
```

The policy learned — from reward signal alone — to weight `budget_frac` (feature 0) and `provider_in_mem` (feature 7) heavily. When budget depletes, INVESTIGATE probability drops. When a provider is in memory, FLAG_REVIEW/APPROVE probability rises. These are exactly the strategies the budget-aware prompt describes in words.

The trained agent (−1,646) doesn't match ThresholdAgent (−799) yet — the linear policy over 10 features can't express full conditional logic. But it demonstrates that the environment contains learnable structure: RL can find policy improvements without any human-written rules.

---

## Finding 7: Memory Half-Life Only Matters If You Use Memory

We ablated `memory_decay_halflife` over [0, 5, 20, 100] with rule-based agents:

| Halflife | RandomAgent | ThresholdAgent |
|---------|-------------|---------------|
| 0 (off) | −2,068 | −763 |
| 5 | −2,068 | −763 |
| 20 | −2,068 | −763 |
| 100 | −2,068 | −763 |

Both are **completely flat**. Two reasons:

1. ThresholdAgent **never investigates** → never populates memory → memory decay rate is irrelevant because memory is always empty
2. RandomAgent **ignores the prompt** → memory content visible in the prompt has no effect on random decisions

This confirms memory is functional in the environment — it's just not activated by agents that don't use INVESTIGATE or don't read it. The LLM memory ablation (requiring the BudgetAware LLM runs across half-life settings) would show real variation — that remains a future experiment.

---

## What the Numbers Actually Mean

To make the reward numbers concrete: a −$455 mean reward means BudgetAware DeepSeek runs a fraud program that loses **$455 per 100 claims** reviewed. ThresholdAgent loses $799. NaiveLLM Qwen loses $2,322.

For a claims department processing 10,000 claims/day:
- NaiveLLM Qwen costs **$232,200/day** in net fraud program losses
- ThresholdAgent costs **$79,900/day**
- BudgetAware DeepSeek costs **$45,500/day**

The API cost to run BudgetAware DeepSeek on 10,000 claims: roughly **$6/day** ($0.26/M tokens × ~23M tokens). The prompt engineering that achieves 2.7× better financial performance costs less than a coffee.

---

## Try It Yourself

The environment is live on Hugging Face Hub:

```python
from environment.client import FraudEnvClient

client = FraudEnvClient("https://huggingface.co/spaces/shylane/healthcare-fraud-openenv")
obs = client.reset()

while not obs["done"]:
    response = your_agent.act(obs["prompt"])
    obs = client.step(response)

print(f"Episode reward: {obs['total_reward']}")
```

Or run locally:
```bash
git clone https://github.com/shylane/healthcare-fraud-openenv
cd healthcare-fraud-openenv
uvicorn environment.server.app:app --port 8000
```

Seeds are fixed. Same claims, same fraud distribution, fully reproducible. Can you beat BudgetAware DeepSeek's −455?

---

## Open Threads

Three questions remain open:

**LLM memory ablation.** The memory ablation only ran on rule-based agents — both were flat because neither builds memory (ThresholdAgent never investigates, RandomAgent ignores the prompt). The interesting case is BudgetAware DeepSeek, which explicitly relies on memory to avoid re-investigating known providers. Does its reward degrade when `memory_decay_halflife` drops to 0? Hypothesis: yes, because providers seen early in the episode would no longer be recognised later, forcing redundant FLAG_REVIEW calls. Remains unrun.

**Full budget ablation with LLM agents.** We got B=5 for both DeepSeek agents and B=10 for NaiveLLM. The budget-aware advantage at B=5 (1.57×) vs B=15 (2.66×) is itself a finding: tighter budgets compress the gap. Whether that gap closes further or reverses at very tight constraints is an open question.

**REINFORCE vs Threshold gap.** The trained RL policy (−1,646) trails ThresholdAgent (−799) by 847 reward points. A non-linear policy (2-layer MLP) or expanded feature set (procedure codes, claim amount directly, fraud pattern type) would likely close this. The environment contains learnable structure — the linear policy is the bottleneck, not the algorithm.

The environment is open. Every result here is reproducible from the JSON files in `experiments/*/results/`. We're curious what you find.

---

## Links

- **Environment**: [HF Space — shylane/healthcare-fraud-openenv](https://huggingface.co/spaces/shylane/healthcare-fraud-openenv)
- **Code + experiments**: [github.com/shylane/healthcare-fraud-openenv](https://github.com/shylane/healthcare-fraud-openenv)
- **Study documentation**: `docs/study_documentation.md` — full technical write-up with code citations
- **Competition**: [AgentX-AgentBeats OpenEnv Track](https://rdi.berkeley.edu/agentx-agentbeats)

---

*Built for the AgentX-AgentBeats OpenEnv Challenge (Berkeley RDI / Hugging Face, April 2026).
6 agents × 2 models × 20 episodes × 100 claims = 14,000 decisions. All open source.*
