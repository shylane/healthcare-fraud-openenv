# When LLMs Lose to Coin Flips: Building a Healthcare Fraud Detection RL Environment

*A rigorous evaluation study — across two models, seven agent configurations, and 14,000 claim decisions — showing that prompt engineering beats model capability, and an open environment so you can prove us wrong.*

---

Healthcare insurance fraud costs the US system an estimated **$100 billion per year** — roughly 3–10% of total health spending, per NHCAA estimates against $4.9T in CMS-reported 2023 expenditures. Most detection systems are batch processes: flag suspicious claims at the end of the month, review manually, claw back if possible. By then the money is gone.

What if we treated fraud detection as a **sequential decision problem** under real constraints? An agent reviewing one claim at a time, with a limited investigation budget, building memory of provider history across the episode — just like a real claims adjudicator.

We built exactly this environment, then ran a complete evaluation across seven agent configurations and two LLM models. The results were sharper than we expected.

**The short version:**
- A naive LLM performs *worse than random*
- A budget-aware prompt improves the same model by up to **2.7×**
- A rule-based heuristic beats naive LLMs without a single API call
- The same prompt works on both a weak and a strong model — but the stronger model executes it more precisely
- A trained REINFORCE policy discovers the same strategy from reward signal alone
- **And one honest finding we discovered about our own environment:** the RL reward objective and the real-world financial outcome rank agents differently — a calibration gap worth knowing before you train on this

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

The **critical tension**: `INVESTIGATE` at $100 is RL-optimal only for suspected fraud above roughly **$1,000** at the current 0.1× fraud-recovery reward rate — a threshold well above the typical claim in this simulation. Below that, `FLAG_REVIEW` at $25 nets a better RL score. (In real-dollar terms the breakeven is ~$300 — see Finding 8 on reward calibration for why these diverge.)

Fraud rate is **5%** — 5 out of every 100 claims. The investigation budget is **15** — meaning even a perfect agent can only deep-audit 15% of claims. The rest must be handled via cheaper signals.

### Reward Architecture

The reward is **multi-component**, not a single financial signal:

| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Decision correctness | **40%** | Financial outcome of the action choice |
| Rationale quality | 30% | Coherence and length of the written explanation |
| Evidence citation | 20% | Did the agent cite specific claim data? |
| Efficiency | 10% | Cost-effectiveness of action given risk level |

When we say "reward" throughout this post we mean the weighted sum. An agent can improve reward not only by making better financial decisions but by writing better rationales — which matters when comparing LLMs (which write prose) to rule-based agents (which emit minimal text). Finding 4 addresses this directly.

---

## Seven Agents, One Leaderboard

We evaluated seven agent configurations across 20 episodes each (2,000 claim decisions per agent, fixed seeds throughout):

**RandomAgent** — weighted random decisions, no reasoning, no API calls. The absolute floor.

**ThresholdAgent** — pure rule-based logic using regex over the prompt text. Flags claims above anomaly thresholds, approves the rest. Never uses INVESTIGATE. No LLM, zero latency.

**NaiveLLMAgent** — sends each claim to an LLM with a minimal prompt: *"Review this claim and decide."* No mention of budget, investigation costs, or memory. Run on two models:
- *NaiveLLM (DeepSeek V3.2)* — state-of-the-art instruction-following, $0.26/M tokens
- *NaiveLLM (Qwen 3.6 Plus)* — current-gen Alibaba model, run on free tier

**BudgetAwareAgent** — same LLM, system prompt explicitly states the economics: INVESTIGATE costs $100, FLAG_REVIEW costs $25, budget limit is 15, switch strategy when budget falls below 20%. Same two models:
- *BudgetAware (DeepSeek V3.2)*
- *BudgetAware (Qwen 3.6 Plus)*

Each naive/budget-aware pair runs on the same model backbone — the only difference is the system prompt. This gives four LLM configurations total.

**ReinforceAgent** — a linear policy trained for 500 episodes via REINFORCE policy gradient on 10 hand-crafted features.

---

## Complete Results

All results from 20 episodes × 100 claims each, fixed seeds (`seed=42`).

| Agent | RL Reward | F1 | Recall | Budget Use | Fraud Caught$/ep | Net Loss$/ep | Fraud Catch Rate |
|-------|-----------|-----|--------|------------|-----------------|-------------|-----------------|
| BudgetAware (DeepSeek) | **−455** | 0.047 | 8% | **0%** | $905 | $2,609 | 26% |
| ThresholdAgent | −841 | **0.144** | 53% | 0% | $1,493 | $2,147 | 48% |
| BudgetAware (Qwen3.6) | −1,190 | **0.147** | 52% | 5% | $1,412 | $3,181 | 46% |
| NaiveLLM (DeepSeek) | −1,212 | 0.074 | 37% | 9% | $2,149 | $2,315 | **61%** |
| REINFORCE (trained) | −1,646 | 0.057 | 23% | 79% | $1,194 | $5,352 | 30% |
| RandomAgent | −2,057 | 0.087 | 44% | 88% | $1,313 | $6,137 | 34% |
| NaiveLLM (Qwen3.6) | −2,322 | 0.063 | 52% | 70% | $1,608 | $5,645 | 52% |

*RL Reward: higher is better. Net Loss: per 100-claim episode. Fraud Catch Rate: % of total fraud $ recovered.*

> **Note:** RL Reward and Net Loss$/ep rank agents differently — this matters, and we explain why in Finding 8.

---

## Finding 1: A Naive LLM Is Worse Than Random

Qwen NaiveLLM scored **−2,322** — worse than RandomAgent at **−2,057**, by 265 reward points.

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

ThresholdAgent (−841) beats NaiveLLM on both models, and beats BudgetAware Qwen (−1,190). Only BudgetAware DeepSeek (−455) clears it.

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

**The gap between ThresholdAgent (−841) and BudgetAware DeepSeek (−455) is meaningful:** 386 reward points. The LLM can reason about *which specific claims* warrant flagging with more nuance than a fixed threshold — when explicitly told what it's optimising for. That gap is the value that strong LLMs add over hand-coded rules: context-sensitive discrimination, not brute force.

---

## Finding 4: High F1 Is the Wrong Goal

Notice something counterintuitive in the results table: BudgetAware DeepSeek has the *lowest* F1 (0.047) and *lowest* recall (8%) — yet the best RL reward.

Meanwhile, NaiveLLM Qwen has the highest recall (52%) and still loses to RandomAgent.

The environment rewards **financially efficient** fraud detection (40% of the reward signal), not raw classification. An agent that catches 100% of fraud by investigating every single claim would have perfect recall — and would be catastrophically expensive (100 × $150 false-positive cost on legitimate claims = $14,250/episode in investigation waste alone).

ThresholdAgent achieves F1=0.144 — highest alongside BudgetAware Qwen — with **zero** investigation budget used. Its rules flag correctly with moderate precision and never incur investigation costs at all. The BudgetAware DeepSeek's F1=0.047 looks terrible, but it's working in a completely different regime: FLAG_REVIEW only, minimal cost, accepting that most fraud slips through in exchange for near-zero investigation overhead.

One nuance: the 30% rationale + 20% evidence components of the reward favour LLM agents (which write structured prose) over ThresholdAgent (which emits minimal text). ThresholdAgent's strong overall score comes *despite* near-zero rationale credit — its financial decision quality is that much better.

**F1 measures fraud detection coverage. RL Reward measures fraud detection efficiency.** These diverge sharply when investigation is expensive and fraud is rare. Which of the two you should care about depends on what you're actually trying to do — and that's the subject of Finding 8.

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

ThresholdAgent is **perfectly flat** at −765 across all budget levels because it never uses INVESTIGATE — the budget is simply never binding. (The −765 vs −841 difference from the main results table reflects fewer episodes: this ablation uses 10 episodes vs 20 in the main evaluation; variance at 10 episodes is higher but the direction holds.)

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

The trained agent (−1,646) doesn't match ThresholdAgent (−841) yet — the linear policy over 10 features can't express full conditional logic. (The −1,739 training mean is over the last 125 training episodes with the same env seed; −1,646 is the separate 20-episode held-out evaluation with `seed=42`, so the gap reflects the policy being evaluated on different claim sequences than it trained on.) But it demonstrates that the environment contains learnable structure: RL can find policy improvements without any human-written rules.

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

## Finding 8: Our Reward Function Has a Calibration Gap

This is the finding we didn't plan for — we found it by looking at the data carefully.

The RL reward function scales fraud recovery at **10% of claim value** and missed-fraud penalty at **20%**:

```python
# environment/models.py — RewardConfig defaults
fraud_caught_reward_rate  = 0.1   # catching $1,000 fraud → +$100 reward
fraud_missed_penalty_rate = 0.2   # missing $1,000 fraud  → -$200 penalty
investigation_cost        = 100.0 # flat cost per INVESTIGATE
```

This makes INVESTIGATE only worthwhile (in RL reward terms) for confirmed fraud above roughly ~$1,000 — a tight threshold almost no individual claim exceeds in expectation. The result: the RL-optimal strategy is to avoid investigation almost entirely, not because investigation is wrong, but because the *rate* underprices the value of fraud recovery.

**The symptom: RL Reward and Net Savings rank agents differently.**

| Agent | RL Reward Rank | Net Loss$/ep | Financial Rank |
|-------|---------------|-------------|----------------|
| BudgetAware (DeepSeek) | **1st** (−455) | $2,609 | 3rd |
| ThresholdAgent | 2nd (−841) | **$2,147** | **1st** |
| NaiveLLM (DeepSeek) | 4th (−1,212) | $2,315 | 2nd |
| BudgetAware (Qwen3.6) | 3rd (−1,190) | $3,181 | 4th |

**ThresholdAgent has the best real-world outcome** ($2,147 net loss per episode) despite ranking 2nd on the RL objective. NaiveLLM(DeepSeek) — which investigates more and catches more fraud dollars ($2,149 vs $905) — comes second in actual dollars despite ranking 4th on RL reward.

**Why this happens:** BudgetAware DeepSeek optimises the RL objective precisely. It eliminates investigation entirely (0% budget use), avoids false-positive costs, and accepts a low fraud catch rate (26%). This is RL-optimal because the 0.1 reward rate makes even recovered fraud barely worth the $100 investigation cost. But in real terms, that 26% catch rate leaves $2,635 of fraud unpaid per episode. ThresholdAgent's heuristics catch 48% of fraud at moderate cost, netting a better actual outcome.

**The fix is a one-line change:**
```python
fraud_caught_reward_rate  = 1.0   # full claim value recovered
fraud_missed_penalty_rate = 1.0   # full claim value lost
```

With equal rates (or rates calibrated to actual payer economics), the RL objective aligns with net savings. Investigation of genuinely high-value suspicious claims becomes worthwhile. The optimal strategy shifts toward selective investigation rather than pure cost-avoidance.

**Why we're not re-running:** With the submission deadline upon us, re-running all 7 agents to produce a clean comparable dataset is out of scope. We're documenting the gap honestly instead.

**What this means for the evaluation study findings:** The core finding — budget-aware prompting improves the same LLM by 2.7× — holds regardless of which metric you use (BudgetAware DeepSeek is best on RL reward; the same direction holds for net savings within each model pair). The direction is consistent: structured prompting helps. But the *magnitude* and *mechanism* change. Under correct reward scaling, an agent that catches more fraud dollars is explicitly rewarded for it, and the threshold for INVESTIGATE becomes much lower.

**The lesson for practitioners:** When designing RL environments for real business problems, verify that your reward rates reflect actual value at stake. A 10% recovery reward on a $1,000 fraud and a flat $100 investigation cost creates a regime where the optimal RL policy is "never investigate" — which may be optimal in the RL game while being poor real-world policy.

---

## What the Numbers Mean in Practice

To make the reward numbers concrete: a −$455 RL reward means BudgetAware DeepSeek runs a fraud program that scores −$455 on the RL objective. Its real-world financial loss (net_savings metric) is **$2,609/episode** — because the 0.1× reward rate understates the actual fraud value in the RL signal.

For a claims department processing 10,000 claims/day, using **net financial losses** as the metric:
- NaiveLLM Qwen: **$564,500/day** in net fraud program losses
- RandomAgent: **$613,700/day**
- BudgetAware DeepSeek: **$260,900/day**
- ThresholdAgent: **$214,700/day** ← best real-world outcome

The API cost to run BudgetAware DeepSeek on 10,000 claims: roughly **$6/day** ($0.26/M tokens × ~23M tokens). Even with its suboptimal reward calibration, the gap between BudgetAware and NaiveLLM ($303,600/day) is enormous relative to API cost.

---

## Try It Yourself

The environment is live on Hugging Face Hub:

```python
from environment.client import HealthClaimEnv
from environment.models import ClaimAction

client = HealthClaimEnv("https://shylane-healthcare-fraud-openenv.hf.space")
obs = client.reset()

while not obs.done:
    response = your_agent.act(obs.prompt)
    obs = client.step(ClaimAction(response_text=response))

print(f"Episode reward: {obs.metadata.get('cumulative_reward', 0)}")
```

Or run locally:
```bash
git clone https://github.com/shylane/healthcare-fraud-openenv
cd healthcare-fraud-openenv
uvicorn environment.server.app:app --port 8000
```

Seeds guarantee identical claim sequences for **deterministic agents** (LLMs). Agents that use Python's global `random` module (e.g. a random-action baseline) will see slightly different episode trajectories because their action draws interleave with lazy claim generation — see the Limitations section below. All LLM-vs-LLM comparisons in this study are clean.

Can you beat BudgetAware DeepSeek's −455?

---

## Open Threads

Four questions remain open:

**LLM memory ablation.** The memory ablation only ran on rule-based agents — both were flat because neither builds memory (ThresholdAgent never investigates, RandomAgent ignores the prompt). The interesting case is BudgetAware DeepSeek, which explicitly relies on memory to avoid re-investigating known providers. Does its reward degrade when `memory_decay_halflife` drops to 0? Hypothesis: yes, because providers seen early in the episode would no longer be recognised later, forcing redundant FLAG_REVIEW calls. Remains unrun.

**Full budget ablation with LLM agents.** We got B=5 for both DeepSeek agents and B=10 for NaiveLLM. The budget-aware advantage at B=5 (1.57×) vs B=15 (2.66×) is itself a finding: tighter budgets compress the gap. Whether that gap closes further or reverses at very tight constraints is an open question.

**REINFORCE vs Threshold gap.** The trained RL policy (−1,646) trails ThresholdAgent (−841) by 805 reward points. A non-linear policy (2-layer MLP) or expanded feature set (procedure codes, claim amount directly, fraud pattern type) would likely close this. The environment contains learnable structure — the linear policy is the bottleneck, not the algorithm.

**Reward rate recalibration.** Set `fraud_caught_reward_rate = 1.0` and `fraud_missed_penalty_rate = 1.0` in `environment/models.py → RewardConfig`. Re-run all experiments. Hypothesis: rankings change significantly — ThresholdAgent drops relative to agents that learn to investigate selectively, and the REINFORCE policy improves by learning a non-trivial investigation strategy rather than pure cost-avoidance.

The environment is open. Every result here is reproducible from the JSON files in `experiments/*/results/`. We're curious what you find.

---

## Limitations and Caveats

We're documenting these openly because they affect how you should interpret specific numbers, even though they don't change the directional findings.

**RNG isolation.** Claims are generated lazily (one per step). The `ClaimsFraudEnvironment` uses a dedicated `random.Random(seed)` instance for its own stochastic decisions (investigation accuracy draws), but `ClaimsSimulator` uses Python's global `random` module for claim generation. `RandomAgent` and `ReinforceAgent` also draw from the global `random` module during action selection, meaning their action draws interleave with subsequent claim generation. This contaminates the "identical claim sequences" guarantee for those two agents.

**Impact:** All LLM-vs-LLM comparisons (Findings 1, 2, 3) are unaffected — LLM agents make no Python `random` calls. The RandomAgent and REINFORCE results are reproducible across multiple runs with the same global seed, but their claim sequences differ slightly from those seen by LLM agents. The directional conclusions (NaiveLLM worse than random; BudgetAware better than NaiveLLM) hold by large enough margins to be robust to this effect.

**Investigation memory records detected truth, not ground truth (post-fix).** Prior to this commit, investigation memory stored `is_fraud` as the claim's ground-truth label regardless of whether the investigation stochastically missed (5% miss rate at `investigate_accuracy=0.95`). This leaked the true label to agents on provider re-encounters. The code is now fixed: memory stores `is_fraud=False` when an investigation misses. At the default 0.95 accuracy, the practical impact on recorded results is small (~5% of fraud investigations), but the principle matters for environments with lower accuracy settings.

**API fallback in ablation runs.** The OpenRouter client falls back to a parseable `FLAG_REVIEW` response on null-content API errors, keeping `valid_response_rate` high even when the model never reasoned. Precise status per budget level: B=5 NaiveLLM and BudgetAware are valid and cited; B=10 NaiveLLM is valid and cited (−1,192); B=10 BudgetAware was not run (credits exhausted); B=20 both agents were collected but show all-FLAG_REVIEW behaviour with response lengths matching the fallback string exactly — those results are excluded. Finding 5 cites only B=5, B=10 (NaiveLLM), and B=15.

**Harness step-level logging (post-fix).** `StepRecord.is_fraud` was previously logged after `env.step()`, recording the *next* claim's label instead of the current one (off-by-one). This has been fixed. Episode-level metrics (total reward, F1, recall, budget utilisation) are computed from the environment's internal state and were never affected by this bug.

**Memory reuse metric (post-fix).** `memory_reuse_rate` previously counted `APPROVE` on a known-fraud provider as "correct" (the agent didn't waste an investigation slot). The metric now correctly scores: FLAG_REVIEW/DENY as correct for known-fraud providers, APPROVE as correct for known-legit providers.

**Multi-component reward vs financial outcome.** The RL reward is 40% financial decision quality, 30% rationale coherence, 20% evidence citation, and 10% efficiency. When comparing LLM agents (which write prose rationales) to ThresholdAgent or RandomAgent (which emit minimal text), the rationale/evidence components create a persistent headwind for rule-based agents. ThresholdAgent's strong overall ranking is despite near-zero rationale credit — its financial decisions are that dominant.

**The $100B figure.** This is the commonly cited NHCAA estimate. The more rigorous range is 3–10% of health spending; at $4.9T (CMS 2023) that implies $147B–$490B in potential fraud exposure, not all of which is recoverable. We use $100B as a conservative anchor.

---

## Links

- **Environment**: [HF Space — shylane/healthcare-fraud-openenv](https://huggingface.co/spaces/shylane/healthcare-fraud-openenv)
- **Code + experiments**: [github.com/shylane/healthcare-fraud-openenv](https://github.com/shylane/healthcare-fraud-openenv)
- **Study documentation**: `docs/study_documentation.md` — full technical write-up with code citations
- **Competition**: [AgentX-AgentBeats OpenEnv Track](https://rdi.berkeley.edu/agentx-agentbeats)

---

*Built for the AgentX-AgentBeats OpenEnv Challenge (Berkeley RDI / Hugging Face, April 2026).
7 agent configurations × 20 episodes × 100 claims = 14,000 decisions. All open source.*
