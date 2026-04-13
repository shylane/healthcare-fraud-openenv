# Healthcare Claims Fraud Detection — OpenEnv Challenge

> **AgentX-AgentBeats / OpenEnv Track** | April 2026

A rigorous evaluation study: *does budget/memory-aware prompting make LLM agents meaningfully better at sequential fraud detection?*

We built a 100-step RL environment, ran seven agent configurations across 14,000 claim decisions, and measured the results. The short answer: yes — by up to **2.7×** — but a 10-line rule-based agent beats every LLM that isn't told the economics first.

> **Blog post:** [HF Space — shylane/healthcare-fraud-openenv](https://huggingface.co/spaces/shylane/healthcare-fraud-openenv)

---

## Key Findings

| Agent | RL Reward | Net Loss$/ep | Fraud Catch Rate |
|-------|-----------|-------------|-----------------|
| BudgetAware (DeepSeek) | **−455** | $2,609 | 26% |
| ThresholdAgent | −841 | **$2,147** | 48% |
| BudgetAware (Qwen3.6) | −1,190 | $3,181 | 46% |
| NaiveLLM (DeepSeek) | −1,212 | $2,315 | 61% |
| REINFORCE (trained) | −1,646 | $5,352 | 30% |
| RandomAgent | −2,057 | $6,137 | 34% |
| NaiveLLM (Qwen3.6) | −2,322 | $5,645 | 52% |

See [`blog/hf_blog_post.md`](blog/hf_blog_post.md) for the full 8-finding analysis and [`docs/study_documentation.md`](docs/study_documentation.md) for the complete technical write-up.

---

## Quick Start

```bash
git clone https://github.com/shylane/healthcare-fraud-openenv
cd healthcare-fraud-openenv

# Run the environment server
pip install -r environment/server/requirements.txt
uvicorn environment.server.app:app --host 0.0.0.0 --port 8000

# In another terminal — interact via curl
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"response_text": "Decision: FLAG_REVIEW\nRationale: Billing anomaly detected.\nEvidence: claim_amount=$4,200 vs provider avg=$480"}'
```

---

## Repository Structure

```
healthcare-fraud-openenv/
├── environment/
│   ├── server/
│   │   ├── app.py              # FastAPI OpenEnv server (/reset /step /state /a2a/*)
│   │   ├── environment.py      # ClaimsFraudEnvironment, reward computation
│   │   └── requirements.txt    # Server dependencies
│   ├── claims_simulator.py     # Synthetic fraud data generator
│   ├── models.py               # RewardConfig, ClaimObservation, ClaimAction
│   ├── client.py               # HTTP client for the environment
│   └── openenv.yaml            # HF Hub OpenEnv manifest
├── green_agent/                # AgentBeats evaluator (Docker + A2A)
│   ├── Dockerfile
│   └── src/                    # server.py, executor.py, agent.py, messenger.py
├── evaluation/
│   ├── agents.py               # All 7 agent implementations
│   └── harness.py              # run_agent(), EvalResults, EpisodeResult
├── experiments/
│   ├── 01_baseline_comparison/ # 7-agent comparison, 20 eps each
│   ├── 02_budget_ablation/     # Budget sweep [5, 10, 15, 20]
│   ├── 03_memory_ablation/     # Half-life sweep [0, 5, 20, 100]
│   └── 04_reinforce_poc/       # REINFORCE: 500 training eps + eval
├── blog/hf_blog_post.md        # Primary submission artifact (8 findings)
├── docs/study_documentation.md # Complete technical write-up (~920 lines)
├── notebooks/01_results_analysis.ipynb
├── scenario.toml               # AgentBeats leaderboard submission config
└── README.md
```

---

## The Environment

**Task:** Sequential claim review with a 15-investigation budget per 100-claim episode.

**Actions:** `APPROVE | FLAG_REVIEW | INVESTIGATE | DENY | REQUEST_INFO`

**Reward signal (4 components):**
- Decision correctness (40%) — financial outcome of the action
- Rationale quality (30%) — coherence of the written explanation
- Evidence citation (20%) — cited specific claim data
- Efficiency (10%) — cost-effectiveness given risk

**Fraud patterns:** Upcoding, phantom billing, duplicate claims, unbundling, provider collusion (synthesised via configurable `ClaimsSimulator`).

**Known reward calibration issue:** `fraud_caught_reward_rate=0.1` creates a regime where the RL-optimal strategy diverges from the real-world optimal. See Finding 8 in the blog post and `RewardConfig` in `environment/models.py`. Fix: set `fraud_caught_reward_rate=1.0`.

---

## Running the Experiments

All experiment results are pre-computed in `experiments/*/results/*.json`. To reproduce:

```bash
# Baseline comparison (all 7 agents, 20 episodes)
python experiments/01_baseline_comparison/run.py --n-episodes 20
python experiments/01_baseline_comparison/run.py --n-episodes 20 --include-deepseek

# Budget ablation (rule-based agents)
python experiments/02_budget_ablation/run.py

# Memory ablation (rule-based agents)
python experiments/03_memory_ablation/run.py

# REINFORCE training (500 episodes, ~56 seconds)
python experiments/04_reinforce_poc/run.py
```

LLM experiments require an `OPENROUTER_API_KEY` environment variable. Budget: ~$1–2 for full DeepSeek runs.

---

## AgentBeats Integration

**Green Agent** (evaluator): runs claim episodes and scores competitor agents via A2A protocol.

```bash
# Build and push green agent image
docker build -f green_agent/Dockerfile . -t ghcr.io/shylane/healthcare-fraud-openenv-evaluator:latest
docker push ghcr.io/shylane/healthcare-fraud-openenv-evaluator:latest
```

See `scenario.toml` for leaderboard submission config (requires AgentBeats registration at agentbeats.dev).

---

## Resources

- [OpenEnv SDK](https://github.com/meta-pytorch/OpenEnv)
- [AgentBeats](https://docs.agentbeats.dev/)
- [Competition](https://rdi.berkeley.edu/agentx-agentbeats)
- [HF Blog Post](https://huggingface.co/spaces/shylane/healthcare-fraud-openenv)
