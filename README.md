# Healthcare Claims Fraud Detection — OpenEnv Challenge

> **AgentX-AgentBeats / OpenEnv Track** | Phase 2 Sprint 1 | April 12, 2026

A rigorous evaluation study: does budget/memory-aware prompting make LLM agents
better fraud detectors? We built a sequential RL environment, ran 4 agent types
(random, rule-based, naive LLM, budget-aware LLM), and measured the results
across 20 episodes × 100 claims each.

**Key finding:** A budget-aware prompt improves the same LLM by **2.7×**. A naive
LLM with no guidance scores *worse than random*. A rule-based heuristic beats
every LLM without a single API call.

> Blog post: [HF Blog — shylane/healthcare-fraud-rl](https://huggingface.co/blog/shylane/healthcare-fraud-rl) *(publishing April 2026)*

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/shylane/healthcare-fraud-openenv
cd healthcare-fraud-openenv

# Run the environment server
cd environment
uv pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal — run the trained agent
curl -X POST http://localhost:8000/reset
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"response_text": "Decision: INVESTIGATE\nRationale: High billing anomaly.\nEvidence: claim_amount=$5000 vs provider avg=$500"}'
```

## Repository Structure

```
healthcare-fraud-openenv/
├── environment/       # OpenEnv-compatible fraud detection environment
│   ├── server/        # FastAPI server (reset/step/state/a2a endpoints)
│   ├── models.py      # ClaimObservation, ClaimAction, ClaimState
│   ├── client.py      # HTTP client for the environment
│   ├── claims_simulator.py  # Synthetic fraud data generator
│   └── openenv.yaml   # HF Hub manifest
├── green_agent/       # AgentBeats evaluator (Docker + A2A)
│   ├── src/           # server.py, executor.py, agent.py, messenger.py
│   └── Dockerfile
├── purple_agent/      # AgentBeats competitor — our trained RL model
│   ├── src/
│   └── Dockerfile
├── training/          # GSPO training pipeline
│   ├── train_gspo_v2.py      # Main GRPO/GSPO training script
│   ├── harvest_episodes.py   # Episode data collection
│   └── orchestrate_cycles_v2.py  # Multi-cycle orchestration
├── scripts/
│   ├── setup_vastai.sh       # Vast.ai environment bootstrap
│   └── early_probe.sh        # Pre-flight validation (< $0.10)
└── scenario.toml      # AgentBeats leaderboard submission config
```

## The Environment

**Task**: Sequential claim review with a 15-investigation budget per episode.

**Actions**: `APPROVE | FLAG_REVIEW | INVESTIGATE | DENY | REQUEST_INFO`

**Reward signal** (8 components):
- Decision correctness (fraud vs legitimate)
- Reasoning quality (structured rationale)
- Budget management (investigation cost)
- Memory utilization (provider history)
- Conciseness penalty
- Response parsability
- Investigation warmup bonus
- Fraud detection bonus

**Fraud patterns**: Upcoding, phantom billing, duplicate claims, unbundling,
provider collusion (synthesised via configurable `ClaimsSimulator`).

## Training

5 RL training cycles using GSPO on a local RTX 4060 (16GB), then a final
run on Vast.ai RTX 3090 (24GB) for quality:

| Cycle | Start Reward | End Reward | Notes |
|-------|-------------|------------|-------|
| C1 | -3.32 | +0.22 | Heuristic baseline |
| C2 | -0.67 | +1.10 | ⚠️ approve-all collapse |
| C3 | -0.55 | -0.53 | Rev 3 reward fixes |
| C4 | +0.51 | +1.08 | Rev 4 anti-hack rewards |
| C5 | +1.00 | -0.03 | Overfitting on C4 data |

Rev 4 key fix: `-0.3` baseline for approving legitimate claims stops pure
approve-all hacking. Final cloud run uses `train_gspo_v2.py` with SIGTERM
handling and automatic checkpoint resumption.

## AgentBeats Integration

**Green Agent** (evaluator): Runs claim episodes and scores the purple agent.
Docker image: `ghcr.io/shylane/healthcare-fraud-green:latest`

**Purple Agent** (competitor): Serves the trained Qwen3-1.7B model.
Docker image: `ghcr.io/shylane/healthcare-fraud-purple:latest`

See `scenario.toml` for leaderboard submission config.

## Resources

- [OpenEnv SDK](https://github.com/meta-pytorch/OpenEnv)
- [AgentBeats](https://docs.agentbeats.dev/)
- [Competition](https://rdi.berkeley.edu/agentx-agentbeats)
- [HF Blog Post](https://huggingface.co/blog/shylane/healthcare-fraud-rl) *(coming soon)*
