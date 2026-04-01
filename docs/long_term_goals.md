# Long-Term Goals

*Created: 2026-04-01. Review and update as goals are completed or refined.*

---

## Goal 1 — File upstream OSS bug reports (Career / Community Cred)

We discovered and documented 7 real bugs/incompatibilities across major ML libraries
during the GRPO training work. Each is a concrete contribution opportunity.

**Why it matters:** Filing well-documented issues (or PRs with fixes) in popular
repos builds public AI engineering credibility faster than most other activities.
These bugs affect hundreds of people running GRPO/GSPO training.

**Status file:** `docs/upstream_issues.md` — all bugs already documented with
root cause, symptoms, minimal fix, and workaround.

### Prioritised action list

| Priority | Repo | Issue | Action |
|----------|------|-------|--------|
| 🔴 HIGH | `unslothai/unsloth` | RoPE shape mismatch in KV-cache inference (`qwen3.py`, `llama.py`) | File issue + PR with 2-line fix |
| 🔴 HIGH | `unslothai/unsloth` | `torch_amp_custom_fwd` dtype mismatch (PyTorch 2.4 API change) | File issue + PR fixing `utils.py` |
| 🟡 MED | `unslothai/unsloth-zoo` | `trl<=0.24.0` constraint too strict, blocks trl 0.29+ | File issue, suggest relaxing bound |
| 🟡 MED | `unslothai/unsloth` | `get_statistics()` crashes on DNS failure, needs try/except | File issue + 3-line PR fix |
| 🟡 MED | `unslothai/unsloth` | Incompatibility with `transformers>=5.4.0` (`auto_docstring`) | File issue, suggest exec globals fix |
| 🟢 LOW | `huggingface/trl` | Document `importance_sampling_level` added in 0.29 only | Add note to GRPO tutorial |
| 🟢 LOW | `vast-ai/vast-ai-client` | Document CDI device error pattern + workaround | File issue or add to community wiki |

### How to approach

1. Search each repo's existing issues first — many may already be filed.
2. If existing: add a comment with your specific reproduction case and stack trace.
3. If new: file a detailed issue using the format in `upstream_issues.md`.
4. For the two HIGH items: prepare a PR. The fixes are already known — it's just
   implementing and testing them in the library's test suite.
5. Mention your fix in the issue even before the PR is ready.

**Time estimate:** 2–4 hours per issue filing; 1–2 days per PR (need to fork,
write tests, open PR).

---

## Goal 2 — Hackathon: AgentX-AgentBeats Option D (Immediate — April 12 deadline)

**Repository:** `github.com/shylane/healthcare-fraud-openenv`

**What we're building:** A rigorous evaluation study showing that our
budget-constrained, memory-augmented fraud detection environment creates
qualitatively harder challenges than naive classification tasks — and that
single-step RL (GRPO) fundamentally cannot exploit the multi-step structure.

### Four experiments

| Experiment | Location | Status |
|-----------|----------|--------|
| 01 — Baseline agent comparison (random vs threshold vs GLM vs DeepSeek) | `experiments/01_baseline_comparison/` | TODO |
| 02 — Budget constraint ablation (what happens with unlimited budget?) | `experiments/02_budget_ablation/` | TODO |
| 03 — Memory ablation (what happens without investigation memory?) | `experiments/03_memory_ablation/` | TODO |
| 04 — REINFORCE proof-of-concept (show multi-step signal exists) | `experiments/04_reinforce_poc/` | TODO |

### Required deliverables (April 12)
1. `environment/` deployed as HF Hub Space (OpenEnv-compatible)
2. Green Agent with Claude API (GLM-4.7-Flash as cost-efficient backend)
3. Blog post on HuggingFace
4. GitHub repo with experiments + results
5. `scenario.toml` for AgentBeats leaderboard

**Key insight to demonstrate:** Budget-awareness gap. An agent without explicit
budget reasoning vs one with it — measurable score difference on our environment.

---

## Goal 3 — Multi-step RL: Build the Right Training System

After the hackathon, build a proper episode-level RL training system that can
actually exploit the environment's multi-step structure. This is a real research
contribution.

### What single-step GRPO cannot do (documented problem)

GRPO optimises per-step reward with no gradient signal across steps. An agent
that investigates provider `P-001` at step 12 (costing budget) to gain memory
that helps at steps 34, 67, 89 cannot receive credit for that decision.

### What's needed for proper multi-step RL

**Component 1: Episode trajectory collector**
- Runs full 100-step episodes, collects `(prompt, completion, log_probs, reward)`
- Must compute per-token log-probs for policy gradient
- Needs batched parallel episode collection (16+ episodes per gradient update)

**Component 2: Return computation**
- Discounted returns with γ=0.99 for 100-step horizon
- GAE (Generalized Advantage Estimation, λ=0.95) to reduce variance
- Return normalisation across the episode batch

**Component 3: Value function / critic**
- Estimates expected future return from current state (budget, memory, claims remaining)
- Options: linear head on LLM hidden state; separate small MLP; GPT2-style value model
- Must be trained jointly with policy (actor-critic) or separately

**Component 4: PPO-clip loss over episodes**
- Standard PPO with episode-level advantages
- KL penalty to prevent catastrophic forgetting
- Gradient clipping (max_norm=1.0)

**Estimated compute requirement:**
- 16 episodes × 100 steps × 150 tokens/step = 240K tokens generation per update
- On A100 40GB (1.7B model): ~8-12 minutes per update step
- 500 updates to convergence ≈ 70-100 GPU hours ≈ $35-50 on Vast.ai at $0.50/hr

**Alternative: Offline-to-Online (practical shortcut)**
1. Collect 200 expert episodes using Claude API (~$2)
2. SFT on top-50 episodes by score (1-2 hours, ~$2 GPU)
3. Run online REINFORCE from this warm-started policy (10× fewer episodes to convergence)

**Reference papers to study:**
- PPO (Schulman 2017) — foundational
- REINFORCE with baseline (Williams 1992)
- GAE (Schulman 2015)
- Decision Transformer (Chen 2021) — alternative: sequence modelling as RL
- RLOO (Leave-One-Out REINFORCE, used in TRL for alignment) — applicable here

**Repository location:** Create `training/multi_step_rl/` after hackathon

---

## Goal 4 — Kaggle Tutorial: RL from Scratch with a Custom Environment

A notebook tutorial series showing how to build an RL environment, train an agent,
understand reward hacking, and design better rewards. Using a toy problem so
readers can run it without GPU access.

### Why Kaggle

- Huge audience of ML practitioners who don't know RL
- Notebook format forces clarity
- Can reuse the healthcare fraud environment as the "real-world" example after
  showing the toy version
- Builds personal brand in the "RL for LLMs" space

### Proposed notebook series

**Notebook 1: Build Your First RL Environment** (~30 min read)
- What is an environment? State, action, reward, done.
- Build a `CoinFlipEnv` in 50 lines (step, reset, render)
- Test with a random agent
- Show how reward shaping changes agent behaviour

**Notebook 2: Your First Policy Gradient Agent**
- REINFORCE from scratch in pure PyTorch (no gym, no framework)
- Train on CoinFlipEnv
- Visualise learning curve
- Explain why vanilla REINFORCE has high variance

**Notebook 3: Reward Hacking — When Your Agent Cheats**
- Introduce a "badly designed" reward
- Show the agent exploiting it (step by step)
- Show the fix (real case from our healthcare fraud training)
- General principles for reward design

**Notebook 4: Scale Up — LLMs as RL Policies**
- Move from CoinFlip to healthcare fraud environment
- Show how text generation = action selection
- Show why per-step RL (GRPO) misses multi-step dynamics
- Motivate episode-level RL

**Timeline:** Post after hackathon (April-May 2026)

---

## Tracking

| Goal | Urgency | Effort | Status |
|------|---------|--------|--------|
| 1 — OSS bug reports | Medium | Low-Medium | Not started |
| 2 — Hackathon Option D | HIGH (deadline Apr 12) | Medium | In progress |
| 3 — Multi-step RL system | Low | High | Not started (post-hackathon) |
| 4 — Kaggle tutorial | Low | Medium | Not started (post-hackathon) |

*Last updated: 2026-04-01*
