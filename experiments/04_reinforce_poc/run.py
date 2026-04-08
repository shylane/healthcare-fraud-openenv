"""
Experiment 04 — REINFORCE Policy Gradient PoC

Demonstrates that a simple REINFORCE agent CAN learn a multi-step policy
on this environment — something GRPO cannot do because it treats each
step as an independent 1-step optimization.

Architecture:
    - Feature extractor: reads claim_features + budget_remaining + memory_size
    - Policy: linear softmax over 3 actions (APPROVE / FLAG_REVIEW / INVESTIGATE)
    - Optimizer: vanilla policy gradient (REINFORCE, no baseline initially)
    - Optional: REINFORCE with baseline (mean reward subtraction)

The key signal: reward shaping that favors budget conservation.
After ~200 episodes, the policy should learn to:
    1. INVESTIGATE high-risk claims early in episode
    2. Switch to FLAG_REVIEW as budget depletes
    3. APPROVE low-risk claims to avoid false-positive costs

This is impossible with GRPO because GRPO optimizes a single-step KL-div loss
and has no credit assignment across the 100-step episode horizon.

Usage:
    python experiments/04_reinforce_poc/run.py
    python experiments/04_reinforce_poc/run.py --episodes 500 --lr 0.01
    python experiments/04_reinforce_poc/run.py --quick    # 50 episodes smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from environment.server.environment import ClaimsFraudEnvironment, EnvironmentConfig
from environment.models import ClaimAction, ClaimObservation

RESULTS_DIR = Path(__file__).parent / "results"

# Actions the REINFORCE agent can choose
# Using 3 actions for tractability (DENY and REQUEST_INFO excluded)
ACTIONS = ["APPROVE", "FLAG_REVIEW", "INVESTIGATE"]
N_ACTIONS = len(ACTIONS)


# ---------------------------------------------------------------------------
# Feature extraction — maps environment state + claim to a flat vector
# ---------------------------------------------------------------------------

def extract_features(obs: ClaimObservation, budget_remaining: int,
                     investigation_budget: int, memory_size: int) -> List[float]:
    """
    Compact 10-feature vector for the policy network.

    Features:
        0: budget fraction remaining (0-1)
        1: memory size (normalized by budget)
        2: claim amount (log-scaled, normalized)
        3: provider risk score (0-1)
        4: member risk score (0-1)
        5: fraud flag rate for provider (0-1)
        6: amount z-score (clipped to [-3, 3], normalized)
        7: is provider in memory (binary)
        8: step fraction through episode (0-1)
        9: high-cost procedure rate (0-1)
    """
    features = obs.claim_features
    provider = obs.provider_profile
    budget_frac = budget_remaining / max(investigation_budget, 1)
    memory_frac = memory_size / max(investigation_budget, 1)
    claim_amount = obs.claim_amount
    log_amount = math.log1p(claim_amount) / math.log1p(50000)  # normalize up to 50k
    provider_risk = min(1.0, max(0.0, features.get("provider_risk_score", 0.0)))
    member_risk = min(1.0, max(0.0, features.get("member_risk_score", 0.0)))
    fraud_flag_rate = min(1.0, provider.get("fraud_flag_rate", 0.0))
    amount_z = features.get("amount_zscore", 0.0)
    amount_z_norm = max(-1.0, min(1.0, amount_z / 3.0))
    # is provider in memory: passed as argument
    provider_in_mem = 0.0  # filled in caller
    # step fraction: not available here directly, handled in caller
    step_frac = 0.0
    high_cost_rate = min(1.0, provider.get("high_cost_procedure_rate", 0.0))

    return [
        budget_frac,
        memory_frac,
        log_amount,
        provider_risk,
        member_risk,
        fraud_flag_rate,
        amount_z_norm,
        provider_in_mem,  # index 7 — set by caller
        step_frac,        # index 8 — set by caller
        high_cost_rate,
    ]


# ---------------------------------------------------------------------------
# Linear policy — softmax over ACTIONS, parameters = W (N_ACTIONS × N_FEATURES)
# ---------------------------------------------------------------------------

N_FEATURES = 10

class LinearPolicy:
    """
    Single-layer linear policy: logits = W @ features + b
    Updated via REINFORCE gradient.
    """

    def __init__(self, lr: float = 0.001, seed: int = 0):
        rng = random.Random(seed)
        # Xavier-ish init: small random weights
        self.W = [
            [rng.gauss(0, 0.1) for _ in range(N_FEATURES)]
            for _ in range(N_ACTIONS)
        ]
        self.b = [0.0] * N_ACTIONS
        self.lr = lr

    def logits(self, features: List[float]) -> List[float]:
        return [
            sum(self.W[a][i] * features[i] for i in range(N_FEATURES)) + self.b[a]
            for a in range(N_ACTIONS)
        ]

    def softmax(self, logits: List[float]) -> List[float]:
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def action_probs(self, features: List[float]) -> List[float]:
        return self.softmax(self.logits(features))

    def sample_action(self, features: List[float]) -> Tuple[int, float]:
        """Sample action index and return (action_idx, log_prob)."""
        probs = self.action_probs(features)
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i, math.log(max(p, 1e-10))
        return N_ACTIONS - 1, math.log(max(probs[-1], 1e-10))

    def update(
        self,
        trajectories: List[dict],
        baseline: float = 0.0,
        entropy_coef: float = 0.05,
        max_grad_norm: float = 0.5,
    ) -> None:
        """
        REINFORCE batch update with three stability fixes:

        Fix 1 — Batch accumulation: accumulate gradients over the full episode,
            divide by episode length, then apply once.  The old per-step loop
            gave an effective lr = lr × T (≈ 0.005 × 100 = 0.5) which is why
            the policy collapsed after a single episode.

        Fix 2 — Advantage normalisation: standardise the advantage signal to
            zero-mean / unit-variance before computing gradients.  Raw returns
            of ~-2800 produced enormous gradient magnitudes.

        Fix 3 — Entropy regularisation: add a bonus proportional to
            -log(p_chosen) so the policy is penalised for being too certain.
            This prevents probability mass from collapsing to one action.
        """
        if not trajectories:
            return

        n = len(trajectories)

        # --- Fix 2: normalise advantages ---
        raw_advantages = [t["return"] - baseline for t in trajectories]
        mean_adv = sum(raw_advantages) / n
        var_adv = sum((a - mean_adv) ** 2 for a in raw_advantages) / n
        std_adv = max(var_adv ** 0.5, 1e-8)
        norm_advantages = [(a - mean_adv) / std_adv for a in raw_advantages]

        # --- Fix 1: accumulate gradients into dW / db ---
        dW = [[0.0] * N_FEATURES for _ in range(N_ACTIONS)]
        db = [0.0] * N_ACTIONS

        for traj, adv in zip(trajectories, norm_advantages):
            features = traj["features"]
            action_idx = traj["action_idx"]
            probs = self.action_probs(features)

            for a in range(N_ACTIONS):
                indicator = 1.0 if a == action_idx else 0.0
                score = indicator - probs[a]

                # Policy gradient term
                pg = adv * score

                # Fix 3: entropy bonus — -log(p_chosen) boosts gradient when
                # the policy is over-confident about the chosen action
                ent = entropy_coef * (-math.log(max(probs[action_idx], 1e-10)) - 1.0) * score

                total = pg + ent
                for i in range(N_FEATURES):
                    dW[a][i] += total * features[i]
                db[a] += total

        # Average over episode length, clip, then apply
        for a in range(N_ACTIONS):
            for i in range(N_FEATURES):
                g = dW[a][i] / n
                g = max(-max_grad_norm, min(max_grad_norm, g))
                self.W[a][i] += self.lr * g
            g_b = db[a] / n
            g_b = max(-max_grad_norm, min(max_grad_norm, g_b))
            self.b[a] += self.lr * g_b

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"W": self.W, "b": self.b, "lr": self.lr}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LinearPolicy":
        with open(path) as f:
            d = json.load(f)
        p = cls(lr=d["lr"])
        p.W = d["W"]
        p.b = d["b"]
        return p


# ---------------------------------------------------------------------------
# REINFORCE agent wrapper — exposes the Agent protocol
# ---------------------------------------------------------------------------

class ReinforceAgent:
    """
    Wraps a LinearPolicy as an Agent for the evaluation harness.
    Used to benchmark trained policy against other agents.
    """
    name = "ReinforceAgent"
    model = "reinforce-linear"

    def __init__(self, policy: LinearPolicy, investigation_budget: int = 15):
        self.policy = policy
        self.investigation_budget = investigation_budget
        self._step = 0
        self._memory_size = 0

    def reset(self) -> None:
        self._step = 0
        self._memory_size = 0

    def act(self, prompt: str) -> str:
        # REINFORCE agent doesn't use the text prompt — it gets features
        # directly from the environment. For harness compatibility, we parse
        # budget and memory from the prompt text.
        import re
        budget_match = re.search(r"Budget:\s*(\d+)/(\d+)\s+remaining", prompt)
        budget_remaining = int(budget_match.group(1)) if budget_match else self.investigation_budget
        mem_lines = len(re.findall(r"^- ⚠", prompt, re.MULTILINE))

        features = [0.0] * N_FEATURES
        features[0] = budget_remaining / self.investigation_budget
        features[1] = mem_lines / max(self.investigation_budget, 1)
        # Parse claim amount
        amount_match = re.search(r"Billed Amount\*\*:\s*\$([\d,]+\.?\d*)", prompt)
        if amount_match:
            amount = float(amount_match.group(1).replace(",", ""))
            features[2] = math.log1p(amount) / math.log1p(50000)
        # Risk level
        if "Risk Level: HIGH" in prompt or "Risk Assessment**: HIGH" in prompt:
            features[3] = 0.8
        elif "Risk Level: Moderate" in prompt or "Risk Assessment**: MODERATE" in prompt:
            features[3] = 0.5
        else:
            features[3] = 0.2
        # Fraud flag rate
        ff_match = re.search(r"Prior Fraud Flags\*\*:\s*([\d.]+)%", prompt)
        if ff_match:
            features[5] = float(ff_match.group(1)) / 100
        # Provider in memory
        features[7] = 1.0 if "KNOWN PROVIDER" in prompt else 0.0
        # Step fraction
        step_match = re.search(r"Step:\s*(\d+)/(\d+)", prompt)
        if step_match:
            features[8] = int(step_match.group(1)) / max(int(step_match.group(2)), 1)

        action_idx, _ = self.policy.sample_action(features)
        decision = ACTIONS[action_idx]
        return (
            f"Decision: {decision}\n"
            f"Rationale: REINFORCE policy decision.\n"
            f"Evidence: budget={budget_remaining}, mem={mem_lines}"
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    episode: int
    total_reward: float
    mean_reward_window: float
    investigations_used: int
    fraud_caught: int
    fraud_missed: int
    policy_entropy: float


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}."""
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


def train(
    n_episodes: int = 300,
    claims_per_episode: int = 100,
    fraud_rate: float = 0.05,
    investigation_budget: int = 15,
    seed: int = 42,
    lr: float = 0.1,
    gamma: float = 0.99,
    use_baseline: bool = True,
    entropy_coef: float = 0.05,
    verbose_every: int = 20,
) -> Tuple[LinearPolicy, List[TrainingMetrics]]:
    """Train a REINFORCE policy on the fraud detection environment."""

    policy = LinearPolicy(lr=lr, seed=seed)
    metrics_history: List[TrainingMetrics] = []
    reward_window: List[float] = []
    window_size = 20

    print(f"Training REINFORCE: {n_episodes} episodes x {claims_per_episode} claims")
    print(f"  lr={lr}  gamma={gamma}  entropy_coef={entropy_coef}")
    print(f"  baseline={'mean-reward (normalised)' if use_baseline else 'none'}")
    print(f"  budget={investigation_budget}  fraud_rate={fraud_rate:.0%}")
    print()

    for ep_idx in range(n_episodes):
        ep_seed = seed + ep_idx
        env_config = EnvironmentConfig(
            claims_per_episode=claims_per_episode,
            fraud_rate=fraud_rate,
            investigation_budget=investigation_budget,
            seed=ep_seed,
        )
        env = ClaimsFraudEnvironment(env_config)
        obs = env.reset()

        episode_rewards: List[float] = []
        episode_trajectories: List[dict] = []

        step = 0
        while not obs.done:
            state = env.state
            budget_remaining = state.budget_remaining
            memory_size = len(state.investigation_memory)
            current_provider = obs.provider_profile.get("provider_id", "")
            provider_in_memory = current_provider in state.investigation_memory

            features = extract_features(
                obs,
                budget_remaining=budget_remaining,
                investigation_budget=investigation_budget,
                memory_size=memory_size,
            )
            # Fill in caller-provided fields
            features[7] = 1.0 if provider_in_memory else 0.0
            total_steps = claims_per_episode
            features[8] = step / max(total_steps, 1)

            action_idx, log_prob = policy.sample_action(features)
            decision = ACTIONS[action_idx]

            response = (
                f"Decision: {decision}\n"
                f"Rationale: REINFORCE policy.\n"
                f"Evidence: step={step}, budget={budget_remaining}"
            )
            action = ClaimAction(response_text=response)
            action.parse_response()
            obs = env.step(action)

            step_reward = obs.reward or 0.0
            episode_rewards.append(step_reward)
            episode_trajectories.append({
                "features": features,
                "action_idx": action_idx,
                "log_prob": log_prob,
                "reward": step_reward,
                "return": 0.0,  # filled below
            })
            step += 1

        # Compute discounted returns
        returns = compute_returns(episode_rewards, gamma=gamma)
        for traj, G in zip(episode_trajectories, returns):
            traj["return"] = G

        # Baseline: mean return over episode (reduces variance)
        baseline = sum(returns) / len(returns) if use_baseline and returns else 0.0

        # REINFORCE batch update (normalised advantages + entropy + clipped)
        policy.update(episode_trajectories, baseline=baseline, entropy_coef=entropy_coef)

        # Track metrics
        total_reward = env.state.cumulative_reward
        reward_window.append(total_reward)
        if len(reward_window) > window_size:
            reward_window.pop(0)

        # Compute policy entropy (average over recent states)
        sample_feats = episode_trajectories[len(episode_trajectories)//2]["features"]
        probs = policy.action_probs(sample_feats)
        entropy = -sum(p * math.log(max(p, 1e-10)) for p in probs)

        m = TrainingMetrics(
            episode=ep_idx,
            total_reward=total_reward,
            mean_reward_window=sum(reward_window) / len(reward_window),
            investigations_used=investigation_budget - env.state.budget_remaining,
            fraud_caught=env.state.true_positives,
            fraud_missed=env.state.false_negatives,
            policy_entropy=entropy,
        )
        metrics_history.append(m)

        if verbose_every > 0 and (ep_idx + 1) % verbose_every == 0:
            print(
                f"  Ep {ep_idx+1:>4}/{n_episodes}  "
                f"reward={total_reward:+7.1f}  "
                f"win_avg={m.mean_reward_window:+7.1f}  "
                f"invests={m.investigations_used:>2}  "
                f"caught={m.fraud_caught}  "
                f"entropy={entropy:.3f}"
            )

    return policy, metrics_history


def parse_args():
    p = argparse.ArgumentParser(description="REINFORCE PoC on fraud detection environment")
    p.add_argument("--episodes", type=int, default=300,
                   help="Training episodes (default: 300)")
    p.add_argument("--claims", type=int, default=100,
                   help="Claims per episode (default: 100)")
    p.add_argument("--fraud-rate", type=float, default=0.05)
    p.add_argument("--budget", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.1,
                   help="Learning rate (default: 0.1 — compensates for divide-by-n batch accumulation)")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor (default: 0.99)")
    p.add_argument("--no-baseline", action="store_true",
                   help="Disable mean-reward baseline (more variance)")
    p.add_argument("--entropy-coef", type=float, default=0.05,
                   help="Entropy regularisation coefficient (default: 0.05)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 50 episodes × 20 claims (smoke test)")
    p.add_argument("--eval-episodes", type=int, default=10,
                   help="Evaluation episodes after training (default: 10)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.episodes = 50
        args.claims = 20
        print("Quick mode: 50 training episodes × 20 claims")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    t_start = time.time()
    policy, metrics = train(
        n_episodes=args.episodes,
        claims_per_episode=args.claims,
        fraud_rate=args.fraud_rate,
        investigation_budget=args.budget,
        seed=args.seed,
        lr=args.lr,
        gamma=args.gamma,
        use_baseline=not args.no_baseline,
        entropy_coef=args.entropy_coef,
        verbose_every=max(1, args.episodes // 10),
    )
    train_time = time.time() - t_start
    print(f"\nTraining complete in {train_time:.1f}s")

    # Save policy weights
    policy_path = RESULTS_DIR / f"{timestamp}_policy.json"
    policy.save(policy_path)
    print(f"Policy saved to {policy_path}")

    # Save training curve
    curve_path = RESULTS_DIR / f"{timestamp}_training_curve.json"
    with open(curve_path, "w") as f:
        json.dump([asdict(m) for m in metrics], f, indent=2)
    print(f"Training curve saved to {curve_path}")

    # Print learning summary
    n = len(metrics)
    first_q = metrics[:n//4]
    last_q = metrics[3*n//4:]
    first_avg = sum(m.total_reward for m in first_q) / max(len(first_q), 1)
    last_avg = sum(m.total_reward for m in last_q) / max(len(last_q), 1)

    print(f"\n{'='*60}")
    print(f"LEARNING SUMMARY")
    print(f"{'='*60}")
    print(f"  First 25% episodes: mean reward = {first_avg:+.1f}")
    print(f"  Last  25% episodes: mean reward = {last_avg:+.1f}")
    print(f"  Improvement:                      {last_avg - first_avg:+.1f}")
    if last_avg > first_avg + 10:
        print(f"  >> Policy LEARNED — reward improved significantly")
    elif last_avg > first_avg:
        print(f"  >> Policy improved marginally")
    else:
        print(f"  >> Minimal learning — try more episodes or different lr")

    # Eval: run trained agent through fresh episodes
    if args.eval_episodes > 0:
        print(f"\n{'='*60}")
        print(f"POST-TRAINING EVALUATION ({args.eval_episodes} episodes)")
        print(f"{'='*60}")
        from evaluation.harness import run_agent
        from evaluation.agents import RandomAgent, ThresholdAgent

        trained_agent = ReinforceAgent(policy, investigation_budget=args.budget)

        for agent in [trained_agent, RandomAgent(), ThresholdAgent()]:
            r = run_agent(
                agent=agent,
                n_episodes=args.eval_episodes,
                claims_per_episode=args.claims,
                fraud_rate=args.fraud_rate,
                investigation_budget=args.budget,
                seed=args.seed + 1000,  # different seed from training
                verbose=False,
            )
            print(f"\n  {agent.name}:")
            print(f"    Reward: {r.mean_reward:+.1f} ± {r.std_reward:.1f}")
            print(f"    F1: {r.mean_f1:.3f}  (P={r.mean_precision:.3f} R={r.mean_recall:.3f})")
            print(f"    Budget utilization: {r.mean_budget_utilization:.0%}")
            print(f"    Budget conservation: {r.mean_budget_conservation_rate:.0%}")
            r.save(RESULTS_DIR / f"{timestamp}_eval_{agent.name}.json")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "04_reinforce_poc",
        "config": {
            "n_episodes": args.episodes,
            "claims_per_episode": args.claims,
            "fraud_rate": args.fraud_rate,
            "investigation_budget": args.budget,
            "lr": args.lr,
            "gamma": args.gamma,
            "use_baseline": not args.no_baseline,
        },
        "training_time_s": train_time,
        "learning": {
            "first_quartile_mean_reward": first_avg,
            "last_quartile_mean_reward": last_avg,
            "improvement": last_avg - first_avg,
        },
    }
    summary_path = RESULTS_DIR / f"{timestamp}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
