"""
Evaluation Harness — runs any agent through N episodes and collects metrics.

Designed for Option D: measuring how different agent strategies exploit
the budget-constrained, memory-augmented fraud detection environment.

Usage:
    from evaluation.harness import run_agent
    from evaluation.agents import BudgetAwareAgent

    agent = BudgetAwareAgent(api_key="...", model="openai/gpt-oss-120b:free")
    results = run_agent(agent, n_episodes=10, claims_per_episode=100, seed=42)
    print(results.summary())
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Protocol

# Make environment importable from repo root
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from environment.server.environment import ClaimsFraudEnvironment, EnvironmentConfig  # noqa: E402
from environment.models import ClaimAction  # noqa: E402


# ---------------------------------------------------------------------------
# Agent protocol — any object with act(prompt) -> str qualifies
# ---------------------------------------------------------------------------

class Agent(Protocol):
    name: str

    def act(self, prompt: str) -> str:
        """Given the environment prompt, return a response string."""
        ...

    def reset(self) -> None:
        """Called before each episode. Override to clear per-episode state."""
        ...


# ---------------------------------------------------------------------------
# Per-step record
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    episode: int
    step: int
    decision: Optional[str]
    reward: float
    is_fraud: bool            # ground truth for this claim
    budget_remaining: int
    investigation_memory_size: int
    response_length: int
    latency_ms: float         # time to get response from agent


# ---------------------------------------------------------------------------
# Per-episode record
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    episode: int
    total_reward: float
    steps: int

    # Fraud detection metrics
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    # Financial metrics
    fraud_caught_amount: float
    fraud_missed_amount: float
    investigation_cost: float
    false_positive_cost: float
    net_savings: float

    # Budget / memory usage
    investigations_used: int
    investigation_budget: int
    budget_utilization: float       # investigations_used / investigation_budget
    providers_investigated: int     # unique providers in memory at episode end

    # Budget conservation — key metric for multi-step reasoning
    # Counts steps where agent correctly used FLAG_REVIEW (not INVESTIGATE)
    # when budget was ≤ 20% remaining
    budget_conservation_correct: int
    budget_conservation_total: int
    budget_conservation_rate: float

    # Memory reuse — did agent avoid re-investigating known providers?
    memory_reuse_opportunities: int   # times known provider appeared
    memory_reuse_correct: int         # times agent used FLAG_REVIEW instead of INVESTIGATE
    memory_reuse_rate: float

    # LLM quality
    valid_response_rate: float
    mean_latency_ms: float
    mean_response_length: float

    # All step records (can be large — serialized separately)
    steps_data: list[StepRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Aggregate results across episodes
# ---------------------------------------------------------------------------

@dataclass
class EvalResults:
    agent_name: str
    model: str
    n_episodes: int
    claims_per_episode: int
    fraud_rate: float
    seed: Optional[int]
    timestamp: str
    total_wall_time_s: float

    episodes: list[EpisodeResult] = field(default_factory=list)

    # Aggregate stats (computed after all episodes)
    mean_reward: float = 0.0
    std_reward: float = 0.0
    mean_f1: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_net_savings: float = 0.0
    mean_budget_utilization: float = 0.0
    mean_budget_conservation_rate: float = 0.0
    mean_memory_reuse_rate: float = 0.0
    mean_valid_response_rate: float = 0.0
    mean_latency_ms: float = 0.0

    def finalize(self) -> "EvalResults":
        """Compute aggregate stats from episode results."""
        if not self.episodes:
            return self
        n = len(self.episodes)

        def mean(vals):
            return sum(vals) / n

        rewards = [e.total_reward for e in self.episodes]
        self.mean_reward = mean(rewards)
        variance = mean([(r - self.mean_reward) ** 2 for r in rewards])
        self.std_reward = variance ** 0.5

        self.mean_f1 = mean([e.f1 for e in self.episodes])
        self.mean_precision = mean([e.precision for e in self.episodes])
        self.mean_recall = mean([e.recall for e in self.episodes])
        self.mean_net_savings = mean([e.net_savings for e in self.episodes])
        self.mean_budget_utilization = mean([e.budget_utilization for e in self.episodes])
        self.mean_budget_conservation_rate = mean(
            [e.budget_conservation_rate for e in self.episodes]
        )
        self.mean_memory_reuse_rate = mean(
            [e.memory_reuse_rate for e in self.episodes
             if e.memory_reuse_opportunities > 0]
        ) if any(e.memory_reuse_opportunities > 0 for e in self.episodes) else 0.0
        self.mean_valid_response_rate = mean([e.valid_response_rate for e in self.episodes])
        self.mean_latency_ms = mean([e.mean_latency_ms for e in self.episodes])
        return self

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"Agent: {self.agent_name}  |  Model: {self.model}",
            f"Episodes: {self.n_episodes}  |  Claims/ep: {self.claims_per_episode}  "
            f"|  Fraud rate: {self.fraud_rate:.0%}",
            f"{'='*60}",
            f"  Reward        mean={self.mean_reward:+.2f}  std={self.std_reward:.2f}",
            f"  F1            {self.mean_f1:.3f}   "
            f"(P={self.mean_precision:.3f}  R={self.mean_recall:.3f})",
            f"  Net savings   ${self.mean_net_savings:,.0f}/episode",
            f"  Budget use    {self.mean_budget_utilization:.0%} of investigations used",
            f"  Budget cons.  {self.mean_budget_conservation_rate:.0%}  "
            f"(correct FLAG_REVIEW when budget <20%)",
            f"  Memory reuse  {self.mean_memory_reuse_rate:.0%}  "
            f"(avoided re-investigating known providers)",
            f"  Valid resp.   {self.mean_valid_response_rate:.0%}",
            f"  Latency       {self.mean_latency_ms:.0f}ms/step",
            f"{'='*60}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict (for JSON export). Strips large step_data by default."""
        d = asdict(self)
        # Strip step-level data for summary files
        for ep in d.get("episodes", []):
            ep.pop("steps_data", None)
        return d

    def save(self, path: Path, include_steps: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        if not include_steps:
            for ep in d.get("episodes", []):
                ep.pop("steps_data", None)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"Saved results to {path}")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_agent(
    agent: Agent,
    n_episodes: int = 10,
    claims_per_episode: int = 100,
    fraud_rate: float = 0.05,
    investigation_budget: int = 15,
    seed: Optional[int] = 42,
    verbose: bool = True,
) -> EvalResults:
    """
    Run agent through n_episodes full episodes. Returns EvalResults.

    Each episode:
    1. Reset environment (seeded for reproducibility)
    2. Call agent.act(prompt) at each step
    3. Submit response to environment
    4. Track all metrics including budget conservation and memory reuse

    The seed is incremented by episode index so each episode is unique
    but reproducible: episode k always uses seed + k.
    """
    import datetime

    t_start = time.time()
    results = EvalResults(
        agent_name=agent.name,
        model=getattr(agent, "model", "N/A"),
        n_episodes=n_episodes,
        claims_per_episode=claims_per_episode,
        fraud_rate=fraud_rate,
        seed=seed,
        timestamp=datetime.datetime.now().isoformat(),
        total_wall_time_s=0.0,
    )

    for ep_idx in range(n_episodes):
        ep_seed = (seed + ep_idx) if seed is not None else None
        env_config = EnvironmentConfig(
            claims_per_episode=claims_per_episode,
            fraud_rate=fraud_rate,
            investigation_budget=investigation_budget,
            seed=ep_seed,
        )
        env = ClaimsFraudEnvironment(env_config)
        obs = env.reset()

        agent.reset()

        step_records: list[StepRecord] = []
        # Budget conservation tracking
        budget_conservation_correct = 0
        budget_conservation_total = 0
        # Memory reuse tracking
        memory_reuse_opportunities = 0
        memory_reuse_correct = 0

        step = 0
        while not obs.done:
            prompt = obs.prompt or ""
            state = env.state
            budget_remaining = state.budget_remaining
            budget_pct = budget_remaining / investigation_budget
            investigation_memory = state.investigation_memory

            # Check if current provider is in memory (memory reuse opportunity)
            current_provider = obs.provider_profile.get("provider_id", "")
            provider_in_memory = current_provider in investigation_memory

            t0 = time.perf_counter()
            response = agent.act(prompt)
            latency_ms = (time.perf_counter() - t0) * 1000

            action = ClaimAction(response_text=response)
            action.parse_response()
            decision = action.parsed_decision

            # Track budget conservation: when budget < 20%, should use FLAG_REVIEW not INVESTIGATE
            if budget_pct < 0.20 and env._current_is_fraud:
                budget_conservation_total += 1
                if decision in ("FLAG_REVIEW", "DENY"):
                    budget_conservation_correct += 1

            # Track memory reuse: when provider is known, should use FLAG_REVIEW not INVESTIGATE
            if provider_in_memory:
                memory_reuse_opportunities += 1
                if decision in ("FLAG_REVIEW", "DENY", "APPROVE"):
                    # Didn't waste an investigation on a known provider
                    memory_reuse_correct += 1

            obs = env.step(action)

            step_records.append(StepRecord(
                episode=ep_idx,
                step=step,
                decision=decision,
                reward=obs.reward or 0.0,
                is_fraud=env._current_is_fraud if hasattr(env, "_current_is_fraud") else False,
                budget_remaining=budget_remaining,
                investigation_memory_size=len(investigation_memory),
                response_length=len(response),
                latency_ms=latency_ms,
            ))
            step += 1

        final_state = env.state
        tp = final_state.true_positives
        fp = final_state.false_positives
        tn = final_state.true_negatives
        fn = final_state.false_negatives
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        investigations_used = investigation_budget - final_state.budget_remaining
        budget_utilization = investigations_used / investigation_budget

        valid_steps = [s for s in step_records if s.decision is not None]
        mean_latency = (
            sum(s.latency_ms for s in step_records) / len(step_records)
            if step_records else 0.0
        )
        mean_length = (
            sum(s.response_length for s in step_records) / len(step_records)
            if step_records else 0.0
        )

        ep_result = EpisodeResult(
            episode=ep_idx,
            total_reward=final_state.cumulative_reward,
            steps=step,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            fraud_caught_amount=final_state.fraud_amount_caught,
            fraud_missed_amount=final_state.fraud_amount_missed,
            investigation_cost=final_state.investigation_cost,
            false_positive_cost=final_state.false_positive_cost,
            net_savings=final_state.net_savings,
            investigations_used=investigations_used,
            investigation_budget=investigation_budget,
            budget_utilization=budget_utilization,
            providers_investigated=len(final_state.investigation_memory),
            budget_conservation_correct=budget_conservation_correct,
            budget_conservation_total=budget_conservation_total,
            budget_conservation_rate=(
                budget_conservation_correct / max(budget_conservation_total, 1)
            ),
            memory_reuse_opportunities=memory_reuse_opportunities,
            memory_reuse_correct=memory_reuse_correct,
            memory_reuse_rate=(
                memory_reuse_correct / max(memory_reuse_opportunities, 1)
            ),
            valid_response_rate=len(valid_steps) / max(step, 1),
            mean_latency_ms=mean_latency,
            mean_response_length=mean_length,
            steps_data=step_records,
        )
        results.episodes.append(ep_result)

        if verbose:
            print(
                f"  Ep {ep_idx+1}/{n_episodes}: "
                f"reward={ep_result.total_reward:+.1f}  "
                f"F1={f1:.3f}  "
                f"budget_used={investigations_used}/{investigation_budget}  "
                f"memory_reuse={memory_reuse_correct}/{max(memory_reuse_opportunities,1)}"
            )

    results.total_wall_time_s = time.time() - t_start
    results.finalize()

    if verbose:
        print(results.summary())

    return results
