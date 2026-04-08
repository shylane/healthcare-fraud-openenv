"""
Experiment 03 — Memory Halflife Ablation Study

Varies memory_decay_halflife (0/5/20/50/∞) to measure how much
investigation memory contributes to agent performance.

Key question: how much does memory decay hurt the BudgetAwareAgent?
With halflife=0, memory is useless (confidence → 0 instantly).
With halflife=∞, memory never decays — perfect recall.

The hypothesis: BudgetAwareAgent degrades most with low halflife
(since it relies on memory), while ThresholdAgent is unaffected
(it reads memory from the prompt but treats it as binary).

Usage:
    python experiments/03_memory_ablation/run.py --api-key YOUR_KEY
    python experiments/03_memory_ablation/run.py --quick --api-key YOUR_KEY
    python experiments/03_memory_ablation/run.py --no-llm
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass

from evaluation.harness import run_agent
from evaluation.agents import (
    RandomAgent,
    ThresholdAgent,
    NaiveLLMAgent,
    BudgetAwareAgent,
)

RESULTS_DIR = Path(__file__).parent / "results"
# 0 = instant decay (memory useless), 5/20 = short/normal decay, 100 = near-perfect
HALFLIFE_CONDITIONS = [0, 5, 20, 100]


def parse_args():
    p = argparse.ArgumentParser(description="Memory halflife ablation experiment")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--claims", type=int, default=100)
    p.add_argument("--fraud-rate", type=float, default=0.05)
    p.add_argument("--budget", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default="qwen/qwen3.6-plus:free")
    p.add_argument("--halflives", type=int, nargs="+", default=HALFLIFE_CONDITIONS,
                   help="Halflife values to test (default: 0 5 20 100)")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 2 episodes × 20 claims, halflives [0, 20]")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--api-key", type=str, default=None)
    return p.parse_args()


def print_ablation_table(all_data: dict) -> None:
    """Print halflife × agent performance matrix."""
    print("\n" + "=" * 90)
    print("MEMORY ABLATION — Mean Memory Reuse Rate by Agent × Halflife")
    print("=" * 90)

    halflives = sorted(all_data.keys())
    if not halflives:
        return
    agent_names = [r.agent_name for r in all_data[halflives[0]]]

    labels = [f"HL={h}" if h > 0 else "HL=0(off)" for h in halflives]
    hdr = f"{'Agent':<30}" + "".join(f"  {lbl:>10}" for lbl in labels)
    print(hdr)
    print("-" * len(hdr))

    print("Memory Reuse Rate:")
    for name in agent_names:
        row = f"  {name:<28}"
        for h in halflives:
            r = next((x for x in all_data[h] if x.agent_name == name), None)
            row += f"  {r.mean_memory_reuse_rate:>9.0%} " if r else "         N/A "
        print(row)

    print("\nMean Reward:")
    for name in agent_names:
        row = f"  {name:<28}"
        for h in halflives:
            r = next((x for x in all_data[h] if x.agent_name == name), None)
            row += f"  {r.mean_reward:>+9.0f} " if r else "         N/A "
        print(row)

    print("=" * 90)


def main():
    args = parse_args()

    if args.quick:
        args.n_episodes = 2
        args.claims = 20
        args.halflives = [0, 20]
        print("Quick mode: 2 episodes × 20 claims, halflives [0, 20]")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.no_llm:
        print("ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_data: dict = {}

    for halflife in args.halflives:
        label = f"HL{halflife}"
        print(f"\n{'#'*70}")
        print(f"# MEMORY HALFLIFE = {halflife} turns{'  (memory disabled)' if halflife == 0 else ''}")
        print(f"{'#'*70}")

        agents = [RandomAgent(), ThresholdAgent()]
        if not args.no_llm:
            agents += [
                NaiveLLMAgent(api_key=api_key, model=args.model),
                BudgetAwareAgent(api_key=api_key, model=args.model),
            ]

        halflife_results = []
        for agent in agents:
            print(f"\n  Agent: {agent.name}  |  halflife={halflife}")

            # Patch the harness to pass halflife to run_agent
            # run_agent doesn't expose halflife directly, so we pass it via
            # the environment config creation in a wrapper.
            r = _run_agent_with_halflife(
                agent=agent,
                n_episodes=args.n_episodes,
                claims_per_episode=args.claims,
                fraud_rate=args.fraud_rate,
                investigation_budget=args.budget,
                memory_halflife=halflife,
                seed=args.seed,
                verbose=True,
            )
            halflife_results.append(r)

            fname = RESULTS_DIR / f"{timestamp}_{label}_{agent.name}.json"
            r.save(fname)

        all_data[halflife] = halflife_results

        comparison = {
            "timestamp": datetime.now().isoformat(),
            "memory_halflife": halflife,
            "n_episodes": args.n_episodes,
            "agents": [
                {
                    "name": r.agent_name,
                    "model": r.model,
                    "mean_reward": r.mean_reward,
                    "mean_f1": r.mean_f1,
                    "mean_memory_reuse_rate": r.mean_memory_reuse_rate,
                    "mean_budget_conservation_rate": r.mean_budget_conservation_rate,
                }
                for r in halflife_results
            ],
        }
        cmp_path = RESULTS_DIR / f"{timestamp}_{label}_comparison.json"
        with open(cmp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"  Saved to {cmp_path}")

    print_ablation_table(all_data)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "03_memory_ablation",
        "n_episodes": args.n_episodes,
        "claims_per_episode": args.claims,
        "fraud_rate": args.fraud_rate,
        "investigation_budget": args.budget,
        "halflives_tested": args.halflives,
        "by_halflife": {
            str(h): [
                {
                    "name": r.agent_name,
                    "mean_reward": r.mean_reward,
                    "mean_f1": r.mean_f1,
                    "mean_memory_reuse_rate": r.mean_memory_reuse_rate,
                    "mean_budget_utilization": r.mean_budget_utilization,
                }
                for r in results
            ]
            for h, results in all_data.items()
        },
    }
    summary_path = RESULTS_DIR / f"{timestamp}_memory_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMemory ablation summary saved to {summary_path}")


def _run_agent_with_halflife(
    agent,
    n_episodes: int,
    claims_per_episode: int,
    fraud_rate: float,
    investigation_budget: int,
    memory_halflife: int,
    seed,
    verbose: bool,
):
    """
    Thin wrapper around the harness that injects memory_halflife into the
    EnvironmentConfig.  The standard run_agent() doesn't expose this parameter,
    so we replicate its loop with the extra config field.
    """
    import time
    import datetime as _dt
    from evaluation.harness import EvalResults, EpisodeResult, StepRecord
    from environment.server.environment import ClaimsFraudEnvironment, EnvironmentConfig
    from environment.models import ClaimAction

    t_start = time.time()
    results = EvalResults(
        agent_name=agent.name,
        model=getattr(agent, "model", "N/A"),
        n_episodes=n_episodes,
        claims_per_episode=claims_per_episode,
        fraud_rate=fraud_rate,
        seed=seed,
        timestamp=_dt.datetime.now().isoformat(),
        total_wall_time_s=0.0,
    )

    for ep_idx in range(n_episodes):
        ep_seed = (seed + ep_idx) if seed is not None else None
        env_config = EnvironmentConfig(
            claims_per_episode=claims_per_episode,
            fraud_rate=fraud_rate,
            investigation_budget=investigation_budget,
            memory_decay_halflife=memory_halflife,
            seed=ep_seed,
        )
        env = ClaimsFraudEnvironment(env_config)
        obs = env.reset()
        agent.reset()

        step_records = []
        budget_conservation_correct = 0
        budget_conservation_total = 0
        memory_reuse_opportunities = 0
        memory_reuse_correct = 0

        step = 0
        while not obs.done:
            prompt = obs.prompt or ""
            state = env.state
            budget_remaining = state.budget_remaining
            budget_pct = budget_remaining / investigation_budget
            investigation_memory = state.investigation_memory
            current_provider = obs.provider_profile.get("provider_id", "")
            provider_in_memory = current_provider in investigation_memory

            import time as _time
            t0 = _time.perf_counter()
            response = agent.act(prompt)
            latency_ms = (_time.perf_counter() - t0) * 1000

            action = ClaimAction(response_text=response)
            action.parse_response()
            decision = action.parsed_decision

            if budget_pct < 0.20 and env._current_is_fraud:
                budget_conservation_total += 1
                if decision in ("FLAG_REVIEW", "DENY"):
                    budget_conservation_correct += 1

            if provider_in_memory:
                memory_reuse_opportunities += 1
                if decision in ("FLAG_REVIEW", "DENY", "APPROVE"):
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
        mean_latency = sum(s.latency_ms for s in step_records) / max(len(step_records), 1)
        mean_length = sum(s.response_length for s in step_records) / max(len(step_records), 1)

        ep_result = EpisodeResult(
            episode=ep_idx,
            total_reward=final_state.cumulative_reward,
            steps=step,
            true_positives=tp, false_positives=fp,
            true_negatives=tn, false_negatives=fn,
            precision=precision, recall=recall, f1=f1,
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
            budget_conservation_rate=budget_conservation_correct / max(budget_conservation_total, 1),
            memory_reuse_opportunities=memory_reuse_opportunities,
            memory_reuse_correct=memory_reuse_correct,
            memory_reuse_rate=memory_reuse_correct / max(memory_reuse_opportunities, 1),
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
                f"mem_reuse={memory_reuse_correct}/{max(memory_reuse_opportunities,1)}"
            )

    results.total_wall_time_s = time.time() - t_start
    results.finalize()
    if verbose:
        print(results.summary())
    return results


if __name__ == "__main__":
    main()
