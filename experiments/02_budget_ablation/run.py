"""
Experiment 02 — Budget Ablation Study

Varies investigation_budget (5/10/15/20) across all 4 agents.
Key question: does BudgetAwareAgent adapt better to tighter budgets?

The hypothesis: NaiveLLMAgent ignores budget and wastes investigations,
so performance degrades more steeply as budget shrinks.
BudgetAwareAgent's explicit budget reasoning should make it more robust.

Usage:
    python experiments/02_budget_ablation/run.py --api-key YOUR_KEY
    python experiments/02_budget_ablation/run.py --quick --api-key YOUR_KEY
    python experiments/02_budget_ablation/run.py --no-llm  # rule-based only
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
BUDGET_CONDITIONS = [5, 10, 15, 20]


def parse_args():
    p = argparse.ArgumentParser(description="Budget ablation experiment")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--claims", type=int, default=100)
    p.add_argument("--fraud-rate", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model", type=str, default="qwen/qwen3.6-plus")
    p.add_argument("--budgets", type=int, nargs="+", default=BUDGET_CONDITIONS,
                   help="Budget values to test (default: 5 10 15 20)")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 2 episodes × 20 claims, budgets [5, 15]")
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--api-key", type=str, default=None)
    return p.parse_args()


def print_ablation_table(all_data: dict) -> None:
    """Print budget × agent performance matrix."""
    print("\n" + "=" * 90)
    print("BUDGET ABLATION — Mean Reward by Agent × Budget")
    print("=" * 90)

    # Collect agent names and budgets
    budgets = sorted(all_data.keys())
    if not budgets:
        return
    agent_names = [r.agent_name for r in all_data[budgets[0]]]

    # Header
    hdr = f"{'Agent':<30}" + "".join(f"  B={b:>2}" for b in budgets)
    print(hdr)
    print("-" * len(hdr))

    for name in agent_names:
        row = f"{name:<30}"
        for b in budgets:
            results_for_budget = all_data[b]
            r = next((x for x in results_for_budget if x.agent_name == name), None)
            row += f"  {r.mean_reward:+6.0f}" if r else "      N/A"
        print(row)

    print()
    print("BUDGET ABLATION — Budget Conservation Rate by Agent × Budget")
    print("-" * len(hdr))
    for name in agent_names:
        row = f"{name:<30}"
        for b in budgets:
            results_for_budget = all_data[b]
            r = next((x for x in results_for_budget if x.agent_name == name), None)
            row += f"  {r.mean_budget_conservation_rate:>5.0%} " if r else "      N/A"
        print(row)

    print("=" * 90)


def main():
    args = parse_args()

    if args.quick:
        args.n_episodes = 2
        args.claims = 20
        args.budgets = [5, 15]
        print("Quick mode: 2 episodes × 20 claims, budgets [5, 15]")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.no_llm:
        print("ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_data: dict = {}  # budget -> list[EvalResults]

    for budget in args.budgets:
        print(f"\n{'#'*70}")
        print(f"# BUDGET = {budget}")
        print(f"{'#'*70}")

        agents = [RandomAgent(), ThresholdAgent()]
        if not args.no_llm:
            agents += [
                NaiveLLMAgent(api_key=api_key, model=args.model),
                BudgetAwareAgent(api_key=api_key, model=args.model),
            ]

        budget_results = []
        for agent in agents:
            print(f"\n  Agent: {agent.name}")
            r = run_agent(
                agent=agent,
                n_episodes=args.n_episodes,
                claims_per_episode=args.claims,
                fraud_rate=args.fraud_rate,
                investigation_budget=budget,
                seed=args.seed,
                verbose=True,
            )
            budget_results.append(r)

            # Save per-agent result
            fname = RESULTS_DIR / f"{timestamp}_B{budget}_{agent.name}.json"
            r.save(fname)

        all_data[budget] = budget_results

        # Save per-budget comparison
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "budget": budget,
            "n_episodes": args.n_episodes,
            "agents": [
                {
                    "name": r.agent_name,
                    "model": r.model,
                    "mean_reward": r.mean_reward,
                    "std_reward": r.std_reward,
                    "mean_f1": r.mean_f1,
                    "mean_budget_utilization": r.mean_budget_utilization,
                    "mean_budget_conservation_rate": r.mean_budget_conservation_rate,
                    "mean_memory_reuse_rate": r.mean_memory_reuse_rate,
                    "mean_net_savings": r.mean_net_savings,
                }
                for r in budget_results
            ],
        }
        cmp_path = RESULTS_DIR / f"{timestamp}_B{budget}_comparison.json"
        with open(cmp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"  Saved comparison to {cmp_path}")

    # Cross-budget summary
    print_ablation_table(all_data)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "02_budget_ablation",
        "n_episodes": args.n_episodes,
        "claims_per_episode": args.claims,
        "fraud_rate": args.fraud_rate,
        "budgets_tested": args.budgets,
        "by_budget": {
            str(b): [
                {
                    "name": r.agent_name,
                    "mean_reward": r.mean_reward,
                    "mean_f1": r.mean_f1,
                    "mean_budget_conservation_rate": r.mean_budget_conservation_rate,
                    "mean_budget_utilization": r.mean_budget_utilization,
                }
                for r in results
            ]
            for b, results in all_data.items()
        },
    }
    summary_path = RESULTS_DIR / f"{timestamp}_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAblation summary saved to {summary_path}")


if __name__ == "__main__":
    main()
