"""
Experiment 01 — Baseline Agent Comparison

Runs all 4 agents (random, threshold, naive LLM, budget-aware LLM) through
the same set of episodes and saves results for analysis.

Usage:
    # Set your OpenRouter API key first:
    export OPENROUTER_API_KEY=sk-or-...

    # Run with default settings (10 episodes, free model):
    python experiments/01_baseline_comparison/run.py

    # Run with DeepSeek V3 as gold standard (adds ~$1 cost):
    python experiments/01_baseline_comparison/run.py --include-deepseek

    # Quick smoke test (2 episodes, 20 claims each):
    python experiments/01_baseline_comparison/run.py --quick

    # Full run (20 episodes — more statistical power):
    python experiments/01_baseline_comparison/run.py --n-episodes 20
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add repo root to path
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from evaluation.harness import run_agent
from evaluation.agents import (
    RandomAgent,
    ThresholdAgent,
    NaiveLLMAgent,
    BudgetAwareAgent,
    DeepSeekNaiveAgent,
    DeepSeekBudgetAwareAgent,
)

RESULTS_DIR = Path(__file__).parent / "results"


def parse_args():
    p = argparse.ArgumentParser(description="Baseline agent comparison experiment")
    p.add_argument("--n-episodes", type=int, default=10,
                   help="Episodes per agent (default: 10)")
    p.add_argument("--claims", type=int, default=100,
                   help="Claims per episode (default: 100)")
    p.add_argument("--fraud-rate", type=float, default=0.05,
                   help="Fraud rate in environment (default: 0.05)")
    p.add_argument("--budget", type=int, default=15,
                   help="Investigation budget per episode (default: 15)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base random seed (default: 42)")
    p.add_argument("--model", type=str, default="openai/gpt-oss-120b:free",
                   help="OpenRouter model ID for LLM agents")
    p.add_argument("--include-deepseek", action="store_true",
                   help="Also run DeepSeek V3 agents (~$1 extra cost)")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: 2 episodes, 20 claims (smoke test)")
    p.add_argument("--no-llm", action="store_true",
                   help="Only run rule-based agents (no API calls)")
    p.add_argument("--api-key", type=str, default=None,
                   help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    return p.parse_args()


def print_comparison_table(all_results: list) -> None:
    """Print a side-by-side comparison table of all agents."""
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    fmt = "{:<30} {:>8} {:>8} {:>8} {:>10} {:>12} {:>12}"
    print(fmt.format(
        "Agent", "Reward", "F1", "Recall",
        "NetSave$", "BudgCons%", "MemReuse%"
    ))
    print("-"*80)
    for r in all_results:
        print(fmt.format(
            r.agent_name[:30],
            f"{r.mean_reward:+.1f}",
            f"{r.mean_f1:.3f}",
            f"{r.mean_recall:.3f}",
            f"${r.mean_net_savings:,.0f}",
            f"{r.mean_budget_conservation_rate:.0%}",
            f"{r.mean_memory_reuse_rate:.0%}",
        ))
    print("="*80)

    # Key insight callout
    if len(all_results) >= 4:
        naive = next((r for r in all_results if "Naive" in r.agent_name), None)
        budget = next((r for r in all_results if "Budget" in r.agent_name), None)
        if naive and budget:
            gap_f1 = budget.mean_f1 - naive.mean_f1
            gap_savings = budget.mean_net_savings - naive.mean_net_savings
            gap_budget_cons = (
                budget.mean_budget_conservation_rate - naive.mean_budget_conservation_rate
            )
            print(f"\nKEY FINDING — Budget awareness gap (same model, different prompt):")
            print(f"  F1 improvement:           {gap_f1:+.3f}")
            print(f"  Net savings/episode:      ${gap_savings:+,.0f}")
            print(f"  Budget conservation rate: {gap_budget_cons:+.0%}")
            if gap_f1 > 0.01:
                print("  → Budget-aware prompt SIGNIFICANTLY outperforms naive prompt")
            elif gap_f1 > 0:
                print("  → Budget-aware prompt marginally outperforms naive prompt")
            else:
                print("  → Minimal difference — model may not follow budget instructions")


def save_comparison(all_results: list, path: Path) -> None:
    """Save side-by-side comparison to JSON."""
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "agents": [
            {
                "name": r.agent_name,
                "model": r.model,
                "n_episodes": r.n_episodes,
                "mean_reward": r.mean_reward,
                "std_reward": r.std_reward,
                "mean_f1": r.mean_f1,
                "mean_precision": r.mean_precision,
                "mean_recall": r.mean_recall,
                "mean_net_savings": r.mean_net_savings,
                "mean_budget_utilization": r.mean_budget_utilization,
                "mean_budget_conservation_rate": r.mean_budget_conservation_rate,
                "mean_memory_reuse_rate": r.mean_memory_reuse_rate,
                "mean_valid_response_rate": r.mean_valid_response_rate,
                "mean_latency_ms": r.mean_latency_ms,
            }
            for r in all_results
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to {path}")


def main():
    args = parse_args()

    if args.quick:
        args.n_episodes = 2
        args.claims = 20
        print("Quick mode: 2 episodes × 20 claims")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.no_llm:
        print("ERROR: Set OPENROUTER_API_KEY env var or pass --api-key")
        print("  export OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    # Build agent list
    agents = [RandomAgent(), ThresholdAgent()]
    if not args.no_llm:
        agents += [
            NaiveLLMAgent(api_key=api_key, model=args.model),
            BudgetAwareAgent(api_key=api_key, model=args.model),
        ]
    if args.include_deepseek and not args.no_llm:
        agents += [
            DeepSeekNaiveAgent(api_key=api_key),
            DeepSeekBudgetAwareAgent(api_key=api_key),
        ]

    print(f"\nExperiment 01 — Baseline Comparison")
    print(f"Agents: {[a.name for a in agents]}")
    print(f"Config: {args.n_episodes} episodes × {args.claims} claims "
          f"| fraud_rate={args.fraud_rate:.0%} | budget={args.budget}")
    print(f"Model: {args.model}\n")

    all_results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for agent in agents:
        print(f"\n{'-'*60}")
        print(f"Running: {agent.name}")
        print(f"{'-'*60}")

        results = run_agent(
            agent=agent,
            n_episodes=args.n_episodes,
            claims_per_episode=args.claims,
            fraud_rate=args.fraud_rate,
            investigation_budget=args.budget,
            seed=args.seed,
            verbose=True,
        )
        all_results.append(results)

        # Save individual results
        safe_name = agent.name.replace("/", "_").replace(":", "_").replace("(", "").replace(")", "")
        results.save(RESULTS_DIR / f"{timestamp}_{safe_name}.json")

    # Print and save comparison
    print_comparison_table(all_results)
    save_comparison(all_results, RESULTS_DIR / f"{timestamp}_comparison.json")

    # Summary message
    print(f"\nTotal API calls made: {sum(getattr(a, '_call_count', 0) for a in agents)}")
    print(f"Results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
