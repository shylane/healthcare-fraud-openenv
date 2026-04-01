"""Compute basic fraud detection metrics from a harvested JSONL file.

Called by orchestrate_cycles_v2.py after each eval harvest to give a
quick read on how the policy is performing before the next training cycle.

Output: JSON summary printed to stdout + written to <data>.metrics.json

Usage:
    python training/eval_metrics.py --data data/eval_after_cycle1.jsonl
"""

import argparse
import json
from pathlib import Path


def compute_metrics(jsonl_path: str) -> dict:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"error": "empty file", "num_records": 0}

    total = len(records)
    decision_counts: dict[str, int] = {}
    tp = fp = tn = fn = 0

    for rec in records:
        gt = rec.get("ground_truth", {})
        is_fraud = gt.get("is_fraud", False)
        action = rec.get("action", "UNKNOWN").upper()

        decision_counts[action] = decision_counts.get(action, 0) + 1

        caught = action in ("INVESTIGATE", "FLAG_REVIEW", "DENY")
        if is_fraud and caught:
            tp += 1
        elif is_fraud and not caught:
            fn += 1
        elif not is_fraud and not caught:
            tn += 1
        else:
            fp += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    approve_rate = decision_counts.get("APPROVE", 0) / total

    metrics = {
        "num_records": total,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "approve_rate": round(approve_rate, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "decision_distribution": decision_counts,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Eval metrics from harvested JSONL")
    parser.add_argument("--data", required=True, help="Path to harvested JSONL file")
    args = parser.parse_args()

    metrics = compute_metrics(args.data)

    print("\n=== Eval Metrics ===")
    print(f"  Records   : {metrics.get('num_records', 0)}")
    print(f"  Precision : {metrics.get('precision', 0):.3f}")
    print(f"  Recall    : {metrics.get('recall', 0):.3f}")
    print(f"  F1        : {metrics.get('f1', 0):.3f}")
    print(f"  Approve % : {metrics.get('approve_rate', 0):.1%}")
    print(f"  Decisions : {metrics.get('decision_distribution', {})}")
    print("====================\n")

    # Write side-car metrics file
    out_path = Path(args.data).with_suffix(".metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to: {out_path}")


if __name__ == "__main__":
    main()
