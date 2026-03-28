#!/usr/bin/env python3
"""
GSPO Cycle Orchestrator v2 — Interruption-safe, auto-resuming, phase-tracked.

Improvements over v1:
  1. Phase lock files — each sub-step (eval/harvest/train) writes a .done marker
     so a re-run skips already-completed phases instead of repeating them.
  2. Auto-resume from any cycle — scans checkpoints dir to detect the last
     completed cycle and starts from there.  No more hardcoded range(3, ...).
  3. Configurable via CLI — WORKSPACE, model, episodes are all arguments.
  4. Graceful SIGTERM handling — writes a resume marker before exiting so the
     next invocation knows exactly where to pick up.
  5. Dry-run mode — prints the full execution plan without running anything.
  6. Per-cycle timeout — kills a hung sub-process and marks the phase as failed
     rather than hanging the whole pipeline.
  7. Cycle 1 management — can optionally launch Cycle 1 (heuristic harvest +
     train) without waiting for a separate process.

Usage:
  # Fresh full run (all 5 cycles, starting Cycle 1 from scratch):
  python scripts/orchestrate_cycles_v2.py --start-fresh

  # Resume from wherever last interrupted:
  python scripts/orchestrate_cycles_v2.py

  # Dry run (print plan only):
  python scripts/orchestrate_cycles_v2.py --dry-run

  # Override workspace (default: /workspace):
  python scripts/orchestrate_cycles_v2.py --workspace /home/user/training

  # Use different model:
  python scripts/orchestrate_cycles_v2.py --model-id unsloth/Qwen3-1.7B-Instruct
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Defaults  (override via CLI args)
# ---------------------------------------------------------------------------
DEFAULT_WORKSPACE = "/workspace"
DEFAULT_MODEL_ID = "unsloth/Qwen2.5-1.5B-Instruct"
DEFAULT_TOTAL_CYCLES = 5
DEFAULT_TRAIN_EPISODES = 15   # 15 eps × 100 claims = 1500 records
DEFAULT_EVAL_EPISODES = 10
DEFAULT_EVAL_SEED_START = 1000
DEFAULT_CLAIMS_PER_EPISODE = 100
PYTHON_CMD = ".venv/bin/python"

# Phase names (used for .done marker filenames)
PHASE_EVAL_HARVEST = "eval_harvest"
PHASE_EVAL_METRICS = "eval_metrics"
PHASE_TRAIN_HARVEST = "train_harvest"
PHASE_TRAIN = "train"
ALL_PHASES = [PHASE_EVAL_HARVEST, PHASE_EVAL_METRICS, PHASE_TRAIN_HARVEST, PHASE_TRAIN]


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_CURRENT_PROC: "subprocess.Popen | None" = None
_SHUTDOWN_REQUESTED = False


def _signal_handler(signum, frame):
    """On SIGTERM: kill current subprocess and write resume marker, then exit."""
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = True
    sig = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
    print(f"\n[ORCHESTRATOR] {sig} received — stopping current sub-process...", flush=True)
    if _CURRENT_PROC is not None:
        try:
            _CURRENT_PROC.terminate()
            time.sleep(3)
            if _CURRENT_PROC.poll() is None:
                _CURRENT_PROC.kill()
        except Exception:
            pass
    print("[ORCHESTRATOR] Exiting cleanly.  Re-run to resume from last completed phase.",
          flush=True)
    sys.exit(0)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)


# ---------------------------------------------------------------------------
# Phase marker helpers
# ---------------------------------------------------------------------------

def _marker_path(workspace: Path, cycle: int, phase: str) -> Path:
    return workspace / "phase_markers" / f"cycle{cycle}_{phase}.done"


def _phase_done(workspace: Path, cycle: int, phase: str) -> bool:
    return _marker_path(workspace, cycle, phase).exists()


def _mark_phase_done(workspace: Path, cycle: int, phase: str) -> None:
    path = _marker_path(workspace, cycle, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(datetime.now().isoformat())


def _clear_phase_markers(workspace: Path, cycle: int) -> None:
    """Clear all phase markers for a cycle (use when re-running a cycle)."""
    for phase in ALL_PHASES:
        p = _marker_path(workspace, cycle, phase)
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# Sub-process runner
# ---------------------------------------------------------------------------

def run_command(
    cmd_list: list[str],
    log_file: Path,
    description: str,
    workspace: Path,
    timeout_hours: float = 4.0,
    dry_run: bool = False,
) -> int:
    """Run a command, tee output to log_file, return exit code.

    Timeout in hours prevents a hung sub-process from blocking forever.
    On timeout the sub-process is killed and the caller can decide whether
    to retry or abort.
    """
    global _CURRENT_PROC

    if dry_run:
        print(f"  [DRY-RUN] Would run: {' '.join(cmd_list)}")
        print(f"            Log: {log_file}")
        return 0

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] START: {description}")
    print(f"  CMD : {' '.join(cmd_list)}")
    print(f"  LOG : {log_file}")

    log_file.parent.mkdir(parents=True, exist_ok=True)
    timeout_sec = int(timeout_hours * 3600)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"=== {description} ===\n")
        f.write(f"Started: {datetime.now()}\n")
        f.write(f"Command: {' '.join(cmd_list)}\n\n")

    with open(log_file, "a", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(workspace),
        )
        _CURRENT_PROC = proc

        start = time.time()
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                f.write(line)
                f.flush()
                if time.time() - start > timeout_sec:
                    print(f"\n[TIMEOUT] {description} exceeded {timeout_hours}h — killing.")
                    proc.kill()
                    f.write(f"\n[TIMEOUT] Killed after {timeout_hours}h\n")
                    return -1
                if _SHUTDOWN_REQUESTED:
                    proc.terminate()
                    f.write("\n[INTERRUPTED] Orchestrator shutdown requested.\n")
                    return -2
        finally:
            proc.wait()
            _CURRENT_PROC = None
            elapsed = time.time() - start
            f.write(f"\nFinished: {datetime.now()}\n")
            f.write(f"Elapsed: {elapsed:.0f}s\n")
            f.write(f"Exit code: {proc.returncode}\n")

    rc = proc.returncode
    status = "OK" if rc == 0 else f"FAILED (exit {rc})"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] END  : {description}  [{status}]")
    return rc


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _detect_last_completed_cycle(workspace: Path, total_cycles: int) -> int:
    """Return the last cycle whose TRAIN phase marker exists.

    Returns 0 if no cycle has completed training (i.e. need to start from 1).
    """
    last = 0
    for c in range(1, total_cycles + 1):
        if _phase_done(workspace, c, PHASE_TRAIN):
            last = c
    return last


def _cycle1_checkpoint_exists(workspace: Path) -> bool:
    """Check if Cycle 1 LoRA checkpoint exists."""
    ckpt = workspace / "checkpoints_v2" / "cycle_1"
    return ckpt.exists() and any((ckpt).iterdir())


# ---------------------------------------------------------------------------
# Per-phase runners
# ---------------------------------------------------------------------------

def run_eval_harvest(
    workspace: Path, cycle: int, model_path: str,
    eval_episodes: int, eval_seed_start: int, claims: int,
    dry_run: bool, model_id: str,
) -> bool:
    """Harvest evaluation episodes using the given model checkpoint."""
    phase = PHASE_EVAL_HARVEST
    if _phase_done(workspace, cycle, phase):
        print(f"  [SKIP] Cycle {cycle} {phase} already done.")
        return True

    output_data = workspace / "data" / f"eval_after_cycle{cycle-1}.jsonl"
    log = workspace / "logs" / f"eval_harvest_cycle{cycle-1}.log"

    cmd = [
        PYTHON_CMD, "training/harvest_episodes.py",
        "--episodes", str(eval_episodes),
        "--seed-start", str(eval_seed_start),
        "--model", model_path,
        "--model-id", model_id,
        "--output", str(output_data),
        "--claims", str(claims),
    ]
    rc = run_command(cmd, log, f"Eval harvest (cycle {cycle-1} model)", workspace,
                     timeout_hours=2.0, dry_run=dry_run)
    if rc == 0:
        _mark_phase_done(workspace, cycle, phase)
        return True
    return False


def run_eval_metrics(workspace: Path, cycle: int, dry_run: bool) -> bool:
    phase = PHASE_EVAL_METRICS
    if _phase_done(workspace, cycle, phase):
        print(f"  [SKIP] Cycle {cycle} {phase} already done.")
        return True

    data = workspace / "data" / f"eval_after_cycle{cycle-1}.jsonl"
    log = workspace / "logs" / f"eval_metrics_cycle{cycle-1}.log"

    cmd = [PYTHON_CMD, "training/eval_metrics.py", "--data", str(data)]
    rc = run_command(cmd, log, f"Eval metrics (cycle {cycle-1})", workspace,
                     timeout_hours=0.5, dry_run=dry_run)
    if rc == 0:
        _mark_phase_done(workspace, cycle, phase)
        return True
    return False


def run_train_harvest(
    workspace: Path, cycle: int, model_path: str,
    train_episodes: int, claims: int,
    dry_run: bool, model_id: str = DEFAULT_MODEL_ID,
) -> bool:
    phase = PHASE_TRAIN_HARVEST
    if _phase_done(workspace, cycle, phase):
        print(f"  [SKIP] Cycle {cycle} {phase} already done.")
        return True

    seed_start = cycle * train_episodes
    output_data = workspace / "data" / f"train_cycle{cycle}.jsonl"
    log = workspace / "logs" / f"harvest_train_cycle{cycle}.log"

    cmd = [
        PYTHON_CMD, "training/harvest_episodes.py",
        "--episodes", str(train_episodes),
        "--seed-start", str(seed_start),
        "--model", model_path,
        "--model-id", model_id,
        "--output", str(output_data),
        "--claims", str(claims),
    ]
    rc = run_command(cmd, log, f"Training harvest (cycle {cycle})", workspace,
                     timeout_hours=2.0, dry_run=dry_run)
    if rc == 0:
        _mark_phase_done(workspace, cycle, phase)
        return True
    return False


def run_train(
    workspace: Path, cycle: int, base_cycle_path: str | None,
    model_id: str, dry_run: bool, group_size: int,
    max_completion_length: int, wandb: bool,
) -> bool:
    phase = PHASE_TRAIN
    if _phase_done(workspace, cycle, phase):
        print(f"  [SKIP] Cycle {cycle} {phase} already done.")
        return True

    data = workspace / "data" / f"train_cycle{cycle}.jsonl"
    output_dir = workspace / "checkpoints_v2"
    log = workspace / "logs" / f"train_cycle{cycle}.log"

    cmd = [
        PYTHON_CMD, "training/train_gspo_v2.py",
        "--cycle", str(cycle),
        "--data", str(data),
        "--output-dir", str(output_dir),
        "--model-id", model_id,
        "--group-size", str(group_size),
        "--max-completion-length", str(max_completion_length),
        "--auto-resume",        # always safe: resumes if interrupted, noop if fresh
    ]
    if base_cycle_path:
        cmd += ["--base-cycle", base_cycle_path]
    if wandb:
        cmd.append("--wandb")

    rc = run_command(cmd, log, f"GSPO Train (cycle {cycle})", workspace,
                     timeout_hours=6.0, dry_run=dry_run)
    if rc == 0:
        _mark_phase_done(workspace, cycle, phase)
        return True
    return False


def run_cycle1_heuristic_harvest(
    workspace: Path, episodes: int, claims: int, dry_run: bool,
) -> bool:
    """Cycle 1 uses heuristic policy (no model needed)."""
    phase = PHASE_TRAIN_HARVEST
    if _phase_done(workspace, 1, phase):
        print("  [SKIP] Cycle 1 heuristic harvest already done.")
        return True

    output_data = workspace / "data" / "train_cycle1.jsonl"
    log = workspace / "logs" / "harvest_train_cycle1.log"

    cmd = [
        PYTHON_CMD, "training/harvest_episodes.py",
        "--episodes", str(episodes),
        "--seed-start", "0",
        "--output", str(output_data),
        "--claims", str(claims),
        # No --model → uses heuristic policy
    ]
    rc = run_command(cmd, log, "Cycle 1 heuristic harvest", workspace,
                     timeout_hours=1.0, dry_run=dry_run)
    if rc == 0:
        _mark_phase_done(workspace, 1, phase)
        return True
    return False


def run_cycle1_train(
    workspace: Path, model_id: str, dry_run: bool,
    group_size: int, max_completion_length: int, wandb: bool,
) -> bool:
    phase = PHASE_TRAIN
    if _phase_done(workspace, 1, phase):
        print("  [SKIP] Cycle 1 training already done.")
        return True

    data = workspace / "data" / "train_cycle1.jsonl"
    output_dir = workspace / "checkpoints_v2"
    log = workspace / "logs" / "train_cycle1.log"

    cmd = [
        PYTHON_CMD, "training/train_gspo_v2.py",
        "--cycle", "1",
        "--data", str(data),
        "--output-dir", str(output_dir),
        "--model-id", model_id,
        "--group-size", str(group_size),
        "--max-completion-length", str(max_completion_length),
        "--auto-resume",
    ]
    if wandb:
        cmd.append("--wandb")

    rc = run_command(cmd, log, "GSPO Train (cycle 1)", workspace,
                     timeout_hours=6.0, dry_run=dry_run)
    if rc == 0:
        _mark_phase_done(workspace, 1, phase)
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GSPO Orchestrator v2")
    parser.add_argument("--workspace", type=str, default=DEFAULT_WORKSPACE)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--total-cycles", type=int, default=DEFAULT_TOTAL_CYCLES)
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_TRAIN_EPISODES)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--claims", type=int, default=DEFAULT_CLAIMS_PER_EPISODE)
    parser.add_argument("--group-size", type=int, default=12,
                        help="num_generations for 3090 (default 12).")
    parser.add_argument("--max-completion-length", type=int, default=384)
    parser.add_argument("--start-fresh", action="store_true",
                        help="Clear all phase markers and restart from Cycle 1.")
    parser.add_argument("--force-cycle", type=int, default=0,
                        help="Force-start from a specific cycle (ignores phase markers "
                             "for that cycle onward).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing.")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "logs").mkdir(exist_ok=True)
    (workspace / "data").mkdir(exist_ok=True)
    (workspace / "checkpoints_v2").mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  GSPO Orchestrator v2")
    print(f"  Workspace  : {workspace}")
    print(f"  Model      : {args.model_id}")
    print(f"  Cycles     : 1 → {args.total_cycles}")
    print(f"  Train eps  : {args.train_episodes}  |  Eval eps: {args.eval_episodes}")
    print(f"  Group size : {args.group_size}  |  Max compl: {args.max_completion_length}")
    print(f"  Dry run    : {args.dry_run}")
    print(f"{'='*60}\n")

    if args.start_fresh:
        print("[FRESH START] Clearing all phase markers...")
        markers_dir = workspace / "phase_markers"
        if markers_dir.exists():
            shutil.rmtree(markers_dir)
        print("  Done.\n")

    if args.force_cycle > 0:
        print(f"[FORCE] Clearing markers for cycles {args.force_cycle}+")
        for c in range(args.force_cycle, args.total_cycles + 1):
            _clear_phase_markers(workspace, c)

    # Detect where to resume
    last_done = _detect_last_completed_cycle(workspace, args.total_cycles)
    start_cycle = max(1, last_done + 1) if not args.start_fresh else 1
    if args.force_cycle > 0:
        start_cycle = args.force_cycle

    print(f"[RESUME] Last completed cycle: {last_done if last_done > 0 else 'None'}")
    print(f"[RESUME] Starting from cycle : {start_cycle}\n")

    if start_cycle > args.total_cycles:
        print(f"All {args.total_cycles} cycles already complete!")
        return

    # ── CYCLE 1 ─────────────────────────────────────────────────────────
    if start_cycle == 1:
        print("\n" + "="*50)
        print("=== CYCLE 1 ===")
        print("="*50)

        # Heuristic harvest
        ok = run_cycle1_heuristic_harvest(
            workspace, args.train_episodes, args.claims, args.dry_run)
        if not ok and not args.dry_run:
            print("!! Cycle 1 harvest failed — aborting.")
            sys.exit(1)
        if _SHUTDOWN_REQUESTED:
            sys.exit(0)

        # Train
        ok = run_cycle1_train(
            workspace, args.model_id, args.dry_run,
            args.group_size, args.max_completion_length, args.wandb)
        if not ok and not args.dry_run:
            print("!! Cycle 1 training failed — aborting.")
            sys.exit(1)
        if _SHUTDOWN_REQUESTED:
            sys.exit(0)

        start_cycle = 2  # fall through to cycles 2+

    # ── CYCLES 2–N ──────────────────────────────────────────────────────
    for cycle in range(max(start_cycle, 2), args.total_cycles + 1):
        prev_cycle = cycle - 1
        prev_model_path = str(workspace / "checkpoints_v2" / f"cycle_{prev_cycle}")

        print("\n" + "="*50)
        print(f"=== CYCLE {cycle} ===")
        print("="*50)
        print(f"  Prev model: {prev_model_path}")

        # Phase 1: Eval harvest (using prev cycle model)
        ok = run_eval_harvest(
            workspace, cycle, prev_model_path,
            args.eval_episodes, DEFAULT_EVAL_SEED_START + (cycle * 100),
            args.claims, args.dry_run, args.model_id)
        if not ok and not args.dry_run:
            print(f"!! Cycle {cycle} eval harvest failed — aborting.")
            sys.exit(1)
        if _SHUTDOWN_REQUESTED:
            sys.exit(0)

        # Phase 2: Eval metrics
        ok = run_eval_metrics(workspace, cycle, args.dry_run)
        if not ok and not args.dry_run:
            print(f"!! Cycle {cycle} eval metrics failed — skipping (non-fatal).")
            # Non-fatal — metrics failure shouldn't stop training

        if _SHUTDOWN_REQUESTED:
            sys.exit(0)

        # Phase 3: Training harvest
        ok = run_train_harvest(
            workspace, cycle, prev_model_path,
            args.train_episodes, args.claims, args.dry_run, model_id=args.model_id)
        if not ok and not args.dry_run:
            print(f"!! Cycle {cycle} train harvest failed — aborting.")
            sys.exit(1)
        if _SHUTDOWN_REQUESTED:
            sys.exit(0)

        # Phase 4: Train
        ok = run_train(
            workspace, cycle,
            base_cycle_path=prev_model_path,  # merge prev LoRA before fresh LoRA
            model_id=args.model_id,
            dry_run=args.dry_run,
            group_size=args.group_size,
            max_completion_length=args.max_completion_length,
            wandb=args.wandb,
        )
        if not ok and not args.dry_run:
            print(f"!! Cycle {cycle} training failed — aborting.")
            sys.exit(1)
        if _SHUTDOWN_REQUESTED:
            sys.exit(0)

        print(f"\n  ✓ Cycle {cycle} complete.\n")

    print("\n" + "="*50)
    print(f"ALL {args.total_cycles} CYCLES COMPLETE!")
    print("="*50)


if __name__ == "__main__":
    main()
