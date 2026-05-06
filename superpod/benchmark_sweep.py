"""Benchmark completed SuperPOD sweep checkpoints.

The benchmark chooses the best agent by validation performance, not by the
largest episode count. That matters in this stochastic game because longer
self-play can drift toward conservative or unstable policies.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import (  # noqa: E402
    BasicHeuristicAgent,
    HeuristicAgent,
    LineBuilderAgent,
    QTableAgent,
    RandomAgent,
    TorchDQNAgent,
    TorchPPOAgent,
    evaluate_matchup,
)
from utils import set_global_seeds  # noqa: E402


def read_tsv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [row for row in reader if row and not row.get("name", "").startswith("#")]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def trained_agent(row: Dict[str, str], run_dir: Path, device: str):
    algo = row["algo"].strip().lower()
    if algo == "ppo":
        path = run_dir / "super_ttt_agent_torchrl.pt"
        if not path.exists():
            path = run_dir / "super_ttt_agent_torch.pt"
        if not path.exists():
            return None, path
        agent = TorchPPOAgent(str(path), device=device, deterministic=True)
    elif algo == "dqn":
        path = run_dir / "dqn_agent_torch.pt"
        if not path.exists():
            return None, path
        agent = TorchDQNAgent(str(path), device=device)
    else:
        return None, run_dir
    agent.name = row["name"]
    return agent, path


def q_agent(row: Dict[str, str], run_dir: Path):
    path = run_dir / "q_table.pkl"
    if not path.exists():
        return None, path
    agent = QTableAgent(str(path), seed=int(row.get("seed", 0)))
    agent.name = row["name"]
    return agent, path


def add_matchup(
    agent,
    baseline,
    games: int,
    seed: int,
    context: Dict[str, object],
    raw_rows: List[Dict[str, object]],
    summary_rows: List[Dict[str, object]],
) -> None:
    rows, summary = evaluate_matchup(agent, baseline, games=games, seed=seed)
    for row in rows:
        row.update(context)
        row["baseline"] = getattr(baseline, "name", "baseline")
    summary.update(context)
    summary["baseline"] = getattr(baseline, "name", "baseline")
    raw_rows.extend(rows)
    summary_rows.append(summary)
    print(
        f"{summary['agent_a']} vs {summary['agent_b']}: "
        f"{summary['agent_a_wins']}-{summary['agent_b_wins']}-"
        f"{summary['draws']} over {summary['games']} games"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark completed SuperPOD sweep checkpoints.")
    parser.add_argument("--run-root", type=str, required=True)
    parser.add_argument("--sweep-name", type=str, required=True)
    parser.add_argument("--neural-config", type=str, default="superpod/experiments_neural.tsv")
    parser.add_argument("--q-config", type=str, default="superpod/experiments_q.tsv")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    sweep_dir = Path(args.run_root) / args.sweep_name
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "benchmarks"

    baselines = [
        RandomAgent(seed=args.seed + 1),
        BasicHeuristicAgent(seed=args.seed + 2),
        LineBuilderAgent(seed=args.seed + 3),
        HeuristicAgent(seed=args.seed + 4),
    ]
    raw_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []

    # Baseline sanity checks; these become the comparison floor in the report.
    add_matchup(
        HeuristicAgent(seed=args.seed + 10),
        RandomAgent(seed=args.seed + 11),
        games=args.games,
        seed=args.seed + 10_000,
        context={"run_name": "baseline_heuristic_vs_random", "algo": "baseline"},
        raw_rows=raw_rows,
        summary_rows=summary_rows,
    )

    seed_offset = 100_000
    neural_rows = read_tsv(ROOT / args.neural_config)
    for row_index, row in enumerate(neural_rows):
        run_dir = sweep_dir / row["name"]
        agent, checkpoint = trained_agent(row, run_dir, args.device)
        if agent is None:
            missing_rows.append({"run_name": row["name"], "checkpoint": str(checkpoint)})
            continue
        context = {
            "run_name": row["name"],
            "algo": row["algo"],
            "train_seed": row["seed"],
            "train_episodes": row["episodes"],
            "train_opponent": row["opponent"],
            "checkpoint": str(checkpoint),
        }
        for baseline_index, baseline in enumerate(baselines):
            add_matchup(
                agent,
                baseline,
                games=args.games,
                seed=args.seed + seed_offset + row_index * 1000 + baseline_index * 100,
                context=context,
                raw_rows=raw_rows,
                summary_rows=summary_rows,
            )

    q_rows = read_tsv(ROOT / args.q_config)
    for row_index, row in enumerate(q_rows):
        run_dir = sweep_dir / row["name"]
        agent, checkpoint = q_agent(row, run_dir)
        if agent is None:
            missing_rows.append({"run_name": row["name"], "checkpoint": str(checkpoint)})
            continue
        context = {
            "run_name": row["name"],
            "algo": "q_learning",
            "train_seed": row["seed"],
            "train_episodes": row["episodes"],
            "train_opponent": "self",
            "checkpoint": str(checkpoint),
        }
        for baseline_index, baseline in enumerate(baselines):
            add_matchup(
                agent,
                baseline,
                games=args.games,
                seed=args.seed + 500_000 + row_index * 1000 + baseline_index * 100,
                context=context,
                raw_rows=raw_rows,
                summary_rows=summary_rows,
            )

    write_csv(output_dir / "benchmark_raw.csv", raw_rows)
    write_csv(output_dir / "benchmark_summary.csv", summary_rows)
    write_csv(output_dir / "missing_checkpoints.csv", missing_rows)
    print(f"Wrote benchmark files to {output_dir}")
    if missing_rows:
        print(f"Missing checkpoints: {len(missing_rows)}")


if __name__ == "__main__":
    main()
