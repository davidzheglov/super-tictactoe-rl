"""Benchmark Super Tic-Tac-Toe agents in pairwise matchups."""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path
from typing import Dict, List

try:
    from .agents import (
        BasicHeuristicAgent,
        HeuristicAgent,
        LineBuilderAgent,
        QTableAgent,
        RandomAgent,
        TorchDQNAgent,
        TorchPPOAgent,
        evaluate_matchup,
    )
    from .utils import project_root, set_global_seeds
except ImportError:  # pragma: no cover
    from agents import (
        BasicHeuristicAgent,
        HeuristicAgent,
        LineBuilderAgent,
        QTableAgent,
        RandomAgent,
        TorchDQNAgent,
        TorchPPOAgent,
        evaluate_matchup,
    )
    from utils import project_root, set_global_seeds


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_agent(name: str, args: argparse.Namespace, seed: int):
    normalized = name.strip().lower()
    if normalized == "random":
        return RandomAgent(seed=seed)
    if normalized == "heuristic":
        return HeuristicAgent(seed=seed)
    if normalized in {"basic", "basic_heuristic"}:
        return BasicHeuristicAgent(seed=seed)
    if normalized in {"line", "line_builder"}:
        return LineBuilderAgent(seed=seed)
    if normalized == "ppo":
        return TorchPPOAgent(args.ppo_path, device=args.device, deterministic=args.deterministic)
    if normalized == "dqn":
        return TorchDQNAgent(args.dqn_path, device=args.device)
    if normalized == "q":
        return QTableAgent(args.q_path, seed=seed)
    raise ValueError(f"Unknown agent {name!r}.")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Run pairwise Super Tic-Tac-Toe benchmarks.")
    parser.add_argument(
        "--agents",
        type=str,
        default="random,basic,line,heuristic",
        help="Comma-separated list from random,basic,line,heuristic,ppo,dqn,q.",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--ppo-path",
        type=str,
        default=str(root / "runs" / "overnight_torch" / "ppo_seed0" / "super_ttt_agent_torchrl.pt"),
    )
    parser.add_argument(
        "--dqn-path",
        type=str,
        default=str(root / "runs" / "overnight_torch" / "dqn_seed0" / "dqn_agent_torch.pt"),
    )
    parser.add_argument(
        "--q-path",
        type=str,
        default=str(root / "runs" / "overnight_torch" / "q_learning_seed0" / "q_table.pkl"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(root / "runs" / "benchmarks"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    names = [name.strip() for name in args.agents.split(",") if name.strip()]
    agents = [build_agent(name, args, args.seed + i * 1000) for i, name in enumerate(names)]
    raw_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for pair_index, (agent_a, agent_b) in enumerate(itertools.combinations(agents, 2)):
        rows, summary = evaluate_matchup(
            agent_a,
            agent_b,
            games=args.games,
            seed=args.seed + pair_index * 100_000,
        )
        raw_rows.extend(rows)
        summary_rows.append(summary)
        print(
            f"{summary['agent_a']} vs {summary['agent_b']}: "
            f"{summary['agent_a_wins']}-{summary['agent_b_wins']}-"
            f"{summary['draws']} over {summary['games']} games"
        )

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "benchmark_raw.csv", raw_rows)
    write_csv(output_dir / "benchmark_summary.csv", summary_rows)
    print(f"Wrote benchmark files to {output_dir}")


if __name__ == "__main__":
    main()
