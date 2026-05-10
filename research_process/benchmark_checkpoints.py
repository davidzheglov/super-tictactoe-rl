"""Benchmark saved neural checkpoints against scripted baselines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

import torch

try:
    from .agents import HeuristicAgent, LineBuilderAgent, TorchDQNAgent, TorchPPOAgent, evaluate_matchup
    from .utils import project_root, set_global_seeds
except ImportError:  # pragma: no cover
    from agents import HeuristicAgent, LineBuilderAgent, TorchDQNAgent, TorchPPOAgent, evaluate_matchup
    from utils import project_root, set_global_seeds


def checkpoint_paths(paths: Iterable[str]) -> List[Path]:
    found: List[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            found.extend(sorted(path.glob("*.pt")))
        elif path.exists():
            found.append(path)
    return found


def build_agent(path: Path, device: str, deterministic: bool):
    payload = torch.load(path, map_location="cpu")
    algo = str(payload.get("algo", ""))
    if algo in {"torch_ppo", "torchrl_ppo"}:
        return TorchPPOAgent(str(path), device=device, deterministic=deterministic), algo
    if algo == "torch_dqn":
        return TorchDQNAgent(str(path), device=device), algo
    raise ValueError(f"Unsupported checkpoint algo {algo!r}: {path}")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Benchmark checkpoint files.")
    parser.add_argument("paths", nargs="+", help="Checkpoint files or directories containing .pt files.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--opponents",
        type=str,
        default="heuristic,line",
        help="Comma-separated scripted opponents: heuristic,line.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(root / "runs" / "checkpoint_benchmark.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    paths = checkpoint_paths(args.paths)
    opponents = [item.strip() for item in args.opponents.split(",") if item.strip()]
    rows: List[Dict[str, object]] = []

    for ckpt_index, path in enumerate(paths):
        try:
            agent, algo = build_agent(path, args.device, args.deterministic)
        except Exception as exc:
            rows.append(
                {
                    "checkpoint": str(path),
                    "algo": "unsupported",
                    "opponent": "",
                    "episodes": "",
                    "agent_wins": 0,
                    "opponent_wins": 0,
                    "draws": 0,
                    "agent_win_rate": 0.0,
                    "error": str(exc),
                }
            )
            continue
        payload = torch.load(path, map_location="cpu")
        for opponent_index, opponent_name in enumerate(opponents):
            if opponent_name == "heuristic":
                opponent = HeuristicAgent(seed=args.seed + 10_000 + opponent_index)
            elif opponent_name == "line":
                opponent = LineBuilderAgent(seed=args.seed + 10_000 + opponent_index)
            else:
                raise ValueError(f"Unknown opponent {opponent_name!r}")

            _, summary = evaluate_matchup(
                opponent,
                agent,
                games=args.games,
                seed=args.seed + ckpt_index * 100_000 + opponent_index * 10_000,
            )
            row = {
                "checkpoint": str(path),
                "algo": algo,
                "opponent": opponent_name,
                "episodes": int(payload.get("episodes", 0)),
                "opponent_wins": summary["agent_a_wins"],
                "agent_wins": summary["agent_b_wins"],
                "draws": summary["draws"],
                "agent_win_rate": summary["agent_b_win_rate"],
                "avg_steps": summary["avg_steps"],
                "avg_forfeits": summary["avg_forfeits"],
                "error": "",
            }
            rows.append(row)
            print(
                f"{path.name} vs {opponent_name}: "
                f"agent_win_rate={row['agent_win_rate']:.3f} "
                f"({row['agent_wins']}/{args.games})"
            )

    output = Path(args.output_csv)
    write_csv(output, rows)
    print(f"Wrote checkpoint benchmark CSV to {output}")


if __name__ == "__main__":
    main()
