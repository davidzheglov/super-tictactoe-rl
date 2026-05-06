"""Create learning-curve and benchmark figures from training outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def save_line_plot(df: pd.DataFrame, x: str, ys: Iterable[str], title: str, path: Path) -> None:
    cols = [col for col in ys if col in df.columns]
    if df.empty or x not in df.columns or not cols:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in cols:
        ax.plot(df[x], df[col], label=col)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_training_logs(run_dir: Path, output_dir: Path) -> List[Path]:
    written: List[Path] = []
    for csv_path in sorted(run_dir.glob("**/*_log.csv")):
        df = read_csv_if_exists(csv_path)
        if df.empty or "episodes" not in df.columns:
            continue
        rel_name = "_".join(csv_path.relative_to(run_dir).with_suffix("").parts)
        if "ppo" in csv_path.name:
            plot_path = output_dir / f"{rel_name}_ppo_learning.png"
            save_line_plot(
                df,
                "episodes",
                ["loss", "policy_loss", "value_loss", "entropy", "avg_forfeits"],
                f"PPO Learning: {csv_path.parent.name}",
                plot_path,
            )
            if plot_path.exists():
                written.append(plot_path)
            mix_path = output_dir / f"{rel_name}_ppo_opponents.png"
            save_line_plot(
                df,
                "episodes",
                ["self_games", "heuristic_games", "line_games", "basic_games", "random_games"],
                f"PPO Opponent Mix: {csv_path.parent.name}",
                mix_path,
            )
            if mix_path.exists():
                written.append(mix_path)
        elif "dqn" in csv_path.name:
            plot_path = output_dir / f"{rel_name}_dqn_learning.png"
            save_line_plot(
                df,
                "episodes",
                ["loss", "epsilon", "replay_size", "forfeits"],
                f"DQN Learning: {csv_path.parent.name}",
                plot_path,
            )
            if plot_path.exists():
                written.append(plot_path)
        elif "q_learning" in csv_path.name:
            plot_path = output_dir / f"{rel_name}_q_learning.png"
            save_line_plot(
                df,
                "episodes",
                ["states", "epsilon", "forfeits", "steps"],
                f"Q-learning Table Growth: {csv_path.parent.name}",
                plot_path,
            )
            if plot_path.exists():
                written.append(plot_path)
    return written


def plot_benchmarks(run_dir: Path, output_dir: Path) -> List[Path]:
    candidates = sorted(run_dir.glob("**/benchmark_summary.csv"))
    if not candidates:
        return []
    frames = []
    for path in candidates:
        df = read_csv_if_exists(path)
        if df.empty:
            continue
        df["source"] = str(path.relative_to(run_dir))
        frames.append(df)
    if not frames:
        return []

    summary = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_cols = [col for col in ["run_name", "algo", "agent_a", "baseline", "agent_b"] if col in summary.columns]
    if "agent_a_win_rate" in summary.columns:
        ranked = summary.sort_values("agent_a_win_rate", ascending=False)
    else:
        ranked = summary
    ranked.to_csv(output_dir / "benchmark_summary_ranked.csv", index=False)

    label_col = "run_name" if "run_name" in summary.columns else "agent_a"
    baseline_col = "baseline" if "baseline" in summary.columns else "agent_b"
    top = ranked.head(30).copy()
    labels = [
        f"{row.get(label_col, 'agent')} vs {row.get(baseline_col, 'baseline')}"
        for _, row in top.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top))))
    ax.barh(labels[::-1], top["agent_a_win_rate"].to_numpy()[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Agent A win rate")
    ax.set_title("Benchmark Win Rates")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    plot_path = output_dir / "benchmark_win_rates.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    _ = rank_cols
    return [plot_path, output_dir / "benchmark_summary_ranked.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Super Tic-Tac-Toe training outputs.")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "figures"
    written = []
    written.extend(plot_training_logs(run_dir, output_dir))
    written.extend(plot_benchmarks(run_dir, output_dir))
    if written:
        print("Wrote figures/reports:")
        for path in written:
            print(path)
    else:
        print(f"No plottable logs found under {run_dir}")


if __name__ == "__main__":
    main()
