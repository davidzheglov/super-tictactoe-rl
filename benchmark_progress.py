"""Benchmark each PPO checkpoint vs smart heuristic to show training progress.

Runs 50 games per checkpoint — lightweight, completes in ~2-3 minutes.
Writes: runs/checkpoint_progress.csv and figures/09_checkpoint_progress.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from agents import HeuristicAgent, TorchPPOAgent, evaluate_matchup
except ImportError:
    from .agents import HeuristicAgent, TorchPPOAgent, evaluate_matchup

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
GAMES = 50


def run_checkpoint_benchmark() -> list[dict]:
    rows = []

    # BC pretrain (episode 0)
    bc_path = RUNS / "research_bc_ppo_300k/behavior_clone_seed0/behavior_clone_torchrl.pt"

    checkpoints = {
        "ppo": [
            (0, bc_path, False),
            *(
                (ep, RUNS / f"research_bc_ppo_300k/ppo_seed0/checkpoints/ppo_ep{ep:07d}.pt", False)
                for ep in [51200, 102400, 153600, 204800, 256000, 300000]
            ),
        ],
        "detppo": [
            (0, bc_path, True),
            *(
                (ep, RUNS / f"research_bc_ppo_300k/ppo_deterministic_seed0/checkpoints/ppo_ep{ep:07d}.pt", True)
                for ep in [51200, 102400, 153600, 204800, 256000, 300000]
            ),
        ],
    }

    heuristic = HeuristicAgent(seed=42)

    for variant, ckpts in checkpoints.items():
        for ep, path, deterministic in ckpts:
            if not path.exists():
                print(f"  skip {path.name} (not found)")
                continue
            try:
                agent = TorchPPOAgent(str(path), device="cpu", deterministic=deterministic)
            except Exception as exc:
                print(f"  skip {path.name}: {exc}")
                continue
            _, summary = evaluate_matchup(agent, heuristic, games=GAMES, seed=ep + 1)
            win_rate = summary["agent_a_win_rate"]
            print(f"  {variant} ep={ep:>7d}: {win_rate:.1%} vs smart heuristic ({GAMES} games)")
            rows.append({
                "variant": variant,
                "episodes": ep,
                "checkpoint": path.name,
                "win_rate": win_rate,
                "wins": summary["agent_a_wins"],
                "losses": summary["agent_b_wins"],
                "games": GAMES,
            })

    return rows


def save_csv(rows: list[dict]) -> Path:
    path = RUNS / "checkpoint_progress.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def plot_progress(rows: list[dict]) -> Path:
    import pandas as pd
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"ppo": "#2196F3", "detppo": "#FF9800"}
    labels = {"ppo": "PPO (stochastic, mixed curriculum)", "detppo": "DetPPO (deterministic, vs heuristic)"}

    for variant, grp in df.groupby("variant"):
        grp = grp.sort_values("episodes")
        ax.plot(grp["episodes"] / 1000, grp["win_rate"],
                marker="o", lw=2, color=colors.get(variant, "grey"),
                label=labels.get(variant, variant))
        for _, row in grp.iterrows():
            ax.annotate(f"{row['win_rate']:.0%}",
                        (row["episodes"] / 1000, row["win_rate"]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=colors.get(variant, "grey"))

    ax.axhline(0.5, color="grey", lw=1, linestyle="--", label="50% baseline")
    ax.set_xlabel("Training episodes (thousands)")
    ax.set_ylabel("Win rate vs smart heuristic")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Training Progress: Win Rate vs Smart Heuristic\nacross All TorchRL PPO Checkpoints")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out = ROOT / "figures" / "09_checkpoint_progress.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    print(f"Benchmarking checkpoints ({GAMES} games each vs smart heuristic)...")
    rows = run_checkpoint_benchmark()
    if not rows:
        print("No checkpoints found.")
        return
    csv_path = save_csv(rows)
    print(f"Wrote {csv_path}")
    fig_path = plot_progress(rows)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
