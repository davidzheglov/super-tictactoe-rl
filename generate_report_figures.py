"""Generate all figures for the research report.

Produces:
  figures/01_qlearning_evolution.png   - Q-table state coverage + epsilon
  figures/02_qvalue_distribution.png  - final Q-value histogram
  figures/03_dqn_learning.png         - DQN loss + epsilon
  figures/04_baseline_ppo.png         - baseline PPO (short run) loss + entropy
  figures/05_research_ppo_winrate.png - PPO & DetPPO win-rate over 300k episodes
  figures/06_research_ppo_entropy.png - entropy comparison PPO vs DetPPO
  figures/07_benchmark_comparison.png - bar chart: agent win rates vs each opponent
  figures/08_bc_pretrain_baseline.png - BC vs random vs heuristic benchmark
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RUNS = ROOT / "runs"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

PALETTE = {
    "ppo": "#2196F3",
    "detppo": "#FF9800",
    "dqn": "#9C27B0",
    "qlearning": "#4CAF50",
    "heuristic": "#F44336",
    "line": "#795548",
    "basic": "#607D8B",
    "bc": "#00BCD4",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # drop repeated header rows (artifact of appended logs)
    if "episodes" in df.columns:
        df = df[df["episodes"] != "episodes"].copy()
        df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
        df = df.dropna(subset=["episodes"]).reset_index(drop=True)
    return df


def savefig(fig: plt.Figure, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def smoothed(series: pd.Series, w: int = 5) -> pd.Series:
    return series.rolling(w, min_periods=1, center=True).mean()


# ── figure 1: Q-learning state coverage ──────────────────────────────────────

def fig_qlearning_evolution() -> None:
    # use overnight run (75k episodes) — much richer than the 15k run
    for candidate in [
        RUNS / "overnight_mixed_torch/q_learning_seed0/q_learning_log.csv",
        RUNS / "heur_line_torch/q_learning_seed0/q_learning_log.csv",
    ]:
        if candidate.exists():
            break
    df = load_csv(candidate)
    if df.empty:
        print("  skip fig 1: no q_learning_log.csv")
        return
    for col in ("states", "epsilon", "steps"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1, ax2 = axes

    ax1.plot(df["episodes"], df["states"] / 1e3, color=PALETTE["qlearning"], lw=1.8)
    ax1.set_ylabel("Unique states discovered (thousands)")
    ax1.set_title("Q-Learning: Table Growth and Exploration Decay")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    ax1.grid(alpha=0.25)

    ax2.plot(df["episodes"], df["epsilon"], color="#FF5722", lw=1.8, label="ε")
    ax2.set_ylabel("Exploration rate (ε)")
    ax2.set_xlabel("Training episodes")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    savefig(fig, "01_qlearning_evolution.png")


# ── figure 2: Q-value distribution ───────────────────────────────────────────

def fig_qvalue_distribution() -> None:
    # prefer overnight run (75k eps, 3.28M states) over short run
    for candidate in [
        RUNS / "overnight_mixed_torch/q_learning_seed0/q_table.pkl",
        RUNS / "heur_line_torch/q_learning_seed0/q_table.pkl",
    ]:
        if candidate.exists():
            pkl_path = candidate
            break
    else:
        print("  skip fig 2: no q_table.pkl")
        return
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    # saved as {"algo":..., "episodes":..., "q_table": {state_key: np.array}}
    if isinstance(payload, dict) and "q_table" in payload:
        raw = payload["q_table"]
    else:
        raw = payload

    all_q: list[np.ndarray] = []
    max_per_state: list[float] = []
    for v in raw.values():
        try:
            arr = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            continue
        if arr.ndim == 0:
            continue
        all_q.append(arr)
        max_per_state.append(float(arr.max()))

    all_q_flat = np.concatenate(all_q)
    max_per_state_arr = np.array(max_per_state)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(all_q_flat, bins=80, color=PALETTE["qlearning"], alpha=0.8, edgecolor="none")
    axes[0].set_xlabel("Q-value")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Distribution of all Q-values\n({len(all_q_flat):,} entries, {len(all_q):,} states)")
    axes[0].grid(alpha=0.2)

    axes[1].hist(max_per_state_arr, bins=60, color="#8BC34A", alpha=0.8, edgecolor="none")
    axes[1].set_xlabel("Max Q-value per state")
    axes[1].set_ylabel("Number of states")
    axes[1].set_title("Best-action Q-value distribution\nacross all visited states")
    axes[1].grid(alpha=0.2)

    fig.suptitle("Final Q-Table Value Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "02_qvalue_distribution.png")


# ── figure 3: DQN learning ────────────────────────────────────────────────────

def fig_dqn_learning() -> None:
    df = load_csv(RUNS / "heur_line_torch/dqn_seed0/dqn_log.csv")
    if df.empty:
        print("  skip fig 3: no dqn_log.csv")
        return
    for col in ("loss", "epsilon", "replay_size", "steps"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(df["episodes"], smoothed(df["loss"], 15),
                 color=PALETTE["dqn"], lw=1.8, label="TD loss (smoothed)")
    axes[0].set_ylabel("TD Loss")
    axes[0].set_title("DQN Training: Loss and Exploration")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(df["episodes"], df["epsilon"], color="#FF5722", lw=1.8, label="ε-greedy")
    if "replay_size" in df.columns:
        ax_r = axes[1].twinx()
        ax_r.plot(df["episodes"], df["replay_size"] / 1e3,
                  color="#9E9E9E", lw=1.2, alpha=0.6, linestyle="--", label="Replay buf (k)")
        ax_r.set_ylabel("Replay buffer size (k)")
        ax_r.legend(loc="upper right")
    axes[1].set_ylabel("ε")
    axes[1].set_xlabel("Training episodes")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left")

    fig.tight_layout()
    savefig(fig, "03_dqn_learning.png")


# ── figure 4: baseline PPO (short run) ───────────────────────────────────────

def fig_baseline_ppo() -> None:
    df = load_csv(RUNS / "heur_line_torch/ppo_seed0/ppo_log.csv")
    if df.empty:
        print("  skip fig 4: no baseline ppo_log.csv")
        return
    for col in ("loss", "entropy", "avg_length", "avg_forfeits"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(df["episodes"], df["loss"], color=PALETTE["ppo"], lw=1.8, label="Total loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Baseline PPO (6k episodes, sparse reward)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(df["episodes"], df["entropy"], color="#9C27B0", lw=1.8, label="Policy entropy")
    axes[1].set_ylabel("Entropy")
    axes[1].set_xlabel("Training episodes")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    savefig(fig, "04_baseline_ppo.png")


# ── figure 5: research PPO win rate over 300k ────────────────────────────────

def fig_research_ppo_winrate() -> None:
    ppo_df = load_csv(RUNS / "research_bc_ppo_300k/ppo_seed0/ppo_log.csv")
    det_df = load_csv(RUNS / "research_bc_ppo_300k/ppo_deterministic_seed0/ppo_log.csv")

    def compute_winrate(df: pd.DataFrame) -> pd.DataFrame:
        for col in ("x_wins", "o_wins", "batch_episodes"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "batch_episodes" not in df.columns:
            df["batch_episodes"] = 512
        df["win_rate"] = df["x_wins"] / df["batch_episodes"]
        return df

    if ppo_df.empty and det_df.empty:
        print("  skip fig 5: no research ppo logs")
        return

    ppo_df = compute_winrate(ppo_df) if not ppo_df.empty else ppo_df
    det_df = compute_winrate(det_df) if not det_df.empty else det_df

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 20
    if not ppo_df.empty:
        ax.plot(ppo_df["episodes"], smoothed(ppo_df["win_rate"], w),
                color=PALETTE["ppo"], lw=2, label="PPO (stochastic, mixed opponents)")
        ax.fill_between(ppo_df["episodes"],
                        ppo_df["win_rate"].rolling(w, min_periods=1).min(),
                        ppo_df["win_rate"].rolling(w, min_periods=1).max(),
                        alpha=0.15, color=PALETTE["ppo"])
    if not det_df.empty:
        ax.plot(det_df["episodes"], smoothed(det_df["win_rate"], w),
                color=PALETTE["detppo"], lw=2, label="DetPPO (deterministic, vs heuristic)")
        ax.fill_between(det_df["episodes"],
                        det_df["win_rate"].rolling(w, min_periods=1).min(),
                        det_df["win_rate"].rolling(w, min_periods=1).max(),
                        alpha=0.15, color=PALETTE["detppo"])

    ax.axhline(0.5, color="grey", lw=1, linestyle="--", label="50% baseline")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Agent win rate (per batch)")
    ax.set_title("Research-scale PPO: Win Rate over 300k Episodes\n(after BC warm-start)")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, "05_research_ppo_winrate.png")


# ── figure 6: entropy comparison ─────────────────────────────────────────────

def fig_entropy_comparison() -> None:
    ppo_df = load_csv(RUNS / "research_bc_ppo_300k/ppo_seed0/ppo_log.csv")
    det_df = load_csv(RUNS / "research_bc_ppo_300k/ppo_deterministic_seed0/ppo_log.csv")

    for df in (ppo_df, det_df):
        if not df.empty and "entropy" in df.columns:
            df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")

    if ppo_df.empty and det_df.empty:
        print("  skip fig 6: no research logs")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 15
    if not ppo_df.empty and "entropy" in ppo_df.columns:
        ax.plot(ppo_df["episodes"], smoothed(ppo_df["entropy"], w),
                color=PALETTE["ppo"], lw=2, label="PPO (stochastic)")
    if not det_df.empty and "entropy" in det_df.columns:
        ax.plot(det_df["episodes"], smoothed(det_df["entropy"], w),
                color=PALETTE["detppo"], lw=2, label="DetPPO (deterministic)")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Policy entropy (nats)")
    ax.set_title("Policy Entropy During Training\n(higher = more exploratory, lower = more confident)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, "06_research_ppo_entropy.png")


# ── figure 7: benchmark comparison (from completed benchmark CSVs) ────────────

def fig_benchmark_comparison() -> None:
    sources = {
        "PPO": RUNS / "benchmarks_ppo/benchmark_summary.csv",
        "DetPPO": RUNS / "benchmarks_detppo/benchmark_summary.csv",
    }
    rows = []
    for agent_name, path in sources.items():
        df = load_csv(path)
        if df.empty:
            continue
        for _, row in df.iterrows():
            opp = str(row.get("agent_b", "")).lower().replace("_", " ")
            wr_a = float(row.get("agent_a_win_rate", 0))
            wr_b = float(row.get("agent_b_win_rate", 0))
            rows.append({"agent": agent_name, "opponent": opp,
                         "agent_win_rate": wr_a, "opp_win_rate": wr_b})

    if not rows:
        print("  skip fig 7: benchmark CSVs not yet written — run after benchmarks complete")
        return

    results = pd.DataFrame(rows)
    opponents = sorted(results["opponent"].unique())
    agents = sorted(results["agent"].unique())
    x = np.arange(len(opponents))
    width = 0.8 / max(len(agents), 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, agent in enumerate(agents):
        subset = results[results["agent"] == agent].drop_duplicates("opponent").set_index("opponent")
        rates = np.array([
            float(subset.loc[opp, "agent_win_rate"]) if opp in subset.index else float("nan")
            for opp in opponents
        ])
        offsets = x + i * width - (len(agents) - 1) * width / 2
        for j, (off, rate) in enumerate(zip(offsets, rates)):
            if np.isnan(rate):
                continue
            bar = ax.bar(off, rate, width * 0.9,
                         label=agent if j == 0 else "_nolegend_",
                         color=PALETTE.get(agent.lower().replace(" ", ""), f"C{i}"))
            if rate > 0.02:
                ax.text(off, rate + 0.01, f"{rate:.0%}",
                        ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="grey", lw=1, linestyle="--", label="50%")
    ax.set_xticks(x)
    ax.set_xticklabels([o.title() for o in opponents], rotation=15)
    ax.set_ylabel("Agent win rate")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Final Benchmark: PPO & DetPPO Win Rates vs All Opponents\n(500 games each)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, "07_benchmark_comparison.png")


# ── figure 8: BC pretrain vs baselines ───────────────────────────────────────

def fig_bc_pretrain_baseline() -> None:
    ckpt = load_csv(RUNS / "research_bc_ppo_300k/checkpoint_benchmark_local.csv")
    if ckpt.empty:
        print("  skip fig 8: no checkpoint_benchmark_local.csv")
        return

    ckpt["agent_win_rate"] = pd.to_numeric(ckpt["agent_win_rate"], errors="coerce")
    ckpt["episodes"] = pd.to_numeric(ckpt["episodes"], errors="coerce")

    fig, ax = plt.subplots(figsize=(9, 5))
    for opp, color in [("heuristic", PALETTE["heuristic"]), ("line", PALETTE["line"])]:
        sub = ckpt[ckpt["opponent"] == opp].sort_values("episodes")
        # label by checkpoint label
        labels = sub.apply(lambda r: "BC" if r["episodes"] == 0 else f"PPO\n{int(r['episodes']//1000)}k",
                           axis=1)
        bars = ax.bar(
            [f"{a}\nvs {b}" for a, b in zip(sub["checkpoint"].apply(
                lambda x: "BC" if "behavior_clone" in str(x) else
                          ("PPO" if "ppo_seed0" in str(x) else "DetPPO")),
                sub["opponent"])],
            sub["agent_win_rate"],
            color=color, alpha=0.8, label=f"vs {opp}"
        )
        for bar, rate in zip(bars, sub["agent_win_rate"]):
            if not np.isnan(rate):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{rate:.0%}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0.5, color="grey", lw=1, linestyle="--", label="50%")
    ax.set_ylabel("Agent win rate")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Checkpoint Benchmark: BC Pretrain and Early PPO\n(100 games each)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, "08_bc_pretrain_baseline.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Generating report figures → {OUT}/")
    fig_qlearning_evolution()
    fig_qvalue_distribution()
    fig_dqn_learning()
    fig_baseline_ppo()
    fig_research_ppo_winrate()
    fig_entropy_comparison()
    fig_benchmark_comparison()
    fig_bc_pretrain_baseline()
    print("Done.")


if __name__ == "__main__":
    main()
