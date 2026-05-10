"""Generate all research report figures.

Figures produced:
  01_qlearning_evolution.png      - Q-table state coverage + epsilon decay (75k episodes)
  03_ppo_full_training.png        - Full PPO & DetPPO training curve (ep 5k → 300k)
  04_ppo_entropy_decay.png        - Entropy comparison: PPO vs DetPPO over 300k
  05_checkpoint_vs_opponents.png  - Checkpoint win rate vs heuristic AND line-builder
  06_checkpoint_head2head.png     - Early checkpoint vs Late checkpoint head-to-head
  07_benchmark_final.png          - Final benchmark bar chart (all agents, both opponents)
  08_teammate_checkpoint.png      - Teammate PPO curriculum: win rate over training
  09_teammate_vs_heuristics.png   - Teammate best model vs full heuristic ladder
"""
from __future__ import annotations

import os
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

TEAMMATE_ROOT = Path("/Users/davidzheglov/Desktop/projects/rl/tic-tac-toe")

PALETTE = {
    "ppo":        "#2196F3",
    "detppo":     "#FF9800",
    "heuristic":  "#E53935",
    "line":       "#6D4C41",
    "basic":      "#78909C",
    "bc":         "#00ACC1",
    "teammate":   "#8E24AA",
}

STYLE = dict(linewidth=2, solid_capstyle="round")


# ─── helpers ──────────────────────────────────────────────────────────────────

def savefig(fig: plt.Figure, name: str) -> None:
    p = OUT / name
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def smooth(s: pd.Series, w: int = 10) -> pd.Series:
    return s.rolling(w, min_periods=1, center=True).mean()


def load_full_ppo_log(seed_dir: Path) -> pd.DataFrame:
    """Combine sequential (pre-vectorized) + full vectorized log for a run."""
    frames = []
    for fname in [
        "ppo_log_sequential_before_vectorized_20260508_175008.csv",
        "ppo_log_full.csv",
    ]:
        p = seed_dir / fname
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["episodes"] != "episodes"]  # drop repeated headers
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        df["episodes"] = pd.to_numeric(df["episodes"], errors="coerce")
        df = df.dropna(subset=["episodes"]).drop_duplicates("episodes").sort_values("episodes")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames).drop_duplicates("episodes").sort_values("episodes").reset_index(drop=True)
    return combined


# ─── figure 01: Q-learning state coverage ─────────────────────────────────────

def fig_qlearning_evolution() -> None:
    # Use the overnight 75k run
    p = RUNS / "overnight_mixed_torch/q_learning_seed0/q_learning_log.csv"
    if not p.exists():
        print("  skip 01: no overnight q_learning_log")
        return
    df = pd.read_csv(p)
    df["states"] = pd.to_numeric(df["states"], errors="coerce")
    df["epsilon"] = pd.to_numeric(df["epsilon"], errors="coerce")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(df["episodes"], df["states"] / 1e6, color=PALETTE["bc"], **STYLE)
    ax1.set_ylabel("Unique states discovered (millions)")
    ax1.set_title("Q-Learning: State Space Coverage over 75,000 Episodes", fontweight="bold")
    ax1.grid(alpha=0.25)
    ax1.annotate(
        f"Final: {df['states'].iloc[-1]/1e6:.2f}M states\n($<10^{{-38}}$% of true state space)",
        xy=(df["episodes"].iloc[-1], df["states"].iloc[-1]/1e6),
        xytext=(-80, -30), textcoords="offset points",
        fontsize=9, color="grey",
        arrowprops=dict(arrowstyle="->", color="grey"),
    )

    ax2.plot(df["episodes"], df["epsilon"], color="#FF5722", **STYLE)
    ax2.set_ylabel("Exploration rate ε")
    ax2.set_xlabel("Training episodes")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.25)
    ax2.set_title("ε-greedy Exploration: Transition from Exploration → Exploitation", fontweight="bold")

    fig.tight_layout(h_pad=2)
    savefig(fig, "01_qlearning_evolution.png")


# ─── figure 03: full PPO training curve (combined sequential + vectorized) ────

def fig_ppo_full_training() -> None:
    ppo_df = load_full_ppo_log(RUNS / "research_bc_ppo_300k/ppo_seed0")
    det_df = load_full_ppo_log(RUNS / "research_bc_ppo_300k/ppo_deterministic_seed0")

    if ppo_df.empty and det_df.empty:
        print("  skip 03: no ppo logs")
        return

    for df in (ppo_df, det_df):
        if not df.empty:
            df["win_rate"] = pd.to_numeric(df["x_wins"], errors="coerce") / pd.to_numeric(df["batch_episodes"], errors="coerce")

    fig, (ax_wr, ax_loss) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    w = 8

    # Win rate panel
    if not ppo_df.empty:
        ax_wr.plot(ppo_df["episodes"]/1000, smooth(ppo_df["win_rate"], w),
                   color=PALETTE["ppo"], label="PPO (stochastic, mixed opponents)", **STYLE)
        ax_wr.fill_between(ppo_df["episodes"]/1000,
                            ppo_df["win_rate"].rolling(w, min_periods=1).min(),
                            ppo_df["win_rate"].rolling(w, min_periods=1).max(),
                            alpha=0.12, color=PALETTE["ppo"])
    if not det_df.empty:
        ax_wr.plot(det_df["episodes"]/1000, smooth(det_df["win_rate"], w),
                   color=PALETTE["detppo"], label="DetPPO (deterministic placement, vs heuristic)", **STYLE)
        ax_wr.fill_between(det_df["episodes"]/1000,
                            det_df["win_rate"].rolling(w, min_periods=1).min(),
                            det_df["win_rate"].rolling(w, min_periods=1).max(),
                            alpha=0.12, color=PALETTE["detppo"])
    ax_wr.axhline(0.5, color="grey", lw=1.2, ls="--", label="50% (coin flip)")
    ax_wr.set_ylabel("Batch win rate (agent as X)")
    ax_wr.set_ylim(0.2, 1.0)
    ax_wr.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_wr.set_title("TorchRL PPO Training: Full 300k Episodes (BC warm-start → PPO)", fontweight="bold", fontsize=13)
    ax_wr.legend(loc="lower right")
    ax_wr.grid(alpha=0.25)

    # Loss panel
    for df, color, label in [(ppo_df, PALETTE["ppo"], "PPO"), (det_df, PALETTE["detppo"], "DetPPO")]:
        if df.empty or "loss" not in df.columns:
            continue
        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
        ax_loss.plot(df["episodes"]/1000, smooth(df["loss"], w),
                     color=color, label=f"{label} total loss", **STYLE)
    ax_loss.set_xlabel("Training episodes (thousands)")
    ax_loss.set_ylabel("PPO Loss")
    ax_loss.set_title("PPO Loss over Training (smoothed)", fontweight="bold")
    ax_loss.legend()
    ax_loss.grid(alpha=0.25)

    fig.tight_layout(h_pad=2)
    savefig(fig, "03_ppo_full_training.png")


# ─── figure 04: entropy comparison ────────────────────────────────────────────

def fig_ppo_entropy() -> None:
    ppo_df = load_full_ppo_log(RUNS / "research_bc_ppo_300k/ppo_seed0")
    det_df = load_full_ppo_log(RUNS / "research_bc_ppo_300k/ppo_deterministic_seed0")

    if ppo_df.empty and det_df.empty:
        print("  skip 04: no logs")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    w = 8
    for df, color, label in [
        (ppo_df, PALETTE["ppo"], "PPO (stochastic) — mixed opponents, diverse policy"),
        (det_df, PALETTE["detppo"], "DetPPO (deterministic) — focused on heuristic, confident policy"),
    ]:
        if df.empty or "entropy" not in df.columns:
            continue
        df["entropy"] = pd.to_numeric(df["entropy"], errors="coerce")
        ax.plot(df["episodes"]/1000, smooth(df["entropy"], w), color=color, label=label, **STYLE)

    ax.set_xlabel("Training episodes (thousands)")
    ax.set_ylabel("Policy entropy (nats)")
    ax.set_title(
        "Policy Entropy Over Training\n"
        "DetPPO converges to confident low-entropy policy; PPO stays exploratory for diverse opponents",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    ax.annotate("DetPPO: ~0.6 nats\n(confident, deterministic)", xy=(290, 0.6),
                 xytext=(200, 1.2), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=PALETTE["detppo"]), color=PALETTE["detppo"])
    ax.annotate("PPO: ~1.7 nats\n(exploratory, diverse)", xy=(290, 1.75),
                 xytext=(200, 2.5), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=PALETTE["ppo"]), color=PALETTE["ppo"])
    fig.tight_layout()
    savefig(fig, "04_ppo_entropy_decay.png")


# ─── figure 05: checkpoint win rate vs BOTH opponents ─────────────────────────

def fig_checkpoint_vs_opponents() -> None:
    """Re-runs checkpoint benchmark vs heuristic AND line builder (100 games each)."""
    try:
        from agents import HeuristicAgent, LineBuilderAgent, TorchPPOAgent, evaluate_matchup
    except ImportError:
        print("  skip 05: cannot import agents")
        return

    results = []
    opponent_map = {
        "heuristic": HeuristicAgent(seed=7),
        "line builder": LineBuilderAgent(seed=8),
    }

    checkpoints = {
        "PPO\n(stochastic)": [
            ("BC\n(ep 0)", RUNS / "research_bc_ppo_300k/behavior_clone_seed0/behavior_clone_torchrl.pt", False),
            *[
                (f"ep {ep//1000}k",
                 RUNS / f"research_bc_ppo_300k/ppo_seed0/checkpoints/ppo_ep{ep:07d}.pt",
                 False)
                for ep in [51200, 102400, 153600, 204800, 256000, 300000]
            ],
        ],
        "DetPPO\n(deterministic)": [
            ("BC\n(ep 0)", RUNS / "research_bc_ppo_300k/behavior_clone_seed0/behavior_clone_torchrl.pt", True),
            *[
                (f"ep {ep//1000}k",
                 RUNS / f"research_bc_ppo_300k/ppo_deterministic_seed0/checkpoints/ppo_ep{ep:07d}.pt",
                 True)
                for ep in [51200, 102400, 153600, 204800, 256000, 300000]
            ],
        ],
    }

    GAMES = 100
    for variant, ckpts in checkpoints.items():
        for label, path, det in ckpts:
            if not path.exists():
                continue
            try:
                agent = TorchPPOAgent(str(path), device="cpu", deterministic=det)
            except Exception:
                continue
            row = {"variant": variant, "label": label}
            for opp_name, opp in opponent_map.items():
                _, s = evaluate_matchup(agent, opp, games=GAMES, seed=42)
                row[opp_name] = s["agent_a_win_rate"]
                print(f"  {variant.replace(chr(10),' ')} {label}: {s['agent_a_win_rate']:.0%} vs {opp_name}")
            results.append(row)

    if not results:
        print("  skip 05: no results")
        return

    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    opponents = ["heuristic", "line builder"]
    colors_opp = [PALETTE["heuristic"], PALETTE["line"]]

    for ax, variant in zip(axes, ["PPO\n(stochastic)", "DetPPO\n(deterministic)"]):
        sub = df[df["variant"] == variant].reset_index(drop=True)
        x = np.arange(len(sub))
        width = 0.38
        for i, (opp, col) in enumerate(zip(opponents, colors_opp)):
            bars = ax.bar(x + i * width - width/2, sub[opp], width * 0.9,
                          color=col, alpha=0.85, label=f"vs {opp}", zorder=3)
            for bar, val in zip(bars, sub[opp]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.0%}", ha="center", va="bottom", fontsize=8)
        ax.axhline(0.5, color="grey", ls="--", lw=1.2, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["label"], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(variant.replace("\n", " "), fontweight="bold", fontsize=11)
        ax.set_xlabel("Training checkpoint")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.25, zorder=0)

    axes[0].set_ylabel("Win rate vs fixed opponent (100 games)")
    fig.suptitle(
        "PPO Training Progress: Win Rate at Each Checkpoint\nvs Smart Heuristic and Line-Builder",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    savefig(fig, "05_checkpoint_vs_opponents.png")


# ─── figure 06: early vs late head-to-head ───────────────────────────────────

def fig_head2head() -> None:
    """Early checkpoint (ep 51k) vs Late checkpoint (ep 300k) — head to head."""
    try:
        from agents import TorchPPOAgent, evaluate_matchup
    except ImportError:
        print("  skip 06: cannot import agents")
        return

    matchups = [
        ("PPO ep 51k",  RUNS / "research_bc_ppo_300k/ppo_seed0/checkpoints/ppo_ep0051200.pt", False),
        ("PPO ep 300k", RUNS / "research_bc_ppo_300k/ppo_seed0/checkpoints/ppo_ep0300000.pt", False),
        ("DetPPO ep 51k",  RUNS / "research_bc_ppo_300k/ppo_deterministic_seed0/checkpoints/ppo_ep0051200.pt", True),
        ("DetPPO ep 300k", RUNS / "research_bc_ppo_300k/ppo_deterministic_seed0/checkpoints/ppo_ep0300000.pt", True),
    ]

    agents = {}
    for name, path, det in matchups:
        if path.exists():
            try:
                agents[name] = TorchPPOAgent(str(path), device="cpu", deterministic=det)
            except Exception:
                pass

    GAMES = 100
    pairs = [
        ("PPO ep 51k", "PPO ep 300k", "PPO: Early vs Late"),
        ("DetPPO ep 51k", "DetPPO ep 300k", "DetPPO: Early vs Late"),
        ("PPO ep 300k", "DetPPO ep 300k", "Final: PPO vs DetPPO"),
    ]
    rows = []
    for a_name, b_name, title in pairs:
        if a_name not in agents or b_name not in agents:
            continue
        _, s = evaluate_matchup(agents[a_name], agents[b_name], games=GAMES, seed=99)
        rows.append({"matchup": title, "A": a_name, "B": b_name,
                     "A_wr": s["agent_a_win_rate"], "B_wr": s["agent_b_win_rate"]})
        print(f"  {title}: {a_name} {s['agent_a_win_rate']:.0%} / {b_name} {s['agent_b_win_rate']:.0%}")

    if not rows:
        print("  skip 06: no head2head results")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w/2, df["A_wr"], w * 0.9, color=PALETTE["ppo"], alpha=0.85, label="First agent")
    ax.bar(x + w/2, df["B_wr"], w * 0.9, color=PALETTE["detppo"], alpha=0.85, label="Second agent")
    for i, row in df.iterrows():
        ax.text(i - w/2, row["A_wr"] + 0.01, f"{row['A_wr']:.0%}", ha="center", fontsize=9)
        ax.text(i + w/2, row["B_wr"] + 0.01, f"{row['B_wr']:.0%}", ha="center", fontsize=9)
        ax.text(i, 0.02, f"{row['A']}\nvs\n{row['B']}", ha="center", fontsize=7.5, color="grey")
    ax.axhline(0.5, color="grey", ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["matchup"], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Win rate (100 games)")
    ax.set_title("Head-to-Head: Early vs Late Checkpoint, PPO vs DetPPO", fontweight="bold", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    savefig(fig, "06_checkpoint_head2head.png")


# ─── figure 07: final benchmark bar chart ─────────────────────────────────────

def fig_final_benchmark() -> None:
    data = {
        "PPO 300k\n(stochastic)": {
            "vs Smart Heuristic": 0.42, "vs Line Builder": 0.39, "vs Basic Heuristic": 0.60,
        },
        "DetPPO 300k\n(deterministic)": {
            "vs Smart Heuristic": 0.635, "vs Line Builder": 0.415, "vs Basic Heuristic": 0.735,
        },
        "BC Pretrain\n(ep 0, no RL)": {
            "vs Smart Heuristic": 0.45, "vs Line Builder": 0.44, "vs Basic Heuristic": None,
        },
    }

    opponents = ["vs Smart Heuristic", "vs Line Builder", "vs Basic Heuristic"]
    agents_order = list(data.keys())
    colors = [PALETTE["ppo"], PALETTE["detppo"], PALETTE["bc"]]

    x = np.arange(len(opponents))
    total_w = 0.75
    w = total_w / len(agents_order)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (agent, col) in enumerate(zip(agents_order, colors)):
        vals = np.array([data[agent].get(o) for o in opponents], dtype=object)
        offset = x + (i - len(agents_order)/2 + 0.5) * w
        for j, (off, val) in enumerate(zip(offset, vals)):
            if val is None:
                continue
            b = ax.bar(off, float(val), w * 0.88, color=col, alpha=0.85,
                       label=agent if j == 0 else "_nolegend_", zorder=3)
            ax.text(off, float(val) + 0.015, f"{val:.0%}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0.5, color="grey", ls="--", lw=1.2, label="50% baseline", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(opponents, fontsize=11)
    ax.set_ylim(0, 0.95)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Win rate (200 games per matchup)")
    ax.set_title(
        "Final Evaluation: Win Rates vs All Opponents\n(200 games each, CPU inference, seed=0)",
        fontweight="bold", fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()
    savefig(fig, "07_benchmark_final.png")


# ─── figure 08: teammate checkpoint progress ──────────────────────────────────

def fig_teammate_checkpoint() -> None:
    """Benchmark teammate checkpoints (every 5th) vs their counter and blocking heuristics."""
    try:
        sys.path.insert(0, str(TEAMMATE_ROOT))
        from super_tictactoe.env import SuperTicTacToeEnv
        from super_tictactoe.model import ActorCritic
        from super_tictactoe.heuristics import counter_heuristic, blocking_agent
    except ImportError as e:
        print(f"  skip 08: cannot import teammate module: {e}")
        return

    ckpt_dir = TEAMMATE_ROOT / "checkpoints_ppo_cl"
    if not ckpt_dir.exists():
        print(f"  skip 08: {ckpt_dir} not found")
        return

    # Every 5th checkpoint
    all_ckpts = sorted(ckpt_dir.glob("model_[0-9]*.pt"))
    selected = all_ckpts[::5] + [ckpt_dir / "model_final.pt"]
    selected = [p for p in selected if p.exists()]

    def play_vs_heuristic(model_path: Path, heuristic_fn, n_games: int = 60) -> float:
        env = SuperTicTacToeEnv()
        model = ActorCritic()
        state_dict = torch.load(str(model_path), map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        wins = 0
        for g in range(n_games):
            state = env.reset()
            # alternate sides
            model_is_p1 = (g % 2 == 0)
            while not env.done:
                if (env.current_player == 1) == model_is_p1:
                    mask = torch.BoolTensor(env.get_action_mask())
                    s_t = torch.FloatTensor(state)
                    with torch.no_grad():
                        action, _, _ = model.get_action(s_t, mask, deterministic=True)
                else:
                    action = heuristic_fn(env)
                state, _, _, _ = env.step(action)
            model_player = 1 if model_is_p1 else 2
            if env.winner == model_player:
                wins += 1
        return wins / n_games

    import torch
    rows = []
    for path in selected:
        name = path.stem
        update = int(name.split("_")[1]) if "_" in name and name.split("_")[1].isdigit() else 3000
        wr_counter = play_vs_heuristic(path, counter_heuristic)
        wr_blocking = play_vs_heuristic(path, blocking_agent)
        rows.append({"update": update, "vs counter": wr_counter, "vs blocking": wr_blocking})
        print(f"  teammate {name}: {wr_counter:.0%} vs counter, {wr_blocking:.0%} vs blocking")

    if not rows:
        print("  skip 08: no results")
        return

    df = pd.DataFrame(rows).sort_values("update")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["update"], df["vs counter"], color=PALETTE["heuristic"], marker="o", ms=5,
            label="vs Counter heuristic", **STYLE)
    ax.plot(df["update"], df["vs blocking"], color=PALETTE["line"], marker="s", ms=5,
            label="vs Blocking heuristic", **STYLE)
    ax.axhline(0.5, color="grey", ls="--", lw=1.2)
    ax.set_xlabel("PPO curriculum training updates")
    ax.set_ylabel("Win rate (60 games, alternating sides)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Teammate's PPO Curriculum: Win Rate at Each Checkpoint\nvs Counter and Blocking Heuristics",
                 fontweight="bold", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, "08_teammate_checkpoint.png")


# ─── figure 09: teammate best model vs full heuristic ladder ──────────────────

def fig_teammate_vs_heuristics() -> None:
    """Teammate's best model vs full heuristic ladder (greedy → stronger)."""
    try:
        sys.path.insert(0, str(TEAMMATE_ROOT))
        import torch
        from super_tictactoe.env import SuperTicTacToeEnv
        from super_tictactoe.model import ActorCritic
        from super_tictactoe.heuristics import (
            greedy_agent, blocking_agent, safe_agent,
            counter_heuristic, stronger_heuristic,
        )
    except ImportError as e:
        print(f"  skip 09: {e}")
        return

    best_path = TEAMMATE_ROOT / "checkpoints_ppo_finetune/model_final.pt"
    if not best_path.exists():
        best_path = TEAMMATE_ROOT / "checkpoints_ppo_cl/model_final.pt"
    if not best_path.exists():
        print("  skip 09: no model found")
        return

    model = ActorCritic()
    sd = torch.load(str(best_path), map_location="cpu")
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()

    heuristics = [
        ("Greedy", greedy_agent),
        ("Blocking", blocking_agent),
        ("Safe", safe_agent),
        ("Counter", counter_heuristic),
        ("Stronger", stronger_heuristic),
    ]

    N = 60
    env = SuperTicTacToeEnv()
    results = []
    for hname, hfn in heuristics:
        wins = 0
        for g in range(N):
            state = env.reset()
            model_is_p1 = (g % 2 == 0)
            while not env.done:
                if (env.current_player == 1) == model_is_p1:
                    mask = torch.BoolTensor(env.get_action_mask())
                    with torch.no_grad():
                        action, _, _ = model.get_action(torch.FloatTensor(state), mask, deterministic=True)
                else:
                    action = hfn(env)
                state, _, _, _ = env.step(action)
            if env.winner == (1 if model_is_p1 else 2):
                wins += 1
        wr = wins / N
        results.append({"opponent": hname, "win_rate": wr})
        print(f"  teammate vs {hname}: {wr:.0%}")

    df = pd.DataFrame(results)
    colors_ladder = [PALETTE["bc"], PALETTE["ppo"], PALETTE["line"], PALETTE["heuristic"], "#B71C1C"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(df["opponent"], df["win_rate"], color=colors_ladder, alpha=0.85, zorder=3)
    for bar, val in zip(bars, df["win_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.015, f"{val:.0%}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="grey", ls="--", lw=1.2, label="50%")
    ax.set_xlabel("Heuristic opponent (weakest → strongest)")
    ax.set_ylabel("Teammate's agent win rate (60 games)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Teammate's Best PPO vs Full Heuristic Ladder\n(60 games each, alternating sides)",
                 fontweight="bold", fontsize=12)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.tight_layout()
    savefig(fig, "09_teammate_vs_heuristics.png")


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Generating all report figures → {OUT}/")
    fig_qlearning_evolution()
    fig_ppo_full_training()
    fig_ppo_entropy()
    fig_final_benchmark()       # fast (no game running)
    fig_checkpoint_vs_opponents()   # ~700 games, ~5 min CPU
    fig_head2head()                 # ~300 games, ~2 min CPU
    fig_teammate_checkpoint()       # ~420 games, ~5 min CPU
    fig_teammate_vs_heuristics()    # ~300 games, ~3 min CPU
    print("\nDone.")


if __name__ == "__main__":
    main()
