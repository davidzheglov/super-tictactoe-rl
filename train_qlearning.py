"""Resumable tabular Q-learning baseline for Super Tic-Tac-Toe."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np

try:
    from .agents import board_potential
    from .env import SuperTicTacToeEnv
    from .utils import project_root, random_legal_action, set_global_seeds
except ImportError:  # pragma: no cover
    from agents import board_potential
    from env import SuperTicTacToeEnv
    from utils import project_root, random_legal_action, set_global_seeds


def state_key(env: SuperTicTacToeEnv) -> tuple:
    return tuple(env.get_observation().astype(np.int8).tolist())


def default_q_table():
    return defaultdict(lambda: np.zeros(96, dtype=np.float32))


def load_q_table(path: Path):
    if not path.exists():
        return default_q_table(), 0
    with path.open("rb") as f:
        payload = pickle.load(f)
    table = default_q_table()
    table.update(payload.get("q_table", payload))
    return table, int(payload.get("episodes", 0))


def save_q_table(path: Path, q_table, episodes: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {
                "algo": "q_learning",
                "episodes": int(episodes),
                "seed": int(seed),
                "q_table": dict(q_table),
                "updated_at_unix": time.time(),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    with (path.with_suffix(path.suffix + ".json")).open("w", encoding="utf-8") as f:
        json.dump({"episodes": int(episodes), "algo": "q_learning", "seed": seed}, f, indent=2)


def append_csv_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train tabular Q-learning baseline.")
    parser.add_argument("--episodes", type=int, default=15000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eps-start", type=float, default=0.9)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shaping-scale", type=float, default=0.03)
    parser.add_argument("--shaping-clip", type=float, default=2.0)
    parser.add_argument("--shaping-defense-weight", type=float, default=0.75)
    parser.add_argument("--forfeit-penalty", type=float, default=0.02)
    parser.add_argument("--save-path", type=str, default=str(root / "models" / "q_table.pkl"))
    parser.add_argument("--log-csv", type=str, default=str(root / "models" / "q_learning_log.csv"))
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--snapshot-interval", type=int, default=0,
                        help="Save a numbered Q-table snapshot every N episodes (0 = off)")
    parser.add_argument("--snapshot-dir", type=str, default="",
                        help="Directory for numbered Q-table snapshots")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument("--done-file", type=str, default="")
    parser.add_argument("--stop-after-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_path = Path(args.save_path)
    done_file = Path(args.done_file or args.save_path + ".done")
    if args.skip_if_done and done_file.exists():
        print(f"Done marker exists; skipping Q-learning: {done_file}")
        return

    set_global_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.resume:
        q_table, start_episode = load_q_table(save_path)
        if start_episode:
            print(f"Resumed Q-table at episode {start_episode}")
    else:
        q_table, start_episode = default_q_table(), 0

    if args.skip_if_done and start_episode >= args.episodes:
        done_file.parent.mkdir(parents=True, exist_ok=True)
        done_file.write_text("done\n", encoding="utf-8")
        print(f"Q-table already has {start_episode} episodes; skipping.")
        return

    env = SuperTicTacToeEnv(seed=args.seed)
    started_at = time.time()
    for episode in range(start_episode, args.episodes):
        frac = min(1.0, episode / max(args.episodes, 1))
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        done = False
        steps = 0
        forfeits = 0
        last_info = {"winner": 0}

        while not done:
            key = state_key(env)
            mask = env.legal_action_mask()
            if rng.random() < epsilon:
                action = random_legal_action(mask, rng)
            else:
                q_values = q_table[key].copy()
                q_values[~mask] = -1.0e9
                action = int(np.argmax(q_values))

            acting_player = int(env.current_player)
            potential_before = (
                board_potential(
                    env.board,
                    acting_player,
                    defense_weight=args.shaping_defense_weight,
                )
                if args.shaping_scale != 0.0
                else 0.0
            )
            _, reward, terminated, truncated, last_info = env.step(action)
            done = bool(terminated or truncated)
            forfeits += int(bool(last_info.get("forfeited", False)))
            shaped_reward = float(reward)
            if args.shaping_scale != 0.0:
                potential_after = board_potential(
                    env.board,
                    acting_player,
                    defense_weight=args.shaping_defense_weight,
                )
                delta = float(np.clip(potential_after - potential_before, -args.shaping_clip, args.shaping_clip))
                shaped_reward += args.shaping_scale * delta
            if bool(last_info.get("forfeited", False)):
                shaped_reward -= args.forfeit_penalty
            if done:
                target = shaped_reward
            else:
                next_key = state_key(env)
                next_mask = env.legal_action_mask()
                next_values = q_table[next_key].copy()
                next_values[~next_mask] = -1.0e9
                target = shaped_reward - args.gamma * float(np.max(next_values))
            q_table[key][action] += args.alpha * (target - q_table[key][action])
            steps += 1

        episode_num = episode + 1
        if episode_num % args.log_interval == 0 or episode_num >= args.episodes:
            # Compute Q-value statistics across all visited states
            all_vals = []
            if q_table:
                for arr in q_table.values():
                    nz = arr[arr != 0.0]
                    if nz.size:
                        all_vals.append(nz)
            if all_vals:
                flat = np.concatenate(all_vals)
                q_mean = float(np.mean(flat))
                q_std = float(np.std(flat))
                q_max = float(np.max(flat))
                q_nonzero = int(flat.size)
            else:
                q_mean = q_std = q_max = float("nan")
                q_nonzero = 0
            row = {
                "time_unix": time.time(),
                "algo": "q_learning",
                "episodes": episode_num,
                "winner": int(last_info["winner"]),
                "steps": steps,
                "forfeits": forfeits,
                "epsilon": epsilon,
                "shaping_scale": args.shaping_scale,
                "states": len(q_table),
                "q_mean": q_mean,
                "q_std": q_std,
                "q_max": q_max,
                "q_nonzero": q_nonzero,
                "elapsed_seconds": time.time() - started_at,
            }
            append_csv_row(Path(args.log_csv), row)
            print(
                f"episodes={episode_num} winner={last_info['winner']} "
                f"steps={steps} states={len(q_table)} epsilon={epsilon:.3f} "
                f"q_mean={q_mean:.4f} q_max={q_max:.4f}"
            )

        if episode_num % args.save_interval == 0 or episode_num >= args.episodes:
            save_q_table(save_path, q_table, episode_num, args.seed)
            print(f"Saved Q-table at episode {episode_num} to {save_path}")

        if (
            args.snapshot_interval > 0
            and args.snapshot_dir
            and episode_num % args.snapshot_interval == 0
        ):
            snap_path = Path(args.snapshot_dir) / f"q_table_ep{episode_num:07d}.pkl"
            save_q_table(snap_path, q_table, episode_num, args.seed)
            print(f"Snapshot saved: {snap_path}")

        if args.stop_after_seconds > 0 and time.time() - started_at >= args.stop_after_seconds:
            save_q_table(save_path, q_table, episode_num, args.seed)
            if args.snapshot_dir:
                snap_path = Path(args.snapshot_dir) / f"q_table_ep{episode_num:07d}.pkl"
                save_q_table(snap_path, q_table, episode_num, args.seed)
            print(f"Stopping early; saved Q-table at episode {episode_num}.")
            return

    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_file.write_text(f"completed episodes={args.episodes} path={save_path}\n", encoding="utf-8")
    print(f"Q-learning completed: {save_path}")


if __name__ == "__main__":
    main()
