"""Resumable DQN-style baseline for Super Tic-Tac-Toe."""

from __future__ import annotations

import argparse
import csv
import json
import os
import importlib.util
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
if (
    importlib.util.find_spec("tf_agents") is not None
    and importlib.util.find_spec("tf_keras") is not None
):
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

try:
    from .env import SuperTicTacToeEnv
    from .utils import checkpoint_exists, project_root, random_legal_action, resolve_tf_device, set_global_seeds
except ImportError:  # pragma: no cover
    from env import SuperTicTacToeEnv
    from utils import checkpoint_exists, project_root, random_legal_action, resolve_tf_device, set_global_seeds


class DQN(tf.keras.Model):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.layers_ = [
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(hidden_size, activation="relu"),
            tf.keras.layers.Dense(96),
        ]

    def call(self, inputs, training: bool = False):
        x = tf.cast(inputs, tf.float32)
        for layer in self.layers_:
            x = layer(x, training=training)
        return x


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.data = deque(maxlen=self.capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, item) -> None:
        self.data.append(item)

    def sample(self, batch_size: int):
        idx = self.rng.choice(len(self.data), size=batch_size, replace=False)
        return [self.data[int(i)] for i in idx]

    def __len__(self) -> int:
        return len(self.data)


def masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    q = np.array(q_values, copy=True)
    q[~mask] = -1.0e9
    return int(np.argmax(q))


def append_csv_row(path: str, row: Dict[str, object]) -> None:
    if not path:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_checkpoint(
    checkpoint: tf.train.Checkpoint,
    episode_counter: tf.Variable,
    save_path: str,
    episodes: int,
    extra: Dict[str, object],
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    episode_counter.assign(int(episodes))
    checkpoint.write(save_path)
    metadata = {
        "episodes": int(episodes),
        "format": "tf.train.Checkpoint prefix",
        "checkpoint_prefix": save_path,
        "updated_at_unix": time.time(),
    }
    metadata.update(extra)
    with open(save_path + ".json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def read_saved_episodes(save_path: str) -> int:
    try:
        with open(save_path + ".json", "r", encoding="utf-8") as f:
            return int(json.load(f).get("episodes", 0))
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train DQN baseline.")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu", "cuda", "mps"])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--warmup-steps", type=int, default=2048)
    parser.add_argument("--target-update-episodes", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-frac", type=float, default=0.7)
    parser.add_argument("--save-path", type=str, default=str(root / "models" / "dqn_agent.pt"))
    parser.add_argument("--log-csv", type=str, default=str(root / "models" / "dqn_log.csv"))
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument("--done-file", type=str, default="")
    parser.add_argument("--stop-after-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    done_file = Path(args.done_file or args.save_path + ".done")
    if args.skip_if_done and done_file.exists():
        print(f"Done marker exists; skipping DQN: {done_file}")
        return

    set_global_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_tf_device(args.device)
    online = DQN(hidden_size=args.hidden_size)
    target = DQN(hidden_size=args.hidden_size)
    online(tf.zeros((1, 97), dtype=tf.float32))
    target(tf.zeros((1, 97), dtype=tf.float32))
    target.set_weights(online.get_weights())
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    episode_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name="episodes")
    checkpoint = tf.train.Checkpoint(
        model=online, target_model=target, optimizer=optimizer, episodes=episode_counter
    )

    start_episode = 0
    if args.resume and checkpoint_exists(args.save_path):
        checkpoint.restore(args.save_path).expect_partial()
        start_episode = int(episode_counter.numpy()) or read_saved_episodes(args.save_path)
        episode_counter.assign(start_episode)
        print(f"Resumed DQN checkpoint {args.save_path} at episode {start_episode}")
    if args.skip_if_done and start_episode >= args.episodes:
        done_file.parent.mkdir(parents=True, exist_ok=True)
        done_file.write_text("done\n", encoding="utf-8")
        print(f"DQN already has {start_episode} episodes; skipping.")
        return

    replay = ReplayBuffer(args.replay_size, args.seed)
    env = SuperTicTacToeEnv(seed=args.seed)
    started_at = time.time()

    with tf.device(device):
        for episode in range(start_episode, args.episodes):
            frac = min(1.0, episode / max(args.episodes * args.eps_decay_frac, 1.0))
            epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done = False
            steps = 0
            forfeits = 0
            last_info = {"winner": 0}
            losses: List[float] = []

            while not done:
                mask = env.legal_action_mask()
                if rng.random() < epsilon:
                    action = random_legal_action(mask, rng)
                else:
                    q = online(obs[None, :], training=False).numpy()[0]
                    action = masked_argmax(q, mask)

                next_obs, reward, terminated, truncated, last_info = env.step(action)
                done = bool(terminated or truncated)
                next_mask = env.legal_action_mask()
                replay.add((obs, action, float(reward), next_obs, done, next_mask))
                obs = next_obs
                forfeits += int(bool(last_info.get("forfeited", False)))
                steps += 1

                if len(replay) >= max(args.warmup_steps, args.batch_size):
                    batch = replay.sample(args.batch_size)
                    b_obs, b_action, b_reward, b_next_obs, b_done, b_next_mask = map(
                        np.asarray, zip(*batch)
                    )
                    next_q = target(b_next_obs, training=False).numpy()
                    next_q[~b_next_mask.astype(bool)] = -1.0e9
                    y = b_reward.astype(np.float32) + (
                        1.0 - b_done.astype(np.float32)
                    ) * (-args.gamma * np.max(next_q, axis=1).astype(np.float32))
                    with tf.GradientTape() as tape:
                        q_all = online(b_obs, training=True)
                        q_action = tf.gather(
                            q_all, b_action.astype(np.int32), batch_dims=1
                        )
                        loss = tf.reduce_mean(tf.square(tf.convert_to_tensor(y) - q_action))
                    grads = tape.gradient(loss, online.trainable_variables)
                    optimizer.apply_gradients(zip(grads, online.trainable_variables))
                    losses.append(float(loss.numpy()))

            episode_num = episode + 1
            episode_counter.assign(episode_num)
            if episode_num % args.target_update_episodes == 0:
                target.set_weights(online.get_weights())
            if episode_num % args.log_interval == 0 or episode_num >= args.episodes:
                row = {
                    "time_unix": time.time(),
                    "algo": "dqn",
                    "episodes": episode_num,
                    "winner": int(last_info["winner"]),
                    "steps": steps,
                    "forfeits": forfeits,
                    "epsilon": epsilon,
                    "replay_size": len(replay),
                    "loss": float(np.mean(losses)) if losses else np.nan,
                    "elapsed_seconds": time.time() - started_at,
                }
                append_csv_row(args.log_csv, row)
                print(
                    f"episodes={episode_num} winner={last_info['winner']} "
                    f"steps={steps} replay={len(replay)} epsilon={epsilon:.3f} "
                    f"loss={row['loss']:.4f}"
                )
            if episode_num % args.save_interval == 0 or episode_num >= args.episodes:
                write_checkpoint(
                    checkpoint,
                    episode_counter,
                    args.save_path,
                    episode_num,
                    {"algo": "dqn", "seed": args.seed},
                )
                print(f"Saved DQN checkpoint at episode {episode_num} to {args.save_path}")
            if args.stop_after_seconds > 0 and time.time() - started_at >= args.stop_after_seconds:
                write_checkpoint(
                    checkpoint,
                    episode_counter,
                    args.save_path,
                    episode_num,
                    {"algo": "dqn", "seed": args.seed, "stopped_early": True},
                )
                print(f"Stopping early; saved DQN at episode {episode_num}.")
                return

    write_checkpoint(
        checkpoint,
        episode_counter,
        args.save_path,
        args.episodes,
        {"algo": "dqn", "seed": args.seed, "completed": True},
    )
    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_file.write_text(f"completed episodes={args.episodes} path={args.save_path}\n", encoding="utf-8")
    print(f"DQN completed: {args.save_path}")


if __name__ == "__main__":
    main()
