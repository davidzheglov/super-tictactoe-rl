"""Train a Super Tic-Tac-Toe agent with TensorFlow self-play.

The project includes a TF-Agents PyEnvironment in env.py. The trainer uses a
custom PPO-style update so legal action masks and symmetric self-play rewards
are handled directly.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
if (
    importlib.util.find_spec("tf_agents") is not None
    and importlib.util.find_spec("tf_keras") is not None
):
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf

try:
    with contextlib.redirect_stderr(io.StringIO()):
        from tf_agents.trajectories import trajectory as tf_agents_trajectory  # noqa: F401

    TF_AGENTS_IMPORT_OK = True
except Exception:
    TF_AGENTS_IMPORT_OK = False

try:
    from .env import SuperTicTacToeEnv, SuperTicTacToePyEnvironment
    from .models import (
        PolicyValueNet,
        action_log_probs,
        categorical_entropy,
        mask_logits,
        select_action,
    )
    from .utils import (
        checkpoint_exists,
        hidden_sizes_from_arg,
        project_root,
        resolve_tf_device,
        set_global_seeds,
    )
except ImportError:  # pragma: no cover
    from env import SuperTicTacToeEnv, SuperTicTacToePyEnvironment
    from models import (
        PolicyValueNet,
        action_log_probs,
        categorical_entropy,
        mask_logits,
        select_action,
    )
    from utils import (
        checkpoint_exists,
        hidden_sizes_from_arg,
        project_root,
        resolve_tf_device,
        set_global_seeds,
    )


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    log_prob: float
    reward: float
    value: float
    done: bool
    action_mask: np.ndarray
    player: int


def run_self_play_episode(
    env: SuperTicTacToeEnv,
    model: PolicyValueNet,
    device: str,
    gamma: float,
    rng: np.random.Generator,
) -> Tuple[List[Transition], Dict[str, int]]:
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    transitions: List[Transition] = []
    done = False
    winner = 0
    forfeits = 0
    illegal = 0

    while not done:
        player = env.current_player
        action_mask = env.legal_action_mask()
        action, log_prob, value = select_action(
            model, obs, action_mask, device=device, deterministic=False
        )
        next_obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        winner = int(info["winner"])
        forfeits += int(bool(info["forfeited"]))
        illegal += int(info["reason"] == "illegal_action")
        transitions.append(
            Transition(
                obs=obs,
                action=action,
                log_prob=log_prob,
                reward=0.0,
                value=value,
                done=done,
                action_mask=action_mask.astype(np.float32),
                player=player,
            )
        )
        obs = next_obs

    final_rewards = _outcome_rewards(transitions, winner, gamma)
    for transition, reward in zip(transitions, final_rewards):
        transition.reward = float(reward)

    stats = {
        "winner": winner,
        "length": len(transitions),
        "forfeits": forfeits,
        "illegal": illegal,
    }
    return transitions, stats


def _outcome_rewards(
    transitions: List[Transition], winner: int, gamma: float
) -> np.ndarray:
    if winner == 0:
        return np.zeros(len(transitions), dtype=np.float32)
    rewards = np.zeros(len(transitions), dtype=np.float32)
    last_index = len(transitions) - 1
    for index, transition in enumerate(transitions):
        outcome = 1.0 if transition.player == winner else -1.0
        rewards[index] = outcome * (gamma ** (last_index - index))
    return rewards


def batch_from_episodes(episodes: List[List[Transition]]) -> Dict[str, np.ndarray]:
    transitions = [transition for episode in episodes for transition in episode]
    rewards = np.asarray([transition.reward for transition in transitions], dtype=np.float32)
    values = np.asarray([transition.value for transition in transitions], dtype=np.float32)
    advantages = rewards - values
    if advantages.size > 1 and float(np.std(advantages)) > 1.0e-8:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1.0e-8)
    return {
        "obs": np.asarray([transition.obs for transition in transitions], dtype=np.float32),
        "actions": np.asarray([transition.action for transition in transitions], dtype=np.int32),
        "old_log_probs": np.asarray(
            [transition.log_prob for transition in transitions], dtype=np.float32
        ),
        "returns": rewards,
        "advantages": advantages.astype(np.float32),
        "action_masks": np.asarray(
            [transition.action_mask for transition in transitions], dtype=np.float32
        ),
    }


def ppo_update(
    model: PolicyValueNet,
    optimizer: tf.keras.optimizers.Optimizer,
    batch: Dict[str, np.ndarray],
    device: str,
    update_epochs: int,
    minibatch_size: int,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
) -> Dict[str, float]:
    n_items = batch["obs"].shape[0]
    indices = np.arange(n_items)
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0

    with tf.device(device):
        for _ in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_items, minibatch_size):
                minibatch = indices[start : start + minibatch_size]
                obs = tf.convert_to_tensor(batch["obs"][minibatch], dtype=tf.float32)
                actions = tf.convert_to_tensor(batch["actions"][minibatch], dtype=tf.int32)
                old_log_probs = tf.convert_to_tensor(
                    batch["old_log_probs"][minibatch], dtype=tf.float32
                )
                returns = tf.convert_to_tensor(batch["returns"][minibatch], dtype=tf.float32)
                advantages = tf.convert_to_tensor(
                    batch["advantages"][minibatch], dtype=tf.float32
                )
                action_masks = tf.convert_to_tensor(
                    batch["action_masks"][minibatch], dtype=tf.bool
                )

                with tf.GradientTape() as tape:
                    logits, values = model(obs, training=True)
                    values = tf.squeeze(values, axis=-1)
                    masked = mask_logits(logits, action_masks)
                    new_log_probs = action_log_probs(masked, actions)
                    ratio = tf.exp(new_log_probs - old_log_probs)
                    clipped_ratio = tf.clip_by_value(
                        ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
                    )
                    policy_loss = -tf.reduce_mean(
                        tf.minimum(ratio * advantages, clipped_ratio * advantages)
                    )
                    value_loss = tf.reduce_mean(tf.square(returns - values))
                    entropy = tf.reduce_mean(categorical_entropy(masked))
                    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                totals["loss"] += float(loss.numpy())
                totals["policy_loss"] += float(policy_loss.numpy())
                totals["value_loss"] += float(value_loss.numpy())
                totals["entropy"] += float(entropy.numpy())
                updates += 1

    return {key: value / max(updates, 1) for key, value in totals.items()}


def parse_args() -> argparse.Namespace:
    default_save_path = project_root() / "models" / "super_ttt_agent.pt"
    parser = argparse.ArgumentParser(description="Train a Super Tic-Tac-Toe agent.")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=str(default_save_path))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "gpu", "cuda", "mps"])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-csv", type=str, default="")
    parser.add_argument("--done-file", type=str, default="")
    parser.add_argument("--stop-after-seconds", type=float, default=0.0)
    parser.add_argument("--skip-if-done", action="store_true")
    return parser.parse_args()


def metadata_path(save_path: str) -> str:
    return save_path + ".json"


def done_path(args: argparse.Namespace) -> str:
    return args.done_file or args.save_path + ".done"


def read_saved_episodes(save_path: str) -> int:
    try:
        with open(metadata_path(save_path), "r", encoding="utf-8") as f:
            return int(json.load(f).get("episodes", 0))
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def write_checkpoint(
    checkpoint: tf.train.Checkpoint,
    episode_counter: tf.Variable,
    save_path: str,
    episodes: int,
    extra: Optional[Dict[str, object]] = None,
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
    if extra:
        metadata.update(extra)
    with open(metadata_path(save_path), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def append_csv_row(path: str, row: Dict[str, object]) -> None:
    if not path:
        return
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    if args.skip_if_done and Path(done_path(args)).exists():
        print(f"Done marker exists; skipping PPO training: {done_path(args)}")
        return
    saved_episodes = read_saved_episodes(args.save_path)
    if args.skip_if_done and saved_episodes >= args.episodes:
        print(
            f"Checkpoint already has {saved_episodes} episodes; requested {args.episodes}. "
            "Marking done and skipping."
        )
        Path(done_path(args)).parent.mkdir(parents=True, exist_ok=True)
        Path(done_path(args)).write_text("done\n", encoding="utf-8")
        return

    set_global_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_tf_device(args.device)
    started_at = time.time()

    hidden_sizes = hidden_sizes_from_arg(args.hidden_size)
    model = PolicyValueNet(hidden_sizes=hidden_sizes)
    model(tf.zeros((1, 97), dtype=tf.float32))
    optimizer_class = getattr(tf.keras.optimizers, "legacy", tf.keras.optimizers).Adam
    optimizer = optimizer_class(learning_rate=args.lr)
    episode_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name="episodes")
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, episodes=episode_counter
    )
    total_episodes = 0
    if args.resume and checkpoint_exists(args.save_path):
        checkpoint.restore(args.save_path).expect_partial()
        total_episodes = int(episode_counter.numpy())
        if total_episodes == 0:
            total_episodes = saved_episodes
            episode_counter.assign(total_episodes)
        print(f"Resumed PPO checkpoint {args.save_path} at episode {total_episodes}")

    if TF_AGENTS_IMPORT_OK:
        tf_env = SuperTicTacToePyEnvironment(seed=args.seed)
        print(
            "TF-Agents environment ready:",
            tf_env.observation_spec(),
            tf_env.action_spec(),
        )
    else:
        print("Warning: tf-agents is not installed; install requirements.txt for bonus support.")

    env = SuperTicTacToeEnv(seed=args.seed)
    while total_episodes < args.episodes:
        episodes_to_collect = min(args.batch_episodes, args.episodes - total_episodes)
        collected: List[List[Transition]] = []
        stats = []
        for _ in range(episodes_to_collect):
            episode, episode_stats = run_self_play_episode(
                env, model, device=device, gamma=args.gamma, rng=rng
            )
            collected.append(episode)
            stats.append(episode_stats)

        batch = batch_from_episodes(collected)
        losses = ppo_update(
            model=model,
            optimizer=optimizer,
            batch=batch,
            device=device,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            clip_ratio=args.clip_ratio,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
        )
        total_episodes += episodes_to_collect
        episode_counter.assign(total_episodes)

        if total_episodes % args.log_interval == 0 or total_episodes >= args.episodes:
            winners = [item["winner"] for item in stats]
            avg_length = float(np.mean([item["length"] for item in stats]))
            avg_forfeits = float(np.mean([item["forfeits"] for item in stats]))
            row = {
                "time_unix": time.time(),
                "algo": "ppo",
                "episodes": total_episodes,
                "batch_episodes": episodes_to_collect,
                "x_wins": winners.count(1),
                "o_wins": winners.count(-1),
                "draws": winners.count(0),
                "avg_length": avg_length,
                "avg_forfeits": avg_forfeits,
                "loss": losses["loss"],
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "elapsed_seconds": time.time() - started_at,
            }
            append_csv_row(args.log_csv, row)
            print(
                f"episodes={total_episodes} "
                f"x_wins={winners.count(1)} o_wins={winners.count(-1)} "
                f"draws={winners.count(0)} avg_len={avg_length:.1f} "
                f"avg_forfeits={avg_forfeits:.1f} loss={losses['loss']:.4f} "
                f"entropy={losses['entropy']:.4f}"
            )
        if (
            args.save_interval > 0
            and (total_episodes % args.save_interval == 0 or total_episodes >= args.episodes)
        ):
            write_checkpoint(
                checkpoint,
                episode_counter,
                args.save_path,
                total_episodes,
                extra={"algo": "ppo", "seed": args.seed},
            )
            print(f"Saved PPO checkpoint at episode {total_episodes} to {args.save_path}")
        if args.stop_after_seconds > 0 and time.time() - started_at >= args.stop_after_seconds:
            write_checkpoint(
                checkpoint,
                episode_counter,
                args.save_path,
                total_episodes,
                extra={"algo": "ppo", "seed": args.seed, "stopped_early": True},
            )
            print(
                f"Stopping early after {time.time() - started_at:.1f}s; "
                f"checkpoint saved at episode {total_episodes}."
            )
            return

    write_checkpoint(
        checkpoint,
        episode_counter,
        args.save_path,
        total_episodes,
        extra={"algo": "ppo", "seed": args.seed, "completed": True},
    )
    Path(done_path(args)).parent.mkdir(parents=True, exist_ok=True)
    Path(done_path(args)).write_text(
        f"completed episodes={total_episodes} path={args.save_path}\n",
        encoding="utf-8",
    )
    print(f"Saved final PPO checkpoint prefix to {args.save_path}")


if __name__ == "__main__":
    main()
