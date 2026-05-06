"""Resumable PyTorch PPO self-play trainer for Super Tic-Tac-Toe."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

try:
    from .agents import HeuristicAgent, RandomAgent
    from .env import SuperTicTacToeEnv
    from .torch_models import TorchPolicyValueNet, mask_logits, resolve_torch_device, select_action_torch
    from .utils import hidden_sizes_from_arg, project_root, set_global_seeds
except ImportError:  # pragma: no cover
    from agents import HeuristicAgent, RandomAgent
    from env import SuperTicTacToeEnv
    from torch_models import TorchPolicyValueNet, mask_logits, resolve_torch_device, select_action_torch
    from utils import hidden_sizes_from_arg, project_root, set_global_seeds


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    log_prob: float
    reward: float
    value: float
    action_mask: np.ndarray
    player: int


def outcome_rewards(transitions: List[Transition], winner: int, gamma: float) -> np.ndarray:
    if winner == 0:
        return np.zeros(len(transitions), dtype=np.float32)
    rewards = np.zeros(len(transitions), dtype=np.float32)
    last_index = len(transitions) - 1
    for index, transition in enumerate(transitions):
        outcome = 1.0 if transition.player == winner else -1.0
        rewards[index] = outcome * (gamma ** (last_index - index))
    return rewards


def run_episode(
    env: SuperTicTacToeEnv,
    model: TorchPolicyValueNet,
    device: torch.device,
    gamma: float,
    rng: np.random.Generator,
    opponent: str = "self",
    agent_player: int = 1,
    random_agent: Optional[RandomAgent] = None,
    heuristic_agent: Optional[HeuristicAgent] = None,
) -> Tuple[List[Transition], Dict[str, int]]:
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    transitions: List[Transition] = []
    done = False
    winner = 0
    forfeits = 0
    illegal = 0
    steps = 0

    while not done:
        player = env.current_player
        policy_turn = opponent == "self" or player == agent_player
        action_mask = env.legal_action_mask()
        if policy_turn:
            action, log_prob, value = select_action_torch(
                model, obs, action_mask, device=device, deterministic=False
            )
        elif opponent == "heuristic":
            action = (heuristic_agent or HeuristicAgent()).select_action(env)
            log_prob = 0.0
            value = 0.0
        elif opponent == "random":
            action = (random_agent or RandomAgent()).select_action(env)
            log_prob = 0.0
            value = 0.0
        else:
            raise ValueError(f"Unknown opponent mode: {opponent}")
        next_obs, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        winner = int(info["winner"])
        forfeits += int(bool(info["forfeited"]))
        illegal += int(info["reason"] == "illegal_action")
        steps += 1
        if policy_turn:
            transitions.append(
                Transition(
                    obs=obs,
                    action=action,
                    log_prob=log_prob,
                    reward=0.0,
                    value=value,
                    action_mask=action_mask.astype(np.float32),
                    player=player,
                )
            )
        obs = next_obs

    rewards = outcome_rewards(transitions, winner, gamma)
    for transition, reward in zip(transitions, rewards):
        transition.reward = float(reward)

    return transitions, {
        "winner": winner,
        "length": steps,
        "policy_steps": len(transitions),
        "forfeits": forfeits,
        "illegal": illegal,
        "opponent_self": int(opponent == "self"),
        "opponent_heuristic": int(opponent == "heuristic"),
        "opponent_random": int(opponent == "random"),
    }


def choose_agent_player(mode: str, episode_index: int, rng: np.random.Generator) -> int:
    if mode == "x":
        return 1
    if mode == "o":
        return -1
    if mode == "random":
        return int(rng.choice([1, -1]))
    return 1 if episode_index % 2 == 0 else -1


def choose_opponent(args: argparse.Namespace, rng: np.random.Generator) -> str:
    if args.opponent != "mixed":
        return args.opponent
    labels = np.asarray(["self", "heuristic", "random"], dtype=object)
    probs = np.asarray(
        [args.mixed_self_prob, args.mixed_heuristic_prob, args.mixed_random_prob],
        dtype=np.float64,
    )
    probs = probs / max(float(np.sum(probs)), 1.0e-12)
    return str(rng.choice(labels, p=probs))


def batch_from_episodes(episodes: List[List[Transition]]) -> Dict[str, np.ndarray]:
    transitions = [transition for episode in episodes for transition in episode]
    rewards = np.asarray([transition.reward for transition in transitions], dtype=np.float32)
    values = np.asarray([transition.value for transition in transitions], dtype=np.float32)
    advantages = rewards - values
    if advantages.size > 1 and float(np.std(advantages)) > 1.0e-8:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1.0e-8)
    return {
        "obs": np.asarray([transition.obs for transition in transitions], dtype=np.float32),
        "actions": np.asarray([transition.action for transition in transitions], dtype=np.int64),
        "old_log_probs": np.asarray([transition.log_prob for transition in transitions], dtype=np.float32),
        "returns": rewards,
        "advantages": advantages.astype(np.float32),
        "action_masks": np.asarray([transition.action_mask for transition in transitions], dtype=np.bool_),
    }


def ppo_update(
    model: TorchPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, np.ndarray],
    device: torch.device,
    update_epochs: int,
    minibatch_size: int,
    clip_ratio: float,
    value_coef: float,
    entropy_coef: float,
) -> Dict[str, float]:
    n_items = int(batch["obs"].shape[0])
    indices = np.arange(n_items)
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0
    model.train()

    for _ in range(update_epochs):
        np.random.shuffle(indices)
        for start in range(0, n_items, minibatch_size):
            mb = indices[start : start + minibatch_size]
            obs = torch.as_tensor(batch["obs"][mb], dtype=torch.float32, device=device)
            actions = torch.as_tensor(batch["actions"][mb], dtype=torch.long, device=device)
            old_log_probs = torch.as_tensor(batch["old_log_probs"][mb], dtype=torch.float32, device=device)
            returns = torch.as_tensor(batch["returns"][mb], dtype=torch.float32, device=device)
            advantages = torch.as_tensor(batch["advantages"][mb], dtype=torch.float32, device=device)
            masks = torch.as_tensor(batch["action_masks"][mb], dtype=torch.bool, device=device)

            logits, values = model(obs)
            masked = mask_logits(logits, masks)
            dist = Categorical(logits=masked)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            value_loss = torch.mean((returns - values) ** 2)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            totals["loss"] += float(loss.item())
            totals["policy_loss"] += float(policy_loss.item())
            totals["value_loss"] += float(value_loss.item())
            totals["entropy"] += float(entropy.item())
            updates += 1

    return {key: value / max(updates, 1) for key, value in totals.items()}


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


def save_checkpoint(
    path: str,
    model: TorchPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    episodes: int,
    args: argparse.Namespace,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    algo_label = os.environ.get("SUPER_TTT_ALGO_LABEL", "torch_ppo")
    framework = os.environ.get("SUPER_TTT_FRAMEWORK", "PyTorch")
    payload = {
        "algo": algo_label,
        "framework": framework,
        "episodes": int(episodes),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "hidden_size": int(args.hidden_size),
        "seed": int(args.seed),
        "updated_at_unix": time.time(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, save_path)
    with (save_path.with_suffix(save_path.suffix + ".json")).open("w", encoding="utf-8") as f:
        json.dump(
            {
                "algo": algo_label,
                "framework": framework,
                "episodes": int(episodes),
                "checkpoint": str(save_path),
                "hidden_size": int(args.hidden_size),
                "seed": int(args.seed),
                **(extra or {}),
            },
            f,
            indent=2,
        )


def load_checkpoint(path: str, model: TorchPolicyValueNet, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    save_path = Path(path)
    if not save_path.exists():
        return 0
    payload = torch.load(save_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload.get("episodes", 0))


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train PyTorch PPO self-play agent.")
    parser.add_argument("--episodes", type=int, default=300000)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=str(root / "models" / "super_ttt_agent_torch.pt"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu", "mps"])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-episodes", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--opponent",
        type=str,
        default="self",
        choices=["self", "random", "heuristic", "mixed"],
    )
    parser.add_argument(
        "--agent-player-mode",
        type=str,
        default="alternate",
        choices=["alternate", "random", "x", "o"],
    )
    parser.add_argument("--mixed-self-prob", type=float, default=0.5)
    parser.add_argument("--mixed-heuristic-prob", type=float, default=0.4)
    parser.add_argument("--mixed-random-prob", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--log-csv", type=str, default="")
    parser.add_argument("--done-file", type=str, default="")
    parser.add_argument("--stop-after-seconds", type=float, default=0.0)
    parser.add_argument("--skip-if-done", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    done_file = Path(args.done_file or args.save_path + ".done")
    if args.skip_if_done and done_file.exists():
        print(f"Done marker exists; skipping PyTorch PPO: {done_file}")
        return

    set_global_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_torch_device(args.device)
    print(f"PyTorch PPO device: {device}")

    model = TorchPolicyValueNet(hidden_sizes=hidden_sizes_from_arg(args.hidden_size)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_episodes = 0
    if args.resume:
        total_episodes = load_checkpoint(args.save_path, model, optimizer, device)
        if total_episodes:
            print(f"Resumed PyTorch PPO checkpoint {args.save_path} at episode {total_episodes}")
    if args.skip_if_done and total_episodes >= args.episodes:
        done_file.parent.mkdir(parents=True, exist_ok=True)
        done_file.write_text("done\n", encoding="utf-8")
        print(f"PyTorch PPO already has {total_episodes} episodes; skipping.")
        return

    env = SuperTicTacToeEnv(seed=args.seed)
    random_agent = RandomAgent(seed=args.seed + 17)
    heuristic_agent = HeuristicAgent(seed=args.seed + 29)
    started_at = time.time()
    while total_episodes < args.episodes:
        episodes_to_collect = min(args.batch_episodes, args.episodes - total_episodes)
        collected: List[List[Transition]] = []
        stats = []
        for local_episode in range(episodes_to_collect):
            episode_index = total_episodes + local_episode
            opponent = choose_opponent(args, rng)
            agent_player = choose_agent_player(args.agent_player_mode, episode_index, rng)
            episode, episode_stats = run_episode(
                env,
                model,
                device,
                args.gamma,
                rng,
                opponent=opponent,
                agent_player=agent_player,
                random_agent=random_agent,
                heuristic_agent=heuristic_agent,
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

        if total_episodes % args.log_interval == 0 or total_episodes >= args.episodes:
            winners = [item["winner"] for item in stats]
            row = {
                "time_unix": time.time(),
                "algo": "torch_ppo",
                "episodes": total_episodes,
                "batch_episodes": episodes_to_collect,
                "x_wins": winners.count(1),
                "o_wins": winners.count(-1),
                "draws": winners.count(0),
                "avg_length": float(np.mean([item["length"] for item in stats])),
                "avg_policy_steps": float(
                    np.mean([item["policy_steps"] for item in stats])
                ),
                "avg_forfeits": float(np.mean([item["forfeits"] for item in stats])),
                "self_games": int(sum(item["opponent_self"] for item in stats)),
                "heuristic_games": int(sum(item["opponent_heuristic"] for item in stats)),
                "random_games": int(sum(item["opponent_random"] for item in stats)),
                "loss": losses["loss"],
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "device": str(device),
                "elapsed_seconds": time.time() - started_at,
            }
            append_csv_row(args.log_csv, row)
            print(
                f"episodes={total_episodes} device={device} "
                f"x_wins={winners.count(1)} o_wins={winners.count(-1)} "
                f"draws={winners.count(0)} avg_len={row['avg_length']:.1f} "
                f"policy_steps={row['avg_policy_steps']:.1f} "
                f"opp_self={row['self_games']} opp_heur={row['heuristic_games']} "
                f"opp_rand={row['random_games']} "
                f"loss={losses['loss']:.4f} entropy={losses['entropy']:.4f}"
            )

        if args.save_interval > 0 and (
            total_episodes % args.save_interval == 0 or total_episodes >= args.episodes
        ):
            save_checkpoint(args.save_path, model, optimizer, total_episodes, args)
            print(f"Saved PyTorch PPO checkpoint at episode {total_episodes} to {args.save_path}")

        if args.stop_after_seconds > 0 and time.time() - started_at >= args.stop_after_seconds:
            save_checkpoint(
                args.save_path,
                model,
                optimizer,
                total_episodes,
                args,
                extra={"stopped_early": True},
            )
            print(f"Stopping early; saved PyTorch PPO at episode {total_episodes}.")
            return

    save_checkpoint(args.save_path, model, optimizer, total_episodes, args, extra={"completed": True})
    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_file.write_text(f"completed episodes={total_episodes} path={args.save_path}\n", encoding="utf-8")
    print(f"PyTorch PPO completed: {args.save_path}")


if __name__ == "__main__":
    main()
