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
    from .agents import BasicHeuristicAgent, HeuristicAgent, LineBuilderAgent, RandomAgent, board_potential
    from .env import SuperTicTacToeEnv
    from .torch_models import TorchPolicyValueNet, mask_logits, resolve_torch_device, select_action_torch
    from .utils import hidden_sizes_from_arg, project_root, set_global_seeds
except ImportError:  # pragma: no cover
    from agents import BasicHeuristicAgent, HeuristicAgent, LineBuilderAgent, RandomAgent, board_potential
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
    line_agent: Optional[LineBuilderAgent] = None,
    basic_agent: Optional[BasicHeuristicAgent] = None,
    shaping_scale: float = 0.03,
    shaping_clip: float = 2.0,
    shaping_defense_weight: float = 0.75,
    forfeit_penalty: float = 0.02,
    start_state_mode: str = "none",
    start_state_min_plies: int = 4,
    start_state_max_plies: int = 18,
) -> Tuple[List[Transition], Dict[str, int]]:
    obs = reset_with_start_state(
        env,
        rng,
        start_state_mode,
        start_state_min_plies,
        start_state_max_plies,
        random_agent or RandomAgent(),
        heuristic_agent or HeuristicAgent(),
        line_agent or LineBuilderAgent(),
        basic_agent or BasicHeuristicAgent(),
    )
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
        potential_before = (
            board_potential(env.board, player, defense_weight=shaping_defense_weight)
            if policy_turn and shaping_scale != 0.0
            else 0.0
        )
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
        elif opponent == "line":
            action = (line_agent or LineBuilderAgent()).select_action(env)
            log_prob = 0.0
            value = 0.0
        elif opponent == "basic":
            action = (basic_agent or BasicHeuristicAgent()).select_action(env)
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
            dense_reward = 0.0
            if shaping_scale != 0.0:
                potential_after = board_potential(
                    env.board,
                    player,
                    defense_weight=shaping_defense_weight,
                )
                delta = float(np.clip(potential_after - potential_before, -shaping_clip, shaping_clip))
                dense_reward += shaping_scale * delta
            if bool(info.get("forfeited", False)):
                dense_reward -= forfeit_penalty
            transitions.append(
                Transition(
                    obs=obs,
                    action=action,
                    log_prob=log_prob,
                    reward=dense_reward,
                    value=value,
                    action_mask=action_mask.astype(np.float32),
                    player=player,
                )
            )
        obs = next_obs

    rewards = outcome_rewards(transitions, winner, gamma)
    for transition, reward in zip(transitions, rewards):
        transition.reward += float(reward)

    return transitions, {
        "winner": winner,
        "length": steps,
        "policy_steps": len(transitions),
        "forfeits": forfeits,
        "illegal": illegal,
        "opponent_self": int(opponent == "self"),
        "opponent_heuristic": int(opponent == "heuristic"),
        "opponent_line": int(opponent == "line"),
        "opponent_basic": int(opponent == "basic"),
        "opponent_random": int(opponent == "random"),
    }


def collect_episodes_vectorized(
    num_episodes: int,
    model: TorchPolicyValueNet,
    device: torch.device,
    gamma: float,
    rng: np.random.Generator,
    placement_mode: str,
    episode_start_index: int,
    agent_player_mode: str,
    args: argparse.Namespace,
    random_agent: RandomAgent,
    heuristic_agent: HeuristicAgent,
    line_agent: LineBuilderAgent,
    basic_agent: BasicHeuristicAgent,
) -> Tuple[List[List[Transition]], List[Dict[str, int]]]:
    """Collect PPO rollout games with batched policy inference.

    The environment and scripted opponents are still Python objects, but policy
    turns across active games share a single neural forward pass per timestep.
    This is the main speedup over collecting one full game at a time.
    """

    envs = [
        SuperTicTacToeEnv(
            seed=int(rng.integers(0, 2**31 - 1)),
            placement_mode=placement_mode,
        )
        for _ in range(num_episodes)
    ]
    observations = [
        reset_with_start_state(
            env,
            rng,
            args.start_state_mode,
            args.start_state_min_plies,
            args.start_state_max_plies,
            random_agent,
            heuristic_agent,
            line_agent,
            basic_agent,
        )
        for env in envs
    ]
    opponents = [choose_opponent(args, rng) for _ in range(num_episodes)]
    agent_players = [
        choose_agent_player(agent_player_mode, episode_start_index + index, rng)
        for index in range(num_episodes)
    ]
    collected: List[List[Transition]] = [[] for _ in range(num_episodes)]
    stats: List[Dict[str, int]] = [
        {
            "winner": 0,
            "length": 0,
            "policy_steps": 0,
            "forfeits": 0,
            "illegal": 0,
            "opponent_self": int(opponents[index] == "self"),
            "opponent_heuristic": int(opponents[index] == "heuristic"),
            "opponent_line": int(opponents[index] == "line"),
            "opponent_basic": int(opponents[index] == "basic"),
            "opponent_random": int(opponents[index] == "random"),
        }
        for index in range(num_episodes)
    ]
    done_flags = [False for _ in range(num_episodes)]
    winners = [0 for _ in range(num_episodes)]
    model.eval()

    while not all(done_flags):
        active_indices = [index for index, done in enumerate(done_flags) if not done]
        step_data: Dict[int, Dict[str, object]] = {}
        policy_indices: List[int] = []

        for index in active_indices:
            env = envs[index]
            player = int(env.current_player)
            opponent = opponents[index]
            policy_turn = opponent == "self" or player == agent_players[index]
            action_mask = env.legal_action_mask()
            potential_before = (
                board_potential(env.board, player, defense_weight=args.shaping_defense_weight)
                if policy_turn and args.shaping_scale != 0.0
                else 0.0
            )
            step_data[index] = {
                "player": player,
                "policy_turn": policy_turn,
                "action_mask": action_mask,
                "potential_before": potential_before,
            }
            if policy_turn:
                policy_indices.append(index)

        if policy_indices:
            obs_batch = torch.as_tensor(
                np.asarray([observations[index] for index in policy_indices], dtype=np.float32),
                dtype=torch.float32,
                device=device,
            )
            mask_batch = torch.as_tensor(
                np.asarray([step_data[index]["action_mask"] for index in policy_indices], dtype=np.bool_),
                dtype=torch.bool,
                device=device,
            )
            with torch.no_grad():
                logits, values = model(obs_batch)
                masked = mask_logits(logits, mask_batch)
                dist = Categorical(logits=masked)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            for batch_index, env_index in enumerate(policy_indices):
                step_data[env_index]["action"] = int(actions[batch_index].item())
                step_data[env_index]["log_prob"] = float(log_probs[batch_index].item())
                step_data[env_index]["value"] = float(values[batch_index].item())

        for index in active_indices:
            data = step_data[index]
            if bool(data["policy_turn"]):
                continue
            env = envs[index]
            opponent = opponents[index]
            if opponent == "heuristic":
                action = heuristic_agent.select_action(env)
            elif opponent == "random":
                action = random_agent.select_action(env)
            elif opponent == "line":
                action = line_agent.select_action(env)
            elif opponent == "basic":
                action = basic_agent.select_action(env)
            else:
                raise ValueError(f"Unknown opponent mode: {opponent}")
            data["action"] = int(action)
            data["log_prob"] = 0.0
            data["value"] = 0.0

        for index in active_indices:
            env = envs[index]
            data = step_data[index]
            action = int(data["action"])
            next_obs, _, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            winner = int(info["winner"])
            stats[index]["winner"] = winner
            stats[index]["length"] += 1
            stats[index]["forfeits"] += int(bool(info["forfeited"]))
            stats[index]["illegal"] += int(info["reason"] == "illegal_action")

            if bool(data["policy_turn"]):
                player = int(data["player"])
                dense_reward = 0.0
                if args.shaping_scale != 0.0:
                    potential_after = board_potential(
                        env.board,
                        player,
                        defense_weight=args.shaping_defense_weight,
                    )
                    delta = float(
                        np.clip(
                            potential_after - float(data["potential_before"]),
                            -args.shaping_clip,
                            args.shaping_clip,
                        )
                    )
                    dense_reward += args.shaping_scale * delta
                if bool(info.get("forfeited", False)):
                    dense_reward -= args.forfeit_penalty
                collected[index].append(
                    Transition(
                        obs=observations[index],
                        action=action,
                        log_prob=float(data["log_prob"]),
                        reward=dense_reward,
                        value=float(data["value"]),
                        action_mask=np.asarray(data["action_mask"], dtype=np.float32),
                        player=player,
                    )
                )
                stats[index]["policy_steps"] += 1

            observations[index] = next_obs
            if done:
                done_flags[index] = True
                winners[index] = winner

    for episode, winner in zip(collected, winners):
        rewards = outcome_rewards(episode, winner, gamma)
        for transition, reward in zip(episode, rewards):
            transition.reward += float(reward)

    return collected, stats


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
    labels = np.asarray(["self", "heuristic", "line", "basic", "random"], dtype=object)
    probs = np.asarray(
        [
            args.mixed_self_prob,
            args.mixed_heuristic_prob,
            args.mixed_line_prob,
            args.mixed_basic_prob,
            args.mixed_random_prob,
        ],
        dtype=np.float64,
    )
    probs = probs / max(float(np.sum(probs)), 1.0e-12)
    return str(rng.choice(labels, p=probs))


def choose_start_actor(mode: str, rng: np.random.Generator) -> str:
    if mode != "mixed":
        return mode
    labels = np.asarray(["heuristic", "line", "random"], dtype=object)
    probs = np.asarray([0.55, 0.30, 0.15], dtype=np.float64)
    return str(rng.choice(labels, p=probs))


def select_scripted_action(
    env: SuperTicTacToeEnv,
    actor: str,
    rng: np.random.Generator,
    random_agent: RandomAgent,
    heuristic_agent: HeuristicAgent,
    line_agent: LineBuilderAgent,
    basic_agent: BasicHeuristicAgent,
) -> int:
    if actor == "heuristic":
        return heuristic_agent.select_action(env)
    if actor == "line":
        return line_agent.select_action(env)
    if actor == "basic":
        return basic_agent.select_action(env)
    return random_agent.select_action(env)


def reset_with_start_state(
    env: SuperTicTacToeEnv,
    rng: np.random.Generator,
    start_state_mode: str,
    min_plies: int,
    max_plies: int,
    random_agent: RandomAgent,
    heuristic_agent: HeuristicAgent,
    line_agent: LineBuilderAgent,
    basic_agent: BasicHeuristicAgent,
) -> np.ndarray:
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    if start_state_mode == "none" or max_plies <= 0:
        return obs

    low = max(0, int(min_plies))
    high = max(low, int(max_plies))
    plies = int(rng.integers(low, high + 1)) if high > 0 else 0
    for _ in range(plies):
        actor = choose_start_actor(start_state_mode, rng)
        action = select_scripted_action(
            env,
            actor,
            rng,
            random_agent,
            heuristic_agent,
            line_agent,
            basic_agent,
        )
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            break
    return obs


def batch_from_episodes(episodes: List[List[Transition]]) -> Dict[str, np.ndarray]:
    transitions = [transition for episode in episodes for transition in episode]
    if not transitions:
        raise RuntimeError("PPO rollout produced no policy transitions.")
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


def load_model_weights(path: str, model: TorchPolicyValueNet, device: torch.device) -> None:
    if not path:
        return
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Initial checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    print(f"Loaded initial policy weights from {checkpoint}")


def save_numbered_checkpoint(
    checkpoint_dir: str,
    model: TorchPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    episodes: int,
    args: argparse.Namespace,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    if not checkpoint_dir:
        return
    path = Path(checkpoint_dir) / f"ppo_ep{int(episodes):07d}.pt"
    save_checkpoint(str(path), model, optimizer, episodes, args, extra=extra)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train PyTorch PPO self-play agent.")
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=str(root / "models" / "super_ttt_agent_torch.pt"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu", "mps"])
    parser.add_argument(
        "--placement-mode",
        type=str,
        default="stochastic",
        choices=["stochastic", "deterministic"],
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-episodes", type=int, default=64)
    parser.add_argument(
        "--rollout-mode",
        type=str,
        default="vectorized",
        choices=["vectorized", "sequential"],
        help="Collect PPO update games concurrently or one game at a time.",
    )
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument(
        "--opponent",
        type=str,
        default="self",
        choices=["self", "random", "heuristic", "line", "basic", "mixed"],
    )
    parser.add_argument(
        "--agent-player-mode",
        type=str,
        default="alternate",
        choices=["alternate", "random", "x", "o"],
    )
    parser.add_argument("--mixed-self-prob", type=float, default=0.2)
    parser.add_argument("--mixed-heuristic-prob", type=float, default=0.45)
    parser.add_argument("--mixed-line-prob", type=float, default=0.3)
    parser.add_argument("--mixed-basic-prob", type=float, default=0.0)
    parser.add_argument("--mixed-random-prob", type=float, default=0.05)
    parser.add_argument("--shaping-scale", type=float, default=0.03)
    parser.add_argument("--shaping-clip", type=float, default=2.0)
    parser.add_argument("--shaping-defense-weight", type=float, default=0.75)
    parser.add_argument("--forfeit-penalty", type=float, default=0.02)
    parser.add_argument(
        "--start-state-mode",
        type=str,
        default="none",
        choices=["none", "random", "heuristic", "line", "basic", "mixed"],
        help="Optionally begin training episodes from scripted mid-game states.",
    )
    parser.add_argument("--start-state-min-plies", type=int, default=4)
    parser.add_argument("--start-state-max-plies", type=int, default=18)
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--checkpoint-dir", type=str, default="")
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
    if args.init_checkpoint and total_episodes == 0:
        load_model_weights(args.init_checkpoint, model, device)
    if args.skip_if_done and total_episodes >= args.episodes:
        done_file.parent.mkdir(parents=True, exist_ok=True)
        done_file.write_text("done\n", encoding="utf-8")
        print(f"PyTorch PPO already has {total_episodes} episodes; skipping.")
        return

    env = SuperTicTacToeEnv(seed=args.seed, placement_mode=args.placement_mode)
    random_agent = RandomAgent(seed=args.seed + 17)
    heuristic_agent = HeuristicAgent(seed=args.seed + 29)
    line_agent = LineBuilderAgent(seed=args.seed + 31)
    basic_agent = BasicHeuristicAgent(seed=args.seed + 37)
    started_at = time.time()
    while total_episodes < args.episodes:
        episodes_to_collect = min(args.batch_episodes, args.episodes - total_episodes)
        if args.rollout_mode == "vectorized":
            collected, stats = collect_episodes_vectorized(
                num_episodes=episodes_to_collect,
                model=model,
                device=device,
                gamma=args.gamma,
                rng=rng,
                placement_mode=args.placement_mode,
                episode_start_index=total_episodes,
                agent_player_mode=args.agent_player_mode,
                args=args,
                random_agent=random_agent,
                heuristic_agent=heuristic_agent,
                line_agent=line_agent,
                basic_agent=basic_agent,
            )
        else:
            collected = []
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
                    line_agent=line_agent,
                    basic_agent=basic_agent,
                    shaping_scale=args.shaping_scale,
                    shaping_clip=args.shaping_clip,
                    shaping_defense_weight=args.shaping_defense_weight,
                    forfeit_penalty=args.forfeit_penalty,
                    start_state_mode=args.start_state_mode,
                    start_state_min_plies=args.start_state_min_plies,
                    start_state_max_plies=args.start_state_max_plies,
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
                "line_games": int(sum(item["opponent_line"] for item in stats)),
                "basic_games": int(sum(item["opponent_basic"] for item in stats)),
                "random_games": int(sum(item["opponent_random"] for item in stats)),
                "loss": losses["loss"],
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "entropy": losses["entropy"],
                "device": str(device),
                "placement_mode": args.placement_mode,
                "rollout_mode": args.rollout_mode,
                "start_state_mode": args.start_state_mode,
                "elapsed_seconds": time.time() - started_at,
            }
            append_csv_row(args.log_csv, row)
            print(
                f"episodes={total_episodes} device={device} "
                f"x_wins={winners.count(1)} o_wins={winners.count(-1)} "
                f"draws={winners.count(0)} avg_len={row['avg_length']:.1f} "
                f"policy_steps={row['avg_policy_steps']:.1f} "
                f"opp_self={row['self_games']} opp_heur={row['heuristic_games']} "
                f"opp_line={row['line_games']} "
                f"opp_rand={row['random_games']} "
                f"loss={losses['loss']:.4f} entropy={losses['entropy']:.4f}"
            )

        if args.save_interval > 0 and (
            total_episodes % args.save_interval == 0 or total_episodes >= args.episodes
        ):
            save_checkpoint(args.save_path, model, optimizer, total_episodes, args)
            save_numbered_checkpoint(args.checkpoint_dir, model, optimizer, total_episodes, args)
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
            save_numbered_checkpoint(
                args.checkpoint_dir,
                model,
                optimizer,
                total_episodes,
                args,
                extra={"stopped_early": True},
            )
            print(f"Stopping early; saved PyTorch PPO at episode {total_episodes}.")
            return

    save_checkpoint(args.save_path, model, optimizer, total_episodes, args, extra={"completed": True})
    save_numbered_checkpoint(
        args.checkpoint_dir,
        model,
        optimizer,
        total_episodes,
        args,
        extra={"completed": True},
    )
    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_file.write_text(f"completed episodes={total_episodes} path={args.save_path}\n", encoding="utf-8")
    print(f"PyTorch PPO completed: {args.save_path}")


if __name__ == "__main__":
    main()
