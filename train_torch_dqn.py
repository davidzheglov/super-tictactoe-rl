"""Resumable PyTorch DQN baseline for Super Tic-Tac-Toe."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .agents import BasicHeuristicAgent, HeuristicAgent, LineBuilderAgent, RandomAgent, board_potential
    from .env import SuperTicTacToeEnv
    from .torch_models import TorchDQN, masked_q_argmax, resolve_torch_device
    from .utils import project_root, random_legal_action, set_global_seeds
except ImportError:  # pragma: no cover
    from agents import BasicHeuristicAgent, HeuristicAgent, LineBuilderAgent, RandomAgent, board_potential
    from env import SuperTicTacToeEnv
    from torch_models import TorchDQN, masked_q_argmax, resolve_torch_device
    from utils import project_root, random_legal_action, set_global_seeds


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int):
        self.data = deque(maxlen=int(capacity))
        self.rng = np.random.default_rng(seed)

    def add(self, item) -> None:
        self.data.append(item)

    def sample(self, batch_size: int):
        indices = self.rng.choice(len(self.data), size=int(batch_size), replace=False)
        return [self.data[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self.data)


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


def select_learning_action(
    online: TorchDQN,
    obs: np.ndarray,
    mask: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> int:
    if rng.random() < epsilon:
        return random_legal_action(mask, rng)
    return masked_q_argmax(online, obs, mask, device)


def select_opponent_action(
    env: SuperTicTacToeEnv,
    opponent: str,
    random_agent: RandomAgent,
    heuristic_agent: HeuristicAgent,
    line_agent: LineBuilderAgent,
    basic_agent: BasicHeuristicAgent,
) -> int:
    if opponent == "heuristic":
        return heuristic_agent.select_action(env)
    if opponent == "line":
        return line_agent.select_action(env)
    if opponent == "basic":
        return basic_agent.select_action(env)
    if opponent == "random":
        return random_agent.select_action(env)
    raise ValueError(f"Unknown non-self opponent mode: {opponent}")


def choose_start_actor(mode: str, rng: np.random.Generator) -> str:
    if mode != "mixed":
        return mode
    labels = np.asarray(["heuristic", "line", "random"], dtype=object)
    probs = np.asarray([0.55, 0.30, 0.15], dtype=np.float64)
    return str(rng.choice(labels, p=probs))


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
        action = select_opponent_action(
            env,
            actor if actor != "mixed" else "heuristic",
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


def save_checkpoint(
    path: str,
    online: TorchDQN,
    target: TorchDQN,
    optimizer: torch.optim.Optimizer,
    episodes: int,
    args: argparse.Namespace,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "algo": "torch_dqn",
        "episodes": int(episodes),
        "online_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
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
                "algo": "torch_dqn",
                "episodes": int(episodes),
                "checkpoint": str(save_path),
                "hidden_size": int(args.hidden_size),
                "seed": int(args.seed),
                **(extra or {}),
            },
            f,
            indent=2,
        )


def load_checkpoint(
    path: str,
    online: TorchDQN,
    target: TorchDQN,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    save_path = Path(path)
    if not save_path.exists():
        return 0
    payload = torch.load(save_path, map_location=device)
    online.load_state_dict(payload["online_state_dict"])
    target.load_state_dict(payload.get("target_state_dict", payload["online_state_dict"]))
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return int(payload.get("episodes", 0))


def save_numbered_checkpoint(
    checkpoint_dir: str,
    online: TorchDQN,
    target: TorchDQN,
    optimizer: torch.optim.Optimizer,
    episodes: int,
    args: argparse.Namespace,
    extra: Optional[Dict[str, object]] = None,
) -> None:
    if not checkpoint_dir:
        return
    path = Path(checkpoint_dir) / f"dqn_ep{int(episodes):07d}.pt"
    save_checkpoint(str(path), online, target, optimizer, episodes, args, extra=extra)


def evaluate_win_rate(
    online: TorchDQN,
    n_games: int,
    device: torch.device,
    placement_mode: str,
    seed: int,
) -> Dict[str, float]:
    """Play n_games vs smart heuristic and line-builder; return win rates."""
    eval_env = SuperTicTacToeEnv(seed=seed + 9999, placement_mode=placement_mode)
    eval_rng = np.random.default_rng(seed + 9999)
    heur = HeuristicAgent(seed=seed + 11)
    line = LineBuilderAgent(seed=seed + 13)
    results: Dict[str, float] = {}
    for opp_name, opp_agent in [("heuristic", heur), ("line", line)]:
        wins = 0
        for g in range(n_games):
            agent_player = 1 if g % 2 == 0 else -1
            obs, _ = eval_env.reset(seed=int(eval_rng.integers(0, 2**31 - 1)))
            done = False
            while not done:
                if eval_env.current_player == agent_player:
                    mask = eval_env.legal_action_mask()
                    action = masked_q_argmax(online, obs, mask, device)
                else:
                    action = opp_agent.select_action(eval_env)
                obs, _, terminated, truncated, info = eval_env.step(action)
                done = bool(terminated or truncated)
            if info.get("winner", 0) == agent_player:
                wins += 1
        results[opp_name] = wins / n_games
    return results


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train PyTorch DQN baseline.")
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu", "mps"])
    parser.add_argument(
        "--placement-mode",
        type=str,
        default="stochastic",
        choices=["stochastic", "deterministic"],
    )
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--warmup-steps", type=int, default=2048)
    parser.add_argument("--target-update-episodes", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-frac", type=float, default=0.7)
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
    )
    parser.add_argument("--start-state-min-plies", type=int, default=4)
    parser.add_argument("--start-state-max-plies", type=int, default=18)
    parser.add_argument("--save-path", type=str, default=str(root / "models" / "dqn_agent_torch.pt"))
    parser.add_argument("--log-csv", type=str, default=str(root / "models" / "dqn_torch_log.csv"))
    parser.add_argument("--checkpoint-dir", type=str, default="")
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=5000,
                        help="Evaluate win rate vs heuristic every N episodes (0 = disable)")
    parser.add_argument("--eval-games", type=int, default=50,
                        help="Number of games per win-rate evaluation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-if-done", action="store_true")
    parser.add_argument("--done-file", type=str, default="")
    parser.add_argument("--stop-after-seconds", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    done_file = Path(args.done_file or args.save_path + ".done")
    if args.skip_if_done and done_file.exists():
        print(f"Done marker exists; skipping PyTorch DQN: {done_file}")
        return

    set_global_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_torch_device(args.device)
    print(f"PyTorch DQN device: {device}")

    online = TorchDQN(hidden_size=args.hidden_size).to(device)
    target = TorchDQN(hidden_size=args.hidden_size).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr)

    start_episode = 0
    if args.resume:
        start_episode = load_checkpoint(args.save_path, online, target, optimizer, device)
        if start_episode:
            print(f"Resumed PyTorch DQN checkpoint {args.save_path} at episode {start_episode}")
    if args.skip_if_done and start_episode >= args.episodes:
        done_file.parent.mkdir(parents=True, exist_ok=True)
        done_file.write_text("done\n", encoding="utf-8")
        print(f"PyTorch DQN already has {start_episode} episodes; skipping.")
        return

    replay = ReplayBuffer(args.replay_size, args.seed)
    env = SuperTicTacToeEnv(seed=args.seed, placement_mode=args.placement_mode)
    random_agent = RandomAgent(seed=args.seed + 17)
    heuristic_agent = HeuristicAgent(seed=args.seed + 29)
    line_agent = LineBuilderAgent(seed=args.seed + 31)
    basic_agent = BasicHeuristicAgent(seed=args.seed + 37)
    started_at = time.time()

    for episode in range(start_episode, args.episodes):
        frac = min(1.0, episode / max(args.episodes * args.eps_decay_frac, 1.0))
        epsilon = args.eps_start + frac * (args.eps_end - args.eps_start)
        opponent = choose_opponent(args, rng)
        agent_player = choose_agent_player(args.agent_player_mode, episode, rng)
        obs = reset_with_start_state(
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
        done = False
        steps = 0
        forfeits = 0
        last_info = {"winner": 0}
        losses: List[float] = []

        while not done:
            if opponent != "self" and env.current_player != agent_player:
                action = select_opponent_action(
                    env,
                    opponent,
                    random_agent,
                    heuristic_agent,
                    line_agent,
                    basic_agent,
                )
                obs, _, terminated, truncated, last_info = env.step(action)
                done = bool(terminated or truncated)
                forfeits += int(bool(last_info.get("forfeited", False)))
                steps += 1
                continue

            mask = env.legal_action_mask()
            action = select_learning_action(online, obs, mask, epsilon, rng, device)
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

            next_obs, reward, terminated, truncated, last_info = env.step(action)
            done = bool(terminated or truncated)
            forfeits += int(bool(last_info.get("forfeited", False)))
            steps += 1
            transition_reward = float(reward)
            if args.shaping_scale != 0.0:
                potential_after = board_potential(
                    env.board,
                    acting_player,
                    defense_weight=args.shaping_defense_weight,
                )
                delta = float(np.clip(potential_after - potential_before, -args.shaping_clip, args.shaping_clip))
                transition_reward += args.shaping_scale * delta
            if bool(last_info.get("forfeited", False)):
                transition_reward -= args.forfeit_penalty
            replay_next_obs = next_obs
            replay_done = done
            next_value_sign = -1.0

            if opponent != "self":
                next_value_sign = 1.0
                if not done:
                    opponent_action = select_opponent_action(
                        env,
                        opponent,
                        random_agent,
                        heuristic_agent,
                        line_agent,
                        basic_agent,
                    )
                    replay_next_obs, _, terminated, truncated, last_info = env.step(
                        opponent_action
                    )
                    replay_done = bool(terminated or truncated)
                    done = replay_done
                    forfeits += int(bool(last_info.get("forfeited", False)))
                    steps += 1
                    if replay_done:
                        winner = int(last_info.get("winner", 0))
                        if winner == agent_player:
                            transition_reward = 1.0
                        elif winner == -agent_player:
                            transition_reward = -1.0
                        else:
                            transition_reward = 0.0

            next_mask = env.legal_action_mask()
            replay.add(
                (
                    obs,
                    action,
                    transition_reward,
                    replay_next_obs,
                    replay_done,
                    next_mask,
                    next_value_sign,
                )
            )
            obs = replay_next_obs

            if len(replay) >= max(args.warmup_steps, args.batch_size):
                batch = replay.sample(args.batch_size)
                b_obs, b_action, b_reward, b_next_obs, b_done, b_next_mask, b_next_sign = map(
                    np.asarray, zip(*batch)
                )
                obs_t = torch.as_tensor(b_obs, dtype=torch.float32, device=device)
                action_t = torch.as_tensor(b_action.astype(np.int64), dtype=torch.long, device=device)
                reward_t = torch.as_tensor(b_reward.astype(np.float32), dtype=torch.float32, device=device)
                next_obs_t = torch.as_tensor(b_next_obs, dtype=torch.float32, device=device)
                done_t = torch.as_tensor(b_done.astype(np.float32), dtype=torch.float32, device=device)
                next_mask_t = torch.as_tensor(b_next_mask.astype(bool), dtype=torch.bool, device=device)
                next_sign_t = torch.as_tensor(
                    b_next_sign.astype(np.float32), dtype=torch.float32, device=device
                )

                with torch.no_grad():
                    next_q = target(next_obs_t).masked_fill(~next_mask_t, -1.0e9)
                    target_q = reward_t + (
                        1.0 - done_t
                    ) * args.gamma * next_sign_t * next_q.max(dim=1).values

                q_values = online(obs_t)
                q_action = q_values.gather(1, action_t[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(q_action, target_q)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.item()))

        episode_num = episode + 1
        if episode_num % args.target_update_episodes == 0:
            target.load_state_dict(online.state_dict())

        # Periodic win-rate evaluation against fixed opponents
        eval_results: Dict[str, float] = {}
        do_eval = (
            args.eval_interval > 0
            and (episode_num % args.eval_interval == 0 or episode_num >= args.episodes)
        )
        if do_eval:
            online.eval()
            with torch.no_grad():
                eval_results = evaluate_win_rate(
                    online, args.eval_games, device, args.placement_mode, args.seed
                )
            online.train()
            print(
                f"  eval ep={episode_num}: vs heuristic {eval_results.get('heuristic', 0):.0%}  "
                f"vs line {eval_results.get('line', 0):.0%}"
            )

        if episode_num % args.log_interval == 0 or episode_num >= args.episodes:
            row = {
                "time_unix": time.time(),
                "algo": "torch_dqn",
                "episodes": episode_num,
                "winner": int(last_info["winner"]),
                "steps": steps,
                "forfeits": forfeits,
                "epsilon": epsilon,
                "opponent": opponent,
                "agent_player": agent_player,
                "shaping_scale": args.shaping_scale,
                "placement_mode": args.placement_mode,
                "start_state_mode": args.start_state_mode,
                "replay_size": len(replay),
                "loss": float(np.mean(losses)) if losses else np.nan,
                "wr_heuristic": eval_results.get("heuristic", np.nan),
                "wr_line": eval_results.get("line", np.nan),
                "device": str(device),
                "elapsed_seconds": time.time() - started_at,
            }
            append_csv_row(args.log_csv, row)
            print(
                f"episodes={episode_num} device={device} winner={last_info['winner']} "
                f"steps={steps} replay={len(replay)} epsilon={epsilon:.3f} "
                f"opponent={opponent} "
                f"loss={row['loss']:.4f}"
            )
        if episode_num % args.save_interval == 0 or episode_num >= args.episodes:
            save_checkpoint(args.save_path, online, target, optimizer, episode_num, args)
            save_numbered_checkpoint(args.checkpoint_dir, online, target, optimizer, episode_num, args)
            print(f"Saved PyTorch DQN checkpoint at episode {episode_num} to {args.save_path}")
        if args.stop_after_seconds > 0 and time.time() - started_at >= args.stop_after_seconds:
            save_checkpoint(
                args.save_path,
                online,
                target,
                optimizer,
                episode_num,
                args,
                {"stopped_early": True},
            )
            save_numbered_checkpoint(
                args.checkpoint_dir,
                online,
                target,
                optimizer,
                episode_num,
                args,
                {"stopped_early": True},
            )
            print(f"Stopping early; saved PyTorch DQN at episode {episode_num}.")
            return

    save_checkpoint(
        args.save_path,
        online,
        target,
        optimizer,
        args.episodes,
        args,
        {"completed": True},
    )
    save_numbered_checkpoint(
        args.checkpoint_dir,
        online,
        target,
        optimizer,
        args.episodes,
        args,
        {"completed": True},
    )
    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_file.write_text(f"completed episodes={args.episodes} path={args.save_path}\n", encoding="utf-8")
    print(f"PyTorch DQN completed: {args.save_path}")


if __name__ == "__main__":
    main()
