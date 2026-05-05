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
    from .env import SuperTicTacToeEnv
    from .torch_models import TorchDQN, masked_q_argmax, resolve_torch_device
    from .utils import project_root, random_legal_action, set_global_seeds
except ImportError:  # pragma: no cover
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


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Train PyTorch DQN baseline.")
    parser.add_argument("--episodes", type=int, default=150000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "gpu", "mps"])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--warmup-steps", type=int, default=2048)
    parser.add_argument("--target-update-episodes", type=int, default=500)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-frac", type=float, default=0.7)
    parser.add_argument("--save-path", type=str, default=str(root / "models" / "dqn_agent_torch.pt"))
    parser.add_argument("--log-csv", type=str, default=str(root / "models" / "dqn_torch_log.csv"))
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
    env = SuperTicTacToeEnv(seed=args.seed)
    started_at = time.time()

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
                action = masked_q_argmax(online, obs, mask, device)

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
                obs_t = torch.as_tensor(b_obs, dtype=torch.float32, device=device)
                action_t = torch.as_tensor(b_action.astype(np.int64), dtype=torch.long, device=device)
                reward_t = torch.as_tensor(b_reward.astype(np.float32), dtype=torch.float32, device=device)
                next_obs_t = torch.as_tensor(b_next_obs, dtype=torch.float32, device=device)
                done_t = torch.as_tensor(b_done.astype(np.float32), dtype=torch.float32, device=device)
                next_mask_t = torch.as_tensor(b_next_mask.astype(bool), dtype=torch.bool, device=device)

                with torch.no_grad():
                    next_q = target(next_obs_t).masked_fill(~next_mask_t, -1.0e9)
                    # The next state belongs to the opponent, so its value is negated.
                    target_q = reward_t + (1.0 - done_t) * (-args.gamma * next_q.max(dim=1).values)

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
        if episode_num % args.log_interval == 0 or episode_num >= args.episodes:
            row = {
                "time_unix": time.time(),
                "algo": "torch_dqn",
                "episodes": episode_num,
                "winner": int(last_info["winner"]),
                "steps": steps,
                "forfeits": forfeits,
                "epsilon": epsilon,
                "replay_size": len(replay),
                "loss": float(np.mean(losses)) if losses else np.nan,
                "device": str(device),
                "elapsed_seconds": time.time() - started_at,
            }
            append_csv_row(args.log_csv, row)
            print(
                f"episodes={episode_num} device={device} winner={last_info['winner']} "
                f"steps={steps} replay={len(replay)} epsilon={epsilon:.3f} "
                f"loss={row['loss']:.4f}"
            )
        if episode_num % args.save_interval == 0 or episode_num >= args.episodes:
            save_checkpoint(args.save_path, online, target, optimizer, episode_num, args)
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
    done_file.parent.mkdir(parents=True, exist_ok=True)
    done_file.write_text(f"completed episodes={args.episodes} path={args.save_path}\n", encoding="utf-8")
    print(f"PyTorch DQN completed: {args.save_path}")


if __name__ == "__main__":
    main()
