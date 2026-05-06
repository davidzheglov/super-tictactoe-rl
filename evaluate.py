"""Evaluate a trained Super Tic-Tac-Toe model against a random opponent."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
import numpy as np

try:
    from .env import SuperTicTacToeEnv
    from .utils import (
        project_root,
        random_legal_action,
        set_global_seeds,
    )
except ImportError:  # pragma: no cover
    from env import SuperTicTacToeEnv
    from utils import (
        project_root,
        random_legal_action,
        set_global_seeds,
    )


@dataclass
class LoadedAgent:
    backend: str
    model: Any
    device: Any


def load_agent(path: str, hidden_size: int, device_arg: str) -> LoadedAgent:
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}. Run training first.")

    try:
        import torch

        try:
            from .torch_models import (
                TorchDQN,
                TorchPolicyValueNet,
                resolve_torch_device,
            )
        except ImportError:  # pragma: no cover
            from torch_models import TorchDQN, TorchPolicyValueNet, resolve_torch_device

        payload = torch.load(model_path, map_location="cpu")
        algo = str(payload.get("algo", ""))
        torch_device = resolve_torch_device(device_arg)
        if algo in {"torch_ppo", "torchrl_ppo"}:
            model = TorchPolicyValueNet(hidden_sizes=(hidden_size, hidden_size))
            model.load_state_dict(payload["model_state_dict"])
            model.to(torch_device).eval()
            return LoadedAgent("torch_ppo", model, torch_device)
        if algo == "torch_dqn":
            model = TorchDQN(hidden_size=hidden_size)
            model.load_state_dict(payload["online_state_dict"])
            model.to(torch_device).eval()
            return LoadedAgent("torch_dqn", model, torch_device)
        raise ValueError(f"Unsupported checkpoint algorithm {algo!r}.")
    except Exception as exc:
        raise RuntimeError(f"Could not load PyTorch checkpoint {model_path}: {exc}") from exc


def agent_action(agent: LoadedAgent, obs: np.ndarray, mask: np.ndarray, deterministic: bool) -> int:
    if agent.backend == "torch_ppo":
        try:
            from .torch_models import select_action_torch
        except ImportError:  # pragma: no cover
            from torch_models import select_action_torch

        action, _, _ = select_action_torch(agent.model, obs, mask, agent.device, deterministic)
        return action
    if agent.backend == "torch_dqn":
        try:
            from .torch_models import masked_q_argmax
        except ImportError:  # pragma: no cover
            from torch_models import masked_q_argmax

        return masked_q_argmax(agent.model, obs, mask, agent.device)
    raise ValueError(f"Unsupported backend {agent.backend!r}")


def parse_args() -> argparse.Namespace:
    default_model_path = project_root() / "models" / "super_ttt_agent_torchrl.pt"
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--model-path", type=str, default=str(default_model_path))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    agent = load_agent(args.model_path, args.hidden_size, args.device)

    results = {"wins": 0, "losses": 0, "draws": 0}
    lengths = []
    forfeits = []
    illegal_losses = 0

    for game_index in range(args.games):
        env = SuperTicTacToeEnv(seed=int(rng.integers(0, 2**31 - 1)))
        obs, _ = env.reset()
        agent_player = 1 if game_index % 2 == 0 else -1
        done = False
        last_info = {"winner": 0, "reason": "none", "forfeited": False}
        length = 0
        game_forfeits = 0

        while not done:
            action_mask = env.legal_action_mask()
            if env.current_player == agent_player:
                action = agent_action(agent, obs, action_mask, args.deterministic)
            else:
                action = random_legal_action(action_mask, rng)

            obs, _, terminated, truncated, last_info = env.step(action)
            done = bool(terminated or truncated)
            length += 1
            game_forfeits += int(bool(last_info["forfeited"]))

        winner = int(last_info["winner"])
        if winner == agent_player:
            results["wins"] += 1
        elif winner == 0:
            results["draws"] += 1
        else:
            results["losses"] += 1
            illegal_losses += int(last_info["reason"] == "illegal_action")
        lengths.append(length)
        forfeits.append(game_forfeits)

    print(f"Evaluation versus random opponent ({agent.backend})")
    print(f"games: {args.games}")
    print(f"wins: {results['wins']}")
    print(f"losses: {results['losses']}")
    print(f"draws: {results['draws']}")
    print(f"illegal losses: {illegal_losses}")
    print(f"average length: {np.mean(lengths):.2f}")
    print(f"average forfeits: {np.mean(forfeits):.2f}")


if __name__ == "__main__":
    main()
