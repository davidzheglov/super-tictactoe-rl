"""Evaluate a trained Super Tic-Tac-Toe model against a random opponent."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

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
    from .models import PolicyValueNet, select_action
    from .utils import (
        checkpoint_exists,
        hidden_sizes_from_arg,
        load_checkpoint,
        project_root,
        random_legal_action,
        resolve_tf_device,
        set_global_seeds,
    )
except ImportError:  # pragma: no cover
    from env import SuperTicTacToeEnv
    from models import PolicyValueNet, select_action
    from utils import (
        checkpoint_exists,
        hidden_sizes_from_arg,
        load_checkpoint,
        project_root,
        random_legal_action,
        resolve_tf_device,
        set_global_seeds,
    )


def parse_args() -> argparse.Namespace:
    default_model_path = project_root() / "models" / "super_ttt_agent.pt"
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
    device = resolve_tf_device(args.device)

    if not checkpoint_exists(args.model_path):
        raise FileNotFoundError(
            f"No checkpoint found at prefix {args.model_path}. Run train.py first."
        )

    model = PolicyValueNet(hidden_sizes=hidden_sizes_from_arg(args.hidden_size))
    model(tf.zeros((1, 97), dtype=tf.float32))
    load_checkpoint(model, args.model_path)

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
                action, _, _ = select_action(
                    model,
                    obs,
                    action_mask,
                    device=device,
                    deterministic=args.deterministic,
                )
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

    print("Evaluation versus random opponent")
    print(f"games: {args.games}")
    print(f"wins: {results['wins']}")
    print(f"losses: {results['losses']}")
    print(f"draws: {results['draws']}")
    print(f"illegal losses: {illegal_losses}")
    print(f"average length: {np.mean(lengths):.2f}")
    print(f"average forfeits: {np.mean(forfeits):.2f}")


if __name__ == "__main__":
    main()
