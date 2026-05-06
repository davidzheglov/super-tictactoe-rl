"""TorchRL entrypoint for PPO training.

This script validates the TorchRL environment wrapper before launching the
project's PyTorch PPO trainer. Checkpoints are labelled `torchrl_ppo` so reports
can distinguish the bonus-framework run from the plain PyTorch baseline.
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_torchrl_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--torchrl-smoke-steps", type=int, default=3)
    return parser.parse_known_args(argv)


def main() -> None:
    torchrl_args, remaining = parse_torchrl_args(sys.argv[1:])
    try:
        from .torchrl_env import check_torchrl_env
        from .train_torch_ppo import main as ppo_main
    except ImportError:  # pragma: no cover
        from torchrl_env import check_torchrl_env
        from train_torch_ppo import main as ppo_main

    seed = 0
    device = "cpu"
    for index, item in enumerate(remaining):
        if item == "--seed" and index + 1 < len(remaining):
            seed = int(remaining[index + 1])
        if item == "--device" and index + 1 < len(remaining):
            device = remaining[index + 1]

    check_torchrl_env(seed=seed, rollout_steps=torchrl_args.torchrl_smoke_steps, device=device)
    os.environ["SUPER_TTT_ALGO_LABEL"] = "torchrl_ppo"
    os.environ["SUPER_TTT_FRAMEWORK"] = "TorchRL"
    sys.argv = [sys.argv[0], *remaining]
    ppo_main()


if __name__ == "__main__":
    main()
