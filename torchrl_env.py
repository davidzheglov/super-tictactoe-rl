"""TorchRL-compatible wrappers for Super Tic-Tac-Toe.

The core game remains a Gymnasium environment because that keeps the assignment
rules easy to test. This module exposes the same game through TorchRL's
TensorDict/GymWrapper stack with an action mask, which is the framework path
used for the bonus requirement.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

try:
    from .env import SuperTicTacToeEnv
except ImportError:  # pragma: no cover
    from env import SuperTicTacToeEnv


class ActionMaskedSuperTicTacToeGym(gym.Env):
    """Gymnasium dict-observation wrapper with a legal-action mask."""

    metadata = SuperTicTacToeEnv.metadata

    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.base_env = SuperTicTacToeEnv(seed=seed)
        self.action_space = self.base_env.action_space
        self.observation_space = spaces.Dict(
            {
                "observation": self.base_env.observation_space,
                "action_mask": spaces.Box(0, 1, shape=(96,), dtype=np.bool_),
            }
        )

    @property
    def board(self):
        return self.base_env.board

    @property
    def current_player(self) -> int:
        return int(self.base_env.current_player)

    def _wrapped_observation(self, observation: np.ndarray) -> dict:
        return {
            "observation": observation.astype(np.float32),
            "action_mask": self.base_env.legal_action_mask().astype(np.bool_),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        observation, info = self.base_env.reset(seed=seed, options=options)
        return self._wrapped_observation(observation), info

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self.base_env.step(int(action))
        return self._wrapped_observation(observation), reward, terminated, truncated, info

    def legal_action_mask(self) -> np.ndarray:
        return self.base_env.legal_action_mask()

    def render(self):
        return self.base_env.render()


def _torch_device(device: str) -> torch.device:
    normalized = (device or "cpu").lower()
    if normalized in {"auto", "cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if normalized in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_torchrl_env(seed: Optional[int] = None, device: str = "cpu"):
    """Create a TorchRL environment with an adaptive action mask."""

    try:
        from torchrl.envs.libs.gym import GymWrapper
        from torchrl.envs.transforms import ActionMask, TransformedEnv
    except ImportError as exc:  # pragma: no cover - depends on optional package.
        raise ImportError(
            "TorchRL is required for make_torchrl_env. Install torchrl and "
            "tensordict from requirements.txt."
        ) from exc

    env = GymWrapper(ActionMaskedSuperTicTacToeGym(seed=seed), device=_torch_device(device))
    return TransformedEnv(env, ActionMask())


def check_torchrl_env(seed: int = 0, rollout_steps: int = 3, device: str = "cpu") -> None:
    """Run TorchRL's environment checker plus a tiny rollout."""

    try:
        from torchrl.envs.utils import check_env_specs
    except ImportError as exc:  # pragma: no cover
        raise ImportError("TorchRL is required to check the environment specs.") from exc

    env = make_torchrl_env(seed=seed, device=device)
    check_env_specs(env)
    rollout = env.rollout(int(rollout_steps))
    print("TorchRL env OK")
    print("rollout batch_size:", rollout.batch_size)
    print("rollout keys:", list(rollout.keys(True, True)))
