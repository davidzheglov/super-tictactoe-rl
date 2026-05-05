"""Gymnasium and TF-Agents environments for Super Tic-Tac-Toe."""

from __future__ import annotations

import os
import contextlib
import io
import importlib.util
from typing import Dict, Optional, Tuple

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
if (
    importlib.util.find_spec("tf_agents") is not None
    and importlib.util.find_spec("tf_keras") is not None
):
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    with contextlib.redirect_stderr(io.StringIO()):
        from tf_agents.environments import py_environment
        from tf_agents.specs import array_spec
        from tf_agents.trajectories import time_step as ts

    TF_AGENTS_AVAILABLE = True
except Exception:  # pragma: no cover - exercised when tf-agents is absent/incompatible.
    py_environment = None
    array_spec = None
    ts = None
    TF_AGENTS_AVAILABLE = False

try:
    from .board import SuperTicTacToeBoard, all_playable_coords
except ImportError:  # pragma: no cover
    from board import SuperTicTacToeBoard, all_playable_coords


class SuperTicTacToeEnv(gym.Env):
    """A Gymnasium-compatible environment.

    Observation is a float32 vector of length 97:
    - 96 playable cell values in action order, using X=1, O=-1, empty=0.
    - 1 current-player value, using X=1 and O=-1.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, seed: Optional[int] = None, render_mode: Optional[str] = None):
        super().__init__()
        self.board = SuperTicTacToeBoard()
        self.current_player = 1
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(len(all_playable_coords()))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(all_playable_coords()) + 1,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board.reset()
        self.current_player = 1
        info = {
            "winner": 0,
            "legal_actions": self.board.legal_actions(),
            "action_mask": self.legal_action_mask(),
        }
        return self.get_observation(), info

    def step(self, action: int):
        current_player_before_move = self.current_player
        winner = 0
        terminated = False
        truncated = False
        reward = 0.0
        intended_coord = None
        actual_coord = None
        accepted_directly = False
        forfeited = False
        reason = "ok"

        try:
            coord = self.board.action_to_coord(int(action))
        except (TypeError, ValueError):
            coord = None

        if coord is None or not self.board.is_empty(coord):
            winner = -current_player_before_move
            reward = -1.0
            terminated = True
            intended_coord = coord
            actual_coord = None
            forfeited = True
            reason = "illegal_action"
        else:
            move_info = self.board.resolve_move(coord, current_player_before_move, self.rng)
            intended_coord = move_info["intended_coord"]
            actual_coord = move_info["actual_coord"]
            accepted_directly = bool(move_info["accepted_directly"])
            forfeited = bool(move_info["forfeited"])
            reason = str(move_info["reason"])

            winner = self.board.check_winner()
            if winner == current_player_before_move:
                reward = 1.0
                terminated = True
            elif winner == -current_player_before_move:
                reward = -1.0
                terminated = True
            elif self.board.is_full():
                reward = 0.0
                terminated = True
            else:
                self.current_player *= -1

        info = {
            "current_player_before_move": current_player_before_move,
            "intended_coord": intended_coord,
            "actual_coord": actual_coord,
            "accepted_directly": accepted_directly,
            "forfeited": forfeited,
            "reason": reason,
            "winner": winner,
            "legal_actions": self.board.legal_actions(),
            "action_mask": self.legal_action_mask(),
        }
        return self.get_observation(), float(reward), terminated, truncated, info

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.bool_)
        legal_actions = self.board.legal_actions()
        mask[legal_actions] = True
        return mask

    def get_observation(self) -> np.ndarray:
        values = list(self.board.iter_values_in_action_order())
        values.append(self.current_player)
        return np.asarray(values, dtype=np.float32)

    def render(self):
        rendered = self.board.render_text()
        if self.render_mode == "human":
            print(rendered)
        return rendered


if TF_AGENTS_AVAILABLE:

    class SuperTicTacToePyEnvironment(py_environment.PyEnvironment):
        """TF-Agents PyEnvironment wrapper with an action mask in observation."""

        def __init__(self, seed: Optional[int] = None):
            super().__init__()
            self._env = SuperTicTacToeEnv(seed=seed)
            self._episode_ended = False
            self._observation_spec = {
                "observation": array_spec.BoundedArraySpec(
                    shape=(97,),
                    dtype=np.float32,
                    minimum=-1.0,
                    maximum=1.0,
                    name="observation",
                ),
                "action_mask": array_spec.BoundedArraySpec(
                    shape=(96,),
                    dtype=np.int32,
                    minimum=0,
                    maximum=1,
                    name="action_mask",
                ),
            }
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.int32, minimum=0, maximum=95, name="action"
            )

        def action_spec(self):
            return self._action_spec

        def observation_spec(self):
            return self._observation_spec

        def _reset(self):
            self._episode_ended = False
            observation, _ = self._env.reset()
            return ts.restart(self._make_observation(observation))

        def _step(self, action):
            if self._episode_ended:
                return self.reset()
            observation, reward, terminated, truncated, _ = self._env.step(int(action))
            self._episode_ended = bool(terminated or truncated)
            tf_observation = self._make_observation(observation)
            if self._episode_ended:
                return ts.termination(tf_observation, np.asarray(reward, dtype=np.float32))
            return ts.transition(
                tf_observation,
                reward=np.asarray(reward, dtype=np.float32),
                discount=np.asarray(1.0, dtype=np.float32),
            )

        def _make_observation(self, observation: np.ndarray) -> Dict[str, np.ndarray]:
            return {
                "observation": observation.astype(np.float32),
                "action_mask": self._env.legal_action_mask().astype(np.int32),
            }

else:

    class SuperTicTacToePyEnvironment:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "tf-agents is not installed. Install requirements.txt to use "
                "SuperTicTacToePyEnvironment."
            )
