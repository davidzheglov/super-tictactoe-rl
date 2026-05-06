"""Gymnasium environment for Super Tic-Tac-Toe."""

from __future__ import annotations

import os
from typing import Optional

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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
