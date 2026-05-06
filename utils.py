"""Shared helpers for training, evaluation, and the Pygame app."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parent


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def hidden_sizes_from_arg(hidden_size: int) -> Tuple[int, int]:
    hidden_size = int(hidden_size)
    return (hidden_size, hidden_size)


def random_legal_action(action_mask: np.ndarray, rng: np.random.Generator) -> int:
    legal = np.flatnonzero(action_mask)
    if legal.size == 0:
        raise ValueError("No legal actions available.")
    return int(rng.choice(legal))


def player_name(player: int) -> str:
    return "X" if player == 1 else "O"


def coord_label(coord) -> str:
    if coord is None:
        return "none"
    level_row, level_col, local_row, local_col = coord
    return (
        f"L{level_row + 1}:{level_col + 1} "
        f"r{local_row + 1} c{local_col + 1}"
    )
