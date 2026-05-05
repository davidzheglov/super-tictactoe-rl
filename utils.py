"""Shared helpers for training, evaluation, and the Streamlit app."""

from __future__ import annotations

import json
import os
import importlib.util
import random
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
if (
    importlib.util.find_spec("tf_agents") is not None
    and importlib.util.find_spec("tf_keras") is not None
):
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np


def project_root() -> Path:
    return Path(__file__).resolve().parent


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def hidden_sizes_from_arg(hidden_size: int) -> Tuple[int, int]:
    hidden_size = int(hidden_size)
    return (hidden_size, hidden_size)


def resolve_tf_device(device: str) -> str:
    import tensorflow as tf

    normalized = (device or "auto").lower()
    if normalized == "cpu":
        return "/CPU:0"

    gpus = tf.config.list_physical_devices("GPU")
    if normalized in {"auto", "cuda", "gpu", "mps"} and gpus:
        return "/GPU:0"
    return "/CPU:0"


PathLike = Union[str, os.PathLike]


def checkpoint_exists(path: PathLike) -> bool:
    path = str(path)
    return os.path.exists(path) or os.path.exists(path + ".index")


def save_checkpoint(model, optimizer, path: PathLike, episodes: int) -> str:
    import tensorflow as tf

    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    episode_counter = tf.Variable(
        int(episodes), trainable=False, dtype=tf.int64, name="episodes"
    )
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, episodes=episode_counter
    )
    checkpoint.write(path)
    metadata = {
        "episodes": int(episodes),
        "format": "tf.train.Checkpoint prefix",
        "checkpoint_prefix": path,
    }
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return path


def load_checkpoint(model, path: PathLike, optimizer=None) -> bool:
    import tensorflow as tf

    path = str(path)
    if not checkpoint_exists(path):
        return False
    checkpoint_kwargs = {"model": model}
    if optimizer is not None:
        checkpoint_kwargs["optimizer"] = optimizer
    checkpoint = tf.train.Checkpoint(**checkpoint_kwargs)
    checkpoint.restore(path).expect_partial()
    return True


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
