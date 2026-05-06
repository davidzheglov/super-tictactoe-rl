"""Reusable agents and matchup utilities for Super Tic-Tac-Toe."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np

try:
    from .board import BOARD_SIZE, Coord, all_playable_coords
    from .env import SuperTicTacToeEnv
    from .utils import random_legal_action
except ImportError:  # pragma: no cover
    from board import BOARD_SIZE, Coord, all_playable_coords
    from env import SuperTicTacToeEnv
    from utils import random_legal_action


class Agent(Protocol):
    name: str

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        ...


def clone_env(env: SuperTicTacToeEnv, seed: Optional[int] = None) -> SuperTicTacToeEnv:
    cloned = SuperTicTacToeEnv(seed=seed)
    cloned.board = env.board.copy()
    cloned.current_player = int(env.current_player)
    if seed is None:
        cloned.rng = copy.deepcopy(env.rng)
    return cloned


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        return random_legal_action(env.legal_action_mask(), self.rng)


def _would_win(env: SuperTicTacToeEnv, action: int, player: int) -> bool:
    coord = env.board.action_to_coord(action)
    if not env.board.is_empty(coord):
        return False
    board = env.board.copy()
    board.place(coord, player)
    return board.check_winner() == player


def _center_score(coord: Coord) -> float:
    _, _, local_row, local_col = coord
    center = (BOARD_SIZE - 1) / 2.0
    return -abs(local_row - center) - abs(local_col - center)


def _corner_score(coord: Coord) -> float:
    _, _, local_row, local_col = coord
    return 1.0 if local_row in {0, BOARD_SIZE - 1} and local_col in {0, BOARD_SIZE - 1} else 0.0


class HeuristicAgent:
    """Simple human-knowledge baseline.

    The heuristic ignores stochastic redirection probabilities and scores the
    chosen target cell. It still respects legal-action masks.
    """

    name = "heuristic"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        legal = env.board.legal_actions()
        player = int(env.current_player)

        winning = [action for action in legal if _would_win(env, action, player)]
        if winning:
            return int(self.rng.choice(winning))

        blocking = [action for action in legal if _would_win(env, action, -player)]
        if blocking:
            return int(self.rng.choice(blocking))

        scored = []
        for action in legal:
            coord = env.board.action_to_coord(action)
            level_row, _, _, _ = coord
            score = 0.0
            score += 4.0 + _center_score(coord)
            score += 0.75 * _corner_score(coord)
            score += 0.25 * (2 - abs(level_row - 1))
            score += 1.0e-3 * float(self.rng.random())
            scored.append((score, action))
        return int(max(scored)[1])


def rollout_random(
    env: SuperTicTacToeEnv,
    root_player: int,
    rng: np.random.Generator,
    max_steps: int = 200,
) -> float:
    done = False
    info = {"winner": 0}
    steps = 0
    while not done and steps < max_steps:
        action = random_legal_action(env.legal_action_mask(), rng)
        _, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
    winner = int(info.get("winner", 0))
    if winner == root_player:
        return 1.0
    if winner == 0:
        return 0.0
    return -1.0


class RolloutMCTSAgent:
    """Lightweight Monte Carlo action-evaluation baseline.

    This is rollout search, not a full transposition-table UCT implementation.
    It samples stochastic move outcomes through the environment and estimates
    each legal action by average rollout return.
    """

    def __init__(self, rollouts_per_action: int = 8, seed: int = 0, max_steps: int = 200):
        self.rollouts_per_action = int(rollouts_per_action)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.name = f"rollout_mcts_{self.rollouts_per_action}"

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        legal = env.board.legal_actions()
        root_player = int(env.current_player)
        best_action = int(legal[0])
        best_score = -float("inf")

        for action in legal:
            total = 0.0
            for _ in range(self.rollouts_per_action):
                sim = clone_env(env, seed=int(self.rng.integers(0, 2**31 - 1)))
                _, _, terminated, truncated, info = sim.step(action)
                if bool(terminated or truncated):
                    winner = int(info["winner"])
                    total += 1.0 if winner == root_player else 0.0 if winner == 0 else -1.0
                else:
                    total += rollout_random(sim, root_player, self.rng, self.max_steps)
            score = total / max(self.rollouts_per_action, 1)
            if score > best_score:
                best_score = score
                best_action = int(action)
        return best_action


class QTableAgent:
    name = "q_table"

    def __init__(self, path: str, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        self.q_table = payload.get("q_table", payload)

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        key = tuple(env.get_observation().astype(np.int8).tolist())
        mask = env.legal_action_mask()
        q_values = np.asarray(self.q_table.get(key, np.zeros(96, dtype=np.float32))).copy()
        q_values[~mask] = -1.0e9
        if np.max(q_values) <= -1.0e8:
            return random_legal_action(mask, self.rng)
        return int(np.argmax(q_values))


class TorchPPOAgent:
    name = "torch_ppo"

    def __init__(self, path: str, device: str = "auto", deterministic: bool = True):
        import torch

        try:
            from .torch_models import TorchPolicyValueNet, resolve_torch_device, select_action_torch
        except ImportError:  # pragma: no cover
            from torch_models import TorchPolicyValueNet, resolve_torch_device, select_action_torch

        self.torch = torch
        self.select_action_torch = select_action_torch
        self.device = resolve_torch_device(device)
        self.deterministic = bool(deterministic)
        payload = torch.load(path, map_location="cpu")
        hidden_size = int(payload.get("hidden_size", 256))
        self.model = TorchPolicyValueNet(hidden_sizes=(hidden_size, hidden_size))
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device).eval()

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        action, _, _ = self.select_action_torch(
            self.model,
            env.get_observation(),
            env.legal_action_mask(),
            self.device,
            deterministic=self.deterministic,
        )
        return action


class TorchDQNAgent:
    name = "torch_dqn"

    def __init__(self, path: str, device: str = "auto"):
        import torch

        try:
            from .torch_models import TorchDQN, masked_q_argmax, resolve_torch_device
        except ImportError:  # pragma: no cover
            from torch_models import TorchDQN, masked_q_argmax, resolve_torch_device

        self.masked_q_argmax = masked_q_argmax
        self.device = resolve_torch_device(device)
        payload = torch.load(path, map_location="cpu")
        hidden_size = int(payload.get("hidden_size", 256))
        self.model = TorchDQN(hidden_size=hidden_size)
        self.model.load_state_dict(payload["online_state_dict"])
        self.model.to(self.device).eval()

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        return self.masked_q_argmax(
            self.model,
            env.get_observation(),
            env.legal_action_mask(),
            self.device,
        )


@dataclass
class GameResult:
    winner: int
    winner_name: str
    steps: int
    forfeits: int
    illegal_loss: bool


def play_game(
    agent_x: Agent,
    agent_o: Agent,
    seed: int = 0,
    max_steps: int = 200,
) -> GameResult:
    env = SuperTicTacToeEnv(seed=seed)
    env.reset(seed=seed)
    done = False
    steps = 0
    forfeits = 0
    info = {"winner": 0, "reason": "none", "forfeited": False}

    while not done and steps < max_steps:
        agent = agent_x if env.current_player == 1 else agent_o
        action = agent.select_action(env)
        _, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        steps += 1
        forfeits += int(bool(info.get("forfeited", False)))

    winner = int(info.get("winner", 0))
    if winner == 1:
        winner_name = getattr(agent_x, "name", "X")
    elif winner == -1:
        winner_name = getattr(agent_o, "name", "O")
    else:
        winner_name = "draw"
    return GameResult(
        winner=winner,
        winner_name=winner_name,
        steps=steps,
        forfeits=forfeits,
        illegal_loss=bool(info.get("reason") == "illegal_action"),
    )


def evaluate_matchup(
    agent_a: Agent,
    agent_b: Agent,
    games: int = 100,
    seed: int = 0,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    a_wins = b_wins = draws = 0
    for game_index in range(int(games)):
        a_is_x = game_index % 2 == 0
        agent_x = agent_a if a_is_x else agent_b
        agent_o = agent_b if a_is_x else agent_a
        result = play_game(agent_x, agent_o, seed=seed + game_index)
        if result.winner == 0:
            draws += 1
            winner_label = "draw"
        elif (result.winner == 1 and a_is_x) or (result.winner == -1 and not a_is_x):
            a_wins += 1
            winner_label = "agent_a"
        else:
            b_wins += 1
            winner_label = "agent_b"
        rows.append(
            {
                "game": game_index,
                "agent_a": getattr(agent_a, "name", "agent_a"),
                "agent_b": getattr(agent_b, "name", "agent_b"),
                "agent_a_player": "X" if a_is_x else "O",
                "winner": winner_label,
                "winner_player": result.winner,
                "steps": result.steps,
                "forfeits": result.forfeits,
                "illegal_loss": result.illegal_loss,
            }
        )
    summary = {
        "agent_a": getattr(agent_a, "name", "agent_a"),
        "agent_b": getattr(agent_b, "name", "agent_b"),
        "games": int(games),
        "agent_a_wins": a_wins,
        "agent_b_wins": b_wins,
        "draws": draws,
        "agent_a_win_rate": a_wins / max(int(games), 1),
        "agent_b_win_rate": b_wins / max(int(games), 1),
        "draw_rate": draws / max(int(games), 1),
        "avg_steps": float(np.mean([row["steps"] for row in rows])) if rows else 0.0,
        "avg_forfeits": float(np.mean([row["forfeits"] for row in rows])) if rows else 0.0,
    }
    return rows, summary
