"""Reusable agents and matchup utilities for Super Tic-Tac-Toe."""

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np

try:
    from .board import (
        ADJACENT_DIRECTIONS,
        BOARD_SIZE,
        Coord,
        SuperTicTacToeBoard,
        all_playable_coords,
    )
    from .env import SuperTicTacToeEnv
    from .utils import random_legal_action
except ImportError:  # pragma: no cover
    from board import (
        ADJACENT_DIRECTIONS,
        BOARD_SIZE,
        Coord,
        SuperTicTacToeBoard,
        all_playable_coords,
    )
    from env import SuperTicTacToeEnv
    from utils import random_legal_action


class Agent(Protocol):
    name: str

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        ...


def clone_env(env: SuperTicTacToeEnv, seed: Optional[int] = None) -> SuperTicTacToeEnv:
    cloned = SuperTicTacToeEnv(seed=seed, placement_mode=getattr(env, "placement_mode", "stochastic"))
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
    for line in _lines_by_coord().get(coord, ()):
        own, opponent, _ = _line_counts(
            env.board,
            line,
            player,
            placed_coord=coord,
            placed_player=player,
        )
        if opponent == 0 and own == len(line.cells):
            return True
    return False


def _center_score(coord: Coord) -> float:
    _, _, local_row, local_col = coord
    center = (BOARD_SIZE - 1) / 2.0
    return -abs(local_row - center) - abs(local_col - center)


def _corner_score(coord: Coord) -> float:
    _, _, local_row, local_col = coord
    return 1.0 if local_row in {0, BOARD_SIZE - 1} and local_col in {0, BOARD_SIZE - 1} else 0.0


@dataclass(frozen=True)
class WinningLine:
    cells: Tuple[Coord, ...]
    kind: str
    direction: Tuple[int, int]


@lru_cache(maxsize=1)
def _global_coord_maps() -> Tuple[Dict[Tuple[int, int], Coord], Dict[Coord, Tuple[int, int]]]:
    by_global: Dict[Tuple[int, int], Coord] = {}
    by_coord: Dict[Coord, Tuple[int, int]] = {}
    for coord in all_playable_coords():
        global_coord = SuperTicTacToeBoard.visual_global_coord(coord)
        by_global[global_coord] = coord
        by_coord[coord] = global_coord
    return by_global, by_coord


@lru_cache(maxsize=1)
def winning_lines() -> Tuple[WinningLine, ...]:
    """Enumerate every real winning window in the visible pyramid board.

    Rows and columns require four cells. Diagonals require five. Vertical
    windows are included only when they span at least two triangular levels,
    matching `SuperTicTacToeBoard.check_winner`.
    """

    by_global, _ = _global_coord_maps()
    windows: List[WinningLine] = []
    seen = set()
    specs = (
        ("horizontal", (0, 1), 4),
        ("vertical", (1, 0), 4),
        ("diagonal", (1, 1), 5),
        ("diagonal", (1, -1), 5),
    )
    for start in by_global:
        for kind, direction, length in specs:
            global_cells = tuple(
                (start[0] + step * direction[0], start[1] + step * direction[1])
                for step in range(length)
            )
            if not all(global_coord in by_global for global_coord in global_cells):
                continue
            cells = tuple(by_global[global_coord] for global_coord in global_cells)
            if kind == "vertical" and len({coord[0] for coord in cells}) <= 1:
                continue
            key = (kind, cells)
            if key in seen:
                continue
            seen.add(key)
            windows.append(WinningLine(cells=cells, kind=kind, direction=direction))
    return tuple(windows)


@lru_cache(maxsize=1)
def _lines_by_coord() -> Dict[Coord, Tuple[WinningLine, ...]]:
    mapping: Dict[Coord, List[WinningLine]] = {coord: [] for coord in all_playable_coords()}
    for line in winning_lines():
        for coord in line.cells:
            mapping[coord].append(line)
    return {coord: tuple(lines) for coord, lines in mapping.items()}


def landing_distribution(
    board: SuperTicTacToeBoard,
    action: int,
) -> Tuple[Dict[Coord, float], float]:
    """Return actual landing probabilities and forfeit probability for action.

    The environment accepts the selected cell with probability 1/2. Otherwise it
    samples one of the eight local neighbours with probability 1/16 each. Invalid
    or occupied redirected cells are forfeits, so risk depends heavily on local
    free space.
    """

    chosen = board.action_to_coord(action)
    outcomes: Dict[Coord, float] = {}
    forfeit_prob = 0.0

    if board.is_empty(chosen):
        outcomes[chosen] = outcomes.get(chosen, 0.0) + 0.5
    else:
        forfeit_prob += 0.5

    level_row, level_col, local_row, local_col = chosen
    for delta_row, delta_col in ADJACENT_DIRECTIONS:
        redirected = (level_row, level_col, local_row + delta_row, local_col + delta_col)
        if board.is_valid_coord(redirected) and board.is_empty(redirected):
            outcomes[redirected] = outcomes.get(redirected, 0.0) + 1.0 / 16.0
        else:
            forfeit_prob += 1.0 / 16.0
    return outcomes, forfeit_prob


def _line_counts(
    board: SuperTicTacToeBoard,
    line: WinningLine,
    player: int,
    placed_coord: Optional[Coord] = None,
    placed_player: Optional[int] = None,
) -> Tuple[int, int, int]:
    own = opponent = empty = 0
    for coord in line.cells:
        if coord == placed_coord and placed_player is not None:
            value = int(placed_player)
        else:
            value = int(board.grid[coord[0], coord[1], coord[2], coord[3]])
        if value == player:
            own += 1
        elif value == -player:
            opponent += 1
        else:
            empty += 1
    return own, opponent, empty


def _line_kind_weight(line: WinningLine) -> float:
    if line.kind == "diagonal":
        return 1.35
    if line.kind == "vertical":
        return 1.15
    return 1.0


def _progress_value(count: int, length: int) -> float:
    if count <= 0:
        return 0.0
    if length == 4:
        return {1: 0.05, 2: 0.32, 3: 1.7, 4: 8.0}.get(count, 0.0)
    return {1: 0.035, 2: 0.16, 3: 0.72, 4: 2.4, 5: 8.0}.get(count, 0.0)


def _threat_value(count: int, length: int) -> float:
    if count <= 0:
        return 0.0
    if count >= length - 1:
        return 9000.0
    if length == 4 and count == 2:
        return 380.0
    if length == 5 and count == 3:
        return 520.0
    if count == 2:
        return 120.0
    return 20.0


def _cell_priority_bonus(coord: Coord) -> float:
    level_row, _, _, _ = coord
    return 0.35 * _center_score(coord) + 0.12 * _corner_score(coord) + 0.08 * level_row


def cell_tactical_score(
    board: SuperTicTacToeBoard,
    coord: Coord,
    player: int,
    offense_weight: float = 1.0,
    defense_weight: float = 1.25,
) -> float:
    """Score an actual landing cell using human tactical knowledge.

    The score is additive over all winning windows containing `coord`, so
    intersections between horizontal, vertical, and diagonal threats naturally
    become high-value cells.
    """

    if not board.is_empty(coord):
        return -10000.0

    offense = 0.0
    defense = 0.0
    own_fork_lines = 0
    defensive_fork_lines = 0
    for line in _lines_by_coord().get(coord, ()):
        length = len(line.cells)
        kind_weight = _line_kind_weight(line)

        own_before, opp_before, _ = _line_counts(board, line, player)
        if opp_before == 0:
            after = own_before + 1
            if after >= length:
                offense += 12000.0 * kind_weight
            else:
                offense += 150.0 * _progress_value(after, length) * kind_weight
                if after >= length - 2:
                    own_fork_lines += 1

        opp_count, own_blockers, _ = _line_counts(board, line, -player)
        if own_blockers == 0 and opp_count > 0:
            value = _threat_value(opp_count, length) * kind_weight
            if line.kind == "horizontal" and opp_count >= 2:
                value *= 1.2
            if line.kind == "vertical" and opp_count >= 2:
                value *= 1.15
            defense += value
            if opp_count >= max(2, length - 2):
                defensive_fork_lines += 1

    if own_fork_lines >= 2:
        offense += 150.0 * (own_fork_lines - 1)
    if defensive_fork_lines >= 2:
        defense += 260.0 * (defensive_fork_lines - 1)

    return offense_weight * offense + defense_weight * defense + _cell_priority_bonus(coord)


def action_expected_score(
    board: SuperTicTacToeBoard,
    action: int,
    player: int,
    offense_weight: float = 1.0,
    defense_weight: float = 1.25,
    forfeit_weight: float = 80.0,
    success_weight: float = 8.0,
    cell_score_cache: Optional[Dict[Coord, float]] = None,
) -> float:
    outcomes, forfeit_prob = landing_distribution(board, action)
    expected = 0.0
    for coord, probability in outcomes.items():
        if cell_score_cache is not None and coord in cell_score_cache:
            cell_score = cell_score_cache[coord]
        else:
            cell_score = cell_tactical_score(
                board,
                coord,
                player,
                offense_weight=offense_weight,
                defense_weight=defense_weight,
            )
        expected += probability * cell_score
    success_prob = sum(outcomes.values())
    return expected + success_weight * success_prob - forfeit_weight * forfeit_prob


def offensive_potential(board: SuperTicTacToeBoard, player: int) -> float:
    total = 0.0
    for line in winning_lines():
        own, opponent, empty = _line_counts(board, line, player)
        if opponent or own == 0:
            continue
        flexibility = 1.0 + 0.04 * empty
        total += _progress_value(own, len(line.cells)) * _line_kind_weight(line) * flexibility
    return total


def board_potential(
    board: SuperTicTacToeBoard,
    player: int,
    defense_weight: float = 0.75,
) -> float:
    """Dense value proxy for reward shaping: build own lines, reduce threats."""

    return offensive_potential(board, player) - defense_weight * offensive_potential(board, -player)


class BasicHeuristicAgent:
    """Old compact heuristic kept as an intentionally weak baseline."""

    name = "basic_heuristic"

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


class HeuristicAgent:
    """Risk-aware human-knowledge baseline.

    Rules encoded:
    - prefer moves that can win after direct or redirected placement;
    - block immediate and multi-line opponent threats, including two-in-a-row
      horizontal/vertical threats and three-in-a-diagonal threats;
    - value intersections where one cell touches several dangerous windows;
    - when no direct danger exists, extend the longest open own construct;
    - choose safer cells with more valid empty neighbours to reduce forfeits.
    """

    name = "smart_heuristic"

    def __init__(
        self,
        seed: int = 0,
        offense_weight: float = 1.0,
        defense_weight: float = 1.35,
        forfeit_weight: float = 90.0,
    ):
        self.rng = np.random.default_rng(seed)
        self.offense_weight = float(offense_weight)
        self.defense_weight = float(defense_weight)
        self.forfeit_weight = float(forfeit_weight)

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        legal = env.board.legal_actions()
        player = int(env.current_player)
        cell_score_cache = {
            coord: cell_tactical_score(
                env.board,
                coord,
                player,
                offense_weight=self.offense_weight,
                defense_weight=self.defense_weight,
            )
            for coord in all_playable_coords()
            if env.board.is_empty(coord)
        }
        scored = []
        for action in legal:
            score = action_expected_score(
                env.board,
                action,
                player,
                offense_weight=self.offense_weight,
                defense_weight=self.defense_weight,
                forfeit_weight=self.forfeit_weight,
                cell_score_cache=cell_score_cache,
            )
            score += 1.0e-4 * float(self.rng.random())
            scored.append((score, action))
        return int(max(scored)[1])


class LineBuilderAgent:
    """Aggressive baseline that mainly extends its own strongest line."""

    name = "line_builder"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: SuperTicTacToeEnv) -> int:
        legal = env.board.legal_actions()
        player = int(env.current_player)
        cell_score_cache = {
            coord: cell_tactical_score(
                env.board,
                coord,
                player,
                offense_weight=1.8,
                defense_weight=0.18,
            )
            for coord in all_playable_coords()
            if env.board.is_empty(coord)
        }
        scored = []
        for action in legal:
            score = action_expected_score(
                env.board,
                action,
                player,
                offense_weight=1.8,
                defense_weight=0.18,
                forfeit_weight=65.0,
                success_weight=10.0,
                cell_score_cache=cell_score_cache,
            )
            score += 1.0e-4 * float(self.rng.random())
            scored.append((score, action))
        return int(max(scored)[1])


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
