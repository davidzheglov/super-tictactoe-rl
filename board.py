"""Board logic for Super Tic-Tac-Toe.

The playable board is represented as six 4x4 boards arranged in triangular
levels. The backing array is rectangular for convenience, but only these level
positions are legal:

    (0, 0)
    (1, 0), (1, 1)
    (2, 0), (2, 1), (2, 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

Coord = Tuple[int, int, int, int]

BOARD_SIZE = 4
LEVEL_ROWS = 3
VALID_LEVEL_POSITIONS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1),
    (2, 2),
)
ADJACENT_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@lru_cache(maxsize=1)
def all_playable_coords() -> Tuple[Coord, ...]:
    """Return every legal cell coordinate in action order."""
    coords: List[Coord] = []
    for level_row, level_col in VALID_LEVEL_POSITIONS:
        for local_row in range(BOARD_SIZE):
            for local_col in range(BOARD_SIZE):
                coords.append((level_row, level_col, local_row, local_col))
    return tuple(coords)


@lru_cache(maxsize=1)
def _coord_to_action_map() -> Dict[Coord, int]:
    return {coord: action for action, coord in enumerate(all_playable_coords())}


def _as_coord(coord: Sequence[int]) -> Optional[Coord]:
    if len(coord) != 4:
        return None
    try:
        return tuple(int(part) for part in coord)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


@dataclass
class MoveResult:
    intended_coord: Coord
    actual_coord: Optional[Coord]
    accepted_directly: bool
    forfeited: bool
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "intended_coord": self.intended_coord,
            "actual_coord": self.actual_coord,
            "accepted_directly": self.accepted_directly,
            "forfeited": self.forfeited,
            "reason": self.reason,
        }


class SuperTicTacToeBoard:
    """Pure NumPy implementation of the Super Tic-Tac-Toe board."""

    def __init__(self) -> None:
        self.grid = np.zeros(
            (LEVEL_ROWS, LEVEL_ROWS, BOARD_SIZE, BOARD_SIZE), dtype=np.int8
        )

    @staticmethod
    def valid_level_positions() -> Tuple[Tuple[int, int], ...]:
        return VALID_LEVEL_POSITIONS

    @staticmethod
    def action_to_coord(action: int) -> Coord:
        action = int(action)
        coords = all_playable_coords()
        if action < 0 or action >= len(coords):
            raise ValueError(f"Action must be in [0, {len(coords) - 1}], got {action}.")
        return coords[action]

    @staticmethod
    def coord_to_action(coord: Sequence[int]) -> int:
        parsed = _as_coord(coord)
        if parsed is None or parsed not in _coord_to_action_map():
            raise ValueError(f"Invalid playable coordinate: {coord!r}")
        return _coord_to_action_map()[parsed]

    def reset(self) -> None:
        self.grid.fill(0)

    def copy(self) -> "SuperTicTacToeBoard":
        copied = SuperTicTacToeBoard()
        copied.grid = self.grid.copy()
        return copied

    def is_valid_coord(self, coord: Sequence[int]) -> bool:
        parsed = _as_coord(coord)
        if parsed is None:
            return False
        level_row, level_col, local_row, local_col = parsed
        return (
            (level_row, level_col) in VALID_LEVEL_POSITIONS
            and 0 <= local_row < BOARD_SIZE
            and 0 <= local_col < BOARD_SIZE
        )

    def is_empty(self, coord: Sequence[int]) -> bool:
        if not self.is_valid_coord(coord):
            return False
        level_row, level_col, local_row, local_col = _as_coord(coord)  # type: ignore[arg-type]
        return self.grid[level_row, level_col, local_row, local_col] == 0

    def legal_actions(self) -> List[int]:
        return [
            action
            for action, coord in enumerate(all_playable_coords())
            if self.is_empty(coord)
        ]

    def place(self, coord: Sequence[int], player: int) -> bool:
        if player not in (-1, 1):
            raise ValueError("Player must be 1 for X or -1 for O.")
        if not self.is_valid_coord(coord):
            raise ValueError(f"Invalid coordinate: {coord!r}")
        if not self.is_empty(coord):
            return False
        level_row, level_col, local_row, local_col = _as_coord(coord)  # type: ignore[arg-type]
        self.grid[level_row, level_col, local_row, local_col] = player
        return True

    def is_full(self) -> bool:
        return len(self.legal_actions()) == 0

    def resolve_move(
        self, chosen_coord: Sequence[int], player: int, rng: np.random.Generator
    ) -> Dict[str, object]:
        """Resolve a stochastic move and mutate the board if it lands legally."""
        parsed = _as_coord(chosen_coord)
        if parsed is None or not self.is_valid_coord(parsed):
            raise ValueError(f"Invalid chosen coordinate: {chosen_coord!r}")
        if player not in (-1, 1):
            raise ValueError("Player must be 1 for X or -1 for O.")

        accepted_directly = bool(rng.random() < 0.5)
        if accepted_directly:
            if not self.is_empty(parsed):
                return MoveResult(
                    intended_coord=parsed,
                    actual_coord=parsed,
                    accepted_directly=True,
                    forfeited=True,
                    reason="chosen_cell_occupied",
                ).as_dict()
            self.place(parsed, player)
            return MoveResult(
                intended_coord=parsed,
                actual_coord=parsed,
                accepted_directly=True,
                forfeited=False,
                reason="accepted_directly",
            ).as_dict()

        direction_index = int(rng.integers(0, len(ADJACENT_DIRECTIONS)))
        delta_row, delta_col = ADJACENT_DIRECTIONS[direction_index]
        level_row, level_col, local_row, local_col = parsed
        redirected = (
            level_row,
            level_col,
            local_row + delta_row,
            local_col + delta_col,
        )

        if not self.is_valid_coord(redirected):
            return MoveResult(
                intended_coord=parsed,
                actual_coord=redirected,
                accepted_directly=False,
                forfeited=True,
                reason="redirected_outside_board",
            ).as_dict()
        if not self.is_empty(redirected):
            return MoveResult(
                intended_coord=parsed,
                actual_coord=redirected,
                accepted_directly=False,
                forfeited=True,
                reason="redirected_cell_occupied",
            ).as_dict()

        self.place(redirected, player)
        return MoveResult(
            intended_coord=parsed,
            actual_coord=redirected,
            accepted_directly=False,
            forfeited=False,
            reason="redirected",
        ).as_dict()

    def check_winner(self) -> int:
        for player in (1, -1):
            if (
                self._has_global_row(player)
                or self._has_global_column(player)
                or self._has_global_diagonal(player)
            ):
                return player
        return 0

    @staticmethod
    def visual_global_coord(coord: Coord) -> Tuple[int, int]:
        """Map a cell to its row/column in the visible pyramid layout."""
        level_row, level_col, local_row, local_col = coord
        global_row = level_row * BOARD_SIZE + local_row
        centered_offset = (LEVEL_ROWS - 1 - level_row) * (BOARD_SIZE // 2)
        global_col = centered_offset + level_col * BOARD_SIZE + local_col
        return global_row, global_col

    def _global_maps(self) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Coord]]:
        values: Dict[Tuple[int, int], int] = {}
        coords: Dict[Tuple[int, int], Coord] = {}
        for coord in all_playable_coords():
            level_row, level_col, local_row, local_col = coord
            global_coord = self.visual_global_coord(coord)
            values[global_coord] = int(
                self.grid[level_row, level_col, local_row, local_col]
            )
            coords[global_coord] = coord
        return values, coords

    def _line_is_player(
        self,
        player: int,
        start: Tuple[int, int],
        direction: Tuple[int, int],
        length: int,
    ) -> Optional[List[Coord]]:
        values, coord_map = self._global_maps()
        global_line = [
            (start[0] + step * direction[0], start[1] + step * direction[1])
            for step in range(length)
        ]
        if not all(values.get(global_coord) == player for global_coord in global_line):
            return None
        return [coord_map[global_coord] for global_coord in global_line]

    def _has_global_row(self, player: int) -> bool:
        values, _ = self._global_maps()
        for row, col in values:
            if self._line_is_player(player, (row, col), (0, 1), 4) is not None:
                return True
        return False

    def _has_global_column(self, player: int) -> bool:
        values, _ = self._global_maps()
        for row, col in values:
            line = self._line_is_player(player, (row, col), (1, 0), 4)
            if line is None:
                continue
            level_rows = {coord[0] for coord in line}
            if len(level_rows) > 1:
                return True
        return False

    def _has_global_diagonal(self, player: int) -> bool:
        values, _ = self._global_maps()
        for row, col in values:
            for direction in ((1, 1), (1, -1)):
                if self._line_is_player(player, (row, col), direction, 5) is not None:
                    return True
        return False

    def render_text(self) -> str:
        symbol = {1: "X", -1: "O", 0: "."}
        lines: List[str] = []
        for level_row in range(LEVEL_ROWS):
            boards = [
                (level_row, level_col)
                for level_col in range(level_row + 1)
                if (level_row, level_col) in VALID_LEVEL_POSITIONS
            ]
            lines.append(f"Level {level_row + 1}")
            for local_row in range(BOARD_SIZE):
                row_chunks = []
                for level_row_i, level_col_i in boards:
                    cells = [
                        symbol[int(self.grid[level_row_i, level_col_i, local_row, c])]
                        for c in range(BOARD_SIZE)
                    ]
                    row_chunks.append(" ".join(cells))
                lines.append("   ".join(row_chunks))
            lines.append("")
        return "\n".join(lines).rstrip()

    def iter_values_in_action_order(self) -> Iterable[int]:
        for level_row, level_col, local_row, local_col in all_playable_coords():
            yield int(self.grid[level_row, level_col, local_row, local_col])
