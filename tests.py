"""Basic test suite for board and environment behavior."""

from __future__ import annotations

import unittest

import numpy as np

try:
    from .board import SuperTicTacToeBoard, all_playable_coords
except ImportError:  # pragma: no cover
    from board import SuperTicTacToeBoard, all_playable_coords

try:
    from .env import SuperTicTacToeEnv

    ENV_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name == "gymnasium":
        SuperTicTacToeEnv = None
        ENV_AVAILABLE = False
    else:
        try:
            from env import SuperTicTacToeEnv

            ENV_AVAILABLE = True
        except ModuleNotFoundError as fallback_exc:
            if fallback_exc.name != "gymnasium":
                raise
            SuperTicTacToeEnv = None
            ENV_AVAILABLE = False
except ImportError:  # pragma: no cover
    try:
        from env import SuperTicTacToeEnv

        ENV_AVAILABLE = True
    except ModuleNotFoundError as exc:
        if exc.name != "gymnasium":
            raise
        SuperTicTacToeEnv = None
        ENV_AVAILABLE = False
else:
    ENV_AVAILABLE = True


class DirectRng:
    def random(self):
        return 0.1

    def integers(self, low, high=None):
        return low


class RedirectOutRng:
    def random(self):
        return 0.9

    def integers(self, low, high=None):
        return 0


class BoardTests(unittest.TestCase):
    def test_action_coordinate_roundtrip(self):
        board = SuperTicTacToeBoard()
        self.assertEqual(len(all_playable_coords()), 96)
        for action, coord in enumerate(all_playable_coords()):
            self.assertEqual(board.action_to_coord(action), coord)
            self.assertEqual(board.coord_to_action(coord), action)

    def test_invalid_level_not_playable(self):
        board = SuperTicTacToeBoard()
        self.assertFalse(board.is_valid_coord((0, 1, 0, 0)))
        self.assertFalse(board.is_valid_coord((1, 2, 0, 0)))
        self.assertFalse(board.is_valid_coord((2, 3, 0, 0)))

    def test_direct_stochastic_move_places_mark(self):
        board = SuperTicTacToeBoard()
        result = board.resolve_move((0, 0, 0, 0), 1, DirectRng())
        self.assertFalse(result["forfeited"])
        self.assertTrue(result["accepted_directly"])
        self.assertEqual(board.grid[0, 0, 0, 0], 1)

    def test_redirected_outside_move_forfeits(self):
        board = SuperTicTacToeBoard()
        result = board.resolve_move((0, 0, 0, 0), 1, RedirectOutRng())
        self.assertTrue(result["forfeited"])
        self.assertEqual(result["reason"], "redirected_outside_board")
        self.assertEqual(board.grid.sum(), 0)

    def test_local_row_is_win(self):
        board = SuperTicTacToeBoard()
        for col in range(4):
            board.place((1, 0, 2, col), 1)
        self.assertEqual(board.check_winner(), 1)

    def test_local_column_alone_is_not_win(self):
        board = SuperTicTacToeBoard()
        for row in range(4):
            board.place((2, 1, row, 3), -1)
        self.assertEqual(board.check_winner(), 0)

    def test_same_local_cell_across_boards_is_not_win(self):
        board = SuperTicTacToeBoard()
        for coord in [(0, 0, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1), (2, 0, 1, 1)]:
            board.place(coord, 1)
        self.assertEqual(board.check_winner(), 0)

    def test_global_row_spanning_boards_win(self):
        board = SuperTicTacToeBoard()
        for coord in [(1, 0, 0, 2), (1, 0, 0, 3), (1, 1, 0, 0), (1, 1, 0, 1)]:
            board.place(coord, 1)
        self.assertEqual(board.check_winner(), 1)

    def test_global_column_spanning_levels_win(self):
        board = SuperTicTacToeBoard()
        for coord in [(0, 0, 2, 1), (0, 0, 3, 1), (1, 0, 0, 3), (1, 0, 1, 3)]:
            board.place(coord, -1)
        self.assertEqual(board.check_winner(), -1)

    def test_global_diagonal_win(self):
        board = SuperTicTacToeBoard()
        for coord in [
            (0, 0, 0, 0),
            (0, 0, 1, 1),
            (0, 0, 2, 2),
            (0, 0, 3, 3),
            (1, 1, 0, 2),
        ]:
            board.place(coord, -1)
        self.assertEqual(board.check_winner(), -1)


@unittest.skipUnless(ENV_AVAILABLE, "gymnasium is not installed")
class EnvironmentTests(unittest.TestCase):
    def test_reset_observation_shape(self):
        env = SuperTicTacToeEnv(seed=123)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (97,))
        self.assertEqual(info["action_mask"].shape, (96,))
        self.assertEqual(int(info["action_mask"].sum()), 96)

    def test_illegal_occupied_action_terminates(self):
        env = SuperTicTacToeEnv(seed=123)
        env.reset()
        action = env.board.coord_to_action((0, 0, 0, 0))
        env.board.place((0, 0, 0, 0), 1)
        _, reward, terminated, _, info = env.step(action)
        self.assertTrue(terminated)
        self.assertEqual(reward, -1.0)
        self.assertEqual(info["winner"], -1)

    def test_legal_action_mask_shrinks_after_direct_move(self):
        env = SuperTicTacToeEnv(seed=123)
        env.reset()
        env.rng = DirectRng()
        action = env.board.coord_to_action((0, 0, 0, 0))
        env.step(action)
        self.assertFalse(env.legal_action_mask()[action])


if __name__ == "__main__":
    unittest.main()
