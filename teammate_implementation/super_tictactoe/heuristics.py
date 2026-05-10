import random
import numpy as np
from super_tictactoe.env import SuperTicTacToeEnv


def greedy_agent(env):
    """Pure offensive: win immediately if possible, else maximise own potential. No blocking."""
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    me = env.current_player

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        if env._check_win(me):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    best_a, best_v = valid[0], -1.0
    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        v = env._evaluate_board(me)
        env.board[r, c] = 0
        if v > best_v:
            best_v, best_a = v, a
    return int(best_a)


def blocking_agent(env):
    """Win if possible, block opponent's immediate win, else maximise own potential."""
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    me  = env.current_player
    opp = 3 - me

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        if env._check_win(me):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = opp
        if env._check_win(opp):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    best_a, best_v = valid[0], -1.0
    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        v = env._evaluate_board(me)
        env.board[r, c] = 0
        if v > best_v:
            best_v, best_a = v, a
    return int(best_a)


def safe_agent(env):
    """Win/block first, then prefer cells with more valid neighbours (lower forfeit risk)."""
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    me  = env.current_player
    opp = 3 - me

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        if env._check_win(me):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = opp
        if env._check_win(opp):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    def valid_neighbour_count(a):
        r, c = a // 12, a % 12
        return sum(
            1 for dr, dc in SuperTicTacToeEnv._DIRECTIONS
            if 0 <= r+dr < 12 and 0 <= c+dc < 12
            and env.valid_mask[r+dr, c+dc]
        )

    best_a, best_v = valid[0], -1.0
    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        v = env._evaluate_board(me) + 0.05 * valid_neighbour_count(a)
        env.board[r, c] = 0
        if v > best_v:
            best_v, best_a = v, a
    return int(best_a)


def one_step_lookahead_agent(env):
    """Depth-2 minimax: win/block immediately, then pick move that maximises
    own potential minus opponent's best greedy response."""
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    me  = env.current_player
    opp = 3 - me

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        if env._check_win(me):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = opp
        if env._check_win(opp):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    # Pre-rank by own potential, only search top 20
    scored = []
    for a in valid:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        scored.append((env._evaluate_board(me), a))
        env.board[r, c] = 0
    scored.sort(reverse=True)
    candidates = [a for _, a in scored[:20]]

    best_a, best_score = candidates[0], float('-inf')
    for a in candidates:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        my_val = env._evaluate_board(me)

        rows, cols = np.where(env.valid_mask & (env.board == 0))
        opp_valid = rows * 12 + cols
        opp_best = 0.0
        for oa in opp_valid:
            or_, oc = oa // 12, oa % 12
            env.board[or_, oc] = opp
            v = env._evaluate_board(opp)
            env.board[or_, oc] = 0
            if v > opp_best:
                opp_best = v

        score = my_val - opp_best
        env.board[r, c] = 0
        if score > best_score:
            best_score, best_a = score, a
    return int(best_a)


def counter_heuristic(env):
    """Win immediately, then block opponent's most advanced line (any direction), else greedy."""
    mask = env.get_action_mask()
    valid_set = set(np.where(mask)[0])
    me  = env.current_player
    opp = 3 - me

    # Win immediately
    for a in valid_set:
        r, c = a // 12, a % 12
        env.board[r, c] = me
        if env._check_win(me):
            env.board[r, c] = 0
            return int(a)
        env.board[r, c] = 0

    # Build all real winning windows (same as _evaluate_board)
    windows = []
    # Horizontal (4)
    for r in range(12):
        for c in range(9):
            cells = [(r, c+i) for i in range(4)]
            if all(env.valid_mask[rr, cc] for rr, cc in cells):
                windows.append(cells)
    # Vertical cross-level (4)
    for c in range(12):
        for r in range(9):
            cells = [(r+i, c) for i in range(4)]
            if all(env.valid_mask[rr, cc] for rr, cc in cells):
                levels = {env._get_level(rr) for rr, _ in cells}
                if len(levels) >= 2:
                    windows.append(cells)
    # Diagonal ↘ (5)
    for r in range(8):
        for c in range(8):
            cells = [(r+i, c+i) for i in range(5)]
            if all(env.valid_mask[rr, cc] for rr, cc in cells):
                windows.append(cells)
    # Diagonal ↙ (5)
    for r in range(8):
        for c in range(4, 12):
            cells = [(r+i, c-i) for i in range(5)]
            if all(env.valid_mask[rr, cc] for rr, cc in cells):
                windows.append(cells)

    # Find window with most opponent pieces (unblocked by me)
    best_block, best_count = None, 0
    for cells in windows:
        opp_count = sum(1 for rr, cc in cells if env.board[rr, cc] == opp)
        own_count  = sum(1 for rr, cc in cells if env.board[rr, cc] == me)
        if own_count > 0:
            continue  # already blocked
        empty_cells = [rr * 12 + cc for rr, cc in cells if env.board[rr, cc] == 0
                       and (rr * 12 + cc) in valid_set]
        if opp_count > best_count and empty_cells:
            best_count = opp_count
            # Pick the empty cell in this window closest to the center of the window
            best_block = empty_cells[len(empty_cells) // 2]

    if best_count >= 2:
        return int(best_block)

    return greedy_agent(env)


def _winning_windows(env):
    windows = []
    for r in range(12):
        for c in range(9):
            cells = tuple((r, c + i) for i in range(4))
            if all(env.valid_mask[rr, cc] for rr, cc in cells):
                windows.append(("horizontal", cells))

    for c in range(12):
        for r in range(9):
            cells = tuple((r + i, c) for i in range(4))
            if all(env.valid_mask[rr, cc] for rr, cc in cells):
                levels = {env._get_level(rr) for rr, _ in cells}
                if len(levels) >= 2:
                    windows.append(("vertical", cells))

    for r in range(8):
        for c in range(8):
            cells = tuple((r + i, c + i) for i in range(5))
            if all(env.valid_mask[rr, cc] for rr, cc in cells):
                windows.append(("diagonal", cells))

    for r in range(8):
        for c in range(4, 12):
            cells = tuple((r + i, c - i) for i in range(5))
            if all(env.valid_mask[rr, cc] for rr, cc in cells):
                windows.append(("diagonal", cells))

    return windows


def _windows_by_cell(env):
    mapping = {
        (r, c): []
        for r in range(12)
        for c in range(12)
        if env.valid_mask[r, c]
    }
    for kind, cells in _winning_windows(env):
        for cell in cells:
            mapping[cell].append((kind, cells))
    return mapping


def _line_kind_weight(kind):
    if kind == "diagonal":
        return 1.35
    if kind == "vertical":
        return 1.15
    return 1.0


def _progress_value(count, length):
    if count <= 0:
        return 0.0
    if length == 4:
        return {1: 0.05, 2: 0.32, 3: 1.7, 4: 8.0}.get(count, 0.0)
    return {1: 0.035, 2: 0.16, 3: 0.72, 4: 2.4, 5: 8.0}.get(count, 0.0)


def _threat_value(count, length):
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


def _cell_priority_bonus(row, col):
    center_bonus = -abs(row - 5.5) * 0.03 - abs(col - 5.5) * 0.03
    level_bonus = 0.04 * env_level_from_row(row)
    return center_bonus + level_bonus


def env_level_from_row(row):
    if row <= 3:
        return 0
    if row <= 7:
        return 1
    return 2


def _line_counts(env, cells, player, placed_cell=None, placed_player=None):
    own = opponent = empty = 0
    opp = 3 - player
    for rr, cc in cells:
        if placed_cell == (rr, cc) and placed_player is not None:
            value = placed_player
        else:
            value = int(env.board[rr, cc])
        if value == player:
            own += 1
        elif value == opp:
            opponent += 1
        else:
            empty += 1
    return own, opponent, empty


def _cell_tactical_score(env, cell, player, windows_for_cell):
    row, col = cell
    if env.board[row, col] != 0:
        return -10000.0

    offense = 0.0
    defense = 0.0
    own_forks = 0
    defense_forks = 0
    opp = 3 - player

    for kind, cells in windows_for_cell.get(cell, ()):
        length = len(cells)
        kind_weight = _line_kind_weight(kind)

        own_before, opp_before, _ = _line_counts(env, cells, player)
        if opp_before == 0:
            after = own_before + 1
            if after >= length:
                offense += 12000.0 * kind_weight
            else:
                offense += 150.0 * _progress_value(after, length) * kind_weight
                if after >= length - 2:
                    own_forks += 1

        opp_count, own_blockers, _ = _line_counts(env, cells, opp)
        if own_blockers == 0 and opp_count > 0:
            value = _threat_value(opp_count, length) * kind_weight
            if kind == "horizontal" and opp_count >= 2:
                value *= 1.2
            if kind == "vertical" and opp_count >= 2:
                value *= 1.15
            defense += value
            if opp_count >= max(2, length - 2):
                defense_forks += 1

    if own_forks >= 2:
        offense += 150.0 * (own_forks - 1)
    if defense_forks >= 2:
        defense += 260.0 * (defense_forks - 1)

    return offense + 1.35 * defense + _cell_priority_bonus(row, col)


def _landing_distribution(env, action):
    row, col = action // 12, action % 12
    outcomes = {}
    forfeit_prob = 0.0

    if env.valid_mask[row, col] and env.board[row, col] == 0:
        outcomes[(row, col)] = outcomes.get((row, col), 0.0) + 0.5
    else:
        forfeit_prob += 0.5

    for dr, dc in SuperTicTacToeEnv._DIRECTIONS:
        rr, cc = row + dr, col + dc
        if 0 <= rr < 12 and 0 <= cc < 12 and env.valid_mask[rr, cc] and env.board[rr, cc] == 0:
            outcomes[(rr, cc)] = outcomes.get((rr, cc), 0.0) + 1.0 / 16.0
        else:
            forfeit_prob += 1.0 / 16.0

    return outcomes, forfeit_prob


def stronger_heuristic(env):
    """Risk-aware tactical heuristic.

    It scores every legal intended move by its full stochastic landing
    distribution. The score values immediate wins, dangerous opponent lines,
    intersections between threats, long own constructs, and low-forfeit cells.
    """
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    player = env.current_player
    windows_for_cell = _windows_by_cell(env)
    cell_cache = {
        (r, c): _cell_tactical_score(env, (r, c), player, windows_for_cell)
        for r in range(12)
        for c in range(12)
        if env.valid_mask[r, c] and env.board[r, c] == 0
    }

    best_action = int(valid[0])
    best_score = float("-inf")
    for action in valid:
        outcomes, forfeit_prob = _landing_distribution(env, int(action))
        score = 0.0
        for cell, probability in outcomes.items():
            score += probability * cell_cache.get(cell, -10000.0)
        success_prob = sum(outcomes.values())
        score += 8.0 * success_prob
        score -= 90.0 * forfeit_prob
        score += 1.0e-6 * random.random()
        if score > best_score:
            best_score = score
            best_action = int(action)

    return best_action


# Weighted pool: heavier on harder opponents to push past current ceiling
HEURISTIC_POOL = [
    (greedy_agent,      0.10),
    (blocking_agent,    0.25),
    (safe_agent,        0.30),
    (counter_heuristic, 0.35),
]

_agents  = [h for h, _ in HEURISTIC_POOL]
_weights = [w for _, w in HEURISTIC_POOL]


def random_heuristic(env):
    """Randomly sample a heuristic opponent each call."""
    agent = random.choices(_agents, weights=_weights, k=1)[0]
    return agent(env)
