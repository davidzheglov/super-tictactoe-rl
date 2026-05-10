import sys
import time
import random as _random
import pygame
import torch
import numpy as np
from super_tictactoe.env import SuperTicTacToeEnv, GRID_POSITIONS
from super_tictactoe.model import ActorCritic

# ── Layout constants ────────────────────────────────────────────────────────
CELL_SIZE   = 50
CELL_PAD    = 3
MARGIN      = 40
BOARD_TOP   = 96
LEVEL_GAP   = 25          # extra vertical space between levels
INFO_HEIGHT = 80          # bottom status bar
PANEL_WIDTH = 310

COLORS = {
    'bg':        (244, 247, 250),
    'panel':     (255, 255, 255),
    'panel_edge': (216, 224, 232),
    'cell':      (37, 171, 218),
    'hover':     (96, 205, 238),
    'grid':      (13, 33, 44),
    'border':    (6, 82, 112),
    'p1':        (21, 29, 39),
    'p2':        (22, 77, 100),
    'p2_fill':   (251, 254, 255),
    'win_line':  (255, 226, 120),
    'forfeit':   (252, 166, 124),
    'text':      (26, 32, 38),
    'muted':     (94, 108, 121),
    'green':     (32, 138, 85),
    'orange':    (198, 111, 34),
    'red':       (190, 62, 62),
}

HEURISTIC_CHOICES = ('random', 'greedy', 'blocking', 'safe', 'counter', 'stronger')


def cell_pixel(row: int, col: int):
    """Top-left pixel of cell (row, col) in 12×12 grid."""
    x = MARGIN + col * (CELL_SIZE + CELL_PAD)
    if row < 4:
        y = BOARD_TOP + row * (CELL_SIZE + CELL_PAD)
    elif row < 8:
        y = BOARD_TOP + 4 * (CELL_SIZE + CELL_PAD) + LEVEL_GAP + (row - 4) * (CELL_SIZE + CELL_PAD)
    else:
        y = BOARD_TOP + 8 * (CELL_SIZE + CELL_PAD) + 2 * LEVEL_GAP + (row - 8) * (CELL_SIZE + CELL_PAD)
    return x, y


def window_size():
    w = MARGIN * 2 + 12 * (CELL_SIZE + CELL_PAD)
    h = BOARD_TOP + MARGIN + 12 * (CELL_SIZE + CELL_PAD) + 2 * LEVEL_GAP + INFO_HEIGHT
    return w + PANEL_WIDTH, h


def board_area_width():
    return MARGIN * 2 + 12 * (CELL_SIZE + CELL_PAD)


def format_name(name: str) -> str:
    return name.replace('_', ' ').title()


def random_agent(env):
    mask = env.get_action_mask()
    valid = np.where(mask)[0]
    return int(_random.choice(valid))


def heuristic_agents():
    from super_tictactoe.heuristics import (
        blocking_agent,
        counter_heuristic,
        greedy_agent,
        safe_agent,
        stronger_heuristic,
    )

    return {
        'random': random_agent,
        'greedy': greedy_agent,
        'blocking': blocking_agent,
        'safe': safe_agent,
        'counter': counter_heuristic,
        'stronger': stronger_heuristic,
    }


def load_model(model_path: str, device: str):
    model = ActorCritic().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def policy_action(model, state, env, device: str):
    action_mask = torch.BoolTensor(env.get_action_mask()).to(device)
    state_tensor = torch.FloatTensor(state).to(device)
    with torch.no_grad():
        action, _, _ = model.get_action(state_tensor, action_mask)
    return int(action)


def draw_board(
    screen,
    env,
    font,
    hover_cell=None,
    last_placed=None,
    forfeit_cell=None,
    message="",
    title="Super Tic-Tac-Toe",
    sidebar_lines=None,
):
    screen.fill(COLORS['bg'])

    title_font = pygame.font.SysFont('arial', 28, bold=True)
    small_font = pygame.font.SysFont('arial', 16)
    mono_font = pygame.font.SysFont('menlo', 14)
    screen.blit(title_font.render(title, True, COLORS['text']), (MARGIN, 16))
    screen.blit(
        small_font.render(
            "Click an empty cell. Stochastic moves may redirect or forfeit.",
            True,
            COLORS['muted'],
        ),
        (MARGIN, 52),
    )

    for r in range(12):
        for c in range(12):
            if not env.valid_mask[r, c]:
                continue
            x, y = cell_pixel(r, c)
            color = COLORS['hover'] if (r, c) == hover_cell else COLORS['cell']
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE), border_radius=4)

            piece = env.board[r, c]
            cx, cy = x + CELL_SIZE // 2, y + CELL_SIZE // 2
            if piece == 1:
                offset = CELL_SIZE // 3
                pygame.draw.line(screen, COLORS['p1'], (cx-offset, cy-offset), (cx+offset, cy+offset), 5)
                pygame.draw.line(screen, COLORS['p1'], (cx+offset, cy-offset), (cx-offset, cy+offset), 5)
            elif piece == 2:
                pygame.draw.circle(screen, COLORS['p2'], (cx, cy), CELL_SIZE // 3, 5)
                pygame.draw.circle(screen, COLORS['p2_fill'], (cx, cy), CELL_SIZE // 3 - 2, 2)

            if last_placed == (r, c):
                pygame.draw.rect(screen, COLORS['win_line'], (x, y, CELL_SIZE, CELL_SIZE), 3, border_radius=4)
            if forfeit_cell == (r, c):
                pygame.draw.rect(screen, COLORS['forfeit'], (x, y, CELL_SIZE, CELL_SIZE), 3, border_radius=4)

    for grid_r, grid_c in GRID_POSITIONS:
        x, y = cell_pixel(grid_r, grid_c)
        width = 4 * CELL_SIZE + 3 * CELL_PAD
        pygame.draw.rect(screen, COLORS['border'], (x, y, width, width), 4, border_radius=5)

    if env.done:
        status = f"Player {env.winner} wins!" if env.winner else "Draw!"
    elif message:
        status = message
    else:
        status = f"Player {'1 (X)' if env.current_player == 1 else '2 (O)'}'s turn"

    panel_x = board_area_width() + 8
    _, h = window_size()
    panel_rect = pygame.Rect(panel_x, 28, PANEL_WIDTH - 32, h - 56)
    pygame.draw.rect(screen, COLORS['panel'], panel_rect, border_radius=8)
    pygame.draw.rect(screen, COLORS['panel_edge'], panel_rect, 1, border_radius=8)

    status_color = COLORS['green']
    if env.done and env.winner:
        status_color = COLORS['orange']
    elif message:
        status_color = COLORS['red'] if 'forfeit' in message.lower() else COLORS['orange']

    screen.blit(font.render(status, True, status_color), (panel_x + 22, 54))
    screen.blit(
        small_font.render(f"Current player: {'X' if env.current_player == 1 else 'O'}", True, COLORS['text']),
        (panel_x + 22, 96),
    )
    screen.blit(
        small_font.render("[R] restart  [Space] pause/continue", True, COLORS['muted']),
        (panel_x + 22, 126),
    )

    if sidebar_lines:
        screen.blit(font.render("Match", True, COLORS['text']), (panel_x + 22, 176))
        y = 212
        for line in sidebar_lines[:12]:
            rendered = small_font.render(line, True, COLORS['muted'])
            screen.blit(rendered, (panel_x + 22, y))
            y += 24

    screen.blit(mono_font.render("Last move", True, COLORS['muted']), (panel_x + 22, h - 128))
    if last_placed:
        last_text = f"placed: ({last_placed[0]}, {last_placed[1]})"
    elif forfeit_cell:
        last_text = f"forfeit: ({forfeit_cell[0]}, {forfeit_cell[1]})"
    else:
        last_text = "none"
    screen.blit(small_font.render(last_text, True, COLORS['text']), (panel_x + 22, h - 98))
    pygame.display.flip()


def get_cell_from_mouse(env, mx, my):
    for r in range(12):
        for c in range(12):
            if not env.valid_mask[r, c]:
                continue
            x, y = cell_pixel(r, c)
            if x <= mx < x + CELL_SIZE and y <= my < y + CELL_SIZE:
                return r, c
    return None


def run_human_vs_agent(
    model_path: str,
    human_player: int = 1,
    device: str = 'cpu',
    num_simulations: int = 0,
    opponent: str = 'model',
):
    model = None
    mcts = None
    opponent_fn = None
    opponent_label = "PPO agent"

    if opponent == 'model':
        from super_tictactoe.mcts import MCTS
        model = load_model(model_path, device)
        mcts = MCTS(model, device=device, num_simulations=num_simulations) if num_simulations > 0 else None
    else:
        heuristics = heuristic_agents()
        if opponent not in heuristics:
            print(f"Unknown opponent '{opponent}'. Choose: model, {list(heuristics)}")
            return
        opponent_fn = heuristics[opponent]
        opponent_label = format_name(opponent)

    pygame.init()
    screen = pygame.display.set_mode(window_size())
    pygame.display.set_caption(f"Super Tic-Tac-Toe - Human vs {opponent_label}")
    font = pygame.font.SysFont('arial', 20, bold=True)
    clock = pygame.time.Clock()

    env = SuperTicTacToeEnv()
    state = env.reset()
    hover_cell = None
    last_placed = None
    forfeit_cell = None
    message = ""
    paused = False  # True while waiting for Space after a forfeit

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state = env.reset()
                    last_placed = None
                    forfeit_cell = None
                    message = ""
                    paused = False
                elif event.key == pygame.K_SPACE and paused:
                    paused = False
                    forfeit_cell = None
                    message = ""

            if event.type == pygame.MOUSEMOTION:
                cell = get_cell_from_mouse(env, *event.pos)
                hover_cell = cell if (cell and env.board[cell[0], cell[1]] == 0) else None

            if event.type == pygame.MOUSEBUTTONDOWN and not env.done and not paused:
                if env.current_player == human_player:
                    cell = get_cell_from_mouse(env, *event.pos)
                    if cell and env.get_action_mask()[cell[0] * 12 + cell[1]]:
                        action = cell[0] * 12 + cell[1]
                        state, _, _, info = env.step(action)
                        last_placed = info['placed']
                        if info['forfeited']:
                            forfeit_cell = (action // 12, action % 12)
                            message = "Your move forfeited! (drifted out of bounds)  [Space to continue]"
                            paused = True
                        else:
                            forfeit_cell = None
                            message = ""

        if not env.done and not paused and env.current_player != human_player:
            time.sleep(0.4)
            if mcts:
                action = mcts.get_action(env)
            elif opponent_fn is not None:
                action = opponent_fn(env)
            else:
                action = policy_action(model, state, env, device)
            state, _, _, info = env.step(action)
            last_placed = info['placed']
            if info['forfeited']:
                forfeit_cell = (action // 12, action % 12)
                message = f"{opponent_label} forfeit at ({action // 12},{action % 12})"
                paused = True
            else:
                forfeit_cell = None
                message = ""

        sidebar_lines = [
            f"Human: Player {human_player}",
            f"Opponent: {opponent_label}",
            "Use R to restart.",
            "Use Space after a forfeit.",
        ]
        draw_board(
            screen,
            env,
            font,
            hover_cell,
            last_placed,
            forfeit_cell,
            message,
            title=f"Human vs {opponent_label}",
            sidebar_lines=sidebar_lines,
        )
        clock.tick(30)


def run_agent_vs_heuristic(model_path: str, heuristic_name: str = 'blocking',
                            agent_player: int = 1, delay: float = 0.5, device: str = 'cpu'):
    heuristics = heuristic_agents()
    if heuristic_name not in heuristics:
        print(f"Unknown heuristic '{heuristic_name}'. Choose: {list(heuristics)}")
        return
    heuristic = heuristics[heuristic_name]

    model = load_model(model_path, device)

    heuristic_player = 3 - agent_player

    pygame.init()
    screen = pygame.display.set_mode(window_size())
    pygame.display.set_caption(
        f"Super Tic-Tac-Toe — PPO (P{agent_player}) vs {heuristic_name.capitalize()} (P{heuristic_player})"
    )
    font   = pygame.font.SysFont('arial', 20, bold=True)
    clock  = pygame.time.Clock()

    wins   = {agent_player: 0, heuristic_player: 0, 'draw': 0}
    game_n = 0

    env   = SuperTicTacToeEnv()
    state = env.reset()
    last_placed  = None
    forfeit_cell = None
    paused       = False
    last_move_time = time.time()
    auto_restart = True   # restart automatically after a game ends

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state = env.reset()
                    last_placed  = None
                    forfeit_cell = None
                    paused       = False
                    last_move_time = time.time()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    last_move_time = time.time()
                elif event.key == pygame.K_a:
                    auto_restart = not auto_restart

        # Auto-restart when game ends
        if env.done and auto_restart and time.time() - last_move_time >= delay * 2:
            if env.winner:
                wins[env.winner] += 1
            else:
                wins['draw'] += 1
            game_n += 1
            state = env.reset()
            last_placed  = None
            forfeit_cell = None
            last_move_time = time.time()

        if not env.done and not paused and time.time() - last_move_time >= delay:
            who = "PPO" if env.current_player == agent_player else heuristic_name.capitalize()
            intended_r, intended_c = None, None

            if env.current_player == agent_player:
                action = policy_action(model, state, env, device)
            else:
                action = heuristic(env)

            intended_r, intended_c = action // 12, action % 12
            state, _, _, info = env.step(action)
            last_placed = info['placed']

            if info['forfeited']:
                forfeit_cell = (intended_r, intended_c)
                print(f"[Game {game_n+1}] FORFEIT  {who} (P{env.current_player ^ 3}) "
                      f"intended ({intended_r},{intended_c}) → drifted out/occupied, turn wasted")
                paused = True
            else:
                placed_r, placed_c = info['placed']
                drifted = (placed_r != intended_r or placed_c != intended_c)
                drift_str = f"→ drifted to ({placed_r},{placed_c})" if drifted else "→ landed exactly"
                print(f"[Game {game_n+1}] MOVE     {who} (P{env.current_player ^ 3}) "
                      f"intended ({intended_r},{intended_c}) {drift_str}"
                      + (f"  WIN!" if env.done and env.winner else ""))
                forfeit_cell = None

            last_move_time = time.time()

        if env.done:
            if env.winner == agent_player:
                msg = "PPO wins!"
            elif env.winner == heuristic_player:
                msg = f"{heuristic_name.capitalize()} wins!"
            else:
                msg = "Draw!"
        elif paused and forfeit_cell:
            msg = f"FORFEIT at ({forfeit_cell[0]},{forfeit_cell[1]}) — orange cell — Space to resume"
        elif paused:
            msg = "PAUSED — Space to resume"
        else:
            who = "PPO" if env.current_player == agent_player else heuristic_name.capitalize()
            msg = f"{who}'s turn (P{env.current_player})"

        sidebar_lines = [
            f"Game {game_n + 1}",
            f"PPO: {wins[agent_player]}",
            f"{format_name(heuristic_name)}: {wins[heuristic_player]}",
            f"Draw: {wins['draw']}",
            f"Auto restart: {'ON' if auto_restart else 'OFF'}",
            "R restart",
            "Space pause",
            "A auto restart",
        ]
        draw_board(
            screen,
            env,
            font,
            last_placed=last_placed,
            forfeit_cell=forfeit_cell,
            message=msg,
            title=f"PPO vs {format_name(heuristic_name)}",
            sidebar_lines=sidebar_lines,
        )
        clock.tick(30)


def run_agent_vs_agent(model1_path: str, model2_path: str, delay: float = 0.5, device: str = 'cpu'):
    def load(path):
        m = ActorCritic().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        return m

    models = {1: load(model1_path), 2: load(model2_path)}

    pygame.init()
    screen = pygame.display.set_mode(window_size())
    pygame.display.set_caption("Super Tic-Tac-Toe — Agent vs Agent")
    font = pygame.font.SysFont('monospace', 18)
    clock = pygame.time.Clock()

    env = SuperTicTacToeEnv()
    state = env.reset()
    last_placed = None
    forfeit_cell = None
    message = ""
    paused = False
    last_move_time = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state = env.reset()
                    last_placed = None
                    forfeit_cell = None
                    message = ""
                    paused = False
                    last_move_time = time.time()
                elif event.key == pygame.K_SPACE and paused:
                    paused = False
                    forfeit_cell = None
                    message = ""
                    last_move_time = time.time()

        if not env.done and not paused and time.time() - last_move_time >= delay:
            model = models[env.current_player]
            action_mask = torch.BoolTensor(env.get_action_mask()).to(device)
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action, _, _ = model.get_action(state_tensor, action_mask)
            state, _, _, info = env.step(action)
            last_placed = info['placed']
            if info['forfeited']:
                forfeit_cell = (action // 12, action % 12)
                message = f"P{3 - env.current_player} forfeit! (intended {action // 12},{action % 12})  [Space to continue]"
                paused = True
            else:
                forfeit_cell = None
                message = ""
                last_move_time = time.time()

        draw_board(screen, env, font, last_placed=last_placed, forfeit_cell=forfeit_cell, message=message)
        clock.tick(30)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['human', 'agent', 'heuristic'])
    parser.add_argument('--model1', type=str, default='checkpoints/model_final.pt')
    parser.add_argument('--model2', type=str, default='checkpoints/model_final.pt')
    parser.add_argument('--human-player', type=int, default=1, choices=[1, 2])
    parser.add_argument('--opponent', type=str, default='model',
                        choices=('model',) + HEURISTIC_CHOICES,
                        help='Human mode opponent: trained model or a heuristic.')
    parser.add_argument('--agent-player', type=int, default=1, choices=[1, 2],
                        help='Which player slot the PPO agent occupies (heuristic mode)')
    parser.add_argument('--heuristic', type=str, default='blocking',
                        choices=HEURISTIC_CHOICES,
                        help='Heuristic opponent (heuristic mode)')
    parser.add_argument('--delay', type=float, default=0.5)
    parser.add_argument('--simulations', type=int, default=0,
                        help='MCTS simulations per move (0 = direct policy, no MCTS)')
    args = parser.parse_args()

    if args.mode == 'human':
        run_human_vs_agent(
            args.model1,
            args.human_player,
            num_simulations=args.simulations,
            opponent=args.opponent,
        )
    elif args.mode == 'heuristic':
        run_agent_vs_heuristic(args.model1, args.heuristic, args.agent_player, args.delay)
    else:
        run_agent_vs_agent(args.model1, args.model2, args.delay)
