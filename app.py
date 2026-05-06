"""Pygame UI for playing Super Tic-Tac-Toe against the trained agent."""

from __future__ import annotations

import argparse
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "hide")
if (
    importlib.util.find_spec("tf_agents") is not None
    and importlib.util.find_spec("tf_keras") is not None
):
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pygame
import tensorflow as tf

try:
    from .board import BOARD_SIZE, VALID_LEVEL_POSITIONS
    from .agents import HeuristicAgent
    from .env import SuperTicTacToeEnv
    from .models import PolicyValueNet, select_action
    from .utils import (
        checkpoint_exists,
        coord_label,
        hidden_sizes_from_arg,
        load_checkpoint,
        player_name,
        project_root,
        random_legal_action,
        resolve_tf_device,
    )
except ImportError:  # pragma: no cover
    from board import BOARD_SIZE, VALID_LEVEL_POSITIONS
    from agents import HeuristicAgent
    from env import SuperTicTacToeEnv
    from models import PolicyValueNet, select_action
    from utils import (
        checkpoint_exists,
        coord_label,
        hidden_sizes_from_arg,
        load_checkpoint,
        player_name,
        project_root,
        random_legal_action,
        resolve_tf_device,
    )


DEFAULT_MODEL_PATH = project_root() / "models" / "super_ttt_agent.pt"

CELL_SIZE = 42
BOARD_LEFT = 64
BOARD_TOP = 118
GLOBAL_COLS = 12
GLOBAL_ROWS = 12
BOARD_WIDTH = GLOBAL_COLS * CELL_SIZE
BOARD_HEIGHT = GLOBAL_ROWS * CELL_SIZE
LEVEL_LABEL_X = BOARD_LEFT + BOARD_WIDTH + 28
PANEL_LEFT = LEVEL_LABEL_X + 128
WINDOW_WIDTH = PANEL_LEFT + 320
WINDOW_HEIGHT = 700
FPS = 60

BG = (244, 247, 250)
PANEL_BG = (255, 255, 255)
PANEL_BORDER = (216, 224, 232)
BOARD_FILL = (37, 171, 218)
BOARD_HOVER = (96, 205, 238)
BOARD_LAST = (255, 226, 120)
BOARD_FORFEIT = (252, 166, 124)
GRID = (13, 33, 44)
BOARD_BORDER = (6, 82, 112)
TEXT = (26, 32, 38)
MUTED = (94, 108, 121)
GREEN = (32, 138, 85)
ORANGE = (198, 111, 34)
RED = (190, 62, 62)
BUTTON = (235, 240, 246)
BUTTON_HOVER = (221, 231, 242)
BUTTON_DISABLED = (238, 240, 243)
X_COLOR = (21, 29, 39)
O_COLOR = (251, 254, 255)
O_OUTLINE = (22, 77, 100)

Coord = Tuple[int, int, int, int]


@dataclass
class LoadedAgent:
    backend: str
    model: Any
    device: Any


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    action: str
    enabled: bool = True


@dataclass
class GameState:
    env: SuperTicTacToeEnv
    human_player: int
    deterministic: bool
    use_model: bool
    rng: np.random.Generator
    done: bool = False
    winner: int = 0
    messages: List[str] = field(default_factory=list)
    last_intended: Optional[Coord] = None
    last_actual: Optional[Coord] = None
    last_forfeited: bool = False


def level_offset_cols(level_row: int) -> int:
    """Return the horizontal offset in cell units for a pyramid level."""
    return (2 - level_row) * 2


def board_origin(level_row: int, level_col: int) -> Tuple[int, int]:
    x = BOARD_LEFT + (level_offset_cols(level_row) + level_col * BOARD_SIZE) * CELL_SIZE
    y = BOARD_TOP + level_row * BOARD_SIZE * CELL_SIZE
    return x, y


def coord_rect(coord: Coord) -> pygame.Rect:
    level_row, level_col, local_row, local_col = coord
    board_x, board_y = board_origin(level_row, level_col)
    return pygame.Rect(
        board_x + local_col * CELL_SIZE,
        board_y + local_row * CELL_SIZE,
        CELL_SIZE,
        CELL_SIZE,
    )


def coord_from_point(pos: Tuple[int, int]) -> Optional[Coord]:
    px, py = pos
    for level_row in range(3):
        y0 = BOARD_TOP + level_row * BOARD_SIZE * CELL_SIZE
        y1 = y0 + BOARD_SIZE * CELL_SIZE
        if not (y0 <= py < y1):
            continue
        local_row = (py - y0) // CELL_SIZE
        offset = level_offset_cols(level_row)
        level_start_x = BOARD_LEFT + offset * CELL_SIZE
        level_width = (level_row + 1) * BOARD_SIZE * CELL_SIZE
        if not (level_start_x <= px < level_start_x + level_width):
            return None
        col_in_level = (px - level_start_x) // CELL_SIZE
        level_col = col_in_level // BOARD_SIZE
        local_col = col_in_level % BOARD_SIZE
        coord = (level_row, int(level_col), int(local_row), int(local_col))
        return coord if (coord[0], coord[1]) in VALID_LEVEL_POSITIONS else None
    return None


def load_agent(model_path: str, hidden_size: int, device_arg: str):
    path = Path(model_path)
    if path.exists():
        try:
            import torch

            try:
                from .torch_models import (
                    TorchDQN,
                    TorchPolicyValueNet,
                    resolve_torch_device,
                )
            except ImportError:  # pragma: no cover
                from torch_models import (
                    TorchDQN,
                    TorchPolicyValueNet,
                    resolve_torch_device,
                )

            payload = torch.load(path, map_location="cpu")
            algo = str(payload.get("algo", ""))
            torch_device = resolve_torch_device(device_arg)
            if algo == "torch_ppo":
                model = TorchPolicyValueNet(
                    hidden_sizes=hidden_sizes_from_arg(int(payload.get("hidden_size", hidden_size)))
                )
                model.load_state_dict(payload["model_state_dict"])
                model.to(torch_device).eval()
                return LoadedAgent("torch_ppo", model, torch_device)
            if algo == "torch_dqn":
                model = TorchDQN(hidden_size=int(payload.get("hidden_size", hidden_size)))
                model.load_state_dict(payload["online_state_dict"])
                model.to(torch_device).eval()
                return LoadedAgent("torch_dqn", model, torch_device)
        except Exception as exc:
            print(f"Could not load PyTorch checkpoint {model_path}: {exc}")

    if not checkpoint_exists(model_path):
        return None
    model = PolicyValueNet(hidden_sizes=hidden_sizes_from_arg(hidden_size))
    model(tf.zeros((1, 97), dtype=tf.float32))
    load_checkpoint(model, model_path)
    return LoadedAgent("tf_ppo", model, resolve_tf_device(device_arg))


def reset_state(state: GameState) -> None:
    state.env = SuperTicTacToeEnv()
    state.env.reset()
    state.done = False
    state.winner = 0
    state.messages.clear()
    state.last_intended = None
    state.last_actual = None
    state.last_forfeited = False


def describe_move(info) -> str:
    player = player_name(int(info["current_player_before_move"]))
    intended = coord_label(info["intended_coord"])
    actual = coord_label(info["actual_coord"])
    if info["reason"] == "illegal_action":
        return f"{player} made an illegal move and loses."
    if info["forfeited"]:
        return f"{player} chose {intended}; redirected to {actual}; forfeited."
    if info["accepted_directly"]:
        return f"{player} placed at {actual}."
    return f"{player} chose {intended}; redirected to {actual}."


def apply_action(state: GameState, action: int) -> None:
    _, _, terminated, truncated, info = state.env.step(action)
    state.done = bool(terminated or truncated)
    state.winner = int(info["winner"])
    state.last_intended = info["intended_coord"]
    state.last_actual = info["actual_coord"]
    state.last_forfeited = bool(info["forfeited"])
    state.messages.insert(0, describe_move(info))
    del state.messages[8:]


def agent_turn(state: GameState, model, device: str) -> None:
    action_mask = state.env.legal_action_mask()
    if model is None or not state.use_model:
        action = random_legal_action(action_mask, state.rng)
    elif isinstance(model, LoadedAgent) and model.backend == "heuristic":
        action = model.model.select_action(state.env)
    elif isinstance(model, LoadedAgent) and model.backend == "torch_ppo":
        try:
            from .torch_models import select_action_torch
        except ImportError:  # pragma: no cover
            from torch_models import select_action_torch

        action, _, _ = select_action_torch(
            model.model,
            state.env.get_observation(),
            action_mask,
            device=model.device,
            deterministic=state.deterministic,
        )
    elif isinstance(model, LoadedAgent) and model.backend == "torch_dqn":
        try:
            from .torch_models import masked_q_argmax
        except ImportError:  # pragma: no cover
            from torch_models import masked_q_argmax

        action = masked_q_argmax(
            model.model,
            state.env.get_observation(),
            action_mask,
            device=model.device,
        )
    else:
        action, _, _ = select_action(
            model.model if isinstance(model, LoadedAgent) else model,
            state.env.get_observation(),
            action_mask,
            device=model.device if isinstance(model, LoadedAgent) else device,
            deterministic=state.deterministic,
        )
    apply_action(state, action)


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: Tuple[int, int],
    color=TEXT,
) -> pygame.Rect:
    rendered = font.render(text, True, color)
    rect = rendered.get_rect(topleft=pos)
    surface.blit(rendered, rect)
    return rect


def draw_wrapped_lines(
    surface: pygame.Surface,
    font: pygame.font.Font,
    lines: List[str],
    start: Tuple[int, int],
    color=TEXT,
    line_gap: int = 8,
) -> int:
    y = start[1]
    for line in lines:
        rendered = font.render(line, True, color)
        surface.blit(rendered, (start[0], y))
        y += rendered.get_height() + line_gap
    return y


def draw_button(
    surface: pygame.Surface,
    font: pygame.font.Font,
    button: Button,
    mouse_pos: Tuple[int, int],
) -> None:
    if not button.enabled:
        fill = BUTTON_DISABLED
        color = MUTED
    elif button.rect.collidepoint(mouse_pos):
        fill = BUTTON_HOVER
        color = TEXT
    else:
        fill = BUTTON
        color = TEXT
    pygame.draw.rect(surface, fill, button.rect, border_radius=7)
    pygame.draw.rect(surface, PANEL_BORDER, button.rect, width=1, border_radius=7)
    rendered = font.render(button.label, True, color)
    surface.blit(rendered, rendered.get_rect(center=button.rect.center))


def draw_mark(surface: pygame.Surface, coord: Coord, value: int) -> None:
    rect = coord_rect(coord)
    cx, cy = rect.center
    pad = CELL_SIZE // 4
    if value == 1:
        pygame.draw.line(
            surface,
            X_COLOR,
            (rect.left + pad, rect.top + pad),
            (rect.right - pad, rect.bottom - pad),
            width=5,
        )
        pygame.draw.line(
            surface,
            X_COLOR,
            (rect.right - pad, rect.top + pad),
            (rect.left + pad, rect.bottom - pad),
            width=5,
        )
    elif value == -1:
        pygame.draw.circle(surface, O_OUTLINE, (cx, cy), CELL_SIZE // 3, width=5)
        pygame.draw.circle(surface, O_COLOR, (cx, cy), CELL_SIZE // 3 - 2, width=2)


def draw_board(
    surface: pygame.Surface,
    state: GameState,
    fonts,
    mouse_pos: Tuple[int, int],
) -> None:
    title_font, _, small_font, _ = fonts
    draw_text(surface, title_font, "Super Tic-Tac-Toe RL", (BOARD_LEFT, 38))
    draw_text(
        surface,
        small_font,
        "Click an empty cell. Redirected moves may land adjacent or be forfeited.",
        (BOARD_LEFT, 78),
        MUTED,
    )

    hover_coord = coord_from_point(mouse_pos)
    human_turn = not state.done and state.env.current_player == state.human_player

    for level_row in range(3):
        level_mid_y = BOARD_TOP + level_row * BOARD_SIZE * CELL_SIZE + 65
        draw_text(
            surface,
            title_font,
            f"Level {level_row + 1}",
            (LEVEL_LABEL_X, level_mid_y),
            TEXT,
        )

        for level_col in range(level_row + 1):
            x0, y0 = board_origin(level_row, level_col)
            board_rect = pygame.Rect(
                x0, y0, BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE
            )
            pygame.draw.rect(surface, BOARD_FILL, board_rect)

            for local_row in range(BOARD_SIZE):
                for local_col in range(BOARD_SIZE):
                    coord = (level_row, level_col, local_row, local_col)
                    cell_rect = coord_rect(coord)
                    if state.last_actual == coord:
                        fill = BOARD_FORFEIT if state.last_forfeited else BOARD_LAST
                        pygame.draw.rect(surface, fill, cell_rect)
                    elif (
                        human_turn
                        and hover_coord == coord
                        and state.env.board.is_empty(coord)
                    ):
                        pygame.draw.rect(surface, BOARD_HOVER, cell_rect)

            for i in range(1, BOARD_SIZE):
                pygame.draw.line(
                    surface,
                    GRID,
                    (x0 + i * CELL_SIZE, y0),
                    (x0 + i * CELL_SIZE, y0 + BOARD_SIZE * CELL_SIZE),
                    width=2,
                )
                pygame.draw.line(
                    surface,
                    GRID,
                    (x0, y0 + i * CELL_SIZE),
                    (x0 + BOARD_SIZE * CELL_SIZE, y0 + i * CELL_SIZE),
                    width=2,
                )
            pygame.draw.rect(surface, BOARD_BORDER, board_rect, width=4)

            for local_row in range(BOARD_SIZE):
                for local_col in range(BOARD_SIZE):
                    coord = (level_row, level_col, local_row, local_col)
                    value = int(
                        state.env.board.grid[level_row, level_col, local_row, local_col]
                    )
                    draw_mark(surface, coord, value)

    if state.last_intended is not None and state.last_intended != state.last_actual:
        pygame.draw.rect(surface, (255, 255, 255), coord_rect(state.last_intended), width=3)


def game_status(state: GameState, model) -> Tuple[str, Tuple[int, int, int]]:
    if state.done:
        if state.winner == 0:
            return "Draw", MUTED
        if state.winner == state.human_player:
            return "You win", GREEN
        return "Agent wins", RED
    if state.env.current_player == state.human_player:
        return f"Your turn ({player_name(state.human_player)})", GREEN
    return f"Agent thinking ({player_name(state.env.current_player)})", ORANGE


def agent_label(model, use_model: bool) -> str:
    if model is None or not use_model:
        return "random"
    if isinstance(model, LoadedAgent) and model.backend == "heuristic":
        return "heuristic"
    if isinstance(model, LoadedAgent) and model.backend == "torch_dqn":
        return "DQN"
    if isinstance(model, LoadedAgent) and model.backend in {"torch_ppo", "tf_ppo"}:
        return "PPO"
    return "model"


def make_buttons(state: GameState, model) -> List[Button]:
    x = PANEL_LEFT + 22
    y = 250
    width = 250
    height = 38
    gap = 12
    return [
        Button(pygame.Rect(x, y, width, height), "New game", "new"),
        Button(
            pygame.Rect(x, y + (height + gap), width, height),
            f"Play as {player_name(state.human_player)}",
            "switch_side",
        ),
        Button(
            pygame.Rect(x, y + 2 * (height + gap), width, height),
            f"Agent: {agent_label(model, state.use_model)}",
            "toggle_model",
            enabled=model is not None,
        ),
        Button(
            pygame.Rect(x, y + 3 * (height + gap), width, height),
            f"Policy: {'greedy' if state.deterministic else 'sampling'}",
            "toggle_policy",
            enabled=model is not None and state.use_model,
        ),
    ]


def draw_panel(
    surface: pygame.Surface,
    state: GameState,
    model,
    fonts,
    mouse_pos: Tuple[int, int],
) -> List[Button]:
    _, body_font, small_font, mono_font = fonts
    panel_rect = pygame.Rect(PANEL_LEFT, 28, 294, WINDOW_HEIGHT - 56)
    pygame.draw.rect(surface, PANEL_BG, panel_rect, border_radius=8)
    pygame.draw.rect(surface, PANEL_BORDER, panel_rect, width=1, border_radius=8)

    status, color = game_status(state, model)
    draw_text(surface, body_font, status, (PANEL_LEFT + 22, 54), color)
    if isinstance(model, LoadedAgent) and model.backend == "heuristic" and state.use_model:
        model_text = "Heuristic opponent"
    elif model is not None and state.use_model:
        model_text = "Loaded checkpoint"
    elif model is not None:
        model_text = "Agent toggled to random"
    else:
        model_text = "No checkpoint; random agent"
    draw_text(
        surface,
        small_font,
        model_text,
        (PANEL_LEFT + 22, 92),
        GREEN if model is not None else ORANGE,
    )
    draw_text(
        surface,
        small_font,
        f"Human: {player_name(state.human_player)}",
        (PANEL_LEFT + 22, 126),
        TEXT,
    )
    draw_text(
        surface,
        small_font,
        f"Current: {player_name(state.env.current_player)}",
        (PANEL_LEFT + 22, 154),
        TEXT,
    )

    buttons = make_buttons(state, model)
    for button in buttons:
        draw_button(surface, body_font, button, mouse_pos)

    draw_text(surface, body_font, "Move log", (PANEL_LEFT + 22, 474), TEXT)
    if state.messages:
        draw_wrapped_lines(
            surface,
            small_font,
            state.messages[:5],
            (PANEL_LEFT + 22, 508),
            MUTED,
            line_gap=7,
        )
    else:
        draw_text(surface, small_font, "No moves yet.", (PANEL_LEFT + 22, 508), MUTED)

    draw_text(surface, mono_font, "N new  S side  G greedy", (PANEL_LEFT + 22, 636), MUTED)
    return buttons


def draw_screen(
    surface: pygame.Surface,
    state: GameState,
    model,
    fonts,
    mouse_pos: Tuple[int, int],
) -> List[Button]:
    surface.fill(BG)
    draw_board(surface, state, fonts, mouse_pos)
    buttons = draw_panel(surface, state, model, fonts, mouse_pos)
    pygame.display.flip()
    return buttons


def handle_button(state: GameState, button: Button) -> None:
    if button.action == "new":
        reset_state(state)
    elif button.action == "switch_side":
        state.human_player *= -1
        reset_state(state)
    elif button.action == "toggle_model":
        state.use_model = not state.use_model
    elif button.action == "toggle_policy":
        state.deterministic = not state.deterministic


def handle_board_click(state: GameState, pos: Tuple[int, int]) -> None:
    if state.done or state.env.current_player != state.human_player:
        return
    coord = coord_from_point(pos)
    if coord is None or not state.env.board.is_empty(coord):
        return
    action = state.env.board.coord_to_action(coord)
    apply_action(state, action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Super Tic-Tac-Toe in Pygame.")
    parser.add_argument(
        "--agent",
        choices=["model", "heuristic", "random"],
        default="model",
        help="Opponent type. Use heuristic to play against the rule-based baseline.",
    )
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--human-player", choices=["X", "O"], default="X")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-agent", action="store_true")
    parser.add_argument("--sampling-agent", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pygame.init()
    pygame.display.set_caption("Super Tic-Tac-Toe RL")
    surface = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("arial", 28, bold=True),
        pygame.font.SysFont("arial", 21, bold=True),
        pygame.font.SysFont("arial", 17),
        pygame.font.SysFont("menlo", 14),
    )

    model = None
    agent_kind = "random" if args.random_agent else args.agent
    if agent_kind == "heuristic":
        model = LoadedAgent("heuristic", HeuristicAgent(seed=args.seed), None)
    elif agent_kind == "model":
        model = load_agent(args.model_path, args.hidden_size, args.device)
    device = resolve_tf_device(args.device)

    state = GameState(
        env=SuperTicTacToeEnv(seed=args.seed),
        human_player=1 if args.human_player == "X" else -1,
        deterministic=not args.sampling_agent,
        use_model=model is not None and agent_kind != "random",
        rng=np.random.default_rng(args.seed),
    )
    state.env.reset(seed=args.seed)

    running = True
    buttons: List[Button] = []
    agent_delay_ms = 220
    next_agent_time = pygame.time.get_ticks() + agent_delay_ms

    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_n:
                    reset_state(state)
                elif event.key == pygame.K_s:
                    state.human_player *= -1
                    reset_state(state)
                elif event.key == pygame.K_g and model is not None and state.use_model:
                    state.deterministic = not state.deterministic
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked_button = next(
                    (button for button in buttons if button.enabled and button.rect.collidepoint(event.pos)),
                    None,
                )
                if clicked_button is not None:
                    handle_button(state, clicked_button)
                else:
                    handle_board_click(state, event.pos)
                next_agent_time = pygame.time.get_ticks() + agent_delay_ms

        if (
            not state.done
            and state.env.current_player != state.human_player
            and pygame.time.get_ticks() >= next_agent_time
        ):
            agent_turn(state, model, device)
            next_agent_time = pygame.time.get_ticks() + agent_delay_ms

        buttons = draw_screen(surface, state, model, fonts, mouse_pos)
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
