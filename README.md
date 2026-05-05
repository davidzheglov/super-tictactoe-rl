# Super Tic-Tac-Toe RL Agent

This project implements the assignment game, trains a self-play reinforcement
learning agent, and provides a Pygame UI for playing against the trained agent.

## Board Interpretation

The assignment describes a triangular board made of six 4x4 square boards. This
project stores the board in a NumPy array with shape `(3, 3, 4, 4)` and treats
only these level positions as playable:

- Level 1: `(0, 0)`
- Level 2: `(1, 0)`, `(1, 1)`
- Level 3: `(2, 0)`, `(2, 1)`, `(2, 2)`

Each playable cell has coordinate:

```text
(level_row, level_col, local_row, local_col)
```

Actions are flattened from `0` to `95` over the 96 playable cells in level order.

The Pygame UI draws the boards as a pyramid on a 12-by-12 cell canvas:

- Level 1: one 4x4 board centered after four empty cell widths.
- Level 2: two 4x4 boards under it, offset by two empty cell widths.
- Level 3: three 4x4 boards across the bottom.

## Winning Rules Implemented

Win detection uses the same centered pyramid coordinates shown in the Pygame UI:

```text
global_r = level_row * 4 + local_row
global_c = (2 - level_row) * 2 + level_col * 4 + local_col
```

- 4 in a horizontal row across the visible pyramid. This can be entirely inside
  one local 4x4 board.
- 4 in a vertical column across the visible pyramid, spanning at least two
  level rows.
- 5 across a visible pyramid diagonal in direction down-right or down-left.

A complete row inside one 4x4 board is a win. A complete column inside only one
4x4 board is not counted as a win because the assignment adds the different-level
condition only to columns.

## Setup

Python 3.9, 3.10, or 3.11 is recommended for the TensorFlow/TF-Agents pins.

```bash
cd super_tictactoe_rl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Apple Silicon, TensorFlow can use the Metal backend if your local TensorFlow
installation supports it. CPU training also works.

## Run Tests

```bash
python tests.py
```

## Train

The trainer uses TensorFlow plus a TF-Agents `PyEnvironment` integration. The
policy/value update is custom PPO-style self-play so action masks and final
winner/loser rewards are handled cleanly. The scripts set
`TF_USE_LEGACY_KERAS=1` automatically because TF-Agents 0.19 depends on the
legacy Keras compatibility path.

```bash
python train.py --episodes 5000 --lr 3e-4 --device cpu
```

Useful faster smoke test:

```bash
python train.py --episodes 20 --batch-episodes 4 --device cpu
```

The checkpoint is saved as a TensorFlow checkpoint prefix:

```text
models/super_ttt_agent.pt
models/super_ttt_agent.pt.index
models/super_ttt_agent.pt.data-00000-of-00001
models/super_ttt_agent.pt.json
```

## Evaluate

```bash
python evaluate.py --games 100 --deterministic --device cpu
```

The model alternates between playing X and O against a random legal opponent.

## Play in Pygame

```bash
python app.py
```

If no trained checkpoint exists, the UI still works and the computer plays
random legal moves.

Useful options:

```bash
python app.py --human-player O
python app.py --model-path models/super_ttt_agent.pt --sampling-agent
python app.py --random-agent
```

Keyboard shortcuts inside the window:

- `N`: new game
- `S`: switch side
- `G`: toggle greedy/sampling policy
- `Esc` or `Q`: quit

## Files

- `board.py`: pure NumPy board, stochastic move resolution, win checks.
- `env.py`: Gymnasium environment and TF-Agents `PyEnvironment`.
- `models.py`: TensorFlow/Keras policy-value network and masked action sampling.
- `train.py`: PPO-style self-play training loop.
- `evaluate.py`: model evaluation against random play.
- `app.py`: Pygame human-vs-agent UI.
- `utils.py`: shared checkpoint, seeding, and device helpers.
- `tests.py`: unit tests for board rules and environment behavior.
