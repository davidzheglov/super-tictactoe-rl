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

Python 3.9+ works for the PyTorch/TorchRL trainers. The normal install is
Torch-only and uses TorchRL for the assignment bonus path.

```bash
cd super_tictactoe_rl
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On the remote Ubuntu GPU server, use `setup_remote_gpu.sh`; it installs a
CUDA-enabled PyTorch wheel from the official PyTorch CUDA index.

## Run Tests

```bash
python tests.py
```

## Train

The recommended bonus-path trainer is TorchRL PPO. The project also keeps a
plain PyTorch PPO baseline and a PyTorch DQN baseline. All neural trainers use
legal-action masking and save resumable `.pt` checkpoints. The one-command
remote runner uses mixed opponent training by default:

- 20% self-play
- 45% smart heuristic opponent
- 30% line-builder opponent
- 5% random opponent

The smart heuristic scores all true winning windows and the stochastic landing
distribution of each move, so it blocks intersections such as horizontal
two-in-a-row plus vertical/diagonal threats. The line-builder is the aggressive
baseline that mostly extends its own longest open line. See
`HEURISTICS_AND_REWARD.md` for the exact rules and sparse-reward shaping.

```bash
python train_torchrl_ppo.py --episodes 5000 --lr 3e-4 --device cuda
python train_torch_dqn.py --episodes 5000 --lr 3e-4 --device cuda
```

Train directly against the heuristic baseline:

```bash
python train_torchrl_ppo.py --episodes 5000 --opponent heuristic --device cuda
python train_torch_dqn.py --episodes 5000 --opponent heuristic --device cuda
python train_torchrl_ppo.py --episodes 5000 --opponent line --device cuda
```

Useful faster smoke test:

```bash
python train_torchrl_ppo.py --episodes 20 --batch-episodes 4 --device cpu
```

The main PPO checkpoint is saved as:

```text
models/super_ttt_agent_torchrl.pt
models/super_ttt_agent_torchrl.pt.json
```

`train_torchrl_ppo.py` validates the TorchRL action-masked environment wrapper
before training and labels checkpoints as `torchrl_ppo`.

## Evaluate

```bash
python evaluate.py --games 100 --deterministic --device cpu
```

The model alternates between playing X and O against a random legal opponent.

For cross-play benchmarking:

```bash
python benchmark.py --agents random,basic,line,heuristic --games 100
python benchmark.py --agents random,basic,line,heuristic,ppo,dqn --games 100 \
  --ppo-path runs/overnight_torch/ppo_seed0/super_ttt_agent_torchrl.pt \
  --dqn-path runs/overnight_torch/dqn_seed0/dqn_agent_torch.pt
```

Plot logs and benchmark outputs:

```bash
python analyze_training.py --run-dir runs/overnight_torch
```

## Play in Pygame

```bash
python app.py
```

If no trained checkpoint exists, the UI still works and the computer plays
random legal moves.

Useful options:

```bash
python app.py --agent heuristic
python app.py --agent line
python app.py --human-player O
python app.py --model-path models/super_ttt_agent_torchrl.pt --sampling-agent
python app.py --random-agent
```

Keyboard shortcuts inside the window:

- `N`: new game
- `S`: switch side
- `G`: toggle greedy/sampling policy
- `Esc` or `Q`: quit

## Files

- `board.py`: pure NumPy board, stochastic move resolution, win checks.
- `env.py`: Gymnasium environment.
- `torchrl_env.py`: TorchRL/GymWrapper environment with legal-action masks.
- `torch_models.py`: PyTorch policy/value and DQN networks.
- `train_torchrl_ppo.py`: TorchRL bonus-path PPO entrypoint.
- `train_torch_ppo.py`: PyTorch PPO-style self-play training loop.
- `train_torch_dqn.py`: PyTorch DQN baseline.
- `agents.py`: random, smart heuristic, line-builder, Q-table, PPO, and DQN agents.
- `benchmark.py`: pairwise cross-play benchmarks and CSV output.
- `analyze_training.py`: training and benchmark plots for reports.
- `evaluate.py`: model evaluation against random play.
- `app.py`: Pygame human-vs-agent UI.
- `utils.py`: shared checkpoint, seeding, and device helpers.
- `tests.py`: unit tests for board rules and environment behavior.
