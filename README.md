# Super Tic-Tac-Toe RL

Reinforcement learning research project training agents to play **Super Tic-Tac-Toe** — a stochastic 3D board game with 6 levels of 4×4 boards arranged as a triangular pyramid. Win by getting 4-in-a-row horizontally, 4-in-a-column spanning multiple levels, or 5-across diagonally. Every move has only a **50% chance** of landing at the intended cell; otherwise it drifts to a random adjacent cell.

**TorchRL framework is used for the final PPO training** — bonus implementation (1.5× score multiplier, capped at 50%).

---

## Quick Start

```bash
pip install -r requirements.txt

# Play against the best trained agent (DetPPO — 300k episodes, 63.5% vs smart heuristic)
python -m app --model-path runs/super_ttt_agent_torchrl_detppo.pt

# Play against the stochastic PPO variant (mixed-opponent curriculum)
python -m app --model-path runs/super_ttt_agent_torchrl_ppo.pt
```

### Controls

| Key / Action | Effect |
|---|---|
| Click a cell | Place your piece |
| `S` | Toggle simulation mode (agent vs agent auto-play) |
| `R` | Restart game |
| `Space` | Pause / resume simulation |

---

## Play Against Different Opponents

### Rule-based opponents

```bash
# Random agent — easiest, useful as sanity check
python -m app --random-agent

# Basic heuristic — wins immediately, blocks immediate threats, else random
python -m app --agent basic

# Line-builder — aggressively constructs long lines, minimal defense
python -m app --agent line

# Smart heuristic — stochastic expected-value scoring across all landing cells
#   (strongest rule-based opponent; used as primary training target)
python -m app --agent heuristic
```

### Trained RL agents (this repo)

```bash
# DetPPO: 300k episodes, deterministic placement mode, pure heuristic opponent
#   Win rates: 63.5% vs smart heuristic, 73.5% vs basic
python -m app --model-path runs/super_ttt_agent_torchrl_detppo.pt

# PPO: 300k episodes, stochastic placement, mixed opponent curriculum
#   Win rates: ~48% vs smart heuristic (balanced, handles diverse opponents)
python -m app --model-path runs/super_ttt_agent_torchrl_ppo.pt

# Play as O instead of X (default is X)
python -m app --model-path runs/super_ttt_agent_torchrl_detppo.pt --human-player O

# Sampling mode: agent draws from policy distribution rather than argmax
python -m app --model-path runs/super_ttt_agent_torchrl_ppo.pt --sampling-agent
```

### Teammate's agents (CNN + AlphaZero implementation)

```bash
cd teammate_implementation
pip install -r requirements.txt

# Human vs PPO curriculum agent (best overall)
python -c "from super_tictactoe.gui import run_human_vs_agent; run_human_vs_agent('checkpoints/ppo_curriculum_final.pt')"

# Human vs finetuned PPO (highest win rate vs heuristics)
python -c "from super_tictactoe.gui import run_human_vs_agent; run_human_vs_agent('checkpoints/ppo_finetuned_final.pt')"

# Human vs AlphaZero (MCTS-guided)
python -c "from super_tictactoe.gui import run_human_vs_agent; run_human_vs_agent('checkpoints/alphazero_best.pt')"
```

> Full teammate source: https://github.com/Anson-1/tic-tac-toe

---

## Auto-Play: Agent vs Agent

```bash
# DetPPO vs smart heuristic (watch the trained agent play)
python -m app --simulate \
  --sim-x ppo --sim-x-model-path runs/super_ttt_agent_torchrl_detppo.pt \
  --sim-o heuristic

# Stochastic PPO vs DetPPO
python -m app --simulate \
  --sim-x ppo --sim-x-model-path runs/super_ttt_agent_torchrl_ppo.pt \
  --sim-o ppo --sim-o-model-path runs/super_ttt_agent_torchrl_detppo.pt
```

---

## Benchmark Results (200 games/matchup)

| Agent | vs Smart Heuristic | vs Line Builder | vs Basic Heuristic |
|---|--:|--:|--:|
| **DetPPO (300k, TorchRL)** | **63.5%** | 41.5% | **73.5%** |
| PPO (300k, TorchRL) | 37% | — | — |
| BC pretrain only (0 PPO steps) | 45% | 44% | — |
| DQN (6k episodes, sparse) | 0% | — | — |
| PPO (6k episodes, sparse) | 6% | — | — |

The jump from 6% → 45% (sparse PPO → BC pretrain) demonstrates the critical value of behavioral cloning warm-start for sparse-reward board games.

Run benchmarks yourself:

```bash
python -m benchmark --agents ppo,heuristic,line,basic \
  --ppo-path runs/super_ttt_agent_torchrl_detppo.pt --deterministic \
  --games 200 --output-dir runs/benchmarks_detppo
```

---

## Training Pipeline

### Phase 1 — Baselines (sparse reward)

```bash
python -m train_qlearning --episodes 15000   # Tabular Q-learning
python -m train_torch_dqn --episodes 6000    # Deep Q-Network
python -m train_torch_ppo --episodes 6000    # PPO, no curriculum
```

**Result:** 6k episodes insufficient for sparse reward. PPO achieves 6%, DQN 0% vs smart heuristic.

### Phase 2 — Full research run (BC + TorchRL PPO, 300k episodes)

Three anti-sparse-reward techniques applied simultaneously:
1. **Behavioral cloning** warm-start from heuristic demonstrations
2. **Mixed-opponent curriculum** (65% heuristic, 20% line, 10% self, 5% random)
3. **Mid-game state starts** (positions sampled 4–18 ply into a game)

```bash
# Step 1: Behavior cloning
python -m train_behavior_clone --samples 200000 --epochs 8

# Step 2a: TorchRL stochastic PPO (mixed curriculum)
python -m train_torch_ppo --episodes 300000 --placement-mode stochastic

# Step 2b: TorchRL deterministic PPO (pure heuristic opponent)
python -m train_torch_ppo --episodes 300000 --placement-mode deterministic
```

---

## Generate Analysis Figures

```bash
python generate_report_figures.py   # Writes figures/ directory (8 plots)
python -m analyze_training --run-dir runs/research_bc_ppo_300k --output-dir runs/figures/research
```

---

## Project Structure

```
super_tictactoe_rl/
├── board.py                      # Board state, win detection, stochastic placement
├── env.py                        # Gymnasium-compatible environment
├── torchrl_env.py                # TorchRL wrapper + action masking  ← BONUS
├── agents.py                     # All agent types (random, heuristics, RL)
├── torch_models.py               # Shared policy-value network architecture
├── train_qlearning.py            # Phase 1: tabular Q-learning
├── train_torch_dqn.py            # Phase 1: DQN
├── train_torch_ppo.py            # Phase 2: PPO (vectorized rollout, TorchRL)
├── train_behavior_clone.py       # Phase 2: BC warm-start
├── app.py                        # Pygame interactive UI
├── benchmark.py                  # Head-to-head benchmark runner
├── evaluate.py                   # Evaluation utilities
├── analyze_training.py           # Learning curve generation
├── generate_report_figures.py    # All 8 report figures
├── runs/
│   ├── super_ttt_agent_torchrl_ppo.pt     # Best stochastic PPO model
│   ├── super_ttt_agent_torchrl_detppo.pt  # Best DetPPO model
│   ├── heur_line_torch/           # Baseline sparse-reward run logs
│   ├── overnight_mixed_torch/     # Extended run: 75k Q-learning, 8k DQN
│   └── research_bc_ppo_300k/      # Full BC + 300k PPO research run
├── figures/                       # Generated report figures (01–08)
├── teammate_implementation/       # Collaborator's CNN + AlphaZero approach
├── LITERATURE_REVIEW.md           # Literature review with academic citations
├── HEURISTICS_AND_REWARD.md       # Heuristic design rationale
└── research_process/              # Cluster configs, deployment scripts, archive
```

---

## Dependencies

Python 3.9+, PyTorch 2.x, TorchRL, Pygame 2.x, NumPy, Pandas, Matplotlib.

```bash
pip install -r requirements.txt
```
