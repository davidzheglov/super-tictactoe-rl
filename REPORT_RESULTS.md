# Report Results Log

This file records benchmark numbers that are useful for the final report.

## 2026-05-07 Local Checkpoint Benchmark

Run directory:

```text
runs/heur_line_torch
```

Training completed checkpoints:

- PPO/TorchRL checkpoint: `runs/heur_line_torch/ppo_seed0/super_ttt_agent_torchrl.pt`
- DQN checkpoint: `runs/heur_line_torch/dqn_seed0/dqn_agent_torch.pt`
- Q-learning table: `runs/heur_line_torch/q_learning_seed0/q_table.pkl`

Benchmark protocol:

- 100 games per matchup.
- Starting player alternated each game.
- Device: CPU.
- PPO evaluated deterministically.
- Opponent: smart heuristic.

Commands:

```bash
.venv/bin/python benchmark.py \
  --agents heuristic,ppo \
  --games 100 \
  --device cpu \
  --deterministic \
  --ppo-path runs/heur_line_torch/ppo_seed0/super_ttt_agent_torchrl.pt \
  --output-dir runs/heur_line_torch/benchmarks_ppo_vs_heuristic_100

.venv/bin/python benchmark.py \
  --agents heuristic,dqn \
  --games 100 \
  --device cpu \
  --dqn-path runs/heur_line_torch/dqn_seed0/dqn_agent_torch.pt \
  --output-dir runs/heur_line_torch/benchmarks_dqn_vs_heuristic_100
```

Results:

| Matchup | Smart Heuristic Wins | Agent Wins | Draws | Agent Win Rate |
| --- | ---: | ---: | ---: | ---: |
| Smart heuristic vs PPO/TorchRL | 94 | 6 | 0 | 6% |
| Smart heuristic vs DQN | 100 | 0 | 0 | 0% |

Interpretation:

The first completed neural checkpoints are useful as a negative baseline:
the current 6,000-episode neural runs do not yet beat the smart heuristic.
This supports the report story that direct sparse-reward RL is not sufficient
against a strong tactical hand-coded opponent without curriculum, imitation
warm-starting, or better state sampling.

Follow-up design:

The next planned run is documented in `RESEARCH_TRAINING_PLAN.md`. It adds
behavior cloning, mid-game state starts, much longer PPO training, numbered
checkpoint selection, and a deterministic-placement PPO ablation.
