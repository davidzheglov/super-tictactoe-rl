# Reference Tic-Tac-Toe Training Parameters

This records the main parameters found in the ignored `tic-tac-toe/` reference
folder.

## PPO Architecture

File: `tic-tac-toe/super_tictactoe/model.py`

- Input state: `(3, 12, 12)` channels.
- Network:
  - Conv2d `3 -> 32`, kernel `3`, padding `1`;
  - Conv2d `32 -> 64`, kernel `3`, padding `1`;
  - flatten;
  - dense layer to `256`;
  - actor head to `144` actions;
  - critic head to scalar value.

## PPO Defaults

Files:

- `tic-tac-toe/super_tictactoe/train.py`
- `tic-tac-toe/super_tictactoe/ppo.py`

Defaults:

- updates: `3000`
- episodes per update: `512`
- total rollout games for a standard long PPO run: `3000 * 512 = 1,536,000`
- optimizer: Adam
- base learning rate: `3e-4`
- learning-rate schedule: linear decay by 90%, from `3e-4` to about `3e-5`
- PPO epochs per update: `4`
- PPO clip epsilon: `0.2`
- entropy coefficient: `0.05`
- value coefficient: `0.5`
- discount factor: `gamma = 0.99`
- GAE lambda: `0.95`
- gradient clipping: `0.5`
- checkpoint save interval: every `100` updates
- opponent pool size: `10`
- common opponent pool probability: `0.5`

## PPO Curriculum

File: `tic-tac-toe/super_tictactoe/train.py`

The curriculum changes the move success rate over training:

- first third: `success_rate = 1.0`, deterministic placement;
- second third: `success_rate = 0.8`, mild stochasticity;
- final third: `success_rate = 0.5`, full assignment stochasticity.

This is important: the reference project did not start directly in the hardest
stochastic environment. It first learned clean tactics, then added noise.

## PPO Job Scripts

Files:

- `tic-tac-toe/job.sh`
- `tic-tac-toe/job_ppo_curriculum.sh`
- `tic-tac-toe/job_ppo_heuristic.sh`
- `tic-tac-toe/job_ppo_phase1.sh`
- `tic-tac-toe/job_ppo_phase2.sh`

Observed jobs:

| Script | Updates | Episodes/Update | Total Games | Opponent Setup | LR |
| --- | ---: | ---: | ---: | --- | ---: |
| `job.sh` | 3000 | 512 | 1,536,000 | 50% checkpoint pool, else self-play | `3e-4` |
| `job_ppo_curriculum.sh` | 3000 | 512 | 1,536,000 | 50% checkpoint pool, curriculum | `3e-4` |
| `job_ppo_heuristic.sh` | 3000 | 512 | 1,536,000 | 40% pool, 30% heuristic, curriculum | `3e-4` |
| `job_ppo_phase1.sh` | 1000 | 512 | 512,000 | 10% pool, 80% heuristic, curriculum | `3e-4` |
| `job_ppo_phase2.sh` | 2000 | 512 | 1,024,000 | resume phase 1, 60% pool, 20% heuristic, curriculum | `1e-4` |
| `job_ppo_finetune.sh` | 1000 | 512 | 512,000 | resume curriculum, 10% pool, 80% heuristic | `1e-4` |
| `job_ppo_finetune2.sh` | 200 | 128 | 25,600 | CPU fine-tune, 10% pool, 80% heuristic | `1e-4` |

## Reward Shaping

File: `tic-tac-toe/super_tictactoe/env.py`

The reference project uses potential-based reward shaping:

- `shaping_gamma = 0.99`
- `defense_weight = 1.5`
- potential = own line potential minus `1.5 * opponent line potential`
- reward adds `gamma * Phi(next_state) - Phi(current_state)`
- fork bonus: `0.05 * min(n_threats - 1, 3)` when a move creates multiple threats.

File: `tic-tac-toe/super_tictactoe/selfplay.py`

When training against a heuristic opponent, it adds a threat-growth penalty:

- `THREAT_PENALTY_COEF = 0.3`
- if the opponent grows its board potential, the learner's next recorded reward
  is penalized.

## Heuristic Opponent Pool

File: `tic-tac-toe/super_tictactoe/heuristics.py`

Weighted random heuristic pool:

- greedy offensive agent: `0.10`
- immediate blocking agent: `0.25`
- safe/low-forfeit agent: `0.30`
- counter heuristic: `0.35`

The counter heuristic blocks the opponent's most advanced line, not only
immediate wins.

## AlphaZero-Style Parameters

Files:

- `tic-tac-toe/super_tictactoe/alphazero_train.py`
- `tic-tac-toe/job_alphazero.sh`
- `tic-tac-toe/job_az_curriculum.sh`

Defaults and jobs:

- iterations: `200`
- games per iteration: `100`
- total self-play games: `20,000`
- MCTS simulations per move: `50`
- train epochs per iteration: `10`
- batch size: `512`
- replay buffer size: `50,000`
- optimizer: Adam
- learning rate: `1e-3`
- weight decay: `1e-4`
- save every `10` iterations
- evaluate every `10` iterations
- optional same stochasticity curriculum as PPO.

## Takeaway For Our Project

The biggest difference is scale and curriculum. The reference PPO runs use over
one million rollout games and curriculum stochasticity, while our first negative
baseline had only 6,000 PPO episodes. The second-stage plan should therefore be
treated as a first serious run, not an excessive run.
