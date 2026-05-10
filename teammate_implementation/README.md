# Teammate's Super Tic-Tac-Toe Implementation

This folder contains a collaborator's independent implementation of the same assignment, trained using a **CNN Actor-Critic architecture** and **AlphaZero with MCTS** on the HKUST SuperPOD GPU cluster.

Full source repository: https://github.com/Anson-1/tic-tac-toe

---

## What's Included Here

```
teammate_implementation/
├── super_tictactoe/           # Core package
│   ├── env.py                 # 12×12 board environment
│   ├── model.py               # CNN Actor-Critic (3×12×12 input)
│   ├── train.py               # PPO training loop
│   ├── ppo.py                 # GAE + PPO update
│   ├── selfplay.py            # Vectorized episode collection
│   ├── mcts.py                # Monte Carlo Tree Search
│   ├── heuristics.py          # 6 heuristic opponents
│   └── gui.py                 # Pygame UI
├── checkpoints/
│   ├── ppo_curriculum_final.pt   # Best PPO (curriculum, 3000 updates)
│   ├── ppo_finetuned_final.pt    # Finetuned variant (best vs heuristics)
│   └── alphazero_best.pt         # AlphaZero peak (iteration 170, 67% vs reference)
├── generated_results/
│   ├── fresh_agent_matchups.csv          # Full benchmark results (36 matchups)
│   ├── fresh_agent_vs_heuristics.png     # Win-rate bar chart
│   └── fresh_winrate_heatmap.png         # Heatmap across agents and opponents
├── compare.py                 # Round-robin benchmark + Elo ratings
├── eval_ppo_progress.py       # PPO checkpoint progression analysis
├── eval_az.py                 # AlphaZero evaluation
├── eval_finetune.py           # Finetune variant comparison
└── plot_winrate.py            # Training progression plots
```

---

## Play Against These Agents

```bash
# Install dependencies
pip install -r requirements.txt

# Human vs PPO curriculum (best overall agent)
python -c "
from super_tictactoe.gui import run_human_vs_agent
run_human_vs_agent('checkpoints/ppo_curriculum_final.pt', device='cpu')
"

# Human vs finetuned PPO (strongest vs heuristics)
python -c "
from super_tictactoe.gui import run_human_vs_agent
run_human_vs_agent('checkpoints/ppo_finetuned_final.pt', device='cpu')
"

# Human vs AlphaZero (MCTS-guided, most strategic)
python -c "
from super_tictactoe.gui import run_human_vs_agent
run_human_vs_agent('checkpoints/alphazero_best.pt', device='cpu')
"
```

---

## Benchmark Results (from `generated_results/`)

Key results from `fresh_agent_matchups.csv` (80 games per matchup):

| Agent | vs Random | vs Blocking | vs Counter | vs Stronger |
|---|--:|--:|--:|--:|
| PPO finetuned | **100%** | 53.75% | 68.75% | 27.5% |
| PPO curriculum | ~90% | 48.75% | ~60% | ~27% |
| PPO base | ~85% | ~45% | ~55% | ~25% |

The `stronger_heuristic` (full stochastic EV scoring with fork detection) beats all PPO agents ~72% of the time — consistent with findings in the main repo.

---

## Architecture Differences vs Main Repo

| Aspect | This folder (Teammate) | Main repo |
|---|---|---|
| Board representation | 12×12 CNN grid | 4D tensor (hierarchical) |
| Action space | 144 actions (flat grid) | 96 actions |
| Model | CNN (conv → FC → actor/critic) | FC policy-value net |
| Extra algorithm | AlphaZero + MCTS | Behavioral cloning |
| Training scale | SuperPOD cluster (multi-GPU) | Single GPU / remote server |
| Opponent curriculum | 6-tier heuristic pool | Mixed: heuristic+line+self+random |

Despite different representations, both approaches implement identical game rules.

---

## Run Evaluation

```bash
# Full round-robin + Elo ratings across all agent variants
python compare.py

# PPO checkpoint progression vs greedy/blocking/random
python eval_ppo_progress.py

# AlphaZero evaluation (policy-only vs policy+MCTS)
python eval_az.py
```
