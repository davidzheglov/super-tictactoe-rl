"""
Evaluate P1 vs P2 win rates across all checkpoints and plot.
Usage: conda run -n tictactoe python plot_winrate.py
"""
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from super_tictactoe.model import ActorCritic
from super_tictactoe.evaluate import evaluate

CHECKPOINTS = [
    (u, f'checkpoints/model_{u:04d}.pt')
    for u in range(100, 3001, 100)
]
N_GAMES = 200

updates, p1_wins, p2_wins, draws = [], [], [], []

for update, path in CHECKPOINTS:
    model = ActorCritic()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    results = evaluate(model, model, num_games=N_GAMES)
    updates.append(update)
    p1_wins.append(results['model1_wins'] * 100)
    p2_wins.append(results['model2_wins'] * 100)
    draws.append(results['draws'] * 100)
    print(f"Update {update:4d} | P1 wins: {results['model1_wins']:.0%} | "
          f"P2 wins: {results['model2_wins']:.0%} | Draws: {results['draws']:.0%}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(updates, p1_wins, 'b-o', label='Player 1 wins %', linewidth=2)
ax.plot(updates, p2_wins, 'r-o', label='Player 2 wins %', linewidth=2)
ax.plot(updates, draws,   'g-o', label='Draws %',          linewidth=2)
ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
ax.set_xlabel('Training update')
ax.set_ylabel('Win rate (%)')
ax.set_title('P1 vs P2 win rate across training (self-play, same model)')
ax.legend()
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('winrate.png', dpi=150)
print('\nSaved to winrate.png')
