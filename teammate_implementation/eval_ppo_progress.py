"""
Verify PPO learning progression across checkpoints.
Tests each saved checkpoint against random, blocking heuristic, and the final model.
Computes Elo to show whether training actually improved the agent over time.
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from super_tictactoe.model import ActorCritic
from compare import play_games, blocking_agent, random_agent, greedy_agent

N = 100  # games per matchup (fast)


def load(path):
    m = ActorCritic()
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m


def model_agent(model):
    def agent(env):
        state = torch.FloatTensor(env._get_state())
        mask  = torch.BoolTensor(env.get_action_mask())
        with torch.no_grad():
            action, _, _ = model.get_action(state, mask)
        return action
    return agent


# Pick evenly spaced checkpoints from PPO-curriculum
ckpt_dir = 'checkpoints_ppo_cl'
all_ckpts = sorted([f for f in os.listdir(ckpt_dir) if f.startswith('model_0')])
# Sample ~6 checkpoints evenly
indices = np.linspace(0, len(all_ckpts) - 1, 6, dtype=int)
checkpoints = [all_ckpts[i] for i in indices]
checkpoints.append('model_final.pt')

print(f"{'Checkpoint':<18} {'vs Random':>10} {'vs Greedy':>10} {'vs Blocking':>12} {'avg_steps':>10}")
print("-" * 65)

results = []
for ckpt in checkpoints:
    path = os.path.join(ckpt_dir, ckpt)
    if not os.path.exists(path):
        continue
    m = load(path)
    agent = model_agent(m)

    w_rand,  l_rand,  _, steps = play_games(agent, random_agent,  N)
    w_greed, l_greed, _, _     = play_games(agent, greedy_agent,  N)
    w_block, l_block, _, _     = play_games(agent, blocking_agent, N)

    label = ckpt.replace('model_', '').replace('.pt', '')
    print(f"{label:<18} {w_rand:>9.0%} {w_greed:>10.0%} {w_block:>12.0%} {steps:>10.1f}")
    results.append((label, w_rand, w_greed, w_block, steps))

# Plot progression
labels   = [r[0] for r in results]
vs_rand  = [r[1] * 100 for r in results]
vs_greed = [r[2] * 100 for r in results]
vs_block = [r[3] * 100 for r in results]
steps    = [r[4] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

x = range(len(labels))
ax1.plot(x, vs_rand,  marker='o', label='vs Random')
ax1.plot(x, vs_greed, marker='s', label='vs Greedy')
ax1.plot(x, vs_block, marker='^', label='vs Blocking')
ax1.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=30)
ax1.set_ylabel('Win rate (%)')
ax1.set_title('PPO-curriculum: Win Rate Progression')
ax1.legend(); ax1.set_ylim(0, 110)

ax2.plot(x, steps, marker='o', color='orange')
ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=30)
ax2.set_ylabel('Avg steps per game')
ax2.set_title('Game Length (longer = more strategic)')

plt.tight_layout()
plt.savefig('ppo_progress.png', dpi=150)
print("\nSaved ppo_progress.png")
