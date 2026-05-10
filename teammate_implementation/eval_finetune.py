import torch
from super_tictactoe.model import ActorCritic
from compare import play_games, greedy_agent, blocking_agent, safe_agent, random_agent

N = 200

def load(path):
    m = ActorCritic()
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m

def make_agent(model):
    def fn(env):
        s    = torch.FloatTensor(env._get_state())
        mask = torch.BoolTensor(env.get_action_mask())
        with torch.no_grad():
            a, _, _ = model.get_action(s, mask)
        return a
    return fn

models = {
    'PPO-curriculum':  load('checkpoints_ppo_cl/model_final.pt'),
    'PPO-finetune200': load('checkpoints_ppo_finetune/model_final.pt'),
    'PPO-ft3-50':      load('checkpoints_ppo_finetune2/model_0050.pt'),
    'PPO-ft3-100':     load('checkpoints_ppo_finetune2/model_0100.pt'),
    'PPO-ft3-150':     load('checkpoints_ppo_finetune2/model_0150.pt'),
    'PPO-ft3-200':     load('checkpoints_ppo_finetune2/model_final.pt'),
}

heuristics = [
    ('Random',   random_agent),
    ('Greedy',   greedy_agent),
    ('Blocking', blocking_agent),
    ('Safe',     safe_agent),
]

header = f"{'Agent':<18} {'Random':>8} {'Greedy':>8} {'Blocking':>10} {'Safe':>8}"
print(header)
print('-' * len(header))

for name, m in models.items():
    ag = make_agent(m)
    results = []
    for hname, h in heuristics:
        p1,  p2,  d,  _ = play_games(ag, h, N // 2)
        p1b, p2b, db, _ = play_games(h,  ag, N // 2)
        win = (p1 + p2b) / 2  # ag wins as P1 + ag wins as P2
        results.append(win)
    row = f"{name:<18} {results[0]:>7.0%} {results[1]:>8.0%} {results[2]:>10.0%} {results[3]:>8.0%}"
    print(row)
