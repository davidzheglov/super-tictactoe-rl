"""Quick evaluation of AZ-curriculum vs heuristic opponents."""
import numpy as np
import torch
from super_tictactoe.model import ActorCritic
from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.mcts import MCTS
from compare import (play_games, greedy_agent, blocking_agent,
                     safe_agent, one_step_lookahead_agent, random_agent)

N_GAMES = 200


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


def mcts_agent(model, simulations=50):
    mcts = MCTS(model, device='cpu', num_simulations=simulations)
    def agent(env):
        return mcts.get_action(env)
    return agent


m = load('checkpoints_az_cl/model_best.pt')
agents = [
    ('AZ-curriculum',       model_agent(m)),
    ('AZ-curriculum+MCTS',  mcts_agent(m, 50)),
]

heuristics = [
    ('Random',         random_agent),
    ('Greedy',         greedy_agent),
    ('Blocking',       blocking_agent),
    ('Safe',           safe_agent),
    ('1-step lookahead', one_step_lookahead_agent),
]

print(f"{'Agent':<22} {'Opponent':<18} {'win':>5}  {'loss':>5}")
print("-" * 55)

for name, agent in agents:
    for h_name, h_agent in heuristics:
        p1,  p2,  d,  _ = play_games(agent,   h_agent, N_GAMES // 2)
        p1b, p2b, db, _ = play_games(h_agent, agent,   N_GAMES // 2)
        win  = (p1 + p1b) / 2
        loss = (p2 + p2b) / 2
        print(f"{name:<22} {h_name:<18} {win:>4.0%}   {loss:>4.0%}")
    print()
