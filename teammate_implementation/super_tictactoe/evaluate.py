import torch
import numpy as np
from typing import Dict
from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.model import ActorCritic


def evaluate(
    model1: ActorCritic,
    model2: ActorCritic,
    num_games: int = 100,
    device: str = 'cpu',
) -> Dict[str, float]:
    """
    Play num_games between model1 (P1) and model2 (P2).
    Returns win rates: {'model1_wins': float, 'model2_wins': float, 'draws': float}
    """
    env = SuperTicTacToeEnv()
    counts = {'model1_wins': 0, 'model2_wins': 0, 'draws': 0}

    for _ in range(num_games):
        state = env.reset()
        while not env.done:
            model = model1 if env.current_player == 1 else model2
            action_mask = torch.BoolTensor(env.get_action_mask()).to(device)
            state_tensor = torch.FloatTensor(state).to(device)
            with torch.no_grad():
                action, _, _ = model.get_action(state_tensor, action_mask, deterministic=True)
            state, _, _, _ = env.step(action)

        if env.winner == 1:
            counts['model1_wins'] += 1
        elif env.winner == 2:
            counts['model2_wins'] += 1
        else:
            counts['draws'] += 1

    return {k: v / num_games for k, v in counts.items()}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, required=True)
    parser.add_argument('--model2', type=str, required=True)
    parser.add_argument('--games', type=int, default=100)
    args = parser.parse_args()

    m1 = ActorCritic()
    m1.load_state_dict(torch.load(args.model1, map_location='cpu'))
    m2 = ActorCritic()
    m2.load_state_dict(torch.load(args.model2, map_location='cpu'))

    results = evaluate(m1, m2, num_games=args.games)
    print(f"Model1 wins: {results['model1_wins']:.1%}")
    print(f"Model2 wins: {results['model2_wins']:.1%}")
    print(f"Draws:       {results['draws']:.1%}")
