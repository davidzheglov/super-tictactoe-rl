import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from collections import deque
from super_tictactoe.model import ActorCritic
from super_tictactoe.env import SuperTicTacToeEnv
from super_tictactoe.mcts import MCTS
from super_tictactoe.evaluate import evaluate


def curriculum_rate(step: int, total: int) -> float:
    """Return success_rate for the current training iteration."""
    if step < total / 3:
        return 1.0   # Phase 1: deterministic
    elif step < 2 * total / 3:
        return 0.8   # Phase 2: mild stochasticity
    else:
        return 0.5   # Phase 3: full stochasticity


def _self_play_worker(args):
    """
    Top-level worker for parallel self-play (must be module-level to be picklable).
    Each worker runs on its own GPU and returns a list of (training_data, winner).
    """
    cpu_state_dict, n_games, num_simulations, temp_threshold, success_rate, device_str = args
    model = ActorCritic().to(device_str)
    model.load_state_dict({k: v.to(device_str) for k, v in cpu_state_dict.items()})
    model.eval()
    mcts = MCTS(model, device=device_str, num_simulations=num_simulations)
    results = []
    for _ in range(n_games):
        data, winner = self_play_game(mcts, temp_threshold=temp_threshold, success_rate=success_rate)
        results.append((data, winner))
    return results


def self_play_game(mcts, temp_threshold=15, success_rate=0.5):
    """
    Play one game with MCTS. Returns:
      - list of (state, mcts_policy, value) training examples
      - winner (1, 2, or None)
    """
    env = SuperTicTacToeEnv(success_rate=success_rate)
    env.reset()
    examples = []
    step = 0

    while not env.done:
        visit_probs = mcts.run(env)

        examples.append({
            'state': env._get_state(),
            'policy': visit_probs,
            'player': env.current_player,
        })

        # Explore early moves, play greedily later
        if step < temp_threshold:
            probs = visit_probs.copy()
            probs /= probs.sum()
            action = int(np.random.choice(len(probs), p=probs))
        else:
            action = int(np.argmax(visit_probs))

        env.step(action)
        step += 1

    winner = env.winner
    training_data = []
    for ex in examples:
        if winner is None:
            z = 0.0
        elif ex['player'] == winner:
            z = 1.0
        else:
            z = -1.0
        training_data.append((ex['state'], ex['policy'], z))

    return training_data, winner


def train_on_batch(model, optimizer, states, policies, values, device):
    states_t  = torch.FloatTensor(np.array(states)).to(device)
    policies_t = torch.FloatTensor(np.array(policies)).to(device)
    values_t   = torch.FloatTensor(np.array(values)).to(device)

    # Channel 2 encodes empty valid cells → use as action mask
    masks_t = states_t[:, 2].reshape(len(states), -1).bool()

    pred_p, pred_v = model(states_t, masks_t)

    policy_loss = -(policies_t * torch.log(pred_p + 1e-8)).sum(dim=1).mean()
    value_loss  = F.mse_loss(pred_v.squeeze(-1), values_t)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy_loss.item(), value_loss.item()


def train_alphazero(
    num_iterations: int = 200,
    games_per_iteration: int = 100,
    num_simulations: int = 50,
    train_epochs: int = 10,
    batch_size: int = 512,
    replay_buffer_size: int = 50_000,
    lr: float = 1e-3,
    device: str = 'cpu',
    checkpoint_dir: str = 'checkpoints_az',
    save_every: int = 10,
    eval_every: int = 10,
    init_from: str = '',
    curriculum: bool = False,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = ActorCritic().to(device)
    if init_from:
        model.load_state_dict(torch.load(init_from, map_location=device))
        print(f"Loaded weights from {init_from}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    mcts = MCTS(model, device=device, num_simulations=num_simulations)
    replay_buffer: deque = deque(maxlen=replay_buffer_size)

    reference = ActorCritic().to(device)
    best_vs_reference = 0.0
    best_path = os.path.join(checkpoint_dir, 'model_best.pt')

    # Multi-GPU parallel self-play: each GPU runs its own MCTS worker
    n_gpus = torch.cuda.device_count() if device == 'cuda' else 0
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs for parallel self-play")
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(n_gpus)
    else:
        pool = None

    for iteration in range(1, num_iterations + 1):
        # ── Self-play ────────────────────────────────────────────────────────
        rate = curriculum_rate(iteration, num_iterations) if curriculum else 0.5
        model.eval()
        p1_wins = p2_wins = draws = 0

        if pool is not None:
            # Distribute games evenly across GPUs
            base = games_per_iteration // n_gpus
            rem  = games_per_iteration % n_gpus
            cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            worker_args = [
                (cpu_sd, base + (1 if i < rem else 0),
                 num_simulations, 15, rate, f'cuda:{i}')
                for i in range(n_gpus)
            ]
            print(f"Iter {iteration:4d}/{num_iterations} | self-play ({n_gpus} GPUs)...",
                  end='\r', flush=True)
            raw = pool.map(_self_play_worker, worker_args)
            game_results = [item for sublist in raw for item in sublist]
        else:
            game_results = []
            for g in range(games_per_iteration):
                data, winner = self_play_game(mcts, temp_threshold=15, success_rate=rate)
                game_results.append((data, winner))
                print(f"Iter {iteration:4d}/{num_iterations} | "
                      f"game {g+1:3d}/{games_per_iteration} | "
                      f"buffer={len(replay_buffer)}",
                      end='\r', flush=True)

        for data, winner in game_results:
            replay_buffer.extend(data)
            if winner == 1:   p1_wins += 1
            elif winner == 2: p2_wins += 1
            else:             draws   += 1

        n = games_per_iteration
        print(f"Iter {iteration:4d}/{num_iterations} | "
              f"rate={rate:.1f} | "
              f"P1={p1_wins/n:.0%} P2={p2_wins/n:.0%} draw={draws/n:.0%} | "
              f"buffer={len(replay_buffer)}")

        if len(replay_buffer) < batch_size:
            print("  Buffer too small, skipping training")
            continue

        # ── Training ─────────────────────────────────────────────────────────
        model.train()
        all_data = list(replay_buffer)
        total_pl = total_vl = 0.0

        for _ in range(train_epochs):
            idx = np.random.choice(len(all_data), size=batch_size, replace=len(all_data) < batch_size)
            batch = [all_data[i] for i in idx]
            states   = [b[0] for b in batch]
            policies = [b[1] for b in batch]
            values   = [b[2] for b in batch]
            pl, vl = train_on_batch(model, optimizer, states, policies, values, device)
            total_pl += pl
            total_vl += vl

        print(f"  policy_loss={total_pl/train_epochs:.4f} | "
              f"value_loss={total_vl/train_epochs:.4f}")

        # ── Checkpoint ───────────────────────────────────────────────────────
        if iteration % save_every == 0:
            path = os.path.join(checkpoint_dir, f'model_{iteration:04d}.pt')
            torch.save(model.state_dict(), path)
            print(f"  Saved {path}")

        if iteration % eval_every == 0:
            model.eval()
            results = evaluate(model, reference, num_games=100, device=device)
            win_rate = results['model1_wins']
            model.train()
            print(f"  vs reference: {win_rate:.0%} wins", end="")
            if win_rate > best_vs_reference:
                best_vs_reference = win_rate
                torch.save(model.state_dict(), best_path)
                print("  ← new best!")
            else:
                print()

    if pool is not None:
        pool.close()
        pool.join()

    final_path = os.path.join(checkpoint_dir, 'model_final.pt')
    torch.save(model.state_dict(), final_path)
    print(f"\nDone. Best: {best_vs_reference:.0%} vs reference. "
          f"Saved to {final_path} and {best_path}")
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations',   type=int,   default=200)
    parser.add_argument('--games',        type=int,   default=100)
    parser.add_argument('--simulations',  type=int,   default=50)
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--batch-size',   type=int,   default=512)
    parser.add_argument('--device',       type=str,   default='cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_az')
    parser.add_argument('--save-every',   type=int,   default=10)
    parser.add_argument('--eval-every',   type=int,   default=10)
    parser.add_argument('--init-from',    type=str,   default='',
                        help='Optional: path to PPO checkpoint to warm-start from')
    parser.add_argument('--curriculum', action='store_true',
                        help='Enable curriculum learning (gradual stochasticity)')
    args = parser.parse_args()

    train_alphazero(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        num_simulations=args.simulations,
        train_epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        eval_every=args.eval_every,
        init_from=args.init_from,
        curriculum=args.curriculum,
    )
