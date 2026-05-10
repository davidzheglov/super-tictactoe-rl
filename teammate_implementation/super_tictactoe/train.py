import os
import copy
import random
import torch
import torch.nn as nn
import argparse
from collections import deque
from super_tictactoe.model import ActorCritic
from super_tictactoe.selfplay import collect_episodes_vectorized, build_buffer
from super_tictactoe.ppo import ppo_update
from super_tictactoe.heuristics import random_heuristic, stronger_heuristic
from super_tictactoe.position_seeder import generate_position_pool


def curriculum_rate(step: int, total: int) -> float:
    """Return success_rate for the current training step."""
    if step < total / 3:
        return 1.0   # Phase 1: deterministic
    elif step < 2 * total / 3:
        return 0.8   # Phase 2: mild stochasticity
    else:
        return 0.5   # Phase 3: full stochasticity


def train(
    num_updates: int = 3000,
    episodes_per_update: int = 512,
    save_every: int = 100,
    device: str = 'cpu',
    checkpoint_dir: str = 'checkpoints',
    pool_size: int = 10,
    pool_prob: float = 0.5,
    curriculum: bool = False,
    heuristic_prob: float = 0.0,
    stronger_prob: float = 0.0,
    position_seed_prob: float = 0.0,
    position_pool_size: int = 500,
    position_refresh_every: int = 200,
    resume: str = None,
    lr: float = 3e-4,
):
    """
    pool_size:              max past checkpoints in opponent pool.
    pool_prob:              probability of using a pool opponent each update.
    curriculum:             gradually increase stochasticity during training.
    heuristic_prob:         probability of using random_heuristic as opponent.
    stronger_prob:          probability of using stronger_heuristic as opponent.
                            Opponent priority order: pool → stronger → random heuristic → self-play.
    position_seed_prob:     fraction of episodes that start from a mid-game position.
    position_pool_size:     number of positions to pre-generate (refreshed periodically).
    position_refresh_every: regenerate the position pool every N updates.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = ActorCritic().to(device)

    if resume:
        model.load_state_dict(torch.load(resume, map_location=device))
        print(f"Resumed from checkpoint: {resume}")

    # Wrap with DataParallel if multiple GPUs are available
    n_gpus = torch.cuda.device_count() if device == 'cuda' else 0
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # raw_model is the underlying ActorCritic (unwrapped) used for saving/pool
    raw_model = model.module if isinstance(model, nn.DataParallel) else model

    base_lr = lr
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

    # Opponent pool: deque of state_dicts (CPU tensors to save GPU memory)
    pool: deque = deque(maxlen=pool_size)
    opponent_model = ActorCritic().to(device)

    # Pre-seed pool with resume checkpoint so it faces its previous best from update 1
    if resume:
        seed_weights = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
        pool.append(seed_weights)
        print(f"Pool pre-seeded with resume checkpoint ({len(pool)} entry)")

    # Build initial position pool if position seeding is enabled
    position_pool = []
    if position_seed_prob > 0:
        init_rate = curriculum_rate(1, num_updates) if curriculum else 0.5
        print(f"Generating initial position pool ({position_pool_size} positions)...")
        position_pool = generate_position_pool(position_pool_size, success_rate=init_rate)
        print(f"  Generated {len(position_pool)} positions.")

    for update in range(1, num_updates + 1):
        # Linear learning rate decay: 3e-4 → 3e-5
        lr = base_lr * (1 - 0.9 * update / num_updates)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # Curriculum: success_rate increases stochasticity across phases
        rate = curriculum_rate(update, num_updates) if curriculum else 0.5

        # Refresh position pool periodically with the current success_rate
        if position_seed_prob > 0 and update > 1 and (update - 1) % position_refresh_every == 0:
            position_pool = generate_position_pool(position_pool_size, success_rate=rate)

        # Opponent priority: pool → stronger_heuristic → random_heuristic → self-play
        roll = random.random()
        if pool and roll < pool_prob:
            state_dict = random.choice(pool)
            opponent_model.load_state_dict(
                {k: v.to(device) for k, v in state_dict.items()}
            )
            opponent_model.eval()
            opp = opponent_model
        elif roll < pool_prob + stronger_prob:
            opp = stronger_heuristic
        elif roll < pool_prob + stronger_prob + heuristic_prob:
            opp = random_heuristic
        else:
            opp = None  # standard self-play

        print(f"Update {update:4d}/{num_updates} collecting...", end='\r', flush=True)
        episodes = collect_episodes_vectorized(
            episodes_per_update, model, device,
            opponent_model=opp, success_rate=rate,
            position_pool=position_pool if position_seed_prob > 0 else None,
            position_seed_prob=position_seed_prob,
        )
        buffer = build_buffer(episodes)
        losses = ppo_update(model, optimizer, buffer)

        # Count P1 wins among episodes that have at least one step
        p1_wins = sum(
            1 for ep in episodes
            if ep and ep[-1]['reward'] > 0.5 and ep[-1]['player'] == 1
        )
        n_ep = max(len(episodes), 1)
        print(
            f"Update {update:4d}/{num_updates} | "
            f"lr={lr:.2e} | "
            f"rate={rate:.1f} | "
            f"actor={losses['actor_loss']:.4f} | "
            f"critic={losses['critic_loss']:.4f} | "
            f"steps={len(buffer['states'])} | "
            f"p1win%={p1_wins/n_ep:.0%} | "
            f"pool={len(pool)}"
        )

        if update % save_every == 0:
            path = os.path.join(checkpoint_dir, f"model_{update:04d}.pt")
            torch.save(raw_model.state_dict(), path)
            print(f"  Saved checkpoint: {path}")
            # Add a CPU copy to the pool
            pool.append({k: v.cpu().clone() for k, v in raw_model.state_dict().items()})

    final_path = os.path.join(checkpoint_dir, 'model_final.pt')
    torch.save(raw_model.state_dict(), final_path)
    print(f"Training complete. Saved to {final_path}")
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=3000)
    parser.add_argument('--episodes', type=int, default=512)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--pool-size', type=int, default=10,
                        help='Max past checkpoints in opponent pool (0 = disabled)')
    parser.add_argument('--pool-prob', type=float, default=0.5,
                        help='Probability of using a pool opponent each update')
    parser.add_argument('--curriculum', action='store_true',
                        help='Enable curriculum learning (gradual stochasticity)')
    parser.add_argument('--heuristic-prob', type=float, default=0.0,
                        help='Probability of using random_heuristic as opponent')
    parser.add_argument('--stronger-prob', type=float, default=0.0,
                        help='Probability of using stronger_heuristic as opponent')
    parser.add_argument('--position-seed-prob', type=float, default=0.0,
                        help='Fraction of episodes that start from a mid-game seeded position')
    parser.add_argument('--position-pool-size', type=int, default=500,
                        help='Number of mid-game positions to pre-generate')
    parser.add_argument('--position-refresh-every', type=int, default=200,
                        help='Regenerate position pool every N updates')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Base learning rate (decays 10x over training)')
    args = parser.parse_args()

    train(
        num_updates=args.updates,
        episodes_per_update=args.episodes,
        save_every=args.save_every,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        pool_size=args.pool_size,
        pool_prob=args.pool_prob,
        curriculum=args.curriculum,
        heuristic_prob=args.heuristic_prob,
        stronger_prob=args.stronger_prob,
        position_seed_prob=args.position_seed_prob,
        position_pool_size=args.position_pool_size,
        position_refresh_every=args.position_refresh_every,
        resume=args.resume,
        lr=args.lr,
    )
