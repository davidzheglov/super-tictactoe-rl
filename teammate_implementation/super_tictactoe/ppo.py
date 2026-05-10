import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation (GAE).
    Returns (advantages, returns).
    """
    advantages = []
    returns = []
    gae = 0.0
    next_value = 0.0
    next_return = 0.0

    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        mask = 1.0 - float(d)
        delta = r + gamma * next_value * mask - v
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        next_return = r + gamma * next_return * mask
        returns.insert(0, next_return)
        next_value = v

    return advantages, returns


def ppo_update(
    model,
    optimizer: torch.optim.Optimizer,
    buffer: Dict,
    epochs: int = 4,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.05,
    value_coef: float = 0.5,
) -> Dict[str, float]:
    """Run PPO update over the collected buffer."""
    device = next(model.parameters()).device
    states = torch.FloatTensor(np.array(buffer['states'])).to(device)
    action_masks = torch.BoolTensor(np.array(buffer['action_masks'])).to(device)
    actions = torch.LongTensor(np.array(buffer['actions'])).to(device)
    old_log_probs = torch.FloatTensor(np.array(buffer['log_probs'])).to(device)
    returns = torch.FloatTensor(np.array(buffer['returns'])).to(device)
    advantages = torch.FloatTensor(np.array(buffer['advantages'])).to(device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_losses, critic_losses = [], []

    for _ in range(epochs):
        probs, values = model(states, action_masks)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        critic_loss = F.mse_loss(values.squeeze(), returns)
        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

    return {
        'actor_loss': float(np.mean(actor_losses)),
        'critic_loss': float(np.mean(critic_losses)),
    }
