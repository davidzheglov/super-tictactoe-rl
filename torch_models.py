"""PyTorch models and helpers for Super Tic-Tac-Toe agents."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


class TorchPolicyValueNet(nn.Module):
    """MLP with policy logits and scalar value heads."""

    def __init__(
        self,
        input_dim: int = 97,
        num_actions: int = 96,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_dim, int(hidden_size)))
            layers.append(nn.ReLU())
            last_dim = int(hidden_size)
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, num_actions)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = obs.float()
        x = self.backbone(x)
        return self.policy_head(x), self.value_head(x).squeeze(-1)


class TorchDQN(nn.Module):
    """MLP Q-network."""

    def __init__(
        self,
        input_dim: int = 97,
        num_actions: int = 96,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), int(hidden_size)),
            nn.ReLU(),
            nn.Linear(int(hidden_size), num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float())


def resolve_torch_device(device: str) -> torch.device:
    normalized = (device or "auto").lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized in {"auto", "cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if normalized in {"auto", "mps"} and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def mask_logits(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    mask = action_mask.bool()
    return logits.masked_fill(~mask, -1.0e9)


@torch.no_grad()
def select_action_torch(
    model: TorchPolicyValueNet,
    obs: np.ndarray,
    action_mask: np.ndarray,
    device: torch.device,
    deterministic: bool = False,
) -> Tuple[int, float, float]:
    if not np.any(action_mask):
        raise ValueError("Cannot select an action when no legal actions remain.")

    model.eval()
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    logits, value = model(obs_t)
    masked = mask_logits(logits, mask_t)
    if deterministic:
        action_t = torch.argmax(masked, dim=-1)
        log_prob_t = torch.log_softmax(masked, dim=-1).gather(1, action_t[:, None]).squeeze(1)
    else:
        dist = Categorical(logits=masked)
        action_t = dist.sample()
        log_prob_t = dist.log_prob(action_t)
    return int(action_t.item()), float(log_prob_t.item()), float(value.item())


@torch.no_grad()
def masked_q_argmax(
    model: TorchDQN,
    obs: np.ndarray,
    action_mask: np.ndarray,
    device: torch.device,
) -> int:
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
    q_values = model(obs_t)
    q_values = q_values.masked_fill(~mask_t, -1.0e9)
    return int(torch.argmax(q_values, dim=-1).item())
