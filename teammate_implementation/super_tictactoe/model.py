import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 12 * 12, 256)
        self.actor = nn.Linear(256, 144)
        self.critic = nn.Linear(256, 1)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc(x))

    def forward(
        self, x: torch.Tensor, action_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, 3, 12, 12)
        action_mask: (batch, 144) bool — True = valid action
        Returns: probs (batch, 144), value (batch, 1)
        """
        features = self._backbone(x)
        logits = self.actor(features)
        logits = logits.masked_fill(~action_mask, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        value = self.critic(features)
        return probs, value

    def get_action(
        self, state: torch.Tensor, action_mask: torch.Tensor, deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        state: (3, 12, 12) — single state, no batch dim
        action_mask: (144,) bool
        Returns: action (int), log_prob (scalar tensor), value (scalar tensor)
        """
        probs, value = self.forward(state.unsqueeze(0), action_mask.unsqueeze(0))
        probs = probs.squeeze(0)
        if deterministic:
            action = probs.argmax().item()
            log_prob = torch.log(probs[action] + 1e-8)
        else:
            dist = torch.distributions.Categorical(probs)
            action_tensor = dist.sample()
            log_prob = dist.log_prob(action_tensor)
            action = action_tensor.item()
        return action, log_prob, value.squeeze()
