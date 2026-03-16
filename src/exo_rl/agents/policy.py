from __future__ import annotations

import math

import torch
from torch import nn


def build_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden))
        layers.append(nn.Tanh())
        prev_dim = hidden
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        self.actor_mean = build_mlp(obs_dim, hidden_sizes, act_dim)
        self.critic = build_mlp(obs_dim, hidden_sizes, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        final_layer = self.actor_mean[-1]
        if isinstance(final_layer, nn.Linear):
            nn.init.constant_(final_layer.bias, -1.5)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def distribution(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor_mean(obs)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def sample_action(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.distribution(obs)
        noise = torch.randn_like(mean)
        pre_sigmoid = mean + std * noise
        action = torch.sigmoid(pre_sigmoid)
        log_prob = self._gaussian_log_prob(pre_sigmoid, mean, std) - torch.log(
            action * (1.0 - action) + 1e-6
        ).sum(dim=-1)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)).sum(dim=-1)
        value = self.value(obs)
        return action, log_prob, entropy, value

    def evaluate_actions(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        squashed = torch.clamp(action, 1e-6, 1.0 - 1e-6)
        pre_sigmoid = torch.log(squashed) - torch.log1p(-squashed)
        mean, std = self.distribution(obs)
        log_prob = self._gaussian_log_prob(pre_sigmoid, mean, std) - torch.log(
            squashed * (1.0 - squashed) + 1e-6
        ).sum(dim=-1)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(std)).sum(dim=-1)
        value = self.value(obs)
        return log_prob, entropy, value

    @staticmethod
    def _gaussian_log_prob(sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        variance = std.pow(2)
        log_scale = torch.log(std)
        return (
            -((sample - mean).pow(2)) / (2 * variance)
            - log_scale
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=-1)
