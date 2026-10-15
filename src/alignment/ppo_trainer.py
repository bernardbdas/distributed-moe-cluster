"""Proximal Policy Optimization (PPO) RLHF Alignment Engine."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.foundation_model import MoEFoundationModel

logger = logging.getLogger("moe.ppo")


class PPOTrainer:
    """Actor-Critic PPO alignment with GAE and KL divergence penalties."""

    def __init__(
        self,
        actor_model: MoEFoundationModel,
        critic_value_head: nn.Linear,
        clip_epsilon: float = 0.2,
        gamma: float = 1.0,
        lam: float = 0.95,
        kl_coef: float = 0.05,
        lr: float = 1e-6,
    ):
        self.actor = actor_model
        self.critic = critic_value_head
        self.clip_eps = clip_epsilon
        self.gamma = gamma
        self.lam = lam
        self.kl_coef = kl_coef

        self.optimizer = torch.optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
            betas=(0.9, 0.95),
        )

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates Generalized Advantage Estimation (GAE) and returns targets."""
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(seq_len)):
            next_value = values[:, t + 1] if t < seq_len - 1 else 0.0
            delta = rewards[:, t] + self.gamma * next_value * masks[:, t] - values[:, t]
            advantages[:, t] = last_gae = delta + self.gamma * self.lam * masks[:, t] * last_gae

        returns = advantages + values
        return advantages, returns

    def compute_ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Clipped surrogate objective and value function loss."""
        # Policy ratio r_t(theta) = exp(log_prob - old_log_prob)
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Critic MSE value loss
        value_loss = 0.5 * F.mse_loss(values, returns)
        return policy_loss, value_loss
