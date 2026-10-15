"""Direct Preference Optimization (DPO) Alignment Engine."""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import torch.nn.functional as F

from ..models.foundation_model import MoEFoundationModel

logger = logging.getLogger("moe.dpo")


class DPOTrainer:
    """Trains policy model directly from pairwise human preference data without a reward model."""

    def __init__(
        self,
        policy_model: MoEFoundationModel,
        reference_model: MoEFoundationModel,
        beta: float = 0.1,
        learning_rate: float = 5e-6,
    ):
        self.policy = policy_model
        self.reference = reference_model
        self.beta = beta

        # Freeze reference model completely
        self.reference.eval()
        for p in self.reference.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

    def compute_log_probs(
        self,
        model: MoEFoundationModel,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log-probabilities masked over completion tokens."""
        _, _, logits = model(input_ids)
        # Shift logits and labels
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute log-softmax over vocab
        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather target token log probs: [batch, seq_len]
        token_log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(2)).squeeze(2)

        # Mask out padding or prompt tokens where label == -100
        mask = (shift_labels != -100).float()
        sequence_log_probs = (token_log_probs * mask).sum(dim=-1)
        return sequence_log_probs

    def compute_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates DPO loss, chosen rewards, and rejected rewards."""
        # 1. Policy model log probs
        pi_chosen = self.compute_log_probs(self.policy, chosen_input_ids, chosen_labels)
        pi_rejected = self.compute_log_probs(self.policy, rejected_input_ids, rejected_labels)

        # 2. Reference model log probs (no grad)
        with torch.no_grad():
            ref_chosen = self.compute_log_probs(self.reference, chosen_input_ids, chosen_labels)
            ref_rejected = self.compute_log_probs(
                self.reference, rejected_input_ids, rejected_labels
            )

        # 3. Log ratios
        pi_logratios = pi_chosen - pi_rejected
        ref_logratios = ref_chosen - ref_rejected

        logits = self.beta * (pi_logratios - ref_logratios)
        # DPO loss: -log(sigmoid(beta * (log_pi - log_ref)))
        loss = -F.logsigmoid(logits).mean()

        # Implicit rewards
        chosen_rewards = self.beta * (pi_chosen - ref_chosen).detach()
        rejected_rewards = self.beta * (pi_rejected - ref_rejected).detach()

        return loss, chosen_rewards.mean(), rejected_rewards.mean()

    def step(
        self,
        chosen_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_ids: torch.Tensor,
        rejected_labels: torch.Tensor,
    ) -> dict:
        self.policy.train()
        self.optimizer.zero_grad()

        loss, chosen_r, rejected_r = self.compute_loss(
            chosen_ids, chosen_labels, rejected_ids, rejected_labels
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        accuracy = (chosen_r > rejected_r).float().item()

        return {
            "dpo_loss": loss.item(),
            "chosen_reward": chosen_r.item(),
            "rejected_reward": rejected_r.item(),
            "reward_margin": (chosen_r - rejected_r).item(),
            "preference_accuracy": accuracy,
        }
