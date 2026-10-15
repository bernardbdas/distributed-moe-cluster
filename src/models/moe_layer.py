"""Sparse Mixture-of-Experts (MoE) Layer with Top-K Gating and Auxiliary Load Balancing."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """Individual Expert MLP using SwiGLU activation."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)  # Gate
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)  # Down
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)  # Up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (Swish(x @ w1) * (x @ w3)) @ w2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SparseMoELayer(nn.Module):
    """Sparse MoE routing layer with Top-K dispatch and load balancing loss."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.aux_loss_coef = aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

        # Gating router: projects input token embeddings to expert logits
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # Expert pool
        self.experts = nn.ModuleList(
            [ExpertMLP(hidden_dim, intermediate_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass dispatching tokens to top-k experts.
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
        Returns:
            Tuple of (output_tensor, total_aux_loss)
        """
        orig_shape = x.shape
        # Flatten tokens across batch and sequence dimensions: [N, D]
        flat_x = x.view(-1, self.hidden_dim)
        _num_tokens = flat_x.size(0)

        # 1. Compute routing logits: [N, num_experts]
        logits = self.gate(flat_x)

        # Router Z-loss: log(sum(exp(logits)))^2 to stabilize large fp16/bf16 activations
        z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2) * self.router_z_loss_coef

        # Softmax probabilities over all experts
        probs = F.softmax(logits, dim=-1)

        # 2. Select top-k experts per token
        top_k_weights, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Re-normalize top-k weights so they sum to 1.0
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # 3. Compute auxiliary load balancing loss (Switch Transformer / DeepSeek style)
        # Fraction of tokens routed to each expert: f_i = (1 / N) * sum(is_top_k(token, expert))
        # Average probability allocated to each expert: P_i = (1 / N) * sum(prob(token, expert))
        expert_mask = F.one_hot(
            top_k_indices, num_classes=self.num_experts
        ).float()  # [N, top_k, num_experts]
        tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)  # [num_experts]
        avg_prob_per_expert = probs.mean(dim=0)  # [num_experts]
        aux_loss = (
            self.num_experts
            * torch.sum(tokens_per_expert * avg_prob_per_expert)
            * self.aux_loss_coef
        )
        total_moe_loss = aux_loss + z_loss

        # 4. Dispatch tokens and accumulate expert outputs
        final_output = torch.zeros_like(flat_x)

        # Group tokens per expert to maximize GEMM throughput
        for expert_idx in range(self.num_experts):
            # Check which tokens route to this expert in any of their top-k slots
            expert_submask = top_k_indices == expert_idx  # [N, top_k]
            token_rows, slot_idx = torch.where(expert_submask)

            if token_rows.numel() == 0:
                continue

            selected_tokens = flat_x[token_rows]  # [M, D]
            expert_out = self.experts[expert_idx](selected_tokens)  # [M, D]

            weights = top_k_weights[token_rows, slot_idx].unsqueeze(-1)  # [M, 1]
            final_output.index_add_(0, token_rows, expert_out * weights)

        return final_output.view(orig_shape), total_moe_loss
