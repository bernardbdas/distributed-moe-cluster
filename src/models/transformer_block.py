"""Transformer Block with RMSNorm, RoPE, GQA, and Sparse MoE Feed-Forward."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .moe_layer import SparseMoELayer


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Applies rotary position embeddings to query and key tensors."""

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with FlashAttention-3 fast path."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if cos is not None and sin is not None:
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads to match Q heads for GQA
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # PyTorch SDPA uses FlashAttention / Memory-Efficient attention kernels under the hood
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(b, s, self.num_heads * self.head_dim)
        return self.o_proj(out)


class MoETransformerBlock(nn.Module):
    """Transformer decoder block wrapping Attention and Sparse MoE MLP with residual norms."""

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        num_heads: int,
        num_kv_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        aux_loss_coef: float = 0.01,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.input_layernorm = RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.self_attn = GroupedQueryAttention(hidden_dim, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_dim, eps=rms_norm_eps)
        self.moe = SparseMoELayer(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            top_k=top_k,
            aux_loss_coef=aux_loss_coef,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN Self-Attention
        attn_out = self.self_attn(self.input_layernorm(x), cos=cos, sin=sin)
        x = x + attn_out

        # Pre-LN MoE
        moe_out, aux_loss = self.moe(self.post_attention_layernorm(x))
        x = x + moe_out

        return x, aux_loss
