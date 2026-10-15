"""Full Mixture-of-Experts Foundation Model Architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_block import MoETransformerBlock, RMSNorm


@dataclass
class MoEConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 8192
    num_experts: int = 8
    num_experts_per_tok: int = 2
    aux_loss_coef: float = 0.01
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0


class MoEFoundationModel(nn.Module):
    """End-to-end 3D-Parallel Compatible MoE Foundation Transformer."""

    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                MoETransformerBlock(
                    hidden_dim=config.hidden_size,
                    intermediate_dim=config.intermediate_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    num_experts=config.num_experts,
                    top_k=config.num_experts_per_tok,
                    aux_loss_coef=config.aux_loss_coef,
                    rms_norm_eps=config.rms_norm_eps,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE rotary frequencies
        self._init_rope()

    def _init_rope(self) -> None:
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through foundation model.
        Returns:
            Tuple of (total_loss, lm_loss, logits)
        """
        _, seq_len = input_ids.shape
        cos, sin = self._get_cos_sin(seq_len, input_ids.device)

        hidden_states = self.embed_tokens(input_ids)
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)

        for layer in self.layers:
            hidden_states, aux_loss = layer(hidden_states, cos=cos, sin=sin)
            total_aux_loss = total_aux_loss + aux_loss

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        total_loss = torch.tensor(0.0, device=input_ids.device)
        lm_loss = torch.tensor(0.0, device=input_ids.device)

        if labels is not None:
            # Shift tokens for next-token autoregressive prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            total_loss = lm_loss + total_aux_loss

        return total_loss, lm_loss, logits
