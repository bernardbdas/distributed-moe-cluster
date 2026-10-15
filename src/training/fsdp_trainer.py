"""Distributed PyTorch FSDP Trainer with Mixed Precision and Gradient Accumulation."""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader

from ..models.foundation_model import MoEConfig, MoEFoundationModel
from ..models.transformer_block import MoETransformerBlock
from ..telemetry.metrics import ModelMetricsTracker
from .dataset import SyntheticPackedDataset

logger = logging.getLogger("moe.trainer")


class FSDPTrainer:
    """Manages multi-GPU FSDP pre-training loop with MoE auxiliary loss tracking."""

    def __init__(
        self,
        config: MoEConfig,
        micro_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_steps: int = 1000,
        device_str: Optional[str] = None,
    ):
        self.config = config
        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = gradient_accumulation_steps
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.max_steps = max_steps

        # Distributed environment detection
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.world_size = dist.get_world_size() if self.is_distributed else 1

        if device_str:
            self.device = torch.device(device_str)
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.metrics_tracker = ModelMetricsTracker(
            num_params=self._estimate_params(),
            seq_len=config.max_position_embeddings,
        )

        self._init_model_and_optimizer()

    def _estimate_params(self) -> int:
        # Approximate parameter count for 8-expert MoE
        dense_params = self.config.vocab_size * self.config.hidden_size * 2
        layer_params = self.config.num_hidden_layers * (
            # Attention
            (self.config.hidden_size * self.config.hidden_size * 2)
            +
            # MoE Experts (SwiGLU: 3 matrices per expert * num_experts)
            (self.config.num_experts * 3 * self.config.hidden_size * self.config.intermediate_size)
        )
        return dense_params + layer_params

    def _init_model_and_optimizer(self) -> None:
        raw_model = MoEFoundationModel(self.config).to(self.device)

        if self.is_distributed and torch.cuda.is_available():
            auto_wrap_policy = transformer_auto_wrap_policy(
                transformer_layer_cls={MoETransformerBlock}
            )
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            self.model = FSDP(
                raw_model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mp_policy,
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                device_id=self.device,
            )
        else:
            self.model = raw_model

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def train_step(self, batch: dict) -> dict:
        """Executes a single forward + backward + step pass."""
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        t0 = time.perf_counter()
        total_loss, lm_loss, _ = self.model(input_ids, labels=labels)
        loss = total_loss / self.grad_accum_steps
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        step_time = time.perf_counter() - t0

        tokens_processed = input_ids.numel() * self.world_size
        tok_per_sec = tokens_processed / max(step_time, 1e-5)
        mfu = self.metrics_tracker.compute_mfu(tokens_processed, step_time)

        return {
            "total_loss": total_loss.item(),
            "lm_loss": lm_loss.item(),
            "grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
            "step_time_sec": step_time,
            "tokens_per_sec": tok_per_sec,
            "mfu": mfu,
        }

    def train_epochs(self, num_steps: int = 10):
        """Run training for a specified number of steps."""
        dataset = SyntheticPackedDataset(
            vocab_size=self.config.vocab_size,
            seq_len=min(512, self.config.max_position_embeddings),
            num_samples=num_steps * self.micro_batch_size,
        )
        loader = DataLoader(dataset, batch_size=self.micro_batch_size)

        step = 0
        for batch in loader:
            metrics = self.train_step(batch)
            step += 1
            if self.rank == 0:
                logger.info(
                    "Step %d/%d | Loss: %.4f | LM Loss: %.4f | Tok/s: %.1f | MFU: %.2f%%",
                    step,
                    num_steps,
                    metrics["total_loss"],
                    metrics["lm_loss"],
                    metrics["tokens_per_sec"],
                    metrics["mfu"] * 100,
                )
            if step >= num_steps:
                break
