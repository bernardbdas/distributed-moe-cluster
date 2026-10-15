"""Ray Train Distributed Cluster Orchestrator."""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..models.foundation_model import MoEConfig

logger = logging.getLogger("moe.ray")


class RayTrainClusterOrchestrator:
    """Launches and scales TorchTrainer across elastic Ray clusters."""

    def __init__(self, num_workers: int = 4, use_gpu: bool = True):
        self.num_workers = num_workers
        self.use_gpu = use_gpu

    def launch(self, config_dict: Dict[str, Any], max_steps: int = 100) -> None:
        """Launch distributed job using Ray Train TorchTrainer."""
        try:
            import ray
            from ray.train import ScalingConfig
            from ray.train.torch import TorchTrainer

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            scaling_config = ScalingConfig(
                num_workers=self.num_workers,
                use_gpu=self.use_gpu,
            )

            def train_loop_per_worker(config: dict):
                from ..models.foundation_model import MoEConfig
                from .fsdp_trainer import FSDPTrainer

                moe_cfg = MoEConfig(**config.get("model", {}))
                trainer = FSDPTrainer(moe_cfg, max_steps=max_steps)
                trainer.train_epochs(num_steps=max_steps)

            trainer = TorchTrainer(
                train_loop_per_worker=train_loop_per_worker,
                train_loop_config=config_dict,
                scaling_config=scaling_config,
            )
            logger.info("Submitting Ray TorchTrainer with %d workers...", self.num_workers)
            trainer.fit()
        except ImportError:
            logger.warning(
                "Ray or Ray Train not installed. Falling back to local standalone trainer."
            )
            from .fsdp_trainer import FSDPTrainer

            cfg = MoEConfig(
                vocab_size=1000,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_experts=4,
            )
            trainer = FSDPTrainer(cfg, max_steps=max_steps)
            trainer.train_epochs(num_steps=5)
