import torch

from src.alignment.dpo_trainer import DPOTrainer
from src.models.foundation_model import MoEConfig, MoEFoundationModel


def test_dpo_loss_computation():
    cfg = MoEConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        num_experts=2,
        num_experts_per_tok=1,
    )
    policy = MoEFoundationModel(cfg)
    reference = MoEFoundationModel(cfg)

    trainer = DPOTrainer(policy, reference, beta=0.1)

    chosen_ids = torch.randint(0, 100, (2, 16))
    chosen_labels = chosen_ids.clone()
    rejected_ids = torch.randint(0, 100, (2, 16))
    rejected_labels = rejected_ids.clone()

    metrics = trainer.step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)

    assert "dpo_loss" in metrics
    assert "reward_margin" in metrics
    assert not torch.isnan(torch.tensor(metrics["dpo_loss"]))
