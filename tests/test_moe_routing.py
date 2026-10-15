import torch

from src.models.foundation_model import MoEConfig, MoEFoundationModel
from src.models.moe_layer import SparseMoELayer


def test_sparse_moe_forward():
    hidden_dim = 128
    intermediate_dim = 256
    num_experts = 4
    top_k = 2

    moe = SparseMoELayer(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_experts=num_experts,
        top_k=top_k,
        aux_loss_coef=0.01,
    )

    batch_size = 2
    seq_len = 8
    x = torch.randn(batch_size, seq_len, hidden_dim)

    out, aux_loss = moe(x)

    assert out.shape == x.shape
    assert aux_loss.item() > 0.0
    assert not torch.isnan(out).any()


def test_moe_foundation_model_loss():
    cfg = MoEConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = MoEFoundationModel(cfg)

    input_ids = torch.randint(0, 100, (2, 16))
    labels = input_ids.clone()

    total_loss, lm_loss, logits = model(input_ids, labels=labels)

    assert total_loss.item() > 0.0
    assert lm_loss.item() > 0.0
    assert logits.shape == (2, 16, 100)

    # Check backprop
    total_loss.backward()
    assert model.embed_tokens.weight.grad is not None
