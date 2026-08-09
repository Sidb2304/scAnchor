import torch

from scanchor.model.correction_head import CorrectionHead


def test_output_shape_matches_embedding():
    head = CorrectionHead(embed_dim=16, vocab_sizes=[5, 3], n_continuous=2)
    embedding = torch.randn(8, 16)
    categorical = torch.randint(0, 3, (8, 2))
    continuous = torch.randn(8, 2)

    corrected = head(embedding, categorical, continuous)

    assert corrected.shape == embedding.shape


def test_identity_at_initialization():
    """Zero-init on the last layer means e' == e before any training."""
    head = CorrectionHead(embed_dim=16, vocab_sizes=[5], n_continuous=1)
    embedding = torch.randn(4, 16)
    categorical = torch.randint(0, 5, (4, 1))
    continuous = torch.randn(4, 1)

    corrected = head(embedding, categorical, continuous)

    assert torch.allclose(corrected, embedding, atol=1e-6)


def test_unseen_categorical_id_does_not_crash():
    """UNK index 0 must be valid for a category value never seen at training time."""
    head = CorrectionHead(embed_dim=16, vocab_sizes=[3], n_continuous=1)
    embedding = torch.randn(2, 16)
    unk_categorical = torch.zeros((2, 1), dtype=torch.long)
    continuous = torch.randn(2, 1)

    corrected = head(embedding, unk_categorical, continuous)

    assert corrected.shape == embedding.shape


def test_delta_norm_is_bounded_even_with_an_extreme_raw_delta():
    """A large raw delta_net output must still be clamped to max_delta_ratio * ||embedding||.

    Directly reproduces the failure mode from real training: an adversarial
    objective pushing delta_net toward huge outputs shouldn't be able to blow
    up the corrected embedding's magnitude.
    """
    head = CorrectionHead(embed_dim=16, vocab_sizes=[3], n_continuous=1, max_delta_ratio=0.5)
    with torch.no_grad():
        head.delta_net[-1].weight.fill_(1000.0)
        head.delta_net[-1].bias.fill_(1000.0)

    embedding = torch.randn(4, 16)
    categorical = torch.zeros((4, 1), dtype=torch.long)
    continuous = torch.randn(4, 1)

    corrected = head(embedding, categorical, continuous)
    delta = corrected - embedding

    assert torch.all(delta.norm(dim=-1) <= 0.5 * embedding.norm(dim=-1) + 1e-4)


def test_max_delta_ratio_zero_forces_identity():
    head = CorrectionHead(embed_dim=8, vocab_sizes=[3], n_continuous=1, max_delta_ratio=0.0)
    with torch.no_grad():
        head.delta_net[-1].weight.fill_(50.0)
        head.delta_net[-1].bias.fill_(50.0)

    embedding = torch.randn(3, 8)
    categorical = torch.zeros((3, 1), dtype=torch.long)
    continuous = torch.randn(3, 1)

    corrected = head(embedding, categorical, continuous)

    assert torch.allclose(corrected, embedding, atol=1e-5)
