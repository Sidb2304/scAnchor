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
