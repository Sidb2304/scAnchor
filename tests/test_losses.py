import torch

from scanchor.model.losses import correction_loss, supervised_contrastive_loss, variance_floor_penalty


def test_supervised_contrastive_loss_is_finite_and_positive():
    embeddings = torch.randn(20, 8)
    labels = torch.randint(0, 4, (20,))

    loss = supervised_contrastive_loss(embeddings, labels)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_variance_floor_penalty_zero_when_variance_preserved():
    torch.manual_seed(0)
    original = torch.randn(30, 8)
    labels = torch.randint(0, 3, (30,))

    penalty = variance_floor_penalty(original, original.clone(), labels, min_ratio=0.8)

    assert penalty.item() == 0.0


def test_variance_floor_penalty_positive_when_collapsed():
    torch.manual_seed(0)
    original = torch.randn(30, 8)
    labels = torch.randint(0, 3, (30,))
    collapsed = original.mean(dim=0, keepdim=True).expand_as(original).clone()

    penalty = variance_floor_penalty(original, collapsed, labels, min_ratio=0.8)

    assert penalty.item() > 0.0


def test_correction_loss_combines_both_terms():
    original = torch.randn(20, 8)
    corrected = original + 0.01 * torch.randn(20, 8)
    labels = torch.randint(0, 4, (20,))

    total, metrics = correction_loss(original, corrected, labels)

    assert torch.isfinite(total)
    assert set(metrics.keys()) == {"contrastive", "variance_penalty", "total"}
