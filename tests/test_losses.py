import torch
import torch.nn.functional as F

from scanchor.model.losses import (
    adversarial_batch_loss,
    correction_loss,
    donor_consistency_loss,
    supervised_contrastive_loss,
    variance_floor_penalty,
)


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
    assert set(metrics.keys()) == {
        "contrastive", "variance_penalty", "donor_consistency", "adversarial_batch", "total",
    }
    assert metrics["donor_consistency"] == 0.0  # no donor_ids passed -> term is inert
    assert metrics["adversarial_batch"] == 0.0  # no batch_logits passed -> term is inert


def test_donor_consistency_loss_zero_when_no_cross_batch_positive():
    """Each donor confined to a single batch -> no valid positive pair -> 0."""
    embeddings = torch.randn(12, 8)
    donor_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    batch_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])  # donor <-> batch 1:1

    loss = donor_consistency_loss(embeddings, donor_ids, batch_ids)

    assert loss.item() == 0.0


def test_donor_consistency_loss_finite_when_donors_crossed_with_batches():
    torch.manual_seed(0)
    embeddings = torch.randn(12, 8)
    donor_ids = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    batch_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])  # every donor in every batch

    loss = donor_consistency_loss(embeddings, donor_ids, batch_ids)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_donor_consistency_loss_ignores_unknown_donor_cells():
    embeddings = torch.randn(6, 8)
    donor_ids = torch.tensor([-1, -1, -1, -1, -1, -1])
    batch_ids = torch.tensor([0, 0, 1, 1, 2, 2])

    loss = donor_consistency_loss(embeddings, donor_ids, batch_ids)

    assert loss.item() == 0.0


def test_correction_loss_includes_donor_term_when_provided():
    torch.manual_seed(0)
    original = torch.randn(12, 8)
    corrected = original + 0.01 * torch.randn(12, 8)
    labels = torch.randint(0, 3, (12,))
    donor_ids = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    batch_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

    total, metrics = correction_loss(original, corrected, labels, donor_ids=donor_ids, batch_ids=batch_ids)

    assert torch.isfinite(total)
    assert metrics["donor_consistency"] > 0.0


def test_adversarial_batch_loss_is_cross_entropy():
    logits = torch.randn(6, 3)
    batch_ids = torch.randint(0, 3, (6,))

    loss = adversarial_batch_loss(logits, batch_ids)

    assert torch.allclose(loss, F.cross_entropy(logits, batch_ids))


def test_correction_loss_includes_adversarial_term_when_provided():
    torch.manual_seed(0)
    original = torch.randn(6, 8)
    corrected = original + 0.01 * torch.randn(6, 8)
    labels = torch.randint(0, 3, (6,))
    batch_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    batch_logits = torch.randn(6, 3)

    total, metrics = correction_loss(original, corrected, labels, batch_ids=batch_ids, batch_logits=batch_logits)

    assert torch.isfinite(total)
    assert metrics["adversarial_batch"] > 0.0
