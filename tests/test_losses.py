import torch
import torch.nn.functional as F

from scanchor.model.losses import (
    adversarial_batch_loss,
    correction_loss,
    donor_consistency_loss,
    mmd_loss,
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
        "contrastive", "variance_penalty", "donor_consistency",
        "adversarial_batch", "batch_absorption", "mmd", "total",
    }
    assert metrics["donor_consistency"] == 0.0  # no donor_ids passed -> term is inert
    assert metrics["adversarial_batch"] == 0.0  # no batch_logits passed -> term is inert
    assert metrics["batch_absorption"] == 0.0  # no absorber_logits passed -> term is inert
    assert metrics["mmd"] == 0.0  # no batch_ids passed -> term is inert


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


def test_correction_loss_includes_absorption_term_when_provided():
    torch.manual_seed(0)
    original = torch.randn(6, 8)
    corrected = original + 0.01 * torch.randn(6, 8)
    labels = torch.randint(0, 3, (6,))
    batch_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    absorber_logits = torch.randn(6, 3)

    total, metrics = correction_loss(
        original, corrected, labels, batch_ids=batch_ids, absorber_logits=absorber_logits
    )

    assert torch.isfinite(total)
    assert metrics["batch_absorption"] > 0.0
    assert metrics["adversarial_batch"] == 0.0  # independent of the absorption term


def test_mmd_loss_zero_with_single_batch():
    embeddings = torch.randn(10, 8)
    batch_ids = torch.zeros(10, dtype=torch.long)

    loss = mmd_loss(embeddings, batch_ids)

    assert loss.item() == 0.0


def test_mmd_loss_finite_and_positive_across_shifted_batches():
    torch.manual_seed(0)
    x = torch.randn(10, 8)
    y = torch.randn(10, 8) + 5.0  # clearly separated distribution
    embeddings = torch.cat([x, y], dim=0)
    batch_ids = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])

    loss = mmd_loss(embeddings, batch_ids)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_mmd_loss_smaller_when_batches_already_aligned():
    torch.manual_seed(0)
    x = torch.randn(10, 8)
    aligned = torch.cat([x, x + 0.01 * torch.randn(10, 8)], dim=0)
    shifted = torch.cat([x, x + 5.0], dim=0)
    batch_ids = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])

    aligned_loss = mmd_loss(aligned, batch_ids)
    shifted_loss = mmd_loss(shifted, batch_ids)

    assert aligned_loss.item() < shifted_loss.item()


def test_mmd_loss_ignores_batches_with_fewer_than_two_cells():
    torch.manual_seed(0)
    embeddings = torch.randn(11, 8)
    batch_ids = torch.cat([torch.zeros(10, dtype=torch.long), torch.tensor([1])])

    loss = mmd_loss(embeddings, batch_ids)

    assert loss.item() == 0.0


def test_correction_loss_mmd_term_inert_when_weight_zero():
    torch.manual_seed(0)
    original = torch.randn(10, 8)
    corrected = original + 5.0 * torch.randn(10, 8)
    labels = torch.randint(0, 3, (10,))
    batch_ids = torch.cat([torch.zeros(5, dtype=torch.long), torch.ones(5, dtype=torch.long)])

    total_no_mmd, metrics_no_mmd = correction_loss(
        original, corrected, labels, batch_ids=batch_ids, mmd_weight=0.0
    )
    total_with_mmd, metrics_with_mmd = correction_loss(
        original, corrected, labels, batch_ids=batch_ids, mmd_weight=1.0
    )

    assert metrics_no_mmd["mmd"] == metrics_with_mmd["mmd"]  # same value computed either way
    assert metrics_with_mmd["mmd"] > 0.0
    assert total_with_mmd.item() > total_no_mmd.item()  # weight 1 pulls it into total, weight 0 doesn't
