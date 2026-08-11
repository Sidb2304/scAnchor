import torch
import torch.nn.functional as F

from scanchor.model.losses import (
    adversarial_batch_loss,
    class_conditional_mmd_loss,
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
        "adversarial_batch", "batch_absorption", "mmd", "conditional_mmd", "total",
    }
    assert metrics["donor_consistency"] == 0.0  # no donor_ids passed -> term is inert
    assert metrics["adversarial_batch"] == 0.0  # no batch_logits passed -> term is inert
    assert metrics["batch_absorption"] == 0.0  # no absorber_logits passed -> term is inert
    assert metrics["mmd"] == 0.0  # no batch_ids passed -> term is inert
    assert metrics["conditional_mmd"] == 0.0  # no batch_ids passed -> term is inert


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


def test_class_conditional_mmd_loss_zero_with_single_batch():
    embeddings = torch.randn(10, 8)
    batch_ids = torch.zeros(10, dtype=torch.long)
    labels = torch.randint(0, 2, (10,))

    loss = class_conditional_mmd_loss(embeddings, batch_ids, labels)

    assert loss.item() == 0.0


def test_class_conditional_mmd_loss_finite_and_positive_when_shifted_within_label():
    torch.manual_seed(0)
    x = torch.randn(10, 8)
    y = torch.randn(10, 8) + 5.0
    embeddings = torch.cat([x, y], dim=0)
    batch_ids = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    labels = torch.zeros(20, dtype=torch.long)  # single cell type -- same as plain mmd_loss here

    loss = class_conditional_mmd_loss(embeddings, batch_ids, labels)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_class_conditional_mmd_loss_ignores_composition_confound_global_mmd_would_see():
    """The exact scenario class-conditional MMD is meant to fix.

    Two batches with different cell-type *composition* (batch 0 is mostly
    label 0, batch 1 is mostly label 1) but embeddings are drawn from the
    identical distribution within each label regardless of batch -- there
    is no real batch effect, only a composition difference. Global mmd_loss
    picks up the composition difference as if it were batch structure;
    class_conditional_mmd_loss, restricted to same-label comparisons only,
    should not.
    """
    torch.manual_seed(0)
    label0_batch0 = torch.randn(20, 8)
    label0_batch1 = torch.randn(4, 8)  # same distribution, same label, different batch
    label1_batch0 = torch.randn(4, 8) + 8.0  # different label -> different distribution
    label1_batch1 = torch.randn(20, 8) + 8.0  # same distribution, same label, different batch

    embeddings = torch.cat([label0_batch0, label0_batch1, label1_batch0, label1_batch1], dim=0)
    batch_ids = torch.cat([
        torch.zeros(20, dtype=torch.long), torch.ones(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long), torch.ones(20, dtype=torch.long),
    ])
    labels = torch.cat([
        torch.zeros(20, dtype=torch.long), torch.zeros(4, dtype=torch.long),
        torch.ones(4, dtype=torch.long), torch.ones(20, dtype=torch.long),
    ])

    global_term = mmd_loss(embeddings, batch_ids)
    conditional_term = class_conditional_mmd_loss(embeddings, batch_ids, labels)

    assert conditional_term.item() < global_term.item()


def test_class_conditional_mmd_loss_uses_shared_bandwidth_not_per_label():
    """The v0.4.0 fix: one bandwidth for all cell types, not one each.

    Recomputing the median heuristic separately on each small per-cell-type
    subset (the original implementation) gave a noisy, inconsistent length
    scale from one cell type to the next -- a real sweep found this
    destabilized training. Verify the fix directly: patching
    `_median_heuristic_sigma` and counting calls should show it's invoked
    once per `class_conditional_mmd_loss` call (the shared bandwidth), not
    once per cell type present.
    """
    import scanchor.model.losses as losses_module

    torch.manual_seed(0)
    embeddings = torch.cat([torch.randn(10, 8), torch.randn(10, 8) + 3.0], dim=0)
    batch_ids = torch.cat([torch.zeros(5, dtype=torch.long), torch.ones(5, dtype=torch.long)]).repeat(2)
    labels = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])

    call_count = 0
    original = losses_module._median_heuristic_sigma

    def counting_wrapper(x):
        nonlocal call_count
        call_count += 1
        return original(x)

    losses_module._median_heuristic_sigma = counting_wrapper
    try:
        losses_module.class_conditional_mmd_loss(embeddings, batch_ids, labels)
    finally:
        losses_module._median_heuristic_sigma = original

    assert call_count == 1  # one shared bandwidth, not one per cell type (2 labels present)


def test_class_conditional_mmd_loss_stable_with_imbalanced_label_sizes():
    torch.manual_seed(0)
    large_label = torch.randn(40, 8)  # 20 cells/batch, plenty of structure
    small_label = torch.randn(4, 8) + 3.0  # right at the minimum threshold
    embeddings = torch.cat([large_label, small_label], dim=0)
    batch_ids = torch.cat([
        torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long),
        torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long),
    ])
    labels = torch.cat([torch.zeros(40, dtype=torch.long), torch.ones(4, dtype=torch.long)])

    loss = class_conditional_mmd_loss(embeddings, batch_ids, labels)

    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_correction_loss_conditional_mmd_term_inert_when_weight_zero():
    torch.manual_seed(0)
    original = torch.randn(12, 8)
    corrected = original + 5.0 * torch.randn(12, 8)
    # 2 labels x 2 batches x 3 cells each -- guarantees every label has
    # enough same-label cells crossing both batches for the term to be
    # provably nonzero, not just "probably" with random labels.
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1])
    batch_ids = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    total_no_cmmd, metrics_no_cmmd = correction_loss(
        original, corrected, labels, batch_ids=batch_ids, conditional_mmd_weight=0.0
    )
    total_with_cmmd, metrics_with_cmmd = correction_loss(
        original, corrected, labels, batch_ids=batch_ids, conditional_mmd_weight=1.0
    )

    assert metrics_no_cmmd["conditional_mmd"] == metrics_with_cmmd["conditional_mmd"]
    assert metrics_with_cmmd["conditional_mmd"] > 0.0
    assert total_with_cmmd.item() > total_no_cmmd.item()
