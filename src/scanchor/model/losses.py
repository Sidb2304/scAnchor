"""Training objective: contrastive alignment + donor consistency + a variance-floor guard.

The variance-floor term exists because the scIB benchmarking blind spot means
a contrastive loss alone can quietly collapse within-cell-type variance to
minimize itself — appearing to "integrate well" while destroying exactly the
biological heterogeneity the correction is supposed to preserve. We guard
against that directly during training rather than relying on it showing up
in a post-hoc metric.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mean_positive_log_prob(sim: torch.Tensor, positive_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean log-prob of positives per row, and which rows have >=1 positive.

    Shared by both contrastive losses below. Uses torch.where rather than
    `log_prob * positive_mask`: sim's diagonal is -inf (self-similarity is
    always excluded), and 0 * -inf is NaN under IEEE float rules.
    """
    has_positive = positive_mask.any(dim=1)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    masked_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    positive_log_prob = masked_log_prob.sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
    return positive_log_prob, has_positive


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """SupCon loss: pulls same-label embeddings together, pushes others apart."""
    z = F.normalize(embeddings, dim=-1)
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))

    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    same_label.fill_diagonal_(False)

    positive_log_prob, has_positive = _mean_positive_log_prob(sim, same_label)
    return -positive_log_prob[has_positive].mean()


def donor_consistency_loss(
    embeddings: torch.Tensor,
    donor_ids: torch.Tensor,
    batch_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Pulls together same-donor cells that come from *different* batches.

    This is the mechanism that's actually supposed to make correction
    donor-preserving rather than just batch-mixing: positives are restricted
    to same-donor-different-batch pairs (not same-donor-same-batch, which is
    already trivially satisfied and wouldn't teach the head anything about
    cross-batch donor identity). It's only learnable when the reference panel
    has donors crossed with batches — a donor represented in only one batch
    gives no valid positive pair and is excluded via `has_positive`.

    Cells with donor_ids < 0 (donor identity unknown/not applicable) are
    dropped before computing the loss.
    """
    valid = donor_ids >= 0
    if valid.sum() < 2:
        return torch.zeros((), device=embeddings.device)

    z = F.normalize(embeddings[valid], dim=-1)
    donor_ids = donor_ids[valid]
    batch_ids = batch_ids[valid]

    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))

    same_donor = donor_ids.unsqueeze(0) == donor_ids.unsqueeze(1)
    different_batch = batch_ids.unsqueeze(0) != batch_ids.unsqueeze(1)
    positive_mask = same_donor & different_batch
    positive_mask.fill_diagonal_(False)

    positive_log_prob, has_positive = _mean_positive_log_prob(sim, positive_mask)
    if not has_positive.any():
        return torch.zeros((), device=embeddings.device)
    return -positive_log_prob[has_positive].mean()


def variance_floor_penalty(
    original: torch.Tensor,
    corrected: torch.Tensor,
    labels: torch.Tensor,
    min_ratio: float = 0.8,
) -> torch.Tensor:
    """Penalize within-label variance dropping below `min_ratio` of the original.

    Computed per unique label present in the batch; labels with fewer than 2
    cells are skipped (variance undefined).
    """
    penalty = torch.zeros((), device=original.device)
    n_terms = 0
    for label in labels.unique():
        mask = labels == label
        if mask.sum() < 2:
            continue
        orig_var = original[mask].var(dim=0, unbiased=True).mean()
        corr_var = corrected[mask].var(dim=0, unbiased=True).mean()
        ratio = corr_var / orig_var.clamp(min=1e-8)
        penalty = penalty + F.relu(min_ratio - ratio)
        n_terms += 1
    return penalty / max(n_terms, 1)


def adversarial_batch_loss(batch_logits: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of a batch discriminator's predictions.

    Call with logits from `BatchDiscriminator(corrected, lambd)` — the
    discriminator's forward pass applies a gradient-reversal layer, so a
    single backward() through this loss trains the discriminator normally
    (better at predicting batch) while pushing the correction head to make
    batch identity LESS predictable. Plain cross-entropy here, deliberately;
    the adversarial direction lives entirely in the discriminator's GRL, not
    in this loss function, so it stays trivially testable on its own.
    """
    return F.cross_entropy(batch_logits, batch_ids)


def correction_loss(
    original: torch.Tensor,
    corrected: torch.Tensor,
    labels: torch.Tensor,
    donor_ids: torch.Tensor | None = None,
    batch_ids: torch.Tensor | None = None,
    batch_logits: torch.Tensor | None = None,
    absorber_logits: torch.Tensor | None = None,
    contrastive_weight: float = 1.0,
    variance_weight: float = 1.0,
    donor_weight: float = 1.0,
    adversarial_weight: float = 1.0,
    absorption_weight: float = 1.0,
    temperature: float = 0.1,
    min_variance_ratio: float = 0.8,
) -> tuple[torch.Tensor, dict[str, float]]:
    contrastive = supervised_contrastive_loss(corrected, labels, temperature=temperature)
    variance_penalty = variance_floor_penalty(original, corrected, labels, min_ratio=min_variance_ratio)
    total = contrastive_weight * contrastive + variance_weight * variance_penalty

    donor_term = torch.zeros((), device=corrected.device)
    if donor_ids is not None and batch_ids is not None:
        donor_term = donor_consistency_loss(corrected, donor_ids, batch_ids, temperature=temperature)
        total = total + donor_weight * donor_term

    adversarial_term = torch.zeros((), device=corrected.device)
    if batch_logits is not None and batch_ids is not None:
        adversarial_term = adversarial_batch_loss(batch_logits, batch_ids)
        total = total + adversarial_weight * adversarial_term

    # Same cross-entropy math as the adversarial term above -- the direction
    # (fight batch signal vs. absorb it) comes entirely from whether the
    # logits passed in went through a GRL (BatchDiscriminator) or not
    # (BatchAbsorber), not from anything in this loss function.
    absorption_term = torch.zeros((), device=corrected.device)
    if absorber_logits is not None and batch_ids is not None:
        absorption_term = adversarial_batch_loss(absorber_logits, batch_ids)
        total = total + absorption_weight * absorption_term

    return total, {
        "contrastive": contrastive.item(),
        "variance_penalty": variance_penalty.item(),
        "donor_consistency": donor_term.item(),
        "adversarial_batch": adversarial_term.item(),
        "batch_absorption": absorption_term.item(),
        "total": total.item(),
    }
