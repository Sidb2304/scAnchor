"""Training objective: supervised contrastive alignment + a variance-floor guard.

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
    has_positive = same_label.any(dim=1)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # torch.where, not `log_prob * same_label`: the diagonal is -inf and
    # always masked out, and 0 * -inf is NaN under IEEE float rules.
    masked_log_prob = torch.where(same_label, log_prob, torch.zeros_like(log_prob))
    positive_log_prob = masked_log_prob.sum(dim=1) / same_label.sum(dim=1).clamp(min=1)
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


def correction_loss(
    original: torch.Tensor,
    corrected: torch.Tensor,
    labels: torch.Tensor,
    contrastive_weight: float = 1.0,
    variance_weight: float = 1.0,
    temperature: float = 0.1,
    min_variance_ratio: float = 0.8,
) -> tuple[torch.Tensor, dict[str, float]]:
    contrastive = supervised_contrastive_loss(corrected, labels, temperature=temperature)
    variance_penalty = variance_floor_penalty(original, corrected, labels, min_ratio=min_variance_ratio)
    total = contrastive_weight * contrastive + variance_weight * variance_penalty
    return total, {
        "contrastive": contrastive.item(),
        "variance_penalty": variance_penalty.item(),
        "total": total.item(),
    }
