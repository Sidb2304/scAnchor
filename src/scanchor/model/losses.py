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


def _median_heuristic_sigma(x: torch.Tensor) -> torch.Tensor:
    """Standard RBF-kernel bandwidth choice: the median pairwise distance.

    Computed under no_grad -- the bandwidth is a fixed scale choice per
    minibatch, not something we want gradients flowing back through (that
    would let the loss cheat by shrinking the kernel width instead of
    actually moving the embeddings).
    """
    with torch.no_grad():
        dist = torch.cdist(x, x, p=2)
        n = dist.shape[0]
        off_diag = dist[~torch.eye(n, dtype=torch.bool, device=dist.device)]
        return off_diag.median().clamp(min=1e-6)


def _pairwise_mmd_sum(embeddings: torch.Tensor, batch_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Sum of pairwise MMD^2 across every pair of batches present, and the pair count.

    Shared kernel math for both `mmd_loss` (global) and
    `class_conditional_mmd_loss` (per cell type) below -- factored out so
    both compute the exact same RBF-MMD rather than risking the two drifting
    apart. Returns (0, 0) if fewer than 2 batches, or every batch present
    has <2 cells -- callers turn that into an inert 0 loss.
    """
    unique_batches = batch_ids.unique()
    if unique_batches.numel() < 2:
        return torch.zeros((), device=embeddings.device), 0

    sigma = _median_heuristic_sigma(embeddings)
    total = torch.zeros((), device=embeddings.device)
    n_pairs = 0
    batch_list = unique_batches.tolist()
    for i in range(len(batch_list)):
        x = embeddings[batch_ids == batch_list[i]]
        if x.shape[0] < 2:
            continue
        for j in range(i + 1, len(batch_list)):
            y = embeddings[batch_ids == batch_list[j]]
            if y.shape[0] < 2:
                continue
            k_xx = torch.exp(-torch.cdist(x, x, p=2).pow(2) / (2 * sigma.pow(2))).mean()
            k_yy = torch.exp(-torch.cdist(y, y, p=2).pow(2) / (2 * sigma.pow(2))).mean()
            k_xy = torch.exp(-torch.cdist(x, y, p=2).pow(2) / (2 * sigma.pow(2))).mean()
            total = total + (k_xx + k_yy - 2 * k_xy)
            n_pairs += 1

    return total, n_pairs


def mmd_loss(embeddings: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
    """Maximum Mean Discrepancy between every pair of batches in this minibatch.

    An explicit alternative mechanism to the adversarial discriminator, not
    another variant of it. Adversarial training relies on a classifier
    "keeping up" in a min-max game -- fragile by construction, and the exact
    failure mode hit earlier in this project (a fixed adversarial strength
    caused runaway divergence). MMD has no learnable parameters and no
    min-max dynamics: it's a direct, differentiable statistical distance
    between each batch's embedding distribution and every other batch's,
    using an RBF kernel. Motivated by a real result, not just theory: Harmony
    -- which works via distributional alignment, not an adversarial
    classifier -- measurably improved batch-mixing on this exact data/metric
    where the adversarial approach here consistently regressed it.

    Requires >=2 batches with >=2 cells each in the minibatch to compute
    anything meaningful; returns 0 (inert) otherwise, same pattern as
    donor_consistency_loss.
    """
    total, n_pairs = _pairwise_mmd_sum(embeddings, batch_ids)
    if n_pairs == 0:
        return torch.zeros((), device=embeddings.device)
    return total / n_pairs


def class_conditional_mmd_loss(
    embeddings: torch.Tensor, batch_ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """MMD computed within each cell type separately, not globally.

    Global `mmd_loss` matches each batch's overall embedding distribution to
    every other batch's -- it has no way to tell "batch structure" apart
    from "these batches just happen to have different cell-type
    composition," so at high weight it can pull cell types together as
    readily as it removes real batch structure (this project's real
    dose-response sweep found exactly that: cell-type purity degrades
    monotonically as `mmd_weight` increases, see README). Pooling only
    same-cell-type cells across batches before computing MMD targets batch
    structure specifically and leaves cross-cell-type structure alone.

    Skips any cell type with fewer than 4 cells in this minibatch (not
    enough to plausibly split across >=2 batches with >=2 cells each) and
    pools the pairwise MMD sum/count across every cell type that did have
    enough structure, rather than averaging per-cell-type averages -- this
    weights each valid (cell type, batch pair) equally regardless of how
    many cell types contributed one that minibatch. Returns 0 (inert) if no
    cell type had enough structure to compute anything, same pattern as
    `mmd_loss` and `donor_consistency_loss`.
    """
    total = torch.zeros((), device=embeddings.device)
    n_pairs = 0
    for label in labels.unique():
        mask = labels == label
        if mask.sum() < 4:
            continue
        label_total, label_n_pairs = _pairwise_mmd_sum(embeddings[mask], batch_ids[mask])
        total = total + label_total
        n_pairs += label_n_pairs

    if n_pairs == 0:
        return torch.zeros((), device=embeddings.device)
    return total / n_pairs


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
    mmd_weight: float = 0.0,
    conditional_mmd_weight: float = 0.0,
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

    mmd_term = torch.zeros((), device=corrected.device)
    if batch_ids is not None:
        mmd_term = mmd_loss(corrected, batch_ids)
        total = total + mmd_weight * mmd_term

    conditional_mmd_term = torch.zeros((), device=corrected.device)
    if batch_ids is not None:
        conditional_mmd_term = class_conditional_mmd_loss(corrected, batch_ids, labels)
        total = total + conditional_mmd_weight * conditional_mmd_term

    return total, {
        "contrastive": contrastive.item(),
        "variance_penalty": variance_penalty.item(),
        "donor_consistency": donor_term.item(),
        "adversarial_batch": adversarial_term.item(),
        "batch_absorption": absorption_term.item(),
        "mmd": mmd_term.item(),
        "conditional_mmd": conditional_mmd_term.item(),
        "total": total.item(),
    }
