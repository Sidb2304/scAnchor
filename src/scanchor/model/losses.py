"""Training objective: contrastive alignment + donor consistency + a variance-floor guard.

The variance-floor term exists because the scIB benchmarking blind spot means
a contrastive loss alone can quietly collapse within-cell-type variance to
minimize itself, appearing to "integrate well" while destroying exactly the
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
    has donors crossed with batches; a donor represented in only one batch
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

    Computed under no_grad, since the bandwidth is a fixed scale choice per
    minibatch, not something we want gradients flowing back through (that
    would let the loss cheat by shrinking the kernel width instead of
    actually moving the embeddings).
    """
    with torch.no_grad():
        dist = torch.cdist(x, x, p=2)
        n = dist.shape[0]
        off_diag = dist[~torch.eye(n, dtype=torch.bool, device=dist.device)]
        return off_diag.median().clamp(min=1e-6)


_MMD_KERNEL_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def _rbf_kernel_mean(a: torch.Tensor, b: torch.Tensor, sigma: torch.Tensor, multi_scale: bool) -> torch.Tensor:
    """Mean RBF kernel value between every row of `a` and every row of `b`.

    `multi_scale=True` sums the kernel at `_MMD_KERNEL_SCALES` multiples of
    `sigma` and averages, instead of using `sigma` alone. This is a standard
    MMD variant (e.g. Long et al.'s Deep Adaptation Networks) meant to make the
    loss less sensitive to picking exactly the right bandwidth, since a
    single median-heuristic estimate can be off for any given minibatch.
    Averaging (not summing) the per-scale kernels keeps the aggregate at a
    comparable magnitude to the single-scale version, so an `mmd_weight`
    tuned for one is a reasonable starting point for the other, not a
    completely different scale.
    """
    sq_dist = torch.cdist(a, b, p=2).pow(2)
    if not multi_scale:
        return torch.exp(-sq_dist / (2 * sigma.pow(2))).mean()
    total = torch.zeros((), device=a.device)
    for scale in _MMD_KERNEL_SCALES:
        total = total + torch.exp(-sq_dist / (2 * (sigma * scale).pow(2))).mean()
    return total / len(_MMD_KERNEL_SCALES)


def _pairwise_mmd_sum(
    embeddings: torch.Tensor,
    batch_ids: torch.Tensor,
    sigma: torch.Tensor | None = None,
    multi_scale: bool = False,
) -> tuple[torch.Tensor, int]:
    """Sum of pairwise MMD^2 across every pair of batches present, and the pair count.

    Shared kernel math for both `mmd_loss` (global) and
    `class_conditional_mmd_loss` (per cell type) below, factored out so
    both compute the exact same RBF-MMD rather than risking the two drifting
    apart. Returns (0, 0) if fewer than 2 batches, or every batch present
    has <2 cells; callers turn that into an inert 0 loss.

    `sigma`: pass a precomputed bandwidth to use instead of the median
    heuristic on this call's own `embeddings`. `class_conditional_mmd_loss`
    uses this, since computing the bandwidth fresh on each small per-cell-type
    subset gave a noisy, inconsistent length scale from one cell type (and
    one minibatch) to the next, which a real sweep found destabilized
    training rather than helping (see README). `mmd_loss` doesn't pass this,
    so its behavior (and the dose-response numbers already validated
    against it) is unchanged.

    `multi_scale`: see `_rbf_kernel_mean`. Defaults to False everywhere, so
    existing callers (and their already-validated numbers) are unaffected
    unless they explicitly opt in.
    """
    unique_batches = batch_ids.unique()
    if unique_batches.numel() < 2:
        return torch.zeros((), device=embeddings.device), 0

    if sigma is None:
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
            k_xx = _rbf_kernel_mean(x, x, sigma, multi_scale)
            k_yy = _rbf_kernel_mean(y, y, sigma, multi_scale)
            k_xy = _rbf_kernel_mean(x, y, sigma, multi_scale)
            total = total + (k_xx + k_yy - 2 * k_xy)
            n_pairs += 1

    return total, n_pairs


def mmd_loss(embeddings: torch.Tensor, batch_ids: torch.Tensor, multi_scale: bool = False) -> torch.Tensor:
    """Maximum Mean Discrepancy between every pair of batches in this minibatch.

    An explicit alternative mechanism to the adversarial discriminator, not
    another variant of it. Adversarial training relies on a classifier
    "keeping up" in a min-max game, which is fragile by construction, and the
    exact failure mode hit earlier in this project (a fixed adversarial
    strength caused runaway divergence). MMD has no learnable parameters and
    no min-max dynamics: it's a direct, differentiable statistical distance
    between each batch's embedding distribution and every other batch's,
    using an RBF kernel. Motivated by a real result, not just theory: Harmony,
    which works via distributional alignment rather than an adversarial
    classifier, measurably improved batch-mixing on this exact data/metric
    where the adversarial approach here consistently regressed it.

    Requires >=2 batches with >=2 cells each in the minibatch to compute
    anything meaningful; returns 0 (inert) otherwise, same pattern as
    donor_consistency_loss.

    `multi_scale=False` (default) reproduces the exact single-bandwidth
    formula this project's dose-response sweep was validated against;
    set True to opt into the multi-kernel variant (see `_rbf_kernel_mean`),
    a separate, not-yet-validated mechanism.
    """
    total, n_pairs = _pairwise_mmd_sum(embeddings, batch_ids, multi_scale=multi_scale)
    if n_pairs == 0:
        return torch.zeros((), device=embeddings.device)
    return total / n_pairs


def class_conditional_mmd_loss(
    embeddings: torch.Tensor, batch_ids: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """MMD computed within each cell type separately, not globally.

    Global `mmd_loss` matches each batch's overall embedding distribution to
    every other batch's; it has no way to tell "batch structure" apart
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
    enough structure, rather than averaging per-cell-type averages. This
    weights each valid (cell type, batch pair) equally regardless of how
    many cell types contributed one that minibatch. Returns 0 (inert) if no
    cell type had enough structure to compute anything, same pattern as
    `mmd_loss` and `donor_consistency_loss`.

    Uses ONE bandwidth computed from all cells in this minibatch (same scale
    `mmd_loss` itself would use), not a bandwidth recomputed per cell type.
    The first version of this function recomputed the median heuristic on
    each small per-cell-type subset, and a real sweep found that destabilized
    training when weighted meaningfully (worse cell-type purity than global
    MMD alone, the opposite of the goal) rather than fixing the composition
    confound it targeted. A shared bandwidth keeps the length scale
    consistent across cell types and across minibatches, which a per-subset
    estimate on a handful of cells can't.
    """
    sigma = _median_heuristic_sigma(embeddings)
    total = torch.zeros((), device=embeddings.device)
    n_pairs = 0
    for label in labels.unique():
        mask = labels == label
        if mask.sum() < 4:
            continue
        label_total, label_n_pairs = _pairwise_mmd_sum(embeddings[mask], batch_ids[mask], sigma=sigma)
        total = total + label_total
        n_pairs += label_n_pairs

    if n_pairs == 0:
        return torch.zeros((), device=embeddings.device)
    return total / n_pairs


def _sinkhorn_pairwise(a: torch.Tensor, b: torch.Tensor, epsilon: float = 0.1, n_iters: int = 50) -> torch.Tensor:
    """Entropic-regularized OT cost (Sinkhorn distance) between two point clouds.

    Log-space dual (Sinkhorn-Knopp) fixed point, not the raw-space scaling
    iteration, since raw-space repeatedly multiplies probability-scale factors
    and underflows to exactly 0 within a handful of iterations for anything
    but a tiny toy example. `cost` is rescaled by its own median before the
    iteration (the same median-heuristic convention `_median_heuristic_sigma`
    uses for the MMD kernels above) purely to keep epsilon in a numerically
    sane range regardless of the embedding's raw distance scale.

    Each dual-update line REPLACES f/g rather than accumulating onto the
    previous iteration's value (`f = ...`, not `f = f + ...`): the
    correct Sinkhorn fixed point recomputes each potential directly from
    the other at every step. An earlier version of this function
    accumulated instead, which is not standard Sinkhorn and diverged to
    NaN within a few iterations regardless of epsilon or cost scaling,
    caught and fixed during the architecture experiment this was ported
    from (see scanchor-architecture-experiment/).
    """
    cost = torch.cdist(a, b, p=2) ** 2
    cost = cost / cost.median().clamp(min=1e-8)
    n_a, n_b = cost.shape
    log_mu = -torch.log(torch.full((n_a,), float(n_a), device=cost.device))
    log_nu = -torch.log(torch.full((n_b,), float(n_b), device=cost.device))
    f = torch.zeros(n_a, device=cost.device)
    g = torch.zeros(n_b, device=cost.device)
    for _ in range(n_iters):
        f = epsilon * (log_mu - torch.logsumexp((-cost + g[None, :]) / epsilon, dim=1))
        g = epsilon * (log_nu - torch.logsumexp((-cost + f[:, None]) / epsilon, dim=0))
    log_plan = (-cost + f[:, None] + g[None, :]) / epsilon
    plan = log_plan.exp()
    return (plan * cost).sum()


def sinkhorn_ot_loss(
    embeddings: torch.Tensor, batch_ids: torch.Tensor, epsilon: float = 0.1, n_iters: int = 50
) -> torch.Tensor:
    """Sum of pairwise Sinkhorn-OT cost across every pair of batches present, averaged over pairs.

    An explicit matching-based alternative to `mmd_loss`'s moment-matching:
    where MMD only requires the two batches' embedding distributions to
    share the same kernel-mean statistics, Sinkhorn explicitly solves for
    an (entropy-smoothed) minimum-cost matching between every batch-A cell
    and every batch-B cell. Motivated by the same real evidence that
    motivated trying neighbor-attention: every moment-matching mechanism
    already validated in this project (adversarial discriminator, every
    MMD variant) lands on the same batch-mixing-vs-cell-type-purity
    trade-off curve regardless of the specific loss used, real evidence
    the limitation could be about *mechanism class* (moment-matching),
    not the specific loss function. Sinkhorn tests that directly by using
    a mechanism from a genuinely different class.

    Real, seed-checked result (3 seeds) on the Stephenson/scGPT reference
    panel already used for the published MMD numbers (see README's
    Current results): at sinkhorn_weight=0.5, this is not just another
    point on the trade-off curve: both batch-mixing regression (+0.030
    vs. MMD's +0.120) and cell-type-purity improvement (+0.091 vs. MMD's
    +0.085) beat the published MMD mechanism simultaneously, with tight
    across-seed variance. NOT yet validated on the donor-retrieval /
    replicate-structure / cross-backbone axes that MMD's full "Current
    results" were checked against. Off by default until those run;
    treat as a promising, real, but narrower-scope result than MMD's.

    Used only as a TRAINING-time loss: the transport plan itself is
    never part of the forward pass or used at inference. That's what
    keeps the correction head fully inductive despite OT's usual reliance
    on having the target batch's cells available in advance: a new/unseen
    batch at inference time never needs its own transport plan solved,
    only the already-trained correction head's forward pass.

    Requires >=2 batches with >=2 cells each to compute anything
    meaningful; returns 0 (inert) otherwise, same pattern as `mmd_loss`.
    """
    unique_batches = batch_ids.unique()
    if unique_batches.numel() < 2:
        return torch.zeros((), device=embeddings.device)
    total = torch.zeros((), device=embeddings.device)
    n_pairs = 0
    batch_list = unique_batches.tolist()
    for i in range(len(batch_list)):
        a = embeddings[batch_ids == batch_list[i]]
        if a.shape[0] < 2:
            continue
        for j in range(i + 1, len(batch_list)):
            b = embeddings[batch_ids == batch_list[j]]
            if b.shape[0] < 2:
                continue
            total = total + _sinkhorn_pairwise(a, b, epsilon=epsilon, n_iters=n_iters)
            n_pairs += 1
    if n_pairs == 0:
        return torch.zeros((), device=embeddings.device)
    return total / n_pairs


def adversarial_batch_loss(batch_logits: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of a batch discriminator's predictions.

    Call with logits from `BatchDiscriminator(corrected, lambd)`. The
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
    mmd_multi_scale: bool = False,
    conditional_mmd_weight: float = 0.0,
    sinkhorn_weight: float = 0.0,
    sinkhorn_epsilon: float = 0.1,
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

    # Same cross-entropy math as the adversarial term above; the direction
    # (fight batch signal vs. absorb it) comes entirely from whether the
    # logits passed in went through a GRL (BatchDiscriminator) or not
    # (BatchAbsorber), not from anything in this loss function.
    absorption_term = torch.zeros((), device=corrected.device)
    if absorber_logits is not None and batch_ids is not None:
        absorption_term = adversarial_batch_loss(absorber_logits, batch_ids)
        total = total + absorption_weight * absorption_term

    mmd_term = torch.zeros((), device=corrected.device)
    if batch_ids is not None:
        mmd_term = mmd_loss(corrected, batch_ids, multi_scale=mmd_multi_scale)
        total = total + mmd_weight * mmd_term

    conditional_mmd_term = torch.zeros((), device=corrected.device)
    if batch_ids is not None:
        conditional_mmd_term = class_conditional_mmd_loss(corrected, batch_ids, labels)
        total = total + conditional_mmd_weight * conditional_mmd_term

    sinkhorn_term = torch.zeros((), device=corrected.device)
    if batch_ids is not None:
        sinkhorn_term = sinkhorn_ot_loss(corrected, batch_ids, epsilon=sinkhorn_epsilon)
        total = total + sinkhorn_weight * sinkhorn_term

    return total, {
        "contrastive": contrastive.item(),
        "variance_penalty": variance_penalty.item(),
        "donor_consistency": donor_term.item(),
        "adversarial_batch": adversarial_term.item(),
        "batch_absorption": absorption_term.item(),
        "mmd": mmd_term.item(),
        "conditional_mmd": conditional_mmd_term.item(),
        "sinkhorn_ot": sinkhorn_term.item(),
        "total": total.item(),
    }
