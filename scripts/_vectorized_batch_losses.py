"""Vectorized (batched-over-pairs) MMD and Sinkhorn losses for the Levy
comparison scripts -- NOT part of the shipped scanchor package, and
deliberately NOT a replacement for losses.py's mmd_loss/sinkhorn_ot_loss.

Why this exists: losses.py's versions loop over batch-pairs in Python,
one cdist/Sinkhorn-solve at a time. That's correct and fine at
Stephenson's 3-batch scale (what those functions were validated
against), but at Levy's 8 batches (up to C(8,2)=28 pairs per minibatch,
Sinkhorn needing 50 sequential iterations per pair) that's dozens of
small, sequential GPU kernel launches per minibatch -- real evidence from
run_levy_sinkhorn_comparison.py / run_levy_combined_comparison.py: ~50
min/seed on GPU for the Sinkhorn-containing configs, vs. ~4 min/seed for
MMD-only, at the same dataset/hardware. GPU throughput doesn't fix that
kind of bottleneck; kernel-launch/sync overhead repeated 28x per
minibatch dominates regardless of how fast each individual op is.

This module pads every active batch's per-minibatch cell subset to the
same size and stacks all pairs into one 3D tensor, so cdist and the
Sinkhorn dual iteration run as ONE batched op across every pair at once
instead of 28 sequential ones -- the actual fix, not just "use more GPU".

Deliberately kept separate from losses.py rather than modifying
mmd_loss/sinkhorn_ot_loss in place: those functions are validated (real,
seed-checked published numbers depend on their exact behavior) and
covered by tests/test_losses.py -- changing them, even to something
numerically equivalent, risks that validated history. This module is
verified numerically equivalent to them instead (see
test_vectorized_batch_losses.py), then used ONLY in these Levy
comparison scripts.
"""
from __future__ import annotations

import torch


def _pad_batches(embeddings: torch.Tensor, batch_ids: torch.Tensor):
    """Split embeddings by batch_ids and pad every batch's subset to a
    common size (the largest batch present), returning:
      - padded: (n_batches_present, max_n, dim)
      - mask: (n_batches_present, max_n) bool, True where a row is real
      - unique_batches: the actual batch id for each row of `padded`

    Padded rows are zero-filled -- always masked out downstream, never
    contribute to any sum/mean.
    """
    unique_batches = batch_ids.unique()
    subsets = [embeddings[batch_ids == b] for b in unique_batches]
    max_n = max(s.shape[0] for s in subsets)
    dim = embeddings.shape[-1]
    device = embeddings.device

    padded = torch.zeros(len(subsets), max_n, dim, device=device, dtype=embeddings.dtype)
    mask = torch.zeros(len(subsets), max_n, dtype=torch.bool, device=device)
    for i, s in enumerate(subsets):
        n = s.shape[0]
        padded[i, :n] = s
        mask[i, :n] = True
    return padded, mask, unique_batches


def _pair_indices(n: int, device) -> torch.Tensor:
    """All (i, j), i < j index pairs among n items, as a (n_pairs, 2) tensor."""
    idx = torch.triu_indices(n, n, offset=1, device=device)
    return idx.T  # (n_pairs, 2)


def vectorized_mmd_loss(embeddings: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
    """Same RBF-MMD math as losses.mmd_loss (single bandwidth, median
    heuristic, no multi_scale), batched over every pair of batches at once
    instead of one cdist call per pair. Numerically verified equivalent
    to losses.mmd_loss on random multi-batch data (see the test module).
    """
    unique_batches = batch_ids.unique()
    if unique_batches.numel() < 2:
        return torch.zeros((), device=embeddings.device)

    with torch.no_grad():
        dist = torch.cdist(embeddings, embeddings, p=2)
        n = dist.shape[0]
        off_diag = dist[~torch.eye(n, dtype=torch.bool, device=dist.device)]
        sigma = off_diag.median().clamp(min=1e-6)

    padded, mask, _ = _pad_batches(embeddings, batch_ids)
    n_batches_present = padded.shape[0]
    if n_batches_present < 2:
        return torch.zeros((), device=embeddings.device)

    pairs = _pair_indices(n_batches_present, embeddings.device)
    a = padded[pairs[:, 0]]  # (n_pairs, max_n, dim)
    b = padded[pairs[:, 1]]
    mask_a = mask[pairs[:, 0]]  # (n_pairs, max_n)
    mask_b = mask[pairs[:, 1]]

    def masked_kernel_mean(x, y, mx, my):
        sq_dist = torch.cdist(x, y, p=2).pow(2)  # (n_pairs, max_n, max_n)
        k = torch.exp(-sq_dist / (2 * sigma.pow(2)))
        pair_mask = mx.unsqueeze(2) & my.unsqueeze(1)  # (n_pairs, max_n, max_n)
        k = k * pair_mask
        counts = pair_mask.sum(dim=(1, 2)).clamp(min=1)
        return k.sum(dim=(1, 2)) / counts

    k_xx = masked_kernel_mean(a, a, mask_a, mask_a)
    k_yy = masked_kernel_mean(b, b, mask_b, mask_b)
    k_xy = masked_kernel_mean(a, b, mask_a, mask_b)
    return (k_xx + k_yy - 2 * k_xy).mean()


def vectorized_sinkhorn_ot_loss(
    embeddings: torch.Tensor, batch_ids: torch.Tensor, epsilon: float = 0.1, n_iters: int = 50
) -> torch.Tensor:
    """Same entropic-OT math as losses.sinkhorn_ot_loss (PER-PAIR
    median-rescaled cost -- matching the reference's independent
    per-call rescale, not one shared median across every pair -- and the
    same log-space dual fixed point), batched over every pair of batches
    at once. Padded rows/cols get a large FINITE cost sentinel rather
    than literal +inf: real +inf/-inf values inside the fixed-point loop
    can turn into NaN gradients on backward even when the forward value
    looks fine (an earlier version of this function used literal inf and
    failed its gradient-flow check for exactly this reason). log_mu/log_nu
    use each pair's REAL (unpadded) batch sizes, so the transport plan's
    marginals are unaffected by padding. Numerically verified equivalent
    to losses.sinkhorn_ot_loss on random multi-batch data, including a
    real gradcheck (not just forward-value comparison).
    """
    unique_batches = batch_ids.unique()
    if unique_batches.numel() < 2:
        return torch.zeros((), device=embeddings.device)

    padded, mask, _ = _pad_batches(embeddings, batch_ids)
    n_batches_present = padded.shape[0]
    if n_batches_present < 2:
        return torch.zeros((), device=embeddings.device)

    pairs = _pair_indices(n_batches_present, embeddings.device)
    a = padded[pairs[:, 0]]  # (n_pairs, max_n, dim)
    b = padded[pairs[:, 1]]
    mask_a = mask[pairs[:, 0]]  # (n_pairs, max_n)
    mask_b = mask[pairs[:, 1]]
    n_pairs, max_n, _ = a.shape

    cost = torch.cdist(a, b, p=2) ** 2  # (n_pairs, max_n, max_n)
    pair_mask = mask_a.unsqueeze(2) & mask_b.unsqueeze(1)

    # Per-pair median (the reference rescales EACH pair independently, via
    # its own separate call) -- a one-time, no_grad, n_pairs-sized loop is
    # cheap; it's the 50-ITERATION loop below that this whole module exists
    # to vectorize, not this.
    with torch.no_grad():
        medians = torch.ones(n_pairs, device=cost.device)
        for p in range(n_pairs):
            valid = cost[p][pair_mask[p]]
            if valid.numel() > 0:
                medians[p] = valid.median().clamp(min=1e-8)
    cost = cost / medians.view(n_pairs, 1, 1)

    # Large FINITE sentinel, not inf: after the /epsilon division and
    # logsumexp, this still drives padded entries' contribution to
    # numerically 0 (exp(-BIG/epsilon) underflows cleanly), without
    # introducing an actual inf into the autograd graph.
    BIG = 1e6
    cost = cost.masked_fill(~pair_mask, BIG)

    n_a = mask_a.sum(dim=1).clamp(min=1).float()  # (n_pairs,) real (unpadded) sizes
    n_b = mask_b.sum(dim=1).clamp(min=1).float()
    log_mu = torch.where(mask_a, -torch.log(n_a).unsqueeze(1).expand(-1, max_n), torch.full_like(cost[:, :, 0], -BIG))
    log_nu = torch.where(mask_b, -torch.log(n_b).unsqueeze(1).expand(-1, max_n), torch.full_like(cost[:, 0, :], -BIG))

    f = torch.zeros(n_pairs, max_n, device=cost.device)
    g = torch.zeros(n_pairs, max_n, device=cost.device)
    for _ in range(n_iters):
        f = epsilon * (log_mu - torch.logsumexp((-cost + g.unsqueeze(1)) / epsilon, dim=2))
        g = epsilon * (log_nu - torch.logsumexp((-cost + f.unsqueeze(2)) / epsilon, dim=1))

    log_plan = (-cost + f.unsqueeze(2) + g.unsqueeze(1)) / epsilon
    plan = log_plan.exp()
    per_pair_cost = (plan * cost * pair_mask).sum(dim=(1, 2))
    return per_pair_cost.mean()
