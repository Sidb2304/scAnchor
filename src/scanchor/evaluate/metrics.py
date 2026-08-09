"""Shared, dependency-light metrics for the two evaluation protocols.

Deliberately not scIB metrics: the documented blind spot (batch-correction
tools scoring well on scIB while erasing within-cell-type biological
variation) is exactly what this project is trying not to reproduce. These
metrics are simpler and check a narrower, more literal claim.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def centroid_per_group(embeddings: np.ndarray, group_ids: np.ndarray) -> dict[str, np.ndarray]:
    return {g: embeddings[group_ids == g].mean(axis=0) for g in np.unique(group_ids)}


def donor_retrieval_accuracy(
    embeddings: np.ndarray,
    donor_ids: np.ndarray,
    batch_ids: np.ndarray,
) -> float:
    """Fraction of (donor, batch) centroids whose nearest other centroid is the
    same donor in a different batch, rather than a different donor in the
    same batch. This is the literal "same-donor-across-batch" claim, computed
    directly rather than through an integration score.
    """
    keys = [f"{d}::{b}" for d, b in zip(donor_ids, batch_ids)]
    unique_keys = sorted(set(keys))
    key_to_donor = {k: k.split("::")[0] for k in unique_keys}
    key_to_batch = {k: k.split("::")[1] for k in unique_keys}

    centroids = np.stack(
        [embeddings[np.array(keys) == k].mean(axis=0) for k in unique_keys]
    )

    n = len(unique_keys)
    if n < 3:
        return float("nan")

    dists = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    np.fill_diagonal(dists, np.inf)

    correct = 0
    for i, key in enumerate(unique_keys):
        nearest_idx = int(np.argmin(dists[i]))
        nearest_key = unique_keys[nearest_idx]
        same_donor = key_to_donor[nearest_key] == key_to_donor[key]
        different_batch = key_to_batch[nearest_key] != key_to_batch[key]
        if same_donor and different_batch:
            correct += 1
    return correct / n


def batch_mixing_purity(embeddings: np.ndarray, batch_ids: np.ndarray, k: int = 30) -> float:
    """Average fraction of a cell's k nearest neighbors that share its batch.

    Lower is better (closer to the batch's overall frequency = well mixed).
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    same_batch = batch_ids[indices[:, 1:]] == batch_ids[:, None]
    return float(same_batch.mean())


def label_knn_purity(embeddings: np.ndarray, labels: np.ndarray, k: int = 30) -> float:
    """Average fraction of a cell's k nearest neighbors sharing its label.

    Higher is better — this is the bio-conservation side of the tradeoff.
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    same_label = labels[indices[:, 1:]] == labels[:, None]
    return float(same_label.mean())
