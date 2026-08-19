"""Real replicate-structure test for Sinkhorn OT (v1.3.0's sinkhorn_weight)
against the shipped mmd_weight=20 default, on the actual private Levy
astrocyte mini-village dataset -- the third validation axis (after the
single-dataset seed-check and Geneformer cross-backbone check, both
already real/positive) that MMD's default status was originally earned on
but Sinkhorn hadn't been checked against yet (see README's Current
results / Status).

Reuses ALREADY-CACHED real scGPT embeddings (levy_run/{reference,heldout}.h5ad,
extracted via submit_uger_levy_full_dataset.sh's GPU job) -- this script
itself needs no GPU, training a small correction head on frozen embeddings
is cheap on CPU.

Purity metrics (batch_mixing_purity, label_knn_purity) are evaluated on a
20k-cell random SUBSAMPLE of the combined reference+held-out set, not the
full ~81k cells: both metrics use sklearn NearestNeighbors, which falls
back to brute-force distance computation for 512-dim embeddings (tree
methods don't help past ~20-30 dims) -- an O(n^2) cost that, run 4 times
(before/after x 2 metrics) on the full ~81k cells, was real enough to risk
exceeding a 6-hour cluster walltime request. This is the first dataset in
this project's history run at Levy's true full scale (every prior dataset
topped out around 19-21k cells), so this scalability wall was never hit
before. donor_retrieval_accuracy is NOT subsampled -- it operates on
(donor, batch) centroids (at most 8x9=72 of them here), independent of
cell count, so it's cheap regardless of scale.

Training deliberately does NOT call the shared `correction_loss` wrapper.
That function always computes mmd_loss, class_conditional_mmd_loss, AND
sinkhorn_ot_loss every step regardless of their weight (by design -- its
metrics dict is meant to report each term's real value even when unused,
see tests/test_losses.py). At Stephenson's 3-batch scale that's cheap and
never mattered; at Levy's real 8-batch x 14-cell-type scale, that's up to
C(8,2)=28 batch-pairs x 14 cell-types = 392 pairwise kernel/Sinkhorn
computations per minibatch for terms whose weight is 0 in every run here
(conditional_mmd always, plus whichever of mmd/sinkhorn isn't this run's
active mechanism) -- a first, real attempt at this exact comparison on
Stephenson-scale local CPU never finished even one seed in ~15 minutes CPU
time. Composing the loss by hand here -- calling only the individual
(already public, tested) functions actually needed -- is 100%
behavior-preserving: a term multiplied by weight=0 contributes exactly 0
to `total` either way, so skipping its computation changes nothing about
the trained model or its evaluation numbers, only the wall-clock cost.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE / "src"))

from scanchor.data.covariates import CovariateVocab
from scanchor.data.datasets import EmbeddedCellDataset
from scanchor.evaluate.metrics import batch_mixing_purity, donor_retrieval_accuracy, label_knn_purity
from scanchor.model.correction_head import CorrectionHead
from scanchor.model.losses import (
    adversarial_batch_loss,
    donor_consistency_loss,
    mmd_loss,
    sinkhorn_ot_loss,
    supervised_contrastive_loss,
    variance_floor_penalty,
)

LEVY_RUN = HERE / "levy_run"
REFERENCE_PATH = LEVY_RUN / "reference.h5ad"
HELD_OUT_PATH = LEVY_RUN / "heldout.h5ad"

EMBED_DIM = 512
EPOCHS = 30
SEEDS = (0, 1, 2)
EVAL_SUBSAMPLE_N = 20_000
EVAL_SUBSAMPLE_SEED = 0  # fixed across all runs -- same eval cells for every config/seed, a fair comparison

MECHANISMS = {
    "mmd20": {"kind": "mmd", "weight": 20.0},
    "sinkhorn0.5": {"kind": "sinkhorn", "weight": 0.5, "sinkhorn_epsilon": 0.1},
}


def run_one(mechanism, seed, reference, held_out, vocab, device):
    torch.manual_seed(seed)

    dataset = EmbeddedCellDataset(
        reference, vocab, embedding_key="X_scGPT", cell_type_col="cell_type",
        batch_col="batch", donor_col="donor_id",
    )
    loader = DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True)

    head = CorrectionHead(
        embed_dim=EMBED_DIM, vocab_sizes=vocab.vocab_sizes(), n_continuous=2,
        cat_embed_dim=8, covariate_dim=32, hidden_dim=128, max_delta_ratio=1.0,
    ).to(device)
    # BatchAbsorber deliberately not built at all: absorption_weight=0.0 in
    # every run here, so its output was never used -- see module docstring.
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    t0 = time.time()
    for epoch in range(EPOCHS):
        for embedding, categorical, continuous, cell_type, batch_code, donor_code in loader:
            embedding = embedding.to(device)
            categorical = categorical.to(device)
            continuous = continuous.to(device)
            cell_type = cell_type.to(device)
            batch_code = batch_code.to(device)
            donor_code = donor_code.to(device)
            corrected = head(embedding, categorical, continuous)

            # Hand-composed loss -- see module docstring for why this doesn't call
            # the shared correction_loss wrapper. Only computes the ONE active
            # mechanism's batch-mixing term, and skips class_conditional_mmd_loss
            # entirely (always 0-weighted in both configs tested here).
            contrastive = supervised_contrastive_loss(corrected, cell_type, temperature=0.1)
            variance_penalty = variance_floor_penalty(embedding, corrected, cell_type, min_ratio=0.8)
            donor_term = donor_consistency_loss(corrected, donor_code, batch_code, temperature=0.1)
            loss = contrastive + variance_penalty + donor_term
            if mechanism["kind"] == "mmd":
                loss = loss + mechanism["weight"] * mmd_loss(corrected, batch_code)
            elif mechanism["kind"] == "sinkhorn":
                loss = loss + mechanism["weight"] * sinkhorn_ot_loss(
                    corrected, batch_code, epsilon=mechanism["sinkhorn_epsilon"]
                )
            else:
                raise ValueError(f"unknown mechanism kind: {mechanism['kind']!r}")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
            optimizer.step()
    elapsed = time.time() - t0

    combined = ad.concat([reference, held_out], join="outer", index_unique="-c")
    head.eval()
    with torch.no_grad():
        combined_embedding = torch.from_numpy(np.asarray(combined.obsm["X_scGPT"], dtype=np.float32)).to(device)
        combined_cat = torch.from_numpy(vocab.encode_categorical(combined)).to(device)
        combined_cont = torch.from_numpy(vocab.encode_continuous(combined)).to(device)
        corrected_combined = head(combined_embedding, combined_cat, combined_cont).cpu().numpy()

    batch_ids = combined.obs["batch"].astype(str).to_numpy()
    cell_types = combined.obs["cell_type"].astype(str).to_numpy()
    donor_ids = combined.obs["donor_id"].astype(str).to_numpy()
    embedding_before = combined.obsm["X_scGPT"]

    # donor retrieval: cheap (centroid-based), full set, no subsampling needed.
    donor_before = donor_retrieval_accuracy(embedding_before, donor_ids, batch_ids)
    donor_after = donor_retrieval_accuracy(corrected_combined, donor_ids, batch_ids)

    # purity metrics: subsampled, same fixed subsample across every run for a fair comparison.
    rng = np.random.default_rng(EVAL_SUBSAMPLE_SEED)
    sub_idx = rng.choice(len(combined), size=min(EVAL_SUBSAMPLE_N, len(combined)), replace=False)
    bmix_before = batch_mixing_purity(embedding_before[sub_idx], batch_ids[sub_idx])
    bmix_after = batch_mixing_purity(corrected_combined[sub_idx], batch_ids[sub_idx])
    knn_before = label_knn_purity(embedding_before[sub_idx], cell_types[sub_idx])
    knn_after = label_knn_purity(corrected_combined[sub_idx], cell_types[sub_idx])

    return {
        "seed": seed,
        "elapsed_s": elapsed,
        "donor_retrieval_before": donor_before,
        "donor_retrieval_after": donor_after,
        "batch_mixing_purity_before": bmix_before,
        "batch_mixing_purity_after": bmix_after,
        "label_knn_purity_before": knn_before,
        "label_knn_purity_after": knn_after,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    reference = ad.read_h5ad(REFERENCE_PATH)
    held_out = ad.read_h5ad(HELD_OUT_PATH)
    print(f"reference: {reference.n_obs} cells, held-out: {held_out.n_obs} cells "
          f"(eval subsampled to {EVAL_SUBSAMPLE_N} of {reference.n_obs + held_out.n_obs} combined cells)")

    vocab = CovariateVocab.build(reference, ["batch"], ["total_counts", "pct_counts_mt"])

    all_results = {}
    for name, mechanism in MECHANISMS.items():
        print(f"\n{'=' * 20} {name} {'=' * 20}")
        seed_results = []
        for seed in SEEDS:
            print(f"--- seed {seed} ---")
            r = run_one(mechanism, seed, reference, held_out, vocab, device)
            print(r)
            seed_results.append(r)
        all_results[name] = seed_results

    print("\n\n=== SUMMARY (mean +/- std across seeds 0,1,2) ===")
    header = (f"{'mechanism':>12} | {'donor_ret after':>15} | {'bmix after':>10} | "
              f"{'knn after':>9} | {'bmix delta':>10} | {'knn delta':>9}")
    print(header)
    print("-" * len(header))
    for name, seed_results in all_results.items():
        donor_after = np.array([r["donor_retrieval_after"] for r in seed_results])
        bmix_before = np.array([r["batch_mixing_purity_before"] for r in seed_results])
        bmix_after = np.array([r["batch_mixing_purity_after"] for r in seed_results])
        knn_before = np.array([r["label_knn_purity_before"] for r in seed_results])
        knn_after = np.array([r["label_knn_purity_after"] for r in seed_results])
        print(f"{name:>12} | {donor_after.mean():>7.4f}+-{donor_after.std():<6.4f} | "
              f"{bmix_after.mean():>10.4f} | {knn_after.mean():>9.4f} | "
              f"{(bmix_after - bmix_before).mean():>+10.4f} | {(knn_after - knn_before).mean():>+9.4f}")
    print("\ndonor_retrieval_before (both mechanisms, same raw embeddings):",
          all_results["mmd20"][0]["donor_retrieval_before"])


if __name__ == "__main__":
    main()
