"""Seed-check the two most promising combined mmd_weight+sinkhorn_weight
points from run_levy_combined_comparison.py's single-seed sweep:
  - mmd20_sinkhorn0.5: nearly eliminated the batch-mixing regression
    (+0.002 vs mmd alone's +0.018) while still beating mmd alone on donor
    retrieval (0.778 vs 0.708).
  - mmd10_sinkhorn0.5: best donor retrieval of every config tested (0.806),
    beating both individual mechanisms.

Uses scripts/_vectorized_batch_losses.py instead of losses.py's
mmd_loss/sinkhorn_ot_loss -- numerically verified equivalent (see
/private/tmp/.../verify_vectorized_batch_losses.py's 9/9 pass, including
gradcheck), but batches every pair of batches into one op instead of
looping sequentially. run_levy_combined_comparison.py's single-seed sweep
took ~50-55 min/seed for the Sinkhorn-containing configs on this exact
hardware; this is the fix motivated directly by that real cost.
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
sys.path.insert(0, str(HERE / "scripts"))

from scanchor.data.covariates import CovariateVocab
from scanchor.data.datasets import EmbeddedCellDataset
from scanchor.evaluate.metrics import batch_mixing_purity, donor_retrieval_accuracy, label_knn_purity
from scanchor.model.correction_head import CorrectionHead
from scanchor.model.losses import donor_consistency_loss, supervised_contrastive_loss, variance_floor_penalty
from _vectorized_batch_losses import vectorized_mmd_loss, vectorized_sinkhorn_ot_loss

LEVY_RUN = HERE / "levy_run"
REFERENCE_PATH = LEVY_RUN / "reference.h5ad"
HELD_OUT_PATH = LEVY_RUN / "heldout.h5ad"

EMBED_DIM = 512
EPOCHS = 30
SEEDS = (0, 1, 2)
EVAL_SUBSAMPLE_N = 20_000
EVAL_SUBSAMPLE_SEED = 0  # same fixed eval cells as every other Levy comparison script
SINKHORN_EPSILON = 0.1

CONFIGS = {
    "mmd20_sinkhorn0.5": {"mmd_weight": 20.0, "sinkhorn_weight": 0.5},
    "mmd10_sinkhorn0.5": {"mmd_weight": 10.0, "sinkhorn_weight": 0.5},
}

# Single-seed (seed 0) reference points already established, for direct comparison.
PUBLISHED_SEED0 = {
    "mmd20_only": {"donor_retrieval_after": 0.7083, "batch_mixing_purity_after": 0.2903, "label_knn_purity_after": 0.5315},
    "sinkhorn0.5_only": {"donor_retrieval_after": 0.7361, "batch_mixing_purity_after": 0.3592, "label_knn_purity_after": 0.6527},
    "mmd20_sinkhorn0.5": {"donor_retrieval_after": 0.7778, "batch_mixing_purity_after": 0.2743, "label_knn_purity_after": 0.4846},
    "mmd10_sinkhorn0.5": {"donor_retrieval_after": 0.8056, "batch_mixing_purity_after": 0.3126, "label_knn_purity_after": 0.5696},
}


def run_one(config, seed, reference, held_out, vocab, device):
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

            contrastive = supervised_contrastive_loss(corrected, cell_type, temperature=0.1)
            variance_penalty = variance_floor_penalty(embedding, corrected, cell_type, min_ratio=0.8)
            donor_term = donor_consistency_loss(corrected, donor_code, batch_code, temperature=0.1)
            loss = contrastive + variance_penalty + donor_term
            if config["mmd_weight"] > 0:
                loss = loss + config["mmd_weight"] * vectorized_mmd_loss(corrected, batch_code)
            if config["sinkhorn_weight"] > 0:
                loss = loss + config["sinkhorn_weight"] * vectorized_sinkhorn_ot_loss(
                    corrected, batch_code, epsilon=SINKHORN_EPSILON
                )

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

    donor_before = donor_retrieval_accuracy(embedding_before, donor_ids, batch_ids)
    donor_after = donor_retrieval_accuracy(corrected_combined, donor_ids, batch_ids)

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
    print(f"reference: {reference.n_obs} cells, held-out: {held_out.n_obs} cells")

    vocab = CovariateVocab.build(reference, ["batch"], ["total_counts", "pct_counts_mt"])

    all_results = {}
    for name, config in CONFIGS.items():
        print(f"\n{'=' * 20} {name} ({config}) {'=' * 20}")
        seed_results = []
        for seed in SEEDS:
            print(f"--- seed {seed} ---")
            r = run_one(config, seed, reference, held_out, vocab, device)
            print(r)
            seed_results.append(r)
        all_results[name] = seed_results

    print("\n\n=== SEED-CHECK SUMMARY (mean +/- std across seeds 0,1,2) ===")
    header = (f"{'config':>20} | {'donor after':>16} | {'bmix after':>16} | {'knn after':>16} | {'mean elapsed_s':>14}")
    print(header)
    print("-" * len(header))
    for name, seed_results in all_results.items():
        donor = np.array([r["donor_retrieval_after"] for r in seed_results])
        bmix = np.array([r["batch_mixing_purity_after"] for r in seed_results])
        knn = np.array([r["label_knn_purity_after"] for r in seed_results])
        elapsed = np.array([r["elapsed_s"] for r in seed_results])
        print(f"{name:>20} | {donor.mean():>7.4f}+-{donor.std():<7.4f} | {bmix.mean():>7.4f}+-{bmix.std():<7.4f} | "
              f"{knn.mean():>7.4f}+-{knn.std():<7.4f} | {elapsed.mean():>14.1f}")

    print("\n=== for reference: single-seed (seed 0) sweep already published ===")
    for name, r in PUBLISHED_SEED0.items():
        print(f"{name}: {r}")


if __name__ == "__main__":
    main()
