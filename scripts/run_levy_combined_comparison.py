"""Does combining mmd_weight and sinkhorn_weight in the SAME correction
head beat either mechanism alone on Levy?

Motivated directly by run_levy_sinkhorn_comparison.py's real result: on
Levy, MMD (mmd_weight=20) and Sinkhorn (sinkhorn_weight=0.5) have
complementary weaknesses: MMD is weak on donor retrieval/cell-type
purity but fine on batch-mixing; Sinkhorn is much stronger on
purity/donor retrieval but worse on batch-mixing. That pattern is real
evidence they might be correcting different parts of the problem, not
just sitting at different points on the same trade-off curve, which is
worth testing directly rather than assuming.

correction_loss already supports both weights simultaneously (no code
change needed), but same as run_levy_sinkhorn_comparison.py, this
composes the loss by hand rather than calling that shared wrapper, to
skip class_conditional_mmd_loss (always 0-weighted here, expensive at
Levy's 8-batch x 14-cell-type scale); this is 100% behavior-preserving,
purely a wall-clock fix.

Single seed (seed=0) across a small grid first, since each combined run
pays Sinkhorn's full per-pair 50-iteration cost on top of MMD's, so this
is at least as expensive per run as the sinkhorn-only comparison (~50
min/seed on GPU there). Seed-check only the best point after this first
pass, same pattern as the original sinkhorn_weight sweep (single-seed
sweep, then seed-check the winner).

Reuses the already-cached real scGPT embeddings
(levy_run/{reference,heldout}.h5ad).
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
EVAL_SUBSAMPLE_N = 20_000
EVAL_SUBSAMPLE_SEED = 0  # same fixed eval cells as run_levy_sinkhorn_comparison.py, for direct comparability

# First-pass grid, single seed. Includes each individual mechanism's own
# best point (mmd20 alone, sinkhorn0.5 alone) as an in-run sanity check
# against run_levy_sinkhorn_comparison.py's already-published numbers,
# plus combined points around both.
CONFIGS = {
    "mmd20_only": {"mmd_weight": 20.0, "sinkhorn_weight": 0.0},
    "sinkhorn0.5_only": {"mmd_weight": 0.0, "sinkhorn_weight": 0.5},
    "mmd20_sinkhorn0.5": {"mmd_weight": 20.0, "sinkhorn_weight": 0.5},
    "mmd10_sinkhorn0.5": {"mmd_weight": 10.0, "sinkhorn_weight": 0.5},
    "mmd20_sinkhorn0.25": {"mmd_weight": 20.0, "sinkhorn_weight": 0.25},
    "mmd10_sinkhorn0.25": {"mmd_weight": 10.0, "sinkhorn_weight": 0.25},
}
SEED = 0
SINKHORN_EPSILON = 0.1


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
                loss = loss + config["mmd_weight"] * mmd_loss(corrected, batch_code)
            if config["sinkhorn_weight"] > 0:
                loss = loss + config["sinkhorn_weight"] * sinkhorn_ot_loss(
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
    print(f"reference: {reference.n_obs} cells, held-out: {held_out.n_obs} cells "
          f"(eval subsampled to {EVAL_SUBSAMPLE_N} of {reference.n_obs + held_out.n_obs} combined cells)")

    vocab = CovariateVocab.build(reference, ["batch"], ["total_counts", "pct_counts_mt"])

    all_results = {}
    for name, config in CONFIGS.items():
        print(f"\n{'=' * 20} {name} ({config}) {'=' * 20}")
        r = run_one(config, SEED, reference, held_out, vocab, device)
        print(r)
        all_results[name] = r

    print("\n\n=== SUMMARY (seed 0) ===")
    header = (f"{'config':>20} | {'donor after':>11} | {'bmix after':>10} | {'knn after':>9} | "
              f"{'bmix delta':>10} | {'knn delta':>9} | {'elapsed_s':>9}")
    print(header)
    print("-" * len(header))
    for name, r in all_results.items():
        bmix_delta = r["batch_mixing_purity_after"] - r["batch_mixing_purity_before"]
        knn_delta = r["label_knn_purity_after"] - r["label_knn_purity_before"]
        print(f"{name:>20} | {r['donor_retrieval_after']:>11.4f} | {r['batch_mixing_purity_after']:>10.4f} | "
              f"{r['label_knn_purity_after']:>9.4f} | {bmix_delta:>+10.4f} | {knn_delta:>+9.4f} | "
              f"{r['elapsed_s']:>9.1f}")


if __name__ == "__main__":
    main()
