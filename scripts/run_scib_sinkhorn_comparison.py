"""Completes Sinkhorn's validation against MMD's original bar: the scIB
atlas-level benchmarks (immune, pancreas, lung) were the one remaining
axis flagged as untested in README's Net assessment for sinkhorn_weight
(cross-backbone: checked, positive; replicate-structure/Levy: checked,
mixed; scIB: this script).

These datasets have real batch counts (immune=10, pancreas=9, lung=16,
all higher than Stephenson's 3, lung higher even than Levy's 8); this
uses scripts/_vectorized_batch_losses.py from the start rather than
losses.py's sequential per-pair loop, given how much that mattered at
Levy's smaller 8-batch scale.

No donor_id in any of these (scIB atlas tasks don't have donor
identity, same as run_scib_benchmark.py's original setup), so only
batch_mixing_purity / label_knn_purity via leave-one-batch-out apply,
not donor_retrieval_accuracy.

Reuses the already-cached real scGPT embeddings
(scib_benchmark_run/{dataset}/{reference,heldout}.h5ad) from the
original MMD scIB validation; no new embedding extraction needed.
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
from scanchor.evaluate.metrics import batch_mixing_purity, label_knn_purity
from scanchor.model.correction_head import CorrectionHead
from scanchor.model.losses import donor_consistency_loss, supervised_contrastive_loss, variance_floor_penalty
from _vectorized_batch_losses import vectorized_mmd_loss, vectorized_sinkhorn_ot_loss

SCIB_RUN = HERE / "scib_benchmark_run"
DATASETS = ["immune", "pancreas", "lung"]
EMBED_DIM = 512
EPOCHS = 30
SEEDS = (0, 1, 2)
SINKHORN_EPSILON = 0.1

MECHANISMS = {
    "mmd20": {"mmd_weight": 20.0, "sinkhorn_weight": 0.0},
    "sinkhorn0.5": {"mmd_weight": 0.0, "sinkhorn_weight": 0.5},
}


def run_one(config, seed, reference, held_out, vocab, device):
    torch.manual_seed(seed)

    dataset = EmbeddedCellDataset(
        reference, vocab, embedding_key="X_scGPT", cell_type_col="cell_type", batch_col="batch",
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
            corrected = head(embedding, categorical, continuous)

            contrastive = supervised_contrastive_loss(corrected, cell_type, temperature=0.1)
            variance_penalty = variance_floor_penalty(embedding, corrected, cell_type, min_ratio=0.8)
            loss = contrastive + variance_penalty
            # no donor_id in scIB tasks, so donor_consistency_loss stays inert
            # (matches run_scib_benchmark.py's original setup), so it's just
            # skipped here rather than computed-and-discarded.
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
    embedding_before = combined.obsm["X_scGPT"]

    return {
        "seed": seed,
        "elapsed_s": elapsed,
        "batch_mixing_purity_before": batch_mixing_purity(embedding_before, batch_ids),
        "batch_mixing_purity_after": batch_mixing_purity(corrected_combined, batch_ids),
        "label_knn_purity_before": label_knn_purity(embedding_before, cell_types),
        "label_knn_purity_after": label_knn_purity(corrected_combined, cell_types),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    all_results = {}
    for dataset_name in DATASETS:
        run_dir = SCIB_RUN / dataset_name
        reference = ad.read_h5ad(run_dir / "reference.h5ad")
        held_out = ad.read_h5ad(run_dir / "heldout.h5ad")
        print(f"\n### {dataset_name}: {reference.n_obs} reference cells, {held_out.n_obs} held-out cells, "
              f"{reference.obs['batch'].nunique()} training batches ###")
        vocab = CovariateVocab.build(reference, ["batch"], ["total_counts", "pct_counts_mt"])

        for mech_name, config in MECHANISMS.items():
            print(f"\n{'=' * 10} {dataset_name} / {mech_name} {'=' * 10}")
            seed_results = []
            for seed in SEEDS:
                r = run_one(config, seed, reference, held_out, vocab, device)
                print(r)
                seed_results.append(r)
            all_results[(dataset_name, mech_name)] = seed_results

    print("\n\n=== SUMMARY (mean +/- std across seeds 0,1,2) ===")
    header = f"{'dataset':>10} | {'mechanism':>11} | {'bmix after':>16} | {'knn after':>16} | {'mean elapsed_s':>14}"
    print(header)
    print("-" * len(header))
    for (dataset_name, mech_name), seed_results in all_results.items():
        bmix = np.array([r["batch_mixing_purity_after"] for r in seed_results])
        knn = np.array([r["label_knn_purity_after"] for r in seed_results])
        elapsed = np.array([r["elapsed_s"] for r in seed_results])
        print(f"{dataset_name:>10} | {mech_name:>11} | {bmix.mean():>7.4f}+-{bmix.std():<7.4f} | "
              f"{knn.mean():>7.4f}+-{knn.std():<7.4f} | {elapsed.mean():>14.1f}")


if __name__ == "__main__":
    main()
