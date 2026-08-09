"""Run the full scAnchor pipeline on the Levy schizophrenia iPSC astrocyte
mini-village dataset: real scGPT embedding extraction, training, and both
evaluation protocols.

Cluster-ready: takes data/checkpoint paths as CLI args instead of hardcoding
them, uses the installed scanchor package (pip install -e ".[scgpt]"), and
device is auto-detected -- GPU if visible, CPU otherwise. No behavior
difference between the two beyond speed.

Example (full dataset, no subsampling):
    python scripts/run_full_dataset.py \
        --metadata-txt /path/to/Levy_astrocyte_mini_village_cell_metadata.txt \
        --counts-h5ad /path/to/Levy_astrocyte_mini_village.h5ad \
        --checkpoint-dir /path/to/scgpt_continual_pretrained_checkpoint \
        --out-dir /path/to/scratch/scanchor_full_run \
        --per-group-n 0

--per-group-n 0 means "use every cell in every (batch, donor) group" --
i.e. the full dataset, not a subsample. Set it to a positive integer to cap
cells per group (what every experiment so far in this repo's history used).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

from scanchor.embeddings.scgpt_extract import extract_embeddings
from scanchor.evaluate.leave_one_batch_out import run as run_leave_one_batch_out
from scanchor.evaluate.replicate_test import run as run_replicate_test
from scanchor.train import train

RNG_SEED = 0


def build_config(ref_path: Path, held_out_path: Path, out_dir: Path, embed_dim: int) -> dict:
    return {
        "reference_panel": {
            "paths": [str(ref_path)],
            "embedding_key": "X_scGPT",
            "cell_type_col": "cell_type",
            "batch_col": "batch",
            "donor_col": "donor_id",
            "categorical_covariate_cols": ["batch"],
            "continuous_covariate_cols": ["total_counts", "pct_counts_mt"],
        },
        "model": {
            "embed_dim": embed_dim,
            "cat_embed_dim": 8,
            "covariate_dim": 32,
            "hidden_dim": 128,
            "max_delta_ratio": 1.0,
        },
        "training": {
            "batch_size": 512,
            "epochs": 30,
            "learning_rate": 1e-3,
            "contrastive_temperature": 0.1,
            "contrastive_weight": 1.0,
            "variance_weight": 1.0,
            "donor_weight": 1.0,
            "adversarial_weight": 1.0,
            "adversarial_lambda": 1.0,
            "grad_clip_norm": 5.0,
            "min_variance_ratio": 0.8,
            "checkpoint_out": str(out_dir / "correction_head.pt"),
        },
        "validation": {
            "leave_one_batch_out_path": str(held_out_path),
            "replicate_dataset_path": str(ref_path),
            "donor_col": "donor_id",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-txt", required=True, type=Path)
    parser.add_argument("--counts-h5ad", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--held-out-batch", default="D60_astrocytes_B")
    parser.add_argument("--per-group-n", type=int, default=0, help="0 = use every cell (full dataset)")
    parser.add_argument("--scgpt-batch-size", type=int, default=64)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    t0 = time.time()
    print("loading metadata...")
    meta = pd.read_csv(args.metadata_txt, index_col=0)
    meta = meta[meta["doublet"] != True]  # noqa: E712

    print("loading counts...")
    adata = ad.read_h5ad(args.counts_h5ad)
    common = adata.obs_names.intersection(meta.index)
    adata = adata[common].copy()
    adata.obs = meta.loc[common]

    if args.per_group_n > 0:
        rng = np.random.default_rng(RNG_SEED)
        groups = adata.obs.groupby(["PREFIX", "DONOR"]).indices
        keep_idx = []
        for _, idx in groups.items():
            idx = np.array(idx)
            n = min(args.per_group_n, len(idx))
            keep_idx.extend(rng.choice(idx, size=n, replace=False))
        adata = adata[sorted(keep_idx)].copy()

    print(f"dataset: {adata.n_obs} cells, {adata.obs['PREFIX'].nunique()} batches, "
          f"{adata.obs['DONOR'].nunique()} donors  ({time.time() - t0:.0f}s elapsed)")

    adata.var_names_make_unique()
    adata.var["gene_name"] = adata.var_names
    adata.obs["batch"] = adata.obs["PREFIX"].astype(str)
    adata.obs["donor_id"] = adata.obs["DONOR"].astype(str)
    adata.obs["cell_type"] = adata.obs["leiden_0.4"].astype(str)
    adata.obs["total_counts"] = adata.obs["n_counts"].astype(float)
    adata.obs["pct_counts_mt"] = adata.obs["percent_mito"].astype(float)

    raw_path = args.out_dir / "input.h5ad"
    adata.write_h5ad(raw_path)

    t0 = time.time()
    print(f"running scGPT embed_data on {adata.n_obs} cells...")
    embedded = extract_embeddings(
        adata_path=raw_path,
        model_dir=args.checkpoint_dir,
        out_path=args.out_dir / "embedded.h5ad",
        gene_col="gene_name",
        batch_size=args.scgpt_batch_size,
        obs_to_save=["batch", "donor_id", "cell_type", "total_counts", "pct_counts_mt"],
        device=device,
        use_fast_transformer=(device == "cuda"),  # flash-attn typically only installed w/ CUDA
    )
    print(f"embedding done: {embedded.obsm['X_scGPT'].shape}  ({time.time() - t0:.0f}s elapsed)")

    held_out = embedded[embedded.obs["batch"] == args.held_out_batch].copy()
    reference = embedded[embedded.obs["batch"] != args.held_out_batch].copy()
    print(f"reference: {reference.n_obs} cells, held-out: {held_out.n_obs} cells")

    ref_path = args.out_dir / "reference.h5ad"
    held_out_path = args.out_dir / "heldout.h5ad"
    reference.write_h5ad(ref_path)
    held_out.write_h5ad(held_out_path)

    config = build_config(ref_path, held_out_path, args.out_dir, embedded.obsm["X_scGPT"].shape[1])

    print("\n=== training ===")
    t0 = time.time()
    train(config)
    print(f"training done  ({time.time() - t0:.0f}s elapsed)")

    print("\n=== replicate test ===")
    print(run_replicate_test(config, config["training"]["checkpoint_out"]))

    print("\n=== leave-one-batch-out test ===")
    print(run_leave_one_batch_out(config, config["training"]["checkpoint_out"]))


if __name__ == "__main__":
    main()
