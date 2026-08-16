"""Run scAnchor + a Harmony baseline on Stephenson et al. 2021's COVID-19 PBMC
atlas (Nat Med, doi 10.1038/s41591-021-01329-2) -- a real public dataset with
a genuinely independent Site (batch) x Status (disease/condition) x donor_id
structure, verified directly from the file (not the paper's description):
  - Site: Cambridge/Ncl/Newcastle/Sanger, 3 processing sites
  - Status: Covid/Healthy/LPS/Non_covid, 4 categories
  - donor_id: 120 donors -- but every donor appears at exactly ONE site, so
    donor and batch are fully confounded here. This dataset is NOT suitable
    for scAnchor's donor-consistency mechanism (no learnable cross-batch
    donor signal exists) -- donor_col is deliberately omitted, same
    treatment as the scIB benchmark tasks. It IS suitable for the
    batch-mixing / cell-type-purity comparison this script runs.

Full dataset is 647,366 cells / ~7GB -- far too large for CPU scGPT
embedding extraction (~91hr at this project's observed throughput). This
script subsamples by donor (capped per-donor) to a tractable scale, shared
with scripts/run_scdisinfact_stephenson.py so both methods are compared
on the identical cells.

Example:
    python scripts/run_stephenson_benchmark.py \
        --checkpoint-dir /path/to/scgpt_continual_pretrained_checkpoint \
        --out-dir /path/to/scratch/stephenson_run \
        --data-cache-dir /path/to/scratch/stephenson_data_cache
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from scanchor.embeddings.scgpt_extract import extract_embeddings
from scanchor.evaluate.baselines import harmony_correct
from scanchor.evaluate.leave_one_batch_out import run as run_leave_one_batch_out
from scanchor.evaluate.metrics import batch_mixing_purity, label_knn_purity
from scanchor.train import train

SOURCE_URL = "https://datasets.cellxgene.cziscience.com/fe2e847c-1602-4f1b-86a4-112e4dc7a8e3.h5ad"
SOURCE_FILENAME = "stephenson_covid_pbmc_full.h5ad"
RNG_SEED = 0
PER_DONOR_CAP = 175  # -> ~15-25k cells total across 120 donors, comparable to the scIB tasks


def download_if_needed(dest: Path) -> None:
    if dest.exists():
        print(f"already downloaded: {dest}")
        return
    print(f"downloading {SOURCE_URL} -> {dest} (7GB, will take a while) ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SOURCE_URL, dest)


def build_subsample(full_path: Path, subsample_path: Path) -> ad.AnnData:
    """Shared subsampling logic -- reused verbatim (same seed) by
    run_scdisinfact_stephenson.py so both methods see identical cells.
    Cached after first build so re-runs (e.g. the scDisInFact script running
    as a separate array task) don't redo this.
    """
    if subsample_path.exists():
        print(f"already subsampled: {subsample_path}")
        return ad.read_h5ad(subsample_path)

    print(f"loading full dataset from {full_path} (backed, for memory) ...")
    full = ad.read_h5ad(full_path, backed="r")
    print(f"full dataset: {full.n_obs} cells, {full.n_vars} genes")

    rng = np.random.default_rng(RNG_SEED)
    groups = full.obs.groupby("donor_id", observed=True).indices
    keep_idx = []
    for _, idx in groups.items():
        idx = np.array(idx)
        n = min(PER_DONOR_CAP, len(idx))
        keep_idx.extend(rng.choice(idx, size=n, replace=False))
    keep_idx = np.array(sorted(keep_idx))

    print(f"subsampling to {len(keep_idx)} cells across {full.obs['donor_id'].nunique()} donors ...")
    sub = full[keep_idx].to_memory()

    # AnnData doesn't subset .uns (it's unstructured, arbitrary content) --
    # this file's uns/antibody_X, uns/antibody_raw.X, and
    # uns/neighbors/rp_forest are all still full-647k-cell-sized structures
    # (confirmed via direct h5py inspection: e.g. antibody_X's indptr has
    # shape (647367,), not (n_obs+1,)) and unused by this pipeline. Clearing
    # before writing keeps the cache at ~subsample scale instead of ~7GB of
    # irrelevant dead weight riding along on every write/read.
    sub.uns.clear()

    # raw counts live in .raw.X per this file's CELLxGENE schema (X itself is
    # normalized/processed) -- confirmed from this dataset's own metadata
    # (raw_data_location: raw.X), not assumed.
    sub.X = sub.raw[:, sub.var_names].X.copy()
    # var_names are Ensembl IDs (e.g. ENSG00000243485) per this file's
    # CELLxGENE schema, NOT gene symbols -- confirmed directly (a first
    # real run against scGPT's vocab matched 0/24245 genes using var_names
    # as-is). Real gene symbols live in the separate feature_name column.
    sub.var["gene_name"] = sub.var["feature_name"].astype(str)
    sub.var_names = sub.var["gene_name"]
    sub.var_names_make_unique()
    sub.var["gene_name"] = sub.var_names
    sub.obs["batch"] = sub.obs["Site"].astype(str)
    sub.obs["cell_type"] = sub.obs["cell_type"].astype(str)
    # total_counts / pct_counts_mt already exist as real columns in this
    # file (unlike Jerber/Levy, no need to compute them from counts).
    sub.obs["total_counts"] = sub.obs["total_counts"].astype(float)
    sub.obs["pct_counts_mt"] = sub.obs["pct_counts_mt"].astype(float)

    print(f"subsample: {sub.n_obs} cells, {sub.obs['batch'].nunique()} sites, "
          f"{sub.obs['donor_id'].nunique()} donors, {sub.obs['cell_type'].nunique()} cell types")
    print("site value counts:", sub.obs["batch"].value_counts().to_dict())

    subsample_path.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(subsample_path)
    return sub


def build_config(ref_path: Path, held_out_path: Path, out_dir: Path, embed_dim: int) -> dict:
    return {
        "reference_panel": {
            "paths": [str(ref_path)],
            "embedding_key": "X_scGPT",
            "cell_type_col": "cell_type",
            "batch_col": "batch",
            # "Status" (Covid/Healthy/LPS/Non_covid) added alongside batch
            # after a real, seed-checked (0/1/2) ablation found it partially
            # closes the batch-mixing regression documented in README's
            # Current results (v0.9.1) -- consistent direction at every
            # seed, no cost to cell-type purity. Doesn't fully close the
            # gap, but a genuine, free win via the same mechanism already
            # used for batch.
            "categorical_covariate_cols": ["batch", "Status"],
            "continuous_covariate_cols": ["total_counts", "pct_counts_mt"],
        },
        "model": {
            "embed_dim": embed_dim,
            "cat_embed_dim": 8,
            "covariate_dim": 32,
            "hidden_dim": 128,
            "max_delta_ratio": 1.0,
            "discriminator_hidden_dim": 128,
        },
        "training": {
            "seed": RNG_SEED,
            "batch_size": 512,
            "epochs": 30,
            "learning_rate": 1e-3,
            "contrastive_temperature": 0.1,
            "contrastive_weight": 1.0,
            "variance_weight": 1.0,
            "donor_weight": 1.0,
            # Current shipped default (README's Current results): MMD alone.
            "adversarial_weight": 0.0,
            "absorption_weight": 0.0,
            "mmd_weight": 20.0,
            "mmd_multi_scale": False,
            "conditional_mmd_weight": 0.0,
            "adversarial_lambda": 1.0,
            "grad_clip_norm": 5.0,
            "min_variance_ratio": 0.8,
            "checkpoint_out": str(out_dir / "correction_head.pt"),
        },
        "validation": {
            "leave_one_batch_out_path": str(held_out_path),
            "replicate_dataset_path": str(ref_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--data-cache-dir", required=True, type=Path)
    parser.add_argument("--scgpt-batch-size", type=int, default=64)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    full_path = args.data_cache_dir / SOURCE_FILENAME
    subsample_path = args.data_cache_dir / "stephenson_subsample.h5ad"
    download_if_needed(full_path)
    sub = build_subsample(full_path, subsample_path)

    held_out_site = sub.obs["batch"].value_counts().idxmin()
    print(f"held-out site (smallest): {held_out_site}")

    raw_path = args.out_dir / "input.h5ad"
    sub.write_h5ad(raw_path)

    t0 = time.time()
    print(f"running scGPT embed_data on {sub.n_obs} cells...")
    embedded = extract_embeddings(
        adata_path=raw_path,
        model_dir=args.checkpoint_dir,
        out_path=args.out_dir / "embedded.h5ad",
        gene_col="gene_name",
        batch_size=args.scgpt_batch_size,
        obs_to_save=["batch", "cell_type", "total_counts", "pct_counts_mt", "Status"],
        device=device,
        use_fast_transformer=(device == "cuda"),
    )
    print(f"embedding done: {embedded.obsm['X_scGPT'].shape}  ({time.time() - t0:.0f}s elapsed)")

    held_out = embedded[embedded.obs["batch"] == held_out_site].copy()
    reference = embedded[embedded.obs["batch"] != held_out_site].copy()
    print(f"reference: {reference.n_obs} cells, held-out: {held_out.n_obs} cells")

    ref_path = args.out_dir / "reference.h5ad"
    held_out_path = args.out_dir / "heldout.h5ad"
    reference.write_h5ad(ref_path)
    held_out.write_h5ad(held_out_path)

    config = build_config(ref_path, held_out_path, args.out_dir, embedded.obsm["X_scGPT"].shape[1])

    print("\n=== training scAnchor (mmd_weight=20, current default) ===")
    t0 = time.time()
    train(config)
    print(f"training done  ({time.time() - t0:.0f}s elapsed)")

    print("\n=== scAnchor leave-one-batch-out ===")
    scanchor_result = run_leave_one_batch_out(config, config["training"]["checkpoint_out"])
    print(scanchor_result)

    print("\n=== Harmony baseline (transductive, full access to held-out site) ===")
    t0 = time.time()
    combined = ad.concat([reference, held_out], join="outer", index_unique="-c")
    harmony_embedding = harmony_correct(combined, embedding_key="X_scGPT", batch_col="batch")
    harmony_result = {
        "batch_mixing_purity_harmony": batch_mixing_purity(
            harmony_embedding, combined.obs["batch"].astype(str).to_numpy()
        ),
        "label_knn_purity_harmony": label_knn_purity(
            harmony_embedding, combined.obs["cell_type"].astype(str).to_numpy()
        ),
    }
    print(f"harmony done  ({time.time() - t0:.0f}s elapsed)")
    print(harmony_result)

    print("\n=== SUMMARY ===")
    print({"dataset": "stephenson", **scanchor_result, **harmony_result})


if __name__ == "__main__":
    main()
