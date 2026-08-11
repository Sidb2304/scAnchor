"""Run scAnchor + a Harmony baseline on one scIB atlas-level integration
benchmark dataset (immune, pancreas, or lung) -- the standard reference
point every batch-correction method gets compared against, not yet run in
this project (see README's Reference panel section).

Cluster-ready: takes paths as CLI args, downloads the dataset from Figshare
on first run (cached after), device is auto-detected (GPU if visible, CPU
otherwise). No behavior difference between the two beyond speed.

These datasets don't carry a donor/individual identity at all (unlike
Levy/Jerber) -- donor_col is left out of the config entirely, so
donor_consistency_loss correctly stays inert, and only the leave-one-batch-out
protocol (batch-mixing purity, cell-type kNN purity) applies, not the
same-donor-across-batch replicate test.

Example:
    python scripts/run_scib_benchmark.py \
        --dataset pancreas \
        --checkpoint-dir /path/to/scgpt_continual_pretrained_checkpoint \
        --out-dir /path/to/scratch/scib_pancreas_run \
        --data-cache-dir /path/to/scratch/scib_data_cache
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

RNG_SEED = 0

# URLs and obs-column names resolved from the Figshare API directly
# (api.figshare.com/v2/articles/12420968) -- batch/celltype column names
# come from the scIB paper's own preprocessing convention for each dataset,
# not verified by actually loading these files locally. The runtime check
# in main() fails loudly with the real available columns if any of these
# are wrong, rather than silently using the wrong column.
DATASETS = {
    "pancreas": {
        "url": "https://ndownloader.figshare.com/files/24539828",
        "filename": "human_pancreas_norm_complexBatch.h5ad",
        "batch_col": "tech",
        "celltype_col": "celltype",
    },
    "lung": {
        "url": "https://ndownloader.figshare.com/files/24539942",
        "filename": "Lung_atlas_public.h5ad",
        "batch_col": "batch",
        "celltype_col": "cell_type",
    },
    "immune": {
        "url": "https://ndownloader.figshare.com/files/25717328",
        "filename": "Immune_ALL_human.h5ad",
        "batch_col": "batch",
        "celltype_col": "final_annotation",
    },
}


def download_if_needed(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"already downloaded: {dest}")
        return
    print(f"downloading {url} -> {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def build_config(ref_path: Path, held_out_path: Path, out_dir: Path, embed_dim: int) -> dict:
    return {
        "reference_panel": {
            "paths": [str(ref_path)],
            "embedding_key": "X_scGPT",
            "cell_type_col": "cell_type",
            "batch_col": "batch",
            "categorical_covariate_cols": ["batch"],
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
            # Current shipped default (README's Current results): MMD alone,
            # not the adversarial discriminator -- a real, seed-checked
            # sweep found the discriminator regresses batch-mixing.
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
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--data-cache-dir", required=True, type=Path)
    parser.add_argument("--scgpt-batch-size", type=int, default=64)
    args = parser.parse_args()

    spec = DATASETS[args.dataset]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.dataset}] device: {device}")

    raw_download_path = args.data_cache_dir / spec["filename"]
    download_if_needed(spec["url"], raw_download_path)

    t0 = time.time()
    print(f"[{args.dataset}] loading {raw_download_path} ...")
    adata = ad.read_h5ad(raw_download_path)
    print(f"[{args.dataset}] {adata.n_obs} cells, {adata.n_vars} genes  "
          f"({time.time() - t0:.0f}s elapsed)")
    print(f"[{args.dataset}] obs columns available: {list(adata.obs.columns)}")

    for col_key in ("batch_col", "celltype_col"):
        col = spec[col_key]
        if col not in adata.obs.columns:
            raise ValueError(
                f"[{args.dataset}] expected {col_key}={col!r} not found in obs. "
                f"Available columns: {list(adata.obs.columns)} -- fix DATASETS[{args.dataset!r}] "
                f"in this script and re-run."
            )

    adata.var_names_make_unique()
    adata.var["gene_name"] = adata.var_names
    adata.obs["batch"] = adata.obs[spec["batch_col"]].astype(str)
    adata.obs["cell_type"] = adata.obs[spec["celltype_col"]].astype(str)

    # scIB benchmark files aren't uniformly raw counts (some ship pre-
    # normalized) and there's no donor/individual identity at all here --
    # total_counts/pct_counts_mt are computed fresh from whatever's in .X
    # for the continuous covariates scAnchor expects; donor_col is omitted
    # from build_config entirely above, so donor_consistency_loss correctly
    # stays inert rather than crashing on a column that doesn't exist.
    total_counts = np.asarray(adata.X.sum(axis=1)).flatten()
    mt_mask = adata.var_names.str.upper().str.startswith("MT-")
    mt_counts = np.asarray(adata.X[:, mt_mask].sum(axis=1)).flatten() if mt_mask.any() else np.zeros(adata.n_obs)
    adata.obs["total_counts"] = total_counts
    adata.obs["pct_counts_mt"] = np.divide(
        mt_counts, total_counts, out=np.zeros_like(total_counts), where=total_counts > 0
    ) * 100

    held_out_batch = adata.obs["batch"].value_counts().idxmin()
    print(f"[{args.dataset}] held-out batch (smallest): {held_out_batch}")

    raw_path = args.out_dir / "input.h5ad"
    adata.write_h5ad(raw_path)

    t0 = time.time()
    print(f"[{args.dataset}] running scGPT embed_data on {adata.n_obs} cells...")
    embedded = extract_embeddings(
        adata_path=raw_path,
        model_dir=args.checkpoint_dir,
        out_path=args.out_dir / "embedded.h5ad",
        gene_col="gene_name",
        batch_size=args.scgpt_batch_size,
        obs_to_save=["batch", "cell_type", "total_counts", "pct_counts_mt"],
        device=device,
        use_fast_transformer=(device == "cuda"),  # flash-attn typically only installed w/ CUDA
    )
    print(f"[{args.dataset}] embedding done: {embedded.obsm['X_scGPT'].shape}  "
          f"({time.time() - t0:.0f}s elapsed)")

    held_out = embedded[embedded.obs["batch"] == held_out_batch].copy()
    reference = embedded[embedded.obs["batch"] != held_out_batch].copy()
    print(f"[{args.dataset}] reference: {reference.n_obs} cells, held-out: {held_out.n_obs} cells")

    ref_path = args.out_dir / "reference.h5ad"
    held_out_path = args.out_dir / "heldout.h5ad"
    reference.write_h5ad(ref_path)
    held_out.write_h5ad(held_out_path)

    config = build_config(ref_path, held_out_path, args.out_dir, embedded.obsm["X_scGPT"].shape[1])

    print(f"\n[{args.dataset}] === training scAnchor (mmd_weight=20, current default) ===")
    t0 = time.time()
    train(config)
    print(f"[{args.dataset}] training done  ({time.time() - t0:.0f}s elapsed)")

    print(f"\n[{args.dataset}] === scAnchor leave-one-batch-out ===")
    scanchor_result = run_leave_one_batch_out(config, config["training"]["checkpoint_out"])
    print(scanchor_result)

    print(f"\n[{args.dataset}] === Harmony baseline (transductive, full access to held-out batch) ===")
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
    print(f"[{args.dataset}] harmony done  ({time.time() - t0:.0f}s elapsed)")
    print(harmony_result)

    print(f"\n[{args.dataset}] === SUMMARY ===")
    print({**scanchor_result, **harmony_result})


if __name__ == "__main__":
    main()
