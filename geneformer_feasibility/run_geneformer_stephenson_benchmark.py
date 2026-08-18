"""Cross-backbone validation: does scAnchor's validated technique
(mmd_weight=20, categorical_covariate_cols=["batch"]) generalize from
scGPT to Geneformer embeddings, on the IDENTICAL cells already used for
the published scGPT-based Stephenson results (README's Current results:
batch_mixing_purity 0.7306->0.8501, label_knn_purity 0.6213->0.7059)?

Reuses scAnchor's own train()/evaluate modules directly (this is scAnchor's
own repo, unlike the abandoned sciplex side-experiment) -- not the PyPI
package, the local src/ during this investigation.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR / "src"))

from scanchor.evaluate.baselines import harmony_correct
from scanchor.evaluate.leave_one_batch_out import run as run_leave_one_batch_out
from scanchor.evaluate.metrics import batch_mixing_purity, label_knn_purity
from scanchor.train import train

HERE = Path(__file__).parent
EMB_CSV = HERE / "embeddings_out_stephenson" / "stephenson.csv"
EMBED_DIM = 256
RNG_SEED = 0


def build_h5ad() -> ad.AnnData:
    df = pd.read_csv(EMB_CSV)
    embed_cols = [str(i) for i in range(EMBED_DIM)]
    embedding = df[embed_cols].to_numpy(dtype="float32")

    obs = df[["batch", "cell_type", "total_counts", "pct_counts_mt"]].copy()
    adata = ad.AnnData(X=np.zeros((len(df), 1), dtype="float32"), obs=obs)
    adata.obsm["X_geneformer"] = embedding
    return adata


def build_config(ref_path, held_out_path, out_dir, seed=RNG_SEED):
    return {
        "reference_panel": {
            "paths": [str(ref_path)],
            "embedding_key": "X_geneformer",
            "cell_type_col": "cell_type",
            "batch_col": "batch",
            "categorical_covariate_cols": ["batch"],
            "continuous_covariate_cols": ["total_counts", "pct_counts_mt"],
        },
        "model": {
            "embed_dim": EMBED_DIM,
            "cat_embed_dim": 8,
            "covariate_dim": 32,
            "hidden_dim": 128,
            "max_delta_ratio": 1.0,
            "discriminator_hidden_dim": 128,
        },
        "training": {
            "seed": seed,
            "batch_size": 512,
            "epochs": 30,
            "learning_rate": 1e-3,
            "contrastive_temperature": 0.1,
            "contrastive_weight": 1.0,
            "variance_weight": 1.0,
            "donor_weight": 1.0,
            "adversarial_weight": 0.0,
            "absorption_weight": 0.0,
            "mmd_weight": 20.0,
            "mmd_multi_scale": False,
            "conditional_mmd_weight": 0.0,
            "adversarial_lambda": 1.0,
            "grad_clip_norm": 5.0,
            "min_variance_ratio": 0.8,
            "checkpoint_out": str(out_dir / f"correction_head_geneformer_seed{seed}.pt"),
        },
        "validation": {
            "leave_one_batch_out_path": str(held_out_path),
            "replicate_dataset_path": str(ref_path),
        },
    }


def main():
    adata = build_h5ad()
    print(f"loaded {adata.n_obs} cells, embedding dim {adata.obsm['X_geneformer'].shape[1]}")

    held_out_site = adata.obs["batch"].value_counts().idxmin()
    print(f"held-out site (smallest, matching the scGPT run): {held_out_site}")

    reference = adata[adata.obs["batch"] != held_out_site].copy()
    held_out = adata[adata.obs["batch"] == held_out_site].copy()
    print(f"reference: {reference.n_obs} cells, held-out: {held_out.n_obs} cells")

    out_dir = HERE / "geneformer_stephenson_run"
    out_dir.mkdir(exist_ok=True)
    ref_path = out_dir / "reference.h5ad"
    held_out_path = out_dir / "heldout.h5ad"
    reference.write_h5ad(ref_path)
    held_out.write_h5ad(held_out_path)

    seed_results = []
    for seed in (0, 1, 2):
        config = build_config(ref_path, held_out_path, out_dir, seed=seed)
        print(f"\n=== training scAnchor on Geneformer embeddings, seed={seed} (mmd_weight=20, same config as scGPT) ===")
        t0 = time.time()
        train(config)
        print(f"training done  ({time.time() - t0:.0f}s elapsed)")

        print(f"=== scAnchor leave-one-batch-out, seed={seed} ===")
        scanchor_result = run_leave_one_batch_out(config, config["training"]["checkpoint_out"])
        scanchor_result["seed"] = seed
        print(scanchor_result)
        seed_results.append(scanchor_result)

    print("\n=== Harmony baseline (transductive) ===")
    t0 = time.time()
    combined = ad.concat([reference, held_out], join="outer", index_unique="-c")
    harmony_embedding = harmony_correct(combined, embedding_key="X_geneformer", batch_col="batch")
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

    print("\n=== SUMMARY (Geneformer backbone, seed-checked) ===")
    for r in seed_results:
        print({"dataset": "stephenson_geneformer", **r})
    print(harmony_result)
    print("\n=== for reference: published scGPT-backbone result (seed 0) ===")
    print({"batch_mixing_purity_before": 0.7306, "batch_mixing_purity_after": 0.8501,
           "label_knn_purity_before": 0.6213, "label_knn_purity_after": 0.7059})


if __name__ == "__main__":
    main()
