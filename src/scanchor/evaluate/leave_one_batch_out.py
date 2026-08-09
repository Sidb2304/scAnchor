"""Leave-one-batch-out generalization test.

Applies a head trained WITHOUT the held-out batch to that batch's embeddings,
using only its covariates (never its cells during training), then checks
whether batch signal is still removed relative to the reference panel the
head *was* trained on. This is the literal inductive-generalization claim —
the part transductive tools (Harmony, scVI, scDisInFact) aren't built for,
since they require the held-out batch present at correction time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import torch

from scanchor.config import load_config
from scanchor.data.covariates import CovariateVocab
from scanchor.evaluate.metrics import batch_mixing_purity, label_knn_purity
from scanchor.model.correction_head import CorrectionHead


def run(config: dict, checkpoint_path: str | Path) -> dict:
    val_cfg = config["validation"]
    ref_cfg = config["reference_panel"]
    model_cfg = config["model"]

    reference = ad.concat(
        [ad.read_h5ad(p) for p in ref_cfg["paths"]], join="outer", index_unique="-r"
    )
    held_out = ad.read_h5ad(val_cfg["leave_one_batch_out_path"])
    combined = ad.concat([reference, held_out], join="outer", index_unique="-c")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = CovariateVocab.from_dict(checkpoint["vocab"])
    # Sanity check: the held-out batch's ID must not be in the training vocab,
    # otherwise this isn't testing inductive generalization at all.
    held_out_batches = set(held_out.obs[ref_cfg["batch_col"]].astype(str))
    trained_batches = set(vocab.vocabs.get(ref_cfg["batch_col"], {}).keys())
    leaked = held_out_batches & trained_batches
    if leaked:
        raise ValueError(f"Held-out batch(es) {leaked} were seen during training — not a valid test.")

    head = CorrectionHead(
        embed_dim=model_cfg["embed_dim"],
        vocab_sizes=vocab.vocab_sizes(),
        n_continuous=len(ref_cfg["continuous_covariate_cols"]),
        cat_embed_dim=model_cfg["cat_embed_dim"],
        covariate_dim=model_cfg["covariate_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        max_delta_ratio=model_cfg.get("max_delta_ratio", 1.0),
    )
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()

    embedding = torch.from_numpy(combined.obsm[ref_cfg["embedding_key"]].astype("float32"))
    categorical = torch.from_numpy(vocab.encode_categorical(combined))
    continuous = torch.from_numpy(vocab.encode_continuous(combined))

    with torch.no_grad():
        corrected = head(embedding, categorical, continuous).numpy()

    batch_ids = combined.obs[ref_cfg["batch_col"]].astype(str).to_numpy()
    labels = combined.obs[ref_cfg["cell_type_col"]].astype(str).to_numpy()

    return {
        "batch_mixing_purity_before": batch_mixing_purity(embedding.numpy(), batch_ids),
        "batch_mixing_purity_after": batch_mixing_purity(corrected, batch_ids),
        "label_knn_purity_before": label_knn_purity(embedding.numpy(), labels),
        "label_knn_purity_after": label_knn_purity(corrected, labels),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", default=None, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_path = args.checkpoint or config["training"]["checkpoint_out"]
    print(run(config, checkpoint_path))


if __name__ == "__main__":
    _cli()
