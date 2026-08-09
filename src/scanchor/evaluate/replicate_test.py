"""Same-donor-across-batch test: apply a trained head, check donor retrieval accuracy."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import torch

from scanchor.config import load_config
from scanchor.data.covariates import CovariateVocab
from scanchor.evaluate.metrics import donor_retrieval_accuracy
from scanchor.model.correction_head import CorrectionHead


def run(config: dict, checkpoint_path: str | Path) -> dict:
    val_cfg = config["validation"]
    ref_cfg = config["reference_panel"]
    model_cfg = config["model"]

    adata = ad.read_h5ad(val_cfg["replicate_dataset_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = CovariateVocab.from_dict(checkpoint["vocab"])

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

    embedding = torch.from_numpy(adata.obsm[ref_cfg["embedding_key"]].astype("float32"))
    categorical = torch.from_numpy(vocab.encode_categorical(adata))
    continuous = torch.from_numpy(vocab.encode_continuous(adata))

    with torch.no_grad():
        corrected = head(embedding, categorical, continuous).numpy()

    donor_ids = adata.obs[val_cfg["donor_col"]].astype(str).to_numpy()
    batch_ids = adata.obs[ref_cfg["batch_col"]].astype(str).to_numpy()

    before = donor_retrieval_accuracy(embedding.numpy(), donor_ids, batch_ids)
    after = donor_retrieval_accuracy(corrected, donor_ids, batch_ids)
    return {"donor_retrieval_accuracy_before": before, "donor_retrieval_accuracy_after": after}


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
