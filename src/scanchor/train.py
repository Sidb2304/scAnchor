"""Train the inductive correction head on a reference panel of embedded batches."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scanchor.config import load_config
from scanchor.data.datasets import load_reference_panel
from scanchor.model.correction_head import CorrectionHead
from scanchor.model.losses import correction_loss


def train(config: dict) -> CorrectionHead:
    ref_cfg = config["reference_panel"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    dataset, vocab = load_reference_panel(
        paths=ref_cfg["paths"],
        categorical_cols=ref_cfg["categorical_covariate_cols"],
        continuous_cols=ref_cfg["continuous_covariate_cols"],
        embedding_key=ref_cfg["embedding_key"],
        cell_type_col=ref_cfg["cell_type_col"],
        batch_col=ref_cfg["batch_col"],
        donor_col=ref_cfg.get("donor_col"),
    )
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = CorrectionHead(
        embed_dim=model_cfg["embed_dim"],
        vocab_sizes=vocab.vocab_sizes(),
        n_continuous=len(ref_cfg["continuous_covariate_cols"]),
        cat_embed_dim=model_cfg["cat_embed_dim"],
        covariate_dim=model_cfg["covariate_dim"],
        hidden_dim=model_cfg["hidden_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=train_cfg["learning_rate"])

    for epoch in range(train_cfg["epochs"]):
        epoch_metrics = {"contrastive": 0.0, "variance_penalty": 0.0, "donor_consistency": 0.0, "total": 0.0}
        n_batches = 0
        for embedding, categorical_ids, continuous, cell_type, batch_code, donor_code in loader:
            embedding = embedding.to(device)
            categorical_ids = categorical_ids.to(device)
            continuous = continuous.to(device)
            cell_type = cell_type.to(device)
            batch_code = batch_code.to(device)
            donor_code = donor_code.to(device)

            corrected = head(embedding, categorical_ids, continuous)
            loss, metrics = correction_loss(
                original=embedding,
                corrected=corrected,
                labels=cell_type,
                donor_ids=donor_code,
                batch_ids=batch_code,
                contrastive_weight=train_cfg["contrastive_weight"],
                variance_weight=train_cfg["variance_weight"],
                donor_weight=train_cfg.get("donor_weight", 1.0),
                temperature=train_cfg["contrastive_temperature"],
                min_variance_ratio=train_cfg["min_variance_ratio"],
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_batches += 1

        avg = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}
        print(f"epoch {epoch:03d} | contrastive {avg['contrastive']:.4f} "
              f"| variance_penalty {avg['variance_penalty']:.4f} "
              f"| donor_consistency {avg['donor_consistency']:.4f} | total {avg['total']:.4f}")

    checkpoint_path = Path(train_cfg["checkpoint_out"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": head.state_dict(), "vocab": vocab.to_dict()}, checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path}")
    return head


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    _cli()
