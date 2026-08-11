"""Train the inductive correction head on a reference panel of embedded batches."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from scanchor.config import load_config
from scanchor.data.datasets import load_reference_panel
from scanchor.model.batch_discriminator import BatchAbsorber, BatchDiscriminator, dann_lambda_schedule
from scanchor.model.correction_head import CorrectionHead
from scanchor.model.losses import correction_loss


def train(config: dict) -> CorrectionHead:
    ref_cfg = config["reference_panel"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    seed = train_cfg.get("seed")
    if seed is not None:
        # Without this, comparing two configs (e.g. discriminator capacity,
        # loss weights) is confounded by a different random model init and
        # DataLoader shuffle order each run -- a real problem hit in practice:
        # reducing adversarial_weight (which should ease pressure on donor
        # retrieval) instead made it *worse* than a higher-weight run, a
        # non-monotonic result impossible to interpret without knowing how
        # much of it is init/shuffle noise vs. the actual change being tested.
        torch.manual_seed(seed)

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
    if len(loader) == 0:
        raise ValueError(
            f"reference panel has {len(dataset)} cells but batch_size={train_cfg['batch_size']} "
            "with drop_last=True -- every epoch would silently run zero minibatches (the head "
            "never trains and every metric stays at its identity-init value with no error raised). "
            "Lower training.batch_size or provide more reference data."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = CorrectionHead(
        embed_dim=model_cfg["embed_dim"],
        vocab_sizes=vocab.vocab_sizes(),
        n_continuous=len(ref_cfg["continuous_covariate_cols"]),
        cat_embed_dim=model_cfg["cat_embed_dim"],
        covariate_dim=model_cfg["covariate_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        max_delta_ratio=model_cfg.get("max_delta_ratio", 1.0),
        batch_latent_dim=model_cfg.get("batch_latent_dim", 32),
    ).to(device)

    # n_batches from the actual training data, not the covariate vocab (which
    # includes the +1 UNK slot for batches never seen during training).
    n_batches = int(dataset.batch_codes.max()) + 1
    discriminator = BatchDiscriminator(
        embed_dim=model_cfg["embed_dim"],
        n_batches=n_batches,
        hidden_dim=model_cfg.get("discriminator_hidden_dim", 256),
    ).to(device)
    absorber = BatchAbsorber(
        latent_dim=model_cfg.get("batch_latent_dim", 32),
        n_batches=n_batches,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(head.parameters()) + list(discriminator.parameters()) + list(absorber.parameters()),
        lr=train_cfg["learning_rate"],
    )
    max_adversarial_lambda = train_cfg.get("adversarial_lambda", 1.0)
    grad_clip_norm = train_cfg.get("grad_clip_norm", 5.0)
    all_params = list(head.parameters()) + list(discriminator.parameters()) + list(absorber.parameters())

    for epoch in range(train_cfg["epochs"]):
        adversarial_lambda = dann_lambda_schedule(
            progress=epoch / max(train_cfg["epochs"] - 1, 1), max_lambda=max_adversarial_lambda
        )
        epoch_metrics = {
            "contrastive": 0.0, "variance_penalty": 0.0, "donor_consistency": 0.0,
            "adversarial_batch": 0.0, "batch_absorption": 0.0, "mmd": 0.0, "total": 0.0,
        }
        n_minibatches = 0
        for embedding, categorical_ids, continuous, cell_type, batch_code, donor_code in loader:
            embedding = embedding.to(device)
            categorical_ids = categorical_ids.to(device)
            continuous = continuous.to(device)
            cell_type = cell_type.to(device)
            batch_code = batch_code.to(device)
            donor_code = donor_code.to(device)

            corrected, z_batch = head(embedding, categorical_ids, continuous, return_batch_latent=True)
            batch_logits = discriminator(corrected, lambd=adversarial_lambda)
            absorber_logits = absorber(z_batch)
            loss, metrics = correction_loss(
                original=embedding,
                corrected=corrected,
                labels=cell_type,
                donor_ids=donor_code,
                batch_ids=batch_code,
                batch_logits=batch_logits,
                absorber_logits=absorber_logits,
                contrastive_weight=train_cfg["contrastive_weight"],
                variance_weight=train_cfg["variance_weight"],
                donor_weight=train_cfg.get("donor_weight", 1.0),
                adversarial_weight=train_cfg.get("adversarial_weight", 1.0),
                absorption_weight=train_cfg.get("absorption_weight", 1.0),
                mmd_weight=train_cfg.get("mmd_weight", 0.0),
                temperature=train_cfg["contrastive_temperature"],
                min_variance_ratio=train_cfg["min_variance_ratio"],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip_norm)
            optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] += v
            n_minibatches += 1

        avg = {k: v / max(n_minibatches, 1) for k, v in epoch_metrics.items()}
        print(f"epoch {epoch:03d} | lambda {adversarial_lambda:.4f} | contrastive {avg['contrastive']:.4f} "
              f"| variance_penalty {avg['variance_penalty']:.4f} "
              f"| donor_consistency {avg['donor_consistency']:.4f} "
              f"| adversarial_batch {avg['adversarial_batch']:.4f} "
              f"| batch_absorption {avg['batch_absorption']:.4f} | mmd {avg['mmd']:.4f} "
              f"| total {avg['total']:.4f}")

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
