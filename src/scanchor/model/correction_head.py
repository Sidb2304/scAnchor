"""The inductive correction head: e' = e + g_theta(e, covariates)."""

from __future__ import annotations

import torch
from torch import nn

from scanchor.data.covariates import CovariateEncoder


class CorrectionHead(nn.Module):
    """Residual correction on top of a frozen foundation-model embedding.

    The residual formulation (`e' = e + delta`) rather than `e' = f(e)` matters:
    it keeps the head close to the identity map at initialization, so training
    starts from "trust the foundation model" and only learns to move cells as
    far as the covariate-conditioned signal justifies — rather than having to
    relearn a faithful reconstruction of `e` from scratch.

    `max_delta_ratio` bounds ||delta|| to at most this fraction of ||embedding||
    per cell. Without it, an adversarial batch-discriminator objective has a
    degenerate way to "win": blow up embedding magnitude in some direction
    that saturates the discriminator's logits, rather than genuinely removing
    batch structure. That's not a hypothetical -- unbounded delta is what
    caused a real runaway divergence on real data (adversarial loss climbing
    into the tens of thousands over training, every downstream metric
    collapsing). The variance-floor loss doesn't catch this: it only
    penalizes variance dropping too low, not growing unbounded.

    A second, small output head produces `z_batch` -- a compact latent meant
    to *absorb* batch-predictive variance (trained normally, not
    adversarially, to actually predict batch well) rather than fight it out
    of the main corrected embedding. Without this split, one shared
    embedding has to simultaneously satisfy "pull together" objectives
    (contrastive on cell type, donor consistency) and "push apart from
    batch" (the adversarial term) -- and that fight is a real, evidenced
    problem: batch-mixing regressed after correction across every dataset,
    scale, and discriminator capacity tried before this. `z_batch` gives
    batch-specific variance somewhere to go instead of forcing it out of the
    representation everything downstream depends on; it's discarded after
    training, only the corrected embedding (`embedding + delta_bio`) is used
    for inference/evaluation.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_sizes: list[int],
        n_continuous: int,
        cat_embed_dim: int = 8,
        covariate_dim: int = 32,
        hidden_dim: int = 256,
        max_delta_ratio: float = 1.0,
        batch_latent_dim: int = 32,
    ):
        super().__init__()
        self.max_delta_ratio = max_delta_ratio
        self.covariate_encoder = CovariateEncoder(
            vocab_sizes=vocab_sizes,
            n_continuous=n_continuous,
            cat_embed_dim=cat_embed_dim,
            out_dim=covariate_dim,
        )
        self.shared_trunk = nn.Sequential(
            nn.Linear(embed_dim + covariate_dim, hidden_dim),
            nn.ReLU(),
        )
        self.bio_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.batch_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, batch_latent_dim),
        )
        # Zero-init the bio head's last layer so delta starts at 0 and
        # e' == e at step 0. batch_head has no such requirement -- it's
        # discarded downstream, only its predictiveness during training
        # matters.
        nn.init.zeros_(self.bio_head[-1].weight)
        nn.init.zeros_(self.bio_head[-1].bias)

    def forward(
        self,
        embedding: torch.Tensor,
        categorical_ids: torch.Tensor,
        continuous: torch.Tensor,
        return_batch_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        c = self.covariate_encoder(categorical_ids, continuous)
        hidden = self.shared_trunk(torch.cat([embedding, c], dim=-1))
        delta = self.bio_head(hidden)

        delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        max_norm = self.max_delta_ratio * embedding.norm(dim=-1, keepdim=True)
        scale = (max_norm / delta_norm).clamp(max=1.0)
        delta = delta * scale
        corrected = embedding + delta

        if return_batch_latent:
            return corrected, self.batch_head(hidden)
        return corrected
