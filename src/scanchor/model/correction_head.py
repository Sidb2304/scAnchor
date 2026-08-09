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
    ):
        super().__init__()
        self.max_delta_ratio = max_delta_ratio
        self.covariate_encoder = CovariateEncoder(
            vocab_sizes=vocab_sizes,
            n_continuous=n_continuous,
            cat_embed_dim=cat_embed_dim,
            out_dim=covariate_dim,
        )
        self.delta_net = nn.Sequential(
            nn.Linear(embed_dim + covariate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # Zero-init the last layer so delta starts at 0 and e' == e at step 0.
        nn.init.zeros_(self.delta_net[-1].weight)
        nn.init.zeros_(self.delta_net[-1].bias)

    def forward(
        self,
        embedding: torch.Tensor,
        categorical_ids: torch.Tensor,
        continuous: torch.Tensor,
    ) -> torch.Tensor:
        c = self.covariate_encoder(categorical_ids, continuous)
        delta = self.delta_net(torch.cat([embedding, c], dim=-1))

        delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        max_norm = self.max_delta_ratio * embedding.norm(dim=-1, keepdim=True)
        scale = (max_norm / delta_norm).clamp(max=1.0)
        delta = delta * scale

        return embedding + delta
