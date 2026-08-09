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
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_sizes: list[int],
        n_continuous: int,
        cat_embed_dim: int = 8,
        covariate_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
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
        return embedding + delta
