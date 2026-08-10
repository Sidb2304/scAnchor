"""Gradient-reversal batch discriminator -- the "push apart from batch
structure" term the correction objective was missing.

Without this, correction_loss only had "pull together" terms (contrastive on
cell type, donor consistency): nothing directly penalized the corrected
embedding for still encoding batch identity. Empirically, that's exactly what
happened -- batch-mixing purity on a held-out batch got WORSE after
correction on real data, across two different scGPT checkpoints.

Standard domain-adversarial (DANN) setup: a small classifier tries to predict
batch identity from the corrected embedding; a gradient-reversal layer flips
the sign of the gradient that flows back through it, so a single backward()
pass trains the discriminator normally (it gets better at predicting batch)
while pushing the correction head in the OPPOSITE direction -- to make batch
identity harder to predict.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def gradient_reversal(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Identity in the forward pass; negates and scales the gradient in backward."""
    return _GradientReversal.apply(x, lambd)


def dann_lambda_schedule(progress: float, max_lambda: float = 1.0, gamma: float = 10.0) -> float:
    """Ganin & Lempitsky (2015)'s ramp: 0 at progress=0, ~max_lambda at progress=1.

    A fixed lambda from step one is a real failure mode, not a theoretical
    one: with lambd=1.0 from epoch 0, the discriminator and correction head
    entered a runaway feedback loop on real data here (adversarial loss went
    2.07 -> 30,224 over 30 epochs, and both downstream metrics collapsed).
    Ramping lets the correction head learn the "pull together" objectives on
    a still-close-to-identity embedding before adversarial pressure kicks in.
    """
    progress = min(max(progress, 0.0), 1.0)
    return max_lambda * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


class BatchDiscriminator(nn.Module):
    """Predicts batch identity from a corrected embedding, through a GRL.

    Training-time only -- discard after training. Evaluation (replicate
    test, leave-one-batch-out) only needs the CorrectionHead; the
    discriminator's job is done once it's shaped the correction head's
    gradients during training.

    Two hidden layers at hidden_dim=256 by default, not one layer at 64: a
    capacity mismatch against CorrectionHead's delta_net (2 hidden layers,
    128 units) is a real, evidenced failure mode, not a hypothetical one.
    On real data, the shallow 1-layer/64-unit discriminator converged to
    chance-level loss (the correct adversarial equilibrium by its own
    metric) at two different dataset scales (3.4k and 18.2k cells), yet
    batch-mixing purity by a kNN metric got WORSE after correction both
    times, by almost the same margin regardless of scale. A discriminator
    that's too weak to detect batch structure a kNN metric picks up can
    reach equilibrium without ever forcing the correction head to remove
    that structure -- it can only push back against what it can see.
    """

    def __init__(self, embed_dim: int, n_batches: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_batches),
        )

    def forward(self, embeddings: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.net(gradient_reversal(embeddings, lambd))
