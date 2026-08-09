"""Transductive baselines, run with full access to the held-out batch.

These get an advantage scAnchor's inductive setup deliberately doesn't have —
the point of the comparison is to see how close the inductive head gets
without needing the held-out batch's cells at correction time, not to prove
it's strictly better.
"""

from __future__ import annotations

import anndata as ad
import numpy as np


def harmony_correct(adata: ad.AnnData, embedding_key: str, batch_col: str) -> np.ndarray:
    """Requires the optional `harmonypy` dependency."""
    import harmonypy

    ho = harmonypy.run_harmony(adata.obsm[embedding_key], adata.obs, [batch_col])
    return ho.Z_corr.T


def scvi_correct(
    adata: ad.AnnData,
    counts_layer: str,
    batch_col: str,
    n_latent: int = 32,
    max_epochs: int = 100,
) -> np.ndarray:
    """Requires the optional `scvi-tools` dependency.

    Note this operates on raw counts, not the foundation-model embedding — scVI
    is a full generative model of expression, not a post-hoc embedding
    corrector, so this baseline answers a related but not identical question.
    """
    import scvi

    adata = adata.copy()
    adata.X = adata.layers[counts_layer]
    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_col)
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    model.train(max_epochs=max_epochs)
    return model.get_latent_representation()
