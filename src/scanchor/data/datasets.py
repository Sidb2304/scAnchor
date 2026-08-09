"""Dataset wrapper over pre-embedded AnnData objects for correction-head training."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import torch
from torch.utils.data import Dataset

from scanchor.data.covariates import CovariateVocab


class EmbeddedCellDataset(Dataset):
    """Yields (embedding, categorical_covariates, continuous_covariates, cell_type, batch, donor).

    `batch` and `donor` are integer codes for grouping in the loss functions
    (donor_consistency_loss needs same-donor-different-batch pairs), separate
    from the covariate encoding pipeline. `donor` is -1 for every cell when
    `donor_col` isn't provided, which correction_loss/donor_consistency_loss
    treat as "identity unknown" and exclude from the donor term entirely.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        vocab: CovariateVocab,
        embedding_key: str = "X_scGPT",
        cell_type_col: str = "cell_type",
        batch_col: str = "batch",
        donor_col: str | None = None,
    ):
        self.embeddings = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
        self.categorical = vocab.encode_categorical(adata)
        self.continuous = vocab.encode_continuous(adata)
        self.cell_type_codes = adata.obs[cell_type_col].astype("category").cat.codes.to_numpy(dtype=np.int64)
        self.batch_codes = adata.obs[batch_col].astype("category").cat.codes.to_numpy(dtype=np.int64)
        if donor_col is not None:
            self.donor_codes = adata.obs[donor_col].astype("category").cat.codes.to_numpy(dtype=np.int64)
        else:
            self.donor_codes = np.full(adata.n_obs, -1, dtype=np.int64)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.embeddings[idx]),
            torch.from_numpy(self.categorical[idx]),
            torch.from_numpy(self.continuous[idx]),
            int(self.cell_type_codes[idx]),
            int(self.batch_codes[idx]),
            int(self.donor_codes[idx]),
        )


def load_reference_panel(
    paths: list[str | Path],
    categorical_cols: list[str],
    continuous_cols: list[str],
    embedding_key: str = "X_scGPT",
    cell_type_col: str = "cell_type",
    batch_col: str = "batch",
    donor_col: str | None = None,
) -> tuple[EmbeddedCellDataset, CovariateVocab]:
    """Concatenate multiple pre-embedded AnnData files into one training dataset.

    Building the covariate vocabulary over the full panel first is what lets a
    batch held out for the leave-one-batch-out test still get a defined (UNK)
    categorical embedding rather than crashing at eval time.
    """
    adatas = [ad.read_h5ad(p) for p in paths]
    combined = ad.concat(adatas, join="outer", label="_source_file", index_unique="-")
    vocab = CovariateVocab.build(combined, categorical_cols, continuous_cols)
    dataset = EmbeddedCellDataset(
        combined,
        vocab,
        embedding_key=embedding_key,
        cell_type_col=cell_type_col,
        batch_col=batch_col,
        donor_col=donor_col,
    )
    return dataset, vocab
