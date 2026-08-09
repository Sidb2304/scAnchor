"""Technical covariate encoding for the correction head.

Categorical covariates (batch ID, platform, chemistry) get a learned
embedding with an explicit UNK slot, so a batch never seen during training
still gets a valid (if generic) representation at inference. Continuous
covariates (sequencing depth, %mito, etc.) are always computable for a new
batch and carry most of the inductive generalization signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import torch
from torch import nn

UNK_TOKEN = "__unk__"


@dataclass
class CovariateVocab:
    categorical_cols: list[str]
    continuous_cols: list[str]
    vocabs: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        adata: ad.AnnData,
        categorical_cols: list[str],
        continuous_cols: list[str],
    ) -> "CovariateVocab":
        vocabs = {}
        for col in categorical_cols:
            values = sorted(adata.obs[col].astype(str).unique().tolist())
            vocabs[col] = {UNK_TOKEN: 0, **{v: i + 1 for i, v in enumerate(values)}}
        return cls(categorical_cols, continuous_cols, vocabs)

    def to_dict(self) -> dict:
        """Plain-dict form for checkpointing.

        Deliberately not the dataclass instance itself: `torch.load` defaults
        to `weights_only=True` (PyTorch >=2.6), which refuses to unpickle
        arbitrary custom classes. Round-tripping through plain dict/list/str
        keeps checkpoints loadable without disabling that safety check.
        """
        return {
            "categorical_cols": self.categorical_cols,
            "continuous_cols": self.continuous_cols,
            "vocabs": self.vocabs,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CovariateVocab":
        return cls(data["categorical_cols"], data["continuous_cols"], data["vocabs"])

    def encode_categorical(self, adata: ad.AnnData) -> np.ndarray:
        cols = []
        for col in self.categorical_cols:
            vocab = self.vocabs[col]
            ids = adata.obs[col].astype(str).map(lambda v: vocab.get(v, vocab[UNK_TOKEN]))
            cols.append(ids.to_numpy(dtype=np.int64))
        return np.stack(cols, axis=1) if cols else np.zeros((adata.n_obs, 0), dtype=np.int64)

    def encode_continuous(self, adata: ad.AnnData) -> np.ndarray:
        if not self.continuous_cols:
            return np.zeros((adata.n_obs, 0), dtype=np.float32)
        vals = adata.obs[self.continuous_cols].to_numpy(dtype=np.float32)
        mean = vals.mean(axis=0, keepdims=True)
        std = vals.std(axis=0, keepdims=True) + 1e-6
        return (vals - mean) / std

    def vocab_sizes(self) -> list[int]:
        return [len(self.vocabs[col]) for col in self.categorical_cols]


class CovariateEncoder(nn.Module):
    """Maps categorical + continuous covariates to a fixed-size embedding `c`."""

    def __init__(
        self,
        vocab_sizes: list[int],
        n_continuous: int,
        cat_embed_dim: int = 8,
        out_dim: int = 32,
    ):
        super().__init__()
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, cat_embed_dim) for vocab_size in vocab_sizes]
        )
        in_dim = cat_embed_dim * len(vocab_sizes) + n_continuous
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, categorical_ids: torch.Tensor, continuous: torch.Tensor) -> torch.Tensor:
        parts = [emb(categorical_ids[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        parts.append(continuous)
        return self.proj(torch.cat(parts, dim=-1))
