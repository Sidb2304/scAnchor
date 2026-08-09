"""Extract frozen scGPT cell embeddings for a reference or query AnnData object."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad


def extract_embeddings(
    adata_path: str | Path,
    model_dir: str | Path,
    out_path: str | Path,
    gene_col: str = "feature_name",
    batch_size: int = 64,
    obs_to_save: list[str] | None = None,
    device: str = "cuda",
) -> ad.AnnData:
    """Embed cells with a frozen scGPT checkpoint and write the result to disk.

    Embeddings land in ``adata.obsm["X_scGPT"]``. Requires the optional
    ``scgpt`` dependency (``pip install -e ".[scgpt]"``).
    """
    from scgpt.tasks import embed_data

    embedded = embed_data(
        adata_or_file=str(adata_path),
        model_dir=str(model_dir),
        gene_col=gene_col,
        batch_size=batch_size,
        obs_to_save=obs_to_save,
        device=device,
        return_new_adata=True,
    )
    embedded.write_h5ad(out_path)
    return embedded


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adata", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gene-col", default="feature_name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--obs-to-save", nargs="*", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    extract_embeddings(
        adata_path=args.adata,
        model_dir=args.model_dir,
        out_path=args.out,
        gene_col=args.gene_col,
        batch_size=args.batch_size,
        obs_to_save=args.obs_to_save,
        device=args.device,
    )


if __name__ == "__main__":
    _cli()
