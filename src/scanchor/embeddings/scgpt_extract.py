"""Extract frozen scGPT cell embeddings for a reference or query AnnData object."""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

import anndata as ad


@contextlib.contextmanager
def _force_single_process_dataloading():
    """Shim os.sched_getaffinity for the duration of an embed_data() call only.

    scGPT's dataloader setup calls os.sched_getaffinity(0) unconditionally,
    which is Linux-only (it was built/tested on a SLURM cluster) and raises
    AttributeError on macOS/BSD. Returning an empty set forces num_workers=0:
    scGPT's embedding function defines its Dataset as a local class, which
    can't be pickled for the "spawn" multiprocessing start method macOS uses
    (Linux's default "fork" start method doesn't need to pickle it, so
    upstream never hit this).

    Scoped as a context manager rather than a module-level monkeypatch:
    torch itself calls os.sched_getaffinity elsewhere (e.g. torch._inductor's
    compile-thread count) and asserts it's non-empty, so leaving an empty-set
    shim in place process-wide breaks unrelated torch internals later in the
    same process (observed when calling scanchor.train.train() afterward).
    """
    original = getattr(os, "sched_getaffinity", None)
    os.sched_getaffinity = lambda _pid: set()
    try:
        yield
    finally:
        if original is None:
            del os.sched_getaffinity
        else:
            os.sched_getaffinity = original


def extract_embeddings(
    adata_path: str | Path,
    model_dir: str | Path,
    out_path: str | Path,
    gene_col: str = "feature_name",
    batch_size: int = 64,
    obs_to_save: list[str] | None = None,
    device: str = "cuda",
    use_fast_transformer: bool = True,
) -> ad.AnnData:
    """Embed cells with a frozen scGPT checkpoint and write the result to disk.

    Embeddings land in ``adata.obsm["X_scGPT"]``. Requires the optional
    ``scgpt`` dependency (``pip install -e ".[scgpt]"``). Expects raw (or at
    least unscaled) counts in ``adata.X``: scGPT applies its own binning
    internally, so pre-normalizing/scaling defeats that.

    Set ``use_fast_transformer=False`` when ``flash-attn`` isn't installed
    (e.g. CPU-only environments); the checkpoint was trained with it enabled
    but scGPT's transformer layer supports a plain-PyTorch-attention fallback.
    """
    from scgpt.tasks import embed_data

    with _force_single_process_dataloading():
        embedded = embed_data(
            adata_or_file=str(adata_path),
            model_dir=str(model_dir),
            gene_col=gene_col,
            batch_size=batch_size,
            obs_to_save=obs_to_save,
            device=device,
            use_fast_transformer=use_fast_transformer,
            return_new_adata=True,
        )
    # With return_new_adata=True, scGPT puts the embedding in .X (it only
    # writes .obsm["X_scGPT"] on the return_new_adata=False path, which
    # mutates the full input adata instead of returning a lightweight one).
    # Move it to .obsm here so callers get one consistent key regardless.
    embedded.obsm["X_scGPT"] = embedded.X
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
    parser.add_argument("--no-fast-transformer", action="store_true")
    args = parser.parse_args()

    extract_embeddings(
        adata_path=args.adata,
        model_dir=args.model_dir,
        out_path=args.out,
        gene_col=args.gene_col,
        batch_size=args.batch_size,
        obs_to_save=args.obs_to_save,
        device=args.device,
        use_fast_transformer=not args.no_fast_transformer,
    )


if __name__ == "__main__":
    _cli()
