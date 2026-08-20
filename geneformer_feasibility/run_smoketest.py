"""Geneformer CPU feasibility smoke test: does the V1-10M checkpoint install
and run on a tiny (200-cell) subset in reasonable time, on CPU only?

Not a scAnchor integration yet, just answering the concrete question this
project always asks before committing engineering effort: does it actually
install and run, and how fast, on real hardware, not assumed from docs.

Selectively downloads only what's needed from the HF repo (it bundles every
model variant + fine-tuned checkpoints in one place, and a naive full clone
would be many unnecessary GB, the same class of problem that caused a real
OOM with scDisInFact's full git history earlier in this project).
"""
from __future__ import annotations

import time
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).parent
MODEL_CACHE = HERE / "geneformer_repo"


def download_geneformer_v1_10m():
    t0 = time.time()
    path = snapshot_download(
        repo_id="ctheodoris/Geneformer",
        local_dir=str(MODEL_CACHE),
        allow_patterns=[
            "Geneformer-V1-10M/*",
            "geneformer/*.py",
            "geneformer/mtl/*.py",
            "geneformer/gene_dictionaries_30m/*",
            "geneformer/*.pkl",  # V2 dicts too, imported at package init time regardless
            "requirements.txt",
            "setup.py",
        ],
    )
    print(f"downloaded to {path}  ({time.time() - t0:.0f}s elapsed)")
    return Path(path)


def main():
    repo_path = download_geneformer_v1_10m()

    import sys
    sys.path.insert(0, str(repo_path))
    t0 = time.time()
    from geneformer import EmbExtractor, TranscriptomeTokenizer
    print(f"import geneformer done  ({time.time() - t0:.0f}s elapsed)")

    tokenized_dir = HERE / "tokenized"
    tokenized_dir.mkdir(exist_ok=True)
    input_dir = HERE  # tiny_smoketest.h5ad lives directly here

    t0 = time.time()
    tokenizer = TranscriptomeTokenizer(
        model_version="V1",
        model_input_size=2048,
        special_token=False,
    )
    tokenizer.tokenize_data(
        data_directory=str(input_dir),
        output_directory=str(tokenized_dir),
        output_prefix="smoketest",
        file_format="h5ad",
    )
    print(f"tokenize done  ({time.time() - t0:.0f}s elapsed)")

    model_dir = repo_path / "Geneformer-V1-10M"
    out_dir = HERE / "embeddings_out"
    out_dir.mkdir(exist_ok=True)

    t0 = time.time()
    embex = EmbExtractor(
        model_type="Pretrained",
        emb_mode="cell",
        max_ncells=None,
        forward_batch_size=8,
        model_version="V1",
    )
    embs = embex.extract_embs(
        model_directory=str(model_dir),
        input_data_file=str(tokenized_dir / "smoketest.dataset"),
        output_directory=str(out_dir),
        output_prefix="smoketest",
    )
    elapsed = time.time() - t0
    print(f"embed done  ({elapsed:.0f}s elapsed for 200 cells, "
          f"{elapsed / 200:.2f}s/cell)")
    print("embedding shape:", embs.shape)


if __name__ == "__main__":
    main()
