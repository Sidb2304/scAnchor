"""Real-scale Geneformer embedding extraction for the Stephenson subsample
(21,000 cells, identical to the already-published scGPT-based results).

Same pattern validated in the smoke test (run_smoketest.py): selectively
download just the V1-10M checkpoint + package code, tokenize, extract cell
embeddings. Only difference here is scale (21K cells vs. 200) and saving
the obs columns scAnchor's training/evaluation actually needs.
"""
from __future__ import annotations

import time
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).parent
INPUT_H5AD = HERE / "stephenson_subsample_geneformer.h5ad"
MODEL_CACHE = HERE / "geneformer_repo"
TOKENIZED_DIR = HERE / "tokenized_stephenson"
OUT_DIR = HERE / "embeddings_out_stephenson"


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
            "geneformer/*.pkl",
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
    from geneformer import EmbExtractor, TranscriptomeTokenizer

    TOKENIZED_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    tokenizer = TranscriptomeTokenizer(
        # emb_label carries these obs columns through tokenization so
        # they survive into the extracted embedding output --
        # everything scAnchor's train()/evaluate need downstream.
        custom_attr_name_dict={"batch": "batch", "cell_type": "cell_type",
                                "total_counts": "total_counts", "pct_counts_mt": "pct_counts_mt"},
        model_version="V1",
        model_input_size=2048,
        special_token=False,
    )
    tokenizer.tokenize_data(
        data_directory=str(HERE),
        output_directory=str(TOKENIZED_DIR),
        output_prefix="stephenson",
        file_format="h5ad",
        # Without this, tokenize_files() picks up every .h5ad in HERE --
        # including the leftover tiny_smoketest.h5ad from the earlier
        # feasibility spike, which lacks the batch/cell_type columns and
        # crashes tokenization (confirmed directly, not assumed).
        input_identifier="stephenson_subsample_geneformer",
    )
    print(f"tokenize done  ({time.time() - t0:.0f}s elapsed)")

    t0 = time.time()
    embex = EmbExtractor(
        model_type="Pretrained",
        emb_mode="cell",
        max_ncells=None,
        forward_batch_size=64,
        emb_label=["batch", "cell_type", "total_counts", "pct_counts_mt"],
        model_version="V1",
    )
    model_dir = MODEL_CACHE / "Geneformer-V1-10M"
    embs_df = embex.extract_embs(
        model_directory=str(model_dir),
        input_data_file=str(TOKENIZED_DIR / "stephenson.dataset"),
        output_directory=str(OUT_DIR),
        output_prefix="stephenson",
    )
    elapsed = time.time() - t0
    n_cells = len(embs_df)
    print(f"embed done  ({elapsed:.0f}s elapsed for {n_cells} cells, {elapsed / n_cells:.3f}s/cell)")
    print("embedding df shape:", embs_df.shape)
    print("columns:", embs_df.columns.tolist()[:10], "...")


if __name__ == "__main__":
    main()
