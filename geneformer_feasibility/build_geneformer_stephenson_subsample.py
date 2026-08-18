"""Build the Geneformer-formatted version of the exact same Stephenson
subsample already used for the published scGPT-based results (README's
Current results), so the two backbones are compared on identical cells --
a real apples-to-apples test of whether scAnchor's validated technique
generalizes beyond scGPT.

Cell-selection logic below is a DELIBERATE, exact duplicate of
scripts/run_stephenson_benchmark.py's build_subsample() (same RNG_SEED,
same PER_DONOR_CAP, same per-donor groupby) -- not reused via import,
since that function also does scGPT-specific gene-symbol remapping we
don't want here. If those constants ever change, this file must change
to match, or the comparison stops being apples-to-apples.

Geneformer needs var["ensembl_id"] + obs["n_counts"] (confirmed directly
from geneformer/tokenizer.py during the feasibility spike) -- notably
LESS reformatting than scGPT needed, since this file's var_names are
already Ensembl IDs before any remapping.
"""
from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np

REPO_DIR = Path(__file__).parent.parent
FULL_SOURCE = REPO_DIR / "stephenson_data_cache" / "stephenson_covid_pbmc_full.h5ad"
OUT_PATH = Path(__file__).parent / "stephenson_subsample_geneformer.h5ad"

# Must exactly match run_stephenson_benchmark.py's build_subsample().
RNG_SEED = 0
PER_DONOR_CAP = 175


def main():
    print(f"loading full dataset from {FULL_SOURCE} (backed, for memory) ...")
    full = ad.read_h5ad(FULL_SOURCE, backed="r")
    print(f"full dataset: {full.n_obs} cells, {full.n_vars} genes")

    rng = np.random.default_rng(RNG_SEED)
    groups = full.obs.groupby("donor_id", observed=True).indices
    keep_idx = []
    for _, idx in groups.items():
        idx = np.array(idx)
        n = min(PER_DONOR_CAP, len(idx))
        keep_idx.extend(rng.choice(idx, size=n, replace=False))
    keep_idx = np.array(sorted(keep_idx))

    print(f"subsampling to {len(keep_idx)} cells across {full.obs['donor_id'].nunique()} donors ...")
    sub = full[keep_idx].to_memory()

    # Same uns-bloat fix as v1.1.0's build_subsample() -- this file's
    # uns/antibody_X etc. are full-647k-cell-sized and unused here.
    sub.uns.clear()

    # Raw counts live in .raw.X (X itself is normalized), confirmed
    # directly from this file's own metadata, same as the scGPT version.
    sub.X = sub.raw[:, sub.var_names].X.copy()

    # var_names ARE ALREADY Ensembl IDs at this point (no remapping done
    # yet, unlike the scGPT version which overwrites them with gene
    # symbols) -- just copy into the column name Geneformer expects.
    sub.var["ensembl_id"] = sub.var_names.astype(str)

    # obs["total_counts"] is a real, already-validated column computed
    # from raw counts by this file's original CELLxGENE processing (used
    # successfully as a scGPT continuous covariate already) -- reuse
    # directly as Geneformer's expected obs["n_counts"], rather than
    # recomputing from sub.X.
    sub.obs["n_counts"] = sub.obs["total_counts"].astype(float)
    sub.obs["batch"] = sub.obs["Site"].astype(str)
    sub.obs["cell_type"] = sub.obs["cell_type"].astype(str)
    sub.obs["pct_counts_mt"] = sub.obs["pct_counts_mt"].astype(float)

    print(f"subsample: {sub.n_obs} cells, {sub.obs['batch'].nunique()} sites, "
          f"{sub.obs['donor_id'].nunique()} donors, {sub.obs['cell_type'].nunique()} cell types")
    print("site value counts:", sub.obs["batch"].value_counts().to_dict())

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(OUT_PATH)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
