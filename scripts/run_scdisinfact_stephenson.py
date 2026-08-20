"""Run scDisInFact on the same Stephenson et al. 2021 COVID-19 PBMC subsample
used by run_stephenson_benchmark.py, for a genuine same-dataset three-way
comparison against scAnchor and Harmony.

scDisInFact (https://github.com/ZhangLabGT/scDisInFact) is a conditional VAE
that disentangles an explicit *condition* variable (here: Status, taking
values Covid/Healthy/LPS/Non_covid) from batch effects (here: Site).
Unlike Harmony/scVI, its whole design assumes an independent condition axis,
which is exactly why this dataset (not Levy/Jerber/scIB) was chosen for it. It
operates on raw/normalized counts via its own generative model, not a
post-hoc correction of a frozen embedding, so this script doesn't touch
scGPT at all and can run independently of (in parallel with) the scGPT
embedding-extraction step in run_stephenson_benchmark.py.

Requires scDisInFact installed from source (not on PyPI):
    git clone https://github.com/ZhangLabGT/scDisInFact.git
    cd scDisInFact && pip install .

Example:
    python scripts/run_scdisinfact_stephenson.py \
        --out-dir /path/to/scratch/stephenson_run \
        --data-cache-dir /path/to/scratch/stephenson_data_cache
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from run_stephenson_benchmark import SOURCE_FILENAME, build_subsample, download_if_needed  # noqa: E402

from scDisInFact import create_scdisinfact_dataset, scdisinfact  # noqa: E402

from scanchor.evaluate.metrics import batch_mixing_purity, label_knn_purity  # noqa: E402

RNG_SEED = 0
NEPOCHS = 50  # scDisInFact's own default; their demo README uses 100 on GPU,
              # halved here as a starting point given CPU-only timing is
              # unknown for this dataset size; see printed epoch timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--data-cache-dir", required=True, type=Path)
    parser.add_argument("--nepochs", type=int, default=NEPOCHS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    full_path = args.data_cache_dir / SOURCE_FILENAME
    subsample_path = args.data_cache_dir / "stephenson_subsample.h5ad"
    download_if_needed(full_path)
    # Shared subsampling logic (same seed, same code path) as
    # run_stephenson_benchmark.py; if that script already ran, this reuses
    # its cached subsample.h5ad instead of rebuilding it, guaranteeing both
    # methods are compared on identical cells.
    sub = build_subsample(full_path, subsample_path)

    print(f"loaded subsample: {sub.n_obs} cells, {sub.n_vars} genes")
    print("Site x Status crosstab:")
    import pandas as pd
    print(pd.crosstab(sub.obs["batch"], sub.obs["Status"]))

    counts = sub.X
    if hasattr(counts, "toarray"):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=np.float32)

    meta_cells = sub.obs[["batch", "Status", "cell_type"]].copy()
    meta_cells.columns = ["Site", "Status", "cell_type"]

    print("\ncreating scDisInFact dataset (condition_key=['Status'], batch_key='Site')...")
    data_dict = create_scdisinfact_dataset(
        counts=counts, meta_cells=meta_cells, condition_key=["Status"], batch_key="Site"
    )
    print(f"split into {len(data_dict['datasets'])} (Site, Status) chunks")

    print(f"\n=== training scDisInFact (Ks=[8,4], nepochs={args.nepochs}) ===")
    t0 = time.time()
    model = scdisinfact(data_dict=data_dict, Ks=[8, 4], seed=RNG_SEED, device=device)
    model.train_model(nepochs=args.nepochs, recon_loss="NB")
    model.eval()
    print(f"training done  ({time.time() - t0:.0f}s elapsed)")

    torch.save(model.state_dict(), args.out_dir / "scdisinfact_model.pt")

    print("\nrunning inference to extract shared-bio (batch+condition-corrected) latent...")
    z_cs, sites, cell_types = [], [], []
    for dataset, meta in zip(data_dict["datasets"], data_dict["meta_cells"]):
        with torch.no_grad():
            dict_inf = model.inference(
                counts=dataset.counts_norm.to(device), batch_ids=dataset.batch_id[:, None].to(device)
            )
        z_cs.append(dict_inf["mu_c"].cpu().numpy())
        sites.append(meta["Site"].to_numpy())
        cell_types.append(meta["cell_type"].to_numpy())

    z_c = np.concatenate(z_cs, axis=0)
    site_labels = np.concatenate(sites, axis=0).astype(str)
    cell_type_labels = np.concatenate(cell_types, axis=0).astype(str)
    print(f"z_c shape: {z_c.shape}")

    results = {
        "dataset": "stephenson",
        "method": "scDisInFact",
        "batch_mixing_purity_scdisinfact": batch_mixing_purity(z_c, site_labels),
        "label_knn_purity_scdisinfact": label_knn_purity(z_c, cell_type_labels),
    }
    print("\n=== SUMMARY ===")
    print(results)
    print("\n(compare against run_stephenson_benchmark.py's before/scAnchor/Harmony numbers on the same subsample)")


if __name__ == "__main__":
    main()
