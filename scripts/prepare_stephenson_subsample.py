"""Download + subsample the Stephenson et al. 2021 COVID-19 PBMC atlas once,
shared by both run_stephenson_benchmark.py (scAnchor+Harmony) and
run_scdisinfact_stephenson.py (scDisInFact) so they compare on identical
cells. Run this first (see scripts/submit_stephenson_pipeline.sh) rather
than letting both downstream scripts race to build the same cache file
independently.

Example:
    python scripts/prepare_stephenson_subsample.py --data-cache-dir /path/to/scratch/stephenson_data_cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_stephenson_benchmark import SOURCE_FILENAME, build_subsample, download_if_needed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-cache-dir", required=True, type=Path)
    args = parser.parse_args()

    args.data_cache_dir.mkdir(parents=True, exist_ok=True)
    full_path = args.data_cache_dir / SOURCE_FILENAME
    subsample_path = args.data_cache_dir / "stephenson_subsample.h5ad"

    download_if_needed(full_path)
    build_subsample(full_path, subsample_path)
    print(f"\nprep done -- subsample cached at {subsample_path}")


if __name__ == "__main__":
    main()
