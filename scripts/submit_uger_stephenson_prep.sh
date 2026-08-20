#!/bin/bash
#$ -N scanchor_stephenson_prep
#$ -o logs/scanchor_stephenson_prep.log
#$ -j y
#$ -cwd
#$ -l h_vmem=32G
#$ -l h_rt=4:00:00
#$ -pe smp 2
#$ -binding linear:2
###############################################################################
# One-time prep for the Stephenson et al. 2021 COVID-19 PBMC benchmark:
# downloads the full 647k-cell / ~7GB dataset from CELLxGENE and subsamples
# it (capped per-donor) to a tractable scale, shared by both
# run_stephenson_benchmark.py (scAnchor+Harmony) and
# run_scdisinfact_stephenson.py (scDisInFact).
#
# Run via scripts/submit_stephenson_pipeline.sh (NOT qsub'd directly);
# that wrapper chains this job before the two comparison jobs via
# -hold_jid, so they don't race to build the same cache file independently.
###############################################################################

set -euo pipefail

REPO_DIR="/stanley/nehme_lab/Siddharth/Projects/scAnchor"
CONDA_ENV_PATH="/stanley/nehme_lab/Siddharth/conda_envs/scanchor"
DATA_CACHE_DIR="${REPO_DIR}/stephenson_data_cache"

for path_var in REPO_DIR CONDA_ENV_PATH; do
    path_value="${!path_var}"
    if [[ ! -e "$path_value" ]]; then
        echo "ERROR: $path_var points to a path that doesn't exist: $path_value" >&2
        exit 1
    fi
done
mkdir -p "$DATA_CACHE_DIR" "$REPO_DIR/logs"

CONDA_SH=""
for conda_base in \
    "/broad/software/free/Linux/redhat_7_x86_64/pkgs/anaconda3_2022.10" \
    "$(conda info --base 2>/dev/null || true)"
do
    if [[ -n "$conda_base" && -e "${conda_base}/etc/profile.d/conda.sh" ]]; then
        CONDA_SH="${conda_base}/etc/profile.d/conda.sh"
        break
    fi
done
if [[ -z "$CONDA_SH" ]]; then
    echo "ERROR: couldn't find conda.sh; see submit_uger_scib.sh's comments." >&2
    exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV_PATH"

cd "$REPO_DIR"
echo "Starting Stephenson prep: $(date)"
python scripts/prepare_stephenson_subsample.py --data-cache-dir "$DATA_CACHE_DIR"
echo "Finished Stephenson prep: $(date)"
###############################################################################
