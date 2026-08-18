#!/bin/bash
#$ -N geneformer_stephenson_subsample
#$ -o geneformer_feasibility/build_subsample.log
#$ -j y
#$ -cwd
#$ -l h_vmem=32G
#$ -l h_rt=2:00:00
#$ -pe smp 2
#$ -binding linear:2
###############################################################################
# Builds the Geneformer-formatted version of the exact same Stephenson
# subsample already used for the published scGPT-based results, for a
# real apples-to-apples backbone comparison. No GPU needed -- just a
# backed-mode h5ad read + subsample, reuses the working `scanchor` conda
# env (has anndata/numpy already).
#
# Submit from the repo root:
#   qsub geneformer_feasibility/submit_uger_build_geneformer_subsample.sh
###############################################################################

set -euo pipefail

REPO_DIR="/stanley/nehme_lab/Siddharth/Projects/scAnchor"
CONDA_ENV_PATH="/stanley/nehme_lab/Siddharth/conda_envs/scanchor"
FULL_SOURCE="${REPO_DIR}/stephenson_data_cache/stephenson_covid_pbmc_full.h5ad"

for path_var in REPO_DIR CONDA_ENV_PATH FULL_SOURCE; do
    path_value="${!path_var}"
    if [[ ! -e "$path_value" ]]; then
        echo "ERROR: $path_var points to a path that doesn't exist: $path_value" >&2
        exit 1
    fi
done

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
    echo "ERROR: couldn't find conda.sh." >&2
    exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV_PATH"

cd "$REPO_DIR"
echo "Starting Geneformer subsample build: $(date)"

python geneformer_feasibility/build_geneformer_stephenson_subsample.py

echo "Finished Geneformer subsample build: $(date)"
###############################################################################
