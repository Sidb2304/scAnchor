#!/bin/bash
#$ -N geneformer_stephenson_benchmark
#$ -o geneformer_feasibility/geneformer_stephenson_benchmark.log
#$ -j y
#$ -cwd
#$ -l h_vmem=16G
#$ -l h_rt=2:00:00
###############################################################################
# Cross-backbone validation: train scAnchor's already-validated config on
# the Geneformer embeddings just extracted, evaluate, and compare directly
# against the published scGPT-backbone numbers. No GPU needed -- just the
# lightweight correction-head training and Harmony, both CPU-cheap. Uses
# the working `scanchor` conda env directly (this is scAnchor's own repo).
#
# Submit from the repo root:
#   qsub geneformer_feasibility/submit_uger_geneformer_stephenson_benchmark.sh
###############################################################################

set -euo pipefail

REPO_DIR="/stanley/nehme_lab/Siddharth/Projects/scAnchor"
CONDA_ENV_PATH="/stanley/nehme_lab/Siddharth/conda_envs/scanchor"

for path_var in REPO_DIR CONDA_ENV_PATH; do
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
echo "Starting Geneformer Stephenson benchmark: $(date)"

python geneformer_feasibility/run_geneformer_stephenson_benchmark.py

echo "Finished Geneformer Stephenson benchmark: $(date)"
###############################################################################
