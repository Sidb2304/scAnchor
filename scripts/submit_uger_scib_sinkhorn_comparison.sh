#!/bin/bash
#$ -N scanchor_scib_sinkhorn
#$ -o scib_benchmark_run/scib_sinkhorn_comparison.log
#$ -j y
#$ -cwd
#$ -l h_vmem=32G
#$ -l h_rt=6:00:00
#$ -l gpu=1
#$ -l operating_system=RedHat8
###############################################################################
# Completes Sinkhorn's validation against MMD's original bar: the scIB
# atlas-level benchmarks (immune, pancreas, lung) were the one remaining
# axis flagged as untested in README's Net assessment for sinkhorn_weight.
#
# Uses scripts/_vectorized_batch_losses.py from the start, since these
# datasets have real batch counts (immune=10, pancreas=9, lung=16, all
# higher than Stephenson's 3, lung higher even than Levy's 8), and that
# vectorization was a ~16x speedup at Levy's smaller scale.
#
# Reuses already-cached real scGPT embeddings
# (scib_benchmark_run/{dataset}/{reference,heldout}.h5ad) from the
# original MMD scIB validation, so no new embedding extraction, no new
# data download.
#
# Same proven GPU fixes/env as every other job in this project.
#
# Submit from the repo root: qsub scripts/submit_uger_scib_sinkhorn_comparison.sh
###############################################################################

set -euo pipefail

REPO_DIR="/stanley/nehme_lab/Siddharth/Projects/scAnchor"
ENV_NAME="scanchor"
ENV_PATH="/stanley/nehme_lab/Siddharth/conda_envs/${ENV_NAME}"

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
    echo "ERROR: couldn't find conda.sh" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate "$ENV_PATH"

cd "$REPO_DIR"
echo "Starting scIB sinkhorn-vs-mmd comparison: $(date)"

python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

python scripts/run_scib_sinkhorn_comparison.py

echo "Finished scIB sinkhorn-vs-mmd comparison: $(date)"
###############################################################################
