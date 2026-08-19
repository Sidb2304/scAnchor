#!/bin/bash
#$ -N scanchor_levy_combined_seedcheck
#$ -o levy_run/levy_combined_seedcheck.log
#$ -j y
#$ -cwd
#$ -l h_vmem=32G
#$ -l h_rt=4:00:00
#$ -l gpu=1
#$ -l operating_system=RedHat8
###############################################################################
# Seed-checks the two most promising combined mmd_weight+sinkhorn_weight
# points found by run_levy_combined_comparison.py's single-seed sweep,
# using scripts/_vectorized_batch_losses.py instead of losses.py's
# sequential-per-pair mmd_loss/sinkhorn_ot_loss -- numerically verified
# equivalent (9/9 checks incl. gradcheck), batches every batch-pair into
# one op instead of looping over up to 28 pairs sequentially. A local
# 2-epoch CPU smoke test of the full training loop ran in ~13s/epoch with
# no errors, extrapolating to well under the ~50 min/seed the sequential
# version took on GPU for a comparable config -- real evidence this fix
# works, not just the isolated numerical-equivalence checks.
#
# Same proven GPU fixes/env as every other job in this project.
#
# Submit from the repo root: qsub scripts/submit_uger_levy_combined_seedcheck.sh
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
mkdir -p levy_run
echo "Starting Levy combined mmd+sinkhorn seed-check (vectorized): $(date)"

python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

python scripts/run_levy_combined_seedcheck.py

echo "Finished Levy combined mmd+sinkhorn seed-check (vectorized): $(date)"
###############################################################################
