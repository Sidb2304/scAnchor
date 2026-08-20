#!/bin/bash
#$ -N scanchor_levy_combined
#$ -o levy_run/levy_combined_comparison.log
#$ -j y
#$ -cwd
#$ -l h_vmem=32G
#$ -l h_rt=8:00:00
#$ -l gpu=1
#$ -l operating_system=RedHat8
###############################################################################
# Does combining mmd_weight + sinkhorn_weight in one correction head beat
# either mechanism alone on Levy? Motivated by run_levy_sinkhorn_comparison.py's
# real result: MMD and Sinkhorn have complementary weaknesses there (MMD
# weak on donor retrieval/cell-type purity, Sinkhorn weak on batch-mixing),
# real evidence worth testing directly, not assuming.
#
# 6 single-seed runs, most including the expensive Sinkhorn term (its
# 50-iteration-per-batch-pair solve dominates cost regardless of whether
# MMD is also active), so h_rt=8:00:00 gives real headroom given
# run_levy_sinkhorn_comparison.py's sinkhorn-only runs took ~50 min/seed.
#
# Same two proven GPU fixes as every other job in this project (gpu=1,
# operating_system=RedHat8) and reuses the already-working scanchor env
# (CUDA torch already installed from the last two Levy jobs, persists on
# shared storage, so no reinstall needed).
#
# Submit from the repo root: qsub scripts/submit_uger_levy_combined_comparison.sh
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
echo "Starting Levy combined mmd+sinkhorn comparison: $(date)"

python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

python scripts/run_levy_combined_comparison.py

echo "Finished Levy combined mmd+sinkhorn comparison: $(date)"
###############################################################################
