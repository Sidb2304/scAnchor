#!/bin/bash
#$ -N scanchor_levy_sinkhorn
#$ -o levy_run/levy_sinkhorn_comparison.log
#$ -j y
#$ -cwd
#$ -l h_vmem=32G
#$ -l h_rt=4:00:00
#$ -l gpu=1
#$ -l operating_system=RedHat8
###############################################################################
# Trains scripts/run_levy_sinkhorn_comparison.py's mmd_weight=20 vs
# sinkhorn_weight=0.5 comparison on the real, already-cached Levy scGPT
# embeddings (levy_run/{reference,heldout}.h5ad -- extracted by
# submit_uger_levy_full_dataset.sh's earlier GPU job, no need to redo).
#
# This was first tried on the local Mac's CPU and found to be much slower
# than expected: correction_loss always computes mmd_loss +
# class_conditional_mmd_loss + sinkhorn_ot_loss every step regardless of
# their weight (by design -- the metrics dict reports each term's real
# value even when unused), and at Levy's real 8-batch x 14-cell-type
# scale that's up to 28 batch-pairs x 14 cell-types = 392 pairwise
# kernel/Sinkhorn computations per minibatch, for terms half our runs
# don't even weight. This is the first time this project's training loop
# has run at this many batches -- Stephenson's 3 batches never surfaced
# this cost. GPU won't eliminate the O(many small ops) Python-loop
# overhead entirely, but the actual tensor math each op does benefits
# from it -- run_levy_sinkhorn_comparison.py was updated to move
# model+tensors to CUDA when available (it didn't originally).
#
# Same two proven fixes as every other GPU job in this project (gpu=1,
# operating_system=RedHat8 -- see geneformer_feasibility scripts for the
# full backstory) and the same torch-CUDA-build fix already validated for
# this env by submit_uger_levy_full_dataset.sh (cu121, NOT cu124 --
# cu124's wheel index only goes back to torch 2.4.0 -- and WITHOUT
# --no-deps, which silently skips installing the nvidia-cublas-cu12/
# nvidia-cudart-cu12 etc. packages the CUDA build actually dlopens at
# import time).
#
# Submit from the repo root: qsub scripts/submit_uger_levy_sinkhorn_comparison.sh
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
echo "Starting Levy sinkhorn-vs-mmd comparison: $(date)"

python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

python scripts/run_levy_sinkhorn_comparison.py

echo "Finished Levy sinkhorn-vs-mmd comparison: $(date)"
###############################################################################
