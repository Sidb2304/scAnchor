#!/bin/bash
# SGE/UGER submission script for scripts/run_full_dataset.py.
#
# Adjust the paths and resource requests below for your cluster before
# submitting with: qsub scripts/submit_uger.sh
#
# The GPU request line is commented out and UNVERIFIED for this cluster --
# exact GPU queue/resource syntax varies a lot between UGER installations.
# Check `qconf -sql` (list queues) or your cluster's docs/admin for the
# correct flag, then uncomment and fix the line below. The Python side
# auto-detects CUDA either way, so worst case without a working GPU flag
# this just runs on CPU -- slower, not wrong.

#$ -N scanchor_full_run
#$ -o scanchor_full_run.out
#$ -e scanchor_full_run.err
#$ -l h_vmem=16G
#$ -l h_rt=12:00:00
#$ -pe smp 4
#$ -binding linear:4
# -l gpu=1                    # <-- verify this flag/queue for your cluster

set -euo pipefail

# --- edit these for your environment ---
REPO_DIR="/path/to/scAnchor"                       # this repo, cluster-side path
CONDA_ENV="scanchor"                                 # created ahead of time, see below
METADATA_TXT="/path/to/Levy_astrocyte_mini_village_cell_metadata.txt"
COUNTS_H5AD="/path/to/Levy_astrocyte_mini_village.h5ad"
CHECKPOINT_DIR="/path/to/scgpt_continual_pretrained_checkpoint"
OUT_DIR="/path/to/scratch/scanchor_full_run"
PER_GROUP_N=0                                        # 0 = full dataset, no subsampling
# ----------------------------------------

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$REPO_DIR"
python scripts/run_full_dataset.py \
    --metadata-txt "$METADATA_TXT" \
    --counts-h5ad "$COUNTS_H5AD" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --out-dir "$OUT_DIR" \
    --per-group-n "$PER_GROUP_N"

# One-time environment setup (run once, not part of the job):
#   conda create -n scanchor python=3.10 -y
#   conda activate scanchor
#   cd /path/to/scAnchor
#   pip install -e ".[scgpt]"
#   # scGPT checkpoint (continual-pretrained, see README "Current results"):
#   pip install gdown
#   gdown --folder "https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing" \
#       -O /path/to/scgpt_continual_pretrained_checkpoint
