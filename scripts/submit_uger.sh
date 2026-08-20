#!/bin/bash
#$ -N scanchor_full_run
#$ -o logs/scanchor_full_run.log
#$ -j y
#$ -cwd
#$ -l h_vmem=32G
#$ -l h_rt=12:00:00
#$ -pe smp 4
#$ -binding linear:4
# -l gpu=1                    # <-- UNVERIFIED for this cluster, see note below
###############################################################################
# Run the full scAnchor pipeline (real scGPT embeddings, training, both
# evaluations) on the Levy schizophrenia iPSC astrocyte mini-village dataset.
#
# Paths below use the /stanley/nehme_lab/... convention seen in this lab's
# other UGER scripts (submit_neuro_gwas.sh, run_star_0801.sh, etc.), i.e.
# the cluster-side path for what's mounted at /Volumes/stanley_nehme_lab
# on macOS. If that mapping is wrong for some reason, the preflight check
# below will say exactly which path doesn't exist rather than failing
# confusingly mid-job.
#
# GPU: no existing script in this lab's history requests one, so the exact
# UGER GPU queue/flag for this cluster is unverified; check `qconf -sql`
# or your cluster admin, then uncomment and fix the `-l gpu=1` line above.
# The Python side auto-detects CUDA either way, so without a working GPU
# flag this just runs on CPU (slower, not wrong); see the timing estimate
# below for what that costs at full dataset scale.
#
# One-time setup (run on a cluster login node, NOT part of this job,
# since conda environments are platform-specific, this can't be prepared from a
# Mac and copied over):
#
#   conda create --prefix /stanley/nehme_lab/Siddharth/conda_envs/scanchor python=3.10 -y
#   source activate /stanley/nehme_lab/Siddharth/conda_envs/scanchor
#   cd /stanley/nehme_lab/Siddharth/Projects/scAnchor
#   pip install -e ".[scgpt]"
#   pip install gdown
#   gdown --folder "https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing" \
#       -O /stanley/nehme_lab/Siddharth/Projects/scAnchor/checkpoints/scgpt_continual_pretrained
#
# Submit with:  qsub scripts/submit_uger.sh
###############################################################################

set -euo pipefail

REPO_DIR="/stanley/nehme_lab/Siddharth/Projects/scAnchor"
CONDA_ENV_PATH="/stanley/nehme_lab/Siddharth/conda_envs/scanchor"
METADATA_TXT="/stanley/nehme_lab/dropseq/libraries/ngn2_astrocytes/Levy_astrocyte_mini_village_cell_metadata.txt"
COUNTS_H5AD="/stanley/nehme_lab/dropseq/libraries/ngn2_astrocytes/Levy_astrocyte_mini_village.h5ad"
CHECKPOINT_DIR="${REPO_DIR}/checkpoints/scgpt_continual_pretrained"
OUT_DIR="${REPO_DIR}/full_dataset_run"
PER_GROUP_N=0    # 0 = full ~81k-cell dataset, no subsampling

# --- preflight check: fail now, loudly, not confusingly mid-job ---
for path_var in REPO_DIR METADATA_TXT COUNTS_H5AD CHECKPOINT_DIR CONDA_ENV_PATH; do
    path_value="${!path_var}"
    if [[ ! -e "$path_value" ]]; then
        echo "ERROR: $path_var points to a path that doesn't exist: $path_value" >&2
        echo "(if the /stanley/nehme_lab path convention is wrong for this cluster, fix the" >&2
        echo " variable block above; if it's just not set up yet, see the one-time setup" >&2
        echo " instructions in this script's header comment)" >&2
        exit 1
    fi
done
mkdir -p "$OUT_DIR" "$REPO_DIR/logs"
# --- end preflight check ---
# Note: this mkdir happens AFTER SGE has already tried to open the -o log
# path above, so it doesn't help THIS run if logs/ was missing at qsub
# time; it's just defensive for next time. logs/ must already exist
# in $REPO_DIR before you run qsub.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_PATH"

cd "$REPO_DIR"
echo "Starting scAnchor full-dataset run: $(date)"
python scripts/run_full_dataset.py \
    --metadata-txt "$METADATA_TXT" \
    --counts-h5ad "$COUNTS_H5AD" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --out-dir "$OUT_DIR" \
    --per-group-n "$PER_GROUP_N"
echo "Finished: $(date)"

# Rough timing estimate from local (Mac CPU) testing: ~0.15-0.2s/cell for
# scGPT embedding extraction alone. At ~81k cells that's roughly 3-4.5 hours
# on CPU before training/eval even start; h_rt=12:00:00 above gives
# generous headroom, adjust if your cluster's CPUs are meaningfully slower.
# With a working GPU this should be dramatically faster.
