#!/bin/bash
#$ -N scanchor_scib_benchmark
#$ -o logs/scanchor_scib_benchmark.$TASK_ID.log
#$ -j y
#$ -cwd
#$ -l h_vmem=32G
#$ -l h_rt=8:00:00
#$ -pe smp 4
#$ -binding linear:4
#$ -t 1-3
# -l gpu=1                    # <-- UNVERIFIED for this cluster, see note below
###############################################################################
# Run scAnchor + a Harmony baseline on all three scIB atlas-level integration
# benchmark datasets (pancreas, lung, immune), the standard reference point
# every batch-correction method gets compared against, not yet run in this
# project (see README's Reference panel section). One array-job task per
# dataset (SGE_TASK_ID 1/2/3 -> pancreas/lung/immune below), so all three run
# in parallel instead of ~3x sequential wall-clock.
#
# Paths below use the /stanley/nehme_lab/... convention seen in this lab's
# other UGER scripts and this repo's existing submit_uger.sh. If that mapping
# is wrong for this cluster, the preflight check below says exactly which
# path doesn't exist rather than failing confusingly mid-job.
#
# Datasets are downloaded fresh from Figshare on first run (cached in
# DATA_CACHE_DIR after), which requires the compute nodes to have internet
# egress. Some clusters only allow that from login nodes; if a task fails
# immediately with a download/connection error, that's almost certainly why,
# so pre-download the 3 files on a login node into DATA_CACHE_DIR instead
# (exact URLs are in scripts/run_scib_benchmark.py's DATASETS dict) and
# re-submit.
#
# GPU: same caveat as submit_uger.sh, since no existing script in this lab's
# history requests one, so the exact UGER GPU queue/flag for this cluster is
# unverified. The Python side auto-detects CUDA either way, so without a
# working GPU flag this just runs on CPU (slower, not wrong); see the
# timing estimate at the bottom for what that costs.
#
# One-time setup (same conda env as submit_uger.sh, reuse it if you
# already ran that script, this doesn't need anything extra installed
# beyond the "baselines" extra for Harmony):
#
#   conda create --prefix /stanley/nehme_lab/Siddharth/conda_envs/scanchor python=3.10 -y
#   source activate /stanley/nehme_lab/Siddharth/conda_envs/scanchor
#   cd /stanley/nehme_lab/Siddharth/Projects/scAnchor
#   pip install -e ".[scgpt,baselines]"
#   pip install gdown
#   gdown --folder "https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB?usp=sharing" \
#       -O /stanley/nehme_lab/Siddharth/Projects/scAnchor/checkpoints/scgpt_continual_pretrained
#
# Submit with:  qsub scripts/submit_uger_scib.sh
###############################################################################

set -euo pipefail

REPO_DIR="/stanley/nehme_lab/Siddharth/Projects/scAnchor"
CONDA_ENV_PATH="/stanley/nehme_lab/Siddharth/conda_envs/scanchor"
CHECKPOINT_DIR="${REPO_DIR}/checkpoints/scgpt_continual_pretrained"
DATA_CACHE_DIR="${REPO_DIR}/scib_data_cache"
OUT_DIR_BASE="${REPO_DIR}/scib_benchmark_run"

DATASETS=(pancreas lung immune)
DATASET="${DATASETS[$((SGE_TASK_ID - 1))]}"
OUT_DIR="${OUT_DIR_BASE}/${DATASET}"

# --- preflight check: fail now, loudly, not confusingly mid-job ---
for path_var in REPO_DIR CHECKPOINT_DIR CONDA_ENV_PATH; do
    path_value="${!path_var}"
    if [[ ! -e "$path_value" ]]; then
        echo "ERROR: $path_var points to a path that doesn't exist: $path_value" >&2
        echo "(if the /stanley/nehme_lab path convention is wrong for this cluster, fix the" >&2
        echo " variable block above; if it's just not set up yet, see the one-time setup" >&2
        echo " instructions in this script's header comment)" >&2
        exit 1
    fi
done
mkdir -p "$DATA_CACHE_DIR" "$OUT_DIR" "$REPO_DIR/logs"
# --- end preflight check ---
# Note: this mkdir happens AFTER SGE has already tried to open the -o log
# path above, so it doesn't help THIS run if logs/ was missing at qsub
# time; it's just defensive for next time. logs/ must already exist
# in $REPO_DIR before you run qsub.

# Batch jobs don't inherit the interactive login shell's setup (the
# `use`/`.bashrc` machinery that makes `conda` resolve on PATH when you're
# typed in manually), since `conda info --base` fails silently here because
# `conda` itself isn't found yet, a real bug hit running this for real.
# Known-good path for this cluster's base anaconda install (confirmed from
# an interactive session, see conda_base candidates below); falls back to
# `conda info --base` in case `conda` IS somehow already on PATH, and fails
# loudly with both attempts shown rather than a confusing "command not
# found" three lines later if neither works.
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
    echo "ERROR: couldn't find conda.sh; tried the hardcoded cluster path and" >&2
    echo "'conda info --base'. Run 'which conda' and 'conda info --base' in an" >&2
    echo "interactive session on this cluster, then hardcode the correct" >&2
    echo "etc/profile.d/conda.sh path into the conda_base list above." >&2
    exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV_PATH"

cd "$REPO_DIR"
echo "Starting scIB benchmark task ${SGE_TASK_ID} (${DATASET}): $(date)"
python scripts/run_scib_benchmark.py \
    --dataset "$DATASET" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --out-dir "$OUT_DIR" \
    --data-cache-dir "$DATA_CACHE_DIR"
echo "Finished task ${SGE_TASK_ID} (${DATASET}): $(date)"

# Rough timing estimate from local (Mac CPU) testing on a similar-scale
# dataset (Jerber day-30, ~9.5k cells): ~118 cells/min for scGPT embedding
# extraction alone. Approximate cell counts for these three, from the scIB
# paper (NOT verified by actually loading these files; exact counts print
# at runtime above): pancreas ~16.4k cells (~2.3hr), lung ~32.5k cells
# (~4.6hr), immune ~33.5k cells (~4.7hr). h_rt=8:00:00 above gives headroom
# for the slowest of the three; adjust if your cluster's CPUs are
# meaningfully slower. With a working GPU this should be dramatically
# faster than any of these CPU estimates.
###############################################################################
