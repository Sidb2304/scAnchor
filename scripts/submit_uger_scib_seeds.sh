#!/bin/bash
#$ -N scanchor_scib_seeds
#$ -o logs/scanchor_scib_seeds.$TASK_ID.log
#$ -j y
#$ -cwd
#$ -l h_vmem=32G
#$ -l h_rt=8:00:00
#$ -pe smp 4
#$ -binding linear:4
#$ -t 1-6
###############################################################################
# Seed-robustness check for the scIB benchmark result (README's Current
# results, v0.8.0): scAnchor beat Harmony on cell-type purity across all
# three atlas datasets at seed 0 -- this reruns seeds 1 and 2 for each
# dataset (seed 0 is already done, don't waste compute repeating it) to
# check that result isn't a single-seed accident, the same rigor already
# applied to the core MMD dose-response result. 6 array-job tasks (3
# datasets x 2 new seeds), same cluster/environment as submit_uger_scib.sh
# -- if that one already ran successfully, no new one-time setup needed.
#
# Writes to per-(dataset, seed) output dirs so nothing collides with the
# existing seed=0 results already in scib_benchmark_run/{dataset}/.
#
# Submit with:  qsub scripts/submit_uger_scib_seeds.sh
###############################################################################

set -euo pipefail

REPO_DIR="/stanley/nehme_lab/Siddharth/Projects/scAnchor"
CONDA_ENV_PATH="/stanley/nehme_lab/Siddharth/conda_envs/scanchor"
CHECKPOINT_DIR="${REPO_DIR}/checkpoints/scgpt_continual_pretrained"
DATA_CACHE_DIR="${REPO_DIR}/scib_data_cache"
OUT_DIR_BASE="${REPO_DIR}/scib_benchmark_run"

DATASETS=(pancreas pancreas lung lung immune immune)
SEEDS=(1 2 1 2 1 2)
IDX=$((SGE_TASK_ID - 1))
DATASET="${DATASETS[$IDX]}"
SEED="${SEEDS[$IDX]}"
OUT_DIR="${OUT_DIR_BASE}/${DATASET}_seed${SEED}"

# --- preflight check: fail now, loudly, not confusingly mid-job ---
for path_var in REPO_DIR CHECKPOINT_DIR CONDA_ENV_PATH; do
    path_value="${!path_var}"
    if [[ ! -e "$path_value" ]]; then
        echo "ERROR: $path_var points to a path that doesn't exist: $path_value" >&2
        echo "(if the /stanley/nehme_lab path convention is wrong for this cluster, fix the" >&2
        echo " variable block above; if it's just not set up yet, see the one-time setup" >&2
        echo " instructions in submit_uger_scib.sh's header comment)" >&2
        exit 1
    fi
done
mkdir -p "$DATA_CACHE_DIR" "$OUT_DIR" "$REPO_DIR/logs"
# --- end preflight check ---
# Note: this mkdir happens AFTER SGE has already tried to open the -o log
# path above, so it doesn't help THIS run if logs/ was missing at qsub
# time -- it's just defensive for next time. logs/ must already exist
# in $REPO_DIR before you run qsub.

# Same conda-activation approach as submit_uger_scib.sh (batch jobs don't
# inherit the interactive login shell's `use`/.bashrc setup that makes
# `conda` resolve on PATH) -- see that script's comments for why.
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
    echo "ERROR: couldn't find conda.sh -- see submit_uger_scib.sh's comments for how to fix." >&2
    exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV_PATH"

cd "$REPO_DIR"
echo "Starting scIB seed-check task ${SGE_TASK_ID} (${DATASET}, seed=${SEED}): $(date)"
python scripts/run_scib_benchmark.py \
    --dataset "$DATASET" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --out-dir "$OUT_DIR" \
    --data-cache-dir "$DATA_CACHE_DIR" \
    --seed "$SEED"
echo "Finished task ${SGE_TASK_ID} (${DATASET}, seed=${SEED}): $(date)"

# Dataset already cached from the earlier seed=0 run (DATA_CACHE_DIR is
# shared across all tasks/seeds), so this should skip straight to embedding
# extraction -- same per-dataset timing as before (~2.3-4.7hr on CPU),
# unaffected by which seed is used.
###############################################################################
