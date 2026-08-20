#!/bin/bash
#$ -N scanchor_stephenson
#$ -o logs/scanchor_stephenson.$TASK_ID.log
#$ -j y
#$ -cwd
#$ -l h_vmem=32G
#$ -l h_rt=8:00:00
#$ -pe smp 4
#$ -binding linear:4
#$ -t 1-2
###############################################################################
# Real three-way comparison on Stephenson et al. 2021's COVID-19 PBMC atlas
# (Site=batch, Status=condition, donor_id, but donor is fully confounded
# with Site here, see run_stephenson_benchmark.py's docstring): task 1 runs
# scAnchor (current default, mmd_weight=20) + a Harmony baseline via scGPT
# embeddings; task 2 runs scDisInFact directly on raw counts (no scGPT
# needed). Both read the same cached subsample, built once by
# submit_uger_stephenson_prep.sh, so they compare on identical cells.
#
# Do NOT qsub this directly; submit via scripts/submit_stephenson_pipeline.sh,
# which chains this after the prep job with -hold_jid so both tasks here
# don't race the prep job's download/subsample step.
#
# One-time setup beyond the existing scanchor env (submit_uger_scib.sh's
# setup); scDisInFact is not on PyPI, needs a source install:
#
#   source activate /stanley/nehme_lab/Siddharth/conda_envs/scanchor
#   cd /stanley/nehme_lab/Siddharth/Projects/scAnchor
#   git clone https://github.com/ZhangLabGT/scDisInFact.git
#   cd scDisInFact && pip install . && cd ..
#
# (this is a NEW dependency, not yet tested on this cluster, so expect it
# may need the same kind of version-pinning treatment already worked
# through for pandas/pyarrow/Pillow/etc. in configs/default.yaml's history,
# if its own dependencies hit the same old-glibc wheel-availability wall.)
###############################################################################

set -euo pipefail

REPO_DIR="/stanley/nehme_lab/Siddharth/Projects/scAnchor"
CONDA_ENV_PATH="/stanley/nehme_lab/Siddharth/conda_envs/scanchor"
CHECKPOINT_DIR="${REPO_DIR}/checkpoints/scgpt_continual_pretrained"
DATA_CACHE_DIR="${REPO_DIR}/stephenson_data_cache"
OUT_DIR="${REPO_DIR}/stephenson_run"

for path_var in REPO_DIR CHECKPOINT_DIR CONDA_ENV_PATH DATA_CACHE_DIR; do
    path_value="${!path_var}"
    if [[ ! -e "$path_value" ]]; then
        echo "ERROR: $path_var points to a path that doesn't exist: $path_value" >&2
        echo "(if this is DATA_CACHE_DIR, the prep job hasn't finished/run yet)" >&2
        exit 1
    fi
done
mkdir -p "$OUT_DIR" "$REPO_DIR/logs"

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
    echo "ERROR: couldn't find conda.sh; see submit_uger_scib.sh's comments." >&2
    exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV_PATH"

cd "$REPO_DIR"
if [[ "$SGE_TASK_ID" == "1" ]]; then
    echo "Starting Stephenson task 1 (scAnchor + Harmony): $(date)"
    python scripts/run_stephenson_benchmark.py \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --out-dir "$OUT_DIR" \
        --data-cache-dir "$DATA_CACHE_DIR"
    echo "Finished Stephenson task 1: $(date)"
else
    echo "Starting Stephenson task 2 (scDisInFact): $(date)"
    python scripts/run_scdisinfact_stephenson.py \
        --out-dir "${OUT_DIR}_scdisinfact" \
        --data-cache-dir "$DATA_CACHE_DIR"
    echo "Finished Stephenson task 2: $(date)"
fi
###############################################################################
