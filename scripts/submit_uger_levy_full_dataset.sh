#!/bin/bash
#$ -N scanchor_levy_full
#$ -o levy_run/levy_full_dataset.log
#$ -j y
#$ -cwd
#$ -l h_vmem=32G
#$ -l h_rt=6:00:00
#$ -l gpu=1
#$ -l operating_system=RedHat8
###############################################################################
# Real-scale Levy replicate-structure test, for the Sinkhorn-OT validation
# (v1.3.0's sinkhorn_weight) -- runs scAnchor's own scripts/run_full_dataset.py
# against the ACTUAL private Levy astrocyte mini-village files (found on the
# lab drive, previously unlocated -- see PROJECT_NOTES.md/conversation
# history), full ~81k-cell panel, no subsampling (--per-group-n 0).
#
# -l gpu=1 / -l operating_system=RedHat8: same two real, previously-diagnosed
# fixes as every other GPU job in this project (see
# geneformer_feasibility/submit_uger_geneformer_smoketest.sh's comments for
# the full story) -- without RedHat8 explicitly, the login node injects
# RedHat7 into the job spec and it sits in `qw` indefinitely with no error,
# regardless of actual GPU availability.
#
# Reuses the existing `scanchor` conda env (already has scgpt 0.2.4 working
# on CPU torch from earlier local-Mac-CPU test runs) rather than building a
# new one -- just swaps its torch build from +cpu to a CUDA build, matching
# the exact torch version already validated to work with this scgpt
# version, so this is a much smaller change than Geneformer's dependency
# chain needed.
#
# This job writes reference.h5ad / heldout.h5ad (with real X_scGPT
# embeddings) to levy_run/ BEFORE training -- those are what actually get
# read back locally afterward to run the real Sinkhorn-vs-MMD comparison
# (fast, CPU, no GPU needed for that part, same pattern as every other
# comparison already done in this project). The training/eval this script
# also runs (mmd_weight=20, the hardcoded default in run_full_dataset.py)
# is a useful bonus -- a fresh, real MMD/Levy baseline number on the actual
# files, not just the historical README number of unverified provenance.
#
# Submit from the repo root: qsub scripts/submit_uger_levy_full_dataset.sh
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
echo "Starting Levy full-dataset run: $(date)"

# cu121, not cu124: a first attempt at this found PyTorch's cu124 wheel
# index only goes back to torch 2.4.0 -- no 2.3.1+cu124 build exists.
# cu121's index does have 2.3.1, and cu121 is still safely within this
# cluster's driver's max-supported CUDA (12.9), so it's compatible.
# --force-reinstall, WITHOUT --no-deps: a first attempt used --no-deps
# (reasoning: same torch version, just a different build, so no other
# dependency should need to change) -- that's wrong. The CPU build has no
# CUDA runtime dependencies at all; the CUDA build depends on separate
# nvidia-cublas-cu12/nvidia-cudart-cu12/etc. pip packages that actually
# provide the .so files torch's CUDA build dlopens at import time.
# --no-deps skipped installing those, so `import torch` failed with
# "libcudart.so.12: cannot open shared object file" / "libcublas.so not
# found" -- the exact same class of failure already diagnosed once before
# in this project's Geneformer work (see its scripts' comments), now
# recurring for the same root-cause reason on a different package.
pip install --quiet --force-reinstall torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

python scripts/run_full_dataset.py \
    --metadata-txt /stanley/nehme_lab/dropseq/libraries/ngn2_astrocytes/Levy_astrocyte_mini_village_cell_metadata.txt \
    --counts-h5ad /stanley/nehme_lab/dropseq/libraries/ngn2_astrocytes/Levy_astrocyte_mini_village.h5ad \
    --checkpoint-dir "${REPO_DIR}/checkpoints/scgpt_continual_pretrained" \
    --out-dir "${REPO_DIR}/levy_run" \
    --per-group-n 0

echo "Finished Levy full-dataset run: $(date)"
###############################################################################
