#!/bin/bash
#$ -N geneformer_stephenson_embed
#$ -o geneformer_feasibility/geneformer_stephenson_embed.log
#$ -j y
#$ -cwd
#$ -l h_vmem=32G
#$ -l h_rt=4:00:00
#$ -l gpu=1
#$ -l operating_system=RedHat8
###############################################################################
# Real-scale Geneformer embedding extraction (21,000 cells), same env
# setup as submit_uger_geneformer_smoketest.sh (which already resolved
# every dependency/GPU/OS issue for this exact package on this exact
# cluster), pointed at the real embedding script instead of the smoke test.
#
# Reuses geneformer_smoketest_env if it still exists, otherwise rebuilds
# it fresh (it was deleted once already to free disk quota; see
# feedback_use_gpu_nodes.md memory note for the full list of fixes below).
#
# Submit from the repo root:
#   qsub geneformer_feasibility/submit_uger_geneformer_stephenson_embed.sh
###############################################################################

set -euo pipefail

REPO_DIR="/stanley/nehme_lab/Siddharth/Projects/scAnchor"
FEASIBILITY_DIR="${REPO_DIR}/geneformer_feasibility"
# Explicit path on the lab's shared storage, NOT `-n` (which defaults to
# ~/.conda/envs/, subject to a hard 20GB home-directory quota that this
# exact job hit three times in a row, even after cleanup). The working
# `scanchor` env already lives this way for the same reason; should
# have matched that pattern from the start instead of using -n.
ENV_PATH="/stanley/nehme_lab/Siddharth/conda_envs/geneformer_stephenson_env"

if [[ ! -e "${FEASIBILITY_DIR}/stephenson_subsample_geneformer.h5ad" ]]; then
    echo "ERROR: run build_geneformer_stephenson_subsample.py first." >&2
    exit 1
fi

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
    echo "ERROR: couldn't find conda.sh." >&2
    exit 1
fi
source "$CONDA_SH"

if [[ ! -d "$ENV_PATH" ]]; then
    echo "creating isolated env at $ENV_PATH (python 3.10)..."
    conda create -y -p "$ENV_PATH" python=3.10
fi
conda activate "$ENV_PATH"

cd "$FEASIBILITY_DIR"
echo "Starting Geneformer Stephenson embedding: $(date)"

export CFLAGS="-std=gnu99"
export CXXFLAGS="-std=c++11"

# Latest torch via the cu124 index, NOT pinned to 2.3.1/cu121: that
# pin was copied by mistake from the sciplex script, which needed it for
# scgpt's torchtext ABI compatibility. Geneformer doesn't use torchtext
# at all, and its own deps (bitsandbytes, peft) need a MODERN torch
# (torch.library.register_fake only exists in torch>=2.4), confirmed
# directly via "AttributeError: module 'torch.library' has no attribute
# 'register_fake'" when 2.3.1 was pinned.
#
# No --no-deps: this env has residual nvidia-cu12-* libraries pinned to
# versions matching the OLD torch==2.3.1/cu121 install from earlier
# attempts, so forcing just the top-level torch package back to
# latest/cu124 without letting its CUDA library deps update too would
# risk the exact same kind of library-version mismatch just fixed above,
# just in the other direction. Let pip reinstall the whole matched set.
pip install --quiet --no-cache-dir --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
pip install --quiet --no-cache-dir huggingface_hub anndata scipy "pandas<2.3" "pyarrow<18" "h5py==3.14.0"
pip install --quiet --no-cache-dir \
    "transformers==4.46" bitsandbytes datasets loompy matplotlib numpy \
    optuna optuna-integration packaging peft "pyarrow<18" pytz ray scanpy \
    scikit-learn scipy seaborn setuptools statsmodels tdigest tensorboard tqdm \
    "greenlet<3" ipython

python run_geneformer_stephenson_embed.py

echo "Finished Geneformer Stephenson embedding: $(date)"
###############################################################################
