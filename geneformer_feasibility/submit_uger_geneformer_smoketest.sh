#!/bin/bash
#$ -N geneformer_smoketest
#$ -o geneformer_feasibility/geneformer_smoketest.log
#$ -j y
#$ -cwd
#$ -l h_vmem=16G
#$ -l h_rt=2:00:00
#$ -l gpu=1
#$ -l operating_system=RedHat8
# No -pe smp / -binding here: this cluster's GPU hosts provide only 1 slot
# each (confirmed via `qconf -sq interactive`'s `slots ...,[@gpuhosts=1]`),
# so any multi-slot `smp` request is permanently unsatisfiable on a GPU
# host. That was a real, necessary fix, but NOT the actual reason this job
# kept sitting in qw afterward. Turns out the login node was also
# injecting an `operating_system=RedHat7` default into every job, while
# the GPU hosts run RedHat8 (confirmed from another user's running GPU
# job's resource list). `-l operating_system=RedHat8` above overrides that
# mismatch, the real root cause of the whole multi-day qw saga.
###############################################################################
# Feasibility smoke test ONLY: does Geneformer's V1-10M checkpoint install
# and run in reasonable time on a tiny 200-cell subset of the already-cached
# Stephenson data? Not a scAnchor integration yet.
#
# Requests an actual GPU node (`-l gpu=1`, matching the `hf:gpu=1.000000`
# host resource confirmed via `qhost -F gpu`, i.e. uger-gpu-d001/d002/d003).
# A prior CPU-only attempt got all the way through install and tokenization
# but crashed in emb_extractor.py's get_embs(), which hardcodes
# `device="cuda"` with no CPU fallback, so this needs a real GPU, not a
# code patch. If this `-l gpu=1` request errors out (wrong resource name
# for this cluster's config), check `qconf -sc | grep -i gpu` for the exact
# complex resource name instead of guessing further.
#
# Builds a NEW, ISOLATED conda env (geneformer_smoketest_env), separate from
# the working `scanchor` env used by every other job in this project, since
# Geneformer's setup.py requires Python >=3.10 and a heavier, training-
# oriented dependency stack (bitsandbytes, ray, optuna, peft, tensorboard)
# than anything else installed so far, and this should not risk breaking
# the working env if something conflicts.
#
# Submit directly: qsub geneformer_feasibility/submit_uger_geneformer_smoketest.sh
###############################################################################

set -euo pipefail

REPO_DIR="/stanley/nehme_lab/Siddharth/Projects/scAnchor"
FEASIBILITY_DIR="${REPO_DIR}/geneformer_feasibility"
ENV_NAME="geneformer_smoketest_env"

if [[ ! -e "$FEASIBILITY_DIR" ]]; then
    echo "ERROR: FEASIBILITY_DIR doesn't exist: $FEASIBILITY_DIR" >&2
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
    echo "ERROR: couldn't find conda.sh; see submit_uger_scib.sh's comments." >&2
    exit 1
fi
source "$CONDA_SH"

if ! conda env list | grep -q "$ENV_NAME"; then
    echo "creating isolated env $ENV_NAME (python 3.10, separate from scanchor env)..."
    conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

cd "$FEASIBILITY_DIR"
echo "Starting Geneformer smoke test: $(date)"

# This cluster's GCC (4.8.5) defaults to a pre-C99/pre-C++11 standard, but
# several packages further down (pillow, greenlet, accumulation-tree/tdigest)
# ship generated C/C++ source that assumes newer standard support (C99 `for`
# loop variable declarations, C++11 `noexcept`) and have no prebuilt wheel
# for this platform+Python 3.10 combo. Forcing the standard directly fixes
# all of them at once rather than hunting a version pin per package.
export CFLAGS="-std=gnu99"
export CXXFLAGS="-std=c++11"

# CUDA 12.4 build specifically, not a plain unpinned `pip install torch`:
# this GPU node's driver reported "found version 12090" (CUDA 12.9) as its
# max supported CUDA version, and the latest PyPI torch release now needs
# something newer than that: a real, fixable driver/toolkit mismatch,
# not the earlier CPU-only-build issue. cu124 is safely backward-
# compatible with a 12.9-capable driver.
# --force-reinstall --no-deps: without this, pip sees "torch requirement
# already satisfied" from whatever's already installed in this reused env
# and silently skips reinstalling even with a different --index-url: a
# real, confirmed recurrence of the same "already satisfied" trap that
# caused the earlier CPU-vs-CUDA build mismatch. --no-deps avoids also
# needlessly reinstalling every already-correct dependency.
pip install --quiet --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cu124
# pandas<2.3 / pyarrow<18: this cluster's RHEL7 GCC (4.8.5) can't compile
# newer pandas/pyarrow's meson+cython source builds when no manylinux2014
# wheel is available; the exact same failure and fix already established
# for the main scanchor env earlier in this project (see other UGER
# scripts' comments). Geneformer's own requirements.txt leaves both
# unpinned (pandas>=2.0, pyarrow>=12.0), so pin them down explicitly here
# rather than letting pip grab whatever's newest.
# h5py==3.14.0: same class of fix as pandas/pyarrow above: no prebuilt
# wheel for this platform+Python combo without pinning, and there's no
# system libhdf5 to link a source build against here. Exact version
# already validated to have a working wheel for the main scanchor env.
pip install --quiet huggingface_hub anndata scipy "pandas<2.3" "pyarrow<18" "h5py==3.14.0"
# greenlet<3 (a ray dependency, pulled in transitively): newer greenlet's
# C++ source uses designated initializers GCC 4.8.5 can't compile at any
# -std setting (a real compiler-version wall, not a flag/standard issue:
# "sorry, unimplemented" from g++ itself). Pin to a pre-3.0 release that
# predates this syntax and still has a prebuilt wheel for this old glibc.
#
# Geneformer's own setup.py dependency list, installed directly rather
# than via `pip install <cloned repo>` since we're selectively downloading
# only the files we need (see run_smoketest.py), not a full repo clone.
pip install --quiet \
    "transformers==4.46" bitsandbytes datasets loompy matplotlib numpy \
    optuna optuna-integration packaging peft "pyarrow<18" pytz ray scanpy \
    scikit-learn scipy seaborn setuptools statsmodels tdigest tensorboard tqdm \
    "greenlet<3"

python run_smoketest.py

echo "Finished Geneformer smoke test: $(date)"
###############################################################################
