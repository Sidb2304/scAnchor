#!/bin/bash
###############################################################################
# Run this directly on the cluster login node (NOT via qsub -- this is a
# plain orchestration script, not a UGE job itself). It submits the
# Stephenson benchmark as two chained UGE jobs so the shared 7GB
# download/subsample step runs exactly once instead of racing across the
# two comparison tasks:
#
#   1. submit_uger_stephenson_prep.sh -- downloads + subsamples once
#   2. submit_uger_stephenson.sh (array, 2 tasks) -- scAnchor+Harmony and
#      scDisInFact, both held until (1) finishes via -hold_jid
#
# Usage:
#   bash scripts/submit_stephenson_pipeline.sh
###############################################################################

set -euo pipefail

echo "Submitting prep job (download + subsample)..."
PREP_JOBID=$(qsub -terse scripts/submit_uger_stephenson_prep.sh)
echo "  prep job ID: ${PREP_JOBID}"

echo "Submitting comparison array job (scAnchor+Harmony, scDisInFact), held until prep finishes..."
qsub -hold_jid "${PREP_JOBID}" scripts/submit_uger_stephenson.sh

echo "Done. Check progress with: qstat -u \$USER"
echo "Logs will land in logs/scanchor_stephenson_prep.log and logs/scanchor_stephenson.{1,2}.log"
###############################################################################
