#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Combined TRAIN + ANALYSE workflow for one (layer, rec/step) on a single GPU box.
#
# This is the one-shot counterpart to running run_full_workflow.sh and
# run_analysis_workflow.sh as two separate fleets. It:
#   1. Trains the per-seed SAEs for LAYER/REC          (run_full_workflow.sh, GPU)
#   2. Uploads the trained checkpoints to S3           (so they persist + the
#      analysis phase can stage them like the standalone analysis fleet does)
#   3. Runs the concept-F1 / consistency benchmark     (run_analysis_workflow.sh)
#
# Driven by the same env vars as its two halves (LAYER, REC, LAYER_TYPE, SEEDS_CSV,
# MAX_STEPS, ...). For diffusion, REC is the diffusion step; run_analysis_workflow.sh
# falls back RECS<-REC so it analyses exactly the step just trained.
#
# Deploy (one GPU instance per layer/step):
#   ./deploy_per_layer.ps1 -e <key> -LayerType diffusion \
#       -WorkflowScript run_train_and_analyse_workflow.sh \
#       -LayerNumbers 12 -RecNumbers 0,1,10 -SelfDestruct -UploadScripts
#
# Run locally (core_env activated, on a GPU box):
#   LAYER=12 REC=10 LAYER_TYPE=diffusion bash run_train_and_analyse_workflow.sh
# -----------------------------------------------------------------------------
set -euo pipefail

log() { printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }

LAYER="${LAYER:-10}"
REC="${REC:-1}"
LAYER_TYPE="${LAYER_TYPE:-pairformer}"
REGION="${AWS_REGION:-eu-west-2}"

# Phase toggles for re-runs / debugging.
SKIP_TRAIN_PHASE="${SKIP_TRAIN_PHASE:-0}"
SKIP_UPLOAD_PHASE="${SKIP_UPLOAD_PHASE:-0}"
SKIP_ANALYSE_PHASE="${SKIP_ANALYSE_PHASE:-0}"

# S3 destination for the trained SAEs. MUST match the convention used by
# server_setup.sh (training upload) and run_analysis_workflow.sh (SAE_S3_ROOT):
# diffusion lives under a 'diffusion/' prefix so it never overwrites pairformer.
if [[ "${LAYER_TYPE}" == "diffusion" ]]; then
    SAE_S3_DEST="s3://boltz-saes-l2/diffusion/rec${REC}/layer${LAYER}/"
else
    SAE_S3_DEST="s3://boltz-saes-l2/rec${REC}/layer${LAYER}/"
fi
SAE_RUNS_DIR="sae_runs_layer${LAYER}"

log "Combined train+analyse: LAYER_TYPE=${LAYER_TYPE} LAYER=${LAYER} REC=${REC}"
log "Trained SAEs -> ${SAE_S3_DEST}"

# -----------------------------------------------------------------------------
# Phase 1: train the per-seed SAEs (GPU). run_full_workflow.sh reads the same
# LAYER / REC / LAYER_TYPE env vars and writes checkpoints under sae_runs_layer<N>/.
# -----------------------------------------------------------------------------
if [[ "${SKIP_TRAIN_PHASE}" != "1" ]]; then
    log "==== Phase 1: TRAIN (run_full_workflow.sh) ===="
    bash run_full_workflow.sh
else
    log "==== Phase 1: TRAIN skipped (SKIP_TRAIN_PHASE=1) ===="
fi

# -----------------------------------------------------------------------------
# Phase 1b: persist the trained SAEs to S3 so (a) they survive self-destruct and
# (b) the analysis phase can stage them exactly like the standalone analysis fleet
# (download_run_checkpoint requires diffusion checkpoints to be S3-staged).
# -----------------------------------------------------------------------------
if [[ "${SKIP_UPLOAD_PHASE}" != "1" ]]; then
    if [[ -d "${SAE_RUNS_DIR}" ]]; then
        log "==== Phase 1b: upload ${SAE_RUNS_DIR} -> ${SAE_S3_DEST} ===="
        aws s3 sync "${SAE_RUNS_DIR}" "${SAE_S3_DEST}" --region "${REGION}"
    else
        echo "Trained-runs dir '${SAE_RUNS_DIR}' not found; nothing to upload (was training skipped?)." >&2
        exit 1
    fi
else
    log "==== Phase 1b: upload skipped (SKIP_UPLOAD_PHASE=1) ===="
fi

# -----------------------------------------------------------------------------
# Phase 2: analyse. run_analysis_workflow.sh stages the just-uploaded SAEs back
# from S3, runs the benchmark + consistency, and uploads results. For diffusion it
# defaults RECS<-REC, so it analyses exactly the step trained above.
#
# The concept sets analysed (including boltz_secondary -- DSSP H/E/C scored against
# Boltz's OWN predicted CIFs) are owned by run_analysis_workflow.sh; this combined
# workflow inherits them. Override via the same env vars it reads, e.g.
# CONCEPT_SETS=... or BUILD_BOLTZ_IF_MISSING=0, which pass straight through.
# -----------------------------------------------------------------------------
if [[ "${SKIP_ANALYSE_PHASE}" != "1" ]]; then
    log "==== Phase 2: ANALYSE (run_analysis_workflow.sh) ===="
    bash run_analysis_workflow.sh
else
    log "==== Phase 2: ANALYSE skipped (SKIP_ANALYSE_PHASE=1) ===="
fi

log "Combined train+analyse complete for LAYER_TYPE=${LAYER_TYPE} layer ${LAYER} rec ${REC}."
