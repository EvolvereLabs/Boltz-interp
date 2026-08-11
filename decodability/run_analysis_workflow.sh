#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# End-to-end SAE *analysis* workflow (the counterpart to run_full_workflow.sh).
#
# run_full_workflow.sh TRAINS SAEs and uploads checkpoints to
#   s3://boltz-saes-l2/rec<REC>/layer<N>/
# This script does the opposite: it pulls those already-trained SAEs (all seeds)
# and the eval activations down from S3, runs the concept-F1 / cross-seed-
# benchmark on CPU, and uploads the analysis results back to S3.
#
# Driven by LAYER (one EC2 instance per layer via deploy_per_layer.ps1); each
# instance loops over RECS x SEEDS x CONCEPT_SETS internally. No GPU required --
# the SAE encode runs on CPU and the null/probes are the (CPU-bound) bulk of the work.
#
# Run on the analysis EC2 image (CPU box, core_env activated):
#   conda activate /home/ec2-user/core_env
#   LAYER=12 bash run_analysis_workflow.sh
# -----------------------------------------------------------------------------
set -euo pipefail

log() { printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }

# Robustly test whether an S3 prefix contains anything. Returns:
#   0 -> ls succeeded and the prefix is NON-empty (has objects)
#   1 -> ls succeeded and the prefix is EMPTY (genuinely untrained -> caller may skip)
#   2 -> ls kept ERRORING after retries (do NOT treat as empty -> caller should abort)
# Why this exists: a bare `aws s3 ls ... 2>/dev/null | grep -q .` makes a transient
# error look identical to "nothing here". At boot a fresh instance often hits
# AccessDenied while its instance-profile role is still propagating, and a whole
# fleet listing the same bucket at once can get S3 SlowDown throttling. Both were
# being silently swallowed and reported as "no trained SAE -- exiting cleanly",
# which is why a solo deploy worked but a fleet of instances all bailed.
s3_prefix_has_objects() {
    local uri="$1" out rc attempt
    for attempt in 1 2 3 4 5; do
        out="$(aws s3 ls "$uri" --region "${REGION}" 2>&1)"; rc=$?
        if [[ $rc -eq 0 ]]; then
            [[ -n "$out" ]] && return 0 || return 1
        fi
        log "  s3 ls '$uri' errored (attempt ${attempt}/5, rc=${rc}): ${out}"
        sleep $(( attempt * 5 ))
    done
    log "  ERROR: s3 ls '$uri' still failing after 5 attempts (last: ${out})."
    return 2
}

# -----------------------------------------------------------------------------
# 0. Configuration (override any of these with env vars before invoking).
# -----------------------------------------------------------------------------
LAYER="${LAYER:-12}"
# Boltz module the SAEs were trained on: "pairformer" (default) or "diffusion".
# Must match what run_full_workflow.sh trained: selects the activation subfolder,
# the SAE run-name tag, the S3 checkpoint/result namespaces, and the local
# sae_explore/[<type>_]layer_sweep_rec<R>/ output tree.
LAYER_TYPE="${LAYER_TYPE:-pairformer}"
LAYER_TYPE="$(printf '%s' "${LAYER_TYPE}" | tr '[:upper:]' '[:lower:]')"
case "${LAYER_TYPE}" in
    pairformer|diffusion) ;;
    *) echo "ERROR: LAYER_TYPE must be 'pairformer' or 'diffusion', got '${LAYER_TYPE}'" >&2; exit 1 ;;
esac
# Local sweep-dir prefix; MUST match run_layer_benchmark._sweep_root_for():
# "" for pairformer, "diffusion_" otherwise.
if [[ "${LAYER_TYPE}" == "pairformer" ]]; then SWEEP_PREFIX=""; else SWEEP_PREFIX="${LAYER_TYPE}_"; fi
# Recs/diffusion-steps to analyse. The analysis workflow loops over RECS internally
# (one instance per LAYER). Pairformer keeps its historical 0,1 default. Diffusion has
# no such default, so it falls back to the single REC the deploy chain set for THIS
# instance -- i.e. deploy_per_layer.ps1 -RecNumbers 0,1,10 fans out 3 instances that
# each analyse one step. Set RECS directly to override either way.
if [[ "${LAYER_TYPE}" == "diffusion" ]]; then
    RECS="${RECS:-${REC:-0}}"
else
    RECS="${RECS:-0,1}"
fi
SEEDS_CSV="${SEEDS_CSV:-1,2,3}"     # SAE seeds to benchmark
# Concept sets to benchmark. boltz_secondary = DSSP H/E/C from Boltz's OWN predicted CIF
# (the principled structural target: labels come from the same forward pass as the
# activations). It is layer-type independent, so it applies to pairformer and diffusion alike.
CONCEPT_SETS="${CONCEPT_SETS:-secondary,swissprot,boltz_secondary}"
N_PERM="${N_PERM:-100}"            # permutation-null count (p-value floor ~ 1/(N_PERM+1))
DEVICE="${DEVICE:-cpu}"

# Benchmark parallelism. On c7i.4xlarge (16 vCPU) keep MAX_WORKERS*THREADS_PER_WORKER ~= 12,
# leaving RAM/cores headroom (each worker holds the full latent matrix, ~5 GB).
MAX_WORKERS="${MAX_WORKERS:-6}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-2}"

# The fixed common-across-layers/recs protein set is BOTH the download list and the
# benchmark's protein filter, so n_proteins is identical everywhere and comparable.
PROTEIN_IDS_FILE="${PROTEIN_IDS_FILE:-common_activation_proteins.txt}"

# Eval activations live in the SwissProt-annotated bucket under this prefix.
EVAL_BUCKET="${EVAL_BUCKET:-swissprot-annotated-proteins-activations}"
EVAL_KEY_PREFIX="${EVAL_KEY_PREFIX:-SwissProtAnnotation5/activations/}"
EVAL_MANIFEST="${EVAL_MANIFEST:-common_activation_proteins.txt}"
EVAL_ACTIVATIONS_DIR="${EVAL_ACTIVATIONS_DIR:-./downloads_layer${LAYER}}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-32}"

# S3 locations. Diffusion SAEs + results live under a 'diffusion/' prefix (set by
# run_full_workflow.sh on upload) so they never collide with pairformer. Annotations
# are layer-type independent (same proteins/concepts), so they stay shared.
if [[ "${LAYER_TYPE}" == "diffusion" ]]; then
    SAE_S3_ROOT="${SAE_S3_ROOT:-s3://boltz-saes-l2/diffusion}"    # trained diffusion SAE checkpoints
    RESULTS_S3="${RESULTS_S3:-s3://boltz-saes-l2/diffusion/analysis}"
else
    SAE_S3_ROOT="${SAE_S3_ROOT:-s3://boltz-saes-l2}"             # trained SAE checkpoints (per rec/layer)
    RESULTS_S3="${RESULTS_S3:-s3://boltz-saes-l2/analysis}"      # where this script writes results
fi
ANNOT_S3="${ANNOT_S3:-s3://boltz-saes-l2/annotations}"       # processed annotation dirs (uploaded once)
REGION="${AWS_REGION:-eu-west-2}"

# Boltz-own-structure secondary set. Built once from the predicted CIFs (the model's own
# output, fetched from the eval bucket) and cached in S3 under ANNOT_S3 like the others.
# BUILD_BOLTZ_IF_MISSING=1 lets the first analysis run build+upload it on demand; set 0 to
# require a pre-built copy in S3. If it ends up unavailable, boltz_secondary is dropped from
# CONCEPT_SETS (with a warning) so the rest of the analysis still runs.
BOLTZ_SECONDARY_DIR="${BOLTZ_SECONDARY_DIR:-processed_swissprot_a5_boltz_secondary}"
BUILD_BOLTZ_IF_MISSING="${BUILD_BOLTZ_IF_MISSING:-1}"

# Step toggles (1 = skip) for debugging / re-runs.
SKIP_VERIFY="${SKIP_VERIFY:-0}"
SKIP_ANNOT="${SKIP_ANNOT:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_STAGE_SAE="${SKIP_STAGE_SAE:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"

IFS=',' read -r -a REC_ARR <<< "${RECS}"
IFS=',' read -r -a SEED_ARR <<< "${SEEDS_CSV}"

# Run name convention must match layer_analysis_utils.run_name(): the benchmark resolves
# checkpoints from sae_explore/layer_sweep_rec<REC>/hf_cache/layer<N>/<run_name>/.
FINAL_STEP="${FINAL_STEP:-500000}"
# Tagged by layer type to match layer_analysis_utils.run_name() and the training upload.
run_name() { printf '%s%s_topk256_lat2048_demean_longtrain%s_l2_3e-3_seed%s' "$LAYER_TYPE" "$LAYER" "$FINAL_STEP" "$1"; }

log "Layer type:       ${LAYER_TYPE}"
log "Layer:            ${LAYER}"
log "Recs:             ${RECS}    Seeds: ${SEEDS_CSV}    Concept sets: ${CONCEPT_SETS}"
log "Eval activations: ${EVAL_ACTIVATIONS_DIR} (bucket ${EVAL_BUCKET})"
log "SAE checkpoints:  ${SAE_S3_ROOT}/rec<REC>/layer${LAYER}/"
log "Results -> S3:    ${RESULTS_S3}"
log "Device:           ${DEVICE}    Python: $(command -v python)"

# -----------------------------------------------------------------------------
# 1. Verify the env (CPU torch is fine; do NOT require CUDA).
# -----------------------------------------------------------------------------
if [[ "${SKIP_VERIFY}" != "1" ]]; then
    log "==== Step 1: verify environment ===="
    python -c "import sys; print('python', sys.version.split()[0])"
    python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda.is_available={torch.cuda.is_available()}")
PY
    python -c "import boto3, tap, numpy, sklearn; print('boto3 + tap + numpy + sklearn import ok')"
else
    log "==== Step 1: env verification SKIPPED ===="
fi

# -----------------------------------------------------------------------------
# 2. Download the processed annotations from S3 (the per-protein label .npz shards
#    are not in git). The CIFs in the secondary dir are not needed for scoring.
# -----------------------------------------------------------------------------
if [[ "${SKIP_ANNOT}" != "1" ]]; then
    log "==== Step 2: sync processed annotations from ${ANNOT_S3} ===="
    aws s3 sync "${ANNOT_S3}/processed_swissprot/" ./processed_swissprot/ --region "${REGION}"
    aws s3 sync "${ANNOT_S3}/processed_swissprot_a5_structure_secondary_n500/" \
        ./processed_swissprot_a5_structure_secondary_n500/ \
        --exclude "alphafold_cifs/*" --region "${REGION}"
    for d in processed_swissprot processed_swissprot_a5_structure_secondary_n500; do
        [[ -f "$d/concept_vocabulary.json" ]] || { echo "Annotations missing after sync: $d/concept_vocabulary.json" >&2; exit 1; }
    done

    # Boltz's-own-structure SS set: sync from S3; build from the predicted CIFs on first run.
    # Only needed when boltz_secondary is in CONCEPT_SETS. Built over the SAME protein set the
    # benchmark uses (PROTEIN_IDS_FILE) so coverage is exact. Unlike the AlphaFold dir, the CIF
    # cache lives outside the annotation dir, so nothing to exclude on scoring.
    if [[ ",${CONCEPT_SETS}," == *",boltz_secondary,"* ]]; then
        log "==== Step 2b: Boltz-own-structure SS set (${BOLTZ_SECONDARY_DIR}) ===="
        aws s3 sync "${ANNOT_S3}/${BOLTZ_SECONDARY_DIR}/" "./${BOLTZ_SECONDARY_DIR}/" \
            --exclude "*cif*" --region "${REGION}" || true
        if [[ ! -f "${BOLTZ_SECONDARY_DIR}/concept_vocabulary.json" && "${BUILD_BOLTZ_IF_MISSING}" == "1" ]]; then
            if [[ -f "${PROTEIN_IDS_FILE}" ]] && python -c "import pydssp" 2>/dev/null; then
                log "Boltz SS set not in S3; building from predicted CIFs over ${PROTEIN_IDS_FILE} ..."
                if python build_boltz_structure_annotations.py \
                        --manifest "${PROTEIN_IDS_FILE}" \
                        --secondary_output_dir "${BOLTZ_SECONDARY_DIR}" \
                        --build_plddt \
                        --bucket "${EVAL_BUCKET}" \
                        --key_template "${EVAL_KEY_PREFIX}{protein_id}/{protein_id}_model_0.cif.gz" \
                        --max_proteins 0 \
                        --download_workers "${DOWNLOAD_WORKERS}"; then
                    log "Uploading built Boltz SS set -> ${ANNOT_S3}/${BOLTZ_SECONDARY_DIR}/"
                    aws s3 sync "./${BOLTZ_SECONDARY_DIR}/" "${ANNOT_S3}/${BOLTZ_SECONDARY_DIR}/" \
                        --exclude "*cif*" --region "${REGION}" || true
                fi
            else
                log "Cannot build Boltz SS set (need ${PROTEIN_IDS_FILE} + importable pydssp)."
            fi
        fi
        if [[ ! -f "${BOLTZ_SECONDARY_DIR}/concept_vocabulary.json" ]]; then
            log "WARNING: Boltz SS set unavailable; dropping boltz_secondary from CONCEPT_SETS so the rest proceeds."
            CONCEPT_SETS="$(printf '%s' "${CONCEPT_SETS}" | tr ',' '\n' | grep -vx 'boltz_secondary' | paste -sd, -)"
            log "CONCEPT_SETS is now: ${CONCEPT_SETS}"
        fi
    fi
else
    log "==== Step 2: annotation sync SKIPPED ===="
fi

# -----------------------------------------------------------------------------
# 3. Build a prefix-form eval manifest (get_activations.py wants S3-prefix paths,
#    common_activation_proteins.txt is bare IDs) -- same auto-fix as run_full_workflow.sh.
# -----------------------------------------------------------------------------
[[ -f "${EVAL_MANIFEST}" ]] || { echo "Eval manifest not found: ${EVAL_MANIFEST}" >&2; exit 1; }
first_eval_entry="$(grep -v '^[[:space:]]*#' "${EVAL_MANIFEST}" | grep -v '^[[:space:]]*$' | head -n 1 || true)"
DL_MANIFEST="${EVAL_MANIFEST}"
if [[ "${first_eval_entry}" != *"/"* ]]; then
    DL_MANIFEST="${EVAL_MANIFEST%.txt}.prefixed.txt"
    log "Eval manifest is bare IDs; writing prefix form to ${DL_MANIFEST} (prefix='${EVAL_KEY_PREFIX}')"
    awk -v prefix="${EVAL_KEY_PREFIX}" '
        /^[[:space:]]*#/ {next}
        /^[[:space:]]*$/ {next}
        { gsub(/[[:space:]]+$/, "", $0); print prefix $0 "/" }
    ' "${EVAL_MANIFEST}" > "${DL_MANIFEST}"
fi

# -----------------------------------------------------------------------------
# 4. Download eval activations for each rec into the shared downloads_layer<N> dir.
#    (RawActivationLoader picks output_<rec>.npz, so both recs coexist in one dir.)
# -----------------------------------------------------------------------------
if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
    for REC in "${REC_ARR[@]}"; do
        log "==== Step 4: download eval activations layer ${LAYER} rec ${REC} -> ${EVAL_ACTIVATIONS_DIR} ===="
        python get_activations.py \
            --bucket "${EVAL_BUCKET}" \
            --manifest "${DL_MANIFEST}" \
            --layer_type "${LAYER_TYPE}" \
            --layer "${LAYER}" \
            --rec "${REC}" \
            --out "${EVAL_ACTIVATIONS_DIR}" \
            --workers "${DOWNLOAD_WORKERS}" \
            --limit 0
    done
else
    log "==== Step 4: activation download SKIPPED ===="
fi

# -----------------------------------------------------------------------------
# 5. Stage the trained SAE checkpoints (all seeds) from S3 into the per-rec cache
#    the benchmark reads. download_run_checkpoint() returns this cached copy on a
#    hit, so the benchmark never touches HuggingFace.
# -----------------------------------------------------------------------------
if [[ "${SKIP_STAGE_SAE}" != "1" ]]; then
    # Guard: an untrained (layer, rec) has an empty S3 prefix under SAE_S3_ROOT ->
    # nothing to analyse, so skip cleanly (exit 0) instead of failing at the per-seed
    # checkpoint check below. NB this is ANALYSIS-only: it never trains. If you meant to
    # train first, use run_train_and_analyse_workflow.sh (or run_full_workflow.sh).
    # (For pairformer, trained SAEs exist only for even layers + 47.)
    have_any_sae=0
    for REC in "${REC_ARR[@]}"; do
        s3_prefix_has_objects "${SAE_S3_ROOT}/rec${REC}/layer${LAYER}/" && rc=0 || rc=$?
        case "$rc" in
            0) have_any_sae=1 ;;
            1) : ;;  # prefix genuinely empty -- this (rec, layer) was never trained
            *) echo "Aborting: could not determine SAE availability for rec${REC}/layer${LAYER} (S3 ls kept erroring -- the instance role is likely still propagating or S3 is throttling the fleet). Refusing to silently skip a layer that may well be trained." >&2
               exit 1 ;;
        esac
    done
    if [[ "${have_any_sae}" == "0" ]]; then
        log "No trained ${LAYER_TYPE} SAE in S3 (${SAE_S3_ROOT}/rec{${RECS}}/layer${LAYER}/); nothing to analyse -- exiting cleanly."
        log "  This workflow only ANALYSES pre-trained SAEs. To train first, deploy with -WorkflowScript run_train_and_analyse_workflow.sh."
        exit 0
    fi
    for REC in "${REC_ARR[@]}"; do
        cache_dir="sae_explore/${SWEEP_PREFIX}layer_sweep_rec${REC}/hf_cache/layer${LAYER}"
        src="${SAE_S3_ROOT}/rec${REC}/layer${LAYER}/"
        log "==== Step 5: stage SAE checkpoints ${src} -> ${cache_dir}/ ===="
        mkdir -p "${cache_dir}"
        aws s3 sync "${src}" "${cache_dir}/" --region "${REGION}"
        for SEED in "${SEED_ARR[@]}"; do
            ckpt="${cache_dir}/$(run_name "${SEED}")/checkpoint_step_${FINAL_STEP}.pt"
            [[ -f "${ckpt}" ]] || { echo "Missing staged checkpoint: ${ckpt} (is rec${REC}/layer${LAYER}/seed${SEED} in S3?)" >&2; exit 1; }
        done
    done
else
    log "==== Step 5: SAE staging SKIPPED ===="
fi

# -----------------------------------------------------------------------------
# 6. Run the multi-seed benchmark (SAE latent vs neuron vs probes + permutation null).
# -----------------------------------------------------------------------------
if [[ "${SKIP_BENCH}" != "1" ]]; then
    log "==== Step 6: benchmark layer ${LAYER} (recs ${RECS}, seeds ${SEEDS_CSV}) ===="
    python run_parallel_benchmark.py \
        --layers "${LAYER}" \
        --recs "${RECS}" \
        --seeds "${SEEDS_CSV}" \
        --concept_sets "${CONCEPT_SETS}" \
        --layer_type "${LAYER_TYPE}" \
        --activations_dir_template "downloads_layer{layer}" \
        --n_perm "${N_PERM}" \
        --probe_on_sae True \
        --protein_ids_file "${PROTEIN_IDS_FILE}" \
        --skip_existing False \
        --max_workers "${MAX_WORKERS}" \
        --threads_per_worker "${THREADS_PER_WORKER}" \
        --device "${DEVICE}"
else
    log "==== Step 6: benchmark SKIPPED ===="
fi

# -----------------------------------------------------------------------------
# 7. Seed-averaged aggregation of the per-seed benchmark JSONs.
# -----------------------------------------------------------------------------
if [[ "${SKIP_AGGREGATE}" != "1" ]]; then
    log "==== Step 7: aggregate seeds ===="
    python aggregate_seed_benchmark.py --recs "${RECS}" --seeds "${SEEDS_CSV}" --concept_sets "${CONCEPT_SETS}" --layer_type "${LAYER_TYPE}"
else
    log "==== Step 7: aggregation SKIPPED ===="
fi

# -----------------------------------------------------------------------------
# 8. Upload results to S3 (the per-rec benchmark tree).
#    Results land in S3 first; you pull them down locally after the fleet finishes.
# -----------------------------------------------------------------------------
if [[ "${SKIP_UPLOAD}" != "1" ]]; then
    for REC in "${REC_ARR[@]}"; do
        bench_dir="sae_explore/${SWEEP_PREFIX}layer_sweep_rec${REC}/benchmark"
        dest="${RESULTS_S3}/rec${REC}/"
        if [[ -d "${bench_dir}" ]]; then
            log "==== Step 8: upload ${bench_dir} -> ${dest} ===="
            aws s3 sync "${bench_dir}" "${dest}benchmark/" --region "${REGION}"
        fi
    done
    log "Upload complete -> ${RESULTS_S3}"
else
    log "==== Step 8: upload SKIPPED ===="
fi

log "Analysis workflow complete for layer ${LAYER}."
