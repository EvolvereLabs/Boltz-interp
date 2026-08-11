#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Simplified end-to-end SAE workflow.
#
# Assumes you're running inside an already-set-up environment that has:
#   - python with torch (CUDA build matching the host driver)
#   - the repo installed (pip install .) so deps like tap/boto3 are present
#   - get_activations.py, train_sae.py, eval_sae.py reachable from the current
#     working directory
#
# On the EC2 image built by server_setup.sh that means: activate core_env first.
#   conda activate /home/ec2-user/core_env
#   bash run_full_workflow_simple.sh
# -----------------------------------------------------------------------------
set -euo pipefail

log() { printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"; }

# -----------------------------------------------------------------------------
# 0. Configuration (override any of these with env vars before invoking).
# -----------------------------------------------------------------------------
LAYER="${LAYER:-10}"
REC="${REC:-1}"

# Which Boltz module the activations come from: "pairformer" (default) or
# "diffusion". This selects the S3 layer subfolder, the activation feature
# dim, and the run-name / output tag so diffusion results never collide with
# pairformer ones. Forwarded as --layer_type to get_activations.py.
LAYER_TYPE="${LAYER_TYPE:-pairformer}"
LAYER_TYPE="$(printf '%s' "${LAYER_TYPE}" | tr '[:upper:]' '[:lower:]')"
case "${LAYER_TYPE}" in
    pairformer|diffusion) ;;
    *) echo "ERROR: LAYER_TYPE must be 'pairformer' or 'diffusion', got '${LAYER_TYPE}'" >&2; exit 1 ;;
esac

TRAIN_MANIFEST="${TRAIN_MANIFEST:-80proteins.txt}"
EVAL_MANIFEST="${EVAL_MANIFEST:-SwissProtproteins.txt}"

TRAIN_BUCKET="${TRAIN_BUCKET:-boltz-1-activations}"
EVAL_BUCKET="${EVAL_BUCKET:-swissprot-annotated-proteins-activations}"

TRAIN_ACTIVATIONS_DIR="${TRAIN_ACTIVATIONS_DIR:-./downloads_train_80k_layer${LAYER}}"
EVAL_ACTIVATIONS_DIR="${EVAL_ACTIVATIONS_DIR:-./downloads_layer${LAYER}}"

DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-32}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"

SEEDS_CSV="${SEEDS_CSV:-1,2,3}"
# Activation feature dim. Pairformer single representation is 384; the
# diffusion transformer operates on 2*token_s = 768. train_sae.py hard-checks
# this against the real activation shape, so a wrong value fails loudly rather
# than training on mis-shaped data. Override INPUT_DIM to force a value.
if [[ "${LAYER_TYPE}" == "diffusion" ]]; then
    INPUT_DIM="${INPUT_DIM:-768}"
else
    INPUT_DIM="${INPUT_DIM:-384}"
fi
LATENT_DIM="${LATENT_DIM:-2048}"
TOP_K="${TOP_K:-256}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
MAX_STEPS="${MAX_STEPS:-500000}"
# Checkpoints are large and only the FINAL one (checkpoint_step_${MAX_STEPS}.pt) is used
# downstream (eval / analysis); the training curve lives in stats.jsonl. So save sparingly
# -- every 100k steps -> ~5 checkpoints/seed instead of 50, cutting disk + S3 + staging time.
SAVE_EVERY="${SAVE_EVERY:-100000}"
LOG_EVERY="${LOG_EVERY:-500}"
LR="${LR:-5e-4}"
WEIGHT_L2="${WEIGHT_L2:-3e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# Namespace the runs directory by layer so a freshly-cloned repo's committed
# artifacts for another layer can never cause a run-name collision or trigger
# the "final checkpoint already exists -> skip training" shortcut below.
RUNS_DIR="${RUNS_DIR:-./sae_runs_layer${LAYER}}"
# Run names are built directly from the hyperparameters (see resolve_run_name
# below) using the existing pairformer{LAYER}_topk{K}_lat{L}_..._seed{S}
# convention. Override RUN_NAME_PREFIX to bolt a custom string onto the front
# (e.g. RUN_NAME_PREFIX="exp42_" for an experiment-tagged run).
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"

SKIP_VERIFY="${SKIP_VERIFY:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# If the eval manifest contains bare protein IDs instead of prefix paths,
# auto-generate a fixed prefix-form manifest by prepending EVAL_KEY_PREFIX
# and appending "/" to each ID. The S3 layout for SwissProt annotation 5 is
# "SwissProtAnnotation5/activations/<id>/", so the default prefix is
# "SwissProtAnnotation5/activations/".
EVAL_KEY_PREFIX="${EVAL_KEY_PREFIX:-SwissProtAnnotation5/activations/}"
# S3 subfolder that holds the per-protein activation files for this layer.
# Must match get_activations.py:get_layer_folder_name().
if [[ "${LAYER_TYPE}" == "diffusion" ]]; then
    LAYER_SUBFOLDER="DiffusionTransformerLayer_${LAYER} from DiffusionTransformer from DiffusionModule from AtomDiffusion"
else
    LAYER_SUBFOLDER="PairformerLayer_${LAYER} s from PairformerLayer"
fi

log "Layer type:           ${LAYER_TYPE}"
log "Layer:                ${LAYER} (subfolder: '${LAYER_SUBFOLDER}')"
log "Input dim:            ${INPUT_DIM}"
log "Train activations:    ${TRAIN_ACTIVATIONS_DIR}"
log "Eval  activations:    ${EVAL_ACTIVATIONS_DIR}"
log "Runs directory:       ${RUNS_DIR}"
log "Seeds:                ${SEEDS_CSV}"
log "Max steps per seed:   ${MAX_STEPS}"
log "Weight L2:            ${WEIGHT_L2}"
log "Python:               $(command -v python)"

mkdir -p "${RUNS_DIR}"

# -----------------------------------------------------------------------------
# 1. Verify the env (replaces the old uv-based setup).
# -----------------------------------------------------------------------------
if [[ "${SKIP_VERIFY}" != "1" ]]; then
    log "==== Step 1: verify environment ===="
    python -c "import sys; print('python', sys.version.split()[0])"
    python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda.is_available={torch.cuda.is_available()}  devices={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "ERROR: torch.cuda.is_available() is False.\n"
        "  Check 'nvidia-smi' for the max CUDA your driver supports, then reinstall torch:\n"
        "    pip install --force-reinstall --index-url https://download.pytorch.org/whl/cu126 \\\n"
        "        torch torchvision torchaudio\n"
        "  (swap cu126 for cu124/cu128/etc. to match your driver.)"
    )
PY
    # Spot-check the deps the workflow scripts actually need.
    python -c "import boto3, tap; print('boto3 + tap import ok')"
else
    log "==== Step 1: env verification SKIPPED (SKIP_VERIFY=1) ===="
fi

# Pick a device for the sub-scripts.
if [[ -z "${DEVICE:-}" ]]; then
    if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        DEVICE="cuda"
    else
        DEVICE="cpu"
    fi
fi
log "Using device: ${DEVICE}"

# -----------------------------------------------------------------------------
# 2. Download training and evaluation activations for the chosen layer.
# -----------------------------------------------------------------------------
if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
    log "==== Step 2: download activations for layer ${LAYER} ===="

    [[ -f "${TRAIN_MANIFEST}" ]] || { echo "Training manifest not found: ${TRAIN_MANIFEST}" >&2; exit 1; }
    [[ -f "${EVAL_MANIFEST}"  ]] || { echo "Eval manifest not found: ${EVAL_MANIFEST}"     >&2; exit 1; }

    # Auto-fix the eval manifest if it's bare IDs instead of prefix paths.
    first_eval_entry="$(grep -v '^[[:space:]]*#' "${EVAL_MANIFEST}" | grep -v '^[[:space:]]*$' | head -n 1 || true)"
    if [[ -z "${first_eval_entry}" ]]; then
        echo "Eval manifest ${EVAL_MANIFEST} is empty or only comments." >&2
        exit 1
    fi
    if [[ "${first_eval_entry}" != *"/"* ]]; then
        FIXED_EVAL_MANIFEST="${EVAL_MANIFEST%.txt}.prefixed.txt"
        log "Eval manifest is bare IDs; writing prefix form to ${FIXED_EVAL_MANIFEST} (prefix='${EVAL_KEY_PREFIX}')"
        # Strip comments + blank lines, drop trailing whitespace, then build
        # the prefix-form path with simple string concatenation. We do NOT
        # use a {ID}-style placeholder because awk's regex parser treats
        # \{ ... \} as repetition syntax in some implementations, which was
        # silently mangling the substitution to just "}".
        awk -v prefix="${EVAL_KEY_PREFIX}" '
            /^[[:space:]]*#/ {next}
            /^[[:space:]]*$/ {next}
            { gsub(/[[:space:]]+$/, "", $0); print prefix $0 "/" }
        ' "${EVAL_MANIFEST}" > "${FIXED_EVAL_MANIFEST}"
        EVAL_MANIFEST="${FIXED_EVAL_MANIFEST}"
        log "First fixed entry: $(head -n 1 "${EVAL_MANIFEST}")"
    fi

    log "Downloading training activations -> ${TRAIN_ACTIVATIONS_DIR}"
    python get_activations.py \
        --bucket "${TRAIN_BUCKET}" \
        --manifest "${TRAIN_MANIFEST}" \
        --layer_type "${LAYER_TYPE}" \
        --layer "${LAYER}" \
        --rec "${REC}" \
        --out "${TRAIN_ACTIVATIONS_DIR}" \
        --workers "${DOWNLOAD_WORKERS}" \
        --limit "${TRAIN_LIMIT}"

    log "Downloading eval activations -> ${EVAL_ACTIVATIONS_DIR}"
    python get_activations.py \
        --bucket "${EVAL_BUCKET}" \
        --manifest "${EVAL_MANIFEST}" \
        --layer_type "${LAYER_TYPE}" \
        --layer "${LAYER}" \
        --rec "${REC}" \
        --out "${EVAL_ACTIVATIONS_DIR}" \
        --workers "${DOWNLOAD_WORKERS}" \
        --limit "${EVAL_LIMIT}"
else
    log "==== Step 2: data download SKIPPED (SKIP_DOWNLOAD=1) ===="
fi

# -----------------------------------------------------------------------------
# 3. Train one SAE per seed.
# -----------------------------------------------------------------------------
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"

resolve_run_name() {
    # Build the run name directly from the hyperparameters. Using printf
    # avoids brittle bash brace-substitution which (depending on extglob /
    # bash version) can mangle literal "{LAYER}" placeholders inside a
    # template string -- earlier versions of this script produced names like
    # "pairformer{LAYER_topk256_..._seed1}" because of that bug.
    local seed="$1"
    # Tag the run by layer type ("pairformer10_..." / "diffusion10_...") so
    # diffusion checkpoints never collide with pairformer ones.
    printf '%s%s\n' \
        "${RUN_NAME_PREFIX}" \
        "${LAYER_TYPE}${LAYER}_topk${TOP_K}_lat${LATENT_DIM}_demean_longtrain${MAX_STEPS}_l2_${WEIGHT_L2}_seed${seed}"
}

CHECKPOINTS=()
RUN_DIRS=()

if [[ "${SKIP_TRAIN}" != "1" ]]; then
    log "==== Step 3: train ${#SEEDS[@]} SAEs (seeds: ${SEEDS_CSV}) ===="
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="$(resolve_run_name "${SEED}")"
        RUN_DIR="${RUNS_DIR}/${RUN_NAME}"
        FINAL_CKPT="${RUN_DIR}/checkpoint_step_${MAX_STEPS}.pt"
        RUN_DIRS+=("${RUN_DIR}")
        CHECKPOINTS+=("${FINAL_CKPT}")

        if [[ -f "${FINAL_CKPT}" ]]; then
            log "Final checkpoint already exists for seed ${SEED}; skipping training."
            continue
        fi

        log "Training seed ${SEED} -> ${RUN_DIR}"
        python train_sae.py \
            --activations_dir "${TRAIN_ACTIVATIONS_DIR}" \
            --layer_subfolder "${LAYER_SUBFOLDER}" \
            --rec "${REC}" \
            --input_dim "${INPUT_DIM}" \
            --latent_dim "${LATENT_DIM}" \
            --k "${TOP_K}" \
            --batch_size "${BATCH_SIZE}" \
            --max_steps "${MAX_STEPS}" \
            --save_every "${SAVE_EVERY}" \
            --log_every "${LOG_EVERY}" \
            --lr "${LR}" \
            --weight_l2 "${WEIGHT_L2}" \
            --weight_decay "${WEIGHT_DECAY}" \
            --grad_clip "${GRAD_CLIP}" \
            --seed "${SEED}" \
            --run_name "${RUN_NAME}" \
            --out_dir "${RUNS_DIR}" \
            --device "${DEVICE}" \
            --tied_init \
            --demean_embeddings \
            --pre_encoder_bias \
            --normalize_decoder
    done
else
    log "==== Step 3: training SKIPPED (SKIP_TRAIN=1) ===="
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="$(resolve_run_name "${SEED}")"
        RUN_DIR="${RUNS_DIR}/${RUN_NAME}"
        RUN_DIRS+=("${RUN_DIR}")
        CHECKPOINTS+=("${RUN_DIR}/checkpoint_step_${MAX_STEPS}.pt")
    done
fi

# -----------------------------------------------------------------------------
# 4. Evaluate each SAE.
# -----------------------------------------------------------------------------
if [[ "${SKIP_EVAL}" != "1" ]]; then
    log "==== Step 4: evaluate each SAE on ${EVAL_ACTIVATIONS_DIR} ===="
    for idx in "${!SEEDS[@]}"; do
        SEED="${SEEDS[${idx}]}"
        RUN_DIR="${RUN_DIRS[${idx}]}"
        CKPT="${CHECKPOINTS[${idx}]}"
        EVAL_OUT="${RUN_DIR}/eval_step_${MAX_STEPS}.json"

        [[ -f "${CKPT}" ]] || { echo "Missing checkpoint for seed ${SEED}: ${CKPT}" >&2; exit 1; }
        if [[ -f "${EVAL_OUT}" ]]; then
            log "Eval JSON already exists for seed ${SEED}; skipping."
            continue
        fi

        log "Evaluating seed ${SEED} (${CKPT})"
        python eval_sae.py \
            --checkpoint "${CKPT}" \
            --activations_dir "${EVAL_ACTIVATIONS_DIR}" \
            --device "${DEVICE}" \
            --out_path "${EVAL_OUT}"
    done
else
    log "==== Step 4: evaluation SKIPPED (SKIP_EVAL=1) ===="
fi
