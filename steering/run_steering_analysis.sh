#!/bin/bash
# On-instance workflow for the FULL helix-steering causal analysis (E1 sufficiency + specificity,
# E2 trunk-vs-diffusion localization, E3 combined-ablation necessity). Launched by boltz_sae_server
# (server_setup.sh runs this in the FIRST cloned repo when a LAYER env var is present; LAYER is
# ignored). Runs as ec2-user in the core_env conda env. Designed to run unattended overnight and
# self-destruct: each experiment is failure-isolated and results upload to S3 as they complete, so a
# late bug never discards earlier work.
#
# PREREQUISITE (do once, locally, before shutting down your machine): fit the directions on the
# TRAINING split and push to S3, so this held-out run is non-circular:
#   (SwissProt repo) python export_probe_direction.py --concepts helix,strand,disulfide_bond \
#       --trunk_layers 10,16,24,32,47 --include_diffusion \
#       --train_ids ../boltz_causal_intervention/probe_train_ids.txt \
#       --out ../boltz_causal_intervention/directions/disulfide_helix_directions.npz
#   aws s3 cp .../disulfide_helix_directions.npz $S3_DIRECTIONS
# The held-out protein list (probe_heldout_ids.txt) is committed in this repo, so it arrives via git.
set -uo pipefail   # NOTE: no -e; we isolate failures per stage and always upload.

REGION="${REGION:-eu-west-2}"
S3_DIRECTIONS="${S3_DIRECTIONS:-s3://aas-processed-data-us/SwissProtAnnotation5/causal/disulfide_helix_directions.npz}"
S3_OUT="${S3_OUT:-s3://activations-at-scale-artefacts/boltz-causal/steering}"
# UNIQUE per-run output prefix so a new run never collides with a previous run's files. aws s3 sync
# skips same-size files (two CIFs of one protein are ~identical size), so writing to a shared prefix
# silently kept STALE CIFs from earlier runs -- decoupling structures from the results jsonl. A fresh
# prefix per run guarantees every CIF + jsonl uploaded here belongs to THIS run.
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
S3_RUN="$S3_OUT/runs/$RUN_ID"
S3_SURVEY="$S3_OUT/survey_cache/helix_survey.json"   # shared across runs (survey is stable for the fixed held-out set)
PROTEIN_IDS="${PROTEIN_IDS:-probe_heldout_ids.txt}"   # held-out set (committed in repo)
MAX_PROTEINS="${MAX_PROTEINS:-40}"                     # cap the held-out build/survey/steer
SURVEY_STEPS="${SURVEY_STEPS:-30}"                     # fast baseline pass for the DSSP gradient
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"                 # steering passes
HELIX_MIN="${HELIX_MIN:-0.25}"; HELIX_MAX="${HELIX_MAX:-0.85}"   # gradient band (dense DSSP)
RUN_DIFFUSION="${RUN_DIFFUSION:-1}"                    # E2 (needs helix@diffusion in the npz)
RUN_COMBINED="${RUN_COMBINED:-1}"                      # E3 (needs helix@diffusion)

echo "[steer] repo=$(pwd)  proteins=$PROTEIN_IDS (cap $MAX_PROTEINS)  region=$REGION"
export TENSORLENS_WORKING_DIR="${TENSORLENS_WORKING_DIR:-$PWD/.tensorlens}"; mkdir -p "$TENSORLENS_WORKING_DIR"

pip install -q pyyaml pydssp || true
python -c "import nutz_and_boltz, tensorlens, pytorch_lightning" \
  || { echo '[steer] FATAL: fork/tensorlens/lightning not importable'; exit 1; }

echo "[steer] RUN_ID=$RUN_ID  outputs -> $S3_RUN/outputs"
mkdir -p directions inputs outputs outputs/select
# resume: restore ONLY the shared DSSP survey cache (keeps the slow survey). NEVER restore predictions
# -- a per-run prefix already isolates structures; restoring stale CIFs is exactly what broke before.
aws s3 cp "$S3_SURVEY" outputs/select/helix_survey.json --region "$REGION" >/dev/null 2>&1 || true
aws s3 cp "$S3_DIRECTIONS" directions/disulfide_helix_directions.npz --region "$REGION" \
  || { echo '[steer] FATAL: could not fetch directions from S3 (did you push the train-fit npz?)'; exit 1; }

# upload everything to the UNIQUE per-run prefix (fresh prefix => no same-size skip of stale files)
upload() { aws s3 sync outputs "$S3_RUN/outputs" --region "$REGION" >/dev/null 2>&1 || true; }
trap upload EXIT

run_stage() {  # name + command; isolate failure, keep going, upload partial results
    local name="$1"; shift
    echo "==================== [steer] $name ===================="
    if "$@"; then echo "[steer] $name OK"; else echo "[steer] $name FAILED (continuing)"; fi
    upload
}

# 0) build held-out per-protein inputs (YAML+MSA from S3)
run_stage build_inputs python scripts/build_inputs.py --proteins "$PROTEIN_IDS" --out-dir inputs --limit "$MAX_PROTEINS"

# 1) survey dense-DSSP helix on the baseline -> the TRUE gradient (SwissProt annotations under-report)
run_stage survey python scripts/select_protein.py --input_dir inputs \
    --sampling_steps "$SURVEY_STEPS" --output_dir outputs/select
# refresh the shared survey cache for future runs (survey is stable for the fixed held-out set)
aws s3 cp outputs/select/helix_survey.json "$S3_SURVEY" --region "$REGION" >/dev/null 2>&1 || true

# 2) E1: trunk steering across the gradient (sufficiency + random specificity, per-protein calibrated)
run_stage E1_trunk python scripts/probe_batch.py --from_survey outputs/select/helix_survey.json \
    --helix_min "$HELIX_MIN" --helix_max "$HELIX_MAX" --site trunk \
    --sampling_steps "$SAMPLING_STEPS" --max_proteins "$MAX_PROTEINS" --output_dir outputs/probe_batch

# 3) E2: diffusion-site steering (the trunk-vs-diffusion localization; needs helix@diffusion)
if [ "$RUN_DIFFUSION" = "1" ]; then
  run_stage E2_diffusion python scripts/probe_batch.py --from_survey outputs/select/helix_survey.json \
      --helix_min "$HELIX_MIN" --helix_max "$HELIX_MAX" --site diffusion \
      --sampling_steps "$SAMPLING_STEPS" --max_proteins "$MAX_PROTEINS" --output_dir outputs/probe_batch
fi

# 3b) E2 multi-layer: single-layer diffusion steering is underpowered (a null there is likely a
#     method artifact, per DiT steering work), so also inject at ALL diffusion layers. Separate
#     output dir so it doesn't clobber the single-layer batch_diffusion.jsonl.
if [ "$RUN_DIFFUSION" = "1" ]; then
  run_stage E2b_diffusion_multilayer python scripts/probe_batch.py --from_survey outputs/select/helix_survey.json \
      --helix_min "$HELIX_MIN" --helix_max "$HELIX_MAX" --site diffusion --diffusion_multilayer \
      --sampling_steps "$SAMPLING_STEPS" --max_proteins "$MAX_PROTEINS" --output_dir outputs/probe_batch_multilayer
fi

# 4) E3: combined multi-site ablation (necessity via redundancy; needs helix@diffusion)
if [ "$RUN_COMBINED" = "1" ]; then
  run_stage E3_combined python scripts/probe_combined.py --from_survey outputs/select/helix_survey.json \
      --helix_min 0.35 --helix_max 0.95 \
      --sampling_steps "$SAMPLING_STEPS" --max_proteins "$MAX_PROTEINS" --output_dir outputs/probe_combined
fi

# 5) E5: SAE-direction trunk steering (cleaner direction?). Runs each SAE variant present in the npz
#    (helix_sae = single top-F1 latent; helix_sae_f1set = all F1-above-null latents). Produce them with
#    export_sae_direction.py locally + re-upload to S3. Steers with --concept, no code change.
for sae_concept in helix_sae helix_sae_f1set; do
  if python -c "import numpy as np,sys; d=np.load('directions/disulfide_helix_directions.npz',allow_pickle=True); sys.exit(0 if '${sae_concept}@trunk_L47' in d.files else 1)" 2>/dev/null; then
    run_stage "E5_${sae_concept}" python scripts/probe_batch.py --from_survey outputs/select/helix_survey.json \
        --helix_min "$HELIX_MIN" --helix_max "$HELIX_MAX" --site trunk --concept "$sae_concept" \
        --sampling_steps "$SAMPLING_STEPS" --max_proteins "$MAX_PROTEINS" --output_dir "outputs/probe_batch_${sae_concept}"
  else
    echo "[steer] no ${sae_concept}@trunk_L47 in npz; skipping (export_sae_direction.py + re-upload to enable)"
  fi
done

# 6) E6: OTHER secondary-structure concepts (beta-strand; coil once a coil direction is exported).
#    Each records the full DSSP confusion matrix (helix/strand/coil) per condition, so we can check
#    specificity (steering strand raises strand, not helix). strand@trunk_L47 is already in the npz;
#    coil needs export_probe_direction.py to add a 'coil' concept + re-upload (see docs).
for ss_concept in strand coil; do
  if python -c "import numpy as np,sys; d=np.load('directions/disulfide_helix_directions.npz',allow_pickle=True); sys.exit(0 if '${ss_concept}@trunk_L47' in d.files else 1)" 2>/dev/null; then
    run_stage "E6_${ss_concept}" python scripts/probe_batch.py --from_survey outputs/select/helix_survey.json \
        --helix_min 0.0 --helix_max 1.0 --site trunk --concept "$ss_concept" \
        --sampling_steps "$SAMPLING_STEPS" --max_proteins "$MAX_PROTEINS" --output_dir "outputs/probe_batch_${ss_concept}"
  else
    echo "[steer] no ${ss_concept}@trunk_L47 in npz; skipping (export that concept + re-upload to enable)"
  fi
done

echo "[steer] DONE. Results at $S3_RUN/outputs  (RUN_ID=$RUN_ID)"
