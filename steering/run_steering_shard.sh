#!/bin/bash
# SHARDED steering: one (concept) per instance, for parallel fan-out across secondary structures x
# direction types. The previous single-instance run did every concept sequentially (~24h); this runs
# one concept per GPU instance so N instances finish in ~1/N the wall-clock.
#
# Shard is chosen by SHARD (or LAYER, which boltz_sae_server's server_setup.sh passes through). Launch
# one instance per shard, e.g. deploy_per_layer.ps1 ... -WorkflowScript run_steering_shard.sh -LayerNumbers 0,1,2,3,4,5
#
#   shard 0 helix       (probe)    3 shard strand_sae  (SAE)
#   shard 1 helix_sae   (SAE)      4 shard coil        (probe)
#   shard 2 strand      (probe)    5 shard coil_sae     (SAE)
#
# Each shard: restore the SHARED survey cache (skip the slow survey), build the held-out inputs, steer
# ONE concept at the trunk, and upload to a per-concept S3 prefix (no cross-shard collision). The
# helix shard's prediction CIFs double as the before/after structure-viz inputs.
set -uo pipefail

REGION="${REGION:-eu-west-2}"
S3_DIRECTIONS="${S3_DIRECTIONS:-s3://aas-processed-data-us/SwissProtAnnotation5/causal/disulfide_helix_directions.npz}"
S3_OUT="${S3_OUT:-s3://activations-at-scale-artefacts/boltz-causal/steering}"
S3_SURVEY="$S3_OUT/survey_cache/helix_survey.json"
# per-RUN tag (shared across the same-day shard fan-out) so a new run's per-concept outputs never
# collide with a previous run's -- avoids aws s3 sync skipping same-size CIFs and serving stale files.
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%d)}"
PROTEIN_IDS="${PROTEIN_IDS:-probe_heldout_ids.txt}"
MAX_PROTEINS="${MAX_PROTEINS:-200}"   # 200 => all buildable held-out proteins (the full ~97), not a subset
SURVEY_STEPS="${SURVEY_STEPS:-30}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"

# Shard -> (S3 label, direction actually steered, n_random). Concept shards are LEAN (NR=1: one paired
# random control per dose). Shard 6 is a dedicated RANDOM-NULL shard (20 random directions) whose
# distribution is REUSED to score every concept -- a random direction's effect is concept-independent,
# so we compute it once instead of 20x per concept (the previous run's overkill).
SHARD="${SHARD:-${LAYER:-0}}"
case "$SHARD" in
  0) LABEL=helix;       STEER=helix;      NR=1 ;;
  1) LABEL=helix_sae;   STEER=helix_sae;  NR=1 ;;
  2) LABEL=strand;      STEER=strand;     NR=1 ;;
  3) LABEL=strand_sae;  STEER=strand_sae; NR=1 ;;
  4) LABEL=coil;        STEER=coil;       NR=1 ;;
  5) LABEL=coil_sae;    STEER=coil_sae;   NR=1 ;;
  6) LABEL=random_null; STEER=helix;      NR=20 ;;   # reusable null: 20 random dirs (concept-independent)
  *) echo "[shard] unknown SHARD=$SHARD (expected 0-6)"; exit 1 ;;
esac
S3_SHARD="$S3_OUT/shards/$RUN_TAG/$LABEL"
echo "[shard] SHARD=$SHARD LABEL=$LABEL STEER=$STEER NR=$NR RUN_TAG=$RUN_TAG  ->  $S3_SHARD/outputs"

export TENSORLENS_WORKING_DIR="${TENSORLENS_WORKING_DIR:-$PWD/.tensorlens}"; mkdir -p "$TENSORLENS_WORKING_DIR"
pip install -q pyyaml pydssp || true
python -c "import nutz_and_boltz, tensorlens, pytorch_lightning" \
  || { echo '[shard] FATAL: fork/tensorlens/lightning not importable'; exit 1; }

mkdir -p directions inputs outputs outputs/select
aws s3 cp "$S3_DIRECTIONS" directions/disulfide_helix_directions.npz --region "$REGION" \
  || { echo '[shard] FATAL: could not fetch directions from S3'; exit 1; }
# fail fast if this shard's concept direction isn't in the npz
python -c "import numpy as np,sys; d=np.load('directions/disulfide_helix_directions.npz',allow_pickle=True); sys.exit(0 if '${STEER}@trunk_L47' in d.files else 1)" \
  || { echo "[shard] FATAL: ${STEER}@trunk_L47 not in npz (export it + re-upload)"; exit 1; }

upload() { aws s3 sync outputs "$S3_SHARD/outputs" --region "$REGION" >/dev/null 2>&1 || true; }
trap upload EXIT

# inputs = ALL held-out proteins. Restore the shared survey cache, then run select_protein anyway:
# it skips proteins already cached and surveys the rest, so the full held-out set gets covered.
python scripts/build_inputs.py --proteins "$PROTEIN_IDS" --out-dir inputs --limit "$MAX_PROTEINS" || true
aws s3 cp "$S3_SURVEY" outputs/select/helix_survey.json --region "$REGION" >/dev/null 2>&1 || true
python scripts/select_protein.py --input_dir inputs --sampling_steps "$SURVEY_STEPS" --output_dir outputs/select
aws s3 cp outputs/select/helix_survey.json "$S3_SURVEY" --region "$REGION" >/dev/null 2>&1 || true

# steer this shard's direction at the trunk over the FULL held-out set. Records the helix/strand/coil
# confusion matrix per condition (specificity checkable). Shard 6 (NR=20) is the reusable random null.
echo "==================== [shard] steer $STEER (label=$LABEL n_random=$NR) ===================="
python scripts/probe_batch.py --from_survey outputs/select/helix_survey.json \
    --helix_min 0.0 --helix_max 1.0 --site trunk --concept "$STEER" --n_random "$NR" \
    --sampling_steps "$SAMPLING_STEPS" --max_proteins "$MAX_PROTEINS" \
    --output_dir "outputs/probe_batch_${LABEL}"
upload
echo "[shard] DONE $LABEL. Results at $S3_SHARD/outputs"
