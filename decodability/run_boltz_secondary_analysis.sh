#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Fleet wrapper: ADD only the boltz_secondary concept set (DSSP H/E/C labelled from
# Boltz-1's OWN predicted CIF) to the existing per-layer analysis already on S3:
#   s3://boltz-saes-l2/analysis/            (pairformer)
#   s3://boltz-saes-l2/diffusion/analysis/  (diffusion)
#
# Deployed exactly like the normal analysis fleet, just with this as the workflow:
#   ./deploy_per_layer.ps1 -e <key> -WorkflowScript run_boltz_secondary_analysis.sh ...
#
# Why a wrapper exists at all:
#   server_setup.sh forwards only LAYER/REC/LAYER_TYPE to the workflow -- NOT CONCEPT_SETS.
#   So we can't ask the fleet for "boltz_secondary only" via an env var on deploy. This
#   wrapper pins that env, then execs the real run_analysis_workflow.sh.
#
# What it deliberately does and skips:
#   * CONCEPT_SETS=boltz_secondary    -> only the NEW set; secondary/swissprot already on S3
#                                        (their permutation nulls are the expensive part -- skip).
#   * BUILD_BOLTZ_IF_MISSING=0        -> the set is pre-built + uploaded to S3 ONCE before the
#                                        fleet launches, so no instance rebuilds it (no race).
#   * SKIP_AGGREGATE                  -> each instance holds only ITS layer; running it
#                                        would clobber the COMBINED cross-layer table at the
#                                        analysis/ root on S3 (benchmark_agg_index.jsonl).
#                                        The per-seed boltz_secondary_seed*.json are
#                                        additive and safe.
#                                        Rebuild the combined tables ONCE locally afterwards
#                                        (see "after the fleet" in the deploy notes).
#
# Everything else (which recs, which instance type, pairformer vs diffusion) is inherited
# from run_analysis_workflow.sh / the deploy, so this stays a 4-line policy file.
# -----------------------------------------------------------------------------
set -euo pipefail

export CONCEPT_SETS="boltz_secondary"
export BUILD_BOLTZ_IF_MISSING="${BUILD_BOLTZ_IF_MISSING:-0}"
export SKIP_AGGREGATE="${SKIP_AGGREGATE:-1}"

exec bash "$(dirname "$0")/run_analysis_workflow.sh"
