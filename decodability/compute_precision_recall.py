#!/usr/bin/env python3
"""Recompute per-concept **precision** and **recall** (not just F1) for the SAE
winner, the single-neuron winner, and the supervised linear probe, by rebuilding
the exact benchmark matrices and re-scoring with the stored F1 recipe.

The layer-sweep benchmark only persists F1 (+ winning feature_idx / threshold_pct),
so precision/recall must be re-derived from activations. Pairformer activations and
SAE checkpoints are staged locally, so this runs offline for the trunk. Diffusion
settings can be added once their activations are pulled (see --help).

Writes a tidy CSV: stack,rec,layer,concept_set,concept,method,precision,recall,f1.
Run:  python compute_precision_recall.py
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from benchmark_f1 import (
    DEFAULT_THRESHOLD_PCTS,
    MIN_POSITIVE_RESIDUES,
    precision_recall_for_feature,
    precompute_feature_thresholds,
    score_best_f1_per_concept,
)
from linear_probe import _protein_groups, out_of_fold_probabilities
from run_layer_benchmark import load_benchmark_matrices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("pr")

CONCEPT_SETS = {"secondary": "processed_swissprot_a5_structure_secondary_n500",
                "swissprot": "processed_swissprot"}
# concepts the comparison notebook reports (skip the long tail)
KEEP = {
    "secondary": ["secondary_structure:helix", "secondary_structure:strand", "secondary_structure:coil"],
    "swissprot": ["beta_strand", "binding_site:substrate",
                  "compositional_bias:basic_and_acidic_residues", "compositional_bias:low_complexity",
                  "compositional_bias:polar_residues", "disulfide_bond",
                  "glycosylation:n-linked_glcnac_asparagine", "helix",
                  "modified_residue:phosphoserine", "region:disordered", "signal_peptide", "turn"],
}
PF_CACHE = Path("sae_explore/layer_sweep/hf_cache")

# (stack, layer_type, layer, rec, activations_dir, cache_dir)
SPECS_PF = [
    ("pairformer", "pairformer", 6,  1, "downloads_layer6",  PF_CACHE),
    ("pairformer", "pairformer", 20, 1, "downloads_layer20", PF_CACHE),
    ("pairformer", "pairformer", 47, 1, "downloads_layer47", PF_CACHE),
]
# Diffusion module output (L22 at final sampling step rec199); checkpoint pre-staged
# into a rec-specific cache, activations downloaded by _dl_diffusion.py.
SPECS_DIFF = [
    ("diffusion", "diffusion", 22, 199, "downloads_diffusion_rec199",
     Path("sae_explore/diffusion_cache/rec199")),
]


def _threshold_value(thresholds: np.ndarray, feat: int, pct: float) -> float:
    """Activation value for a (feature, percentile) cell of the threshold table."""
    if feat < 0 or np.isnan(pct):
        return np.inf
    return float(thresholds[feat, DEFAULT_THRESHOLD_PCTS.index(pct)])


def unsup_pr(fm, labels, offsets, concept_list, keep):
    """Precision/recall/F1 for each concept's best single feature (SAE or neuron)."""
    thr = precompute_feature_thresholds(fm)
    best_f1, best_feat, best_thr = score_best_f1_per_concept(fm, thr, labels, offsets)
    out = {}
    for c in keep:
        if c not in concept_list:
            continue
        ci = concept_list.index(c)
        feat, pct = int(best_feat[ci]), float(best_thr[ci])
        tval = _threshold_value(thr, feat, pct)
        p, r, f = precision_recall_for_feature(fm[:, feat], tval, labels[:, ci], offsets)
        out[c] = (p, r, f)
    return out


def probe_pr(fm, labels, offsets, concept_list, keep):
    """Precision/recall/F1 for the held-out linear probe (OOF probs, same F1 recipe)."""
    groups = _protein_groups(offsets, fm.shape[0])
    alive = fm.std(axis=0) > 0.0
    fm_alive = fm[:, alive]
    out = {}
    for c in keep:
        if c not in concept_list:
            continue
        ci = concept_list.index(c)
        y = labels[:, ci].astype(np.int64)
        if int(y.sum()) < MIN_POSITIVE_RESIDUES:
            continue
        oof = out_of_fold_probabilities(fm_alive, y, groups, n_splits=5, c_reg=1.0, max_iter=2000)
        prob = oof[:, None].astype(np.float32)
        thr = precompute_feature_thresholds(prob)
        _, _, best_thr = score_best_f1_per_concept(prob, thr, labels[:, ci:ci + 1], offsets)
        tval = _threshold_value(thr, 0, float(best_thr[0]))
        p, r, f = precision_recall_for_feature(prob[:, 0], tval, labels[:, ci], offsets)
        out[c] = (p, r, f)
    return out


def main(specs, out_csv: str, max_proteins: int = 250) -> None:
    rows = []
    for stack, ltype, layer, rec, act_dir, cache_dir in specs:
        if not Path(act_dir).exists():
            LOG.warning("activations dir %s missing; skipping %s L%d", act_dir, stack, layer)
            continue
        for cset, ann_dir in CONCEPT_SETS.items():
            LOG.info("=== %s L%d rec%d / %s ===", stack, layer, rec, cset)
            m = load_benchmark_matrices(
                layer=layer, concept_set_name=cset, annotation_dir=ann_dir,
                activations_dir=act_dir, seed=1, rec=rec, max_proteins=max_proteins,
                device="cpu", cache_dir=Path(cache_dir), require_full_rec=False,
                protein_subset=None, layer_type=ltype,
            )
            if m is None:
                LOG.warning("no matrices for %s L%d / %s; skipping", stack, layer, cset)
                continue
            keep = KEEP[cset]
            methods = {
                "sae": unsup_pr(m.sae_matrix, m.label_matrix, m.protein_offsets, m.concept_list, keep),
                "neuron": unsup_pr(m.raw_matrix, m.label_matrix, m.protein_offsets, m.concept_list, keep),
                "probe_raw": probe_pr(m.raw_matrix, m.label_matrix, m.protein_offsets, m.concept_list, keep),
            }
            for method, res in methods.items():
                for c, (p, r, f) in res.items():
                    rows.append(dict(stack=stack, rec=rec, layer=layer, concept_set=cset,
                                     concept=c, method=method, precision=round(p, 5),
                                     recall=round(r, 5), f1=round(f, 5),
                                     n_proteins=len(m.used_ids)))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stack", "rec", "layer", "concept_set", "concept",
                                           "method", "precision", "recall", "f1", "n_proteins"])
        w.writeheader()
        w.writerows(rows)
    LOG.info("wrote %d rows -> %s", len(rows), out_csv)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "pf"
    mp = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    if which == "diff":
        main(SPECS_DIFF, "sae_explore/precision_recall_diffusion.csv", max_proteins=mp)
    elif which == "all":
        main(SPECS_PF + SPECS_DIFF, "sae_explore/precision_recall.csv", max_proteins=mp)
    else:
        main(SPECS_PF, "sae_explore/precision_recall.csv", max_proteins=mp)
