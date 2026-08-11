#!/usr/bin/env python3
"""Grouped (by-protein) bootstrap CIs for the concept-F1 benchmark.

Reviewer concern: the eval set is small and uneven (99 proteins; e.g. 25 disulfide
domains, 22 signal-peptide domains, 47 glycosylation sites), so some per-concept F1
values are fragile. The cross-seed repeats do **not** address this -- they only retrain
the SAE dictionary; `probe_raw` and `neuron` F1 are deterministic across seeds, and the
SAE seed std measures dictionary-init stability, not evaluation-sample sampling error.

The right uncertainty here is *finite-sample* sampling error: if a different draw of
proteins had been evaluated, how much would F1 move? We estimate it with a **cluster
bootstrap over proteins** (proteins are the independent unit; the probe already groups
folds by protein). The operating point (winning feature + threshold, or probe winner) is
**fixed** at the full-sample value -- we do NOT re-select it inside each resample -- so the
CI isolates pure sampling uncertainty and does not double-count the threshold-grid optimism.

Mechanics, per (stack, layer, concept_set, concept, method):
  1. Reproduce the full-sample winner exactly as `compute_precision_recall.py` does, giving
     a fixed boolean prediction vector `pred` over all residues.
  2. Reduce to per-protein sufficient statistics: predicted-positive residues, true-positive
     residues (for per-residue precision), and total / hit domains (for per-domain recall).
  3. Resample proteins with replacement B times (vectorised as a multinomial count matrix
     times the per-protein stats) and recompute precision / recall / F1 each time.
  4. Report the point estimate plus the 2.5 / 97.5 percentile CI.

Writes  sae_explore/concept_f1_ci.csv  with one row per (..., method):
  stack,rec,layer,concept_set,concept,method,n_proteins,pos_domains,pos_residues,
  precision,precision_lo,precision_hi,recall,recall_lo,recall_hi,f1,f1_lo,f1_hi,
  f1_se,n_boot,boot_seed

Run:  python bootstrap_concept_f1.py            # trunk (pairformer), staged locally
      python bootstrap_concept_f1.py all 250    # + diffusion, once its activations exist
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

from benchmark_f1 import (
    DEFAULT_THRESHOLD_PCTS,
    MIN_POSITIVE_RESIDUES,
    precompute_feature_thresholds,
    score_best_f1_per_concept,
)
from compute_precision_recall import (
    CONCEPT_SETS,
    KEEP,
    SPECS_DIFF,
    SPECS_PF,
    _threshold_value,
)
from linear_probe import _protein_groups, out_of_fold_probabilities
from run_layer_benchmark import load_benchmark_matrices

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger("boot")

N_BOOT = 2000
BOOT_SEED = 0
CI_PCTS = (2.5, 97.5)


def per_protein_stats(
    pred: np.ndarray, label_col: np.ndarray, offsets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-protein sufficient statistics for the asymmetric F1.

    The F1 recipe is per-residue precision, per-domain recall (a "domain" is a maximal
    contiguous run of positive labels within a protein). Both numerator and denominator are
    sums over proteins, so a cluster bootstrap only needs four per-protein totals.

    Args:
        pred: ``(N,)`` boolean prediction mask at the fixed operating point.
        label_col: ``(N,)`` binary labels for one concept.
        offsets: ``(P + 1,)`` protein boundary offsets.

    Returns:
        ``(tp_res, predpos_res, hit_dom, total_dom)`` each ``(P,)``:
        per-residue true positives and predicted positives (precision), and
        per-protein hit / total domain counts (recall).
    """
    pred = pred.astype(bool)
    label_col = label_col.astype(bool)
    n_prot = len(offsets) - 1
    tp_res = np.zeros(n_prot, dtype=np.float64)
    predpos_res = np.zeros(n_prot, dtype=np.float64)
    hit_dom = np.zeros(n_prot, dtype=np.float64)
    total_dom = np.zeros(n_prot, dtype=np.float64)
    for p in range(n_prot):
        s, e = int(offsets[p]), int(offsets[p + 1])
        yp, pp = label_col[s:e], pred[s:e]
        predpos_res[p] = pp.sum()
        tp_res[p] = (pp & yp).sum()
        if not yp.any():
            continue
        # contiguous positive runs within this protein
        starts = np.flatnonzero(yp & ~np.concatenate(([False], yp[:-1])))
        ends = np.flatnonzero(yp & ~np.concatenate((yp[1:], [False]))) + 1
        total_dom[p] = len(starts)
        hit_dom[p] = sum(1 for a, b in zip(starts, ends) if pp[a:b].any())
    return tp_res, predpos_res, hit_dom, total_dom


def _f1(prec: np.ndarray, rec: np.ndarray) -> np.ndarray:
    denom = prec + rec
    return np.where(denom > 0, 2.0 * prec * rec / denom, 0.0)


def bootstrap_ci(
    pred: np.ndarray, label_col: np.ndarray, offsets: np.ndarray,
    n_boot: int = N_BOOT, seed: int = BOOT_SEED,
) -> dict:
    """Cluster-bootstrap precision / recall / F1 CIs at a fixed operating point.

    Resamples proteins with replacement (multinomial counts) and recomputes the pooled
    per-residue precision and per-domain recall from per-protein sufficient statistics.

    Returns a dict with point estimates, lo/hi CI bounds, and the F1 bootstrap SE.
    """
    tp_res, predpos_res, hit_dom, total_dom = per_protein_stats(pred, label_col, offsets)
    n_prot = len(tp_res)

    def _point(tp, pp, hd, td):
        prec = tp.sum() / pp.sum() if pp.sum() > 0 else 0.0
        rec = hd.sum() / td.sum() if td.sum() > 0 else 0.0
        return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)

    p0, r0, f0 = _point(tp_res, predpos_res, hit_dom, total_dom)

    rng = np.random.default_rng(seed)
    # (n_boot, n_prot) multinomial counts: each row is one resample of n_prot proteins.
    counts = rng.multinomial(n_prot, np.full(n_prot, 1.0 / n_prot), size=n_boot).astype(np.float64)
    num_p = counts @ tp_res
    den_p = counts @ predpos_res
    num_r = counts @ hit_dom
    den_r = counts @ total_dom
    with np.errstate(divide="ignore", invalid="ignore"):
        prec_b = np.where(den_p > 0, num_p / den_p, 0.0)
        rec_b = np.where(den_r > 0, num_r / den_r, 0.0)
    f1_b = _f1(prec_b, rec_b)

    p_lo, p_hi = np.percentile(prec_b, CI_PCTS)
    r_lo, r_hi = np.percentile(rec_b, CI_PCTS)
    f_lo, f_hi = np.percentile(f1_b, CI_PCTS)
    # f1_mean / f1_bias are a DIAGNOSTIC only. We report the full-sample point estimate `f1`, not the
    # bootstrap mean: F1 is a harmonic mean of ratios-of-sums, so its bootstrap distribution is biased
    # (bias = point - bootstrap_mean) and right-skewed at small n. The point estimate keeps the reported
    # value consistent with the *_agg.json headline F1; f1_bias just confirms the skew is small.
    f1_mean = float(f1_b.mean())
    return dict(
        precision=p0, precision_lo=p_lo, precision_hi=p_hi,
        recall=r0, recall_lo=r_lo, recall_hi=r_hi,
        f1=f0, f1_lo=f_lo, f1_hi=f_hi, f1_se=float(f1_b.std(ddof=1)),
        f1_boot_mean=f1_mean, f1_bias=f0 - f1_mean,
    )


def _domain_count(label_col: np.ndarray, offsets: np.ndarray) -> int:
    """Total positive domains for a concept (for reporting alongside the CI)."""
    _, _, _, total_dom = per_protein_stats(np.zeros_like(label_col, dtype=bool), label_col, offsets)
    return int(total_dom.sum())


def unsup_pred(fm, labels, offsets, concept_list, c):
    """Fixed boolean prediction vector for the best single feature (SAE / neuron) of concept ``c``."""
    thr = precompute_feature_thresholds(fm)
    _, best_feat, best_thr = score_best_f1_per_concept(fm, thr, labels, offsets)
    ci = concept_list.index(c)
    feat, pct = int(best_feat[ci]), float(best_thr[ci])
    if feat < 0:
        return None
    tval = _threshold_value(thr, feat, pct)
    return fm[:, feat] >= np.float32(tval)


def probe_pred(fm, labels, offsets, concept_list, c):
    """Fixed boolean prediction vector for the held-out linear probe of concept ``c``."""
    ci = concept_list.index(c)
    y = labels[:, ci].astype(np.int64)
    if int(y.sum()) < MIN_POSITIVE_RESIDUES:
        return None
    groups = _protein_groups(offsets, fm.shape[0])
    alive = fm.std(axis=0) > 0.0
    oof = out_of_fold_probabilities(fm[:, alive], y, groups, n_splits=5, c_reg=1.0, max_iter=2000)
    prob = oof[:, None].astype(np.float32)
    thr = precompute_feature_thresholds(prob)
    _, _, best_thr = score_best_f1_per_concept(prob, thr, labels[:, ci:ci + 1], offsets)
    tval = _threshold_value(thr, 0, float(best_thr[0]))
    return prob[:, 0] >= np.float32(tval)


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
            for c in KEEP[cset]:
                if c not in m.concept_list:
                    continue
                ci = m.concept_list.index(c)
                y = m.label_matrix[:, ci]
                if int(y.sum()) < MIN_POSITIVE_RESIDUES:
                    continue
                pos_dom = _domain_count(y, m.protein_offsets)
                preds = {
                    "sae": unsup_pred(m.sae_matrix, m.label_matrix, m.protein_offsets, m.concept_list, c),
                    "neuron": unsup_pred(m.raw_matrix, m.label_matrix, m.protein_offsets, m.concept_list, c),
                    "probe_raw": probe_pred(m.raw_matrix, m.label_matrix, m.protein_offsets, m.concept_list, c),
                }
                for method, pred in preds.items():
                    if pred is None:
                        continue
                    ci_stats = bootstrap_ci(pred, y, m.protein_offsets)
                    rows.append(dict(
                        stack=stack, rec=rec, layer=layer, concept_set=cset, concept=c,
                        method=method, n_proteins=len(m.used_ids), pos_domains=pos_dom,
                        pos_residues=int(y.sum()),
                        **{k: round(float(v), 5) for k, v in ci_stats.items()},
                        n_boot=N_BOOT, boot_seed=BOOT_SEED,
                    ))
                LOG.info("  %s: pos_domains=%d done", c, pos_dom)
    fields = ["stack", "rec", "layer", "concept_set", "concept", "method", "n_proteins",
              "pos_domains", "pos_residues", "precision", "precision_lo", "precision_hi",
              "recall", "recall_lo", "recall_hi", "f1", "f1_lo", "f1_hi", "f1_se",
              "f1_boot_mean", "f1_bias", "n_boot", "boot_seed"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    LOG.info("wrote %d rows -> %s", len(rows), out_csv)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "pf"
    mp = int(sys.argv[2]) if len(sys.argv) > 2 else 250
    specs = {"pf": SPECS_PF, "diff": SPECS_DIFF, "all": SPECS_PF + SPECS_DIFF}[which]
    main(specs, "sae_explore/concept_f1_ci.csv", max_proteins=mp)
