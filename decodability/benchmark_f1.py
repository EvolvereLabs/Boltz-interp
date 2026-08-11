#!/usr/bin/env python3
"""Core scoring for the SAE-vs-neuron-vs-probe concept-F1 benchmark.

This module provides a single, fast, vectorised implementation of the
"best feature per concept" F1 statistic used throughout the layer sweep, plus a
label-permutation null for it. Using one scorer for the *observed* value and the
*null* guarantees the empirical p-values are comparable.

The F1 recipe mirrors :class:`embeddings_concepts_evaluation.ConceptEvaluator`
(per-residue precision, per-domain recall, positive-activation percentile
thresholds) with one deliberate fix: contiguous "domains" are detected **within
each protein** so a positive run cannot leak across a protein boundary.

Shapes used throughout:

* ``feature_matrix``  -- ``(N, F)`` float32, N = total residues, F = features
  (SAE latents, raw neurons, or a single supervised probe score).
* ``label_matrix``    -- ``(N, C)`` uint8, C = concepts.
* ``protein_offsets`` -- ``(P + 1,)`` int; protein ``p`` spans
  ``[offsets[p], offsets[p + 1])``.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# Same positive-activation percentile grid as the SAE F1 sweep.
DEFAULT_THRESHOLD_PCTS: tuple[float, ...] = (50.0, 75.0, 90.0, 95.0, 99.0)
MIN_POSITIVE_RESIDUES = 5
# Features are scored in column blocks so the dense prediction mask stays small:
# peak memory is ``(N, FEATURE_BLOCK)`` float32 instead of ``(N, n_features)``.
DEFAULT_FEATURE_BLOCK = 256


def precompute_feature_thresholds(
    feature_matrix: np.ndarray,
    threshold_pcts: tuple[float, ...] = DEFAULT_THRESHOLD_PCTS,
) -> np.ndarray:
    """Per-feature activation thresholds at each positive-activation percentile.

    Thresholds depend only on the activations (not the labels), so they are
    computed once and reused across every permutation.

    Args:
        feature_matrix: ``(N, F)`` activations.
        threshold_pcts: Percentiles applied to each feature's positive activations.

    Returns:
        ``(F, T)`` thresholds; rows for features that never fire are ``NaN``.
    """
    n_features = feature_matrix.shape[1]
    n_thresholds = len(threshold_pcts)
    thresholds = np.full((n_features, n_thresholds), np.nan, dtype=np.float32)
    pcts = np.asarray(threshold_pcts, dtype=np.float64)
    for feat_idx in range(n_features):
        activ = feature_matrix[:, feat_idx]
        positive = activ[activ > 0.0]
        if positive.size == 0:
            continue
        thresholds[feat_idx] = np.percentile(positive, pcts)
    return thresholds


class DomainStructure:
    """Precomputed per-concept domain structure for per-domain recall.

    Attributes:
        domain_csr: ``(n_domains, N)`` CSR matrix with 1s over each domain's residues.
        concept_selector: ``(C, n_domains)`` CSR matrix selecting a concept's domains
            (used to segment-sum domain hits into per-concept recall via one matmul).
        total_domains: ``(C,)`` per-concept domain count.
        too_few: ``(C,)`` bool mask of concepts with too few positive residues.
    """

    __slots__ = ("domain_csr", "concept_selector", "total_domains", "too_few")

    def __init__(
        self,
        domain_csr: csr_matrix,
        concept_selector: csr_matrix,
        total_domains: np.ndarray,
        too_few: np.ndarray,
    ) -> None:
        self.domain_csr = domain_csr
        self.concept_selector = concept_selector
        self.total_domains = total_domains
        self.too_few = too_few


def build_domain_structure(
    label_matrix: np.ndarray,
    protein_offsets: np.ndarray,
    min_positive: int = MIN_POSITIVE_RESIDUES,
) -> DomainStructure:
    """Build the per-concept :class:`DomainStructure`, respecting protein boundaries.

    A "domain" is a maximal contiguous run of 1s within a single protein.

    Args:
        label_matrix: ``(N, C)`` binary labels.
        protein_offsets: ``(P + 1,)`` protein boundary offsets.
        min_positive: Concepts with fewer positive residues are flagged in ``too_few``.

    Returns:
        A populated :class:`DomainStructure`.
    """
    n_residues, n_concepts = label_matrix.shape
    boundary = np.zeros(n_residues, dtype=bool)
    boundary[protein_offsets[:-1]] = True  # first residue of each protein

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    domain_concept: list[np.ndarray] = []
    total_domains = np.zeros(n_concepts, dtype=np.int64)
    running_count = 0

    for concept_idx in range(n_concepts):
        col = label_matrix[:, concept_idx].astype(bool)
        if not col.any():
            continue
        prev = np.empty(n_residues, dtype=bool)
        prev[0] = False
        prev[1:] = col[:-1]
        prev = prev & ~boundary  # treat each protein's first residue as a fresh start
        starts = col & ~prev
        domain_ordinal = np.cumsum(starts)  # 1-based ordinal within this concept
        pos_idx = np.nonzero(col)[0]
        n_concept_domains = int(domain_ordinal[pos_idx[-1]])
        rows.append(running_count + domain_ordinal[pos_idx] - 1)
        cols.append(pos_idx)
        domain_concept.append(np.full(n_concept_domains, concept_idx, dtype=np.int64))
        total_domains[concept_idx] = n_concept_domains
        running_count += n_concept_domains

    too_few = label_matrix.sum(axis=0) < min_positive
    if running_count == 0:
        empty = csr_matrix((0, n_residues), dtype=np.float32)
        selector = csr_matrix((n_concepts, 0), dtype=np.float32)
        return DomainStructure(empty, selector, total_domains, too_few)

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    domain_csr = csr_matrix(
        (np.ones(row_idx.shape[0], dtype=np.float32), (row_idx, col_idx)),
        shape=(running_count, n_residues),
    )
    domain_concept_arr = np.concatenate(domain_concept)
    concept_selector = csr_matrix(
        (
            np.ones(running_count, dtype=np.float32),
            (domain_concept_arr, np.arange(running_count)),
        ),
        shape=(n_concepts, running_count),
    )
    return DomainStructure(domain_csr, concept_selector, total_domains, too_few)


def _f1_matrix(
    pred: np.ndarray,
    pred_pos: np.ndarray,
    labels_f: np.ndarray,
    domains: DomainStructure,
) -> np.ndarray:
    """Per-feature, per-concept F1 for one prediction mask.

    Args:
        pred: ``(N, F)`` float32 prediction mask (1.0 where the feature fires).
        pred_pos: ``(F,)`` positives per feature (``pred.sum(0)``).
        labels_f: ``(N, C)`` float32 labels.
        domains: Precomputed domain structure for ``labels_f``.

    Returns:
        ``(F, C)`` F1 matrix (per-residue precision, per-domain recall).
    """
    n_features = pred.shape[1]
    n_concepts = labels_f.shape[1]
    tp = pred.T @ labels_f  # (F, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_pos[:, None] > 0, tp / pred_pos[:, None], 0.0)

    if domains.domain_csr.shape[0]:
        domain_hits = (domains.domain_csr @ pred) > 0  # (n_domains, F)
        tp_domains = domains.concept_selector @ domain_hits.astype(np.float32)  # (C, F)
    else:
        tp_domains = np.zeros((n_concepts, n_features), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(
            domains.total_domains[:, None] > 0, tp_domains / domains.total_domains[:, None], 0.0
        ).T  # (F, C)

    denom = precision + recall
    f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)
    f1[:, domains.too_few] = 0.0
    return f1


def _prediction_mask(feature_matrix: np.ndarray, threshold_col: np.ndarray) -> np.ndarray:
    """Float32 ``(N, F)`` mask of ``feature >= threshold`` (dead features never fire)."""
    safe_thr = np.where(np.isnan(threshold_col), np.inf, threshold_col).astype(np.float32)
    return (feature_matrix >= safe_thr[None, :]).astype(np.float32)


def score_best_f1_per_concept(
    feature_matrix: np.ndarray,
    thresholds: np.ndarray,
    label_matrix: np.ndarray,
    protein_offsets: np.ndarray,
    threshold_pcts: tuple[float, ...] = DEFAULT_THRESHOLD_PCTS,
    min_positive: int = MIN_POSITIVE_RESIDUES,
    feature_block: int = DEFAULT_FEATURE_BLOCK,
    return_per_feature: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best per-domain F1 for every concept, maximised over features and thresholds.

    Features are processed in column blocks so the dense prediction mask never
    exceeds ``(N, feature_block)`` -- this caps peak memory for wide dictionaries.

    Args:
        feature_matrix: ``(N, F)`` activations.
        thresholds: ``(F, T)`` from :func:`precompute_feature_thresholds`.
        label_matrix: ``(N, C)`` binary labels.
        protein_offsets: ``(P + 1,)`` protein boundary offsets.
        threshold_pcts: Percentile grid matching ``thresholds`` columns.
        min_positive: Concepts with fewer positive residues score 0.
        feature_block: Number of feature columns scored per block.
        return_per_feature: If True, also return the ``(F, C)`` matrix of each
            feature's best F1 over thresholds. The full block F1 is already
            computed below to find the argmax; retaining the per-feature maximum
            adds only an ``np.maximum`` accumulation, no extra scoring work. This
            is what the stability-vs-interpretability analysis needs (F1 for *all*
            latents, not just the per-concept winner).

    Returns:
        ``(best_f1, best_feature, best_threshold_pct)`` each of shape ``(C,)``.
        When ``return_per_feature`` is True, a 4th element ``per_feature_f1`` of
        shape ``(F, C)`` is appended.
    """
    n_concepts = label_matrix.shape[1]
    n_features = feature_matrix.shape[1]
    domains = build_domain_structure(label_matrix, protein_offsets, min_positive)
    labels_f = label_matrix.astype(np.float32)

    best_f1 = np.zeros(n_concepts, dtype=np.float64)
    best_feature = np.full(n_concepts, -1, dtype=np.int64)
    best_threshold = np.full(n_concepts, np.nan, dtype=np.float64)
    per_feature_f1 = (
        np.zeros((n_features, n_concepts), dtype=np.float64) if return_per_feature else None
    )

    for start in range(0, n_features, feature_block):
        end = min(start + feature_block, n_features)
        block = feature_matrix[:, start:end]
        for thr_i, thr_pct in enumerate(threshold_pcts):
            pred = _prediction_mask(block, thresholds[start:end, thr_i])
            f1 = _f1_matrix(pred, pred.sum(axis=0), labels_f, domains)  # (block, C)
            if per_feature_f1 is not None:
                np.maximum(per_feature_f1[start:end], f1, out=per_feature_f1[start:end])
            feat_argmax = np.argmax(f1, axis=0)  # (C,)
            feat_best = f1[feat_argmax, np.arange(n_concepts)]
            improved = feat_best > best_f1
            best_f1 = np.where(improved, feat_best, best_f1)
            best_feature = np.where(improved, start + feat_argmax, best_feature)
            best_threshold = np.where(improved, thr_pct, best_threshold)

    if return_per_feature:
        return best_f1, best_feature, best_threshold, per_feature_f1
    return best_f1, best_feature, best_threshold


def precision_recall_for_feature(
    feature_col: np.ndarray,
    threshold_value: float,
    label_column: np.ndarray,
    protein_offsets: np.ndarray,
    min_positive: int = MIN_POSITIVE_RESIDUES,
) -> tuple[float, float, float]:
    """Per-residue precision and per-domain recall for one feature at one threshold.

    Recomputes the two components of the asymmetric F1 used by
    :func:`score_best_f1_per_concept` for a *single* (feature, threshold, concept),
    so an earlier run's stored ``feature_idx`` / ``threshold_pct`` can be expanded
    into precision and recall without redoing the search. Returns the same F1 the
    search would have recorded, which doubles as an alignment check.

    Args:
        feature_col: ``(N,)`` activations of the chosen feature.
        threshold_value: Activation threshold (the value, not the percentile); the
            prediction mask is ``feature_col >= threshold_value`` to match the scorer.
        label_column: ``(N,)`` or ``(N, 1)`` binary labels for the concept.
        protein_offsets: ``(P + 1,)`` protein boundary offsets.
        min_positive: Concepts with fewer positive residues score 0 (matches scorer).

    Returns:
        ``(precision, recall, f1)`` -- per-residue precision, per-domain recall, F1.
    """
    label_col = np.asarray(label_column).reshape(-1).astype(np.uint8)
    if int(label_col.sum()) < min_positive:
        return 0.0, 0.0, 0.0
    pred = feature_col >= np.float32(threshold_value)  # (N,) bool, matches _prediction_mask
    n_pred = int(pred.sum())
    precision = float(np.sum(pred & label_col.astype(bool))) / n_pred if n_pred else 0.0

    # Per-domain recall: fraction of contiguous within-protein label runs the feature hits.
    domains = build_domain_structure(label_col[:, None], protein_offsets, min_positive)
    total = int(domains.total_domains[0])
    if total == 0 or domains.domain_csr.shape[0] == 0:
        recall = 0.0
    else:
        domain_hits = (domains.domain_csr @ pred.astype(np.float32)) > 0  # (n_domains,)
        recall = float(domain_hits.sum()) / total

    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom > 0 else 0.0
    return precision, recall, f1


def circular_shift_labels(
    label_matrix: np.ndarray,
    protein_offsets: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Circularly shift each protein's label block by a random offset.

    This breaks alignment between activations and labels while preserving each
    protein's concept prevalence, domain-length distribution, and the
    co-occurrence structure between concepts (the whole block is rolled together).

    Args:
        label_matrix: ``(N, C)`` binary labels.
        protein_offsets: ``(P + 1,)`` protein boundary offsets.
        rng: NumPy random generator.

    Returns:
        A shifted copy of ``label_matrix``.
    """
    shifted = label_matrix.copy()
    for p in range(len(protein_offsets) - 1):
        start, end = int(protein_offsets[p]), int(protein_offsets[p + 1])
        length = end - start
        if length > 1:
            k = int(rng.integers(1, length))
            shifted[start:end] = np.roll(label_matrix[start:end], k, axis=0)
    return shifted


def permutation_null(
    feature_matrix: np.ndarray,
    thresholds: np.ndarray,
    label_matrix: np.ndarray,
    protein_offsets: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    threshold_pcts: tuple[float, ...] = DEFAULT_THRESHOLD_PCTS,
    min_positive: int = MIN_POSITIVE_RESIDUES,
    feature_block: int = DEFAULT_FEATURE_BLOCK,
) -> np.ndarray:
    """Null distribution of the best-F1-over-features statistic per concept.

    Because each permutation re-takes the max over all features, the resulting
    null automatically accounts for the multiple-comparison selection across the
    feature dictionary (e.g. 2048 SAE latents vs 384 raw neurons).

    Args:
        feature_matrix: ``(N, F)`` activations (held fixed across permutations).
        thresholds: ``(F, T)`` precomputed feature thresholds.
        label_matrix: ``(N, C)`` binary labels.
        protein_offsets: ``(P + 1,)`` protein boundary offsets.
        n_perm: Number of label permutations.
        rng: NumPy random generator.
        threshold_pcts: Percentile grid matching ``thresholds`` columns.
        min_positive: Minimum positive residues to score a concept.
        feature_block: Number of feature columns scored per block (caps peak memory).

    Returns:
        ``(n_perm, C)`` array of null best-F1 values.
    """
    n_concepts = label_matrix.shape[1]
    n_features = feature_matrix.shape[1]
    if n_perm == 0:
        return np.zeros((0, n_concepts), dtype=np.float64)

    # Precompute each permutation's shuffled labels + domain structure once, so each
    # prediction mask (built per feature block × threshold) is reused across all perms.
    perm_labels: list[np.ndarray] = []
    perm_domains: list[DomainStructure] = []
    for _ in range(n_perm):
        shuffled = circular_shift_labels(label_matrix, protein_offsets, rng)
        perm_labels.append(shuffled.astype(np.float32))
        perm_domains.append(build_domain_structure(shuffled, protein_offsets, min_positive))

    null = np.zeros((n_perm, n_concepts), dtype=np.float64)
    n_blocks = (n_features + feature_block - 1) // feature_block
    for block_i, start in enumerate(range(0, n_features, feature_block)):
        end = min(start + feature_block, n_features)
        block = feature_matrix[:, start:end]
        for thr_i in range(len(threshold_pcts)):
            pred = _prediction_mask(block, thresholds[start:end, thr_i])
            pred_pos = pred.sum(axis=0)
            for perm_i in range(n_perm):
                f1 = _f1_matrix(pred, pred_pos, perm_labels[perm_i], perm_domains[perm_i])
                null[perm_i] = np.maximum(null[perm_i], f1.max(axis=0))
        LOGGER.info(msg=f"permutation null: feature block {block_i + 1}/{n_blocks} done")
    return null


def empirical_p_value(observed: float, null_samples: np.ndarray) -> float:
    """One-sided permutation p-value with the standard +1 correction."""
    n_perm = null_samples.shape[0]
    if n_perm == 0:
        return float("nan")
    n_ge = int(np.sum(null_samples >= observed))
    return (1 + n_ge) / (n_perm + 1)
