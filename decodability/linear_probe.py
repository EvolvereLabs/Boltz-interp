#!/usr/bin/env python3
"""Supervised linear-probe baseline for the concept-F1 benchmark.

For each concept we train an L2-regularised logistic regression on residue-level
features to predict the binary concept label, using **grouped K-fold by protein**
so no residues from a protein appear in both train and test. Out-of-fold (OOF)
probabilities are then scored with the *same* per-domain F1 recipe used for the
SAE and single-neuron baselines (max over the positive-activation percentile
grid), making the probe directly comparable -- it is the supervised upper bound
on how linearly decodable a concept is from that layer.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from benchmark_f1 import (
    DEFAULT_THRESHOLD_PCTS,
    MIN_POSITIVE_RESIDUES,
    precompute_feature_thresholds,
    score_best_f1_per_concept,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def _protein_groups(protein_offsets: np.ndarray, n_residues: int) -> np.ndarray:
    """Return a ``(N,)`` array mapping each residue to its protein index."""
    groups = np.zeros(n_residues, dtype=np.int64)
    for p in range(len(protein_offsets) - 1):
        groups[protein_offsets[p] : protein_offsets[p + 1]] = p
    return groups


def out_of_fold_probabilities(
    feature_matrix: np.ndarray,
    label_column: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    c_reg: float,
    max_iter: int,
) -> np.ndarray:
    """Out-of-fold positive-class probabilities for one concept.

    Args:
        feature_matrix: ``(N, F)`` features.
        label_column: ``(N,)`` binary labels for one concept.
        groups: ``(N,)`` protein index per residue.
        n_splits: Number of grouped folds.
        c_reg: Inverse L2 regularisation strength.
        max_iter: Solver iteration cap.

    Returns:
        ``(N,)`` OOF probabilities (residues never held out stay 0).
    """
    oof = np.zeros(feature_matrix.shape[0], dtype=np.float64)
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        LOGGER.warning(msg="Not enough protein groups for cross-validation; skipping probe.")
        return oof

    splitter = GroupKFold(n_splits=splits)
    for train_idx, test_idx in splitter.split(feature_matrix, label_column, groups):
        if label_column[train_idx].sum() == 0:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(feature_matrix[train_idx])
        x_test = scaler.transform(feature_matrix[test_idx])
        model = LogisticRegression(
            class_weight="balanced",
            C=c_reg,
            max_iter=max_iter,
            solver="lbfgs",
        )
        model.fit(x_train, label_column[train_idx])
        oof[test_idx] = model.predict_proba(x_test)[:, 1]
    return oof


def probe_f1_per_concept(
    feature_matrix: np.ndarray,
    label_matrix: np.ndarray,
    protein_offsets: np.ndarray,
    concept_list: list[str],
    n_splits: int = 5,
    c_reg: float = 1.0,
    max_iter: int = 2000,
    threshold_pcts: tuple[float, ...] = DEFAULT_THRESHOLD_PCTS,
    min_positive: int = MIN_POSITIVE_RESIDUES,
    restrict_alive: bool = True,
) -> dict[str, float]:
    """Held-out per-domain F1 of a logistic probe for each concept.

    Args:
        feature_matrix: ``(N, F)`` features (raw neurons or SAE latents).
        label_matrix: ``(N, C)`` binary labels.
        protein_offsets: ``(P + 1,)`` protein boundary offsets.
        concept_list: Concept names indexed by column.
        n_splits: Grouped folds for cross-validation.
        c_reg: Inverse L2 regularisation strength.
        max_iter: Solver iteration cap.
        threshold_pcts: Percentile grid for scoring the OOF probabilities.
        min_positive: Concepts with fewer positive residues are skipped.
        restrict_alive: Drop zero-variance feature columns before fitting. A constant
            column is mapped to all-zeros by :class:`StandardScaler` and then driven to a
            zero coefficient by the L2 penalty, so it cannot affect any prediction -- the
            F1 is identical, but for a Top-K SAE this skips the (typically large) dead-latent
            population and the fit runs much faster. Variance, not ``>0``, is the right test:
            a raw neuron that is always negative is still informative once standardised.

    Returns:
        Mapping ``concept_name -> held-out per-domain F1``.
    """
    n_residues, n_concepts = label_matrix.shape
    if restrict_alive:
        alive = feature_matrix.std(axis=0) > 0.0
        n_alive = int(alive.sum())
        if n_alive < feature_matrix.shape[1]:
            LOGGER.info(
                msg=f"Probing on {n_alive}/{feature_matrix.shape[1]} non-constant features "
                f"(dropped {feature_matrix.shape[1] - n_alive} dead/constant)."
            )
            feature_matrix = feature_matrix[:, alive]
    groups = _protein_groups(protein_offsets, n_residues)
    pos_per_concept = label_matrix.sum(axis=0)

    probe_f1: dict[str, float] = {}
    for concept_idx in range(n_concepts):
        concept_name = concept_list[concept_idx]
        if pos_per_concept[concept_idx] < min_positive:
            continue
        LOGGER.info(msg=f"Training probe for concept '{concept_name}'")
        oof = out_of_fold_probabilities(
            feature_matrix,
            label_matrix[:, concept_idx].astype(np.int64),
            groups,
            n_splits=n_splits,
            c_reg=c_reg,
            max_iter=max_iter,
        )
        # Score the OOF probabilities with the identical per-domain F1 recipe.
        prob_feature = oof[:, None].astype(np.float32)
        label_column = label_matrix[:, concept_idx : concept_idx + 1]
        thresholds = precompute_feature_thresholds(prob_feature, threshold_pcts)
        best_f1, _, _ = score_best_f1_per_concept(
            prob_feature,
            thresholds,
            label_column,
            protein_offsets,
            threshold_pcts=threshold_pcts,
            min_positive=min_positive,
        )
        probe_f1[concept_name] = float(best_f1[0])
    return probe_f1
