# embedding_concept_evaluation.py
"""
Embedding-Concept F1 Evaluation Toolkit (optimized)
========================================
Compute Precision, Recall, Domain Recall, and F1 scores between
protein embeddings and SwissProt concept annotations.

This optimized version precomputes percentiles per feature and
vectorizes per-threshold comparisons across all concepts at once,
dramatically reducing repeated work.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    thresholds: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    min_positive_residues: int = 5  # Min residues with annotation to evaluate concept
    min_domains: int = 1  # Min domains to evaluate concept
    use_percentile_thresholds: bool = True  # Use percentile-based thresholds
    percentile_thresholds: List[float] = field(default_factory=lambda: [50, 60, 70, 80, 90, 95, 99])
    use_feature_presence: bool = False  # For TopK SAEs, use activation > threshold as presence
    feature_presence_threshold: float = 0.0
    use_positive_activation_percentiles: bool = False
    positive_activation_percentile_thresholds: List[float] = field(
        default_factory=lambda: [50, 75, 90, 95, 99]
    )
    batch_size: int = 100  # Proteins to process at once (unused in this optimized version)
    n_jobs: int = 1  # Parallel jobs (1 = sequential) (not used)

@dataclass
class ProteinData:
    """Container for matched protein data."""
    protein_id: str
    embeddings: np.ndarray  # Shape: [seq_len, embedding_dim]
    annotations: np.ndarray  # Shape: [seq_len, n_concepts]
    sequence_length: int

    def validate(self) -> bool:
        """Check that embeddings and annotations have matching sequence length."""
        return self.embeddings.shape[0] == self.annotations.shape[0] == self.sequence_length

@dataclass
class ConceptMetrics:
    """Metrics for a single concept at a specific threshold."""
    concept_name: str
    concept_idx: int
    feature_idx: int
    threshold: float
    threshold_pct: float  # Percentile threshold

    # Per-residue metrics
    tp: int = 0  # True positives
    fp: int = 0  # False positives
    fn: int = 0  # False negatives
    tn: int = 0  # True negatives

    # Per-domain metrics
    tp_domains: int = 0  # Domains with any true positive
    total_domains: int = 0  # Total annotated domains

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def recall_per_domain(self) -> float:
        if self.total_domains == 0:
            return 0.0
        return self.tp_domains / self.total_domains

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def f1_per_domain(self) -> float:
        """F1 using domain-level recall (as in InterPLM)."""
        p, r = self.precision, self.recall_per_domain
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def to_dict(self) -> Dict:
        return {
            'concept': self.concept_name,
            'concept_idx': self.concept_idx,
            'feature_idx': self.feature_idx,
            'threshold': self.threshold,
            'threshold_pct': self.threshold_pct,
            'tp': self.tp,
            'fp': self.fp,
            'fn': self.fn,
            'tn': self.tn,
            'tp_domains': self.tp_domains,
            'total_domains': self.total_domains,
            'precision': self.precision,
            'recall': self.recall,
            'recall_per_domain': self.recall_per_domain,
            'f1': self.f1,
            'f1_per_domain': self.f1_per_domain
        }

# ============================================================================
# EMBEDDING LOADERS
# ============================================================================

class EmbeddingLoader(ABC):
    """Abstract base class for loading embeddings."""

    @abstractmethod
    def load(self, protein_id: str) -> Optional[np.ndarray]:
        """Load embeddings for a protein. Returns None if not found."""
        pass

    @abstractmethod
    def list_available(self) -> List[str]:
        """List all available protein IDs."""
        pass

class NPZEmbeddingLoader(EmbeddingLoader):
    """Load embeddings from NPZ files organized by protein ID folders."""

    def __init__(
        self,
        base_dir: str,
        npz_filename: str = "output_3.npz",
        array_key: str = "arr_0",
        subfolder_pattern: Optional[str] = None,
        squeeze_batch_dim: bool = True
    ):
        """
        Args:
            base_dir: Base directory containing protein folders
            npz_filename: Name of NPZ file within each protein folder
            array_key: Key to access array within NPZ file
            subfolder_pattern: Optional subfolder pattern (e.g., "PairformerLayer_20 s from PairformerLayer")
            squeeze_batch_dim: If True, squeeze batch dimension from shape (1, L, D) to (L, D)
        """
        self.base_dir = Path(base_dir)
        self.npz_filename = npz_filename
        self.array_key = array_key
        self.subfolder_pattern = subfolder_pattern
        self.squeeze_batch_dim = squeeze_batch_dim

        # Cache of available proteins
        self._available_proteins = None

    def _get_npz_path(self, protein_id: str) -> Path:
        """Construct path to NPZ file for a protein."""
        if self.subfolder_pattern:
            return self.base_dir / protein_id / self.subfolder_pattern / self.npz_filename
        return self.base_dir / protein_id / self.npz_filename

    def load(self, protein_id: str) -> Optional[np.ndarray]:
        """Load embeddings for a protein."""
        npz_path = self._get_npz_path(protein_id)

        if not npz_path.exists():
            logger.debug(f"Embedding file not found: {npz_path}")
            return None

        try:
            data = np.load(npz_path)
            embeddings = data[self.array_key]

            # Squeeze batch dimension if present
            if self.squeeze_batch_dim and embeddings.ndim == 3 and embeddings.shape[0] == 1:
                embeddings = embeddings.squeeze(0)

            return embeddings

        except Exception as e:
            logger.error(f"Error loading embeddings for {protein_id}: {e}")
            return None

    def list_available(self) -> List[str]:
        """List all available protein IDs."""
        if self._available_proteins is not None:
            return self._available_proteins

        available = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                protein_id = item.name
                if self._get_npz_path(protein_id).exists():
                    available.append(protein_id)

        self._available_proteins = available
        return available

class SingleNPZLoader(EmbeddingLoader):
    """Load embeddings from a single NPZ file with protein IDs as keys."""

    def __init__(self, npz_path: str):
        self.npz_path = Path(npz_path)
        self._data = None

    def _ensure_loaded(self):
        if self._data is None:
            self._data = dict(np.load(self.npz_path, allow_pickle=True))

    def load(self, protein_id: str) -> Optional[np.ndarray]:
        self._ensure_loaded()
        return self._data.get(protein_id)

    def list_available(self) -> List[str]:
        self._ensure_loaded()
        return list(self._data.keys())

# ============================================================================
# ANNOTATION LOADER
# ============================================================================

class AnnotationLoader:
    """Load annotations from processed SwissProt data."""

    def __init__(self, processed_dir: str):
        self.processed_dir = Path(processed_dir)
        self._annotations = {}
        self._concept_list = None
        self._load_all()

    def _load_all(self):
        """Load all annotations from all shards."""
        # Load concept vocabulary
        vocab_path = self.processed_dir / "concept_vocabulary.json"
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        self._concept_list = vocab['concepts']

        # Load all shards
        shard_dirs = sorted([d for d in self.processed_dir.iterdir()
                           if d.is_dir() and d.name.startswith('shard_')])

        for shard_dir in shard_dirs:
            npz_path = shard_dir / "annotations.npz"
            if npz_path.exists():
                data = np.load(npz_path, allow_pickle=True)
                for key in data.files:
                    self._annotations[key] = data[key]

        logger.info(f"Loaded annotations for {len(self._annotations)} proteins, "
                   f"{len(self._concept_list)} concepts")

    def load(self, protein_id: str) -> Optional[np.ndarray]:
        """Load annotations for a protein."""
        return self._annotations.get(protein_id)

    def list_available(self) -> List[str]:
        """List all available protein IDs."""
        return list(self._annotations.keys())

    @property
    def concept_list(self) -> List[str]:
        return self._concept_list

    @property
    def n_concepts(self) -> int:
        return len(self._concept_list)

# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================

def find_domains(binary_vector: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find contiguous domains (regions of 1s) in a binary vector.

    Returns:
        List of (start, end) tuples (0-indexed, inclusive)
    """
    domains = []
    in_domain = False
    start = 0

    for i, val in enumerate(binary_vector):
        if val == 1 and not in_domain:
            start = i
            in_domain = True
        elif val == 0 and in_domain:
            domains.append((start, i - 1))
            in_domain = False

    if in_domain:
        domains.append((start, len(binary_vector) - 1))

    return domains

# ============================================================================
# MAIN EVALUATOR CLASS (optimized evaluate_all)
# ============================================================================

class ConceptEvaluator:
    """
    Evaluate embeddings against concept annotations.

    Computes precision, recall, domain recall, and F1 scores
    for all embedding dimension - concept pairs.
    """

    def __init__(
        self,
        embedding_loader: EmbeddingLoader,
        annotation_loader: AnnotationLoader,
        config: Optional[EvaluationConfig] = None
    ):
        self.embedding_loader = embedding_loader
        self.annotation_loader = annotation_loader
        self.config = config or EvaluationConfig()

        # Find matching proteins
        emb_proteins = set(embedding_loader.list_available())
        ann_proteins = set(annotation_loader.list_available())
        self.matched_proteins = sorted(emb_proteins & ann_proteins)

        logger.info(f"Found {len(self.matched_proteins)} proteins with both "
                   f"embeddings and annotations")
        logger.info(f"  Embeddings only: {len(emb_proteins - ann_proteins)}")
        logger.info(f"  Annotations only: {len(ann_proteins - emb_proteins)}")

    def load_protein_data(self, protein_id: str) -> Optional[ProteinData]:
        """Load and validate data for a single protein."""
        embeddings = self.embedding_loader.load(protein_id)
        annotations = self.annotation_loader.load(protein_id)

        if embeddings is None or annotations is None:
            return None

        # Check dimension match
        if embeddings.shape[0] != annotations.shape[0]:
            logger.warning(f"Dimension mismatch for {protein_id}: "
                          f"embeddings={embeddings.shape[0]}, "
                          f"annotations={annotations.shape[0]}")
            return None

        protein_data = ProteinData(
            protein_id=protein_id,
            embeddings=embeddings,
            annotations=annotations,
            sequence_length=embeddings.shape[0]
        )

        if not protein_data.validate():
            logger.warning(f"Validation failed for {protein_id}")
            return None

        return protein_data

    def aggregate_activations_and_annotations(
        self,
        protein_ids: Optional[List[str]] = None,
        feature_indices: Optional[List[int]] = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, List]]:
        """
        Aggregate activations and annotations across proteins for each concept.

        Args:
            protein_ids: Optional protein IDs to evaluate. Defaults to all matched proteins.
            feature_indices: Optional embedding/latent feature indices to keep. If omitted,
                all features are aggregated.

        Returns:
            feature_activations: {feature_idx: concatenated activations}
            concept_annotations: {concept_idx: concatenated annotations}
            concept_domains: {concept_idx: list of (protein_id, start, end) tuples}
        """
        if protein_ids is None:
            protein_ids = self.matched_proteins

        selected_feature_indices = None
        if feature_indices is not None:
            selected_feature_indices = sorted({int(feature_idx) for feature_idx in feature_indices})
            if len(selected_feature_indices) == 0:
                raise ValueError("feature_indices must contain at least one feature.")

        n_concepts = self.annotation_loader.n_concepts

        # Initialize storage
        feature_activations = defaultdict(list)
        concept_annotations = defaultdict(list)
        concept_domains = defaultdict(list)

        n_loaded = 0
        n_skipped = 0
        checked_feature_indices = False

        for protein_id in protein_ids:
            protein_data = self.load_protein_data(protein_id)

            if protein_data is None:
                n_skipped += 1
                continue

            n_loaded += 1

            # Store activations for each embedding dimension (feature)
            n_features = protein_data.embeddings.shape[1]
            if selected_feature_indices is None:
                current_feature_indices = range(n_features)
            else:
                if not checked_feature_indices:
                    invalid_features = [
                        feature_idx
                        for feature_idx in selected_feature_indices
                        if feature_idx < 0 or feature_idx >= n_features
                    ]
                    if invalid_features:
                        raise ValueError(
                            f"feature_indices out of range for {n_features} features: "
                            f"{invalid_features[:10]}"
                        )
                    checked_feature_indices = True
                current_feature_indices = selected_feature_indices

            for feat_idx in current_feature_indices:
                feature_activations[feat_idx].append(
                    protein_data.embeddings[:, feat_idx]
                )

            # Store annotations for each concept
            for concept_idx in range(n_concepts):
                concept_annotations[concept_idx].append(
                    protein_data.annotations[:, concept_idx]
                )

                # Track domains (relative to concatenated coordinate; we will stitch later)
                domains = find_domains(protein_data.annotations[:, concept_idx])
                # store domains as per-protein (protein_id, start, end)
                for start, end in domains:
                    concept_domains[concept_idx].append((protein_id, start, end))

        logger.info(f"Loaded {n_loaded} proteins, skipped {n_skipped}")

        # Concatenate
        for feat_idx in feature_activations:
            feature_activations[feat_idx] = np.concatenate(feature_activations[feat_idx])

        for concept_idx in concept_annotations:
            concept_annotations[concept_idx] = np.concatenate(concept_annotations[concept_idx])

        return dict(feature_activations), dict(concept_annotations), dict(concept_domains)

    def evaluate_feature_concept_pair(
        self,
        feature_activations: np.ndarray,
        concept_annotations: np.ndarray,
        feature_idx: int,
        concept_idx: int,
        concept_name: str
    ) -> List[ConceptMetrics]:
        """
        Legacy single-pair evaluator (kept for reference). Not used by optimized evaluate_all.
        """
        results = []

        # Skip if no positive annotations
        n_positive = concept_annotations.sum()
        if n_positive < self.config.min_positive_residues:
            return results

        thresholds = self.config.percentile_thresholds if self.config.use_percentile_thresholds \
                    else self.config.thresholds

        for thresh in thresholds:
            tp, fp, fn, tn, tp_domains, total_domains = compute_metrics_at_threshold(
                feature_activations,
                concept_annotations,
                thresh,
                use_percentile=self.config.use_percentile_thresholds
            )

            metrics = ConceptMetrics(
                concept_name=concept_name,
                concept_idx=concept_idx,
                feature_idx=feature_idx,
                threshold=np.percentile(feature_activations, thresh) if self.config.use_percentile_thresholds else thresh,
                threshold_pct=thresh if self.config.use_percentile_thresholds else thresh * 100,
                tp=int(tp),
                fp=int(fp),
                fn=int(fn),
                tn=int(tn),
                tp_domains=int(tp_domains),
                total_domains=int(total_domains)
            )
            results.append(metrics)

        return results

    def evaluate_all(
        self,
        protein_ids: Optional[List[str]] = None,
        max_features: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
        feature_indices: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Evaluate all feature-concept pairs (optimized).

        Args:
            protein_ids: Optional protein IDs to evaluate. Defaults to all matched proteins.
            max_features: Optional cap on the number of features to evaluate after filtering.
            progress_callback: Optional callback invoked periodically with progress counts.
            feature_indices: Optional embedding/latent feature indices to evaluate.

        Vectorizes across concepts for each feature+threshold:
          - Precompute percentiles per feature once
          - Stack concept annotations into (L, C) uint8 matrix
          - Compute pred once per feature+threshold
          - Compute tp/fp/fn/tn through vectorized reductions
          - Compute domain recall per concept using precomputed domain boundaries
        """
        logger.info("Aggregating activations and annotations...")
        feature_acts, concept_anns, concept_domains = self.aggregate_activations_and_annotations(
            protein_ids=protein_ids,
            feature_indices=feature_indices,
        )

        n_concepts = len(concept_anns)
        concept_list = self.annotation_loader.concept_list

        # Prepare features iteration order, preserving explicit non-contiguous feature IDs.
        features_to_evaluate = sorted(feature_acts.keys())
        if max_features:
            features_to_evaluate = features_to_evaluate[:max_features]

        logger.info(f"Evaluating {len(features_to_evaluate)} features x {n_concepts} concepts...")

        # Build concept matrix: shape (L, C) as uint8/bool to save memory
        # Ensure consistent ordering: concept indices 0 .. n_concepts-1
        concept_order = list(range(n_concepts))
        # If any concept index missing in dictionary, fill with zeros (shouldn't happen)
        example_lengths = [arr.shape[0] for arr in concept_anns.values()]
        if len(example_lengths) == 0:
            logger.warning("No concept annotations found. Exiting.")
            return pd.DataFrame()

        # Verify all annotation lengths match
        L = None
        for cidx in concept_order:
            arr = concept_anns.get(cidx)
            if arr is None:
                raise ValueError(f"Concept {cidx} missing in concept_anns.")
            if L is None:
                L = arr.shape[0]
            elif arr.shape[0] != L:
                raise ValueError("Mismatched concatenated lengths between concepts.")

        # Stack into matrix of dtype uint8 (0/1)
        logger.info(f"Building concept annotation matrix of shape ({L}, {n_concepts})")
        concept_matrix = np.stack([concept_anns[cidx].astype(np.uint8) for cidx in concept_order], axis=1)  # (L, C)

        # Precompute per-concept domains (on concatenated coordinates)
        # convert concept_matrix columns to bool for domain finding
        concept_domains_list: List[List[Tuple[int, int]]] = []
        concept_total_domains: List[int] = []
        for cidx in concept_order:
            col = concept_matrix[:, cidx].astype(np.uint8)
            domains = find_domains(col)
            concept_domains_list.append(domains)
            concept_total_domains.append(len(domains))

        if self.config.use_feature_presence and self.config.use_positive_activation_percentiles:
            raise ValueError(
                "use_feature_presence and use_positive_activation_percentiles are mutually exclusive."
            )

        if self.config.use_feature_presence:
            logger.info(
                f"Using feature-presence threshold: activation > "
                f"{self.config.feature_presence_threshold}"
            )
            thresholds = [self.config.feature_presence_threshold]
        elif self.config.use_positive_activation_percentiles:
            logger.info(
                f"Using positive-activation percentiles: "
                f"{self.config.positive_activation_percentile_thresholds}"
            )
            thresholds = self.config.positive_activation_percentile_thresholds
        else:
            thresholds = self.config.percentile_thresholds if self.config.use_percentile_thresholds \
                        else self.config.thresholds

        all_results = []
        total_pairs = len(features_to_evaluate) * n_concepts
        current = 0

        # To reduce memory traffic use uint8 for pred and concept_matrix (already uint8)
        for feat_idx in features_to_evaluate:
            activ = feature_acts[feat_idx]  # shape (L,)
            if activ.shape[0] != L:
                logger.warning(f"Feature {feat_idx} activations length {activ.shape[0]} != expected {L}. Skipping.")
                continue

            # Compute absolute thresholds for this feature once.
            if self.config.use_feature_presence:
                abs_thresholds = np.array(thresholds)
            elif self.config.use_positive_activation_percentiles:
                positive_activ = activ[activ > self.config.feature_presence_threshold]
                if positive_activ.shape[0] == 0:
                    logger.warning(f"Feature {feat_idx} has no positive activations. Skipping.")
                    continue
                abs_thresholds = np.percentile(positive_activ, thresholds)
            elif self.config.use_percentile_thresholds:
                # thresholds are percentiles 0-100
                abs_thresholds = np.percentile(activ, thresholds)
            else:
                abs_thresholds = np.array(thresholds)

            for t_i, thresh_val in enumerate(abs_thresholds):
                # Compute prediction mask once. TopK SAE presence should exclude zero activations.
                if self.config.use_feature_presence:
                    pred = (activ > thresh_val).astype(np.uint8)  # shape (L,)
                elif self.config.use_positive_activation_percentiles:
                    pred = (activ >= thresh_val).astype(np.uint8)  # shape (L,)
                else:
                    pred = (activ >= thresh_val).astype(np.uint8)  # shape (L,)
                # broadcast pred to (L,1) and compare to concept_matrix (L,C)
                # vectorized TP/FP/FN/TN across concepts
                pred_col = pred[:, None]  # (L,1)
                ann_mat = concept_matrix  # (L,C) uint8

                # Compute per-concept counts (axis=0 sums over positions)
                tp_per_concept = np.sum((pred_col == 1) & (ann_mat == 1), axis=0).astype(np.int64)
                fp_per_concept = np.sum((pred_col == 1) & (ann_mat == 0), axis=0).astype(np.int64)
                fn_per_concept = np.sum((pred_col == 0) & (ann_mat == 1), axis=0).astype(np.int64)
                tn_per_concept = np.sum((pred_col == 0) & (ann_mat == 0), axis=0).astype(np.int64)

                # Domain-level true positives: for each concept, check each domain if any pred in domain
                # This loops per-domain but domains << L so it's cheap comparatively.
                tp_domains_per_concept = np.zeros(n_concepts, dtype=np.int64)
                for cidx in range(n_concepts):
                    domains = concept_domains_list[cidx]
                    if len(domains) == 0:
                        tp_domains_per_concept[cidx] = 0
                        continue
                    # For each domain check any(pred[start:end+1])
                    count = 0
                    for (s, e) in domains:
                        # slice is small relative to L; using np.any on uint8 is fast
                        if np.any(pred[s:e+1] == 1):
                            count += 1
                    tp_domains_per_concept[cidx] = count

                # Now build ConceptMetrics for each concept
                for cidx in range(n_concepts):
                    # skip concepts with too few positives
                    total_pos = int(np.sum(concept_matrix[:, cidx] == 1))
                    if total_pos < self.config.min_positive_residues:
                        continue

                    metrics = ConceptMetrics(
                        concept_name=concept_list[cidx],
                        concept_idx=cidx,
                        feature_idx=feat_idx,
                        threshold=float(thresh_val),
                        threshold_pct=float(thresholds[t_i]) if self.config.use_percentile_thresholds else float(thresholds[t_i]),
                        tp=int(tp_per_concept[cidx]),
                        fp=int(fp_per_concept[cidx]),
                        fn=int(fn_per_concept[cidx]),
                        tn=int(tn_per_concept[cidx]),
                        tp_domains=int(tp_domains_per_concept[cidx]),
                        total_domains=int(concept_total_domains[cidx])
                    )
                    all_results.append(metrics.to_dict())

                    current += 1
                    if progress_callback and current % 10000 == 0:
                        progress_callback(current, total_pairs)

        logger.info(f"Evaluation complete. {len(all_results)} results.")

        return pd.DataFrame(all_results)

    def find_best_features_per_concept(
        self,
        results_df: pd.DataFrame,
        metric: str = 'f1_per_domain',
        top_k: int = 1
    ) -> pd.DataFrame:
        """
        Find the best feature(s) for each concept.

        Args:
            results_df: Results from evaluate_all()
            metric: Metric to optimize ('f1', 'f1_per_domain', 'precision', 'recall')
            top_k: Number of top features to return per concept

        Returns:
            DataFrame with best feature-concept pairs
        """
        # Group by concept and find best
        best_results = []

        if results_df.empty:
            return pd.DataFrame()

        for concept_name in results_df['concept'].unique():
            concept_df = results_df[results_df['concept'] == concept_name]

            # Find best threshold for each feature
            best_per_feature = concept_df.loc[concept_df.groupby('feature_idx')[metric].idxmax()]

            # Get top-k features
            top_features = best_per_feature.nlargest(top_k, metric)
            best_results.append(top_features)

        if not best_results:
            return pd.DataFrame()

        return pd.concat(best_results, ignore_index=True)

# ============================================================================
# CONVENIENCE FUNCTIONS (kept for compatibility; some are unused by optimized flow)
# ============================================================================

def compute_metrics_at_threshold(
    activations: np.ndarray,
    annotations: np.ndarray,
    threshold: float,
    use_percentile: bool = True
) -> Tuple[int, int, int, int, int, int]:
    """
    Compute TP, FP, FN, TN and domain metrics at a given threshold.

    (Retained for compatibility; optimized evaluate_all does not call this repeatedly)
    """
    if use_percentile:
        thresh_val = np.percentile(activations, threshold)
    else:
        thresh_val = threshold

    predictions = (activations >= thresh_val).astype(int)
    annotations = annotations.astype(int)

    tp = int(np.sum((predictions == 1) & (annotations == 1)))
    fp = int(np.sum((predictions == 1) & (annotations == 0)))
    fn = int(np.sum((predictions == 0) & (annotations == 1)))
    tn = int(np.sum((predictions == 0) & (annotations == 0)))

    domains = find_domains(annotations)
    total_domains = len(domains)
    tp_domains = 0
    for start, end in domains:
        if np.any(predictions[start:end+1] == 1):
            tp_domains += 1

    return tp, fp, fn, tn, tp_domains, total_domains

def compute_precision(tp: int, fp: int) -> float:
    """Compute precision from TP and FP counts."""
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def compute_recall(tp: int, fn: int) -> float:
    """Compute recall from TP and FN counts."""
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def compute_domain_recall(tp_domains: int, total_domains: int) -> float:
    """Compute domain-level recall."""
    if total_domains == 0:
        return 0.0
    return tp_domains / total_domains

def compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# convert numpy scalars / pandas NaN to native Python types for json
def _to_native(o):
    # numpy integers, floats, bools
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        # convert nan -> None
        if np.isnan(o):
            return None
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    # pandas NA / numpy nan
    try:
        if pd.isna(o):
            return None
    except Exception:
        pass
    return o


# ============================================================================
# RUN EVALUATION (same API)
# ============================================================================

def run_evaluation(
    embeddings_dir: str,
    annotations_dir: str,
    output_dir: str,
    npz_filename: str = "output_3.npz",
    array_key: str = "arr_0",
    subfolder_pattern: Optional[str] = None,
    config: Optional[EvaluationConfig] = None,
    max_features: Optional[int] = None
) -> pd.DataFrame:
    """
    Run complete evaluation pipeline (optimized).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize loaders
    embedding_loader = NPZEmbeddingLoader(
        base_dir=embeddings_dir,
        npz_filename=npz_filename,
        array_key=array_key,
        subfolder_pattern=subfolder_pattern
    )

    annotation_loader = AnnotationLoader(annotations_dir)

    # Create evaluator
    evaluator = ConceptEvaluator(
        embedding_loader=embedding_loader,
        annotation_loader=annotation_loader,
        config=config
    )

    # Run evaluation
    def progress(current, total):
        if current % 10000 == 0:
            logger.info(f"Progress: {current}/{total} ({100*current/total:.1f}%)")

    results_df = evaluator.evaluate_all(
        max_features=max_features,
        progress_callback=progress
    )

    # Save full results
    results_df.to_csv(output_path / "all_results.csv", index=False)
    logger.info(f"Saved full results to {output_path / 'all_results.csv'}")

    # Find best features per concept
    best_df = evaluator.find_best_features_per_concept(results_df)
    best_df.to_csv(output_path / "best_features_per_concept.csv", index=False)
    logger.info(f"Saved best features to {output_path / 'best_features_per_concept.csv'}")

    # Summary statistics
    summary = {
        'n_proteins': len(evaluator.matched_proteins),
        'n_concepts': annotation_loader.n_concepts,
        'n_features_evaluated': results_df['feature_idx'].nunique() if not results_df.empty else 0,
        'n_concept_feature_pairs': len(results_df),
        'concepts_with_f1_gt_0.5': (best_df['f1_per_domain'] > 0.5).sum() if not best_df.empty else 0,
        'mean_best_f1': best_df['f1_per_domain'].mean() if not best_df.empty else 0.0,
        'max_f1': best_df['f1_per_domain'].max() if not best_df.empty else 0.0
    }

    summary_native = {k: _to_native(v) for k, v in summary.items()}
    
    with open(output_path / "summary.json", 'w') as f:
        json.dump(summary_native, f, indent=2)

    logger.info(f"\nSummary:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v}")

    return results_df

# ============================================================================
# MAIN (CLI compatibility)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate embeddings against SwissProt concepts (optimized)")

    parser.add_argument("--embeddings_dir", "-e", required=True,
                       help="Directory containing protein embedding folders")
    parser.add_argument("--annotations_dir", "-a", required=True,
                       help="Directory with processed SwissProt annotations")
    parser.add_argument("--output_dir", "-o", required=True,
                       help="Output directory for results")
    parser.add_argument("--npz_filename", default="output_3.npz",
                       help="Name of NPZ file in each protein folder")
    parser.add_argument("--array_key", default="arr_0",
                       help="Key for array in NPZ file")
    parser.add_argument("--subfolder", default=None,
                       help="Subfolder pattern within protein folders")
    parser.add_argument("--max_features", type=int, default=None,
                       help="Limit features for testing")

    args = parser.parse_args()

    run_evaluation(
        embeddings_dir=args.embeddings_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        npz_filename=args.npz_filename,
        array_key=args.array_key,
        subfolder_pattern=args.subfolder,
        max_features=args.max_features
    )
