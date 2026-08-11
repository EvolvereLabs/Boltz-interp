#!/usr/bin/env python3
"""Augment existing benchmark JSONs in place -- without redoing the expensive parts.

The permutation null, the single-SAE-latent / single-neuron F1, and the raw linear
probe are already stored by ``run_layer_benchmark.py``. The only things missing from a
run made with ``--probe_on_sae False`` are (a) the SAE-latent probe upper bound and
(b) the precision/recall split behind each best-feature F1. Both can be added by
reloading *only* the activation matrices (a cheap SAE forward pass) and reusing the
indices the original run already chose -- no null, no neuron rescoring, no raw probe.

For each existing ``*_benchmark.json`` this script:

1. Rebuilds the raw + SAE + label matrices via ``load_benchmark_matrices`` over the
   *same* protein set, so the matrices line up residue-for-residue with the originals.
2. Trains the SAE-latent probe **on non-constant (alive) latents only** -- identical F1,
   far less compute -- and patches ``probe_sae_f1``.
3. Recomputes per-residue precision and per-domain recall for the stored best SAE latent
   and best neuron (using their saved ``feature_idx`` / ``threshold_pct``), and patches
   ``precision`` / ``recall`` next to each ``f1``. The recomputed F1 is checked against
   the stored F1 as an alignment guard.

The original ``f1`` / ``null`` / ``p_value`` / ``probe_raw_f1`` fields are never touched.

Example::

    uv run python augment_benchmark.py \\
        --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47 \\
        --recs 0,1 --concept_sets secondary,swissprot \\
        --activations_dir_template "downloads_layer{layer}" \\
        --protein_ids_file common_activation_proteins.txt
"""

import json
import logging
from pathlib import Path

import numpy as np
from tap import tapify

from benchmark_f1 import DEFAULT_THRESHOLD_PCTS, precision_recall_for_feature
from linear_probe import probe_f1_per_concept
from run_layer_benchmark import (
    DEFAULT_CONCEPT_SETS,
    cache_root_for,
    load_benchmark_matrices,
    output_root_for,
    rebuild_index,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# Stored F1 recomputed from the saved (feature, threshold) must match the original to
# within this tolerance; a larger gap means the rebuilt matrices are misaligned.
F1_ALIGN_TOL = 1e-4


def _parse_ints(value: str) -> list[int]:
    return [int(token) for token in str(value).replace(" ", "").split(",") if token]


def _threshold_value(feature_col: np.ndarray, pct: float) -> float:
    """Activation threshold at a positive-activation percentile (matches the scorer)."""
    positive = feature_col[feature_col > 0.0]
    if positive.size == 0:
        return float("nan")
    return float(np.percentile(positive, pct))


def _load_protein_subset(protein_ids_file: str | None) -> set[str] | None:
    """Parse the fixed protein-ID list exactly as ``run_layer_benchmark`` does."""
    if not protein_ids_file:
        return None
    path = Path(protein_ids_file)
    if not path.exists():
        raise FileNotFoundError(f"protein_ids_file '{protein_ids_file}' not found.")
    subset = {
        line.strip().rstrip("/").split("/")[-1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    LOGGER.info(msg=f"Restricting to {len(subset)} protein IDs from {protein_ids_file}")
    return subset


def _patch_precision_recall(
    entry: dict,
    method: str,
    feature_matrix: np.ndarray,
    label_column: np.ndarray,
    protein_offsets: np.ndarray,
) -> str | None:
    """Add precision/recall for one method's stored best feature; return a warning or None."""
    stats = entry.get(method)
    if not stats:
        return None
    feature_idx = stats.get("feature_idx")
    threshold_pct = stats.get("threshold_pct")
    if feature_idx is None or feature_idx < 0 or threshold_pct is None or np.isnan(threshold_pct):
        stats["precision"] = None
        stats["recall"] = None
        return None
    feature_col = feature_matrix[:, int(feature_idx)]
    thr_value = _threshold_value(feature_col, float(threshold_pct))
    precision, recall, f1 = precision_recall_for_feature(
        feature_col, thr_value, label_column, protein_offsets
    )
    stats["precision"] = precision
    stats["recall"] = recall
    stored_f1 = stats.get("f1")
    if stored_f1 is not None and abs(f1 - stored_f1) > F1_ALIGN_TOL:
        return f"{method} f1 mismatch (recomputed {f1:.5f} vs stored {stored_f1:.5f})"
    return None


def augment_layer_concept_set(
    json_path: Path,
    activations_dir: str,
    annotation_dir: str,
    protein_subset: set[str] | None,
    max_proteins: int,
    device: str | None,
    cache_dir: Path,
    recompute_probe_sae: bool,
    recompute_pr: bool,
) -> bool:
    """Augment one benchmark JSON in place. Returns True on success."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    layer = int(data["layer"])
    rec = int(data["rec"])
    seed = int(data["seed"])
    concept_set_name = data["concept_set"]

    matrices = load_benchmark_matrices(
        layer=layer,
        concept_set_name=concept_set_name,
        annotation_dir=annotation_dir,
        activations_dir=activations_dir,
        seed=seed,
        rec=rec,
        max_proteins=max_proteins,
        device=device,
        cache_dir=cache_dir,
        require_full_rec=False,
        protein_subset=protein_subset,
    )
    if matrices is None:
        LOGGER.error(msg=f"{json_path}: could not rebuild matrices; skipping.")
        return False

    # Guard: the rebuilt protein set must match what the original run scored, otherwise
    # the stored feature indices refer to a different residue ordering.
    if matrices.raw_matrix.shape[0] != data["n_residues"]:
        LOGGER.error(
            msg=(
                f"{json_path}: residue count changed ({matrices.raw_matrix.shape[0]} vs stored "
                f"{data['n_residues']}); activations differ from the original run. Skipping."
            )
        )
        return False

    concept_to_idx = {name: i for i, name in enumerate(matrices.concept_list)}
    warnings: list[str] = []

    if recompute_pr:
        for concept_name, entry in data["per_concept"].items():
            idx = concept_to_idx.get(concept_name)
            if idx is None:
                continue
            label_column = matrices.label_matrix[:, idx]
            for method, fmatrix in (("sae", matrices.sae_matrix), ("neuron", matrices.raw_matrix)):
                warn = _patch_precision_recall(
                    entry, method, fmatrix, label_column, matrices.protein_offsets
                )
                if warn:
                    warnings.append(f"{concept_name}: {warn}")

    if recompute_probe_sae:
        LOGGER.info(msg=f"{json_path.parent.name}/{concept_set_name}: SAE-latent probe (alive only)")
        probe_sae = probe_f1_per_concept(
            matrices.sae_matrix, matrices.label_matrix, matrices.protein_offsets, matrices.concept_list
        )
        for concept_name, entry in data["per_concept"].items():
            entry["probe_sae_f1"] = probe_sae.get(concept_name)

    if warnings:
        LOGGER.warning(msg=f"{json_path}: {len(warnings)} alignment warning(s): {warnings[:3]}")

    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    LOGGER.info(msg=f"Augmented {json_path}")
    return True


def augment_benchmark(
    layers: str = "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47",
    recs: str = "0,1",
    concept_sets: str = "secondary,swissprot",
    activations_dir_template: str = "downloads_layer{layer}",
    secondary_dir: str = DEFAULT_CONCEPT_SETS["secondary"],
    swissprot_dir: str = DEFAULT_CONCEPT_SETS["swissprot"],
    protein_ids_file: str | None = None,
    max_proteins: int = 0,
    device: str | None = None,
    recompute_probe_sae: bool = True,
    recompute_pr: bool = True,
    write_index: bool = True,
) -> int:
    """Add SAE-probe F1 and precision/recall to existing benchmark JSONs in place.

    Args:
        layers: Comma-separated pairformer layers to augment.
        recs: Comma-separated recycle indices to augment.
        concept_sets: Comma-separated sets from ``{secondary, swissprot}``.
        activations_dir_template: Activation dir; may contain ``{layer}``. Must point at
            the *same* activations the original run used, or the script aborts that file.
        secondary_dir: Directory for the secondary-structure labels.
        swissprot_dir: Directory for the SwissProt concept labels.
        protein_ids_file: Fixed protein-ID list -- pass the SAME file the original run used
            (e.g. ``common_activation_proteins.txt``) so the rebuilt matrices align.
        max_proteins: Cap on matched proteins (0 = all); match the original run's value.
        device: Optional torch device override for the SAE forward pass.
        recompute_probe_sae: Train + patch the SAE-latent probe (alive latents only).
        recompute_pr: Patch precision/recall for the stored best SAE latent and neuron.
        write_index: Rebuild ``benchmark_index.jsonl`` at the end. Set False when running
            many single-job processes in parallel (they'd race on the shared index);
            rebuild it once afterwards with ``rebuild_index(output_root_for(rec))``.

    Returns:
        Exit code; 0 if at least one JSON was augmented.
    """
    set_dirs = {"secondary": secondary_dir, "swissprot": swissprot_dir}
    requested = [s for s in concept_sets.replace(" ", "").split(",") if s]
    for name in requested:
        if name not in set_dirs:
            raise ValueError(f"Unknown concept set '{name}'. Choose from {sorted(set_dirs)}.")

    protein_subset = _load_protein_subset(protein_ids_file)
    rec_list = _parse_ints(recs)
    layer_list = _parse_ints(layers)
    augmented = 0

    for rec in rec_list:
        output_root = output_root_for(rec)
        cache_dir = cache_root_for(rec)
        for layer in layer_list:
            activations_dir = activations_dir_template.format(layer=layer)
            for name in requested:
                json_path = output_root / f"layer{layer}" / f"{name}_benchmark.json"
                if not json_path.exists():
                    LOGGER.info(msg=f"No JSON at {json_path}; nothing to augment.")
                    continue
                if not Path(activations_dir).exists():
                    LOGGER.warning(
                        msg=f"Activations '{activations_dir}' missing; cannot augment {json_path}."
                    )
                    continue
                ok = augment_layer_concept_set(
                    json_path=json_path,
                    activations_dir=activations_dir,
                    annotation_dir=set_dirs[name],
                    protein_subset=protein_subset,
                    max_proteins=max_proteins,
                    device=device,
                    cache_dir=cache_dir,
                    recompute_probe_sae=recompute_probe_sae,
                    recompute_pr=recompute_pr,
                )
                augmented += int(ok)

    if write_index:
        for rec in rec_list:
            index_path = rebuild_index(output_root_for(rec))
            LOGGER.info(msg=f"rec{rec} index -> {index_path}")

    if augmented == 0:
        LOGGER.error(msg="No benchmark JSONs were augmented.")
        return 1
    LOGGER.info(msg=f"Augmented {augmented} benchmark JSON(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(tapify(augment_benchmark, explicit_bool=True))
