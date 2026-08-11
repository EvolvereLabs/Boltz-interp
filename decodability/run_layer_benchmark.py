#!/usr/bin/env python3
"""Benchmark SAE latents vs single neurons vs linear probes for concept F1.

For each requested pairformer layer and concept set this script:

1. Loads the layer's raw Boltz activations (single-neuron baseline) and the same
   activations passed through the layer's trained SAE (latent baseline), over a
   **fixed, shared protein set** so the three methods are directly comparable.
2. Computes the best per-domain F1 per concept for (a) the best single SAE latent
   and (b) the best single raw neuron, using the shared scorer in ``benchmark_f1``.
3. Builds a **label-permutation null** for each unsupervised statistic (circular
   shift of each protein's labels), giving an empirical p-value that accounts for
   the multiple-comparison selection over the feature dictionary.
4. Trains **logistic-regression probes** (grouped CV by protein) on the raw
   activations and on the SAE latents as a supervised upper bound.
5. Writes per-(layer, concept set) JSON plus a tidy long-format index that the
   notebook reads.

Activations are the same files used by ``evaluate_f1_across_layers.py``; no extra
downloads are required for the single-neuron baseline. For the SwissProt concept
set, make sure each layer has the 100 annotated proteins downloaded (see
``swissprot_concept_proteins_manifest.txt``).

Example::

    uv run python run_layer_benchmark.py --layers 12,20 \\
        --concept_sets secondary,swissprot \\
        --activations_dir_template "downloads_layer{layer}" --n_perm 200
"""

import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tap import tapify

from loaders.raw_activation_loader import RawActivationLoader
from loaders.sae_latent_loader import SAELatentActivationLoader
from benchmark_f1 import (
    DEFAULT_THRESHOLD_PCTS,
    empirical_p_value,
    permutation_null,
    precompute_feature_thresholds,
    score_best_f1_per_concept,
)
from embeddings_concepts_evaluation import AnnotationLoader
from layer_analysis_utils import FINAL_STEP, download_run_checkpoint, layer_subfolder, repo_for_rec
from linear_probe import probe_f1_per_concept

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def _sweep_root_for(rec: int, layer_type: str = "pairformer") -> Path:
    """Per-(rec, layer_type) sweep root. Pairformer keeps the original un-prefixed
    path for backward compatibility; diffusion (and any future type) lives under a
    ``{layer_type}_`` prefix so its benchmark / cache never clobbers pairformer's.
    """
    prefix = "" if layer_type == "pairformer" else f"{layer_type}_"
    return Path(f"sae_explore/{prefix}layer_sweep_rec{rec}")


def output_root_for(rec: int, layer_type: str = "pairformer") -> Path:
    """Per-rec output tree so rec0 and rec1 runs never overwrite each other.

    The recycle index is part of the path (not just a flag) because the SAE
    weights, the activations, and therefore every number downstream differ by
    rec; sharing a directory silently clobbered one run with the other. The
    layer type is part of the path for the same reason (see ``_sweep_root_for``).
    """
    return _sweep_root_for(rec, layer_type) / "benchmark"


def cache_root_for(rec: int, layer_type: str = "pairformer") -> Path:
    """Per-rec SAE-checkpoint cache so rec0/rec1 (and types) never share a slot."""
    return _sweep_root_for(rec, layer_type) / "hf_cache"

DEFAULT_CONCEPT_SETS: dict[str, str] = {
    "secondary": "processed_swissprot_a5_structure_secondary_n500",
    "swissprot": "processed_swissprot",
    # Boltz's OWN predicted structure (build_boltz_structure_annotations.py). The principled
    # structural target -- labels come from the same forward pass as the activations.
    "boltz_secondary": "processed_swissprot_a5_boltz_secondary",
    "boltz_plddt": "processed_swissprot_a5_boltz_plddt",
}


def _parse_layers(layers: str) -> list[int]:
    """Parse a comma-separated layer string like ``"12,20"`` into ints."""
    return [int(token) for token in str(layers).replace(" ", "").split(",") if token]


def load_dual_matrices(
    raw_loader: RawActivationLoader,
    sae_loader: SAELatentActivationLoader,
    annotation_loader: AnnotationLoader,
    protein_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    """Load raw neurons, SAE latents, and labels on one shared protein set.

    Loading all three in lockstep guarantees identical protein ordering and
    residue offsets across the neuron, latent, and label matrices.

    Returns:
        ``(raw_matrix, sae_matrix, label_matrix, protein_offsets, used_ids)`` or
        ``None`` if no protein produced usable data.
    """
    raw_blocks: list[np.ndarray] = []
    sae_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    offsets: list[int] = [0]
    used_ids: list[str] = []

    for protein_id in protein_ids:
        raw = raw_loader.load(protein_id)
        if raw is None:
            continue
        sae = sae_loader.load(protein_id)
        if sae is None:
            continue
        ann = annotation_loader.load(protein_id)
        if ann is None:
            continue
        if not (raw.shape[0] == sae.shape[0] == ann.shape[0]):
            LOGGER.warning(msg=f"Length mismatch for {protein_id}; skipping.")
            continue
        raw_blocks.append(np.asarray(raw, dtype=np.float32))
        sae_blocks.append(np.asarray(sae, dtype=np.float32))
        label_blocks.append(np.asarray(ann, dtype=np.uint8))
        offsets.append(offsets[-1] + raw.shape[0])
        used_ids.append(protein_id)

    if not raw_blocks:
        return None

    raw_matrix = np.concatenate(raw_blocks)
    sae_matrix = np.concatenate(sae_blocks)
    label_matrix = np.concatenate(label_blocks)
    protein_offsets = np.asarray(offsets, dtype=np.int64)
    return raw_matrix, sae_matrix, label_matrix, protein_offsets, used_ids


def _unsupervised_records(
    feature_matrix: np.ndarray,
    label_matrix: np.ndarray,
    protein_offsets: np.ndarray,
    concept_list: list[str],
    method: str,
    n_perm: int,
    rng: np.random.Generator,
    feature_block: int,
    return_per_feature: bool = False,
) -> dict[str, dict] | tuple[dict[str, dict], np.ndarray]:
    """Observed best-F1 + permutation null for one unsupervised representation.

    With ``return_per_feature`` the ``(F, C)`` per-feature F1 matrix is returned
    alongside the records, so callers can persist F1 for every latent (not just
    each concept's winner). Adds no scoring work -- the matrix is a by-product of
    the argmax search.
    """
    thresholds = precompute_feature_thresholds(feature_matrix)
    scored = score_best_f1_per_concept(
        feature_matrix, thresholds, label_matrix, protein_offsets,
        feature_block=feature_block, return_per_feature=return_per_feature,
    )
    if return_per_feature:
        best_f1, best_feature, best_threshold, per_feature_f1 = scored
    else:
        best_f1, best_feature, best_threshold = scored
    LOGGER.info(msg=f"[{method}] running {n_perm} label permutations")
    null = permutation_null(
        feature_matrix,
        thresholds,
        label_matrix,
        protein_offsets,
        n_perm=n_perm,
        rng=rng,
        feature_block=feature_block,
    )

    records: dict[str, dict] = {}
    for concept_idx, concept_name in enumerate(concept_list):
        null_col = null[:, concept_idx]
        records[concept_name] = {
            "f1": float(best_f1[concept_idx]),
            "feature_idx": int(best_feature[concept_idx]),
            "threshold_pct": float(best_threshold[concept_idx]),
            "null_mean": float(np.mean(null_col)) if n_perm else float("nan"),
            "null_p95": float(np.percentile(null_col, 95)) if n_perm else float("nan"),
            "p_value": empirical_p_value(float(best_f1[concept_idx]), null_col),
        }
    if return_per_feature:
        return records, per_feature_f1
    return records


class LayerMatrices(NamedTuple):
    """Aligned raw/SAE/label matrices for one (layer, concept set) on a shared protein set."""

    raw_matrix: np.ndarray       # (N, n_neurons)
    sae_matrix: np.ndarray       # (N, n_latents)
    label_matrix: np.ndarray     # (N, C)
    protein_offsets: np.ndarray  # (P + 1,)
    used_ids: list[str]
    concept_list: list[str]
    n_annotated: int


def load_benchmark_matrices(
    layer: int,
    concept_set_name: str,
    annotation_dir: str,
    activations_dir: str,
    seed: int,
    rec: int,
    max_proteins: int,
    device: str | None,
    cache_dir: Path,
    require_full_rec: bool,
    protein_subset: set[str] | None,
    layer_type: str = "pairformer",
) -> LayerMatrices | None:
    """Build the raw + SAE + label matrices for one (layer, concept set).

    Extracted from :func:`benchmark_layer_concept_set` so that re-analysis passes
    (e.g. ``augment_benchmark.py``) reconstruct the *exact same* matrices from the
    same protein matching -- a hard requirement for patching metrics that reference
    feature indices stored by an earlier run. Given the same activations, SAE repo,
    and ``protein_subset``, this is deterministic, so the rebuilt matrices line up
    residue-for-residue with the originals.
    """
    # The SAE must be the one trained on THIS recycle's activations, pulled into a
    # rec-specific cache so it can't be served a stale checkpoint from another rec.
    # Pairformer SAEs are HF-backed (repo per rec); diffusion SAEs are S3-staged only,
    # so no HF repo is passed and download_run_checkpoint requires a cache hit.
    repo_id = repo_for_rec(rec) if layer_type == "pairformer" else None
    checkpoint = (
        download_run_checkpoint(
            layer, seed=seed, cache_dir=cache_dir, repo_id=repo_id, layer_type=layer_type
        )
        / f"checkpoint_step_{FINAL_STEP}.pt"
    )
    subfolder = layer_subfolder(layer, layer_type)
    raw_loader = RawActivationLoader(activations_dir, subfolder=subfolder, rec=rec)
    sae_loader = SAELatentActivationLoader(
        embeddings_dir=activations_dir,
        subfolder=subfolder,
        rec=rec,
        sae_checkpoint=str(checkpoint),
        device=device,
    )
    annotation_loader = AnnotationLoader(annotation_dir)

    ann_ids = set(annotation_loader.list_available())
    act_ids = set(raw_loader.list_available())
    # When a fixed subset is given (the common set across layers/recs), the candidate
    # pool is constant for this concept set regardless of layer or rec -- so n_proteins
    # is identical everywhere and the benchmark is directly comparable across both axes.
    target_ids = ann_ids & protein_subset if protein_subset is not None else ann_ids
    matched = sorted(target_ids & act_ids)
    if not matched:
        LOGGER.error(msg=f"Layer {layer}/{concept_set_name}: no protein overlap at rec={rec}; skipping.")
        return None
    # Loudly flag target proteins with no activation at THIS rec -- the silent
    # drop here is what made a "rec0" secondary run quietly fall back to whichever
    # proteins happened to have rec0 files (often far fewer than rec1). With a proper
    # common-set subset this should be empty (every layer/rec has all of them).
    missing_act = sorted(target_ids - act_ids)
    if missing_act:
        LOGGER.warning(
            msg=(
                f"Layer {layer}/{concept_set_name} rec={rec}: "
                f"{len(missing_act)}/{len(target_ids)} target proteins have NO output_{rec} "
                f"activation and will be DROPPED (using {len(matched)} proteins). "
                f"Download rec{rec} activations for the missing proteins for a complete run."
            )
        )
        if require_full_rec:
            raise RuntimeError(
                f"require_full_rec=True but {len(missing_act)} annotated proteins lack a rec={rec} "
                f"activation (layer {layer}/{concept_set_name}). First missing: {missing_act[:5]}"
            )
    if max_proteins > 0:
        matched = matched[:max_proteins]
    LOGGER.info(msg=f"Layer {layer}/{concept_set_name}: loading {len(matched)} proteins at rec={rec}")

    loaded = load_dual_matrices(raw_loader, sae_loader, annotation_loader, matched)
    if loaded is None:
        LOGGER.error(msg=f"Layer {layer}/{concept_set_name}: no usable proteins; skipping.")
        return None
    raw_matrix, sae_matrix, label_matrix, protein_offsets, used_ids = loaded
    LOGGER.info(
        msg=(
            f"Layer {layer}/{concept_set_name}: {len(used_ids)} proteins, "
            f"{raw_matrix.shape[0]} residues, {raw_matrix.shape[1]} neurons, "
            f"{sae_matrix.shape[1]} latents"
        )
    )
    return LayerMatrices(
        raw_matrix=raw_matrix,
        sae_matrix=sae_matrix,
        label_matrix=label_matrix,
        protein_offsets=protein_offsets,
        used_ids=used_ids,
        concept_list=annotation_loader.concept_list,
        n_annotated=len(ann_ids),
    )


def benchmark_layer_concept_set(
    layer: int,
    concept_set_name: str,
    annotation_dir: str,
    activations_dir: str,
    seed: int,
    rec: int,
    max_proteins: int,
    n_perm: int,
    probe_on_sae: bool,
    feature_block: int,
    device: str | None,
    rng: np.random.Generator,
    cache_dir: Path,
    require_full_rec: bool,
    protein_subset: set[str] | None,
    collect_per_latent: bool = False,
    layer_type: str = "pairformer",
) -> dict | None:
    """Run the full benchmark for one (layer, concept set); return a summary dict.

    When ``collect_per_latent`` is set, the summary carries a private
    ``"_sae_per_latent_f1"`` (``(F, C)`` array) and aligned ``"_concept_order"``
    under non-JSON keys; the caller pops these to write the per-latent sidecar and
    strips them before serialising the JSON summary.
    """
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
        require_full_rec=require_full_rec,
        protein_subset=protein_subset,
        layer_type=layer_type,
    )
    if matrices is None:
        return None
    raw_matrix = matrices.raw_matrix
    sae_matrix = matrices.sae_matrix
    label_matrix = matrices.label_matrix
    protein_offsets = matrices.protein_offsets
    used_ids = matrices.used_ids
    concept_list = matrices.concept_list

    sae_per_latent_f1 = None
    if collect_per_latent:
        sae_records, sae_per_latent_f1 = _unsupervised_records(
            sae_matrix, label_matrix, protein_offsets, concept_list, "sae", n_perm, rng,
            feature_block, return_per_feature=True,
        )
    else:
        sae_records = _unsupervised_records(
            sae_matrix, label_matrix, protein_offsets, concept_list, "sae", n_perm, rng, feature_block
        )
    neuron_records = _unsupervised_records(
        raw_matrix, label_matrix, protein_offsets, concept_list, "neuron", n_perm, rng, feature_block
    )
    LOGGER.info(msg=f"Layer {layer}/{concept_set_name}: training raw-activation probes")
    probe_raw = probe_f1_per_concept(raw_matrix, label_matrix, protein_offsets, concept_list)
    probe_sae: dict[str, float] = {}
    if probe_on_sae:
        LOGGER.info(msg=f"Layer {layer}/{concept_set_name}: training SAE-latent probes")
        probe_sae = probe_f1_per_concept(sae_matrix, label_matrix, protein_offsets, concept_list)

    scored_concepts = sorted(set(sae_records) | set(neuron_records) | set(probe_raw))
    per_concept: dict[str, dict] = {}
    for concept_name in scored_concepts:
        per_concept[concept_name] = {
            "sae": sae_records.get(concept_name),
            "neuron": neuron_records.get(concept_name),
            "probe_raw_f1": probe_raw.get(concept_name),
            "probe_sae_f1": probe_sae.get(concept_name),
        }

    summary = {
        "layer": layer,
        "concept_set": concept_set_name,
        "annotation_dir": annotation_dir,
        "seed": seed,
        "rec": rec,
        "layer_type": layer_type,
        # Diffusion SAEs have no HF repo (they're S3-staged); only pairformer maps rec->repo.
        "sae_repo": repo_for_rec(rec) if layer_type == "pairformer" else f"s3 ({layer_type})",
        "n_annotated": matrices.n_annotated,
        "n_proteins": len(used_ids),
        "n_residues": int(raw_matrix.shape[0]),
        "n_neurons": int(raw_matrix.shape[1]),
        "n_latents": int(sae_matrix.shape[1]),
        "n_perm": n_perm,
        "threshold_pcts": list(DEFAULT_THRESHOLD_PCTS),
        "per_concept": per_concept,
    }
    if sae_per_latent_f1 is not None:
        # Private (non-JSON) payload: the caller writes the .npz sidecar and pops
        # these before json.dump. null_mean lets the analysis form signal-above-null
        # per latent without re-reading the per-concept records.
        summary["_sae_per_latent_f1"] = sae_per_latent_f1
        summary["_concept_order"] = list(concept_list)
        summary["_null_mean"] = [
            float(sae_records[c]["null_mean"]) for c in concept_list
        ]
    return summary


def _long_records_from_summary(summary: dict) -> list[dict]:
    """Tidy long-format rows (one per layer/concept/method) for one summary."""
    rows: list[dict] = []
    for concept_name, entry in summary["per_concept"].items():
        base = {
            "layer": summary["layer"],
            "rec": summary.get("rec"),
            "seed": summary.get("seed"),
            "concept_set": summary["concept_set"],
            "concept": concept_name,
        }
        for method in ("sae", "neuron"):
            stats = entry.get(method)
            if stats is not None:
                rows.append({**base, "method": method, "f1": stats["f1"], "p_value": stats["p_value"], "null_mean": stats["null_mean"]})
        if entry.get("probe_raw_f1") is not None:
            rows.append({**base, "method": "probe_raw", "f1": entry["probe_raw_f1"], "p_value": None, "null_mean": None})
        if entry.get("probe_sae_f1") is not None:
            rows.append({**base, "method": "probe_sae", "f1": entry["probe_sae_f1"], "p_value": None, "null_mean": None})
    return rows


def rebuild_index(output_root: Path) -> Path:
    """Rewrite ``benchmark_index.jsonl`` from every per-layer JSON under ``output_root``.

    Rebuilding (vs appending) makes the index a pure function of the JSONs on disk,
    so reruns can't leave behind duplicate or stale rows from an earlier run.
    """
    index_path = output_root / "benchmark_index.jsonl"
    rows: list[dict] = []
    # Only the per-seed files ({set}_seed{n}_benchmark.json); a legacy seedless
    # {set}_benchmark.json (older single-seed runs) is deliberately ignored so it
    # can't double-count a (layer, concept) once the seeded files exist.
    for path in sorted(output_root.glob("layer*/*_seed*_benchmark.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(_long_records_from_summary(summary))
    with index_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return index_path


def run_layer_benchmark(
    layers: str = "12,20",
    concept_sets: str = "secondary,swissprot",
    activations_dir_template: str = "downloads_layer{layer}",
    secondary_dir: str = DEFAULT_CONCEPT_SETS["secondary"],
    swissprot_dir: str = DEFAULT_CONCEPT_SETS["swissprot"],
    boltz_secondary_dir: str = DEFAULT_CONCEPT_SETS["boltz_secondary"],
    boltz_plddt_dir: str = DEFAULT_CONCEPT_SETS["boltz_plddt"],
    seed: int = 1,
    rec: int = 1,
    max_proteins: int = 0,
    n_perm: int = 200,
    probe_on_sae: bool = True,
    feature_block: int = 256,
    skip_existing: bool = True,
    require_full_rec: bool = False,
    protein_ids_file: str | None = None,
    write_index: bool = True,
    rng_seed: int = 0,
    device: str | None = None,
    collect_per_latent: bool = False,
    layer_type: str = "pairformer",
) -> int:
    """Run the SAE-vs-neuron-vs-probe benchmark over one or more layers.

    Args:
        layers: Comma-separated layers, e.g. ``"12,20"``.
        layer_type: ``"pairformer"`` (default, HF-backed) or ``"diffusion"`` (S3-staged
            only). Selects the activation subfolder, the SAE run-name tag, and the
            ``sae_explore/[<type>_]layer_sweep_rec{rec}/`` output + cache tree.
        concept_sets: Comma-separated sets from
            ``{secondary, swissprot, boltz_secondary, boltz_plddt}``.
        activations_dir_template: Activation dir; may contain ``{layer}``.
        secondary_dir: Override directory for the AlphaFold secondary-structure labels.
        swissprot_dir: Override directory for the SwissProt concept labels.
        boltz_secondary_dir: Override directory for Boltz's-own-structure SS labels.
        boltz_plddt_dir: Override directory for the Boltz pLDDT-band labels.
        seed: SAE seed run to evaluate (checkpoint pulled from HuggingFace).
        rec: Recycle/output index. Selects BOTH the ``output_{rec}`` activations and
            the SAE repo trained on that recycle (see ``REC_REPO_IDS``); results are
            written under ``sae_explore/layer_sweep_rec{rec}/`` so recs never clobber.
        max_proteins: Cap on matched proteins (0 = all available).
        n_perm: Number of label permutations for the null (0 to skip).
        probe_on_sae: Also train probes on SAE latents (in addition to raw).
        feature_block: Feature columns scored per block (lower = less peak memory).
        skip_existing: Skip a (layer, concept set) whose benchmark JSON already exists.
        require_full_rec: Error out if any target protein lacks an ``output_{rec}``
            activation, instead of silently benchmarking on the subset that has it.
        protein_ids_file: Optional path to a newline-delimited list of protein IDs to
            restrict every (layer, concept set) to. Pass the common-across-layers-and-recs
            set (see ``compute_common_proteins.py``) so ``n_proteins`` is identical for a
            given concept set across all layers and recs -- the only way the numbers are
            directly comparable. IDs not in a concept's annotation set are simply ignored.
        write_index: Rebuild ``benchmark_index.jsonl`` at the end. Set False when running
            many single-layer processes in parallel (they'd race on the shared index);
            rebuild it once afterwards with ``rebuild_index(output_root_for(rec))``.
        rng_seed: Seed for the permutation/probe RNG.
        device: Optional torch device override for the SAE forward pass.

    Returns:
        Exit code; 0 if at least one (layer, concept set) produced results.
    """
    set_dirs = {
        "secondary": secondary_dir,
        "swissprot": swissprot_dir,
        "boltz_secondary": boltz_secondary_dir,
        "boltz_plddt": boltz_plddt_dir,
    }
    requested = [s for s in concept_sets.replace(" ", "").split(",") if s]
    for name in requested:
        if name not in set_dirs:
            raise ValueError(f"Unknown concept set '{name}'. Choose from {sorted(set_dirs)}.")

    protein_subset: set[str] | None = None
    if protein_ids_file:
        if not Path(protein_ids_file).exists():
            raise FileNotFoundError(
                f"protein_ids_file '{protein_ids_file}' not found. Generate it with "
                f"compute_common_proteins.py, or omit the flag to use all available proteins."
            )
        protein_subset = {
            line.strip().rstrip("/").split("/")[-1]
            for line in Path(protein_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        LOGGER.info(msg=f"Restricting to {len(protein_subset)} protein IDs from {protein_ids_file}")

    output_root = output_root_for(rec, layer_type)
    cache_dir = cache_root_for(rec, layer_type)
    rng = np.random.default_rng(rng_seed)
    output_root.mkdir(parents=True, exist_ok=True)
    sae_src = repo_for_rec(rec) if layer_type == "pairformer" else f"S3-staged ({layer_type})"
    LOGGER.info(msg=f"rec={rec} layer_type={layer_type}: outputs -> {output_root}, SAE source -> {sae_src}")
    produced = 0

    for layer in _parse_layers(layers):
        activations_dir = activations_dir_template.format(layer=layer)
        if not Path(activations_dir).exists():
            LOGGER.warning(msg=f"Activation directory '{activations_dir}' missing; skipping layer {layer}.")
            continue
        for name in requested:
            out_dir = output_root / f"layer{layer}"
            # Seed in the filename so the 3 seed runs for one (layer, concept set)
            # coexist instead of overwriting; the seedless name is kept free for
            # legacy single-seed outputs that predate multi-seed support.
            out_name = f"{name}_seed{seed}_benchmark.json"
            out_path = out_dir / out_name
            npz_path = out_dir / f"{name}_seed{seed}_perlatent_f1.npz"
            # When collecting per-latent F1, a pre-existing JSON is only "done" if
            # its sidecar exists too -- otherwise re-run to backfill the sidecar.
            already_done = out_path.exists() and (not collect_per_latent or npz_path.exists())
            if skip_existing and already_done:
                LOGGER.info(msg=f"Skipping layer {layer}/{name} seed {seed}: {out_path} already exists.")
                continue
            summary = benchmark_layer_concept_set(
                layer=layer,
                concept_set_name=name,
                annotation_dir=set_dirs[name],
                activations_dir=activations_dir,
                seed=seed,
                rec=rec,
                max_proteins=max_proteins,
                n_perm=n_perm,
                probe_on_sae=probe_on_sae,
                feature_block=feature_block,
                device=device,
                rng=rng,
                cache_dir=cache_dir,
                require_full_rec=require_full_rec,
                protein_subset=protein_subset,
                collect_per_latent=collect_per_latent,
                layer_type=layer_type,
            )
            if summary is None:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            # Pop the private numpy payload (un-JSON-able) and write it as a compact
            # sidecar: f1 is (F, C), columns aligned to `concepts`, null_mean per concept.
            per_latent = summary.pop("_sae_per_latent_f1", None)
            concept_order = summary.pop("_concept_order", None)
            null_mean = summary.pop("_null_mean", None)
            (out_dir / out_name).write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            if per_latent is not None:
                np.savez_compressed(
                    npz_path,
                    f1=per_latent.astype(np.float32),
                    concepts=np.array(concept_order),  # unicode dtype -> no pickle on load
                    null_mean=np.array(null_mean, dtype=np.float32),
                )
                LOGGER.info(msg=f"Wrote per-latent F1 sidecar {npz_path.name} {per_latent.shape}")
            produced += 1
            LOGGER.info(msg=f"Wrote benchmark for layer {layer}/{name} seed {seed}")

    # Rebuild the index from the JSONs on disk so it reflects exactly what was computed
    # (no stale/duplicate rows). Skipped under parallel single-layer runs to avoid a
    # write race on the shared index; rebuild once at the end instead.
    if write_index:
        index_path = rebuild_index(output_root)
        LOGGER.info(msg=f"Index at {index_path}")
    if produced == 0:
        LOGGER.error(msg="No new (layer, concept set) combinations produced results.")
        return 1
    LOGGER.info(msg=f"Completed {produced} benchmarks (write_index={write_index})")
    return 0


if __name__ == "__main__":
    # explicit_bool lets flags be set either way, e.g. ``--skip_existing False``.
    raise SystemExit(tapify(run_layer_benchmark, explicit_bool=True))
