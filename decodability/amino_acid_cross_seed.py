#!/usr/bin/env python3
"""Cross-seed stability of the amino-acid-identity SAE latents.

The single-seed sweep (``amino_acid_sanity_check.py``) shows that one SAE latent
recovers each amino acid. This asks the follow-up: is it *the same* latent across
independently trained seeds? For each layer it loads every seed's SAE, finds the
best latent per amino acid in each, and measures whether those latents agree:

* ``sae_f1_seed{s}``    -- best single-latent per-residue F1 per seed (+ mean/std).
* ``decoder_cos_mean``  -- mean pairwise cosine between the best-AA latents'
  *decoder directions* across seeds. High => seeds learned the same direction.
* ``act_corr_mean``     -- mean pairwise Pearson correlation between those latents'
  per-residue *activation profiles*. High => they fire on the same residues.

``neuron_f1`` and ``probe_f1`` (seed-independent) are reported alongside for
reference. Results stream to ``sae_explore/amino_acid_sanity/cross_seed_results.jsonl``
(one row per layer/amino-acid), read by ``amino_acid_identity_analysis.ipynb``.

Examples::

    # pairformer, all even layers, 3 seeds
    uv run python amino_acid_cross_seed.py \\
        --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47
    # diffusion layers x recs (after stage_diffusion_data.py)
    uv run python amino_acid_cross_seed.py --layer_type diffusion --layers 4,14,22 \\
        --rec 199 --checkpoint_cache_dir sae_explore/diffusion_cache/rec199 --append
"""

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from tap import tapify

from amino_acid_sanity_check import (
    PERCENTILES,
    best_feature_f1_per_aa,
    gather_activations,
    list_available_proteins,
    probe_f1_per_aa,
)
from build_amino_acid_concepts import STANDARD_AMINO_ACIDS
from embeddings_concepts_evaluation import AnnotationLoader
from layer_analysis_utils import FINAL_STEP, download_run_checkpoint, layer_subfolder, repo_for_rec
from sae_utils import apply_demean, resolve_device
from loaders.sae_latent_loader import load_sae_from_checkpoint, resolve_mean_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

OUTPUT_ROOT = Path("sae_explore/amino_acid_sanity")


def encode_and_decoder(
    raw: np.ndarray, checkpoint: Path, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Encode raw activations to latents and return the decoder direction matrix.

    Returns:
        ``latents`` -- ``(n_residues, latent_dim)`` SAE activations.
        ``decoder`` -- ``(input_dim, latent_dim)``; column ``j`` is latent ``j``'s
            decoder direction in (demeaned) input space.
    """
    model, cfg = load_sae_from_checkpoint(checkpoint, device)
    mean_vec = resolve_mean_vector(checkpoint, cfg)
    pre_encoder_bias = bool(cfg.get("pre_encoder_bias", False))
    processed = apply_demean(raw, mean_vec) if mean_vec is not None else raw

    latent_dim = int(cfg["latent_dim"])
    latents = np.empty((processed.shape[0], latent_dim), dtype=np.float32)
    batch_size = 8192
    with torch.no_grad():
        for start in range(0, processed.shape[0], batch_size):
            chunk = torch.from_numpy(processed[start : start + batch_size]).to(device)
            if pre_encoder_bias:
                chunk = chunk - model.decoder.bias
            latents[start : start + batch_size] = model.encode(chunk).cpu().numpy()
    decoder = model.decoder.weight.detach().cpu().numpy()  # (input_dim, latent_dim)
    return latents, decoder


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two vectors (0.0 if either is constant)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    sa, sb = a.std(), b.std()
    if sa == 0 or sb == 0:
        return 0.0
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two direction vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_one_layer(
    layer: int,
    layer_type: str,
    embeddings_dir: Path,
    rec: int,
    annotation_loader: AnnotationLoader,
    seeds: list[int],
    target_per_aa: int,
    max_proteins: int,
    probe_test_fraction: float,
    probe_max_residues: int,
    checkpoint_cache_dir: str,
    device: torch.device,
) -> list[dict]:
    """Score every seed for one layer and return per-amino-acid stability rows."""
    subfolder = layer_subfolder(layer, layer_type)
    available = sorted(
        set(list_available_proteins(embeddings_dir, subfolder, rec))
        & set(annotation_loader.list_available())
    )
    if not available:
        LOGGER.error(msg=f"{layer_type} layer {layer} rec {rec}: no proteins overlap with labels.")
        return []

    raw, labels, row_ids = gather_activations(
        embeddings_dir, subfolder, rec, annotation_loader, available, target_per_aa, max_proteins
    )
    if raw.size == 0:
        return []

    n_aa = len(STANDARD_AMINO_ACIDS)
    neuron_f1, _ = best_feature_f1_per_aa(raw, labels)
    probe = probe_f1_per_aa(raw, labels, row_ids, probe_test_fraction, probe_max_residues, seeds[0])

    # Per seed: best latent per AA, its activation profile, and decoder direction.
    per_seed_f1: dict[int, np.ndarray] = {}
    per_seed_feat: dict[int, np.ndarray] = {}
    per_seed_act: dict[int, np.ndarray] = {}  # (n_aa, L) best-latent activations
    per_seed_dir: dict[int, np.ndarray] = {}  # (n_aa, input_dim) best-latent decoder dirs
    for seed in seeds:
        repo_id = repo_for_rec(rec) if layer_type == "pairformer" else None
        ckpt = (
            download_run_checkpoint(
                layer,
                seed=seed,
                cache_dir=Path(checkpoint_cache_dir),
                repo_id=repo_id,
                layer_type=layer_type,
            )
            / f"checkpoint_step_{FINAL_STEP}.pt"
        )
        latents, decoder = encode_and_decoder(raw, ckpt, device)
        f1, feat = best_feature_f1_per_aa(latents, labels)
        per_seed_f1[seed] = f1
        per_seed_feat[seed] = feat
        per_seed_act[seed] = np.stack(
            [latents[:, feat[i]] if feat[i] >= 0 else np.zeros(latents.shape[0]) for i in range(n_aa)]
        )
        per_seed_dir[seed] = np.stack(
            [decoder[:, feat[i]] if feat[i] >= 0 else np.zeros(decoder.shape[0]) for i in range(n_aa)]
        )

    rows = []
    for i, amino_acid in enumerate(STANDARD_AMINO_ACIDS):
        f1_values = [float(per_seed_f1[s][i]) for s in seeds]
        cos_pairs, corr_pairs = [], []
        for s_a, s_b in combinations(seeds, 2):
            if per_seed_feat[s_a][i] < 0 or per_seed_feat[s_b][i] < 0:
                continue
            cos_pairs.append(_cosine(per_seed_dir[s_a][i], per_seed_dir[s_b][i]))
            corr_pairs.append(_pearson(per_seed_act[s_a][i], per_seed_act[s_b][i]))
        row = {
            "layer_type": layer_type,
            "layer": layer,
            "rec": rec,
            "amino_acid": amino_acid,
            "n_positive": int(labels[:, i].sum()),
            "n_proteins": int(np.unique(row_ids).size),
            "neuron_f1": float(neuron_f1[i]),
            "probe_f1": None if np.isnan(probe[i]) else float(probe[i]),
            "sae_f1_mean": float(np.mean(f1_values)),
            "sae_f1_std": float(np.std(f1_values)),
            "decoder_cos_mean": float(np.mean(cos_pairs)) if cos_pairs else None,
            "decoder_cos_min": float(np.min(cos_pairs)) if cos_pairs else None,
            "act_corr_mean": float(np.mean(corr_pairs)) if corr_pairs else None,
            "act_corr_min": float(np.min(corr_pairs)) if corr_pairs else None,
        }
        for s in seeds:
            row[f"sae_f1_seed{s}"] = float(per_seed_f1[s][i])
            row[f"sae_feature_seed{s}"] = int(per_seed_feat[s][i])
        rows.append(row)

    LOGGER.info(
        msg=(
            f"{layer_type} layer {layer} rec {rec}: "
            f"mean SAE F1 {np.mean([r['sae_f1_mean'] for r in rows]):.3f} "
            f"(std {np.mean([r['sae_f1_std'] for r in rows]):.3f}), "
            f"decoder cos {np.nanmean([r['decoder_cos_mean'] for r in rows if r['decoder_cos_mean'] is not None]):.3f}, "
            f"act corr {np.nanmean([r['act_corr_mean'] for r in rows if r['act_corr_mean'] is not None]):.3f}"
        )
    )
    return rows


def amino_acid_cross_seed(
    layers: str = "20",
    layer_type: str = "pairformer",
    activations_dir: str = "",
    aa_concept_dir: str = "processed_swissprot_aa",
    rec: int = 1,
    seeds: str = "1,2,3",
    target_per_aa: int = 500,
    max_proteins: int = 250,
    probe_test_fraction: float = 0.3,
    probe_max_residues: int = 150000,
    checkpoint_cache_dir: str = "",
    device: str = "",
    out_jsonl: str = "",
    append: bool = False,
) -> int:
    """Sweep cross-seed amino-acid-latent stability across layers.

    Args mirror ``amino_acid_sanity_check.py`` plus ``seeds`` (comma-separated SAE
    seeds to compare). Pairformer runs automatically use the SAE repo and checkpoint
    cache for ``rec``. For diffusion, point ``checkpoint_cache_dir`` at
    ``sae_explore/diffusion_cache/rec{rec}`` and ``activations_dir`` at the matching
    ``downloads_diffusion_rec{rec}`` (the default when ``activations_dir`` is empty).

    Returns:
        Process exit code (0 if at least one layer produced rows).
    """
    if not Path(aa_concept_dir).exists():
        LOGGER.error(msg=f"Concept dir '{aa_concept_dir}' not found; run build_amino_acid_concepts.py.")
        return 1

    layer_list = [int(t) for t in layers.replace(" ", "").split(",") if t]
    seed_list = [int(t) for t in seeds.replace(" ", "").split(",") if t]
    if len(seed_list) < 2:
        LOGGER.error(msg="Need at least two seeds for cross-seed stability.")
        return 1

    device_obj = resolve_device(device or None)
    annotation_loader = AnnotationLoader(aa_concept_dir)
    default_template = (
        f"downloads_diffusion_rec{rec}" if layer_type == "diffusion" else "downloads_layer{layer}"
    )
    dir_template = activations_dir or default_template
    if not checkpoint_cache_dir:
        checkpoint_cache_dir = (
            f"sae_explore/diffusion_cache/rec{rec}"
            if layer_type == "diffusion"
            else f"sae_explore/layer_sweep_rec{rec}/hf_cache"
        )

    out_path = Path(out_jsonl) if out_jsonl else OUTPUT_ROOT / "cross_seed_results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    with out_path.open("a" if append else "w", encoding="utf-8") as handle:
        for layer in layer_list:
            embeddings_dir = Path(dir_template.format(layer=layer))
            if not embeddings_dir.exists():
                LOGGER.warning(msg=f"Activation dir '{embeddings_dir}' missing; skipping layer {layer}.")
                continue
            rows = run_one_layer(
                layer=layer,
                layer_type=layer_type,
                embeddings_dir=embeddings_dir,
                rec=rec,
                annotation_loader=annotation_loader,
                seeds=seed_list,
                target_per_aa=target_per_aa,
                max_proteins=max_proteins,
                probe_test_fraction=probe_test_fraction,
                probe_max_residues=probe_max_residues,
                checkpoint_cache_dir=checkpoint_cache_dir,
                device=device_obj,
            )
            for row in rows:
                handle.write(json.dumps(row) + "\n")
                handle.flush()
            all_rows.extend(rows)

    if not all_rows:
        LOGGER.error(msg="No layers produced results.")
        return 1
    LOGGER.info(msg=f"Wrote {len(all_rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(tapify(amino_acid_cross_seed))
