#!/usr/bin/env python3
"""Sanity check: do SAE latents / neurons / probes encode amino-acid identity?

Residue identity is the most local, model-visible concept there is -- the model
reads the one-hot amino acid straight off its input -- so it is the concept we
*most* expect to find cleanly represented. This sweep therefore calibrates the
whole interpretability story: it asks, layer by layer, how well the easiest
possible concept survives in three representations.

For each requested layer it scores three representations against the one-hot
amino-acid concept set built by ``build_amino_acid_concepts.py``:

1. ``sae``    -- best single SAE latent per amino acid.
2. ``neuron`` -- best single raw activation dimension per amino acid.
3. ``probe``  -- a per-amino-acid L2 logistic regression over the *full* raw
   activation vector (protein-level held-out test F1), an upper bound on what is
   *linearly* decodable. Requires scikit-learn; skipped with a warning otherwise.

The metric is per-residue F1 at the activation threshold (over positive-activation
percentiles) that maximises it -- the honest metric for residue identity, where
each occurrence is independent. This is intentionally faster and simpler than the
domain-level ``ConceptEvaluator`` so a 2048-latent SAE can be swept across every
pairformer layer plus the diffusion stack in minutes.

Results stream to ``sae_explore/amino_acid_sanity/results.jsonl`` (one row per
layer/amino-acid), which ``amino_acid_identity_analysis.ipynb`` reads.

Examples::

    uv run python build_amino_acid_concepts.py --source_dir processed_swissprot_a5
    # one pairformer layer
    uv run python amino_acid_sanity_check.py --layers 20
    # the full local sweep (all even pairformer layers + diffusion layer 22)
    uv run python amino_acid_sanity_check.py \\
        --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47
    uv run python amino_acid_sanity_check.py --layer_type diffusion --layers 22 \\
        --activations_dir downloads_diffusion_rec199 --rec 199 \\
        --checkpoint_cache_dir sae_explore/diffusion_cache/rec199
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from tap import tapify

from build_amino_acid_concepts import STANDARD_AMINO_ACIDS
from embeddings_concepts_evaluation import AnnotationLoader
from layer_analysis_utils import FINAL_STEP, download_run_checkpoint, layer_subfolder
from sae_utils import apply_demean, list_activation_files, load_activation, resolve_device
from loaders.sae_latent_loader import load_sae_from_checkpoint, resolve_mean_vector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

OUTPUT_ROOT = Path("sae_explore/amino_acid_sanity")
PERCENTILES = [50, 75, 90, 95, 99]


def _activation_path(embeddings_dir: Path, protein_id: str, subfolder: str, rec: int) -> Path | None:
    """Return the raw activation file for a protein, or None if missing."""
    layer_dir = embeddings_dir / protein_id / subfolder
    for name in (f"output_{rec}.npz", f"output_{rec}.npy.gz"):
        candidate = layer_dir / name
        if candidate.exists():
            return candidate
    return None


def list_available_proteins(embeddings_dir: Path, subfolder: str, rec: int) -> list[str]:
    """List protein IDs with a raw activation file under ``embeddings_dir``."""
    files = list_activation_files(embeddings_dir, subfolder, rec, None)
    return sorted({path.parent.parent.name for path in files})


def gather_activations(
    embeddings_dir: Path,
    subfolder: str,
    rec: int,
    annotation_loader: AnnotationLoader,
    protein_ids: list[str],
    target_per_aa: int,
    max_proteins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw activations + one-hot labels until each amino acid is well sampled.

    Proteins are consumed in order; loading stops once every standard amino acid
    has at least ``target_per_aa`` positive residues (or ``max_proteins`` is hit).

    Returns:
        ``raw`` -- ``(n_residues, input_dim)`` float32 activations.
        ``labels`` -- ``(n_residues, 20)`` int8 one-hot amino-acid labels.
        ``protein_row_ids`` -- ``(n_residues,)`` int index identifying each
            residue's source protein (for protein-level probe splits).
    """
    raw_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    row_ids: list[np.ndarray] = []
    aa_counts = np.zeros(len(STANDARD_AMINO_ACIDS), dtype=np.int64)
    n_used = 0

    for protein_id in protein_ids:
        if max_proteins > 0 and n_used >= max_proteins:
            break
        if target_per_aa > 0 and aa_counts.min() >= target_per_aa:
            break
        path = _activation_path(embeddings_dir, protein_id, subfolder, rec)
        labels = annotation_loader.load(protein_id)
        if path is None or labels is None:
            continue
        acts = load_activation(path).astype(np.float32, copy=False)
        if acts.shape[0] != labels.shape[0]:
            LOGGER.warning(
                msg=f"Length mismatch for {protein_id}: acts {acts.shape[0]} vs labels {labels.shape[0]}; skipping."
            )
            continue
        raw_blocks.append(acts)
        label_blocks.append(np.asarray(labels, dtype=np.int8))
        row_ids.append(np.full(acts.shape[0], n_used, dtype=np.int64))
        aa_counts += np.asarray(labels, dtype=np.int64).sum(axis=0)
        n_used += 1

    if not raw_blocks:
        return np.empty((0, 0), np.float32), np.empty((0, len(STANDARD_AMINO_ACIDS)), np.int8), np.empty(0, np.int64)

    LOGGER.info(
        msg=(
            f"Loaded {n_used} proteins, {sum(b.shape[0] for b in raw_blocks)} residues; "
            f"min per-AA count = {int(aa_counts.min())} ({STANDARD_AMINO_ACIDS[int(aa_counts.argmin())]})."
        )
    )
    return np.concatenate(raw_blocks), np.concatenate(label_blocks), np.concatenate(row_ids)


def encode_with_sae(raw: np.ndarray, checkpoint: Path, device: torch.device) -> np.ndarray:
    """Encode raw activations into SAE latents, mirroring the training preprocessing."""
    model, cfg = load_sae_from_checkpoint(checkpoint, device)
    mean_vec = resolve_mean_vector(checkpoint, cfg)
    pre_encoder_bias = bool(cfg.get("pre_encoder_bias", False))
    processed = apply_demean(raw, mean_vec) if mean_vec is not None else raw
    out = np.empty((processed.shape[0], int(cfg["latent_dim"])), dtype=np.float32)
    batch_size = 8192
    with torch.no_grad():
        for start in range(0, processed.shape[0], batch_size):
            chunk = torch.from_numpy(processed[start : start + batch_size]).to(device)
            if pre_encoder_bias:
                chunk = chunk - model.decoder.bias
            out[start : start + batch_size] = model.encode(chunk).cpu().numpy()
    return out


def best_feature_f1_per_aa(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best single-feature per-residue F1 for each amino acid (vectorised).

    For every feature the activation threshold is swept over positive-activation
    percentiles; the best (feature, threshold) F1 is kept per amino acid.

    Args:
        features: ``(n_residues, n_features)`` activations.
        labels: ``(n_residues, 20)`` one-hot amino-acid labels.

    Returns:
        ``best_f1`` -- ``(20,)`` best F1 per amino acid.
        ``best_feature`` -- ``(20,)`` index of the feature achieving it (-1 if none).
    """
    n_aa = labels.shape[1]
    best_f1 = np.zeros(n_aa, dtype=np.float64)
    best_feature = np.full(n_aa, -1, dtype=np.int64)
    pos_per_aa = labels.sum(axis=0).astype(np.float64)  # (20,)

    for thresh_pct in PERCENTILES:
        # Per-feature threshold from that feature's positive activations.
        positive = np.where(features > 0.0, features, np.nan)
        with np.errstate(invalid="ignore"):
            thresholds = np.nanpercentile(positive, thresh_pct, axis=0)  # (F,)
        thresholds = np.where(np.isfinite(thresholds), thresholds, np.inf)
        pred = features >= thresholds[None, :]  # (L, F) bool

        pred_pos = pred.sum(axis=0).astype(np.float64)  # (F,)
        # tp[c, f] = number of residues where pred fires AND label is amino acid c.
        tp = labels.T.astype(np.float64) @ pred.astype(np.float64)  # (20, F)
        fp = pred_pos[None, :] - tp
        fn = pos_per_aa[:, None] - tp
        denom = 2.0 * tp + fp + fn
        f1 = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)  # (20, F)

        f1_max = f1.max(axis=1)
        f1_arg = f1.argmax(axis=1)
        improved = f1_max > best_f1
        best_feature = np.where(improved, f1_arg, best_feature)
        best_f1 = np.where(improved, f1_max, best_f1)

    return best_f1, best_feature


def probe_f1_per_aa(
    raw: np.ndarray,
    labels: np.ndarray,
    protein_row_ids: np.ndarray,
    test_fraction: float,
    max_train_residues: int,
    seed: int,
) -> np.ndarray:
    """Per-amino-acid linear-probe held-out F1 (protein-level train/test split)."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score
    except ImportError:
        LOGGER.warning(msg="scikit-learn not installed; skipping the linear-probe baseline.")
        return np.full(labels.shape[1], np.nan)

    rng = np.random.default_rng(seed)
    proteins = np.unique(protein_row_ids)
    rng.shuffle(proteins)
    n_test = max(1, int(len(proteins) * test_fraction))
    test_proteins = set(proteins[:n_test].tolist())
    is_test = np.isin(protein_row_ids, list(test_proteins))

    train_x, train_y = raw[~is_test], labels[~is_test]
    test_x, test_y = raw[is_test], labels[is_test]
    if train_x.shape[0] == 0 or test_x.shape[0] == 0:
        LOGGER.warning(msg="Probe: empty train/test split; skipping.")
        return np.full(labels.shape[1], np.nan)

    if train_x.shape[0] > max_train_residues:
        keep = rng.choice(train_x.shape[0], size=max_train_residues, replace=False)
        train_x, train_y = train_x[keep], train_y[keep]

    f1s = np.full(labels.shape[1], np.nan)
    for aa_index in range(labels.shape[1]):
        y_train = train_y[:, aa_index]
        y_test = test_y[:, aa_index]
        if y_train.sum() < 5 or y_test.sum() < 5:
            continue
        clf = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced")
        clf.fit(train_x, y_train)
        f1s[aa_index] = float(f1_score(y_test, clf.predict(test_x), zero_division=0))
    return f1s


def run_one_layer(
    layer: int,
    layer_type: str,
    embeddings_dir: Path,
    rec: int,
    annotation_loader: AnnotationLoader,
    representations: list[str],
    target_per_aa: int,
    max_proteins: int,
    probe_test_fraction: float,
    probe_max_residues: int,
    seed: int,
    checkpoint: str,
    checkpoint_cache_dir: str,
    device: torch.device,
) -> list[dict]:
    """Score all requested representations for one layer; return per-AA rows."""
    subfolder = layer_subfolder(layer, layer_type)
    available = sorted(
        set(list_available_proteins(embeddings_dir, subfolder, rec))
        & set(annotation_loader.list_available())
    )
    if not available:
        LOGGER.error(
            msg=f"{layer_type} layer {layer}: no proteins overlap with amino-acid labels under {embeddings_dir}."
        )
        return []

    raw, labels, row_ids = gather_activations(
        embeddings_dir, subfolder, rec, annotation_loader, available, target_per_aa, max_proteins
    )
    if raw.size == 0:
        LOGGER.error(msg=f"{layer_type} layer {layer}: no residues loaded.")
        return []

    n_proteins = int(np.unique(row_ids).size)
    n_residues = int(raw.shape[0])
    rows = [
        {
            "layer_type": layer_type,
            "layer": layer,
            "seed": seed,
            "n_proteins": n_proteins,
            "n_residues": n_residues,
            "amino_acid": STANDARD_AMINO_ACIDS[i],
            "n_positive": int(labels[:, i].sum()),
        }
        for i in range(len(STANDARD_AMINO_ACIDS))
    ]

    if "neuron" in representations:
        f1, feat = best_feature_f1_per_aa(raw, labels)
        for i, row in enumerate(rows):
            row["neuron_f1"] = float(f1[i])
            row["neuron_feature"] = int(feat[i])

    if "sae" in representations:
        if checkpoint:
            ckpt_path = Path(checkpoint)
        else:
            cache_dir = Path(checkpoint_cache_dir)
            ckpt_path = (
                download_run_checkpoint(layer, seed=seed, cache_dir=cache_dir, layer_type=layer_type)
                / f"checkpoint_step_{FINAL_STEP}.pt"
            )
        latents = encode_with_sae(raw, ckpt_path, device)
        f1, feat = best_feature_f1_per_aa(latents, labels)
        for i, row in enumerate(rows):
            row["sae_f1"] = float(f1[i])
            row["sae_feature"] = int(feat[i])

    if "probe" in representations:
        probe = probe_f1_per_aa(
            raw, labels, row_ids, probe_test_fraction, probe_max_residues, seed
        )
        for i, row in enumerate(rows):
            row["probe_f1"] = None if np.isnan(probe[i]) else float(probe[i])

    means = {
        rep: np.mean([r[f"{rep}_f1"] for r in rows if r.get(f"{rep}_f1") is not None])
        for rep in representations
        if any(r.get(f"{rep}_f1") is not None for r in rows)
    }
    LOGGER.info(
        msg=f"{layer_type} layer {layer}: mean F1 "
        + ", ".join(f"{rep}={val:.3f}" for rep, val in means.items())
    )
    return rows


def amino_acid_sanity_check(
    layers: str = "20",
    layer_type: str = "pairformer",
    activations_dir: str = "",
    aa_concept_dir: str = "processed_swissprot_aa",
    rec: int = 1,
    representations: str = "sae,neuron,probe",
    target_per_aa: int = 500,
    max_proteins: int = 250,
    probe_test_fraction: float = 0.3,
    probe_max_residues: int = 150000,
    seed: int = 1,
    checkpoint: str = "",
    checkpoint_cache_dir: str = "sae_explore/layer_sweep/hf_cache",
    device: str = "",
    out_jsonl: str = "",
    append: bool = False,
) -> int:
    """Sweep amino-acid-identity F1 across layers for SAE latents, neurons, and a probe.

    Args:
        layers: Comma-separated layers, e.g. ``"0,2,20,47"``.
        layer_type: ``"pairformer"`` or ``"diffusion"`` (selects activation
            subfolder and SAE source).
        activations_dir: Raw-activation directory. May contain ``{layer}``. Empty
            -> ``downloads_layer{layer}`` for pairformer, ``downloads_diffusion_rec{rec}``
            for diffusion.
        aa_concept_dir: Output of ``build_amino_acid_concepts.py``.
        rec: Recycle/output index of the activations (diffusion uses 199).
        representations: Comma-separated subset of ``{sae, neuron, probe}``.
        target_per_aa: Load proteins until every amino acid has at least this many
            positive residues (0 disables; then ``max_proteins`` governs).
        max_proteins: Hard cap on proteins loaded per layer.
        probe_test_fraction: Fraction of proteins held out for the probe test F1.
        probe_max_residues: Cap on probe training residues (speed).
        seed: Seed for the SAE run, the probe split, and subsampling.
        checkpoint: Explicit SAE checkpoint path; empty -> resolve per layer.
        checkpoint_cache_dir: Cache dir for resolved checkpoints. Diffusion SAEs
            live under ``sae_explore/diffusion_cache/rec{rec}``.
        device: Optional torch device override.
        out_jsonl: Output JSONL path. Empty -> ``sae_explore/amino_acid_sanity/results.jsonl``.
        append: Append to the JSONL instead of overwriting (for multi-run sweeps).

    Returns:
        Process exit code (0 if at least one layer produced rows).
    """
    if not Path(aa_concept_dir).exists():
        LOGGER.error(
            msg=f"Amino-acid concept dir '{aa_concept_dir}' not found. Run build_amino_acid_concepts.py first."
        )
        return 1

    requested = [r for r in representations.replace(" ", "").split(",") if r]
    valid = {"sae", "neuron", "probe"}
    unknown = [r for r in requested if r not in valid]
    if unknown:
        LOGGER.error(msg=f"Unknown representation(s) {unknown}; choose from {sorted(valid)}.")
        return 1

    layer_list = [int(token) for token in layers.replace(" ", "").split(",") if token]
    device_obj = resolve_device(device or None)
    annotation_loader = AnnotationLoader(aa_concept_dir)

    default_template = (
        f"downloads_diffusion_rec{rec}" if layer_type == "diffusion" else "downloads_layer{layer}"
    )
    dir_template = activations_dir or default_template

    out_path = Path(out_jsonl) if out_jsonl else OUTPUT_ROOT / "results.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"

    all_rows: list[dict] = []
    with out_path.open(mode, encoding="utf-8") as handle:
        for layer in layer_list:
            embeddings_dir = Path(dir_template.format(layer=layer))
            if not embeddings_dir.exists():
                LOGGER.warning(msg=f"Activation dir '{embeddings_dir}' for layer {layer} not found; skipping.")
                continue
            rows = run_one_layer(
                layer=layer,
                layer_type=layer_type,
                embeddings_dir=embeddings_dir,
                rec=rec,
                annotation_loader=annotation_loader,
                representations=requested,
                target_per_aa=target_per_aa,
                max_proteins=max_proteins,
                probe_test_fraction=probe_test_fraction,
                probe_max_residues=probe_max_residues,
                seed=seed,
                checkpoint=checkpoint,
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
    LOGGER.info(msg=f"Wrote {len(all_rows)} rows ({len(layer_list)} layers requested) to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(tapify(amino_acid_sanity_check))
