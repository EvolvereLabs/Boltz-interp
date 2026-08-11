#!/usr/bin/env python3
"""Export an SAE-derived "helix" steering direction for causal steering of Boltz-1.

This is the SAE-feature analogue of ``export_probe_direction.py``. Instead of fitting a
supervised logistic probe, it derives a helix steering direction from the trunk L47 Top-K
SAE by selecting concept-relevant latents and combining their decoder columns.

Latent selection (``--select``)
-------------------------------
Two selection statistics are available:

* ``f1`` (DEFAULT) -- rank latents by their per-latent **helix F1** (discrimination quality).
  F1 is the SAME statistic the benchmark reports: each latent's activation is treated as a
  score, the benchmark's percentile thresholds are swept, and the best-threshold per-domain
  F1 is kept per latent (reusing ``benchmark_f1.score_best_f1_per_concept`` so the numbers
  match ``run_layer_benchmark``). A label-permutation null (``benchmark_f1.permutation_null``)
  gives a baseline for the ``--f1_above_null`` mode. Selection sub-modes:

    - default (k=1): the SINGLE highest-F1 latent's decoder column -- the cleanest
      interpretability variant (one latent -> one direction).
    - ``--f1_above_null``: ALL latents with F1 above the permutation-null baseline
      (data-driven count).
    - ``--f1_min <float>``: ALL latents with F1 >= f1_min (data-driven count, fixed cutoff).
    - ``--k N``: force the top-N latents by F1.

* ``crs`` -- the original **Concept Relevance Score**: rank latents by how much more they fire
  on positives than on residues in general,
  ``R(i) = mean_act(i | positive) - mean_act(i | all)`` (see ``--crs_norm``), and take top-k.
  Kept available because it is the paper's stated method, but F1 measures discrimination
  quality (precision + recall) rather than just "fires on positives", so ``f1`` is the default.

Building the steering vector
----------------------------
The selected latents' **decoder columns** are combined into the steering vector (each decoder
column is a direction in the same raw 384-dim activation space the probe directions live in,
because the SAE reconstructs ``raw ~= decoder @ latents + decoder.bias + mean_vector``)::

    v_c = sum_{i in selected} w_i * decoder_column_i
    w_i = mean_activation(latent i | helix-POSITIVE residues)     (CRS-style weighting)

When exactly one latent is selected (the default top-1-F1 case) this is simply that latent's
decoder column. ``v_c`` is then L2-normalised to a unit direction, exactly matching how
``export_probe_direction.fit_direction`` returns a unit direction.

Variants coexisting in one npz (``--out_concept``)
--------------------------------------------------
``--out_concept`` sets the concept name in the output key so multiple variants live side by
side, e.g. ``helix_sae`` for the top-1 latent and ``helix_sae_f1set`` for the F1-above-null set.

Output / plumbing
-----------------
The direction is written into the SAME ``.npz`` the probe exporter uses (default
``../steering/directions/disulfide_helix_directions.npz``), adding two keys:

    helix_sae@trunk_L47        the unit steering direction (float32, shape (384,))
    helix_sae@trunk_L47.mean   the SAE's mean_vector (the demean vector), so the intervention
                               can mean-centre the raw activation before projecting, exactly
                               like the probe directions' ``.mean`` companion.

Because the steering runners (``probe_batch.py`` / ``probe_control.py``) select a direction by
its ``{concept}@{where}`` key via ``--concepts``, the new ``helix_sae`` concept plugs into the
existing steering machinery with **no code change**: pass ``--concepts helix_sae`` (or a key
like ``helix_sae@trunk_L47``) and it steers the SAE direction.

``np.savez`` cannot append, so we load every existing array from the target ``.npz`` first and
re-save them alongside the two new keys (existing keys are preserved / only the two SAE keys
are added or updated).

Held-out fitting
----------------
``--train_ids <file>`` restricts latent selection (F1 or CRS) to a training split of protein IDs
(fit on train, steer on the held-out rest), matching ``export_probe_direction.py`` exactly. It
uses the same benchmark loader (``run_layer_benchmark.load_benchmark_matrices``), so the SAE
latents, helix labels, and per-protein offsets are aligned residue-for-residue.

Heavy dependencies (torch, the SAE, ``run_layer_benchmark``) are imported lazily inside
functions so the module imports cleanly where activations are not staged.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

# Reuse the probe exporter's concept table, location resolver, and cache locations so the two
# exporters stay in lockstep (same labels, same layer keying, same npz layout).
from export_probe_direction import CONCEPT_SPECS, PF_CACHE, _where_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def load_sae_matrices(
    concept: str,
    where: str,
    seed: int,
    activations_dir_template: str = "downloads_layer{layer}",
    max_proteins: int = 0,
    protein_subset: list[str] | set[str] | None = None,
):
    """Return ``(sae_matrix (N, n_latents), label_column (N,))`` for one concept at one location.

    Uses the exact benchmark matrix builder that ``export_probe_direction`` uses, but keeps the
    SAE latent matrix (unused by the probe exporter) instead of the raw matrix. Returns the full
    ``LayerMatrices`` object too so callers can pull the SAE checkpoint details from the same
    (layer, rec) resolution.
    """
    from run_layer_benchmark import load_benchmark_matrices  # heavy deps (torch/tap); lazy import

    if concept not in CONCEPT_SPECS:
        raise ValueError(f"Unknown concept {concept!r}; known: {sorted(CONCEPT_SPECS)}")
    set_name, ann_dir, label_name = CONCEPT_SPECS[concept]
    layer_type, layer, rec, act_dir, cache_dir = _where_spec(where, activations_dir_template)

    if not Path(act_dir).exists():
        raise FileNotFoundError(
            f"Activations dir {act_dir!r} for {where} not found. Stage it first "
            f"(get_activations.py --layer {layer} --rec {rec} for pairformer)."
        )

    m = load_benchmark_matrices(
        layer=layer,
        concept_set_name=set_name,
        annotation_dir=ann_dir,
        activations_dir=act_dir,
        seed=seed,
        rec=rec,
        max_proteins=max_proteins,
        device="cpu",
        cache_dir=cache_dir,
        require_full_rec=False,
        protein_subset=set(protein_subset) if protein_subset is not None else None,
        layer_type=layer_type,
    )
    if m is None:
        raise RuntimeError(f"No matrices for {concept}@{where} (no protein overlap?).")
    if label_name not in m.concept_list:
        raise KeyError(f"Label {label_name!r} not in concept_list for set {set_name!r}: {m.concept_list}")
    y = m.label_matrix[:, m.concept_list.index(label_name)]
    # protein_offsets are needed for the benchmark's per-domain F1 scorer (F1 is computed
    # per protein-domain, not globally), so pass them back to the F1 selection code.
    return m.sae_matrix, y, m.protein_offsets, (layer_type, layer, rec, cache_dir)


def load_sae_decoder_and_mean(
    layer: int,
    rec: int,
    seed: int,
    cache_dir: Path,
    layer_type: str = "pairformer",
) -> tuple[np.ndarray, np.ndarray]:
    """Load the SAE decoder matrix and mean (demean) vector for one (layer, rec, seed).

    The SAE is a :class:`sae_utils.TopKSAE`; ``model.decoder`` is a
    ``torch.nn.Linear(latent_dim, input_dim)``, so ``model.decoder.weight`` has shape
    ``(input_dim=384, latent_dim)`` and column ``[:, i]`` is latent ``i``'s direction in the
    raw 384-dim activation space (``raw ~= decoder.weight @ latents + decoder.bias + mean``).

    Returns:
        ``(decoder (384, n_latents), mean_vector (384,))`` as float32. ``mean_vector`` is the
        training-set demean vector loaded from the checkpoint's ``mean_vector.npy``.
    """
    import torch  # heavy dep; lazy import

    from loaders.sae_latent_loader import load_sae_from_checkpoint, resolve_mean_vector
    from layer_analysis_utils import FINAL_STEP, download_run_checkpoint, repo_for_rec

    repo_id = repo_for_rec(rec) if layer_type == "pairformer" else None
    run_dir = download_run_checkpoint(
        layer, seed=seed, cache_dir=cache_dir, repo_id=repo_id, layer_type=layer_type
    )
    checkpoint_path = run_dir / f"checkpoint_step_{FINAL_STEP}.pt"

    model, cfg = load_sae_from_checkpoint(checkpoint_path, torch.device("cpu"))
    # decoder.weight: (input_dim, latent_dim) -> column i is the raw-space direction of latent i.
    decoder = model.decoder.weight.detach().cpu().numpy().astype(np.float32)

    mean_vec = resolve_mean_vector(checkpoint_path, cfg)
    if mean_vec is None:
        # The trunk SAEs are demean-trained; a missing mean vector means an unexpected config.
        raise ValueError(
            f"SAE at {checkpoint_path} reports demean_embeddings=False; expected a demeaned "
            f"trunk SAE with a mean_vector.npy companion."
        )
    return decoder, np.asarray(mean_vec, dtype=np.float32)


def concept_relevance_scores(
    sae_matrix: np.ndarray,
    label_column: np.ndarray,
    crs_norm: bool = True,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Concept Relevance Score per latent, plus the per-latent mean over positives (weights).

    ``R(i) = mean_activation(latent i | positive residues) - mean_activation(latent i | ALL)``.

    When ``crs_norm`` is True each residue's latent activations are normalised to sum to 1
    across latents (a per-residue distribution over latents) before averaging, so R reflects
    relative firing (the paper's normalisation). When False, raw mean latent activations are
    used instead.

    Returns:
        ``(R (n_latents,), pos_mean (n_latents,))`` where ``pos_mean`` is the mean latent
        activation over positive residues, used as the CRS weighting ``w_i``. When
        ``crs_norm`` is on, ``pos_mean`` is the mean of the normalised activations (kept on the
        same scale as R so the weighting is consistent with the selection statistic).
    """
    acts = np.asarray(sae_matrix, dtype=np.float64)
    pos = np.asarray(label_column).astype(bool)
    if pos.sum() == 0:
        raise ValueError("No positive residues for the concept; cannot compute CRS.")

    if crs_norm:
        # Per-residue normalisation: divide each residue's latent activations by their sum, so
        # each row is a distribution over latents. Residues with all-zero activations (possible
        # under Top-K if every active latent is exactly 0) contribute a zero row rather than NaN.
        row_sums = acts.sum(axis=1, keepdims=True)
        acts = np.divide(acts, row_sums, out=np.zeros_like(acts), where=row_sums > eps)

    all_mean = acts.mean(axis=0)
    pos_mean = acts[pos].mean(axis=0)
    r = pos_mean - all_mean
    return r.astype(np.float64), pos_mean.astype(np.float64)


def per_latent_helix_f1(
    sae_matrix: np.ndarray,
    label_column: np.ndarray,
    protein_offsets: np.ndarray,
    n_perm: int = 200,
    rng_seed: int = 0,
    feature_block: int = 256,
) -> tuple[np.ndarray, float]:
    """Per-latent helix F1 + permutation-null baseline, reusing the benchmark's scorer.

    Reuses ``benchmark_f1.score_best_f1_per_concept`` (the exact statistic ``run_layer_benchmark``
    reports for SAE latents): each latent's activation is a score, the benchmark's percentile
    thresholds (``DEFAULT_THRESHOLD_PCTS``) are swept, and the best-threshold per-domain F1 is
    kept per latent. A label-permutation null (``benchmark_f1.permutation_null``) gives the
    baseline used by ``--f1_above_null``; because each permutation re-takes the max over all
    latents, that null already accounts for the multiple-comparison selection over the dictionary.

    Args:
        sae_matrix: ``(N, n_latents)`` SAE latent activations.
        label_column: ``(N,)`` binary helix labels.
        protein_offsets: ``(P + 1,)`` per-protein boundaries (F1 is per protein-domain).
        n_perm: Permutations for the null (0 to skip; then the null baseline is NaN).
        rng_seed: Seed for the permutation RNG.
        feature_block: Latent columns scored per block (caps peak memory).

    Returns:
        ``(f1 (n_latents,), null_baseline (float))`` where ``null_baseline`` is the 95th
        percentile of the best-F1-over-latents permutation null (NaN if ``n_perm == 0``).
    """
    from benchmark_f1 import (
        DEFAULT_THRESHOLD_PCTS,
        permutation_null,
        precompute_feature_thresholds,
        score_best_f1_per_concept,
    )

    # The benchmark scorer works on a (N, C) label matrix; use a single helix column (C=1).
    labels = np.asarray(label_column).astype(np.uint8).reshape(-1, 1)
    feats = np.asarray(sae_matrix, dtype=np.float32)
    thresholds = precompute_feature_thresholds(feats, DEFAULT_THRESHOLD_PCTS)
    _best_f1, _best_feature, _best_threshold, per_feature_f1 = score_best_f1_per_concept(
        feats, thresholds, labels, protein_offsets,
        threshold_pcts=DEFAULT_THRESHOLD_PCTS, feature_block=feature_block,
        return_per_feature=True,
    )
    f1 = per_feature_f1[:, 0].astype(np.float64)  # (n_latents,) helix F1

    null_baseline = float("nan")
    if n_perm > 0:
        rng = np.random.default_rng(rng_seed)
        null = permutation_null(
            feats, thresholds, labels, protein_offsets,
            n_perm=n_perm, rng=rng, threshold_pcts=DEFAULT_THRESHOLD_PCTS,
            feature_block=feature_block,
        )  # (n_perm, 1)
        null_baseline = float(np.percentile(null[:, 0], 95))
    return f1, null_baseline


def _select_latents(
    stat: np.ndarray,
    k: int | None,
    threshold: float | None,
) -> np.ndarray:
    """Return selected latent indices (descending by ``stat``).

    Exactly one of ``k`` (top-N) or ``threshold`` (all with stat >= threshold) drives the count;
    when ``threshold`` is given it wins. If a threshold selects nothing, fall back to the single
    best latent so the direction is never empty.
    """
    order = np.argsort(stat)[::-1]
    if threshold is not None:
        sel = order[stat[order] >= threshold]
        if sel.size == 0:
            sel = order[:1]  # never return an empty selection
        return sel
    n = 1 if k is None else max(1, int(k))
    return order[: min(n, order.size)]


def build_sae_direction(
    sae_matrix: np.ndarray,
    label_column: np.ndarray,
    decoder: np.ndarray,
    protein_offsets: np.ndarray | None = None,
    select: str = "f1",
    k: int | None = None,
    f1_above_null: bool = False,
    f1_min: float | None = None,
    crs_norm: bool = True,
    n_perm: int = 200,
    rng_seed: int = 0,
) -> tuple[np.ndarray, dict[str, object]]:
    """Select concept latents and combine their decoder columns into a unit steering direction.

    Selection is driven by ``select``:

    * ``"f1"`` -- rank by per-latent helix F1 (needs ``protein_offsets``). Count is data-driven:
      ``f1_above_null`` selects all latents above the permutation-null 95th-percentile baseline;
      ``f1_min`` selects all latents with F1 >= f1_min; otherwise top-``k`` (default k=1, the
      single best latent).
    * ``"crs"`` -- rank by Concept Relevance Score and take top-``k`` (default k=1).

    The combined vector ``sum_i w_i * decoder_column_i`` (``w_i`` = mean activation over
    helix-positive residues) is L2-normalised. When a single latent is selected this is just
    that latent's decoder column (unit-normalised).

    Returns ``(unit_direction (384,), info)`` with the selection method, indices, F1/CRS scores,
    weights, null/threshold, and count in ``info``.
    """
    n_latents = sae_matrix.shape[1]
    if decoder.shape[1] != n_latents:
        raise ValueError(
            f"Decoder/latent mismatch: decoder has {decoder.shape[1]} columns but sae_matrix "
            f"has {n_latents} latents."
        )
    if select not in ("f1", "crs"):
        raise ValueError(f"Unknown --select {select!r}; expected 'f1' or 'crs'.")

    # CRS-style weighting (mean activation over positives) is used to combine columns in BOTH
    # modes, so always compute it. `r` is only used to RANK under --select crs.
    r, pos_mean = concept_relevance_scores(sae_matrix, label_column, crs_norm=crs_norm)

    info: dict[str, object] = {
        "select": select,
        "crs_norm": bool(crs_norm),
        "n_positive": int(np.asarray(label_column).astype(bool).sum()),
        "n_residues": int(sae_matrix.shape[0]),
        "n_latents": int(n_latents),
    }

    if select == "f1":
        if protein_offsets is None:
            raise ValueError("--select f1 requires protein_offsets for the per-domain F1 scorer.")
        f1, null_baseline = per_latent_helix_f1(
            sae_matrix, label_column, protein_offsets, n_perm=n_perm, rng_seed=rng_seed,
        )
        # threshold beats k: --f1_above_null uses the null baseline, --f1_min a fixed cutoff.
        threshold: float | None = None
        if f1_above_null:
            if not np.isfinite(null_baseline):
                raise ValueError("--f1_above_null needs a null baseline; run with n_perm > 0.")
            threshold = null_baseline
            info["selection_mode"] = "f1_above_null"
        elif f1_min is not None:
            threshold = float(f1_min)
            info["selection_mode"] = "f1_min"
        else:
            info["selection_mode"] = "top_k_f1"
        sel = _select_latents(f1, k=k, threshold=threshold)
        info.update({
            "f1_scores": [float(f1[i]) for i in sel],
            "null_baseline_p95": null_baseline,
            "f1_min": None if f1_min is None else float(f1_min),
            "n_perm": int(n_perm),
        })
    else:  # crs
        sel = _select_latents(r, k=k, threshold=None)
        info["selection_mode"] = "top_k_crs"

    weights = pos_mean[sel]  # w_i = mean activation over positive residues (CRS-style weighting)
    # v_c = sum_i w_i * decoder_column_i  ==  decoder[:, sel] @ weights
    v = decoder[:, sel] @ weights.astype(decoder.dtype)
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError("Degenerate SAE direction (zero norm); selected weights or columns are zero.")
    u = (v / norm).astype(np.float32)

    info.update({
        "selected_latents": [int(i) for i in sel],
        "n_selected": int(sel.size),
        "relevance_scores": [float(r[i]) for i in sel],  # CRS for the selected latents (always logged)
        "weights": [float(x) for x in weights],
        "raw_norm": norm,
        "dim": int(u.shape[0]),
    })
    return u, info


def load_existing_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load all arrays + ``__meta__`` from an existing directions ``.npz`` (empty if absent).

    ``np.savez`` cannot append, so every key must be re-saved. We read them all into memory,
    split off ``__meta__`` (the object-array metadata dict), and return the rest as plain
    vectors ready to be merged with the new SAE keys.
    """
    p = Path(path)
    vectors: dict[str, np.ndarray] = {}
    meta: dict[str, object] = {}
    if not p.exists():
        LOGGER.info(msg=f"No existing npz at {p}; creating a new one.")
        return vectors, meta

    with np.load(p, allow_pickle=True) as data:
        for key in data.files:
            if key == "__meta__":
                meta = data["__meta__"].item()
                continue
            vectors[key] = data[key]
    LOGGER.info(msg=f"Loaded {len(vectors)} existing directions from {p}")
    return vectors, meta


def save_directions(path: str | Path, vectors: dict[str, np.ndarray], meta: dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, __meta__=np.array(meta, dtype=object), **vectors)
    LOGGER.info(msg=f"Wrote {len(vectors)} directions -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export an SAE-derived helix steering direction (F1- or CRS-selected latents) to the probe .npz"
    )
    ap.add_argument("--concept", default="helix",
                    help="friendly concept name from CONCEPT_SPECS (default helix = DSSP "
                         "secondary_structure:helix positive control)")
    ap.add_argument("--out_concept", default="helix_sae",
                    help="concept name used in the output npz key (default helix_sae). Change it so "
                         "variants coexist, e.g. helix_sae_f1set for the F1-above-null set. "
                         "probe_batch.py/probe_control.py steer it via --concepts with no code change.")
    ap.add_argument("--layer", type=int, default=47, help="pairformer trunk layer (default 47)")
    ap.add_argument("--select", choices=("f1", "crs"), default="f1",
                    help="latent selection statistic: f1 (default, discrimination quality) or crs "
                         "(concept relevance score, the paper's fires-on-positives method)")
    ap.add_argument("--k", type=int, default=None,
                    help="force top-N latents by the selected statistic. Default (unset) = 1 (the single "
                         "best latent under --select f1). Ignored when --f1_above_null/--f1_min is set.")
    ap.add_argument("--f1_above_null", action="store_true",
                    help="[--select f1] select ALL latents with F1 above the permutation-null 95th "
                         "percentile (data-driven count).")
    ap.add_argument("--f1_min", type=float, default=None,
                    help="[--select f1] select ALL latents with F1 >= this fixed cutoff (data-driven count).")
    ap.add_argument("--n_perm", type=int, default=200,
                    help="label permutations for the F1 null baseline (0 to skip; needed for --f1_above_null)")
    ap.add_argument("--seed", type=int, default=1, help="SAE seed run to use (default 1)")
    ap.add_argument("--rng_seed", type=int, default=0, help="RNG seed for the F1 permutation null")
    ap.add_argument("--crs_norm", dest="crs_norm", action="store_true", default=True,
                    help="per-residue normalise latent activations before CRS/weighting (default; the paper's norm)")
    ap.add_argument("--no_crs_norm", dest="crs_norm", action="store_false",
                    help="use raw mean latent activations for CRS/weighting instead of per-residue normalisation")
    ap.add_argument("--activations_dir_template", default="downloads_layer{layer}")
    ap.add_argument("--max_proteins", type=int, default=0, help="0 = all matched proteins")
    ap.add_argument("--train_ids", default=None,
                    help="file of protein IDs to FIT on (held-out steering: fit on train, steer "
                         "on the rest). Omit to fit on all available proteins.")
    ap.add_argument("--out", default="../steering/directions/disulfide_helix_directions.npz")
    args = ap.parse_args()

    where = f"trunk_L{args.layer}"
    out_key = f"{args.out_concept}@{where}"

    train_ids = set(Path(args.train_ids).read_text().split()) if args.train_ids else None
    if train_ids:
        LOGGER.info(msg=f"Fitting on {len(train_ids)} training proteins (held-out steering).")

    # 1. Load the SAE decoder + demean vector FIRST (fail-fast: the matrix load below takes ~20 min,
    #    so surface any checkpoint/import error in seconds rather than after it).
    layer_type, layer, rec, _act_dir, cache_dir = _where_spec(where, args.activations_dir_template)
    decoder, mean_vec = load_sae_decoder_and_mean(layer, rec, args.seed, cache_dir, layer_type)

    # 2. SAE latents + helix labels + protein offsets aligned per residue (the slow step).
    sae_matrix, y, protein_offsets, _where = load_sae_matrices(
        args.concept, where, args.seed, args.activations_dir_template,
        args.max_proteins, protein_subset=train_ids,
    )

    # 4. Select latents (F1 or CRS) + weighted decoder-column sum, L2-normalised to a unit direction.
    u, info = build_sae_direction(
        sae_matrix, y, decoder,
        protein_offsets=protein_offsets,
        select=args.select,
        k=args.k,
        f1_above_null=args.f1_above_null,
        f1_min=args.f1_min,
        crs_norm=args.crs_norm,
        n_perm=args.n_perm,
        rng_seed=args.rng_seed,
    )
    score_key = "f1_scores" if args.select == "f1" else "relevance_scores"
    LOGGER.info(msg=(
        f"{out_key}: select={info['select']}/{info['selection_mode']} "
        f"n_selected={info['n_selected']} latents={info['selected_latents']} "
        f"{score_key}={[round(x, 5) for x in info[score_key]]} raw_norm={info['raw_norm']:.4f} "
        f"n_pos={info['n_positive']} dim={info['dim']}"
    ))

    # 5. Merge into the existing npz without clobbering other keys, then re-save everything.
    vectors, meta = load_existing_npz(args.out)
    vectors[out_key] = u
    vectors[f"{out_key}.mean"] = mean_vec  # SAE demean vector for mean-centred projection
    meta[out_key] = {
        **info,
        "layer": layer,
        "rec": rec,
        "seed": args.seed,
        "concept": args.concept,
        "concept_set": CONCEPT_SPECS[args.concept][0],
        "source": f"sae_{args.select}",
        "trained_on": "train_ids" if train_ids else "all",
    }
    save_directions(args.out, vectors, meta)
    LOGGER.info(msg=f"Added keys: {out_key}, {out_key}.mean")
    LOGGER.info(msg=f"Directions now: {sorted(vectors)}")


if __name__ == "__main__":
    main()
