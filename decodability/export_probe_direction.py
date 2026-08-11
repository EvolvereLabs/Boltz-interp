#!/usr/bin/env python3
"""Export intervention/steering directions for the causal experiments (`boltz_causal_intervention`).

`linear_probe.py` fits an L2 logistic probe per fold and keeps only the out-of-fold probabilities —
it throws the fitted model away. The causal experiments need the *weight vector* instead, mapped
back into **raw-activation space** and L2-normalised:

    probe fits on StandardScaled features:   p(y=1) = sigmoid(w · z + b),  z = (x - mu) / sigma
    the separating direction in raw x-space:  d = w / sigma        (elementwise)
    unit direction:                           u = d / ||d||

**One probe per (concept, layer).** We fit an independent probe on each layer's raw activations and
key the output `{concept}@trunk_L{n}` (pairformer layer n) or `{concept}@diffusion`, so the steering
runner picks the direction that matches the layer it injects at (the depth sweep). We always include
a **secondary-structure positive control** — the dense DSSP `secondary_structure:helix` direction
(probe-raw ≈0.85, the knife that visibly cuts the fold) — so we can confirm steering works before
trusting the disulfide result.

Reuses the exact benchmark loaders (`run_layer_benchmark.load_benchmark_matrices`) so the raw
activations and labels are aligned residue-for-residue the same way the correlational F1 numbers are.
Fit on **raw** activations (no demean) — the intervention operates on raw `s`, consistent with the
probe-raw readout the paper's claims rest on.

Activations must be staged locally per layer (see `get_activations.py` / `_dl_diffusion.py`):
    pairformer:  downloads_layer{n}/<id>/<subfolder>/output_1.npz     (rec 1)
    diffusion:   downloads_diffusion_rec199/<id>/<subfolder>/output_199.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

PF_CACHE = Path("sae_explore/layer_sweep/hf_cache")
DIFFUSION_CACHE = Path("sae_explore/diffusion_cache/rec199")

# friendly concept name -> (concept_set_name, annotation_dir, label as it appears in concept_list)
# `helix`/`strand` intentionally use the *dense DSSP* labels (the strong steering positive control),
# NOT SwissProt's sparse experimental SS annotation.
CONCEPT_SPECS: dict[str, tuple[str, str, str]] = {
    "disulfide_bond": ("swissprot", "processed_swissprot", "disulfide_bond"),
    "helix": ("secondary", "processed_swissprot_a5_structure_secondary_n500", "secondary_structure:helix"),
    "strand": ("secondary", "processed_swissprot_a5_structure_secondary_n500", "secondary_structure:strand"),
    # coil steering target (increase loop/disorder). If this exact label isn't in the secondary
    # concept_list, fit_direction will raise listing the available labels -- adjust the label then.
    "coil": ("secondary", "processed_swissprot_a5_structure_secondary_n500", "secondary_structure:coil"),
    "signal_peptide": ("swissprot", "processed_swissprot", "signal_peptide"),
}


def fit_direction(
    feature_matrix: np.ndarray,
    label_column: np.ndarray,
    c_reg: float = 1.0,
    max_iter: int = 2000,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit one L2 logistic probe on ALL residues; return the unit raw-space direction + info."""
    scaler = StandardScaler()
    x = scaler.fit_transform(feature_matrix)
    model = LogisticRegression(class_weight="balanced", C=c_reg, max_iter=max_iter, solver="lbfgs")
    model.fit(x, label_column.astype(np.int64))

    w_scaled = model.coef_.reshape(-1)
    scale = np.where(scaler.scale_ == 0.0, 1.0, scaler.scale_)
    d_raw = w_scaled / scale
    norm = float(np.linalg.norm(d_raw))
    if norm == 0.0:
        raise ValueError("Degenerate probe direction (zero norm).")
    u = (d_raw / norm).astype(np.float32)
    info = {
        "raw_norm": norm,
        "train_acc": float(model.score(x, label_column)),
        "n_positive": int(label_column.sum()),
        "n_residues": int(label_column.shape[0]),
        "dim": int(u.shape[0]),
    }
    # The probe logit is d_raw . (x - mean); the concept signal is u . (x - mean), NOT u . x. A few
    # activation dims have huge magnitude, so u . mean dominates u . x and a raw-space ablation removes
    # that constant offset instead of the concept (a random vector does the same -> no specificity).
    # Export the feature mean so the intervention can project the mean-centred activation.
    mean = scaler.mean_.astype(np.float32)
    return u, mean, info


def random_matched(u_ref: np.ndarray, raw_norm: float, seed: int = 0) -> np.ndarray:
    """Random unit direction scaled to the reference raw-space norm (specificity control S1)."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(u_ref.shape).astype(np.float32)
    r /= np.linalg.norm(r)
    return (r * raw_norm).astype(np.float32)


def save_directions(path: str | Path, vectors: dict[str, np.ndarray], meta: dict[str, object]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, __meta__=np.array(meta, dtype=object), **vectors)
    LOGGER.info(msg=f"Wrote {len(vectors)} directions -> {out}")


def _where_spec(where: str, activations_dir_template: str) -> tuple[str, int, int, str, Path]:
    """Resolve a location key into (layer_type, layer, rec, activations_dir, cache_dir)."""
    if where == "diffusion":
        return "diffusion", 22, 199, "downloads_diffusion_rec199", DIFFUSION_CACHE
    if where.startswith("trunk_L"):
        layer = int(where.removeprefix("trunk_L"))
        return "pairformer", layer, 1, activations_dir_template.format(layer=layer), PF_CACHE
    raise ValueError(f"Unknown location {where!r} (expected 'trunk_L<n>' or 'diffusion').")


def load_activations_and_labels(
    concept: str,
    where: str,
    activations_dir_template: str = "downloads_layer{layer}",
    max_proteins: int = 0,
    protein_subset: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(raw_features (N,D), label_column (N,))`` for one concept at one location.

    Uses the benchmark's own matrix builder so activations/labels align exactly as in the F1 runs.
    The SAE matrix it also builds is unused here (we only need the raw activations).
    """
    from run_layer_benchmark import load_benchmark_matrices  # heavy deps (torch/tap); import lazily

    if concept not in CONCEPT_SPECS:
        raise ValueError(f"Unknown concept {concept!r}; known: {sorted(CONCEPT_SPECS)}")
    set_name, ann_dir, label_name = CONCEPT_SPECS[concept]
    layer_type, layer, rec, act_dir, cache_dir = _where_spec(where, activations_dir_template)

    if not Path(act_dir).exists():
        raise FileNotFoundError(
            f"Activations dir {act_dir!r} for {where} not found. Stage it first "
            f"(get_activations.py --layer {layer} --rec {rec} for pairformer; _dl_diffusion.py for diffusion)."
        )

    m = load_benchmark_matrices(
        layer=layer,
        concept_set_name=set_name,
        annotation_dir=ann_dir,
        activations_dir=act_dir,
        seed=1,
        rec=rec,
        max_proteins=max_proteins,
        device="cpu",
        cache_dir=cache_dir,
        require_full_rec=False,
        protein_subset=protein_subset,  # restrict fitting to the training split (held-out steering)
        layer_type=layer_type,
    )
    if m is None:
        raise RuntimeError(f"No matrices for {concept}@{where} (no protein overlap?).")
    if label_name not in m.concept_list:
        raise KeyError(f"Label {label_name!r} not in concept_list for set {set_name!r}: {m.concept_list}")
    y = m.label_matrix[:, m.concept_list.index(label_name)]
    return m.raw_matrix, y


def main() -> None:
    ap = argparse.ArgumentParser(description="Export per-layer causal-intervention directions to .npz")
    ap.add_argument("--concepts", default="disulfide_bond,helix",
                    help="friendly concept names (helix = DSSP secondary-structure positive control)")
    ap.add_argument("--trunk_layers", default="10,16,24,32,47")
    ap.add_argument("--include_diffusion", action="store_true",
                    help="also fit each requested concept in diffusion space (C2 control; "
                         "helix@diffusion enables the diffusion-module arm of the causal 2x2)")
    ap.add_argument("--activations_dir_template", default="downloads_layer{layer}")
    ap.add_argument("--max_proteins", type=int, default=0, help="0 = all matched proteins")
    ap.add_argument("--train_ids", default=None,
                    help="file of protein IDs to FIT on (held-out steering: fit on train, steer on the "
                         "rest). Omit to fit on all available proteins.")
    ap.add_argument("--out", default="../steering/directions/disulfide_helix_directions.npz")
    args = ap.parse_args()

    concepts = [c for c in args.concepts.split(",") if c]
    trunk_layers = [int(x) for x in args.trunk_layers.split(",") if x]
    train_ids = set(Path(args.train_ids).read_text().split()) if args.train_ids else None  # loader intersects as a set
    if train_ids:
        LOGGER.info(msg=f"Fitting on {len(train_ids)} training proteins (held-out steering).")

    vectors: dict[str, np.ndarray] = {}
    meta: dict[str, object] = {"c_reg": 1.0, "space": "raw", "trunk_layers": trunk_layers, "concepts": concepts}

    for concept in concepts:
        for layer in trunk_layers:
            where = f"trunk_L{layer}"
            x, y = load_activations_and_labels(concept, where, args.activations_dir_template,
                                               args.max_proteins, protein_subset=train_ids)
            u, mean, info = fit_direction(x, y)
            vectors[f"{concept}@{where}"] = u
            vectors[f"{concept}@{where}.mean"] = mean  # feature mean for mean-centred projection
            meta[f"{concept}@{where}"] = {**info, "layer": layer, "concept_set": CONCEPT_SPECS[concept][0]}
            LOGGER.info(msg=f"{concept}@{where}: raw_norm={info['raw_norm']:.3f} "
                            f"train_acc={info['train_acc']:.3f} n_pos={info['n_positive']} dim={info['dim']}")
        if args.include_diffusion:  # fit every requested concept in diffusion space, not just disulfide
            x, y = load_activations_and_labels(concept, "diffusion", args.activations_dir_template,
                                               args.max_proteins, protein_subset=train_ids)
            u, mean, info = fit_direction(x, y)
            vectors[f"{concept}@diffusion"] = u
            vectors[f"{concept}@diffusion.mean"] = mean
            meta[f"{concept}@diffusion"] = {**info, "layer": 22, "concept_set": "swissprot_diffusion"}
            LOGGER.info(msg=f"{concept}@diffusion: raw_norm={info['raw_norm']:.3f} dim={info['dim']}")

    # random matched-norm control keyed to the final trunk layer (specificity control S1); it reuses
    # the reference concept's mean so the random control is mean-centred identically (fair comparison).
    ref_key = f"disulfide_bond@trunk_L{trunk_layers[-1]}"
    if ref_key in vectors:
        ref_norm = float(meta[ref_key]["raw_norm"])  # type: ignore[index]
        vectors[f"random@trunk_L{trunk_layers[-1]}"] = random_matched(vectors[ref_key], ref_norm)
        vectors[f"random@trunk_L{trunk_layers[-1]}.mean"] = vectors[f"{ref_key}.mean"]
        LOGGER.info(msg=f"Added random@trunk_L{trunk_layers[-1]} (matched norm {ref_norm:.3f})")

    save_directions(args.out, vectors, meta)
    LOGGER.info(msg=f"Directions: {sorted(vectors)}")


if __name__ == "__main__":
    main()
