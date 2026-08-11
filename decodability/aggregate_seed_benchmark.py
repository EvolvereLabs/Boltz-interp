#!/usr/bin/env python3
"""Average the per-seed benchmark JSONs into one ``*_agg.json`` per (layer, concept set).

``run_layer_benchmark.py`` now writes one file per SAE seed
(``{set}_seed{seed}_benchmark.json``). Only the SAE-derived numbers (``sae.*`` and
``probe_sae_f1``) change across seeds -- each seed is a *different* trained SAE -- while the
single-neuron baseline (``neuron.*``) and the raw-activation probe (``probe_raw_f1``) are
seed-independent (their RNG is ``rng_seed``, not the SAE ``seed``). So aggregating means:

* **SAE / SAE-probe**: report ``mean`` and (sample) ``std`` across seeds. The std on
  ``sae_sig = f1 - null_mean`` is the error bar that tells you whether a layer-to-layer F1
  difference is real or just seed noise.
* **neuron / probe_raw**: carried straight through, with an assertion that they actually match
  across seeds (a mismatch means something leaked the SAE seed into a baseline -- a bug).

Outputs, per rec, mirror the per-seed tree:
    sae_explore/layer_sweep_rec{rec}/benchmark/layer{N}/{set}_agg.json
plus a tidy ``benchmark_agg_index.jsonl`` the notebook reads for the error-bar plots.

Example::

    uv run python aggregate_seed_benchmark.py --recs 0,1 --seeds 1,2,3 \\
        --concept_sets secondary,swissprot
"""

import json
import logging
from pathlib import Path

import numpy as np
from tap import tapify

from run_layer_benchmark import output_root_for

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# Tolerance for "are the seed-invariant baselines really identical across seeds?".
_BASELINE_TOL = 1e-6


def _parse_ints(value: str) -> list[int]:
    return [int(token) for token in str(value).replace(" ", "").split(",") if token]


def _mean_std(values: list[float]) -> tuple[float, float, int]:
    """Mean, sample std (ddof=1 when n>1 else 0.0), and n over the finite values."""
    finite = [v for v in values if v is not None and np.isfinite(v)]
    n = len(finite)
    if n == 0:
        return float("nan"), float("nan"), 0
    arr = np.asarray(finite, dtype=float)
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    return float(arr.mean()), std, n


def _check_invariant(name: str, values: list[float], layer: int, concept: str) -> float | None:
    """Return the (shared) value of a seed-invariant field, warning if seeds disagree."""
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None
    spread = max(finite) - min(finite)
    if spread > _BASELINE_TOL:
        LOGGER.warning(
            msg=(
                f"layer {layer}/{concept}: '{name}' varies across seeds by {spread:.2e} "
                f"(expected seed-invariant); using the mean. Values: {finite}"
            )
        )
    return float(np.mean(finite))


def _aggregate_concept(layer: int, concept: str, entries: list[dict]) -> dict:
    """Aggregate one concept across the per-seed entries (one entry per seed)."""
    sae_f1 = [(e.get("sae") or {}).get("f1") for e in entries]
    sae_null = [(e.get("sae") or {}).get("null_mean") for e in entries]
    sae_sig = [
        (f1 - nm) if (f1 is not None and nm is not None) else None
        for f1, nm in zip(sae_f1, sae_null)
    ]
    sae_p = [(e.get("sae") or {}).get("p_value") for e in entries]
    sae_feat = [(e.get("sae") or {}).get("feature_idx") for e in entries]
    probe_sae = [e.get("probe_sae_f1") for e in entries]

    f1_mean, f1_std, n_seeds = _mean_std(sae_f1)
    null_mean, _, _ = _mean_std(sae_null)
    sig_mean, sig_std, _ = _mean_std(sae_sig)
    psae_mean, psae_std, _ = _mean_std(probe_sae)

    # The winning latent index may differ by seed (different dictionaries); record them all so
    # the consistency study can ask whether the *same* concept-direction recurs across seeds.
    winning_idx = [int(i) for i in sae_feat if i is not None]

    return {
        "n_seeds": n_seeds,
        "sae_f1_mean": f1_mean,
        "sae_f1_std": f1_std,
        "sae_null_mean": null_mean,
        "sae_sig_mean": sig_mean,        # mean of (f1 - null_mean) across seeds
        "sae_sig_std": sig_std,          # the error bar that separates signal from seed noise
        "sae_p_value_max": float(max([p for p in sae_p if p is not None], default=float("nan"))),
        "sae_winning_idx_per_seed": winning_idx,
        "probe_sae_f1_mean": psae_mean,
        "probe_sae_f1_std": psae_std,
        # seed-invariant baselines (carried through with a consistency check)
        "neuron_f1": _check_invariant("neuron.f1", [(e.get("neuron") or {}).get("f1") for e in entries], layer, concept),
        "neuron_null_mean": _check_invariant("neuron.null_mean", [(e.get("neuron") or {}).get("null_mean") for e in entries], layer, concept),
        "probe_raw_f1": _check_invariant("probe_raw_f1", [e.get("probe_raw_f1") for e in entries], layer, concept),
    }


def _agg_long_rows(summary: dict) -> list[dict]:
    """Tidy long-format rows (one per layer/concept) from one aggregated summary."""
    rows: list[dict] = []
    for concept, agg in summary["per_concept"].items():
        rows.append({
            "rec": summary["rec"],
            "layer": summary["layer"],
            "concept_set": summary["concept_set"],
            "concept": concept,
            **agg,
        })
    return rows


def aggregate_seed_benchmark(
    recs: str = "0,1",
    seeds: str = "1,2,3",
    concept_sets: str = "secondary,swissprot",
    write_index: bool = True,
    layer_type: str = "pairformer",
) -> int:
    """Aggregate per-seed benchmark JSONs into ``*_agg.json`` plus a long-format index.

    Args:
        recs: Comma-separated recycle indices to aggregate.
        layer_type: ``"pairformer"`` (default) or ``"diffusion"``; selects the per-type
            ``sae_explore/[<type>_]layer_sweep_rec{rec}/benchmark`` tree to aggregate.
        seeds: Comma-separated SAE seeds whose per-seed JSONs should be combined.
        concept_sets: Comma-separated concept sets (file stems) to aggregate.
        write_index: Write ``benchmark_agg_index.jsonl`` per rec.

    Returns:
        Exit code; 0 if at least one (layer, concept set) was aggregated.
    """
    rec_list = _parse_ints(recs)
    seed_list = _parse_ints(seeds)
    set_names = [s for s in concept_sets.replace(" ", "").split(",") if s]
    produced = 0

    for rec in rec_list:
        root = output_root_for(rec, layer_type)
        if not root.exists():
            LOGGER.warning(msg=f"rec{rec}: {root} does not exist; skipping.")
            continue
        index_rows: list[dict] = []
        for layer_dir in sorted(root.glob("layer*")):
            if not layer_dir.is_dir():
                continue
            layer = int(layer_dir.name.removeprefix("layer"))
            for name in set_names:
                per_seed: list[dict] = []
                found_seeds: list[int] = []
                for seed in seed_list:
                    path = layer_dir / f"{name}_seed{seed}_benchmark.json"
                    if path.exists():
                        per_seed.append(json.loads(path.read_text(encoding="utf-8")))
                        found_seeds.append(seed)
                if not per_seed:
                    continue
                if len(found_seeds) < len(seed_list):
                    LOGGER.warning(
                        msg=f"layer {layer}/{name} rec{rec}: only seeds {found_seeds} present "
                            f"(requested {seed_list}); aggregating over the available ones."
                    )
                concepts = sorted({c for e in per_seed for c in e["per_concept"]})
                per_concept = {
                    concept: _aggregate_concept(
                        layer, concept, [e["per_concept"][concept] for e in per_seed if concept in e["per_concept"]]
                    )
                    for concept in concepts
                }
                base = per_seed[0]
                summary = {
                    "layer": layer,
                    "rec": rec,
                    "concept_set": name,
                    "seeds": found_seeds,
                    "n_proteins": base.get("n_proteins"),
                    "n_residues": base.get("n_residues"),
                    "n_neurons": base.get("n_neurons"),
                    "n_latents": base.get("n_latents"),
                    "n_perm": base.get("n_perm"),
                    "per_concept": per_concept,
                }
                (layer_dir / f"{name}_agg.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                index_rows.extend(_agg_long_rows(summary))
                produced += 1
                LOGGER.info(msg=f"Aggregated layer {layer}/{name} rec{rec} over seeds {found_seeds}")

        if write_index and index_rows:
            index_path = root / "benchmark_agg_index.jsonl"
            with index_path.open("w", encoding="utf-8") as handle:
                for row in index_rows:
                    handle.write(json.dumps(row) + "\n")
            LOGGER.info(msg=f"rec{rec} aggregate index -> {index_path}")

    if produced == 0:
        LOGGER.error(msg="No per-seed benchmark JSONs found to aggregate.")
        return 1
    LOGGER.info(msg=f"Aggregated {produced} (layer, concept set) combinations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(tapify(aggregate_seed_benchmark, explicit_bool=True))
