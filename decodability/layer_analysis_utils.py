#!/usr/bin/env python3
"""Helpers for analysing the per-layer Boltz pairformer SAEs hosted on HuggingFace.

The SAEs live in the public HuggingFace model repo ``evolve-away/Boltz1-SAEs-L2``.
For every analysed pairformer layer there is one folder ``layer{N}/`` containing
three seed runs plus a layer-level cross-seed consistency report::

    layer{N}/
        pairformer{N}_topk256_lat2048_demean_longtrain500000_l2_3e-3_seed{1,2,3}/
            checkpoint_step_500000.pt   # SAE weights (18.9 MB)
            config.json                 # training config
            eval_step_500000.json       # reconstruction / sparsity metrics
            mean_vector.npy             # demeaning vector
            stats.jsonl                 # 1001-point training curve
        pairformer{N}_..._l2_3e-3_alive_cross_seed.json

This module downloads (and locally caches) the small JSON / JSONL metadata files
and parses them into tidy pandas tables. It only needs ``requests`` -- no
``huggingface_hub`` install and no GPU -- so the overview analysis runs anywhere.

Use :func:`download_run_checkpoint` to also fetch the heavy ``.pt`` checkpoint
plus its ``config.json`` / ``mean_vector.npy`` when you need to run the SAE
itself (see ``evaluate_f1_across_layers.py``).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# --- HuggingFace repo layout -------------------------------------------------
REPO_ID = "evolve-away/Boltz1-SAEs-L2"
RESOLVE_BASE = f"https://huggingface.co/{REPO_ID}/resolve/main"

# SAEs trained on a given recycle of the activations live in separate repos.
# ``rec`` here is the Boltz recycle index whose activations the SAE was trained on,
# NOT just which activations the *benchmark* loads -- the two must match.
#   rec 1 -> the original public repo (Boltz1-SAEs-L2)
#   rec 0 -> the rec0-retrained repo (Boltz1-SAEs-L2-rec0; now public too)
# Edit this mapping if a repo name is wrong; a wrong entry silently reintroduces
# the "results don't change between recs" bug because the SAE weights won't differ.
REC_REPO_IDS: dict[int, str] = {
    1: "evolve-away/Boltz1-SAEs-L2",
    0: "evolve-away/Boltz1-SAEs-L2-rec0",
}

# Optional HuggingFace token, kept for private repos / higher rate limits. Both SAE
# repos are currently public, so this is normally unset. Read from the environment
# so it never gets committed.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def repo_for_rec(rec: int) -> str:
    """Return the SAE HF repo whose model was trained on the given recycle index."""
    if rec not in REC_REPO_IDS:
        raise ValueError(f"No SAE repo configured for rec={rec}; known: {sorted(REC_REPO_IDS)}")
    return REC_REPO_IDS[rec]


def resolve_base_for(repo_id: str | None) -> str:
    """Build the ``/resolve/main`` base URL for a repo (defaults to ``REPO_ID``)."""
    return f"https://huggingface.co/{repo_id or REPO_ID}/resolve/main"

# Pairformer layers with a trained SAE (every 2nd layer, plus the final layer 47).
LAYERS: list[int] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 47]
SEEDS: list[int] = [1, 2, 3]
L2_TAG = "3e-3"
FINAL_STEP = 500000

# Local cache for downloaded HF artefacts.
CACHE_DIR = Path("sae_explore/layer_sweep/hf_cache")


def run_name(layer: int, seed: int, layer_type: str = "pairformer") -> str:
    """Return the seed-run directory name for a layer (matches the training folders).

    ``layer_type`` tags the run ("pairformer10_..." / "diffusion10_...") so diffusion
    runs never collide with pairformer ones; it must match the tag the training
    workflow (run_full_workflow.sh) wrote.
    """
    return f"{layer_type}{layer}_topk256_lat2048_demean_longtrain{FINAL_STEP}_l2_{L2_TAG}_seed{seed}"


def cross_seed_name(layer: int, layer_type: str = "pairformer") -> str:
    """Return the layer-level cross-seed JSON file name."""
    return f"{layer_type}{layer}_topk256_lat2048_demean_longtrain{FINAL_STEP}_l2_{L2_TAG}_alive_cross_seed.json"


def layer_subfolder(layer: int, layer_type: str = "pairformer") -> str:
    """Return the raw-activation S3 subfolder name for a layer.

    Must match ``get_activations.py:get_layer_folder_name``.
    """
    if layer_type == "diffusion":
        return (
            f"DiffusionTransformerLayer_{layer} from DiffusionTransformer "
            f"from DiffusionModule from AtomDiffusion"
        )
    return f"PairformerLayer_{layer} s from PairformerLayer"


# --- Download + cache --------------------------------------------------------
def _download(
    rel_path: str,
    cache_dir: Path = CACHE_DIR,
    force: bool = False,
    retries: int = 4,
    resolve_base: str | None = None,
    token: str | None = None,
) -> Path:
    """Download one repo-relative file from HF into the local cache.

    Args:
        rel_path: Path within the repo, e.g. ``layer0/<run>/eval_step_500000.json``.
        cache_dir: Root directory for cached downloads.
        force: Re-download even if a cached copy exists.
        retries: Number of attempts on transient HTTP errors.
        resolve_base: ``/resolve/main`` base URL; defaults to the public ``REPO_ID``.
        token: Optional HF bearer token (only needed if a repo is made private).

    Returns:
        Local path to the cached file.
    """
    local_path = cache_dir / rel_path
    if local_path.exists() and not force:
        return local_path

    url = f"{resolve_base or RESOLVE_BASE}/{rel_path}"
    headers = {"Authorization": f"Bearer {token}"} if token else None
    local_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as resp:
                resp.raise_for_status()
                tmp_path = local_path.with_suffix(local_path.suffix + ".part")
                with tmp_path.open("wb") as handle:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
                tmp_path.replace(local_path)
            return local_path
        except (requests.RequestException, OSError) as exc:  # noqa: PERF203
            last_error = exc
            wait = 2 ** attempt
            LOGGER.warning(msg=f"Download failed ({rel_path}) attempt {attempt + 1}: {exc}; retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {url}") from last_error


def fetch_json(rel_path: str, **kwargs) -> dict:
    """Download (cached) and parse a JSON file from the repo."""
    return json.loads(_download(rel_path, **kwargs).read_text(encoding="utf-8"))


def fetch_jsonl(rel_path: str, **kwargs) -> list[dict]:
    """Download (cached) and parse a JSONL file from the repo."""
    text = _download(rel_path, **kwargs).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --- Tidy tables -------------------------------------------------------------
_EVAL_FIELDS = [
    "mean_mse",
    "input_rms",
    "baseline_mse_zero",
    "baseline_mse_mean",
    "relative_mse_zero",
    "relative_mse_mean",
    "active_feature_frac",
    "fraction_features_active",
    "total_tokens",
]


def load_eval_table(layers: list[int] = LAYERS, seeds: list[int] = SEEDS) -> pd.DataFrame:
    """Per-(layer, seed) reconstruction and sparsity metrics from ``eval_step``.

    Columns: ``layer``, ``seed`` plus the fields in ``_EVAL_FIELDS``. ``dead_feature_frac``
    is derived as ``1 - fraction_features_active``.
    """
    rows: list[dict] = []
    for layer in layers:
        for seed in seeds:
            rel = f"layer{layer}/{run_name(layer, seed)}/eval_step_{FINAL_STEP}.json"
            try:
                data = fetch_json(rel)
            except RuntimeError as exc:
                LOGGER.warning(msg=f"Skipping eval for layer {layer} seed {seed}: {exc}")
                continue
            row = {"layer": layer, "seed": seed}
            row.update({field: data.get(field) for field in _EVAL_FIELDS})
            row["dead_feature_frac"] = (
                1.0 - row["fraction_features_active"] if row["fraction_features_active"] is not None else None
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["layer", "seed"]).reset_index(drop=True)


def load_training_curves(layers: list[int] = LAYERS, seeds: list[int] = SEEDS) -> pd.DataFrame:
    """Long-format training curves (one row per logged step) from ``stats.jsonl``.

    Columns: ``layer``, ``seed``, ``step``, ``loss``, ``mse_loss``,
    ``active_feature_frac``, ``l2_penalty``.
    """
    frames: list[pd.DataFrame] = []
    for layer in layers:
        for seed in seeds:
            rel = f"layer{layer}/{run_name(layer, seed)}/stats.jsonl"
            try:
                records = fetch_jsonl(rel)
            except RuntimeError as exc:
                LOGGER.warning(msg=f"Skipping stats for layer {layer} seed {seed}: {exc}")
                continue
            frame = pd.DataFrame(records)
            frame.insert(0, "seed", seed)
            frame.insert(0, "layer", layer)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_cross_seed_table(layers: list[int] = LAYERS, thresholds: tuple[str, ...] = ("0.7", "0.8", "0.9")) -> pd.DataFrame:
    """Per-layer cross-seed consistency from the ``alive_cross_seed`` reports.

    Reads ``aggregate_fraction_consistent`` (mean over seed pairs of the fraction
    of decoder/encoder directions with a cosine match above each threshold).

    Columns: ``layer`` plus ``decoder_consistent_{t}`` / ``encoder_consistent_{t}``
    for each threshold ``t``, and ``mean_pair_decoder_cosine`` averaged over pairs.
    """
    rows: list[dict] = []
    for layer in layers:
        rel = f"layer{layer}/{cross_seed_name(layer)}"
        try:
            data = fetch_json(rel)
        except RuntimeError as exc:
            LOGGER.warning(msg=f"Skipping cross-seed for layer {layer}: {exc}")
            continue
        agg = data.get("aggregate_fraction_consistent", {})
        row: dict = {"layer": layer}
        for thr in thresholds:
            entry = agg.get(thr, {})
            row[f"decoder_consistent_{thr}"] = entry.get("decoder_fraction_consistent_mean")
            row[f"encoder_consistent_{thr}"] = entry.get("encoder_fraction_consistent_mean")
        pair_cosines = [p.get("decoder", {}).get("mean_cosine") for p in data.get("pairs", [])]
        pair_cosines = [c for c in pair_cosines if c is not None]
        row["mean_pair_decoder_cosine"] = sum(pair_cosines) / len(pair_cosines) if pair_cosines else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values("layer").reset_index(drop=True)


def load_config(layer: int, seed: int = 1) -> dict:
    """Return the training ``config.json`` for one run."""
    return fetch_json(f"layer{layer}/{run_name(layer, seed)}/config.json")


def download_run_checkpoint(
    layer: int,
    seed: int = 1,
    cache_dir: Path = CACHE_DIR,
    repo_id: str | None = None,
    token: str | None = None,
    layer_type: str = "pairformer",
) -> Path:
    """Download a run's checkpoint plus its ``config.json`` / ``mean_vector.npy``.

    Args:
        layer: Layer index.
        seed: SAE seed run to fetch.
        cache_dir: Root directory for cached downloads. Use a rec-specific dir so
            checkpoints from different ``repo_id``s never share a cache slot.
        repo_id: SAE repo to pull from (defaults to the public ``REPO_ID``). Pass
            ``repo_for_rec(rec)`` to get the SAE trained on a given recycle. Ignored
            when ``layer_type`` is not pairformer (those have no HF mirror).
        token: Optional HF bearer token (defaults to ``HF_TOKEN`` from the
            environment; only needed for private repos).
        layer_type: ``"pairformer"`` (HF-backed) or ``"diffusion"``. Diffusion SAEs
            are only on S3, so they must be pre-staged into ``cache_dir`` by the
            analysis workflow; this function refuses to fall back to the pairformer
            HF repo for them (which would silently load the wrong weights).

    Returns the local run directory containing ``checkpoint_step_{FINAL_STEP}.pt``,
    ``config.json`` and ``mean_vector.npy`` -- the layout expected by
    ``eval_sae.load_config_from_run`` and ``SAELatentActivationLoader``.
    """
    run_dir = f"layer{layer}/{run_name(layer, seed, layer_type)}"
    fnames = ("config.json", "mean_vector.npy", f"checkpoint_step_{FINAL_STEP}.pt")
    local_dir = cache_dir / run_dir

    if layer_type != "pairformer":
        # No HuggingFace mirror for non-pairformer SAEs -- they live only on S3 and
        # must have been staged into cache_dir by run_analysis_workflow.sh. Refuse to
        # silently download pairformer weights from HF if the stage step was skipped.
        missing = [f for f in fnames if not (local_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"{layer_type} checkpoint not staged at {local_dir}: missing {missing}. "
                f"Stage it from S3 first (run_analysis_workflow.sh step 5); there is no HF fallback."
            )
        LOGGER.info(msg=f"Checkpoint ready at {local_dir} (staged from S3, layer_type={layer_type})")
        return local_dir

    resolve_base = resolve_base_for(repo_id)
    token = token if token is not None else HF_TOKEN
    for fname in fnames:
        _download(f"{run_dir}/{fname}", cache_dir=cache_dir, resolve_base=resolve_base, token=token)
    LOGGER.info(msg=f"Checkpoint ready at {local_dir} (repo={repo_id or REPO_ID})")
    return local_dir


if __name__ == "__main__":
    # Quick smoke test: print a compact per-layer overview table.
    eval_df = load_eval_table()
    summary = (
        eval_df.groupby("layer")[["relative_mse_mean", "fraction_features_active"]]
        .mean()
        .round(4)
    )
    LOGGER.info(msg=f"Loaded eval metrics for {eval_df['layer'].nunique()} layers")
    print(summary.to_string())
