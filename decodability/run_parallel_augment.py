#!/usr/bin/env python3
"""Parallel driver for ``augment_benchmark.py`` across ``(layer, rec)`` on CPU cores.

Augmenting a benchmark JSON is independent per ``(layer, rec)`` and dominated by
reloading that layer's activations (protein IO + a cheap SAE forward), so the jobs
fan out cleanly across cores. As in ``run_parallel_benchmark.py``:

1. **BLAS threads are capped per worker** (``OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS``)
   so N processes don't oversubscribe the cores. The augment work is lighter than the
   full benchmark (no permutation null), so more workers fit in RAM -- but the loading
   is IO-bound, so raising ``max_workers`` helps more than ``threads_per_worker`` here.
2. **No per-worker index writes** (``--write_index False``); the shared index is rebuilt
   once by the parent after every worker finishes, avoiding a write race.

Each worker augments one ``(layer, rec)`` for all requested concept sets, patching
``probe_sae_f1`` + precision/recall in place (originals untouched).

Example::

    uv run python run_parallel_augment.py \\
        --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47 \\
        --recs 1 --concept_sets secondary,swissprot \\
        --activations_dir_template "downloads_layer{layer}" \\
        --protein_ids_file common_activation_proteins.txt \\
        --max_workers 8 --threads_per_worker 2
"""

import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tap import tapify

from run_layer_benchmark import output_root_for, rebuild_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

SCRIPT = str(Path(__file__).parent / "augment_benchmark.py")


def _thread_env(threads_per_worker: int) -> dict[str, str]:
    """Cap each worker's BLAS thread pools so workers don't oversubscribe cores."""
    value = str(max(1, threads_per_worker))
    return {
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
    }


def _parse_ints(value: str) -> list[int]:
    return [int(token) for token in str(value).replace(" ", "").split(",") if token]


def _run_one(
    layer: int,
    rec: int,
    concept_sets: str,
    activations_dir_template: str,
    protein_ids_file: str | None,
    max_proteins: int,
    recompute_probe_sae: bool,
    recompute_pr: bool,
    threads_per_worker: int,
) -> tuple[int, int, int, float, str]:
    """Augment one ``(layer, rec)`` as a thread-capped subprocess."""
    cmd = [
        sys.executable, SCRIPT,
        "--layers", str(layer),
        "--recs", str(rec),
        "--concept_sets", concept_sets,
        "--activations_dir_template", activations_dir_template,
        "--recompute_probe_sae", str(recompute_probe_sae),
        "--recompute_pr", str(recompute_pr),
        "--write_index", "False",  # rebuilt once by the parent after all workers finish
    ]
    if protein_ids_file:
        cmd += ["--protein_ids_file", protein_ids_file]
    if max_proteins:
        cmd += ["--max_proteins", str(max_proteins)]

    env = {**os.environ, **_thread_env(threads_per_worker)}
    start = time.monotonic()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    tail = (proc.stderr or proc.stdout or "")[-800:]
    return layer, rec, proc.returncode, elapsed, tail


def run_parallel_augment(
    layers: str = "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47",
    recs: str = "1",
    concept_sets: str = "secondary,swissprot",
    activations_dir_template: str = "downloads_layer{layer}",
    protein_ids_file: str | None = None,
    max_workers: int = 8,
    threads_per_worker: int = 2,
    max_proteins: int = 0,
    recompute_probe_sae: bool = True,
    recompute_pr: bool = True,
) -> int:
    """Fan ``augment_benchmark`` out over ``(layer, rec)`` with a worker cap.

    Args:
        layers: Comma-separated pairformer layers.
        recs: Comma-separated recycle indices (both rec SAE repos are public; no token needed).
        concept_sets: Comma-separated sets from ``{secondary, swissprot}``.
        activations_dir_template: Activation dir; may contain ``{layer}``. Must be the SAME
            activations the original benchmark used or the per-file alignment guard aborts it.
        protein_ids_file: Fixed protein-ID list -- pass the SAME file the original run used.
        max_workers: Concurrent layer-jobs. Each holds the layer's latent matrix (~1.5 GB for
            secondary, less for swissprot); far lighter than the benchmark (no null), so RAM
            allows more workers. Loading is IO-bound, so this is the main throughput lever.
        threads_per_worker: BLAS threads per worker (mostly helps the probe fit, not loading).
        max_proteins: Cap on matched proteins (0 = all); must match the original run or the
            rebuilt residue count won't align and the file is skipped.
        recompute_probe_sae: Train + patch the SAE-latent probe (alive latents only).
        recompute_pr: Patch precision/recall for the stored best SAE latent and neuron.

    Returns:
        Exit code; 0 only if every job succeeded.
    """
    if protein_ids_file and not Path(protein_ids_file).exists():
        LOGGER.error(msg=f"protein_ids_file '{protein_ids_file}' not found.")
        return 1

    layer_list = _parse_ints(layers)
    rec_list = _parse_ints(recs)
    jobs = [(layer, rec) for rec in rec_list for layer in layer_list]
    LOGGER.info(
        msg=f"Launching {len(jobs)} augment jobs ({len(layer_list)} layers x {len(rec_list)} recs) "
            f"across {max_workers} workers x {threads_per_worker} thread(s)"
    )

    results: list[tuple[int, int, int]] = []
    wall_start = time.monotonic()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_one, layer, rec, concept_sets, activations_dir_template,
                protein_ids_file, max_proteins, recompute_probe_sae, recompute_pr,
                threads_per_worker,
            ): (layer, rec)
            for (layer, rec) in jobs
        }
        for future in as_completed(futures):
            layer, rec, code, elapsed, tail = future.result()
            done += 1
            status = "ok" if code == 0 else "FAILED"
            LOGGER.info(msg=f"[{done}/{len(jobs)}] {status}: layer {layer} rec {rec}  ({elapsed:.0f}s)")
            if code != 0:
                LOGGER.error(msg=f"  layer {layer} rec {rec} stderr/out tail:\n{tail}")
            results.append((layer, rec, code))

    LOGGER.info(msg=f"All jobs finished in {time.monotonic() - wall_start:.0f}s; rebuilding indexes")
    for rec in rec_list:
        index_path = rebuild_index(output_root_for(rec))
        LOGGER.info(msg=f"  rec{rec} index -> {index_path}")

    failed = [(layer, rec) for (layer, rec, code) in results if code != 0]
    if failed:
        LOGGER.error(msg=f"{len(failed)}/{len(jobs)} job(s) FAILED: {failed}")
        return 1
    LOGGER.info(msg=f"All {len(jobs)} jobs succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(tapify(run_parallel_augment, explicit_bool=True))
