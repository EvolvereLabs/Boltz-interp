#!/usr/bin/env python3
"""Parallel driver for ``run_layer_benchmark.py`` across ``(layer, rec)`` on CPU cores.

Each ``(layer, rec)`` is an independent job writing to its own file, so they fan out
across cores. The work is pure NumPy/sklearn (no GPU), so the two things that make this
actually faster rather than slower are baked in here:

1. **One BLAS thread per worker** (``OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1``) so N
   single-threaded processes scale cleanly instead of fighting over all cores.
2. **No per-worker index writes** (``--write_index False``); the shared index is rebuilt
   once here after every worker finishes, avoiding a write race.

Workers run one layer each, so the RNG resets fresh per ``(layer, rec)`` -- the null is
deterministic per job and identical across recs.

Example::

    uv run python run_parallel_benchmark.py \\
        --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47 \\
        --recs 0,1 --concept_sets secondary,swissprot \\
        --activations_dir_template "downloads_layer{layer}" --n_perm 200 \\
        --protein_ids_file common_activation_proteins.txt \\
        --skip_existing False --max_workers 8
"""

import json
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


def _thread_env(threads_per_worker: int) -> dict[str, str]:
    """Cap each worker's BLAS thread pools so workers don't oversubscribe cores.

    With ``max_workers * threads_per_worker`` <= physical cores, the heavy BLAS work
    (permutation null, logistic-regression probes) runs multi-threaded *within* each
    job while jobs still run in parallel -- the right trade when RAM caps the worker
    count below the core count.
    """
    value = str(max(1, threads_per_worker))
    return {
        "OMP_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
    }


SCRIPT = str(Path(__file__).parent / "run_layer_benchmark.py")


def _parse_ints(value: str) -> list[int]:
    return [int(token) for token in str(value).replace(" ", "").split(",") if token]


def _run_one(
    layer: int,
    rec: int,
    seed: int,
    concept_sets: str,
    activations_dir_template: str,
    protein_ids_file: str | None,
    n_perm: int,
    skip_existing: bool,
    max_proteins: int,
    probe_on_sae: bool,
    threads_per_worker: int,
    device: str | None,
    layer_type: str,
) -> tuple[int, int, int, int, float, str]:
    """Run the benchmark for one ``(layer, rec, seed)`` as a thread-capped subprocess."""
    cmd = [
        sys.executable, SCRIPT,
        "--layers", str(layer),
        "--rec", str(rec),
        "--seed", str(seed),
        "--concept_sets", concept_sets,
        "--activations_dir_template", activations_dir_template,
        "--n_perm", str(n_perm),
        "--probe_on_sae", str(probe_on_sae),
        "--skip_existing", str(skip_existing),
        "--layer_type", layer_type,
        "--write_index", "False",  # rebuilt once by the parent after all workers finish
    ]
    if protein_ids_file:
        cmd += ["--protein_ids_file", protein_ids_file]
    if max_proteins:
        cmd += ["--max_proteins", str(max_proteins)]
    if device:
        cmd += ["--device", device]

    env = {**os.environ, **_thread_env(threads_per_worker)}
    start = time.monotonic()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    tail = (proc.stderr or proc.stdout or "")[-800:]
    return layer, rec, seed, proc.returncode, elapsed, tail


def _report_consistency(rec_list: list[int], concept_sets: str, layer_type: str) -> None:
    """Log whether ``n_proteins`` is constant per concept set (across layers and recs)."""
    names = [s for s in concept_sets.replace(" ", "").split(",") if s]
    per_set_single: dict[str, set[int]] = {name: set() for name in names}
    for rec in rec_list:
        root = output_root_for(rec, layer_type)
        for name in names:
            counts: dict[int, list[int]] = {}
            for path in sorted(root.glob(f"layer*/{name}_seed*_benchmark.json")):
                data = json.loads(path.read_text(encoding="utf-8"))
                counts.setdefault(int(data["n_proteins"]), []).append(int(data["layer"]))
            if not counts:
                continue
            if len(counts) == 1:
                n = next(iter(counts))
                per_set_single[name].add(n)
                LOGGER.info(msg=f"  rec{rec}/{name}: n_proteins={n} across all {len(counts[n])} layers  OK")
            else:
                detail = ", ".join(f"{n} ({len(ls)} layers)" for n, ls in sorted(counts.items()))
                LOGGER.warning(msg=f"  rec{rec}/{name}: n_proteins VARIES -> {detail}")
    # Cross-rec check: a concept set should land on one shared count across recs.
    for name, singles in per_set_single.items():
        if len(rec_list) > 1 and len(singles) == 1:
            LOGGER.info(msg=f"  {name}: same n_proteins={next(iter(singles))} across recs {rec_list}  OK")
        elif len(singles) > 1:
            LOGGER.warning(msg=f"  {name}: n_proteins differs between recs -> {sorted(singles)}")


def run_parallel_benchmark(
    layers: str = "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47",
    recs: str = "0,1",
    seeds: str = "1",
    concept_sets: str = "secondary,swissprot",
    activations_dir_template: str = "downloads_layer{layer}",
    protein_ids_file: str | None = None,
    n_perm: int = 200,
    max_workers: int = 8,
    threads_per_worker: int = 1,
    probe_on_sae: bool = True,
    skip_existing: bool = False,
    max_proteins: int = 0,
    device: str | None = None,
    layer_type: str = "pairformer",
) -> int:
    """Fan ``run_layer_benchmark`` out over ``(layer, rec, seed)`` with a worker cap.

    Args:
        layers: Comma-separated layers.
        layer_type: ``"pairformer"`` (default) or ``"diffusion"``; forwarded to each
            worker and used to resolve the per-type output tree for the index rebuild.
        recs: Comma-separated recycle indices to run (e.g. ``"0,1"``).
        seeds: Comma-separated SAE seeds to evaluate (e.g. ``"1,2,3"``). Each seed loads
            a different trained SAE, so its ``sae``/``probe_sae`` numbers differ while the
            ``neuron``/``probe_raw`` baselines are identical across seeds -- averaging over
            seeds turns layer-to-layer F1 wiggle into mean +/- std (see aggregate_seed_benchmark.py).
        concept_sets: Comma-separated sets from
            ``{secondary, swissprot, boltz_secondary, boltz_plddt}``.
        activations_dir_template: Activation dir; may contain ``{layer}``.
        protein_ids_file: Optional fixed protein-ID list (the common set) so every
            layer/rec uses an identical protein pool.
        n_perm: Permutations for the null per job. The null is a big cost; 100 gives a
            p-value floor of 0.01 (vs 0.005 at 200) and roughly halves null time.
        max_workers: Concurrent layer-jobs. Each holds the full latent matrix in RAM
            (~5 GB measured), so RAM, not cores, is usually the limit. Lower if you swap.
        threads_per_worker: BLAS threads per worker. When RAM caps ``max_workers`` below
            your physical core count, raise this so ``max_workers * threads_per_worker``
            ~= physical cores -- otherwise the heavy per-job BLAS work runs single-threaded
            and the spare cores sit idle.
        probe_on_sae: Train the 2048-dim SAE-latent probe too. This is the single most
            expensive step; set False to skip it (you keep sae_f1/neuron_f1/probe_raw,
            only the probe_sae upper bound is dropped).
        skip_existing: Skip a (layer, concept set) whose JSON already exists. Default
            False here because re-running on a new protein set must overwrite.
        max_proteins: Cap on matched proteins (0 = all in the set).
        device: Optional torch device for the SAE forward pass (e.g. ``"cpu"`` on the
            CPU analysis fleet, ``"cuda"`` on a GPU box). Forwarded to each worker.

    Returns:
        Exit code; 0 only if every job succeeded.
    """
    if protein_ids_file and not Path(protein_ids_file).exists():
        LOGGER.error(
            msg=(
                f"protein_ids_file '{protein_ids_file}' not found. Generate it first:\n"
                f"  uv run python compute_common_proteins.py --layers {layers} --recs {recs} "
                f"--out {protein_ids_file}\n"
                f"Or drop --protein_ids_file to use every available protein (n_proteins will "
                f"then vary by layer/rec)."
            )
        )
        return 1

    layer_list = _parse_ints(layers)
    rec_list = _parse_ints(recs)
    seed_list = _parse_ints(seeds)
    jobs = [(layer, rec, seed) for rec in rec_list for layer in layer_list for seed in seed_list]
    LOGGER.info(msg=f"Launching {len(jobs)} jobs ({len(layer_list)} layers x {len(rec_list)} recs "
                    f"x {len(seed_list)} seeds) across {max_workers} workers x "
                    f"{threads_per_worker} thread(s) (probe_on_sae={probe_on_sae}, n_perm={n_perm})")

    results: list[tuple[int, int, int, int]] = []
    wall_start = time.monotonic()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_one, layer, rec, seed, concept_sets, activations_dir_template,
                protein_ids_file, n_perm, skip_existing, max_proteins,
                probe_on_sae, threads_per_worker, device, layer_type,
            ): (layer, rec, seed)
            for (layer, rec, seed) in jobs
        }
        for future in as_completed(futures):
            layer, rec, seed, code, elapsed, tail = future.result()
            done += 1
            status = "ok" if code == 0 else "FAILED"
            LOGGER.info(msg=f"[{done}/{len(jobs)}] {status}: layer {layer} rec {rec} seed {seed}  ({elapsed:.0f}s)")
            if code != 0:
                LOGGER.error(msg=f"  layer {layer} rec {rec} seed {seed} stderr/out tail:\n{tail}")
            results.append((layer, rec, seed, code))

    LOGGER.info(msg=f"All jobs finished in {time.monotonic() - wall_start:.0f}s; rebuilding indexes")
    for rec in rec_list:
        index_path = rebuild_index(output_root_for(rec, layer_type))
        LOGGER.info(msg=f"  rec{rec} index -> {index_path}")

    LOGGER.info(msg="n_proteins consistency:")
    _report_consistency(rec_list, concept_sets, layer_type)

    failed = [(layer, rec, seed) for (layer, rec, seed, code) in results if code != 0]
    if failed:
        LOGGER.error(msg=f"{len(failed)}/{len(jobs)} job(s) FAILED: {failed}")
        return 1
    LOGGER.info(msg=f"All {len(jobs)} jobs succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(tapify(run_parallel_benchmark, explicit_bool=True))
