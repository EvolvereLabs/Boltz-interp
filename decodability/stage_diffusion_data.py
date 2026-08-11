#!/usr/bin/env python3
"""Stage diffusion activations + SAE checkpoints from S3 for the AA sweep.

Downloads, for each requested diffusion layer x recycle/step:

* raw activations for a pool of amino-acid-annotated proteins into
  ``downloads_diffusion_rec{REC}/<pid>/<layer_subfolder>/output_{REC}.npz``
  (the layout ``amino_acid_sanity_check.py`` / ``amino_acid_cross_seed.py`` read), and
* the trained SAE checkpoints for every seed
  (``s3://boltz-saes-l2/diffusion/rec{REC}/layer{N}/<run>/``) into
  ``sae_explore/diffusion_cache/rec{REC}/layer{N}/<run>/`` (the cache
  ``download_run_checkpoint(..., layer_type="diffusion")`` returns).

Example::

    uv run python stage_diffusion_data.py --layers 4,14,22 --recs 0,50,199 \\
        --seeds 1,2,3 --n_proteins 160
"""

import concurrent.futures as cf
import logging
from pathlib import Path

import boto3
from botocore.config import Config
from tap import tapify

from embeddings_concepts_evaluation import AnnotationLoader
from layer_analysis_utils import FINAL_STEP, layer_subfolder, run_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

ACT_BUCKET = "swissprot-annotated-proteins-activations"
ACT_PREFIX = "SwissProtAnnotation5/activations"
SAE_BUCKET = "boltz-saes-l2"
SAE_ROOT = "diffusion"  # s3://boltz-saes-l2/diffusion/rec{REC}/layer{N}/<run>/
CHECKPOINT_FILES = ("config.json", "mean_vector.npy", f"checkpoint_step_{FINAL_STEP}.pt")


def _parse_ints(text: str) -> list[int]:
    return [int(token) for token in str(text).replace(" ", "").split(",") if token]


def stage_diffusion_data(
    layers: str = "4,14,22",
    recs: str = "0,50,199",
    seeds: str = "1,2,3",
    n_proteins: int = 160,
    aa_concept_dir: str = "processed_swissprot_aa",
    max_workers: int = 24,
) -> int:
    """Download diffusion activations and per-seed SAE checkpoints from S3.

    Args:
        layers: Comma-separated diffusion transformer layers.
        recs: Comma-separated recycle/diffusion-step indices.
        seeds: Comma-separated SAE seeds to stage.
        n_proteins: Number of amino-acid-annotated proteins to attempt to fetch
            (missing ones are skipped; coverage is roughly 70-75%).
        aa_concept_dir: Concept dir whose protein IDs define the download pool.
        max_workers: Thread pool size for activation downloads.

    Returns:
        Process exit code (0 on success).
    """
    layer_list = _parse_ints(layers)
    rec_list = _parse_ints(recs)
    seed_list = _parse_ints(seeds)
    protein_ids = list(AnnotationLoader(aa_concept_dir).list_available())[:n_proteins]
    s3 = boto3.client("s3", config=Config(max_pool_connections=max_workers + 4))

    # --- 1. Activations -----------------------------------------------------
    jobs: list[tuple[str, Path]] = []
    for rec in rec_list:
        out_root = Path(f"downloads_diffusion_rec{rec}")
        for layer in layer_list:
            sub = layer_subfolder(layer, "diffusion")
            for pid in protein_ids:
                key = f"{ACT_PREFIX}/{pid}/{sub}/output_{rec}.npz"
                dst = out_root / pid / sub / f"output_{rec}.npz"
                jobs.append((key, dst))

    def fetch(job: tuple[str, Path]) -> str:
        key, dst = job
        if dst.exists() and dst.stat().st_size > 0:
            return "skip"
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            s3.download_file(ACT_BUCKET, key, str(dst))
            return "ok"
        except Exception:
            return "miss"

    LOGGER.info(msg=f"Activations: {len(jobs)} files ({len(protein_ids)} proteins x {len(layer_list)} layers x {len(rec_list)} recs)")
    counts = {"ok": 0, "skip": 0, "miss": 0}
    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for i, result in enumerate(pool.map(fetch, jobs), 1):
            counts[result] += 1
            if i % 200 == 0:
                LOGGER.info(msg=f"  {i}/{len(jobs)} ok={counts['ok']} skip={counts['skip']} miss={counts['miss']}")
    LOGGER.info(msg=f"Activations done: ok={counts['ok']} skip={counts['skip']} miss={counts['miss']}")

    # --- 2. SAE checkpoints (all seeds) -------------------------------------
    ckpt_ok = ckpt_miss = 0
    for rec in rec_list:
        cache_dir = Path(f"sae_explore/diffusion_cache/rec{rec}")
        for layer in layer_list:
            for seed in seed_list:
                run = run_name(layer, seed, "diffusion")
                dst_dir = cache_dir / f"layer{layer}" / run
                dst_dir.mkdir(parents=True, exist_ok=True)
                for fname in CHECKPOINT_FILES:
                    dst = dst_dir / fname
                    if dst.exists() and dst.stat().st_size > 0:
                        ckpt_ok += 1
                        continue
                    key = f"{SAE_ROOT}/rec{rec}/layer{layer}/{run}/{fname}"
                    try:
                        s3.download_file(SAE_BUCKET, key, str(dst))
                        ckpt_ok += 1
                    except Exception as exc:
                        ckpt_miss += 1
                        LOGGER.warning(msg=f"checkpoint miss rec{rec} layer{layer} seed{seed} {fname}: {exc}")
    LOGGER.info(msg=f"Checkpoints done: ok={ckpt_ok} miss={ckpt_miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(tapify(stage_diffusion_data))
