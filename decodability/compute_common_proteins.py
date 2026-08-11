#!/usr/bin/env python3
"""Compute the protein IDs whose activations are present at EVERY (layer, rec).

The layer benchmark normally uses whatever proteins each layer happens to have, so
``n_proteins`` drifts between layers (different downloads succeed/fail) and between
recycles. To make results directly comparable across both axes, restrict the benchmark
to a single shared protein set: the *intersection* of available proteins over every
requested ``(layer, rec)``. Intersecting that with each concept's annotation set then
yields an identical protein pool for that concept set at every layer and rec.

Write the list, then pass it to the benchmark::

    uv run python compute_common_proteins.py \\
        --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47 \\
        --recs 0,1 --out common_activation_proteins.txt

    uv run python run_layer_benchmark.py --layers ... --rec 0 \\
        --concept_sets secondary,swissprot \\
        --activations_dir_template "downloads_layer{layer}" --n_perm 200 \\
        --protein_ids_file common_activation_proteins.txt
"""

import logging
from pathlib import Path

from tap import tapify

from loaders.raw_activation_loader import RawActivationLoader
from layer_analysis_utils import layer_subfolder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def compute_common_proteins(
    layers: str = "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,47",
    recs: str = "0,1",
    activations_dir_template: str = "downloads_layer{layer}",
    out: str = "common_activation_proteins.txt",
) -> int:
    """Intersect available protein IDs over all ``(layer, rec)`` and write the result.

    Args:
        layers: Comma-separated pairformer layers to require coverage on.
        recs: Comma-separated recycle indices to require coverage on (e.g. ``"0,1"``).
        activations_dir_template: Activation dir; may contain ``{layer}``.
        out: Output path for the newline-delimited protein IDs.

    Returns:
        Exit code; 0 if a non-empty common set was written, else 1.
    """
    layer_list = [int(x) for x in layers.replace(" ", "").split(",") if x]
    rec_list = [int(x) for x in recs.replace(" ", "").split(",") if x]

    common: set[str] | None = None
    for layer in layer_list:
        activations_dir = activations_dir_template.format(layer=layer)
        if not Path(activations_dir).exists():
            LOGGER.warning(msg=f"Activation dir '{activations_dir}' missing; layer {layer} contributes nothing.")
            common = set()
            continue
        subfolder = layer_subfolder(layer)
        for rec in rec_list:
            ids = set(RawActivationLoader(activations_dir, subfolder=subfolder, rec=rec).list_available())
            LOGGER.info(msg=f"layer {layer} rec {rec}: {len(ids)} proteins with output_{rec}")
            common = ids if common is None else (common & ids)

    common = common or set()
    Path(out).write_text("\n".join(sorted(common)) + "\n", encoding="utf-8")
    LOGGER.info(msg=f"Common across all (layer, rec): {len(common)} proteins -> {out}")
    if not common:
        LOGGER.error(msg="Common set is EMPTY -- some (layer, rec) has no overlap. Check downloads.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(tapify(compute_common_proteins))
