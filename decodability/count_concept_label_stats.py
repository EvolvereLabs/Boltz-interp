#!/usr/bin/env python3
"""Per-concept label statistics for the evaluation set (residues + domains).

Reviewers need to know, per concept, how many positive residues and positive
*domains* enter the benchmark, and over how many proteins. None of this is stored
in the benchmark JSONs (they keep only F1 / precision / recall), but it is a pure
property of the labels and is therefore identical across every layer, recycle and
SAE seed -- so it is computed once here, directly from the processed annotation
shards, over the *same* protein set the probe sees.

The protein set is pinned by ``common_activation_proteins.txt`` (the shared
download list + benchmark filter used by ``run_analysis_workflow.sh``); since
``run_layer_benchmark.py`` is called with ``max_proteins=0``, every protein in
that set that also has the concept's annotation enters the probe. Restricting to
``common ∩ annotated`` reproduces the saved ``n_proteins`` / ``n_residues`` in the
benchmark JSONs exactly (99 / 44968 swissprot, 393 / 154299 secondary, 486 /
196136 boltz_secondary), which is the correctness check for this count.

A "domain" is a contiguous run of positive residues -- the identical definition
used for domain-level recall (``embeddings_concepts_evaluation.find_domains``).

Writes a tidy long CSV (one row per concept set x concept):
    concept_set, annotation_dir, concept, pos_residues, pos_domains,
    n_proteins_with_concept, set_n_proteins, set_n_residues
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np
from tap import tapify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# concept-set label -> processed annotation directory (matches run_layer_benchmark.DEFAULT_CONCEPT_SETS)
CONCEPT_SET_DIRS: dict[str, str] = {
    "swissprot": "processed_swissprot",
    "secondary": "processed_swissprot_a5_structure_secondary_n500",
    "boltz_secondary": "processed_swissprot_a5_boltz_secondary",
}


def _count_domains(col: np.ndarray) -> int:
    """Number of contiguous positive runs in a binary column (== find_domains length)."""
    padded = np.concatenate([[0], (col > 0).astype(np.int8), [0]])
    return int((np.diff(padded) == 1).sum())


def _load_shard_annotations(processed_dir: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    """Concept list + {protein_id: (L, C) label matrix} across all shards (as AnnotationLoader)."""
    vocab = json.loads((processed_dir / "concept_vocabulary.json").read_text(encoding="utf-8"))
    concepts: list[str] = vocab["concepts"]
    ann: dict[str, np.ndarray] = {}
    for shard in sorted(p for p in processed_dir.iterdir() if p.is_dir() and p.name.startswith("shard_")):
        npz = shard / "annotations.npz"
        if npz.exists():
            data = np.load(npz, allow_pickle=True)
            for key in data.files:
                ann[key] = data[key]
    return concepts, ann


def count_concept_label_stats(
    protein_ids_file: str = "common_activation_proteins.txt",
    out_csv: str = "sae_explore/concept_label_stats.csv",
) -> int:
    """Compute per-concept positive residues + domains over the common protein set."""
    common = {
        line.strip().rstrip("/").split("/")[-1]
        for line in Path(protein_ids_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    LOGGER.info(msg=f"Protein filter: {len(common)} ids from {protein_ids_file}")

    rows: list[dict] = []
    for set_name, dirname in CONCEPT_SET_DIRS.items():
        processed_dir = Path(dirname)
        if not processed_dir.exists():
            LOGGER.warning(msg=f"Annotation dir '{dirname}' missing; skipping {set_name}.")
            continue
        concepts, ann = _load_shard_annotations(processed_dir)
        used = sorted(set(ann) & common)
        pos_res = np.zeros(len(concepts), dtype=np.int64)
        pos_dom = np.zeros(len(concepts), dtype=np.int64)
        n_prot = np.zeros(len(concepts), dtype=np.int64)
        set_n_residues = 0
        for pid in used:
            mat = np.asarray(ann[pid])
            if mat.ndim == 1:
                mat = mat[:, None]
            set_n_residues += mat.shape[0]
            for ci in range(len(concepts)):
                col = mat[:, ci]
                r = int((col > 0).sum())
                pos_res[ci] += r
                pos_dom[ci] += _count_domains(col)
                if r > 0:
                    n_prot[ci] += 1
        LOGGER.info(
            msg=f"{set_name}: {len(used)} proteins, {set_n_residues} residues, {len(concepts)} concepts"
        )
        for ci, concept in enumerate(concepts):
            rows.append(
                {
                    "concept_set": set_name,
                    "annotation_dir": dirname,
                    "concept": concept,
                    "pos_residues": int(pos_res[ci]),
                    "pos_domains": int(pos_dom[ci]),
                    "n_proteins_with_concept": int(n_prot[ci]),
                    "set_n_proteins": len(used),
                    "set_n_residues": set_n_residues,
                }
            )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info(msg=f"Wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(tapify(count_concept_label_stats))
