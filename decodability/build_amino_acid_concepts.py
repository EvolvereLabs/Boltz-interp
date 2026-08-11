#!/usr/bin/env python3
"""Build a residue-identity concept set from SwissProt sequences.

This is the scaffolding for a *sanity check*: amino-acid identity is the most
local, model-visible property there is, so it is the concept we most expect an
SAE latent / neuron / linear probe to recover. The output is an ordinary
processed-concept directory -- ``concept_vocabulary.json`` plus per-shard
``annotations.npz`` keyed by UniProt ID -- so it drops straight into the
existing :class:`AnnotationLoader` / :class:`ConceptEvaluator` tooling
(``evaluate_f1_across_layers.py``, ``amino_acid_sanity_check.py``).

Each protein gets a one-hot annotation matrix of shape ``(seq_len, 20)`` over
the 20 standard amino acids (concepts ``aa:A`` ... ``aa:Y``). Non-standard
residues (``X``, ``B``, ``Z``, ``U``, ``O``, ``*``) are left all-zero so they
contribute no positives to any concept.

Example::

    uv run python build_amino_acid_concepts.py \\
        --source_dir processed_swissprot_a5 --out_dir processed_swissprot_aa
"""

import json
import logging
from pathlib import Path

import numpy as np
from tap import tapify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

# The 20 standard amino acids, in a fixed canonical order.
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(STANDARD_AMINO_ACIDS)}
CONCEPTS = [f"aa:{aa}" for aa in STANDARD_AMINO_ACIDS]


def parse_fasta(fasta_path: Path) -> dict[str, str]:
    """Parse a FASTA file into an ``{id: sequence}`` mapping.

    Args:
        fasta_path: Path to a ``sequences.fasta`` file.

    Returns:
        Mapping from the FASTA header id (first whitespace-delimited token,
        ``>`` stripped) to the concatenated sequence string.
    """
    sequences: dict[str, str] = {}
    current_id: str | None = None
    chunks: list[str] = []

    def _flush() -> None:
        if current_id is not None:
            sequences[current_id] = "".join(chunks)

    with fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _flush()
                current_id = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
    _flush()
    return sequences


def one_hot_sequence(sequence: str) -> np.ndarray:
    """Return a ``(len(sequence), 20)`` int8 one-hot of residue identity.

    Args:
        sequence: A protein sequence (single-letter amino-acid codes).

    Returns:
        One-hot matrix over :data:`STANDARD_AMINO_ACIDS`; rows for
        non-standard residues are all-zero.
    """
    matrix = np.zeros((len(sequence), len(STANDARD_AMINO_ACIDS)), dtype=np.int8)
    for position, residue in enumerate(sequence.upper()):
        index = AA_TO_INDEX.get(residue)
        if index is not None:
            matrix[position, index] = 1
    return matrix


def build_amino_acid_concepts(
    source_dir: str = "processed_swissprot_a5",
    out_dir: str = "processed_swissprot_aa",
) -> int:
    """Build a one-hot amino-acid concept directory from a processed source dir.

    Args:
        source_dir: Existing processed dir holding ``shard_*/sequences.fasta``.
        out_dir: Destination concept directory to create (mirrors the standard
            processed layout: one ``shard_0/annotations.npz`` plus
            ``concept_vocabulary.json``).

    Returns:
        Process exit code (0 on success, 1 if no sequences were found).
    """
    source = Path(source_dir)
    shard_dirs = (
        sorted(d for d in source.iterdir() if d.is_dir() and d.name.startswith("shard_"))
        if source.is_dir()
        else []
    )
    fasta_paths = [d / "sequences.fasta" for d in shard_dirs]
    fasta_paths = [p for p in fasta_paths if p.exists()]

    # Fall back to the committed flat FASTA of the same SwissProt-A5 pool when the
    # processed shards have not been built (swissprot_annotation_pipeline.py).
    if not fasta_paths:
        fallback = Path(__file__).parent / "inputs" / "swissprot_a5.fasta"
        if not fallback.exists():
            LOGGER.error(msg=f"No sequences.fasta under {source} and no {fallback}.")
            return 1
        LOGGER.info(msg=f"No processed shards under {source}; using {fallback}.")
        fasta_paths = [fallback]

    annotations: dict[str, np.ndarray] = {}
    for fasta_path in fasta_paths:
        for protein_id, sequence in parse_fasta(fasta_path).items():
            if protein_id in annotations:
                continue
            annotations[protein_id] = one_hot_sequence(sequence)

    if not annotations:
        LOGGER.error(msg="No sequences parsed; nothing to write.")
        return 1

    out = Path(out_dir)
    shard_out = out / "shard_0"
    shard_out.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(shard_out / "annotations.npz", **annotations)
    (out / "concept_vocabulary.json").write_text(
        json.dumps({"concepts": CONCEPTS}, indent=2), encoding="utf-8"
    )

    total_residues = sum(int(matrix.shape[0]) for matrix in annotations.values())
    LOGGER.info(
        msg=(
            f"Wrote {len(annotations)} proteins ({total_residues} residues) over "
            f"{len(CONCEPTS)} amino-acid concepts to {out}."
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(tapify(build_amino_acid_concepts))
