#!/usr/bin/env python3
"""Build per-residue structural annotations from **Boltz-1's own** predicted CIF files.

This is the principled counterpart to ``build_structure_secondary_structure_annotations.py``.
That script labels structure from *AlphaFold* models downloaded from EBI -- a different
model's output, on the canonical UniProt sequence, which is why it drops proteins on
length mismatch and hits 404s. Here we use the structure Boltz-1 *itself* predicted in the
same forward pass that produced the activations we probe (``<id>_model_0.cif.gz``, stored
next to the activations on S3). Two consequences make this the better target:

* **Causal/self-consistent.** The label is downstream of the very activations being
  interpreted -- we test "can we decode what the model computed", not "does the feature
  correlate with a third party's structure".
* **Alignment is guaranteed.** Boltz's CIF is per-token identical to the activation tensor
  (same input, same pass), so residue ``i`` in the CIF is token ``i`` in the activations.

Critical implementation detail: ``ConceptEvaluator`` *drops* any protein whose annotation
length differs from its activation length. Incomplete-backbone residues (e.g. disordered
termini) cannot be passed to pydssp, so we run pydssp on the complete-backbone residues
only and then **scatter** the resulting labels back to full length by ``label_seq_id``
(missing residues default to coil). The emitted annotation length therefore equals
``max(label_seq_id)`` == the activation token count, and a dropped residue never shifts the
positions after it.

Outputs (per concept set) an ``AnnotationLoader``-compatible directory identical in format
to the AlphaFold builder, so ``evaluate_f1_across_layers.py`` consumes it unchanged.

S3 note: the activations bucket grants GetObject but not ListBucket, so we fetch each CIF by
its exact key (``head``/``get_object``) -- never a LIST.
"""

import gzip
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pydssp
from botocore.exceptions import ClientError
from tap import tapify

from build_structure_secondary_structure_annotations import (
    ATOM_ORDER,
    THREE_TO_ONE_AA,
    _column_index,
    _optional_column_index,
    _parse_atom_site_loop,
)
from get_activations import create_s3_client, is_missing_key_error

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_BUCKET = "swissprot-annotated-proteins-activations"
# Manifest lines look like "SwissProtAnnotation5/activations/<id>/"; the CIF sits inside that
# folder as "<id>_model_0.cif.gz". For plain-id manifests we build the key from this template.
DEFAULT_KEY_TEMPLATE = "SwissProtAnnotation5/activations/{protein_id}/{protein_id}_model_0.cif.gz"

SECONDARY_STRUCTURE_CONCEPTS = [
    "secondary_structure:helix",
    "secondary_structure:strand",
    "secondary_structure:coil",
]
SS3_TO_CONCEPT_IDX = {"H": 0, "E": 1, "C": 2}

# Boltz pLDDT is on the same 0-100 lddt scale as AlphaFold; use the standard AlphaFold bands.
PLDDT_CONCEPTS = [
    "plddt:very_low",   # < 50
    "plddt:low",        # 50-70
    "plddt:confident",  # 70-90
    "plddt:very_high",  # >= 90
]


def parse_manifest_entries(manifest_path: Path, key_template: str, max_proteins: int) -> list[tuple[str, str]]:
    """Read ``(protein_id, s3_key)`` pairs from a manifest.

    Accepts both prefix manifests (``SwissProtAnnotation5/activations/<id>/``) and plain-id
    manifests (``<id>``). For a prefix line the CIF key is built inside that prefix; for a
    bare id it is built from ``key_template``.
    """
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip().strip("/").replace("\\", "/")
            if not line or line.startswith("#"):
                continue
            protein_id = line.split("/")[-1]
            if protein_id in seen:
                continue
            if "/" in line:  # full activation prefix -> CIF lives inside it
                key = f"{line}/{protein_id}_model_0.cif.gz"
            else:  # bare accession -> use the template
                key = key_template.format(protein_id=protein_id)
            entries.append((protein_id, key))
            seen.add(protein_id)
            if max_proteins > 0 and len(entries) >= max_proteins:
                break
    return entries


def fetch_cif_text(s3, bucket: str, key: str, cache_path: Path, overwrite: bool) -> str | None:
    """Fetch + gunzip one Boltz CIF by exact key (no LIST). Returns text or None if absent.

    The decompressed CIF is cached on disk; an empty cache file records a confirmed miss.
    """
    if cache_path.exists() and not overwrite:
        text = cache_path.read_text(encoding="utf-8")
        return text or None
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if is_missing_key_error(exc):
            cache_path.write_text("", encoding="utf-8")  # cache the miss
            return None
        raise
    text = gzip.decompress(body).decode("utf-8")
    cache_path.write_text(text, encoding="utf-8")
    return text


def _atom_site_rows(cif_text: str) -> tuple[list[str], list[list[str]]]:
    """Return atom_site headers and rows, robust to 80-column line wrapping.

    Boltz CIFs wrap long atom_site rows across physical lines (the writer breaks at ~80
    columns, e.g. pushing ``pdbx_PDB_model_num`` onto its own line when coordinates are wide).
    The shared ``_parse_atom_site_loop`` assumes one row per line, so a wrapped row becomes a
    short row that the length guard silently drops -- losing residues and breaking alignment.

    An mmCIF loop has a fixed column count, so we flatten the loop's whitespace tokens and
    regroup into rows of ``len(headers)``. For unwrapped files (e.g. AlphaFold's, which keep
    each row on one line) this is a no-op: every row already has exactly ``n`` tokens. If the
    flat-token count is not divisible by ``n`` (some unexpected value), we fall back to the
    line-based rows so a parsing anomaly degrades gracefully rather than misaligning everything.
    """
    headers, raw_rows = _parse_atom_site_loop(cif_text)
    n = len(headers)
    flat = [token for row in raw_rows for token in row]
    if n == 0 or len(flat) % n != 0:
        LOGGER.warning(msg=f"atom_site token count {len(flat)} not divisible by {n} columns; using line-based rows.")
        return headers, [row for row in raw_rows if len(row) == n]
    rows = [flat[i : i + n] for i in range(0, len(flat), n)]
    return headers, rows


def parse_cif_residues(cif_text: str) -> dict[str, Any]:
    """Parse the first protein chain's backbone, indexed by ``label_seq_id``.

    Returns a dict with:
        ``n_residues``: ``max(label_seq_id)`` -- the full token count (== activation length).
        ``coords``: ``(m, 4, 3)`` N/CA/C/O coords for the ``m`` complete-backbone residues.
        ``coord_seq_ids``: the 1-based ``label_seq_id`` of each retained residue (length ``m``).
        ``sequence``: one-letter sequence of the retained residues (length ``m``).
        ``plddt``: ``(n_residues,)`` per-residue pLDDT (B-factor), NaN where no atoms seen.
    """
    headers, rows = _atom_site_rows(cif_text)
    group_idx = _optional_column_index(headers, "group_PDB")
    atom_idx = _column_index(headers, "label_atom_id")
    comp_idx = _column_index(headers, "label_comp_id")
    seq_idx = _column_index(headers, "label_seq_id")
    chain_idx = _column_index(headers, "label_asym_id")
    x_idx = _column_index(headers, "Cartn_x")
    y_idx = _column_index(headers, "Cartn_y")
    z_idx = _column_index(headers, "Cartn_z")
    b_idx = _optional_column_index(headers, "B_iso_or_equiv")
    model_idx = _optional_column_index(headers, "pdbx_PDB_model_num")

    residues: dict[int, dict[str, Any]] = {}
    first_chain = ""
    for row in rows:
        if len(row) < len(headers):
            continue
        if group_idx is not None and row[group_idx] != "ATOM":
            continue
        if model_idx is not None and row[model_idx] not in {"", "1"}:
            continue
        atom_name = row[atom_idx]
        if atom_name not in ATOM_ORDER:
            continue
        chain_id = row[chain_idx]
        if not first_chain:
            first_chain = chain_id
        if chain_id != first_chain:
            continue
        try:
            seq_id = int(row[seq_idx])
            xyz = np.array([float(row[x_idx]), float(row[y_idx]), float(row[z_idx])], dtype=np.float32)
        except ValueError:
            continue
        residue = residues.setdefault(
            seq_id,
            {
                "seq_id": seq_id,
                "aa1": THREE_TO_ONE_AA.get(row[comp_idx].upper(), "X"),
                "atoms": np.full((4, 3), np.nan, dtype=np.float32),
                "b_vals": [],
            },
        )
        residue["atoms"][ATOM_ORDER[atom_name]] = xyz
        if b_idx is not None:
            try:
                residue["b_vals"].append(float(row[b_idx]))
            except ValueError:
                pass

    if not residues:
        raise ValueError("No protein-chain atoms found in CIF.")

    n_residues = max(residues)  # label_seq_id is 1-based and contiguous for Boltz models
    plddt = np.full(n_residues, np.nan, dtype=np.float32)
    for sid, res in residues.items():
        if res["b_vals"]:
            plddt[sid - 1] = float(np.mean(res["b_vals"]))

    coords: list[np.ndarray] = []
    coord_seq_ids: list[int] = []
    seq_parts: list[str] = []
    for sid in sorted(residues):
        atoms = residues[sid]["atoms"]
        if np.isnan(atoms).any():
            continue  # incomplete backbone -> scattered to coil later
        coords.append(atoms)
        coord_seq_ids.append(sid)
        seq_parts.append(str(residues[sid]["aa1"]))

    return {
        "n_residues": n_residues,
        "coords": np.stack(coords, axis=0) if coords else np.empty((0, 4, 3), dtype=np.float32),
        "coord_seq_ids": coord_seq_ids,
        "sequence": "".join(seq_parts),
        "plddt": plddt,
    }


def build_secondary_structure_annotation(parsed: dict[str, Any]) -> tuple[np.ndarray, Counter]:
    """Run pydssp on complete-backbone residues and scatter H/E/C to full length (coil default)."""
    n = parsed["n_residues"]
    annotation = np.zeros((n, len(SECONDARY_STRUCTURE_CONCEPTS)), dtype=np.uint8)
    annotation[:, SS3_TO_CONCEPT_IDX["C"]] = 1  # default every residue to coil
    counts: Counter = Counter()

    coords = parsed["coords"]
    if coords.shape[0] >= 1:
        seq = parsed["sequence"]
        donor_mask = np.array([aa != "P" for aa in seq], dtype=bool)
        labels = np.asarray(pydssp.assign(coords, donor_mask=donor_mask, out_type="c3")).astype(str).reshape(-1)
        for seq_id, lab in zip(parsed["coord_seq_ids"], labels):
            ss3 = "C" if lab == "-" else lab
            idx = SS3_TO_CONCEPT_IDX[ss3]
            annotation[seq_id - 1, :] = 0
            annotation[seq_id - 1, idx] = 1
            counts[ss3] += 1
    counts["C_scattered"] = int(n - coords.shape[0])  # residues with no DSSP label, left as coil
    return annotation, counts


def build_plddt_annotation(parsed: dict[str, Any]) -> np.ndarray:
    """One-hot pLDDT confidence bands per residue (NaN pLDDT -> all-zero row)."""
    plddt = parsed["plddt"]
    annotation = np.zeros((parsed["n_residues"], len(PLDDT_CONCEPTS)), dtype=np.uint8)
    valid = ~np.isnan(plddt)
    bins = np.full(plddt.shape, -1, dtype=int)
    bins[valid & (plddt < 50)] = 0
    bins[valid & (plddt >= 50) & (plddt < 70)] = 1
    bins[valid & (plddt >= 70) & (plddt < 90)] = 2
    bins[valid & (plddt >= 90)] = 3
    for row_idx, b in enumerate(bins):
        if b >= 0:
            annotation[row_idx, b] = 1
    return annotation


def write_processed_annotations(
    output_dir: Path,
    concepts: list[str],
    annotations: dict[str, np.ndarray],
    manifest_path: Path,
    source_desc: str,
) -> None:
    """Write an ``AnnotationLoader``-compatible processed directory (one shard)."""
    shard_dir = output_dir / "shard_0"
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(shard_dir / "annotations.npz", **annotations)
    vocabulary = {
        "concepts": concepts,
        "metadata": {
            "source": source_desc,
            "manifest": str(manifest_path),
            "n_annotated_proteins": len(annotations),
        },
    }
    (output_dir / "concept_vocabulary.json").write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    (output_dir / "protein_ids.txt").write_text(
        "\n".join(annotations.keys()) + ("\n" if annotations else ""), encoding="utf-8"
    )


def build_boltz_structure_annotations(
    manifest: str = "swissprot_a5_prefixes.txt",
    secondary_output_dir: str = "processed_swissprot_a5_boltz_secondary",
    plddt_output_dir: str = "processed_swissprot_a5_boltz_plddt",
    build_plddt: bool = True,
    bucket: str = DEFAULT_BUCKET,
    key_template: str = DEFAULT_KEY_TEMPLATE,
    cif_cache_dir: str = "structure_cache/boltz_cifs",
    max_proteins: int = 500,
    download_workers: int = 16,
    overwrite_cifs: bool = False,
    endpoint_url: str | None = None,
) -> int:
    """Download Boltz ``_model_0.cif.gz`` files and build per-residue structural annotations.

    Args:
        manifest: Prefix manifest (e.g. ``swissprot_a5_prefixes.txt``) or a plain-id list.
        secondary_output_dir: Output dir for the H/E/C secondary-structure concept set.
        plddt_output_dir: Output dir for the pLDDT-confidence concept set.
        build_plddt: Also emit the pLDDT-band concept set (nearly free; the model's own confidence).
        bucket: S3 bucket holding the activations + CIFs.
        key_template: CIF key template for bare-id manifests.
        cif_cache_dir: Where decompressed CIFs are cached.
        max_proteins: Number of manifest proteins to process. 0 = all.
        download_workers: Concurrent S3 GETs.
        overwrite_cifs: Re-fetch CIFs already cached.
        endpoint_url: Optional S3-compatible endpoint override (e.g. R2).

    Returns:
        Exit code; 0 if at least one protein was annotated.
    """
    manifest_path = Path(manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    cache_dir = Path(cif_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries = parse_manifest_entries(manifest_path, key_template, max_proteins)
    LOGGER.info(msg=f"Processing {len(entries)} proteins from {manifest_path}")
    s3 = create_s3_client(endpoint_url=endpoint_url, max_pool_connections=max(10, download_workers * 2))

    def fetch_one(item: tuple[str, str]) -> tuple[str, str | None]:
        protein_id, key = item
        cache_path = cache_dir / f"{protein_id}_model_0.cif"
        try:
            return protein_id, fetch_cif_text(s3, bucket, key, cache_path, overwrite_cifs)
        except ClientError as exc:
            LOGGER.warning(msg=f"{protein_id}: S3 error {exc}")
            return protein_id, None

    cif_texts: dict[str, str | None] = {}
    with ThreadPoolExecutor(max_workers=max(1, download_workers)) as executor:
        futures = {executor.submit(fetch_one, item): item[0] for item in entries}
        for idx, future in enumerate(as_completed(futures), start=1):
            protein_id, text = future.result()
            cif_texts[protein_id] = text
            if idx % 25 == 0 or idx == len(futures):
                LOGGER.info(msg=f"Fetched {idx}/{len(futures)} CIFs")

    ss_annotations: dict[str, np.ndarray] = {}
    plddt_annotations: dict[str, np.ndarray] = {}
    n_missing = n_parse_error = 0
    for protein_id, _ in entries:
        text = cif_texts.get(protein_id)
        if not text:
            n_missing += 1
            continue
        try:
            parsed = parse_cif_residues(text)
            ss_ann, _counts = build_secondary_structure_annotation(parsed)
        except (AssertionError, ValueError, KeyError) as exc:
            LOGGER.debug(msg=f"{protein_id}: parse/DSSP failed: {exc}")
            n_parse_error += 1
            continue
        ss_annotations[protein_id] = ss_ann
        if build_plddt:
            plddt_annotations[protein_id] = build_plddt_annotation(parsed)

    write_processed_annotations(
        Path(secondary_output_dir),
        SECONDARY_STRUCTURE_CONCEPTS,
        ss_annotations,
        manifest_path,
        "Boltz-1 _model_0.cif (own predicted structure) processed with pydssp",
    )
    LOGGER.info(msg=f"Wrote {len(ss_annotations)} SS annotations to {secondary_output_dir}")
    if build_plddt:
        write_processed_annotations(
            Path(plddt_output_dir),
            PLDDT_CONCEPTS,
            plddt_annotations,
            manifest_path,
            "Boltz-1 _model_0.cif per-residue pLDDT (B-factor) binned",
        )
        LOGGER.info(msg=f"Wrote {len(plddt_annotations)} pLDDT annotations to {plddt_output_dir}")

    LOGGER.info(msg=f"Missing CIFs: {n_missing}; parse/DSSP errors: {n_parse_error}")
    return 0 if ss_annotations else 1


if __name__ == "__main__":
    raise SystemExit(tapify(build_boltz_structure_annotations))
