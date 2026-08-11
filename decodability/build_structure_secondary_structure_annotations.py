#!/usr/bin/env python3
"""Build per-residue secondary-structure annotations from AlphaFold CIF files."""

import csv
import json
import logging
import shlex
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pydssp
import requests
from tap import tapify

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ALPHAFOLD_CIF_URL = "https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v{version}.cif"
SECONDARY_STRUCTURE_CONCEPTS = [
    "secondary_structure:helix",
    "secondary_structure:strand",
    "secondary_structure:coil",
]
SS3_TO_CONCEPT_IDX = {"H": 0, "E": 1, "C": 2}
ATOM_ORDER = {"N": 0, "CA": 1, "C": 2, "O": 3}
THREE_TO_ONE_AA = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def parse_manifest_protein_id(line: str) -> str | None:
    """Extract a UniProt accession from one activation manifest line."""
    stripped = line.strip().strip("/").replace("\\", "/")
    if not stripped:
        return None
    return stripped.split("/")[-1]


def read_manifest_protein_ids(manifest_path: Path, max_proteins: int) -> list[str]:
    """Read the first unique protein IDs from the notebook manifest."""
    protein_ids: list[str] = []
    seen: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            protein_id = parse_manifest_protein_id(line)
            if protein_id is None or protein_id in seen:
                continue
            protein_ids.append(protein_id)
            seen.add(protein_id)
            if max_proteins > 0 and len(protein_ids) >= max_proteins:
                break
    return protein_ids


def parse_versions(version_csv: str) -> list[int]:
    """Parse AlphaFoldDB model versions in preferred order."""
    versions = [int(part.strip()) for part in version_csv.split(",") if part.strip()]
    if not versions:
        raise ValueError("alphafold_versions must contain at least one integer version.")
    return versions


def download_alphafold_cif(
    protein_id: str,
    cif_dir: Path,
    versions: list[int],
    timeout_seconds: int,
    overwrite: bool,
) -> dict[str, Any]:
    """Download one AlphaFold CIF, trying the requested model versions in order."""
    cif_path = cif_dir / f"{protein_id}.cif"
    if cif_path.exists() and not overwrite:
        return {
            "protein_id": protein_id,
            "download_status": "skipped_existing",
            "cif_path": str(cif_path),
            "alphafold_url": "",
            "download_error": "",
        }

    last_error = ""
    for version in versions:
        url = ALPHAFOLD_CIF_URL.format(protein_id=protein_id, version=version)
        try:
            response = requests.get(url, timeout=timeout_seconds)
            if response.status_code == 404:
                last_error = f"404 for {url}"
                continue
            response.raise_for_status()
            tmp_path = cif_path.with_suffix(".cif.tmp")
            tmp_path.write_bytes(response.content)
            tmp_path.replace(cif_path)
            return {
                "protein_id": protein_id,
                "download_status": "downloaded",
                "cif_path": str(cif_path),
                "alphafold_url": url,
                "download_error": "",
            }
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    return {
        "protein_id": protein_id,
        "download_status": "missing_or_error",
        "cif_path": str(cif_path),
        "alphafold_url": "",
        "download_error": last_error,
    }


def _clean_cif_value(value: str) -> str:
    """Remove simple mmCIF quotes/placeholders from one token."""
    if value in {".", "?"}:
        return ""
    return value.strip("\"'")


def _tokenize_cif_row(line: str) -> list[str]:
    """Tokenize one mmCIF data row."""
    return [_clean_cif_value(token) for token in shlex.split(line, posix=False)]


def _parse_atom_site_loop(cif_text: str) -> tuple[list[str], list[list[str]]]:
    """Return atom_site headers and rows from the first atom_site loop."""
    lines = cif_text.splitlines()
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() != "loop_":
            idx += 1
            continue

        idx += 1
        headers: list[str] = []
        while idx < len(lines) and lines[idx].strip().startswith("_"):
            headers.append(lines[idx].strip())
            idx += 1

        if not headers or not headers[0].startswith("_atom_site."):
            continue

        rows: list[list[str]] = []
        while idx < len(lines):
            stripped = lines[idx].strip()
            if not stripped or stripped == "#":
                break
            if stripped == "loop_" or stripped.startswith("_"):
                break
            rows.append(_tokenize_cif_row(stripped))
            idx += 1
        return headers, rows

    raise ValueError("No _atom_site loop found in CIF file.")


def _column_index(headers: list[str], column_name: str) -> int:
    """Return a required atom_site column index."""
    full_name = f"_atom_site.{column_name}"
    if full_name not in headers:
        raise ValueError(f"Missing required CIF column: {full_name}")
    return headers.index(full_name)


def _optional_column_index(headers: list[str], column_name: str) -> int | None:
    """Return an optional atom_site column index."""
    full_name = f"_atom_site.{column_name}"
    return headers.index(full_name) if full_name in headers else None


def parse_cif_backbone(cif_path: Path) -> tuple[np.ndarray, str]:
    """Parse N/CA/C/O backbone coordinates from an AlphaFold mmCIF file.

    Args:
        cif_path: Path to a downloaded AlphaFoldDB mmCIF.

    Returns:
        A coordinate array with shape ``(n_residues, 4, 3)`` in N, CA, C, O order
        plus the one-letter amino-acid sequence for the retained residues.

    Raises:
        ValueError: If required atom-site data are missing or no complete backbone
            residues can be parsed.
    """
    headers, rows = _parse_atom_site_loop(cif_path.read_text(encoding="utf-8"))
    group_idx = _optional_column_index(headers, "group_PDB")
    atom_idx = _column_index(headers, "label_atom_id")
    comp_idx = _column_index(headers, "label_comp_id")
    seq_idx = _column_index(headers, "label_seq_id")
    chain_idx = _column_index(headers, "label_asym_id")
    x_idx = _column_index(headers, "Cartn_x")
    y_idx = _column_index(headers, "Cartn_y")
    z_idx = _column_index(headers, "Cartn_z")
    model_idx = _optional_column_index(headers, "pdbx_PDB_model_num")

    residues: dict[tuple[str, int], dict[str, Any]] = {}
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

        residue_key = (chain_id, seq_id)
        residue = residues.setdefault(
            residue_key,
            {
                "seq_id": seq_id,
                "aa1": THREE_TO_ONE_AA.get(row[comp_idx].upper(), "X"),
                "atoms": np.full((4, 3), np.nan, dtype=np.float32),
            },
        )
        residue["atoms"][ATOM_ORDER[atom_name]] = xyz

    coords: list[np.ndarray] = []
    sequence_parts: list[str] = []
    for residue in sorted(residues.values(), key=lambda item: int(item["seq_id"])):
        atoms = residue["atoms"]
        if np.isnan(atoms).any():
            LOGGER.debug(msg=f"Skipping residue with incomplete backbone in {cif_path}")
            continue
        coords.append(atoms)
        sequence_parts.append(str(residue["aa1"]))

    if not coords:
        raise ValueError(f"No complete N/CA/C/O residues found in {cif_path}")
    return np.stack(coords, axis=0), "".join(sequence_parts)


def ss3_to_annotation(ss3_labels: np.ndarray, sequence: str) -> tuple[np.ndarray, Counter[str]]:
    """Convert pydssp C3 labels into a one-hot H/E/C annotation matrix."""
    flat_labels = np.asarray(ss3_labels).astype(str).reshape(-1)
    if flat_labels.shape[0] != len(sequence):
        raise ValueError(
            f"pydssp label length mismatch: labels={flat_labels.shape[0]}, sequence={len(sequence)}"
        )

    annotation = np.zeros((len(flat_labels), len(SECONDARY_STRUCTURE_CONCEPTS)), dtype=np.uint8)
    ss3_counts: Counter[str] = Counter()

    for row_idx, ss3 in enumerate(flat_labels):
        if ss3 == "-":
            ss3 = "C"
        concept_idx = SS3_TO_CONCEPT_IDX[ss3]
        annotation[row_idx, concept_idx] = 1
        ss3_counts[ss3] += 1

    return annotation, ss3_counts


def process_cif_with_pydssp(protein_id: str, cif_path: Path) -> dict[str, Any]:
    """Run pydssp for one CIF and return annotation payload plus metadata."""
    try:
        coords, sequence = parse_cif_backbone(cif_path)
        donor_mask = np.array([aa != "P" for aa in sequence], dtype=bool)
        ss3_labels = pydssp.assign(coords, donor_mask=donor_mask, out_type="c3")
        annotation, ss3_counts = ss3_to_annotation(ss3_labels, sequence)
    except (AssertionError, ValueError) as exc:
        return {
            "protein_id": protein_id,
            "annotation": None,
            "sequence": "",
            "pydssp_residues": 0,
            "helix_residues": 0,
            "strand_residues": 0,
            "coil_residues": 0,
            "pydssp_status": "error",
            "pydssp_error": str(exc),
        }

    if annotation.shape[0] == 0:
        return {
            "protein_id": protein_id,
            "annotation": None,
            "sequence": "",
            "pydssp_residues": 0,
            "helix_residues": 0,
            "strand_residues": 0,
            "coil_residues": 0,
            "pydssp_status": "empty",
            "pydssp_error": "pydssp returned no residues.",
        }

    return {
        "protein_id": protein_id,
        "annotation": annotation,
        "sequence": sequence,
        "pydssp_residues": int(annotation.shape[0]),
        "helix_residues": int(ss3_counts["H"]),
        "strand_residues": int(ss3_counts["E"]),
        "coil_residues": int(ss3_counts["C"]),
        "pydssp_status": "ok",
        "pydssp_error": "",
    }


def write_metadata_csv(metadata_rows: list[dict[str, Any]], metadata_path: Path) -> None:
    """Write per-protein download/pydssp metadata for reproducibility."""
    fieldnames = [
        "protein_id",
        "download_status",
        "pydssp_status",
        "pydssp_residues",
        "helix_residues",
        "strand_residues",
        "coil_residues",
        "cif_path",
        "alphafold_url",
        "download_error",
        "pydssp_error",
        "sequence",
    ]
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metadata_rows)


def write_processed_annotations(
    output_dir: Path,
    annotations: dict[str, np.ndarray],
    metadata_rows: list[dict[str, Any]],
    manifest_path: Path,
    max_proteins: int,
) -> None:
    """Write an AnnotationLoader-compatible processed annotation directory."""
    shard_dir = output_dir / "shard_0"
    shard_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(shard_dir / "annotations.npz", **annotations)
    vocabulary = {
        "concepts": SECONDARY_STRUCTURE_CONCEPTS,
        "metadata": {
            "source": "AlphaFoldDB CIF files processed with pydssp",
            "manifest": str(manifest_path),
            "max_proteins": max_proteins,
            "n_annotated_proteins": len(annotations),
        },
    }
    (output_dir / "concept_vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2),
        encoding="utf-8",
    )
    write_metadata_csv(metadata_rows, output_dir / "secondary_structure_metadata.csv")
    (output_dir / "protein_ids.txt").write_text(
        "\n".join(annotations.keys()) + ("\n" if annotations else ""),
        encoding="utf-8",
    )


def build_structure_secondary_structure_annotations(
    manifest: str = "swissprot_a5_prefixes.txt",
    output_dir: str = "processed_swissprot_a5_structure_secondary_n500",
    max_proteins: int = 500,
    cif_dir: str | None = None,
    download_workers: int = 8,
    overwrite_cifs: bool = False,
    overwrite_annotations: bool = True,
    alphafold_versions: str = "6,5,4,3,2,1",
    timeout_seconds: int = 60,
) -> int:
    """Download AlphaFold CIFs and build pydssp H/E/C annotation matrices.

    Args:
        manifest: Notebook activation manifest. The first ``max_proteins`` IDs are used.
        output_dir: Processed annotation directory compatible with ``AnnotationLoader``.
        max_proteins: Number of manifest proteins to process. Use 0 for all proteins.
        cif_dir: Optional CIF cache directory. Defaults to ``output_dir/alphafold_cifs``.
        download_workers: Number of concurrent AlphaFoldDB downloads.
        overwrite_cifs: Redownload CIF files that are already present.
        overwrite_annotations: Replace an existing processed annotation directory.
        alphafold_versions: Comma-separated AlphaFoldDB model versions to try.
        timeout_seconds: HTTP timeout per CIF download.

    Returns:
        Exit code, where 0 means at least one protein was annotated.
    """
    manifest_path = Path(manifest)
    out_path = Path(output_dir)
    cif_path = Path(cif_dir) if cif_dir is not None else out_path / "alphafold_cifs"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if out_path.exists() and not overwrite_annotations:
        raise FileExistsError(f"Output directory already exists: {out_path}")

    out_path.mkdir(parents=True, exist_ok=True)
    cif_path.mkdir(parents=True, exist_ok=True)
    versions = parse_versions(alphafold_versions)
    protein_ids = read_manifest_protein_ids(manifest_path, max_proteins)
    LOGGER.info(msg=f"Processing {len(protein_ids)} proteins from {manifest_path}")

    download_rows: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, download_workers)) as executor:
        futures = {
            executor.submit(
                download_alphafold_cif,
                protein_id,
                cif_path,
                versions,
                timeout_seconds,
                overwrite_cifs,
            ): protein_id
            for protein_id in protein_ids
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            download_rows[str(row["protein_id"])] = row
            if idx % 25 == 0 or idx == len(futures):
                LOGGER.info(msg=f"Downloaded/checked {idx}/{len(futures)} CIF files")

    annotations: dict[str, np.ndarray] = {}
    metadata_rows: list[dict[str, Any]] = []
    for idx, protein_id in enumerate(protein_ids, start=1):
        download_row = download_rows[protein_id]
        if download_row["download_status"] == "missing_or_error":
            metadata_rows.append({**download_row, "pydssp_status": "skipped_no_cif"})
            continue

        pydssp_row = process_cif_with_pydssp(protein_id, Path(str(download_row["cif_path"])))
        annotation = pydssp_row.pop("annotation")
        metadata_rows.append({**download_row, **pydssp_row})
        if annotation is not None:
            annotations[protein_id] = annotation
        if idx % 25 == 0 or idx == len(protein_ids):
            LOGGER.info(msg=f"Processed pydssp for {idx}/{len(protein_ids)} proteins")

    write_processed_annotations(out_path, annotations, metadata_rows, manifest_path, max_proteins)
    LOGGER.info(msg=f"Wrote {len(annotations)} pydssp-derived annotations to {out_path}")
    return 0 if annotations else 1


if __name__ == "__main__":
    raise SystemExit(tapify(build_structure_secondary_structure_annotations))
