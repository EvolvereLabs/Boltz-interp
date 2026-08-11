"""Structural readouts from Boltz output CIFs — the causal effect sizes.

Flagship: **Sγ–Sγ distance** for annotated disulfide pairs (bonded ≈ 2.05 Å vs broken). Effect of an
intervention = Δdistance(intervened − baseline). Also: DSSP helix fraction (geometry positive
control C3), CA-RMSD vs the baseline structure, and mean pLDDT (CIF B-factor).

The mmCIF ``_atom_site`` parsing mirrors
``SwissProt_annotations/build_structure_secondary_structure_annotations.py`` (kept self-contained
here so the runner has no cross-repo import). Boltz writes coordinates via ``BoltzWriter`` in mmcif.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------------------------------
# mmCIF atom_site parsing
# --------------------------------------------------------------------------------------------------
def _clean(value: str) -> str:
    if value in {".", "?"}:
        return ""
    return value.strip("\"'")


def _parse_atom_site(cif_text: str) -> tuple[list[str], list[list[str]]]:
    """Headers and rows of the first ``_atom_site`` loop."""
    lines = cif_text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip() != "loop_":
            i += 1
            continue
        i += 1
        headers: list[str] = []
        while i < len(lines) and lines[i].strip().startswith("_"):
            headers.append(lines[i].strip())
            i += 1
        if not headers or not headers[0].startswith("_atom_site."):
            continue
        rows: list[list[str]] = []
        while i < len(lines):
            s = lines[i].strip()
            if not s or s == "#" or s == "loop_" or s.startswith("_"):
                break
            rows.append([_clean(t) for t in shlex.split(s, posix=False)])
            i += 1
        return headers, rows
    raise ValueError("No _atom_site loop found in CIF file.")


def _col(headers: list[str], name: str, required: bool = True) -> int | None:
    full = f"_atom_site.{name}"
    if full in headers:
        return headers.index(full)
    if required:
        raise ValueError(f"Missing required CIF column: {full}")
    return None


@dataclass
class Atom:
    chain: str
    seq_id: int
    comp_id: str
    atom_id: str
    xyz: np.ndarray
    bfactor: float


def parse_atoms(cif_path: str | Path, first_chain_only: bool = True) -> list[Atom]:
    """Parse all ATOM records (first model, optionally first chain) from a Boltz mmCIF."""
    headers, rows = _parse_atom_site(Path(cif_path).read_text(encoding="utf-8"))
    g = _col(headers, "group_PDB", required=False)
    a = _col(headers, "label_atom_id")
    c = _col(headers, "label_comp_id")
    s = _col(headers, "label_seq_id")
    ch = _col(headers, "label_asym_id")
    x = _col(headers, "Cartn_x")
    y = _col(headers, "Cartn_y")
    z = _col(headers, "Cartn_z")
    b = _col(headers, "B_iso_or_equiv", required=False)
    m = _col(headers, "pdbx_PDB_model_num", required=False)

    atoms: list[Atom] = []
    first_chain = ""
    for row in rows:
        if len(row) < len(headers):
            continue
        if g is not None and row[g] != "ATOM":
            continue
        if m is not None and row[m] not in {"", "1"}:
            continue
        chain = row[ch]
        if first_chain_only:
            first_chain = first_chain or chain
            if chain != first_chain:
                continue
        try:
            seq_id = int(row[s])
            xyz = np.array([float(row[x]), float(row[y]), float(row[z])], dtype=np.float32)
        except ValueError:
            continue
        bfac = float(row[b]) if (b is not None and row[b]) else np.nan
        atoms.append(Atom(chain, seq_id, row[c].upper(), row[a], xyz, bfac))
    return atoms


# --------------------------------------------------------------------------------------------------
# readouts
# --------------------------------------------------------------------------------------------------
def sg_coords_by_residue(cif_path: str | Path) -> dict[int, np.ndarray]:
    """Map ``seq_id -> Sγ coordinate`` for every cysteine SG atom in the structure."""
    return {
        at.seq_id: at.xyz
        for at in parse_atoms(cif_path)
        if at.atom_id == "SG"
    }


def sg_sg_distances(
    cif_path: str | Path,
    pairs: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    """Sγ–Sγ distance (Å) for each annotated disulfide pair. NaN if either SG is missing.

    Args:
        cif_path: Boltz output mmCIF.
        pairs: 1-based ``(resA, resB)`` residue pairs from UniProt ``DISULFID a..b``.
    """
    sg = sg_coords_by_residue(cif_path)
    out: dict[tuple[int, int], float] = {}
    for a, b in pairs:
        if a in sg and b in sg:
            out[(a, b)] = float(np.linalg.norm(sg[a] - sg[b]))
        else:
            out[(a, b)] = float("nan")
    return out


def is_bonded(distance: float, cutoff: float = 2.5) -> bool:
    """A disulfide is 'formed' if Sγ–Sγ ≤ cutoff (bonded ≈ 2.05 Å; use 2.5 Å slack)."""
    return np.isfinite(distance) and distance <= cutoff


def ca_by_residue(cif_path: str | Path) -> dict[int, np.ndarray]:
    return {at.seq_id: at.xyz for at in parse_atoms(cif_path) if at.atom_id == "CA"}


def mean_plddt(cif_path: str | Path) -> float:
    """Mean CA pLDDT (Boltz stores per-atom confidence in the B-factor column)."""
    vals = [at.bfactor for at in parse_atoms(cif_path) if at.atom_id == "CA" and np.isfinite(at.bfactor)]
    return float(np.mean(vals)) if vals else float("nan")


def ca_rmsd(cif_a: str | Path, cif_b: str | Path) -> float:
    """Kabsch-superposed CA RMSD (Å) between two structures over shared residues."""
    ca_a, ca_b = ca_by_residue(cif_a), ca_by_residue(cif_b)
    shared = sorted(set(ca_a) & set(ca_b))
    if len(shared) < 3:
        return float("nan")
    p = np.stack([ca_a[i] for i in shared]).astype(np.float64)
    q = np.stack([ca_b[i] for i in shared]).astype(np.float64)
    p -= p.mean(0)
    q -= q.mean(0)
    h = p.T @ q
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    diff = q - p @ r.T
    return float(np.sqrt((diff**2).sum() / len(shared)))


def helix_fraction(cif_path: str | Path) -> float:
    """Fraction of residues assigned helix by DSSP (geometry positive control C3).

    Requires ``pydssp`` and a complete N/CA/C/O backbone; returns NaN if unavailable.
    """
    try:
        import pydssp  # type: ignore
    except ImportError:
        return float("nan")

    order = {"N": 0, "CA": 1, "C": 2, "O": 3}
    res: dict[int, np.ndarray] = {}
    for at in parse_atoms(cif_path):
        if at.atom_id in order:
            res.setdefault(at.seq_id, np.full((4, 3), np.nan, dtype=np.float32))[order[at.atom_id]] = at.xyz
    coords = [res[i] for i in sorted(res) if not np.isnan(res[i]).any()]
    if not coords:
        return float("nan")
    ss = np.asarray(pydssp.assign(np.stack(coords)[None], out_type="c3")).reshape(-1)
    return float(np.mean([s == "H" for s in ss.astype(str)]))


def ss_fractions(cif_path: str | Path) -> dict[str, float]:
    """DSSP 3-state fractions {'helix','strand','coil'} in one pass (H / E / everything-else).

    Lets a steering run read the concept it steered (strand -> 'strand') and the full confusion
    matrix (steer helix, watch strand/coil too). Values are NaN if pydssp or the backbone is missing.
    """
    nan = {"helix": float("nan"), "strand": float("nan"), "coil": float("nan")}
    try:
        import pydssp  # type: ignore
    except ImportError:
        return nan

    order = {"N": 0, "CA": 1, "C": 2, "O": 3}
    res: dict[int, np.ndarray] = {}
    for at in parse_atoms(cif_path):
        if at.atom_id in order:
            res.setdefault(at.seq_id, np.full((4, 3), np.nan, dtype=np.float32))[order[at.atom_id]] = at.xyz
    coords = [res[i] for i in sorted(res) if not np.isnan(res[i]).any()]
    if not coords:
        return nan
    ss = np.asarray(pydssp.assign(np.stack(coords)[None], out_type="c3")).reshape(-1).astype(str)
    n = len(ss)
    h = float(np.mean(ss == "H"))
    e = float(np.mean(ss == "E"))
    return {"helix": h, "strand": e, "coil": 1.0 - h - e}  # coil = not H and not E (pydssp c3 loop)
