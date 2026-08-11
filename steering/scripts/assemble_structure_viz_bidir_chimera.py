"""Assemble the Chimera-rendered A6NI15 panels into the bidirectional steering figure.

Produces ``figures/structure_bidir_3d_chimera.{png,pdf}`` (manuscript Fig. 2, right
panel). The three structures (coil-steered / baseline / helix-steered) are rendered as
ray-quality secondary-structure ribbons in UCSF Chimera, then composited here into a
2x2 layout (three panels + title/legend cell).

Panel PNGs are a PREREQUISITE and are not produced by this repo: render them in the
Chimera GUI first, with transparent backgrounds and a SHARED camera, into ``PANEL_DIR``
as ``panel_coil_steered.png`` / ``panel_baseline.png`` / ``panel_helix_steered.png``.
All three are then cropped to one common bounding box -> relative size/extent is
preserved (the coil-steered chain reads as visibly more extended, exactly as in the
data).

SS fractions in the titles are recomputed with the repo's own DSSP path
(``boltz_causal.structural_readout`` + pydssp, via ``Struct`` below) so the numbers
match the rest of the paper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from boltz_causal.structural_readout import parse_atoms  # noqa: E402

# Secondary-structure color scheme (matches the Chimera ribbon colors).
SS_COLORS = {"H": "#d62728", "E": "#e6c200", "-": "#9e9e9e"}  # helix red, strand yellow, coil grey
SS_LABELS = {"H": "Helix", "E": "Strand", "-": "Coil"}
_BACKBONE = {"N": 0, "CA": 1, "C": 2, "O": 3}


def residue_backbone(cif_path: str | Path) -> tuple[list[int], np.ndarray]:
    """Return (sorted residue ids, (L,4,3) backbone coords) for residues with a full N/CA/C/O.

    Residues missing any backbone atom are skipped (DSSP needs the complete backbone). Mirrors
    the coordinate assembly in ``structural_readout.helix_fraction`` / ``ss_fractions``.
    """
    res: dict[int, np.ndarray] = {}
    for at in parse_atoms(cif_path):
        if at.atom_id in _BACKBONE:
            res.setdefault(at.seq_id, np.full((4, 3), np.nan, dtype=np.float32))[_BACKBONE[at.atom_id]] = at.xyz
    ids = [i for i in sorted(res) if not np.isnan(res[i]).any()]
    if not ids:
        return [], np.zeros((0, 4, 3), dtype=np.float32)
    return ids, np.stack([res[i] for i in ids])  # (L, 4, 3)


def dssp_ss(backbone: np.ndarray) -> np.ndarray:
    """Per-residue 3-state SS ('H'/'E'/'-') via pydssp, same call as ``ss_fractions``."""
    import pydssp  # type: ignore

    ss = np.asarray(pydssp.assign(backbone[None], out_type="c3")).reshape(-1)
    return ss.astype(str)


class Struct:
    """One parsed + DSSP-annotated structure."""

    def __init__(self, name: str, cif_path: str | Path):
        self.name = name
        self.ids, bb = residue_backbone(cif_path)
        if not self.ids:
            raise ValueError(f"No complete-backbone residues in {cif_path}")
        self.ss = dssp_ss(bb)
        self.helix_frac = float(np.mean(self.ss == "H"))
        self.strand_frac = float(np.mean(self.ss == "E"))
        self.coil_frac = float(np.mean(self.ss == "-"))


CIF_DIR = _REPO_ROOT / "data" / "shard4_20260713" / "cifs_A6NI15"
# Pre-rendered Chimera ribbon panels (see the Chimera step in steering/README.md).
# Override with: python scripts/assemble_structure_viz_bidir_chimera.py <panel-dir>
DEFAULT_PANEL_DIR = _REPO_ROOT / "panels"

# (struct label, cif filename, panel png) in the display order coil | baseline | helix.
ITEMS = [
    ("coil-steered", "A6NI15_coil_steered.cif", "panel_coil_steered.png"),
    ("baseline", "A6NI15_baseline.cif", "panel_baseline.png"),
    ("helix-steered", "A6NI15_helix_steered.cif", "panel_helix_steered.png"),
]


def alpha_bbox(im: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of non-transparent (alpha>0) pixels."""
    a = np.asarray(im.convert("RGBA"))[:, :, 3]
    ys, xs = np.where(a > 0)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def load_common_crop(paths: list[Path], pad_frac: float = 0.04) -> list[Image.Image]:
    """Crop every panel to the UNION alpha-bbox so relative scale is preserved."""
    ims = [Image.open(p).convert("RGBA") for p in paths]
    boxes = [alpha_bbox(im) for im in ims]
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    pad = int(round(max(x1 - x0, y1 - y0) * pad_frac))
    W, H = ims[0].size
    box = (max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad))
    return [im.crop(box) for im in ims]


def main() -> None:
    panel_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PANEL_DIR
    if not panel_dir.is_dir():
        raise SystemExit(
            f"panel directory not found: {panel_dir}\n"
            "Render the three Chimera ribbon panels first (see steering/README.md), "
            "then pass their directory as the first argument."
        )
    structs = {label: Struct(label, CIF_DIR / cif) for label, cif, _ in ITEMS}
    panels = load_common_crop([panel_dir / png for *_, png in ITEMS])

    fig = plt.figure(figsize=(10.5, 9.6))
    FS_TITLE, FS_LEG, FS_HEAD, FS_NOTE = 18, 16, 19, 13

    for pos, (label, _, _), im in zip((1, 2, 3), ITEMS, panels):
        ax = fig.add_subplot(2, 2, pos)
        ax.imshow(np.asarray(im), interpolation="lanczos")
        ax.axis("off")
        s = structs[label]
        ax.set_title(
            f"{label}\nhelix = {s.helix_frac:.3f}   coil = {s.coil_frac:.3f}",
            fontsize=FS_TITLE, pad=6,
        )

    axL = fig.add_subplot(2, 2, 4); axL.axis("off")
    axL.text(0.5, 0.86, "A6NI15", ha="center", va="center",
             fontsize=FS_HEAD, fontweight="bold", transform=axL.transAxes)
    axL.text(0.5, 0.72, "bidirectional trunk steering", ha="center", va="center",
             fontsize=FS_NOTE, transform=axL.transAxes)
    legend_items = [
        Line2D([0], [0], color=SS_COLORS["H"], lw=4, label=SS_LABELS["H"]),
        Line2D([0], [0], color=SS_COLORS["E"], lw=4, label=SS_LABELS["E"]),
        Line2D([0], [0], color=SS_COLORS["-"], lw=4, label=SS_LABELS["-"]),
    ]
    axL.legend(handles=legend_items, loc="center", bbox_to_anchor=(0.5, 0.38),
               ncol=1, frameon=False, fontsize=FS_LEG, handlelength=2.2,
               title="secondary structure", title_fontsize=FS_LEG)

    fig.tight_layout()
    out_dirs = [_REPO_ROOT / "figures"]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        png = d / "structure_bidir_3d_chimera.png"
        fig.savefig(png, dpi=300)
        fig.savefig(png.with_suffix(".pdf"))
        print(f"wrote {png}")
    plt.close(fig)

    print("-" * 60)
    for label, cif, _ in ITEMS:
        s = structs[label]
        print(f"  {label:14s} helix={s.helix_frac:.3f} strand={s.strand_frac:.3f} "
              f"coil={s.coil_frac:.3f}  n={len(s.ids)}")


if __name__ == "__main__":
    main()
