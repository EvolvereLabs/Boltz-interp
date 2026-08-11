#!/usr/bin/env python
"""Generate the steering confusion-matrix figure for the 6-concept Boltz-1
secondary-structure steering results (helix / strand / coil, each probe + SAE).

Reads six per-protein JSONL shards, computes summary statistics locally, and writes:
  <out>/figures/*.png   (300 dpi, colorblind-friendly)
  <out>/data/*.csv      (exact plotted values, one CSV per figure)

Figures:
  fig7_confusion_matrix.png        — 6x3 heatmap of concept-add SS changes

Usage:
    python scripts/make_ss_figures.py [INPUT_DIR] [OUTPUT_DIR]

Data-schema notes:
  * `kind` is always in {helix_add, rand_add, helix_ablate, rand_ablate}
    regardless of the steered concept: "helix_add"/"rand_add" mean
    concept-add / random-add. The steered concept is given by the file name.
  * `d_target` is BUGGED for the SAE shards (defaulted to d_helix for strand_sae /
    coil_sae). Do NOT use it. This figure reads only `d_helix`/`d_strand`/`d_coil`,
    which are all three correct. (Own-state effects live in analyze_full97.py.)
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(REPO, "data", "shard4_20260713")
DEFAULT_OUTPUT = REPO

# The six shards, in display order: (key, filename, human label, method)
CONCEPTS = [
    ("helix", "helix.jsonl", "Helix", "probe"),
    ("helix_sae", "helix_sae.jsonl", "Helix", "SAE"),
    ("strand", "strand.jsonl", "Strand", "probe"),
    ("strand_sae", "strand_sae.jsonl", "Strand", "SAE"),
    ("coil", "coil.jsonl", "Coil", "probe"),
    ("coil_sae", "coil_sae.jsonl", "Coil", "SAE"),
]

# ---------------------------------------------------------------------------
# Consistent style
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.autolayout": False,
    }
)

K = 16  # steering dose multiplier used for all figures


# ---------------------------------------------------------------------------
# Data loading / stats helpers
# ---------------------------------------------------------------------------
def load_jsonl(path):
    with open(path, "r") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def mean(values):
    a = np.asarray(values, dtype=float)
    return float(np.mean(a)) if a.size else float("nan")


def rows_for(rows, kind, mult):
    return [r for r in rows if r.get("kind") == kind and r.get("mult") == float(mult)]


def write_csv(path, header, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(records)


def verify_png(path):
    ok = os.path.isfile(path) and os.path.getsize(path) > 0
    size = os.path.getsize(path) if os.path.isfile(path) else 0
    print(f"  {'OK ' if ok else 'FAIL'} {path} ({size} bytes)")
    return ok


# ===========================================================================
# Figure 7: Confusion matrix — 6x3 heatmap of concept-add SS change at k=16
# ===========================================================================
def fig7_confusion(shards, fig_dir, data_dir):
    """Rows = steered shard, cols = measured SS (d_helix/d_strand/d_coil),
    values = mean concept-add effect at k=16. NOTE: this is the raw concept-add
    mean (not paired), matching the requested expected values."""
    ss_cols = ["d_helix", "d_strand", "d_coil"]
    col_labels = [r"$\Delta$ helix", r"$\Delta$ strand", r"$\Delta$ coil"]
    row_labels = [f"{lbl} ({method})" for _k, _f, lbl, method in CONCEPTS]

    M = np.zeros((len(CONCEPTS), 3))
    for i, (key, _fn, _lbl, _m) in enumerate(CONCEPTS):
        rows = rows_for(shards[key], "helix_add", K)  # "helix_add" == concept-add
        for j, col in enumerate(ss_cols):
            M[i, j] = mean([r[col] for r in rows])

    vmax = float(np.max(np.abs(M))) or 1e-3
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(CONCEPTS)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Measured secondary-structure change")
    ax.set_ylabel("Steered concept")
    ax.set_title("Confusion matrix: helix↔coil trade off;\n"
                 "strand column is ~0 (strand not inducible)", fontsize=11)
    ax.grid(False)
    # Minor-grid cell borders.
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CONCEPTS), 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="minor", color="white", linewidth=1.2)

    for i in range(len(CONCEPTS)):
        for j in range(3):
            v = M[i, j]
            # Contrast: white text on saturated cells.
            frac = abs(v) / vmax
            color = "white" if frac > 0.55 else "black"
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Mean concept-add $\Delta$ fraction ($k{=}16$)", fontsize=9)

    fig.tight_layout()
    png = os.path.join(fig_dir, "fig7_confusion_matrix.png")
    fig.savefig(png)
    plt.close(fig)

    recs = []
    for i, (key, _fn, lbl, method) in enumerate(CONCEPTS):
        recs.append([key, lbl, method, M[i, 0], M[i, 1], M[i, 2]])
    write_csv(
        os.path.join(data_dir, "fig7_confusion_matrix.csv"),
        ["shard", "concept", "method", "mean_d_helix", "mean_d_strand", "mean_d_coil"],
        recs,
    )
    return png, M


# ===========================================================================
def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    fig_dir = os.path.join(out_dir, "figures")
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    print(f"Input dir : {in_dir}")
    print(f"Output dir: {out_dir}")

    shards = {}
    for key, fn, _lbl, _m in CONCEPTS:
        shards[key] = load_jsonl(os.path.join(in_dir, fn))

    print("\nGenerating figures...")
    pngs = []
    p7, M7 = fig7_confusion(shards, fig_dir, data_dir); pngs.append(p7)

    print("\nVerifying PNGs:")
    all_ok = all(verify_png(p) for p in pngs)

    print("\n=== Fig 7: confusion matrix (mean concept-add @ k=16) ===")
    print(f"  {'concept':14s} {'d_helix':>9s} {'d_strand':>9s} {'d_coil':>9s}")
    for i, (key, _fn, lbl, method) in enumerate(CONCEPTS):
        print(f"  {lbl+' ('+method+')':14s} {M7[i,0]:+9.4f} {M7[i,1]:+9.4f} {M7[i,2]:+9.4f}")

    print("\nDONE" if all_ok else "\nDONE (with FAILED pngs)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
