#!/usr/bin/env python
"""Generate publication-quality figures + underlying CSV data for the
Boltz-1 causal steering paper.

Reads the per-protein steering JSONL outputs, computes summary statistics
(means, SEM = std/sqrt(n)) locally, and writes:
  <out>/figures/*.png   (300 dpi, colorblind-friendly)
  <out>/data/*.csv      (exact plotted values, one CSV per figure)

Figures (manuscript supplementary "steering is dose-dependent" / "necessity is null"):
  fig1_sufficiency_dose_response.png  — additive dose sweep vs matched-norm random
  fig4_necessity_ablation.png         — trunk / diffusion / both ablation is null

Usage:
    python scripts/make_paper_plots.py [INPUT_DIR] [OUTPUT_DIR]

Defaults:
    INPUT_DIR  = steering/data/section5_20260708 (committed)
    OUTPUT_DIR = steering/  -> writes figures/ and data/
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The section-5 single-instance steering rows, committed under steering/data/.
DEFAULT_INPUT = os.path.join(REPO, "data", "section5_20260708")
DEFAULT_OUTPUT = REPO

# ---------------------------------------------------------------------------
# Colorblind-friendly palette (Wong 2011) + consistent style
# ---------------------------------------------------------------------------
CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "skyblue": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#999999",
    "black": "#000000",
}

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

MULTS = [1, 2, 4, 8, 16]


# ---------------------------------------------------------------------------
# Data loading / stats helpers
# ---------------------------------------------------------------------------
def load_jsonl(path):
    with open(path, "r") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sem(values):
    """Standard error of the mean = std(ddof=1) / sqrt(n)."""
    a = np.asarray(values, dtype=float)
    n = a.size
    if n <= 1:
        return 0.0
    return float(np.std(a, ddof=1) / math.sqrt(n))


def mean(values):
    a = np.asarray(values, dtype=float)
    return float(np.mean(a)) if a.size else float("nan")


def rows_for(rows, kind, mult):
    return [r for r in rows if r.get("kind") == kind and r.get("mult") == float(mult)]


def paired_values(rows, kind_a, kind_b, mult):
    """Per-protein (kind_a - kind_b) d_helix at a given mult."""
    a = {r["protein"]: r["d_helix"] for r in rows_for(rows, kind_a, mult)}
    b = {r["protein"]: r["d_helix"] for r in rows_for(rows, kind_b, mult)}
    common = sorted(set(a) & set(b))
    return [a[p] - b[p] for p in common], common


def write_csv(path, header, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(records)


def verify_png(path):
    ok = os.path.isfile(path) and os.path.getsize(path) > 0
    print(f"  {'OK ' if ok else 'FAIL'} {path} ({os.path.getsize(path) if os.path.isfile(path) else 0} bytes)")
    return ok


# ===========================================================================
# Figure 1: Sufficiency dose-response
# ===========================================================================
def fig1_dose_response(e1, e5, fig_dir, data_dir):
    series = {
        "probe_helix": (e1, "helix_add", CB["blue"], "Probe helix-add (E1)", "-", "o"),
        "probe_rand": (e1, "rand_add", CB["skyblue"], "Probe random-add", "--", "o"),
        "sae_helix": (e5, "helix_add", CB["vermillion"], "SAE helix-add (E5)", "-", "s"),
        "sae_rand": (e5, "rand_add", CB["orange"], "SAE random-add", "--", "s"),
    }

    plotted = {}  # name -> (means, sems)
    for name, (rows, kind, _c, _lbl, _ls, _mk) in series.items():
        means, sems = [], []
        for k in MULTS:
            vals = [r["d_helix"] for r in rows_for(rows, kind, k)]
            means.append(mean(vals))
            sems.append(sem(vals))
        plotted[name] = (means, sems)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for name, (rows, kind, c, lbl, ls, mk) in series.items():
        means, sems = plotted[name]
        ax.errorbar(
            MULTS, means, yerr=sems, label=lbl, color=c, linestyle=ls,
            marker=mk, markersize=5, linewidth=1.8, capsize=3, capthick=1.0,
        )
    ax.axhline(0.0, color=CB["grey"], linewidth=0.8, zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(MULTS)
    ax.set_xticklabels([str(k) for k in MULTS])
    ax.set_xlabel("Steering dose multiplier $k$ (log$_2$)")
    ax.set_ylabel(r"Mean $\Delta$ helix fraction")
    ax.set_title("Sufficiency: dose-response of helix steering (trunk)")
    ax.legend(loc="upper left", fontsize=9)

    # Annotate paired concept-minus-random at k=16
    probe_paired, _ = paired_values(e1, "helix_add", "rand_add", 16)
    sae_paired, _ = paired_values(e5, "helix_add", "rand_add", 16)
    p16 = mean(probe_paired)
    s16 = mean(sae_paired)
    ax.annotate(
        f"paired (helix$-$rand) @k=16:\nprobe {p16:+.3f}, SAE {s16:+.3f}",
        xy=(16, plotted["probe_helix"][0][-1]),
        xytext=(3.0, max(plotted["probe_helix"][0]) * 0.62),
        fontsize=8.5, color=CB["black"],
        arrowprops=dict(arrowstyle="->", color=CB["grey"], lw=0.8),
    )
    fig.tight_layout()
    png = os.path.join(fig_dir, "fig1_sufficiency_dose_response.png")
    fig.savefig(png)
    plt.close(fig)

    # CSV: one row per (series, k)
    recs = []
    for name, (rows, kind, _c, lbl, _ls, _mk) in series.items():
        means, sems = plotted[name]
        for i, k in enumerate(MULTS):
            n = len(rows_for(rows, kind, k))
            recs.append([name, lbl, k, means[i], sems[i], n])
    write_csv(
        os.path.join(data_dir, "fig1_sufficiency_dose_response.csv"),
        ["series", "label", "k", "mean_d_helix", "sem_d_helix", "n"],
        recs,
    )
    return png, {"probe_paired_k16": p16, "sae_paired_k16": s16}


# ===========================================================================
# Figure 4: Necessity (E3 ablation) — mean d_helix for trunk_only/diffusion_only/both
# ===========================================================================
def fig4_necessity(e3, fig_dir, data_dir):
    sites = ["trunk_only", "diffusion_only", "both"]
    means, sems, ns = [], [], []
    for s in sites:
        vals = [r["d_helix"] for r in e3 if r["site"] == s]
        means.append(mean(vals))
        sems.append(sem(vals))
        ns.append(len(vals))

    colors = [CB["blue"], CB["orange"], CB["purple"]]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    x = np.arange(len(sites))
    ax.bar(x, means, yerr=sems, color=colors, capsize=5, width=0.6,
           edgecolor="black", linewidth=0.6, error_kw=dict(lw=1.1))
    ax.axhline(0.0, color=CB["grey"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Trunk only", "Diffusion only", "Both"])
    ax.set_ylabel(r"Mean $\Delta$ helix fraction (ablation)")
    ax.set_title("Necessity: helix ablation has no effect (E3)")
    span = max(abs(m) + s for m, s in zip(means, sems)) or 1e-3
    ax.set_ylim(-span * 2.2, span * 2.2)
    for xi, m, s, n in zip(x, means, sems, ns):
        ax.text(xi, m + s + span * 0.2 * (1 if m >= 0 else -1),
                f"{m:+.4f}\n(n={n})", ha="center",
                va="bottom" if m >= 0 else "top", fontsize=8.5)
    fig.tight_layout()
    png = os.path.join(fig_dir, "fig4_necessity_ablation.png")
    fig.savefig(png)
    plt.close(fig)

    write_csv(
        os.path.join(data_dir, "fig4_necessity_ablation.csv"),
        ["site", "mean_d_helix", "sem", "n"],
        [[s, m, se, n] for s, m, se, n in zip(sites, means, sems, ns)],
    )
    return png, {s: m for s, m in zip(sites, means)}


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

    e1 = load_jsonl(os.path.join(in_dir, "probe_batch", "batch_trunk.jsonl"))
    e5 = load_jsonl(os.path.join(in_dir, "probe_batch_helix_sae", "batch_trunk.jsonl"))
    e3 = load_jsonl(os.path.join(in_dir, "probe_combined", "combined.jsonl"))

    print("\nGenerating figures...")
    pngs = []
    p1, s1 = fig1_dose_response(e1, e5, fig_dir, data_dir); pngs.append(p1)
    p4, s4 = fig4_necessity(e3, fig_dir, data_dir); pngs.append(p4)

    print("\nVerifying PNGs:")
    all_ok = all(verify_png(p) for p in pngs)

    print("\n=== Computed statistics ===")
    print(f"Fig1 paired @k16: probe {s1['probe_paired_k16']:+.4f}  SAE {s1['sae_paired_k16']:+.4f}")
    print(f"Fig4 ablation means: {', '.join(f'{k}={v:+.4f}' for k,v in s4.items())}")

    print("\nDONE" if all_ok else "\nDONE (with FAILED pngs)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
