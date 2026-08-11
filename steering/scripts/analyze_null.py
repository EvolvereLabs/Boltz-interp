#!/usr/bin/env python
"""Random-direction NULL-distribution analysis for the 6-concept steering results.

The `random_null` shard applied 19 random matched-norm directions to each of the
97 held-out proteins at k=16 (reused across concepts). This gives, per protein, a
19-sample null of "what a random direction does to each DSSP channel", which we use
two ways:

  (A) POPULATION test: each random direction yields one population-mean own-SS effect
      (mean over 97 proteins). The 19 values form a null; we z-score / percentile the
      real concept-add population-mean effect against it. Automatically absorbs any
      generic "adding a big vector perturbs structure" bias.

  (B) PER-PROTEIN paired test: concept-add own-SS minus the SAME protein's mean over
      its 19 random directions -> a lower-variance random baseline than the single
      rand_add draw. Reported as mean + bootstrap 95% CI.

d_target is NOT used (bugged for SAE shards); own-SS is read from CONCEPT_SS.

Usage: python scripts/analyze_null.py [INPUT_DIR] [OUTPUT_DIR]
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import zlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(REPO, "data", "shard4_20260713")
DEFAULT_OUTPUT = REPO

CONCEPTS = [
    ("helix", "helix.jsonl", "Helix", "probe"),
    ("helix_sae", "helix_sae.jsonl", "Helix", "SAE"),
    ("strand", "strand.jsonl", "Strand", "probe"),
    ("strand_sae", "strand_sae.jsonl", "Strand", "SAE"),
    ("coil", "coil.jsonl", "Coil", "probe"),
    ("coil_sae", "coil_sae.jsonl", "Coil", "SAE"),
]
CONCEPT_SS = {"helix": "d_helix", "helix_sae": "d_helix",
              "strand": "d_strand", "strand_sae": "d_strand",
              "coil": "d_coil", "coil_sae": "d_coil"}
K = 16.0
CB = {"blue": "#0072B2", "vermillion": "#D55E00", "grey": "#999999", "purple": "#CC79A7"}
plt.rcParams.update({"figure.dpi": 300, "savefig.dpi": 300, "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def mean(v):
    v = np.asarray(v, float)
    return float(v.mean()) if v.size else float("nan")


def boot_ci(v, seed, nboot=10000, alpha=0.05):
    v = np.asarray(v, float)
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, v.size, size=(nboot, v.size))].mean(axis=1)
    return float(np.percentile(b, 100 * alpha / 2)), float(np.percentile(b, 100 * (1 - alpha / 2)))


def write_csv(path, header, recs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(recs)


def null_by_direction(null_rows, field):
    """{rand_idx -> {protein -> d_field}} at k=16 for rand_null rows."""
    out = {}
    for r in null_rows:
        if r.get("kind") == "rand_null" and r.get("mult") == K:
            out.setdefault(r["rand_idx"], {})[r["protein"]] = r[field]
    return out


def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    fig_dir = os.path.join(out_dir, "figures")
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    null_rows = load(os.path.join(in_dir, "random_null.jsonl"))
    shards = {k: load(os.path.join(in_dir, fn)) for k, fn, *_ in CONCEPTS}

    # Null: per random direction, its per-protein d_{helix,strand,coil} at k=16.
    null_dir = {ss: null_by_direction(null_rows, ss) for ss in ("d_helix", "d_strand", "d_coil")}
    n_dirs = len(null_dir["d_helix"])
    dir_ids = sorted(null_dir["d_helix"])

    recs = []           # summary per concept
    null_pop = {}       # concept -> list of 19 population-mean null effects (own SS)
    real_pop = {}       # concept -> real population-mean concept-add own-SS effect
    paired19 = {}       # concept -> list of per-protein (concept - mean19random)

    for key, _fn, lbl, method in CONCEPTS:
        ss = CONCEPT_SS[key]
        add = {r["protein"]: r[ss] for r in shards[key]
               if r.get("kind") == "helix_add" and r.get("mult") == K}
        proteins = sorted(add)

        # (A) population null: mean over proteins of each random direction's own-SS effect
        pop_null = []
        for i in dir_ids:
            vals = [null_dir[ss][i][p] for p in proteins if p in null_dir[ss][i]]
            pop_null.append(mean(vals))
        real = mean([add[p] for p in proteins])
        nm, nsd = mean(pop_null), float(np.std(pop_null, ddof=1))
        z = (real - nm) / nsd if nsd else float("nan")
        # one-sided empirical percentile: fraction of null strictly below real
        pct = 100.0 * sum(1 for v in pop_null if v < real) / len(pop_null)
        p_emp = (sum(1 for v in pop_null if v >= real) + 1) / (len(pop_null) + 1)  # right-tail
        null_pop[key] = pop_null
        real_pop[key] = real

        # (B) per-protein paired vs mean-of-19-random
        per = []
        for p in proteins:
            rv = [null_dir[ss][i][p] for i in dir_ids if p in null_dir[ss][i]]
            if rv:
                per.append(add[p] - mean(rv))
        # crc32, not hash(): Python randomizes str hashing per process (PYTHONHASHSEED),
        # which made these bootstrap CIs differ on every run.
        lo, hi = boot_ci(per, seed=zlib.crc32(key.encode()))
        paired19[key] = per

        recs.append([key, lbl, method, ss, real, nm, nsd, z, pct, p_emp,
                     mean(per), lo, hi, len(per), n_dirs])

    write_csv(os.path.join(data_dir, "null_distribution_full97.csv"),
              ["shard", "concept", "method", "own_field",
               "real_pop_effect", "null_pop_mean", "null_pop_std", "z_score",
               "percentile_below", "p_emp_right_tail",
               "paired_vs_mean19", "ci_lo", "ci_hi", "n_proteins", "n_null_dirs"],
              recs)

    # ---- Figure: real effect vs 19-direction null, per concept ----
    labels = [f"{lbl}\n({method})" for _k, _f, lbl, method in CONCEPTS]
    x = np.arange(len(CONCEPTS))
    # Half-width figure (fits ~0.4*textwidth); fonts sized to stay legible at 100% zoom.
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    for i, (key, *_r) in enumerate(CONCEPTS):
        yn = null_pop[key]
        ax.scatter([x[i]] * len(yn), yn, s=28, color=CB["grey"], alpha=0.6,
                   zorder=2, label="random dirs (null)" if i == 0 else None)
    ax.scatter(x, [real_pop[k] for k, *_ in CONCEPTS], s=120, color=CB["vermillion"],
               marker="D", edgecolor="black", linewidth=0.7, zorder=3, label="concept-add (real)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel(r"Population-mean own-SS effect @ $k{=}16$", fontsize=13)
    ax.set_title(f"Real concept-add effect vs\n{n_dirs}-direction random null", fontsize=14)
    # Headroom so the largest real diamond (coil-probe ~+0.066) is not clipped.
    ymax = max(max(real_pop.values()), max(max(v) for v in null_pop.values()))
    ymin = min(min(real_pop.values()), min(min(v) for v in null_pop.values()))
    pad = 0.12 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    fig.tight_layout()
    _null_png = os.path.join(fig_dir, "null_distribution_full97.png")
    fig.savefig(_null_png)
    fig.savefig(_null_png[:-4] + ".pdf")   # vector copy for the manuscript
    plt.close(fig)

    # ---- Report ----
    print(f"random_null: {n_dirs} directions x {len(shards['helix'])//13 or 97} proteins @ k=16\n")
    print("=" * 92)
    print("PER-CONCEPT own-SS effect vs random-direction null (population + per-protein)")
    print("-" * 92)
    print(f"{'concept':14s} {'real':>8s} {'null_mu':>8s} {'null_sd':>8s} {'z':>7s} "
          f"{'pctile':>7s} {'p':>6s} | {'paired_vs_mean19 [95% CI]':>28s}")
    for r in recs:
        key, lbl, method, ss, real, nm, nsd, z, pct, pe, pm, lo, hi, npn, nd = r
        print(f"{lbl+' ('+method+')':14s} {real:+8.4f} {nm:+8.4f} {nsd:8.4f} {z:+7.1f} "
              f"{pct:6.1f}% {pe:6.3f} | {pm:+8.4f} [{lo:+.4f},{hi:+.4f}]")
    print("-" * 92)
    print("z = (real - null_mean)/null_std over the null directions; pctile = % of null below real;")
    print(f"p = right-tail empirical p (min {1/(n_dirs+1):.3f} with {n_dirs} dirs). paired_vs_mean19 =")
    print("per-protein concept-add minus that protein's mean over the 19 random dirs (bootstrap CI).")
    print(f"\nWrote {os.path.join(out_dir, 'data', 'null_distribution_full97.csv')}"
          f" + {os.path.join(out_dir, 'figures', 'null_distribution_full97.png')}")


if __name__ == "__main__":
    main()
