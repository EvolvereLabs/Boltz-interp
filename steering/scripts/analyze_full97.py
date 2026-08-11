#!/usr/bin/env python
"""Corrected full-97 held-out analysis of the 6-concept Boltz-1 SS steering shards.

CSV-only: this script writes no figures. `own_state_effect_full97.csv` is the source for
the manuscript's paired-steering-effects table.

Covers four deliverables, all as CSVs under <out>/data/:
  1. confusion_full97.csv       — 6 concepts x d_helix/d_strand/d_coil, raw concept-add
                                  mean AND paired (concept-add - rand-add) with CIs, k=16.
  2. own_state_effect_full97.csv — per-concept own-state paired effect over k, with the
                                  k=16 bootstrap 95% CI (manuscript steering table).
  3. plddt_dependence_full97.csv — low <75 vs high >=75 bins + Pearson r, all 6 concepts.
  4. high_strand_full97.csv      — strand-rich deep dive (does strand steering do anything?).

CRITICAL CORRECTION
-------------------
The stored `d_target` field is BUGGED for the SAE shards: it defaulted to d_helix for
strand_sae and coil_sae (verified: d_target == d_helix in 1164/1164 rows for all three
SAE shards). We therefore NEVER trust stored `d_target`; we recompute the own-state effect
from the correct per-SS field (d_helix/d_strand/d_coil) via CONCEPT_SS below.

Usage:
    python scripts/analyze_full97.py [INPUT_DIR] [OUTPUT_DIR]
Defaults: INPUT_DIR = <repo>/data/shard4_20260713, OUTPUT_DIR = <repo>
"""
from __future__ import annotations

import csv
import json
import math
import os
import zlib
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(REPO, "data", "shard4_20260713")
DEFAULT_OUTPUT = REPO

# (key, filename, human label, method)
CONCEPTS = [
    ("helix", "helix.jsonl", "Helix", "probe"),
    ("helix_sae", "helix_sae.jsonl", "Helix", "SAE"),
    ("strand", "strand.jsonl", "Strand", "probe"),
    ("strand_sae", "strand_sae.jsonl", "Strand", "SAE"),
    ("coil", "coil.jsonl", "Coil", "probe"),
    ("coil_sae", "coil_sae.jsonl", "Coil", "SAE"),
]
# The DSSP field each shard's OWN-STATE effect must be read from (fixes the d_target bug).
CONCEPT_SS = {
    "helix": "d_helix", "helix_sae": "d_helix",
    "strand": "d_strand", "strand_sae": "d_strand",
    "coil": "d_coil", "coil_sae": "d_coil",
}
SS_COLS = ["d_helix", "d_strand", "d_coil"]
K = 16.0
KS = [1.0, 2.0, 4.0, 8.0, 16.0]
PLDDT_CUT = 75.0
RNG_SEED = 0
N_BOOT = 10000


# --------------------------------------------------------------------------- #
# Loading / stats
# --------------------------------------------------------------------------- #
def load_jsonl(path):
    with open(path, "r") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def rows_for(rows, kind, mult):
    return [r for r in rows if r.get("kind") == kind and r.get("mult") == float(mult)]


def paired(rows, field, mult=K, kind_a="helix_add", kind_b="rand_add"):
    """Per-protein (concept-add - random-add) on `field`. kind labels are generic
    ('helix_add'/'rand_add' = concept-add/random-add) regardless of steered SS."""
    a = {r["protein"]: r[field] for r in rows_for(rows, kind_a, mult)}
    b = {r["protein"]: r[field] for r in rows_for(rows, kind_b, mult)}
    common = sorted(set(a) & set(b))
    return np.array([a[p] - b[p] for p in common]), common


def mean(v):
    v = np.asarray(v, float)
    return float(v.mean()) if v.size else float("nan")


def sem(v):
    v = np.asarray(v, float)
    return float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0


def boot_ci(v, seed=RNG_SEED, nboot=N_BOOT, alpha=0.05):
    v = np.asarray(v, float)
    if v.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(nboot, v.size))
    boots = v[idx].mean(axis=1)
    return (float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def write_csv(path, header, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(records)


# --------------------------------------------------------------------------- #
# 1. Confusion matrices (raw + paired) at k=16
# --------------------------------------------------------------------------- #
def confusion(shards, data_dir):
    raw = np.zeros((len(CONCEPTS), 3))
    pair = np.zeros((len(CONCEPTS), 3))
    pair_lo = np.zeros((len(CONCEPTS), 3))
    pair_hi = np.zeros((len(CONCEPTS), 3))
    for i, (key, *_r) in enumerate(CONCEPTS):
        add_rows = rows_for(shards[key], "helix_add", K)
        for j, col in enumerate(SS_COLS):
            raw[i, j] = mean([r[col] for r in add_rows])
            d, _ = paired(shards[key], col, K)
            pair[i, j] = mean(d)
            lo, hi = boot_ci(d, seed=RNG_SEED + i * 3 + j)
            pair_lo[i, j], pair_hi[i, j] = lo, hi

    # CSV: both raw and paired, with CIs
    recs = []
    for i, (key, _fn, lbl, method) in enumerate(CONCEPTS):
        recs.append([key, lbl, method,
                     raw[i, 0], raw[i, 1], raw[i, 2],
                     pair[i, 0], pair[i, 1], pair[i, 2],
                     pair_lo[i, 0], pair_hi[i, 0],
                     pair_lo[i, 1], pair_hi[i, 1],
                     pair_lo[i, 2], pair_hi[i, 2]])
    write_csv(os.path.join(data_dir, "confusion_full97.csv"),
              ["shard", "concept", "method",
               "raw_d_helix", "raw_d_strand", "raw_d_coil",
               "paired_d_helix", "paired_d_strand", "paired_d_coil",
               "helix_lo", "helix_hi", "strand_lo", "strand_hi", "coil_lo", "coil_hi"],
              recs)
    return raw, pair, pair_lo, pair_hi


# --------------------------------------------------------------------------- #
# 1b. Own-state (diagonal) effect vs k, with bootstrap CI at k=16
# --------------------------------------------------------------------------- #
def own_state(shards, data_dir):
    recs = []
    diag = {}
    for key, _fn, lbl, method in CONCEPTS:
        field = CONCEPT_SS[key]
        row = {"shard": key, "label": lbl, "method": method}
        for k in KS:
            d, _ = paired(shards[key], field, k)
            row[f"k{int(k)}"] = mean(d)
        d16, common = paired(shards[key], field, K)
        # crc32, not hash(): Python randomizes str hashing per process (PYTHONHASHSEED),
        # which made these bootstrap CIs differ on every run.
        lo, hi = boot_ci(d16, seed=zlib.crc32(key.encode()))
        row["mean_k16"] = mean(d16)
        row["sem_k16"] = sem(d16)
        row["ci_lo"] = lo
        row["ci_hi"] = hi
        row["n"] = len(common)
        diag[key] = row
        recs.append([key, lbl, method] + [row[f"k{int(k)}"] for k in KS] +
                    [row["mean_k16"], row["sem_k16"], lo, hi, row["n"]])
    write_csv(os.path.join(data_dir, "own_state_effect_full97.csv"),
              ["shard", "concept", "method"] + [f"k{int(k)}" for k in KS] +
              ["mean_k16", "sem_k16", "ci_lo", "ci_hi", "n"], recs)
    return diag


# --------------------------------------------------------------------------- #
# 2. pLDDT dependence, all 6 concepts
# --------------------------------------------------------------------------- #
def pearson(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if xs.size < 2:
        return float("nan")
    mx, my = xs.mean(), ys.mean()
    sx, sy = math.sqrt(((xs - mx) ** 2).sum()), math.sqrt(((ys - my) ** 2).sum())
    return float(((xs - mx) * (ys - my)).sum() / (sx * sy)) if sx and sy else float("nan")


def plddt_analysis(shards, data_dir):
    per = {}  # key -> list of (plddt, paired_own_effect)
    for key, *_ in CONCEPTS:
        field = CONCEPT_SS[key]
        add = {r["protein"]: r[field] for r in rows_for(shards[key], "helix_add", K)}
        rnd = {r["protein"]: r[field] for r in rows_for(shards[key], "rand_add", K)}
        pl = {r["protein"]: r["base_plddt"] for r in shards[key] if r["condition"] == "baseline"}
        per[key] = [(pl[p], add[p] - rnd[p]) for p in sorted(set(add) & set(rnd) & set(pl))]

    recs = []
    for key, _fn, lbl, method in CONCEPTS:
        lo = [e for p, e in per[key] if p < PLDDT_CUT]
        hi = [e for p, e in per[key] if p >= PLDDT_CUT]
        r = pearson([p for p, _ in per[key]], [e for _, e in per[key]])
        recs.append([key, lbl, method, mean(lo), sem(lo), len(lo),
                     mean(hi), sem(hi), len(hi), r])
    write_csv(os.path.join(data_dir, "plddt_dependence_full97.csv"),
              ["shard", "concept", "method", "low_mean", "low_sem", "low_n",
               "high_mean", "high_sem", "high_n", "pearson_r"], recs)
    return per


# --------------------------------------------------------------------------- #
# 3. High-strand protein deep dive
# --------------------------------------------------------------------------- #
def high_strand(shards, data_dir, strand_cut=0.30):
    base = {r["protein"]: r for r in shards["strand"] if r["condition"] == "baseline"}
    strand_rich = sorted([p for p, r in base.items() if r["strand"] > strand_cut],
                         key=lambda p: -base[p]["strand"])

    def get(shard, kind, k, field):
        return {r["protein"]: r[field] for r in rows_for(shards[shard], kind, k)}

    recs = []
    for shard in ("strand", "strand_sae"):
        add_s = get(shard, "helix_add", K, "d_strand")
        rnd_s = get(shard, "rand_add", K, "d_strand")
        add_h = get(shard, "helix_add", K, "d_helix")
        add_c = get(shard, "helix_add", K, "d_coil")
        for p in strand_rich:
            recs.append([shard, p, base[p]["strand"], base[p]["base_plddt"],
                         add_s.get(p, float("nan")), rnd_s.get(p, float("nan")),
                         add_s.get(p, float("nan")) - rnd_s.get(p, float("nan")),
                         add_h.get(p, float("nan")), add_c.get(p, float("nan"))])
    write_csv(os.path.join(data_dir, "high_strand_full97.csv"),
              ["shard", "protein", "base_strand", "base_plddt",
               "d_strand_add", "d_strand_rand", "d_strand_paired",
               "d_helix_add", "d_coil_add"], recs)

    # Fragility test: on strand-rich proteins, is |d_strand| under random perturbation
    # larger than under structured strand-add? (strand fragile / destroyed by noise)
    frag = {}
    for shard in ("strand", "strand_sae"):
        add_s = {r["protein"]: r["d_strand"] for r in rows_for(shards[shard], "helix_add", K)}
        rnd_s = {r["protein"]: r["d_strand"] for r in rows_for(shards[shard], "rand_add", K)}
        common = [p for p in strand_rich if p in add_s and p in rnd_s]
        frag[shard] = {
            "mean_d_strand_add": mean([add_s[p] for p in common]),
            "mean_d_strand_rand": mean([rnd_s[p] for p in common]),
            "mean_abs_add": mean([abs(add_s[p]) for p in common]),
            "mean_abs_rand": mean([abs(rnd_s[p]) for p in common]),
            "n": len(common),
        }
    return strand_rich, recs, frag


# --------------------------------------------------------------------------- #
def main():
    in_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    print(f"Input : {in_dir}\nOutput: {out_dir}\n")

    shards = {key: load_jsonl(os.path.join(in_dir, fn)) for key, fn, *_ in CONCEPTS}

    raw, pair, plo, phi = confusion(shards, data_dir)
    diag = own_state(shards, data_dir)
    per = plddt_analysis(shards, data_dir)
    strand_rich, hs_recs, frag = high_strand(shards, data_dir)

    # ---- Report ----
    print("=" * 74)
    print("1. OWN-STATE (diagonal) PAIRED EFFECT  concept-add - rand-add  @ k=16")
    print("   (d_target recomputed from correct per-SS field; SAE bug bypassed)")
    print("-" * 74)
    print(f"   {'concept':14s} {'k16':>8s} {'95% CI':>20s} {'n':>4s}   verdict")
    for key, _fn, lbl, method in CONCEPTS:
        d = diag[key]
        real = d["ci_lo"] > 0 or d["ci_hi"] < 0
        v = "STEERS" if (real and d["mean_k16"] > 0) else ("WRONG-SIGN" if real else "null")
        print(f"   {lbl+' ('+method+')':14s} {d['mean_k16']:+8.4f} "
              f"[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}] {d['n']:4d}   {v}")

    print("\n" + "=" * 74)
    print("2. CONFUSION MATRIX (paired concept-add - rand-add, k=16)")
    print("-" * 74)
    print(f"   {'steered':14s} {'Dhelix':>9s} {'Dstrand':>9s} {'Dcoil':>9s}")
    for i, (key, _fn, lbl, method) in enumerate(CONCEPTS):
        print(f"   {lbl+' ('+method+')':14s} {pair[i,0]:+9.4f} {pair[i,1]:+9.4f} {pair[i,2]:+9.4f}")
    print("   (raw concept-add version in confusion_full97.csv)")

    print("\n" + "=" * 74)
    print("3. pLDDT DEPENDENCE (paired own-SS effect, k=16)")
    print("-" * 74)
    print(f"   {'concept':14s} {'low<75':>9s} {'high>=75':>9s} {'ratio':>7s} {'pearson_r':>10s}")
    for key, _fn, lbl, method in CONCEPTS:
        lo = [e for p, e in per[key] if p < PLDDT_CUT]
        hi = [e for p, e in per[key] if p >= PLDDT_CUT]
        r = pearson([p for p, _ in per[key]], [e for _, e in per[key]])
        ratio = (mean(lo) / mean(hi)) if mean(hi) else float("nan")
        print(f"   {lbl+' ('+method+')':14s} {mean(lo):+9.4f} {mean(hi):+9.4f} "
              f"{ratio:7.1f} {r:+10.2f}   (n_lo={len(lo)}, n_hi={len(hi)})")

    print("\n" + "=" * 74)
    print(f"4. HIGH-STRAND DEEP DIVE  (baseline strand>0.30: n={len(strand_rich)} proteins)")
    print("-" * 74)
    print("   Held-out set is strand-POOR (max baseline strand ~0.47). Underpowered by design.")
    for shard in ("strand", "strand_sae"):
        f = frag[shard]
        print(f"\n   [{shard}] on {f['n']} strand-rich proteins, k=16:")
        print(f"     mean d_strand  concept-add {f['mean_d_strand_add']:+.4f}  "
              f"random-add {f['mean_d_strand_rand']:+.4f}  "
              f"paired {f['mean_d_strand_add']-f['mean_d_strand_rand']:+.4f}")
        print(f"     mean |d_strand|  concept-add {f['mean_abs_add']:.4f}  "
              f"random-add {f['mean_abs_rand']:.4f}  "
              f"(if rand>=add: strand fragile to any perturbation)")
    print("\n   Per-protein (strand shard), most strand-rich first:")
    print(f"   {'protein':12s} {'base_s':>7s} {'plddt':>6s} {'dS_add':>8s} {'dS_rand':>8s} "
          f"{'dS_pair':>8s} {'dH_add':>8s} {'dC_add':>8s}")
    for rec in hs_recs:
        if rec[0] != "strand":
            continue
        _, p, bs, pl, dsa, dsr, dsp, dha, dca = rec
        print(f"   {p:12s} {bs:7.3f} {pl:6.1f} {dsa:+8.4f} {dsr:+8.4f} {dsp:+8.4f} "
              f"{dha:+8.4f} {dca:+8.4f}")

    print("\nWrote CSVs -> paper/data/")
    print("DONE")


if __name__ == "__main__":
    main()
