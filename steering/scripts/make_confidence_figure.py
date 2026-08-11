#!/usr/bin/env python3
"""Figure: steering success depends on model confidence (pLDDT).

Low-confidence structures steer far more than high-confidence ones -- the trunk conditioning has
more sway where the fold is not yet committed. Reads the per-concept shard JSONLs and plots, per
concept, the paired (concept-add - random-add) effect on the concept's own DSSP state (d_target,
k=16) split by baseline pLDDT (low <75 vs high >=75), plus a per-protein scatter vs pLDDT.

    python scripts/make_confidence_figure.py [SHARDS_DIR] [OUT_DIR]
"""

from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CUT = 75.0
K = 16.0
CONCEPTS = ["helix", "coil", "helix_sae"]  # inducible/interesting; coil is the strongest lever
COLORS = {"helix": "#0072B2", "coil": "#D55E00", "helix_sae": "#56B4E9"}


def per_protein(shards_dir: Path, concept: str):
    rows = [json.loads(l) for l in (shards_dir / f"{concept}.jsonl").read_text().splitlines() if l.strip()]
    by: dict[str, dict] = {}
    for r in rows:
        if r.get("mult") == K and r.get("kind") in ("helix_add", "rand_add"):
            by.setdefault(r["protein"], {"plddt": r["base_plddt"]})[r["kind"]] = r["d_target"]
    return [(v["plddt"], v["helix_add"] - v["rand_add"]) for v in by.values()
            if "helix_add" in v and "rand_add" in v]


def ms(d):
    return (st.mean(d), st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else 0.0, len(d))


def pearson(xs, ys):
    n = len(xs); mx, my = st.mean(xs), st.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    shards = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "data" / "shard4_20260713"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else repo
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    data = {c: per_protein(shards, c) for c in CONCEPTS}
    fig, (axb, axs) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: grouped bars low vs high pLDDT
    rows_csv = [("concept", "bin", "mean_paired", "sem", "n")]
    x = range(len(CONCEPTS)); w = 0.38
    for j, (binname, lo_hi) in enumerate([("low pLDDT (<75)", True), ("high pLDDT (>=75)", False)]):
        means, sems = [], []
        for c in CONCEPTS:
            d = [e for p, e in data[c] if (p < CUT) == lo_hi]
            m, se, n = ms(d); means.append(m); sems.append(se)
            rows_csv.append((c, binname, f"{m:.4f}", f"{se:.4f}", n))
        axb.bar([i + (j - 0.5) * w for i in x], means, w, yerr=sems, capsize=4,
                label=binname, color=("#CC79A7" if lo_hi else "#999999"))
    axb.set_xticks(list(x)); axb.set_xticklabels(CONCEPTS)
    axb.axhline(0, color="k", lw=0.8)
    axb.set_ylabel("paired effect  (concept-add - random-add) on own DSSP state, k=16")
    axb.set_title("Steering success is larger in low-confidence structures\n(coil: +0.094 low vs +0.025 high; full 97 held-out)")
    axb.legend(frameon=False)

    # Panel B: scatter d_target-vs-random per protein vs pLDDT, with fit + Pearson r
    for c in CONCEPTS:
        ps = [p for p, e in data[c]]; es = [e for p, e in data[c]]
        r = pearson(ps, es)
        axs.scatter(ps, es, s=22, alpha=0.7, color=COLORS[c], label=f"{c} (r={r:+.2f})")
        # simple least-squares line
        mx, my = st.mean(ps), st.mean(es); b = (sum((p - mx) * (e - my) for p, e in zip(ps, es))
                                                / sum((p - mx) ** 2 for p in ps))
        xs = [min(ps), max(ps)]; axs.plot(xs, [my + b * (xx - mx) for xx in xs], color=COLORS[c], lw=1.2)
    axs.axhline(0, color="k", lw=0.8); axs.axvline(CUT, color="grey", ls="--", lw=0.8)
    axs.set_xlabel("baseline pLDDT"); axs.set_ylabel("paired steering effect (own DSSP state)")
    axs.set_title("Per-protein: effect vs confidence"); axs.legend(frameon=False)

    fig.tight_layout()
    p = out / "figures" / "fig9_confidence_by_pLDDT.png"
    fig.savefig(p, dpi=300)
    (out / "data" / "fig9_confidence_by_pLDDT.csv").write_text(
        "\n".join(",".join(map(str, r)) for r in rows_csv) + "\n", encoding="utf-8")
    print(f"wrote {p}")
    for c in CONCEPTS:
        lo = [e for pp, e in data[c] if pp < CUT]; hi = [e for pp, e in data[c] if pp >= CUT]
        print(f"  {c:10} low {ms(lo)[0]:+.3f} (n{len(lo)})  high {ms(hi)[0]:+.3f} (n{len(hi)})")


if __name__ == "__main__":
    main()
