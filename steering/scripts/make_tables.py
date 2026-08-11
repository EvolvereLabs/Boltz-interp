#!/usr/bin/env python3
"""Emit the manuscript's steering table (tab:steering) from the committed CSVs.

The paired effect and its 95% bootstrap CI come from make_bootstrap_ci.py
(fig10_bootstrap_ci.csv); the z-score against the 19-direction matched-norm
random null comes from analyze_null.py (null_distribution_full97.csv). Joining
them by hand was the last un-generated number in the paper.

Writes CSV (machine-checkable) and LaTeX (paste-ready). Run from steering/:

    python scripts/make_tables.py [DATA_DIR] [OUT_DIR]

Defaults: DATA_DIR = steering/data, OUT_DIR = steering/data/tables
"""
from __future__ import annotations

import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# shard name -> (concept, direction) as printed in the manuscript
SHARDS = {
    "coil": ("Coil", "probe"),
    "helix": ("Helix", "probe"),
    "helix_sae": ("Helix", "SAE-1feat"),
    "strand": ("Strand", "probe"),
    "strand_sae": ("Strand", "SAE-1feat"),
    "coil_sae": ("Coil", "SAE-1feat"),
}
# Manuscript row order: steering directions first, ranked by effect, then the nulls.
ROW_ORDER = ["coil", "helix", "helix_sae", "strand", "strand_sae", "coil_sae"]


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Regenerate it with the steering analysis scripts.")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verdict(mean: float, z: float, ci_lo: float, ci_hi: float) -> str:
    if ci_lo > 0 and z > 2:
        return "steers"
    return "null"


def build(data_dir: str) -> list[dict]:
    ci = {r["concept"]: r for r in read_csv(os.path.join(data_dir, "fig10_bootstrap_ci.csv"))}
    null = {r["shard"]: r for r in read_csv(os.path.join(data_dir, "null_distribution_full97.csv"))}

    rows = []
    for shard in ROW_ORDER:
        if shard not in ci or shard not in null:
            print(f"  WARNING: no data for shard {shard}; skipping")
            continue
        concept, direction = SHARDS[shard]
        mean = float(ci[shard]["mean"])
        lo, hi = float(ci[shard]["ci_lo"]), float(ci[shard]["ci_hi"])
        z = float(null[shard]["z_score"])
        rows.append(dict(concept=concept, direction=direction, effect=mean,
                         ci_lo=lo, ci_hi=hi, z=z, n=int(ci[shard]["n"]),
                         verdict=verdict(mean, z, lo, hi)))

    rows.sort(key=lambda r: -r["effect"])
    if rows:
        rows[0]["verdict"] = "steers (strongest)" if rows[0]["verdict"] == "steers" else rows[0]["verdict"]
    return rows


def latex(rows: list[dict]) -> str:
    out = [
        r"\begin{table}[H]\centering",
        r"  \caption{Paired steering effects on the concept's own DSSP state (held-out,",
        r"  dose $k{=}16$, $n{=}97$ proteins). Effect $=$ per-protein",
        r"  (concept-add $-$ random-add) with 95\% paired-protein bootstrap CI",
        r"  ($B{=}10{,}000$); $z$ is the population effect against a 19-direction",
        r"  matched-norm random null (Fig.~\ref{fig:steer_null}). Mean pLDDT stays stable",
        r"  ($\sim$79) across conditions.}",
        r"  \label{tab:steering}",
        r"  \small",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    Concept & Direction & Paired effect (95\% CI) & $z$ vs.\ null & Verdict \\",
        r"    \midrule",
    ]
    for r in rows:
        eff = f"${r['effect']:+.3f}\\ [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]$"
        out.append(f"    {r['concept']:<6} & {r['direction']:<9} & {eff} & "
                   f"${r['z']:+.1f}$ & {r['verdict']} \\\\")
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def main() -> None:
    data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "data")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "data", "tables")
    os.makedirs(out_dir, exist_ok=True)

    rows = build(data_dir)

    csv_path = os.path.join(out_dir, "tab_steering.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tex_path = os.path.join(out_dir, "tab_steering.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex(rows))

    print("== tab:steering ==")
    print(f"{'Concept':<8}{'Direction':<11}{'Effect':>9}  {'95% CI':<20}{'z':>7}  Verdict")
    for r in rows:
        ci = f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]"
        print(f"{r['concept']:<8}{r['direction']:<11}{r['effect']:>+9.3f}  {ci:<20}"
              f"{r['z']:>+7.1f}  {r['verdict']}")
    print(f"\nWrote {csv_path}\n      {tex_path}")


if __name__ == "__main__":
    main()
