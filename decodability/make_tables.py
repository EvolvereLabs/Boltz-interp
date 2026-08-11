#!/usr/bin/env python3
"""Emit the manuscript's decodability tables from the committed summary data.

Two tables, each written as both CSV (machine-checkable) and LaTeX (paste-ready):

  tab:headline                  depth-matched trunk (pf L47) vs diffusion (L22) probe-raw F1
  tab:dataset_label_prevalence  per-concept label prevalence for the three evaluation sets

Both were previously transcribed by hand out of the figure notebooks, so nothing
in the repo could confirm the printed numbers. Run from decodability/:

    python make_tables.py [--out-dir data/tables]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
PF_BENCH = DATA / "sae_explore/layer_sweep_rec1/benchmark"
DIFF_BENCH = DATA / "sae_explore/diffusion_analysis"
AA_RESULTS = DATA / "sae_explore/amino_acid_sanity/results.jsonl"
LABEL_STATS = DATA / "sae_explore/concept_label_stats.csv"

PF_REC, DIFF_REC = 1, 199
PF_LAYER, DIFF_LAYER = 47, 22

# Manuscript row order and display names for tab:headline.
HEADLINE_ROWS = [
    ("secondary_structure:coil", "Coil"),
    ("secondary_structure:helix", "Helix"),
    ("secondary_structure:strand", "Strand"),
    ("region:disordered", "Disordered"),
    ("signal_peptide", "Signal peptide"),
    ("disulfide_bond", "Disulfide bond"),
]

DATASET_LABELS = {
    "swissprot": "SwissProt",
    "secondary": "DSSP Secondary",
    "boltz_secondary": "Boltz Secondary",
}


def _flatten(path: str, stack: str) -> list[dict]:
    d = json.load(open(path))
    return [
        dict(stack=stack, rec=d["rec"], layer=d["layer"], concept=c,
             probe_raw_f1=v.get("probe_raw_f1"), sae_f1=v.get("sae_f1_mean"))
        for c, v in d["per_concept"].items()
    ]


def load_benchmark() -> pd.DataFrame:
    rows: list[dict] = []
    for p in glob.glob(str(PF_BENCH / "layer*/[sw]*_agg.json")):
        rows += _flatten(p, "pairformer")
    for p in glob.glob(str(DIFF_BENCH / "rec*/benchmark/layer*/[sw]*_agg.json")):
        rows += _flatten(p, "diffusion")
    if not rows:
        raise SystemExit(f"No *_agg.json found under {PF_BENCH} or {DIFF_BENCH}.")
    return pd.DataFrame(rows).drop_duplicates(["stack", "rec", "layer", "concept"])


def headline_table() -> pd.DataFrame:
    df = load_benchmark()
    # NB: df["stack"], not df.stack — the latter is DataFrame.stack, the reshape method.
    pf = df[(df["stack"] == "pairformer") & (df.rec == PF_REC) & (df.layer == PF_LAYER)]
    di = df[(df["stack"] == "diffusion") & (df.rec == DIFF_REC) & (df.layer == DIFF_LAYER)]
    pf_f1 = pf.set_index("concept")["probe_raw_f1"]
    di_f1 = di.set_index("concept")["probe_raw_f1"]

    rows = [dict(concept=label, trunk=pf_f1.get(key), diffusion=di_f1.get(key))
            for key, label in HEADLINE_ROWS]

    # Amino-acid identity is scored per residue by amino_acid_sanity_check.py, not by
    # the concept benchmark, so it is averaged over the 20 residues here (same as Fig 3A).
    aa = pd.DataFrame([json.loads(line) for line in AA_RESULTS.read_text().splitlines() if line.strip()])
    for col, label in [("probe_f1", "AA identity (probe)"), ("sae_f1", "AA identity (SAE-1feat)")]:
        rows.append(dict(
            concept=label,
            trunk=aa[(aa.layer_type == "pairformer") & (aa.layer == PF_LAYER)][col].mean(),
            diffusion=aa[(aa.layer_type == "diffusion") & (aa.layer == DIFF_LAYER)][col].mean(),
        ))

    out = pd.DataFrame(rows)
    out["delta"] = out["diffusion"] - out["trunk"]
    return out


def headline_latex(t: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[H]\centering",
        r"  \caption{Depth-matched module outputs (probe-raw F1 unless noted). Geometry",
        r"  transfers; sequence chemistry is reduced.}",
        r"  \label{tab:headline}",
        r"  \small",
        r"  \begin{tabular}{lccc}",
        r"    \toprule",
        r"    Concept & Trunk (pf L47) & Diffusion (L22) & $\Delta$ \\",
        r"    \midrule",
    ]
    for r in t.itertuples():
        # Anything that rounds to zero prints as a positive-signed 0.00, matching the
        # manuscript; only a genuinely negative rounded delta gets the minus sign.
        if round(r.delta, 2) < 0:
            d = f"$-{abs(r.delta):.2f}$"
        else:
            d = rf"$\,${abs(r.delta):.2f}"
        lines.append(f"    {r.concept:<23} & {r.trunk:.2f} & {r.diffusion:.2f} & {d} \\\\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def prevalence_table() -> pd.DataFrame:
    if not LABEL_STATS.exists():
        raise SystemExit(f"Missing {LABEL_STATS}; regenerate with count_concept_label_stats.py.")
    df = pd.read_csv(LABEL_STATS)
    df["dataset"] = df["concept_set"].map(DATASET_LABELS).fillna(df["concept_set"])
    cols = ["dataset", "concept", "pos_residues", "pos_domains",
            "n_proteins_with_concept", "set_n_proteins", "set_n_residues"]
    return df[cols].sort_values(["dataset", "concept"]).reset_index(drop=True)


def prevalence_latex(t: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[ht]\centering",
        r"\small",
        r"\caption{Dataset size and per-concept label prevalence for the evaluation sets.}",
        r"\label{tab:dataset_label_prevalence}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Concept} & \textbf{Positive Residues} & "
        r"\textbf{Positive Domains} & \textbf{Proteins w/ Concept} & "
        r"\textbf{Total Proteins} & \textbf{Total Residues} \\",
        r"\midrule",
    ]
    prev = None
    for r in t.itertuples():
        if prev is not None and r.dataset != prev:
            lines.append(r"\midrule")
        ds = r.dataset if r.dataset != prev else ""
        concept = r.concept.replace("_", " ").replace("&", r"\&").title()
        lines.append(
            f"{ds} & {concept} & {r.pos_residues:,} & {r.pos_domains:,} & "
            f"{r.n_proteins_with_concept} & {r.set_n_proteins} & {r.set_n_residues:,} \\\\"
        )
        prev = r.dataset
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DATA / "tables"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    head = headline_table()
    head.round(3).to_csv(out / "tab_headline.csv", index=False)
    (out / "tab_headline.tex").write_text(headline_latex(head), encoding="utf-8")

    prev = prevalence_table()
    prev.to_csv(out / "tab_dataset_label_prevalence.csv", index=False)
    (out / "tab_dataset_label_prevalence.tex").write_text(prevalence_latex(prev), encoding="utf-8")

    print("== tab:headline (probe-raw F1) ==")
    print(head.round(3).to_string(index=False))
    print(f"\n== tab:dataset_label_prevalence: {len(prev)} concepts over "
          f"{prev.dataset.nunique()} datasets ==")
    print(f"\nWrote 4 files to {out}")


if __name__ == "__main__":
    main()
