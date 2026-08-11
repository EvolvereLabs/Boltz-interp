#!/usr/bin/env python3
"""Split the activation/probe pool into a train set (fit the steering direction) and a held-out set
(run the causal steering test), stratified by helix content so the held-out set still spans the
gradient. Held-out steering avoids the circularity of steering with a direction fit on those proteins.

The pool (common_activation_proteins.txt == the annotation TSV set) is the ONLY set with staged
activations + MSAs, so we split it rather than seek new proteins. Deterministic (seeded).

    python scripts/split_probe_proteins.py \
        --pool ../SwissProt_annotations/common_activation_proteins.txt \
        --annotations uniprot_annotations.tsv --frac 0.8 --seed 0

Writes probe_train_ids.txt + probe_heldout_ids.txt. Then:
  * fit on train:  export_probe_direction.py ... --train_ids probe_train_ids.txt
  * steer on heldout: make_protein_gradient.py --exclude probe_train_ids.txt   (draws from heldout)
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

BINS = (("poor", 0.00, 0.15), ("moderate", 0.15, 0.55), ("rich", 0.55, 1.01))


def helix_fraction(row: dict) -> float:
    h, length = row.get("Helix", ""), row.get("Length", "")
    if not h or not length:
        return 0.0
    res = sum(int(b) - int(a) + 1 for a, b in re.findall(r"HELIX (\d+)\.\.(\d+)", h))
    try:
        return res / int(length)
    except ValueError:
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified train/held-out split of the probe pool.")
    ap.add_argument("--pool", default="../SwissProt_annotations/common_activation_proteins.txt")
    ap.add_argument("--annotations", default="uniprot_annotations.tsv")
    ap.add_argument("--frac", type=float, default=0.8, help="training fraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_out", default="probe_train_ids.txt")
    ap.add_argument("--heldout_out", default="probe_heldout_ids.txt")
    args = ap.parse_args()

    pool = [p for p in Path(args.pool).read_text().split() if p]
    hf = {r["Entry"]: helix_fraction(r) for r in csv.DictReader(open(args.annotations, encoding="utf-8"), delimiter="\t")}
    rng = random.Random(args.seed)

    train: list[str] = []
    heldout: list[str] = []
    for name, lo, hi in BINS:
        band = sorted(p for p in pool if lo <= hf.get(p, 0.0) <= hi)
        rng.shuffle(band)
        cut = round(len(band) * args.frac)
        train += band[:cut]
        heldout += band[cut:]
        print(f"[split] {name:9s} [{lo:.2f}-{hi:.2f}]: {len(band)} -> {cut} train / {len(band) - cut} held-out")

    Path(args.train_out).write_text("\n".join(sorted(train)) + "\n", encoding="utf-8")
    Path(args.heldout_out).write_text("\n".join(sorted(heldout)) + "\n", encoding="utf-8")
    print(f"\n[split] {len(train)} train -> {args.train_out}")
    print(f"[split] {len(heldout)} held-out -> {args.heldout_out}")
    assert not (set(train) & set(heldout)), "train/heldout overlap!"
    print("[split] disjoint OK. Fit direction with --train_ids probe_train_ids.txt; "
          "build gradient from held-out with make_protein_gradient.py --exclude probe_train_ids.txt")


if __name__ == "__main__":
    main()
