#!/usr/bin/env python3
"""Build a helix-gradient protein list for steering-generalization runs (E1/E2).

Selects proteins spanning low/moderate/high helix content from the SwissProt annotation TSV so the
batch steering runs cover a range rather than one point. Two important caveats baked in:

  * SwissProt HELIX annotations UNDER-REPORT (Paper A Sec 3.5), so this is a *prefilter* -- the true
    per-protein helix fraction comes from scripts/select_protein.py (dense DSSP on the baseline). Bins
    here are deliberately wide; the survey refines them.
  * HELD-OUT: pass --exclude with the IDs used to FIT the probes, so the causal test is not run on
    probe-training proteins (avoids the circularity a reviewer will flag). Without it, the list is not
    guaranteed held-out and the script says so.

    python scripts/make_protein_gradient.py --annotations uniprot_annotations.tsv \
        --exclude probe_train_ids.txt --per_bin 8 --max_len 400 --out proteins_gradient.txt

Then build + survey the result:
    python scripts/build_inputs.py --proteins proteins_gradient.txt --out-dir inputs
    python scripts/select_protein.py --input_dir inputs --sampling_steps 30
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

# (name, lo, hi) helix-fraction bins on the (under-reporting) SwissProt annotation
BINS = (("poor", 0.00, 0.15), ("moderate", 0.20, 0.55), ("rich", 0.55, 1.01))


def helix_fraction(row: dict) -> float | None:
    h, length = row.get("Helix", ""), row.get("Length", "")
    if not h or not length:
        return None
    res = sum(int(b) - int(a) + 1 for a, b in re.findall(r"HELIX (\d+)\.\.(\d+)", h))
    try:
        return res / int(length)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Select a helix-gradient protein list from the annotation TSV.")
    ap.add_argument("--annotations", default="uniprot_annotations.tsv")
    ap.add_argument("--exclude", default=None, help="file of IDs to drop (e.g. probe-training set) for held-out")
    ap.add_argument("--per_bin", type=int, default=8, help="proteins per helix bin")
    ap.add_argument("--min_len", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=400, help="cap for Boltz runtime")
    ap.add_argument("--out", default="proteins_gradient.txt")
    args = ap.parse_args()

    exclude: set[str] = set()
    if args.exclude and Path(args.exclude).exists():
        exclude = set(Path(args.exclude).read_text().split())

    rows = list(csv.DictReader(open(args.annotations, encoding="utf-8"), delimiter="\t"))
    cand: list[tuple[str, float, int]] = []
    for r in rows:
        hf, length = helix_fraction(r), int(r["Length"]) if r.get("Length") else 0
        if hf is None or not (args.min_len <= length <= args.max_len) or r["Entry"] in exclude:
            continue
        cand.append((r["Entry"], hf, length))

    chosen: list[tuple[str, float, int]] = []
    print(f"[gradient] {len(cand)} candidates (len {args.min_len}-{args.max_len}, "
          f"{len(exclude)} excluded)")
    for name, lo, hi in BINS:
        band = sorted((c for c in cand if lo <= c[1] <= hi), key=lambda c: c[1])
        # spread evenly across the band rather than clustering at one end
        pick = band if len(band) <= args.per_bin else [band[round(i * (len(band) - 1) / (args.per_bin - 1))]
                                                        for i in range(args.per_bin)]
        chosen.extend(pick)
        print(f"[gradient] {name:9s} [{lo:.2f}-{hi:.2f}]: {len(band)} available -> picked {len(pick)}")

    # de-dup (bins can overlap at edges), keep order
    seen: set[str] = set()
    uniq = [c for c in chosen if not (c[0] in seen or seen.add(c[0]))]
    ids = [c[0] for c in uniq]
    Path(args.out).write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"\n[gradient] wrote {len(ids)} ids -> {args.out}")
    for pid, hf, length in sorted(uniq, key=lambda c: c[1]):
        print(f"    {pid:14s} swiss_helix={hf:.3f}  len={length}")
    if not exclude:
        print("\n[gradient] WARNING: no --exclude given -> NOT guaranteed held-out from probe fitting. "
              "Pass your probe-training IDs to --exclude before using these for the causal claim.")
    print("[gradient] NOTE: buildability depends on YAML+MSA existing under the S3 "
          "SwissProtAnnotation5 prefix; build_inputs.py will skip any that are missing.")


if __name__ == "__main__":
    main()
