#!/usr/bin/env python3
"""Held-out validity check for the steering directions (reviewer point 4 + precision question).

For each secondary-structure concept, evaluate the EXACT probe direction stored in the steering npz
on HELD-OUT proteins: score = (x - mean) . u, sweep thresholds, and report held-out F1, precision,
and recall. Answers (a) did we use the right (high-F1) vector -- compare to the paper's probe-raw F1
~0.79-0.90; and (b) is the direction high-recall/low-precision (which would argue for steering a
precision-optimal direction instead).

Run locally in the SwissProt venv where downloads_layer47/ is staged:
    python validate_directions_f1.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from export_probe_direction import load_activations_and_labels

NPZ = "../boltz_causal_intervention/directions/disulfide_helix_directions.npz"
HELDOUT = "../boltz_causal_intervention/probe_heldout_ids.txt"
PCTS = (50, 75, 90, 95, 99)  # same threshold grid as the benchmark


def prf(score: np.ndarray, y: np.ndarray, t: float):
    pred = score >= t
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return f, p, r


def main() -> None:
    d = np.load(NPZ, allow_pickle=True)
    heldout = set(Path(HELDOUT).read_text().split())
    print(f"held-out proteins: {len(heldout)}")
    print(f"{'concept':8} {'heldoutF1':>9} {'precision':>9} {'recall':>7} {'best_pct':>8} {'pos_frac':>8}")
    for concept in ("helix", "strand", "coil"):
        x, y = load_activations_and_labels(concept, "trunk_L47", protein_subset=heldout)
        y = np.asarray(y).astype(int)
        u = d[f"{concept}@trunk_L47"].ravel(); u = u / np.linalg.norm(u)
        mean = d[f"{concept}@trunk_L47.mean"].ravel()
        score = (x - mean) @ u
        best = max(((prf(score, y, np.percentile(score, pc)), pc) for pc in PCTS), key=lambda z: z[0][0])
        (f, p, r), pc = best
        print(f"  {concept:6} {f:>9.3f} {p:>9.3f} {r:>7.3f} {pc:>8} {y.mean():>8.3f}")


if __name__ == "__main__":
    main()
