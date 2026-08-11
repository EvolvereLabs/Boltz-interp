#!/usr/bin/env python3
"""Paired-protein bootstrap 95% CIs for the steering effect, per concept (reviewer point 3).

Effect = per-protein paired (concept-add - random-add) on the concept's own DSSP state at dose k=16.
Bootstrap resamples proteins with replacement (B=10000) to get a distribution-free 95% CI. Writes
`data/fig10_bootstrap_ci.csv`, the source of the 95% CIs in the manuscript's steering table. CSV
only -- no figure. Pure analysis on the shard JSONLs -- no GPU.

    python scripts/make_bootstrap_ci.py [SHARDS_DIR] [OUT_DIR]
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

K = 16.0
B = 10000
SEED = 0
CONCEPTS = ["coil", "helix", "helix_sae", "strand_sae", "coil_sae", "strand"]
# Own-state DSSP field per concept. NOTE: stored `d_target` is BUGGED for the SAE shards
# (defaulted to d_helix for strand_sae/coil_sae), so we NEVER read it — we take the correct
# per-SS delta below. Verified: d_target == d_helix in 1164/1164 rows for all 3 SAE shards.
OWN_FIELD = {
    "helix": "d_helix", "helix_sae": "d_helix",
    "strand": "d_strand", "strand_sae": "d_strand",
    "coil": "d_coil", "coil_sae": "d_coil",
}


def paired_vals(shards: Path, concept: str) -> list[float]:
    field = OWN_FIELD[concept]
    rows = [json.loads(l) for l in (shards / f"{concept}.jsonl").read_text().splitlines() if l.strip()]
    by: dict[str, dict] = {}
    for r in rows:
        if r.get("mult") == K and r.get("kind") in ("helix_add", "rand_add"):
            by.setdefault(r["protein"], {})[r["kind"]] = r[field]
    return [v["helix_add"] - v["rand_add"] for v in by.values() if "helix_add" in v and "rand_add" in v]


def boot_ci(vals: list[float], b: int = B):
    rng = random.Random(SEED)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    return sum(vals) / n, means[int(0.025 * b)], means[int(0.975 * b)]


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    shards = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "data" / "shard4_20260713"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else repo
    (out / "data").mkdir(parents=True, exist_ok=True)

    res = []
    for c in CONCEPTS:
        v = paired_vals(shards, c)
        m, lo, hi = boot_ci(v)
        res.append((c, m, lo, hi, len(v)))

    csv = ["concept,mean,ci_lo,ci_hi,n,excludes_zero"]
    for c, m, lo, hi, n in res:
        csv.append(f"{c},{m:.4f},{lo:.4f},{hi:.4f},{n},{lo > 0 or hi < 0}")
    p = out / "data" / "fig10_bootstrap_ci.csv"
    p.write_text("\n".join(csv) + "\n", encoding="utf-8")
    print(f"wrote {p}")
    for c, m, lo, hi, n in res:
        print(f"  {c:11} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  n={n}")


if __name__ == "__main__":
    main()
