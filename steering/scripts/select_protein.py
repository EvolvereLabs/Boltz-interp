#!/usr/bin/env python3
"""Survey baseline helix content to pick the right protein(s) for the steering test.

The steering readout is dense DSSP helix fraction on the Boltz *baseline* prediction -- SwissProt
HELIX annotations under-report (the paper's own point, Sec 3.5) and cover only a fraction of the set,
so the authoritative signal is a real baseline pass + DSSP. This script runs a fast baseline-only
prediction (no interventions) for every built input, measures helix fraction + mean pLDDT, caches
the result, and recommends proteins for each steering role:

  * moderate helix (0.30-0.60): BEST for a bidirectional demo -- headroom to push helix both down
    (ablate) and up (add) on the SAME protein, the cleanest single figure.
  * helix-rich (>= 0.55): for the negative-steering (ablate -> helix down) arm.
  * helix-poor (<= 0.15): for the positive-steering (add -> helix up) arm.

    python scripts/select_protein.py --input_dir inputs --sampling_steps 30

Results cache to outputs/helix_survey.json; re-runs skip proteins already measured (use --force to
recompute). Geometry converges early in diffusion (paper Sec 3.3), so a low --sampling_steps is fine
for ranking -- but re-confirm the chosen protein's baseline at the probe's real step count.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import torch

from boltz_causal.boltz_runtime import (
    download_cache,
    load_boltz_model,
    make_datamodule,
    predict_to_cif,
    process_protein,
)
from boltz_causal.config import ExperimentConfig
from boltz_causal.structural_readout import helix_fraction, mean_plddt


def swissprot_helix(annotations: Path) -> dict[str, float]:
    """Crude helix fraction from SwissProt HELIX feature spans / length (a hint, not ground truth)."""
    out: dict[str, float] = {}
    if not annotations.exists():
        return out
    with open(annotations, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            h, length = row.get("Helix", ""), row.get("Length", "")
            if not h or not length:
                continue
            res = sum(int(b) - int(a) + 1 for a, b in re.findall(r"HELIX (\d+)\.\.(\d+)", h))
            out[row["Entry"]] = res / int(length)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Survey baseline helix content to pick a steering protein.")
    ap.add_argument("--input_dir", default="inputs")
    ap.add_argument("--annotations", default="uniprot_annotations.tsv", help="SwissProt TSV (helix hint)")
    ap.add_argument("--directions", default=None)
    ap.add_argument("--output_dir", default="outputs/select")
    ap.add_argument("--sampling_steps", type=int, default=30, help="low is fine for ranking")
    ap.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--limit", type=int, default=0, help="0 = all built inputs")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    args = ap.parse_args()

    cfg = ExperimentConfig()
    if args.directions:
        cfg.directions_npz = args.directions
    cfg.sampling_steps = args.sampling_steps
    cfg.accelerator = args.accelerator

    torch.set_grad_enabled(False)
    torch.manual_seed(cfg.seed)

    proteins = sorted(p.stem for p in Path(args.input_dir).glob(f"*{cfg.input_suffix}"))
    if args.limit:
        proteins = proteins[: args.limit]
    if not proteins:
        raise SystemExit(f"No inputs in {args.input_dir}/ (run build_inputs.py first).")
    swiss = swissprot_helix(Path(args.annotations))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "helix_survey.json"
    survey: dict[str, dict] = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    todo = [p for p in proteins if args.force or p not in survey]
    print(f"[select] {len(proteins)} built inputs; {len(todo)} to measure "
          f"({len(proteins) - len(todo)} cached)  sampling_steps={cfg.sampling_steps}")

    if todo:
        ckpt = Path(cfg.checkpoint_path).expanduser() if cfg.checkpoint_path else download_cache(Path(cfg.cache_dir))
        model = load_boltz_model(
            ckpt, recycling_steps=cfg.recycling_steps, sampling_steps=cfg.sampling_steps,
            diffusion_samples=cfg.diffusion_samples, step_scale=cfg.step_scale,
        )
        for i, pid in enumerate(todo, 1):
            try:
                processed = process_protein(
                    Path(args.input_dir) / f"{pid}{cfg.input_suffix}", out_dir / "proc" / pid,
                    ckpt.parent / "ccd.pkl", use_msa_server=cfg.use_msa_server,
                )
                datamodule = make_datamodule(processed, num_workers=cfg.num_workers)
                cif = predict_to_cif(
                    model, datamodule, targets_dir=processed.targets_dir,
                    predictions_dir=out_dir / "predictions" / pid, accelerator=cfg.accelerator, devices=cfg.devices,
                )
                survey[pid] = {"helix": helix_fraction(cif), "plddt": mean_plddt(cif),
                               "swiss_helix": swiss.get(pid)}
            except Exception as e:  # noqa: BLE001  # one bad protein shouldn't sink the survey
                survey[pid] = {"helix": None, "plddt": None, "swiss_helix": swiss.get(pid), "error": str(e)[:200]}
                print(f"[select] {pid}: FAILED ({str(e)[:80]})")
            cache_path.write_text(json.dumps(survey, indent=2))  # checkpoint each protein
            h = survey[pid].get("helix")
            print(f"[select] ({i}/{len(todo)}) {pid}: helix={h if h is None else round(h,3)}  "
                  f"plddt={survey[pid].get('plddt')}")

    # rank + recommend
    ok = {p: d for p, d in survey.items() if isinstance(d.get("helix"), (int, float))}
    ranked = sorted(ok.items(), key=lambda kv: kv[1]["helix"], reverse=True)
    print("\n" + "=" * 68)
    print(f"HELIX SURVEY  ({len(ok)} measured, dense DSSP on baseline)")
    print("=" * 68)
    print(f"  {'protein':14s} {'DSSP helix':>10s} {'pLDDT':>7s} {'SwissProt':>10s}")
    for pid, d in ranked:
        sw = d.get("swiss_helix")
        print(f"  {pid:14s} {d['helix']:>10.3f} {(d['plddt'] or float('nan')):>7.1f} "
              f"{('-' if sw is None else f'{sw:.3f}'):>10s}")

    def pick(lo, hi, target):
        band = [(p, d) for p, d in ranked if lo <= d["helix"] <= hi]
        return min(band, key=lambda kv: abs(kv[1]["helix"] - target), default=(None, None))

    mod_p, mod_d = pick(0.30, 0.60, 0.45)   # bidirectional
    rich_p, rich_d = pick(0.55, 1.01, 0.70)  # negative steering
    poor_p, poor_d = pick(0.00, 0.15, 0.07)  # positive steering
    print("\nRECOMMENDATIONS")
    if mod_p:
        print(f"  BIDIRECTIONAL (primary): {mod_p}  (helix={mod_d['helix']:.3f}) -- ablate down AND add up")
        print(f"    python scripts/probe_batch.py --proteins {mod_p} --input_dir {args.input_dir}")
    else:
        print("  BIDIRECTIONAL: none in 0.30-0.60; use the rich+poor pair below instead")
    if rich_p:
        print(f"  NEGATIVE steering (ablate): {rich_p}  (helix={rich_d['helix']:.3f})")
    if poor_p:
        print(f"  POSITIVE steering (add):    {poor_p}  (helix={poor_d['helix']:.3f})")
    print(f"\n[select] cache: {cache_path}")


if __name__ == "__main__":
    main()
