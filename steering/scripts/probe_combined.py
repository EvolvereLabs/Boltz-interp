#!/usr/bin/env python3
"""E3 -- necessity via combined multi-site ablation, aggregated over proteins.

Ablate the helix direction at the trunk conditioning ALONE, the diffusion module ALONE, and BOTH at
once, and measure the DSSP helix change. Prediction (redundancy): single-site ablation is weak, both
together is strong -- necessity emerges only when a redundantly-encoded feature is removed everywhere.
Runs on any protein with a built input (no disulfide pairs needed), noise-matched and mean-centred.

    python scripts/probe_combined.py --from_survey outputs/select/helix_survey.json \
        --helix_min 0.35 --helix_max 0.90 --sampling_steps 50 --max_proteins 8
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from boltz_causal.boltz_runtime import (
    download_cache,
    load_boltz_model,
    make_datamodule,
    predict_to_cif,
    process_protein,
)
from boltz_causal.config import DIFFUSION_PROBE_LAYER, ExperimentConfig
from boltz_causal.directions import InterventionSpec, load_directions
from boltz_causal.hooks import register_interventions
from boltz_causal.structural_readout import ca_rmsd, helix_fraction, mean_plddt


def _spec(bank, where, target, layer):
    mean = bank.get_mean("helix", where)
    mean_t = None if mean is None else torch.as_tensor(mean, dtype=torch.float32)
    return InterventionSpec("helix", bank.get("helix", where), target, "ablate", 1.0, layer, "all", mean=mean_t)


def run_protein(model, pid, cfg, bank, ckpt, out_dir) -> list[dict]:
    processed = process_protein(
        Path(cfg.input_yaml_dir) / f"{pid}{cfg.input_suffix}", out_dir / "proc" / pid,
        ckpt.parent / "ccd.pkl", use_msa_server=cfg.use_msa_server,
    )
    datamodule = make_datamodule(processed, num_workers=cfg.num_workers)
    n_recycles = cfg.recycling_steps + 1

    def predict(tag: str) -> Path:
        torch.manual_seed(cfg.seed)  # noise-matched across baseline + every ablation
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        return predict_to_cif(model, datamodule, targets_dir=processed.targets_dir,
                              predictions_dir=out_dir / "predictions" / pid / tag,
                              accelerator=cfg.accelerator, devices=cfg.devices)

    trunk = _spec(bank, "trunk_L47", "trunk_output", None)
    diff = _spec(bank, "diffusion", "diffusion", DIFFUSION_PROBE_LAYER)
    combos = {"trunk_only": [trunk], "diffusion_only": [diff], "both": [trunk, diff]}

    base_cif = predict("baseline")
    base_helix = helix_fraction(base_cif)
    rows = [{"protein": pid, "site": "baseline", "helix": base_helix, "d_helix": 0.0,
             "ca_rmsd": 0.0, "plddt": mean_plddt(base_cif)}]
    for site, specs in combos.items():
        with register_interventions(model, [(s, None) for s in specs], n_recycles=n_recycles,
                                    last_recycle_only=cfg.trunk_depth_last_recycle_only):
            cif = predict(site)
        h = helix_fraction(cif)
        rows.append({"protein": pid, "site": site, "helix": h, "d_helix": h - base_helix,
                     "ca_rmsd": ca_rmsd(base_cif, cif), "plddt": mean_plddt(cif)})
        print(f"[E3] {pid:14s} {site:14s} d_helix={h - base_helix:+.3f} rmsd={rows[-1]['ca_rmsd']:.1f} plddt={rows[-1]['plddt']:.1f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="E3 combined multi-site helix ablation across proteins.")
    ap.add_argument("--proteins", default=None, help="comma-separated ids (overrides --from_survey)")
    ap.add_argument("--from_survey", default="outputs/select/helix_survey.json")
    ap.add_argument("--helix_min", type=float, default=0.35)  # ablation needs helix to remove
    ap.add_argument("--helix_max", type=float, default=0.95)
    ap.add_argument("--max_proteins", type=int, default=8)
    ap.add_argument("--input_dir", default="inputs")
    ap.add_argument("--directions", default=None)
    ap.add_argument("--output_dir", default="outputs/probe_combined")
    ap.add_argument("--sampling_steps", type=int, default=50)
    ap.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()

    cfg = ExperimentConfig()
    if args.directions:
        cfg.directions_npz = args.directions
    cfg.sampling_steps = args.sampling_steps
    cfg.accelerator = args.accelerator
    cfg.input_yaml_dir = args.input_dir
    torch.set_grad_enabled(False)

    bank = load_directions(cfg.directions_npz)
    try:
        bank.get("helix", "diffusion")
    except KeyError:
        raise SystemExit("E3 needs helix@diffusion in the directions .npz "
                         "(export_probe_direction.py --concepts helix --include_diffusion).")

    built = {p.stem for p in Path(args.input_dir).glob(f"*{cfg.input_suffix}")}
    if args.proteins:
        proteins = [p for p in args.proteins.split(",") if p]
    elif Path(args.from_survey).exists():
        survey = json.loads(Path(args.from_survey).read_text())
        proteins = sorted((p for p, d in survey.items()
                           if isinstance(d.get("helix"), (int, float)) and args.helix_min <= d["helix"] <= args.helix_max),
                          key=lambda p: -survey[p]["helix"])  # most helical first (most to ablate)
    else:
        proteins = sorted(built)
    proteins = [p for p in proteins if p in built][: args.max_proteins]
    if not proteins:
        raise SystemExit("No proteins to run (check --from_survey band / --input_dir).")
    print(f"[E3] {len(proteins)} proteins: {proteins}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(cfg.checkpoint_path).expanduser() if cfg.checkpoint_path else download_cache(Path(cfg.cache_dir))
    model = load_boltz_model(ckpt, recycling_steps=cfg.recycling_steps, sampling_steps=cfg.sampling_steps,
                             diffusion_samples=cfg.diffusion_samples, step_scale=cfg.step_scale)

    all_rows: list[dict] = []
    with open(out_dir / "combined.jsonl", "w", encoding="utf-8") as fh:
        for pid in proteins:
            try:
                rows = run_protein(model, pid, cfg, bank, ckpt, out_dir)
            except Exception as e:  # noqa: BLE001
                print(f"[E3] {pid}: FAILED ({str(e)[:100]})")
                rows = []
            for r in rows:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
            all_rows.extend(rows)

    print("\n" + "=" * 60)
    print(f"E3 COMBINED ABLATION -- {len({r['protein'] for r in all_rows})} proteins (mean d_helix)")
    print("=" * 60)
    for site in ("trunk_only", "diffusion_only", "both"):
        ds = [r["d_helix"] for r in all_rows if r["site"] == site]
        m = statistics.mean(ds) if ds else float("nan")
        s = statistics.stdev(ds) if len(ds) > 1 else 0.0
        print(f"  {site:14s} mean d_helix = {m:+.3f} +/- {s:.3f}  (n={len(ds)})")
    print("  redundancy result holds if 'both' >> trunk_only and diffusion_only.")


if __name__ == "__main__":
    main()
