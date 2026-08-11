#!/usr/bin/env python3
"""Replicate the helix-steering battery across many proteins and aggregate the specificity curve.

Runs the noise-matched, mean-centred steering battery over a set of proteins and aggregates the
helix-direction vs random-direction dose-response. Because each protein has its own activation scale,
the additive strength is calibrated per protein (alpha = k * mean|(x-mean).u|) and results are
aggregated by the MULTIPLIER k -- a protein-scale-invariant x-axis -- not absolute alpha.

Output: the paper figure's data -- per multiplier, mean +/- std helix change for the helix direction
and for a matched-norm random direction, plus the dose where pLDDT starts to drop (off-manifold edge).

    # pick proteins spanning a helix gradient first (writes outputs/select/helix_survey.json):
    python scripts/select_protein.py --input_dir inputs --sampling_steps 30
    # then aggregate the steering battery over them:
    python scripts/probe_batch.py --from_survey outputs/select/helix_survey.json \
        --helix_min 0.25 --helix_max 0.85 --site trunk --sampling_steps 50
"""

from __future__ import annotations

import argparse
import hashlib
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
from boltz_causal.hooks import register_intervention
from boltz_causal.steering import capture_scale as _capture_scale
from boltz_causal.structural_readout import ca_rmsd, mean_plddt, ss_fractions

# steered direction key -> the DSSP state it should move (readout follows the concept). MUST cover the
# _sae/_f1set variants or d_target silently defaults to helix (bug: strand_sae/coil_sae read helix).
_TARGET_SS = {"helix": "helix", "helix_sae": "helix", "helix_sae_f1set": "helix",
              "strand": "strand", "strand_sae": "strand", "strand_sae_f1set": "strand",
              "coil": "coil", "coil_sae": "coil", "coil_sae_f1set": "coil"}

MULTIPLIERS = (1.0, 2.0, 4.0, 8.0, 16.0)  # add alpha = k * per-protein projection scale


def run_protein(model, pid, cfg, bank, ckpt, where, target, layer, out_dir, rand_seed, concept="helix",
                n_random=1) -> list[dict]:
    """Baseline + concept/random addition sweep + a=1 ablations for one protein (noise-matched).
    ``concept`` is the direction key (e.g. 'helix', 'helix_sae', 'strand'); readout stays DSSP helix.
    ``n_random`` > 1 adds (n_random-1) EXTRA matched-norm random directions at the top dose (kind
    'rand_null'), giving a per-protein RANDOM NULL DISTRIBUTION to compare the concept effect against
    (reviewer: a random-direction distribution, not a single control)."""
    helix_v = bank.get(concept, where)
    mean = bank.get_mean(concept, where)
    mean_t = None if mean is None else torch.as_tensor(mean, dtype=torch.float32)
    # PINNED random seed: derive from the protein ID (stable hash), NOT the enumeration index, so the
    # matched-norm random control(s) are identical across runs regardless of protein order/subset.
    base_seed = int(hashlib.sha1(pid.encode()).hexdigest()[:8], 16)
    rand_v = bank.random_like(concept, where, seed=base_seed)

    processed = process_protein(
        Path(cfg.input_yaml_dir) / f"{pid}{cfg.input_suffix}", out_dir / "proc" / pid,
        ckpt.parent / "ccd.pkl", use_msa_server=cfg.use_msa_server,
    )
    datamodule = make_datamodule(processed, num_workers=cfg.num_workers)
    n_recycles = cfg.recycling_steps + 1

    def predict(tag: str) -> Path:
        torch.manual_seed(cfg.seed)  # identical diffusion noise for baseline and every condition
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        return predict_to_cif(
            model, datamodule, targets_dir=processed.targets_dir,
            predictions_dir=out_dir / "predictions" / pid / tag,
            accelerator=cfg.accelerator, devices=cfg.devices,
        )

    cap = _capture_scale(model, torch.as_tensor(helix_v, dtype=torch.float32), args_site(target), mean_t)
    base_cif = predict("baseline")
    cap["_restore"]()
    base_ss = ss_fractions(base_cif)            # {'helix','strand','coil'} in one DSSP pass
    base_plddt = mean_plddt(base_cif)
    target_ss = _TARGET_SS.get(concept, "helix")  # which SS this concept should move
    proj = cap.get("proj", float("nan"))
    if not (proj == proj) or proj <= 0:
        print(f"[batch] {pid}: bad projection scale ({proj}); skipping")
        return []

    def spec(concept_key, vec, mode, a):
        return InterventionSpec(concept_key, vec, target, mode, a, layer, "all", mean=mean_t)

    def measure(tag, s):
        with register_intervention(model, s, None, n_recycles=n_recycles,
                                   last_recycle_only=cfg.trunk_depth_last_recycle_only):
            cif = predict(tag)
        ss = ss_fractions(cif)
        row = {"protein": pid, "condition": tag, "base_helix": base_ss["helix"],
               "ca_rmsd": ca_rmsd(base_cif, cif), "plddt": mean_plddt(cif), "base_plddt": base_plddt,
               "target_ss": target_ss}
        for k in ("helix", "strand", "coil"):     # full confusion-matrix readout each condition
            row[k] = ss[k]
            row[f"d_{k}"] = ss[k] - base_ss[k]
        row["d_target"] = row[f"d_{target_ss}"]    # the steered concept's own SS change
        return row

    rows: list[dict] = [{"protein": pid, "condition": "baseline", "base_helix": base_ss["helix"],
                         "helix": base_ss["helix"], "strand": base_ss["strand"], "coil": base_ss["coil"],
                         "d_helix": 0.0, "d_strand": 0.0, "d_coil": 0.0, "d_target": 0.0,
                         "ca_rmsd": 0.0, "plddt": base_plddt, "base_plddt": base_plddt, "target_ss": target_ss}]
    for k in MULTIPLIERS:
        a = k * proj
        rows.append({**measure(f"helix_add_k{k:g}", spec("helix", helix_v, "add", a)), "kind": "helix_add", "mult": k})
        rows.append({**measure(f"rand_add_k{k:g}", spec("random", rand_v, "add", a)), "kind": "rand_add", "mult": k})
    rows.append({**measure("helix_ablate_a1", spec("helix", helix_v, "ablate", 1.0)), "kind": "helix_ablate", "mult": 1.0})
    rows.append({**measure("rand_ablate_a1", spec("random", rand_v, "ablate", 1.0)), "kind": "rand_ablate", "mult": 1.0})
    # random NULL DISTRIBUTION at the top dose: extra matched-norm random directions so the concept
    # effect can be scored against a per-protein spread of random effects (not one control draw).
    top_k = MULTIPLIERS[-1]; top_a = top_k * proj
    for j in range(1, n_random):
        rv = bank.random_like(concept, where, seed=base_seed + j)  # pinned per (protein, j)
        rows.append({**measure(f"randnull{j}_k{top_k:g}", spec("random", rv, "add", top_a)),
                     "kind": "rand_null", "mult": top_k, "rand_idx": j})
    for r in rows:
        print(f"[batch] {pid:14s} {r['condition']:18s} d_{target_ss}={r.get('d_target', 0.0):+.3f} "
              f"(H{r['d_helix']:+.2f} E{r['d_strand']:+.2f} C{r['d_coil']:+.2f}) "
              f"rmsd={r['ca_rmsd']:.1f} plddt={r['plddt']:.1f}")
    return rows


def args_site(target: str) -> str:
    return "diffusion" if target == "diffusion" else "trunk"


def _agg(vals: list[float]) -> str:
    vals = [v for v in vals if isinstance(v, (int, float)) and v == v]
    if not vals:
        return "  n/a"
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:+.3f}+/-{s:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate the helix-steering battery across proteins.")
    ap.add_argument("--proteins", default=None, help="comma-separated ids (overrides --from_survey)")
    ap.add_argument("--from_survey", default="outputs/select/helix_survey.json",
                    help="helix survey from select_protein.py; proteins filtered by --helix_min/max")
    ap.add_argument("--helix_min", type=float, default=0.25)
    ap.add_argument("--helix_max", type=float, default=0.85)
    ap.add_argument("--max_proteins", type=int, default=8)
    ap.add_argument("--input_dir", default="inputs")
    ap.add_argument("--site", default="trunk", choices=["trunk", "diffusion"])
    ap.add_argument("--concept", default="helix", help="direction key to steer (helix | helix_sae | strand)")
    ap.add_argument("--n_random", type=int, default=1,
                    help="matched-norm random directions for a NULL DISTRIBUTION (e.g. 20); >1 adds "
                         "(n_random-1) extra randoms at the top dose as kind='rand_null'")
    ap.add_argument("--diffusion_multilayer", action="store_true",
                    help="inject at ALL diffusion layers (multi-steering), not just the probe layer -- "
                         "single-layer diffusion steering is underpowered (a null there is likely a "
                         "method artifact, per DiT steering work)")
    ap.add_argument("--directions", default=None)
    ap.add_argument("--output_dir", default="outputs/probe_batch")
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
    where, target, layer = ("diffusion", "diffusion", DIFFUSION_PROBE_LAYER) if args.site == "diffusion" \
        else ("trunk_L47", "trunk_output", None)
    if args.site == "diffusion" and args.diffusion_multilayer:
        layer = None  # inject at ALL diffusion layers (multi-steering), every sampling step
    bank.get(args.concept, where)  # fail fast if the direction key is missing
    if bank.get_mean(args.concept, where) is None:
        print(f"[batch] WARNING: no exported mean for {args.concept}@{where} -> raw-space ablation "
              f"(re-export directions for mean-centred steering).")
    print(f"[batch] concept={args.concept} site={args.site} "
          f"{'(multi-layer: all diffusion layers)' if layer is None and args.site=='diffusion' else ''}")

    # protein list
    built = {p.stem for p in Path(args.input_dir).glob(f"*{cfg.input_suffix}")}
    if args.proteins:
        proteins = [p for p in args.proteins.split(",") if p]
    elif Path(args.from_survey).exists():
        survey = json.loads(Path(args.from_survey).read_text())
        proteins = sorted(
            (p for p, d in survey.items()
             if isinstance(d.get("helix"), (int, float)) and args.helix_min <= d["helix"] <= args.helix_max),
            key=lambda p: survey[p]["helix"],
        )
    else:
        proteins = sorted(built)
    proteins = [p for p in proteins if p in built][: args.max_proteins]
    if not proteins:
        raise SystemExit("No proteins to run (check --from_survey band / --input_dir).")
    print(f"[batch] {len(proteins)} proteins, site={args.site}: {proteins}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(cfg.checkpoint_path).expanduser() if cfg.checkpoint_path else download_cache(Path(cfg.cache_dir))
    model = load_boltz_model(ckpt, recycling_steps=cfg.recycling_steps, sampling_steps=cfg.sampling_steps,
                             diffusion_samples=cfg.diffusion_samples, step_scale=cfg.step_scale)

    all_rows: list[dict] = []
    results_path = out_dir / f"batch_{args.site}.jsonl"
    with open(results_path, "w", encoding="utf-8") as fh:
        for i, pid in enumerate(proteins):
            try:
                rows = run_protein(model, pid, cfg, bank, ckpt, where, target, layer, out_dir,
                                   rand_seed=i, concept=args.concept, n_random=args.n_random)
            except Exception as e:  # noqa: BLE001  # one protein shouldn't sink the batch
                print(f"[batch] {pid}: FAILED ({str(e)[:100]})")
                rows = []
            for r in rows:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
            all_rows.extend(rows)

    # aggregate by multiplier: concept-direction vs random-direction (the specificity curve).
    # d_target = the steered concept's own DSSP state (helix for helix/SAE, strand for strand, ...).
    n_prot = len({r["protein"] for r in all_rows if r.get("kind")})
    print("\n" + "=" * 74)
    print(f"{args.concept.upper()} STEERING -- {n_prot} proteins, site={args.site}  "
          f"(mean+/-std d_{ _TARGET_SS.get(args.concept,'helix') } by dose multiplier)")
    print("=" * 74)
    print(f"  {'mult k':>7s} {'concept-add':>14s} {'random-add':>14s} {'cpt pLDDT':>12s} {'rand pLDDT':>11s}")
    for k in MULTIPLIERS:
        ha = [r["d_target"] for r in all_rows if r.get("kind") == "helix_add" and r["mult"] == k]
        ra = [r["d_target"] for r in all_rows if r.get("kind") == "rand_add" and r["mult"] == k]
        hp = [r["plddt"] for r in all_rows if r.get("kind") == "helix_add" and r["mult"] == k]
        rp = [r["plddt"] for r in all_rows if r.get("kind") == "rand_add" and r["mult"] == k]
        print(f"  {k:>7g} {_agg(ha):>14s} {_agg(ra):>14s} {statistics.mean(hp) if hp else float('nan'):>12.1f} "
              f"{statistics.mean(rp) if rp else float('nan'):>11.1f}")
    print("  ablation a=1: concept", _agg([r["d_target"] for r in all_rows if r.get("kind") == "helix_ablate"]),
          " random", _agg([r["d_target"] for r in all_rows if r.get("kind") == "rand_ablate"]))
    print(f"\n[batch] rows -> {results_path}")
    print("[batch] SUFFICIENCY holds if helix-add stays positive & pLDDT-stable while random-add is "
          "flat/negative; the multiplier where helix pLDDT drops or helix-add reverses = manifold edge.")


if __name__ == "__main__":
    main()
