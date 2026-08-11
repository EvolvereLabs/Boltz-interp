#!/usr/bin/env python3
"""Driver for the causal-intervention runs.

Per protein: process inputs once, load the model once (reused across every condition), then for the
baseline (α=0) and each condition install hooks → Boltz forward → mmCIF → structural readout.

    baseline pass (no hooks)                  → baseline CIF + Sγ–Sγ reference
    for condition × α × (trunk layer):
        with register_intervention(model, spec, residue_idx):
            predict_to_cif(...)               → intervened CIF
        Δ Sγ–Sγ vs baseline                   → effect size

Model-run glue lives in ``boltz_causal.boltz_runtime`` (ported from the fork's ``main.predict`` and
``aas-boltz-runner/task/run_npz.py``). Validate on 2–3 proteins on a GPU box before wiring the
EC2/S3 shard loop (see README status + PAPER_A_CAUSAL_PLAN.md build order).

Token/residue mapping: for single-chain proteins Boltz uses one token per residue, so a 1-based
UniProt residue ``n`` maps to 0-based token index ``n-1``. Multi-chain / modified inputs would need
the record's token map — out of scope for the monomeric SwissProt disulfide set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from boltz_causal.boltz_runtime import (
    download_cache,
    load_boltz_model,
    make_datamodule,
    predict_to_cif,
    process_protein,
)
from boltz_causal.config import DIFFUSION_PROBE_LAYER, ConditionSpec, ExperimentConfig
from boltz_causal.directions import DirectionBank, InterventionSpec, load_directions
from boltz_causal.hooks import register_interventions, residue_indices
from boltz_causal.structural_readout import helix_fraction, is_bonded, mean_plddt, sg_sg_distances


# --------------------------------------------------------------------------------------------------
# expand the logic table into concrete interventions
# --------------------------------------------------------------------------------------------------
def build_specs(
    cond: ConditionSpec,
    bank: DirectionBank,
    trunk_output_where: str = "trunk_L47",
) -> list[tuple[list[InterventionSpec], str]]:
    """Expand one ConditionSpec into ``(specs, tag)`` groups. ``specs`` is applied together in one
    forward -- length 1 for single-site conditions, several for target='combined' (E3 multi-site)."""
    groups: list[tuple[list[InterventionSpec], str]] = []
    for alpha in cond.alphas:
        if cond.target == "combined":
            specs = [_make_spec(cond, bank, t, lyr, alpha, trunk_output_where)
                     for t, lyr in cond.combined_sites]
            groups.append((specs, f"{cond.name}_a{alpha}"))
        elif cond.target == "trunk_depth":
            for layer in cond.trunk_layers:
                groups.append(([_make_spec(cond, bank, "trunk_depth", layer, alpha, trunk_output_where)],
                               f"{cond.name}_L{layer}_a{alpha}"))
        else:
            layer = cond.diffusion_layer if cond.target == "diffusion" else None
            groups.append(([_make_spec(cond, bank, cond.target, layer, alpha, trunk_output_where)],
                           f"{cond.name}_a{alpha}"))
    return groups


def _site_where(target: str, layer: int | None, trunk_output_where: str) -> str:
    """Direction key for a site: trunk_output->final conditioning, diffusion->diffusion, depth->layer."""
    if target == "diffusion":
        return "diffusion"
    if target == "trunk_depth":
        return f"trunk_L{layer}"
    return trunk_output_where  # trunk_output


def _make_spec(cond: ConditionSpec, bank: DirectionBank, target: str, layer: int | None,
               alpha: float, trunk_output_where: str) -> InterventionSpec:
    where = _site_where(target, layer, trunk_output_where)
    site_layer = DIFFUSION_PROBE_LAYER if (target == "diffusion" and layer is None) else layer
    return InterventionSpec(cond.concept, _direction_vector(cond, bank, where), target, cond.mode,
                            alpha, site_layer, cond.residue_key, mean=_direction_mean(cond, bank, where))


def _direction_vector(cond: ConditionSpec, bank: DirectionBank, where: str):
    if cond.concept == "random":
        return bank.random_like("disulfide_bond", where)  # matched-norm control (S1)
    return bank.get(cond.concept, where)


def _direction_mean(cond: ConditionSpec, bank: DirectionBank, where: str):
    """Feature mean for mean-centred ablation; random reuses the disulfide reference mean (fair)."""
    ref = "disulfide_bond" if cond.concept == "random" else cond.concept
    return bank.get_mean(ref, where)


# --------------------------------------------------------------------------------------------------
# per-protein run
# --------------------------------------------------------------------------------------------------
def run_protein(
    model,
    protein_id: str,
    cfg: ExperimentConfig,
    bank: DirectionBank,
    ckpt_cache: Path,
    pairs_by_protein: dict[str, list[tuple[int, int]]],
    out_dir: Path,
) -> list[dict]:
    """Baseline + all conditions for one protein; return result rows."""
    device = next(model.parameters()).device
    pairs = pairs_by_protein[protein_id]
    cys_positions = sorted({p for pair in pairs for p in pair})
    cys_idx = residue_indices([p - 1 for p in cys_positions], device)  # 1-based → 0-based token
    residue_idx_for = {"cys": cys_idx, "concept": cys_idx, "all": None, "free_cys": cys_idx}

    # input processing + datamodule (once per protein)
    input_path = Path(cfg.input_yaml_dir) / f"{protein_id}{cfg.input_suffix}"
    work_dir = out_dir / "proc" / protein_id
    processed = process_protein(
        input_path, work_dir, ckpt_cache.parent / "ccd.pkl", use_msa_server=cfg.use_msa_server
    )
    datamodule = make_datamodule(processed, num_workers=cfg.num_workers)

    def _predict(tag: str) -> Path:
        # Reseed before every pass so baseline and all conditions denoise from IDENTICAL diffusion
        # noise -- otherwise the global RNG advances per condition and the effect size is confounded
        # with an independent noise draw. Noise-matched baseline vs condition
        # is what makes delta Sg-Sg a causal effect rather than sample-to-sample variation.
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        return predict_to_cif(
            model,
            datamodule,
            targets_dir=processed.targets_dir,
            predictions_dir=out_dir / "predictions" / protein_id / tag,
            accelerator=cfg.accelerator,
            devices=cfg.devices,
        )

    rows: list[dict] = []

    # 1) baseline
    base_cif = _predict("baseline")
    base_dist = sg_sg_distances(base_cif, pairs)
    rows.append(_row(protein_id, "baseline", None, base_dist, base_cif))

    # 2) conditions
    n_recycles = cfg.recycling_steps + 1  # trunk runs the stack this many times
    for cond in cfg.conditions:
        residue_idx = residue_idx_for.get(cond.residue_key, cys_idx)
        for specs, tag in build_specs(cond, bank):
            with register_interventions(
                model, [(s, residue_idx) for s in specs],
                n_recycles=n_recycles,
                last_recycle_only=cfg.trunk_depth_last_recycle_only,
            ):
                cif_path = _predict(tag)
            dist = sg_sg_distances(cif_path, pairs)
            # log the first spec (single-site) or a synthetic combined marker for multi-site groups
            rows.append(_row(protein_id, tag, specs[0] if len(specs) == 1 else _combined_marker(specs),
                             dist, cif_path, base_dist))
    return rows


def _combined_marker(specs: list[InterventionSpec]):
    """A lightweight stand-in for _row's spec fields when several sites were applied together."""
    from types import SimpleNamespace
    return SimpleNamespace(target="combined(" + "+".join(s.target for s in specs) + ")",
                           concept=specs[0].concept, alpha=specs[0].alpha, layer=None)


def _row(protein_id, tag, spec, dist, cif_path, base_dist=None) -> dict:
    row = {
        "protein": protein_id,
        "condition": tag,
        "target": None if spec is None else spec.target,
        "concept": None if spec is None else spec.concept,
        "alpha": None if spec is None else spec.alpha,
        "layer": None if spec is None else spec.layer,
        "sg_sg": {f"{a}-{b}": d for (a, b), d in dist.items()},
        "n_bonded": int(sum(is_bonded(d) for d in dist.values())),
        "mean_plddt": mean_plddt(cif_path),
        "helix_fraction": helix_fraction(cif_path),  # geometry readout for the C3 control
        "cif": str(cif_path),
    }
    if base_dist is not None:
        row["delta_sg_sg"] = {f"{a}-{b}": (dist[(a, b)] - base_dist[(a, b)]) for (a, b) in dist}
        row["mean_delta_sg_sg"] = _nanmean([row["delta_sg_sg"][k] for k in row["delta_sg_sg"]])
    return row


def _nanmean(xs: list[float]) -> float:
    vals = [x for x in xs if x == x]  # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")


def load_disulfide_pairs(path: str | Path) -> dict[str, list[tuple[int, int]]]:
    """Load ``{protein_id: [[a, b], ...]}`` (1-based) produced from UniProt DISULFID annotations."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {pid: [tuple(p) for p in pairs] for pid, pairs in raw.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Causal directional-ablation runs on Boltz-1.")
    ap.add_argument("--directions", default=None)
    ap.add_argument("--proteins", default=None)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--input_dir", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--accelerator", default=None, choices=["gpu", "cpu"])
    args = ap.parse_args()

    cfg = ExperimentConfig()
    if args.directions:
        cfg.directions_npz = args.directions
    if args.proteins:
        cfg.proteins_file = args.proteins
    if args.pairs:
        cfg.disulfide_pairs_json = args.pairs
    if args.input_dir:
        cfg.input_yaml_dir = args.input_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.accelerator:
        cfg.accelerator = args.accelerator

    torch.set_grad_enabled(False)
    torch.manual_seed(cfg.seed)

    bank = load_directions(cfg.directions_npz)
    pairs_by_protein = load_disulfide_pairs(cfg.disulfide_pairs_json)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(cfg.checkpoint_path).expanduser() if cfg.checkpoint_path else download_cache(Path(cfg.cache_dir))
    model = load_boltz_model(
        ckpt,
        recycling_steps=cfg.recycling_steps,
        sampling_steps=cfg.sampling_steps,
        diffusion_samples=cfg.diffusion_samples,
        step_scale=cfg.step_scale,
    )

    # Run set: an explicit proteins_file if present, else exactly the inputs build_inputs produced
    # (inputs/*.yaml), else all proteins with pairs. Deriving from the inputs dir keeps the run in
    # lockstep with build_inputs' --limit.
    if Path(cfg.proteins_file).exists():
        proteins = Path(cfg.proteins_file).read_text().split()
    else:
        built = sorted(p.stem for p in Path(cfg.input_yaml_dir).glob(f"*{cfg.input_suffix}"))
        proteins = built or list(pairs_by_protein.keys())
        print(f"[run] no {cfg.proteins_file}; running {len(proteins)} built inputs from {cfg.input_yaml_dir}/")

    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as fh:
        for pid in proteins:
            if pid not in pairs_by_protein:
                print(f"[skip] {pid}: no disulfide pairs in {cfg.disulfide_pairs_json}")
                continue
            for row in run_protein(model, pid, cfg, bank, ckpt, pairs_by_protein, out_dir):
                fh.write(json.dumps(row) + "\n")
                fh.flush()
    print(f"Wrote {out_dir / 'results.jsonl'}")


if __name__ == "__main__":
    main()
