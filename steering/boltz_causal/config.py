"""Experiment configuration for the causal-intervention runs.

Mirrors the config/dataclass pattern in ``boltz_pruning/run_pruning_on_aws_s3_v2.py`` and the shard
model in ``aas-boltz-runner`` so the EC2/S3 orchestration drops in with minimal glue.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# The depth-sweep injection points for Condition 1 (PairformerLayer indices spanning the mid-trunk
# disulfide hotspot at frac depth 0.2–0.35 through the final conditioning at L47).
DEFAULT_TRUNK_SWEEP: tuple[int, ...] = (10, 16, 24, 32, 47)

# Graded ablation strengths for Panel G.
DEFAULT_ALPHAS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

# The diffusion layer the C2 disulfide direction is fit on (module output, depth-matched to the
# trunk L47 output). MUST equal export_probe_direction._where_spec("diffusion")'s layer — the
# direction is a property of *this* layer's activation space, so C2 projects only here (applying it
# at other diffusion layers would inject off-distribution noise, not a cleaner control).
DIFFUSION_PROBE_LAYER: int = 22


@dataclass
class ConditionSpec:
    """One row of the logic table, expanded per protein into InterventionSpecs at run time."""

    name: str  # "C1", "C2", "C3", "S1", "G", "Suf"
    concept: str  # "disulfide_bond" | "helix" | "random"
    target: str  # "trunk_output" | "trunk_depth" | "diffusion" | "combined"
    mode: str = "ablate"  # or "add" for sufficiency
    alphas: tuple[float, ...] = (1.0,)
    trunk_layers: tuple[int, ...] = ()  # for trunk_depth sweep; empty otherwise
    diffusion_layer: int | None = None  # for target="diffusion": the single layer to project at
    residue_key: str = "concept"  # "concept" (annotated residues) | "cys" | "all" | "free_cys"
    # For target="combined" (E3 multi-site ablation): the sites to perturb SIMULTANEOUSLY in one
    # forward, each as (target, layer). The direction key per site is derived from its target
    # (trunk_output->trunk_L47, diffusion->DIFFUSION_PROBE_LAYER, trunk_depth->trunk_L{layer}).
    combined_sites: tuple[tuple[str, int | None], ...] = ()


DEFAULT_CONDITIONS: tuple[ConditionSpec, ...] = (
    ConditionSpec("C1_output", "disulfide_bond", "trunk_output", residue_key="cys"),
    ConditionSpec("C1_depth", "disulfide_bond", "trunk_depth", trunk_layers=DEFAULT_TRUNK_SWEEP, residue_key="cys"),
    ConditionSpec("C2", "disulfide_bond", "diffusion", diffusion_layer=DIFFUSION_PROBE_LAYER, residue_key="cys"),
    ConditionSpec("C3", "helix", "trunk_output", residue_key="all"),
    ConditionSpec("S1_random", "random", "trunk_output", residue_key="cys"),
    ConditionSpec("G_graded", "disulfide_bond", "trunk_output", alphas=DEFAULT_ALPHAS, residue_key="cys"),
)


# E3 (opt-in): combined multi-site ablation -- remove helix from BOTH the trunk conditioning and the
# diffusion token repr in one forward, to test whether necessity emerges only when a redundantly
# encoded feature is knocked out everywhere. Requires helix@diffusion in the directions .npz
# (export_probe_direction.py --concepts helix --include_diffusion). Append to conditions to run.
COMBINED_ABLATION = ConditionSpec(
    "E3_combined", "helix", "combined", residue_key="all",
    combined_sites=(("trunk_output", None), ("diffusion", DIFFUSION_PROBE_LAYER)),
)


@dataclass
class ExperimentConfig:
    """Top-level run config."""

    # data / directions
    directions_npz: str = "directions/disulfide_helix_directions.npz"
    proteins_file: str = "disulfide_proteins.txt"  # UniProt IDs, filtered to bonded-at-baseline
    input_yaml_dir: str = "inputs"  # per-protein Boltz YAML (sequence + MSA), aas-boltz-runner style
    input_suffix: str = ".yaml"  # ".yaml" or ".fasta"; each named "{protein_id}{suffix}"
    disulfide_pairs_json: str = "disulfide_pairs.json"  # {protein_id: [[a,b], ...]} 1-based
    cache_dir: str = "~/.boltz"

    # model / inference
    checkpoint_path: str = ""  # empty → download boltz1_conf.ckpt into cache_dir
    # MUST match the recycle the steering directions were fit at. The correlational study's
    # canonical conditioning (what diffusion reads) is trunk **recycle 1**, and
    # export_probe_direction fits at rec=1 (PFLayer_{L}_rec_1). The trunk conditioning `s_trunk`
    # handed to diffusion is the FINAL recycle iteration, so recycling_steps=1 makes that final
    # == rec1 == the direction's fit space. Using >1 here misaligns the direction. (Paper S8:
    # rec0 vs rec1 probe-F1 |Δ|=0.029, so the representation is stable across recycles, but we
    # match it exactly rather than rely on that.)
    recycling_steps: int = 1
    # 200-step trajectory: the diffusion direction is fit at step 199 (rec199 = final step), so the
    # last DiffusionTransformerLayer pass the C2 hook sees matches the fit. Keep at 200.
    sampling_steps: int = 200
    diffusion_samples: int = 1
    step_scale: float = 1.638
    accelerator: str = "gpu"
    devices: int = 1
    num_workers: int = 2
    use_msa_server: bool = False
    seed: int = 0

    # experiment
    conditions: tuple[ConditionSpec, ...] = DEFAULT_CONDITIONS
    run_baseline: bool = True  # the α=0 pass every effect size is measured against
    # depth-sweep ablation on the final recycle only, to match the rec-1 direction (see hooks.py);
    # set False to ablate at layer L on every recycle pass (stronger, but not rec-matched).
    trunk_depth_last_recycle_only: bool = True
    disulfide_bond_cutoff: float = 2.5  # Å; "formed" threshold for the baseline filter

    # io
    output_dir: str = "outputs"
    # S3 (from aas-boltz-runner); leave empty for local runs
    source_bucket: str = ""
    source_prefix: str = ""
    target_bucket: str = ""
    target_prefix: str = ""

    extra: dict[str, object] = field(default_factory=dict)
