# Decodability pipeline (Paper A, Results §3.1–3.5)

Trains sparse autoencoders on Boltz-1 activations and benchmarks four readouts — **probe-raw**
(held-out L2 logistic probe on raw activations), **probe-SAE**, **SAE-1feat** (best single latent),
**neuron-1feat** — against secondary-structure, sequence-chemistry, and amino-acid-identity labels,
separately for the Pairformer trunk and the diffusion module.

Run everything with this directory as the working directory (scripts import each other as flat
siblings). Install: `pip install -e .` (or `uv sync`). Large inputs (activations, SAE checkpoints)
are pulled on demand — see [`../docs/DATA_ACCESS.md`](../docs/DATA_ACCESS.md).

## Layout

```
decodability/
├── data/            committed SUMMARY data (regenerate figures offline; ≈ 4 MB)
│   ├── sae_explore/ benchmark aggregates (*_agg.json), per-seed SS benchmarks,
│   │                precision_recall*.csv, concept_f1_ci.csv, concept_label_stats.csv,
│   │                amino_acid_sanity/*.jsonl
│   ├── homology/    sequence-identity leakage report (check_probe_homology.py)
│   └── tables/      generated manuscript tables (make_tables.py)
├── figures/         the 12 decodability figures the manuscript includes (.pdf/.png)
├── inputs/          protein-ID lists, manifests, eval-set sequences
├── loaders/         shared activation loaders (raw / SAE-latent)
├── *.py             pipeline + analysis modules (below)
├── *.sh             training / benchmark orchestration
├── make_tables.py   manuscript table generator
└── _build_paper_a_*.py   figure-notebook generators
```

## Pipeline order (from raw activations)

1. **Labels** — `swissprot_annotation_pipeline.py` (UniProt → per-residue multi-hot; builds
   `processed_swissprot_a5/`, the prerequisite for most of what follows),
   `build_structure_secondary_structure_annotations.py` (DSSP on AlphaFold),
   `build_boltz_structure_annotations.py` (DSSP on Boltz's own CIF; the self-consistency control),
   `build_amino_acid_concepts.py` (one-hot residue identity; the positive control).
2. **Activations** — `get_activations.py` (Pairformer, per layer/rec → `downloads_layer{N}/`),
   `stage_diffusion_data.py` (diffusion activations + per-seed SAE checkpoints).
3. **SAE training** — `train_sae.py` (TopK, k=256, 2048 latents, demeaned, L2=3e-3, 3 seeds,
   500k steps). Trunk SAEs are also released on HuggingFace and fetched by
   `layer_analysis_utils.py`.
4. **Benchmarks** — `run_layer_benchmark.py` (per layer/rec/seed → the four readouts + permutation
   null), parallelised by `run_parallel_benchmark.py`, averaged across the 3 seeds by
   `aggregate_seed_benchmark.py` → `*_agg.json`. `augment_benchmark.py` adds the probe-SAE upper
   bound.
5. **Derived analyses** — `compute_precision_recall.py` (precision/recall split),
   `bootstrap_concept_f1.py` (protein-bootstrap CIs), `amino_acid_sanity_check.py` /
   `amino_acid_cross_seed.py` (residue-identity calibration),
   `count_concept_label_stats.py` (dataset-prevalence table), `check_probe_homology.py`
   (sequence-identity leakage control, Limitations).
6. **Figures and tables** — `_build_paper_a_figures.py` / `_build_paper_a_supplementary.py`
   generate the figure notebooks, which read `data/` and write `figures/`; `make_tables.py` writes
   `data/tables/`.

`run_analysis_workflow.sh` / `run_full_workflow.sh` / `run_train_and_analyse_workflow.sh` chain the
above; `run_boltz_secondary_analysis.sh` runs the Boltz-own-DSSP variant.

## Regenerate figures and tables from committed data

```bash
python _build_paper_a_figures.py
python _build_paper_a_supplementary.py
python -m jupyter nbconvert --to notebook --execute --inplace paper_a_figures.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace paper_a_supplementary.ipynb
python make_tables.py
```

The builders read `data/sae_explore/...` and write to `figures/`. Full
figure→data→script provenance: [`../docs/DECODABILITY_FIGURE_PROVENANCE.md`](../docs/DECODABILITY_FIGURE_PROVENANCE.md);
per-script reference: [`../docs/DECODABILITY_SCRIPT_MAP.md`](../docs/DECODABILITY_SCRIPT_MAP.md).

## Bridge to steering

`export_probe_direction.py` and `export_sae_direction.py` fit the steering directions used by
[`../steering/`](../steering/), and `validate_directions_f1.py` re-scores them for held-out F1 on
the 97 steered proteins. They reuse this project's probe/SAE machinery, so they live here; by
default they write to `../steering/directions/`. See
[`../docs/STEERING_METHODS.md`](../docs/STEERING_METHODS.md) §2.
