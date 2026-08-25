# Boltz-Paper-A — reproduction suite

Reproducibility code and summary data for **"Probing and steering biology across Boltz-1"**

The paper has two halves, and so does this repo:

1. **Decodability** (correlational) — where biological annotations are *linearly decodable* in
   Boltz-1. Trains sparse autoencoders (SAEs), fits held-out linear probes, and benchmarks four
   readouts (probe-raw, probe-SAE, SAE-1feat, neuron-1feat) against secondary-structure,
   sequence-chemistry, and amino-acid-identity labels across the Pairformer trunk and the diffusion
   module. → [`decodability/`](decodability/)
2. **Steering** (causal) — whether the trunk's decodable secondary-structure signal is functionally
   *used*. Fits held-out steering directions on the trunk single representation, re-folds under
   additive/ablative interventions, and reads out DSSP secondary structure, pLDDT, and Cα-RMSD.
   → [`steering/`](steering/)

```
boltz-paper-a-repro/
├── decodability/          # SAE training, probes, F1 benchmarks, figure + table builders
│   ├── data/              #   committed summary data (agg JSONs + CSVs) → regenerate figs offline
│   ├── inputs/            #   protein-ID lists, manifests, eval-set sequences
│   └── figures/           #   the 12 decodability figures the manuscript includes
├── steering/              # causal steering package, scripts, analysis, figure + table makers
│   ├── data/              #   committed steering rows (JSONL) + frozen plotted CSVs + CIFs
│   ├── directions/        #   fitted steering directions (.npz)
│   └── figures/           #   the 6 steering figures the manuscript includes
└── docs/                  # methods, script map, figure provenance, data access, EC2 launch
```

**Scope.** This repo contains only what reproduces the manuscript. The SAE cross-seed stability
analysis (the ℓ2 recipe cited to the companion stable-SAE study) and the LLM auto-interpretability
pipeline (the companion study referenced in §3.4) live in their own repos.

## Where the data lives

- **Committed here** (≈ 11 MB): all *summary* data needed to regenerate every figure and table —
  benchmark aggregates, probe/precision-recall CSVs, bootstrap CIs, steering rows. See each
  subproject's `data/`.
- **External** (large): the trunk **SAE checkpoints** are public on **HuggingFace**
  ([collection](https://huggingface.co/collections/evolve-away/boltz-saes)) and pulled automatically.
  The **diffusion-module SAEs** are public too. Only the raw per-residue
  **activations** live in private storage (Cloudflare R2 / S3) — **credentials on request**, since
  they run to hundreds of GB. None of this is needed for the quick start below. Full instructions:
  **[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md)**.

## Quick start — regenerate every figure and table from committed data

Decodability (no GPU; the S7 cell imports the benchmark stack, which pulls in `torch`, so install
the full `decodability` env):

```bash
cd decodability
pip install -e .                       # or: uv sync
python _build_paper_a_figures.py       # writes paper_a_figures.ipynb
python _build_paper_a_supplementary.py # writes paper_a_supplementary.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace paper_a_figures.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace paper_a_supplementary.ipynb
python make_tables.py
# → 12 figures to decodability/figures/, tables to decodability/data/tables/
```

Steering (no GPU, no Boltz needed — these read the committed rows):

```bash
cd steering
pip install -e .
python scripts/analyze_full97.py         data/shard4_20260713 .
python scripts/analyze_null.py           data/shard4_20260713 .
python scripts/make_bootstrap_ci.py      data/shard4_20260713 .
python scripts/make_ss_figures.py        data/shard4_20260713 .
python scripts/make_confidence_figure.py data/shard4_20260713 .
python scripts/make_paper_plots.py
python scripts/make_tables.py
# → 5 figures to steering/figures/, CSVs to steering/data/, tables to steering/data/tables/
```

The sixth steering figure needs one manual step — see below.

### The Chimera structure figure

`structure_bidir_3d_chimera.png` composites three **pre-rendered** UCSF Chimera ribbon panels. The
CIFs are committed but the panel PNGs are not, because rendering them is a GUI step:

1. Open `steering/data/shard4_20260713/cifs_A6NI15/A6NI15_baseline.cif`,
   `A6NI15_helix_steered.cif`, and `A6NI15_coil_steered.cif` in the UCSF Chimera **GUI**
   (`--nogui` cannot produce these renders).
2. Apply secondary-structure ribbon colouring, set a transparent background, and keep the **same
   camera** across all three so the panels are comparable.
3. Save as `panel_baseline.png`, `panel_helix_steered.png`, `panel_coil_steered.png`.
4. Pass that directory to the assembler (it defaults to `steering/panels/`):

```bash
python scripts/assemble_structure_viz_bidir_chimera.py /path/to/panels
```

Every other figure regenerates from committed data with no extra input.

## Reproduce end-to-end (from raw activations)

Full pipelines — activation download → SAE training → benchmarks (decodability), and direction
fitting → re-folding interventions on GPU (steering) — are documented per subproject:

- Decodability: [`decodability/README.md`](decodability/README.md),
  [`docs/DECODABILITY_SCRIPT_MAP.md`](docs/DECODABILITY_SCRIPT_MAP.md),
  [`docs/DECODABILITY_FIGURE_PROVENANCE.md`](docs/DECODABILITY_FIGURE_PROVENANCE.md).
- Steering: [`steering/README.md`](steering/README.md),
  [`docs/STEERING_METHODS.md`](docs/STEERING_METHODS.md) (self-contained protocol).

## Figure → script map

The manuscript includes 18 figures. Every one is listed here. Filenames are historical and do not
track the manuscript's own figure numbering — the `.tex` includes them by filename.

| File the manuscript includes | Produced by |
|---|---|
| `fig1_aa_calibration.pdf` | `decodability/_build_paper_a_figures.py` |
| `fig2_trunk_geom_chem.pdf` | `decodability/_build_paper_a_figures.py` |
| `fig3_geom_retained_chem_discarded.pdf` | `decodability/_build_paper_a_figures.py` |
| `fig4_precision_recall.pdf` | `decodability/_build_paper_a_figures.py` |
| `fig5_under_annotation.pdf` | `decodability/_build_paper_a_figures.py` |
| `s1_aa_per_residue.pdf` | `decodability/_build_paper_a_supplementary.py` |
| `s2_diffusion_layer_step.pdf` | `decodability/_build_paper_a_supplementary.py` |
| `s4_insample_vs_heldout.pdf` | `decodability/_build_paper_a_supplementary.py` |
| `s5_sae_faithfulness.pdf` | `decodability/_build_paper_a_supplementary.py` |
| `s7_ss_self_consistency.pdf` | `decodability/_build_paper_a_supplementary.py` |
| `s9_diffusion_aa_grid.pdf` | `decodability/_build_paper_a_supplementary.py` |
| `s11_trunk_vs_diffusion_ci.pdf` | `decodability/_build_paper_a_supplementary.py` |
| `null_distribution_full97.pdf` | `steering/scripts/analyze_null.py` |
| `structure_bidir_3d_chimera.png` | `steering/scripts/assemble_structure_viz_bidir_chimera.py` (+ Chimera GUI) |
| `fig1_sufficiency_dose_response.png` | `steering/scripts/make_paper_plots.py` |
| `fig4_necessity_ablation.png` | `steering/scripts/make_paper_plots.py` |
| `fig7_confusion_matrix.png` | `steering/scripts/make_ss_figures.py` |
| `fig9_confidence_by_pLDDT.png` | `steering/scripts/make_confidence_figure.py` |

## Table → script map

| Manuscript table | Produced by |
|---|---|
| `tab:headline` (depth-matched F1) | `decodability/make_tables.py` |
| `tab:dataset_label_prevalence` | `decodability/make_tables.py` |
| `tab:steering` (paired effects) | `steering/scripts/make_tables.py` |

## License

MIT — see [`LICENSE`](LICENSE).
