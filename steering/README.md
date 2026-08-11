# Causal steering pipeline (Paper A, Results §3.6)

Tests whether the Pairformer trunk's decodable secondary-structure signal is causally **used** by the
diffusion decoder. Fits held-out steering directions on the trunk single representation
`s_trunk` (layer 47), re-runs the full Boltz-1 forward under additive (sufficiency) and ablative
(necessity) interventions, and reads out DSSP secondary-structure fraction, mean Cα pLDDT (on-manifold
monitor), and Cα-RMSD.

**The complete, self-contained protocol — split, direction fitting, intervention math, controls, run
parameters, results — is [`../docs/STEERING_METHODS.md`](../docs/STEERING_METHODS.md).** This README is
the map; that doc is the manual.

## Layout

```
steering/
├── boltz_causal/     intervention library (hooks, projection math, structural readout, config)
├── scripts/          pipeline + analysis + figure-maker scripts
├── directions/       fitted steering directions (disulfide_helix_directions.npz)
├── data/             committed steering data
│   ├── shard4_20260713/         6-concept + random_null shard rows (*.jsonl) + A6NI15 CIFs
│   └── *.csv                    frozen plotted values for the figures
├── figures/          rendered steering figures
├── tests/            projection-math unit tests (no GPU)
└── run_*.sh          EC2 orchestration (steering battery / shards)
```

`boltz_causal/`: `directions.py` (load `.npz` + add/ablate projection math), `hooks.py`
(register/remove the trunk-output monkeypatch, trunk-depth and diffusion forward-hooks),
`steering.py`, `structural_readout.py` (DSSP fraction, Cα-RMSD, pLDDT from output CIF), `config.py`,
`boltz_runtime.py`.

## Pipeline order

Steps 1–2 fit directions using the sibling [`../decodability/`](../decodability/) machinery on saved
eval activations; steps 3–7 rerun Boltz on GPU (the `nutz_and_boltz` fork + TensorLens).

1. **Split** — `scripts/split_probe_proteins.py` → `probe_train_ids.txt` (389) / `probe_heldout_ids.txt` (97), stratified by helix content.
2. **Fit directions** — in `../decodability/`: `export_probe_direction.py` (probe) + `export_sae_direction.py` (SAE-1feat latent), both `--train_ids ../steering/probe_train_ids.txt`, writing `directions/disulfide_helix_directions.npz`.
3. **Inputs** — `scripts/build_inputs.py` (per-protein YAML+MSA from S3).
4. **Baseline survey** — `scripts/select_protein.py` → DSSP helix gradient.
5. **Steering battery** — `scripts/probe_batch.py` (sufficiency dose sweep + matched-norm random control, noise-matched), `scripts/probe_combined.py` (multi-site ablation / necessity), `scripts/run_intervention.py` (low-level single-run intervention driver).
6. **Sharded run** — `run_steering_shard.sh` fans one concept per GPU across `helix / helix_sae / strand / strand_sae / coil / coil_sae` (+ `random_null`); `run_steering_analysis.sh` runs the single-instance battery. Both take their stage commands from [`../docs/STEERING_METHODS.md`](../docs/STEERING_METHODS.md).
7. **Analysis + figures** — see below.

## Regenerate figures from committed data

```bash
python scripts/make_ss_figures.py        data/shard4_20260713 .   # fig7 confusion matrix
python scripts/make_confidence_figure.py data/shard4_20260713 .   # fig9 effect by pLDDT
python scripts/analyze_null.py           data/shard4_20260713 .   # null_distribution_full97
python scripts/assemble_structure_viz_bidir_chimera.py            # A6NI15 bidirectional 3D
```

Two more scripts are CSV-only — they write no figure, but they are the source for the
manuscript's paired-steering-effects table, so keep them in the reproduction path:

```bash
python scripts/analyze_full97.py    data/shard4_20260713 .   # own_state_effect_full97.csv (+3 more)
python scripts/make_bootstrap_ci.py data/shard4_20260713 .   # fig10_bootstrap_ci.csv (95% CIs)
```

`assemble_structure_viz_bidir_chimera.py` needs three **pre-rendered** UCSF Chimera ribbon panels
(`panel_baseline.png` / `panel_helix_steered.png` / `panel_coil_steered.png`; transparent
background, shared camera), passed as the script's first argument (default: `panels/`). They are not
committed — render them from `data/shard4_20260713/cifs_A6NI15/*.cif` in the Chimera GUI first. The DSSP
fractions in the panel titles are recomputed from those CIFs by the script itself.

Figures land in `figures/` and derived CSVs in `data/`. Plotting reads only committed data — no GPU,
no Boltz, no network.

**Steering supplementary fig1 (dose response) and fig4 (necessity ablation)**
(`scripts/make_paper_plots.py`) come from the earlier single-instance run whose raw JSONL rows are
**not** committed; they live on S3/R2 (`docs/STEERING_METHODS.md` §6). The frozen plotted values are
in `data/fig{1,4}*.csv`.

## Model / environment

Interventions need the Boltz-1 fork on GPU: `pip install -e ../nutz_and_boltz` (branch
`feature/update-boltz`) + `tensorlens` (branch `feature/io-queue-2`). Plotting and the unit tests
(`pytest tests/`) need neither. Run parameters (recycles, sampling steps, seeds) are in
`boltz_causal/config.py` and `../docs/STEERING_METHODS.md` §4.
