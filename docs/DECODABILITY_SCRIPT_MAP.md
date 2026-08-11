# Decodability — script map

Every script in `decodability/`, what it consumes, and what it emits. Run everything with
`decodability/` as the working directory (the scripts import each other as flat siblings).

Common assumptions:
- `downloads_layer{N}/` holds Pairformer activations for layer N (see `get_activations.py`).
- `downloads_diffusion_rec{R}/` holds diffusion activations (see `stage_diffusion_data.py`).
- `processed_swissprot_a5/` holds the processed SwissProt shards (see
  `swissprot_annotation_pipeline.py`). Two scripts fall back to `inputs/swissprot_a5.fasta` when it
  is absent; the rest need the shards.
- SAE checkpoints are pulled from HuggingFace on demand by `layer_analysis_utils.py`.

Large-artifact locations and credentials: [`DATA_ACCESS.md`](DATA_ACCESS.md).

---

## 1. Labels

`swissprot_annotation_pipeline.py`
- Downloads UniProt annotations for a protein-ID list and writes per-residue multi-hot concept
  encodings as processed shards.
- In: `--input_ids inputs/swissprot_a5_ids.txt` (the 5,000-protein SwissProt-A5 eval pool).
- Out: `--output_dir processed_swissprot_a5/` — `shard_*/annotations.npz`,
  `shard_*/sequences.fasta`, `concept_vocabulary.json`.
- This is the prerequisite for most of the rest of the pipeline. `inputs/swissprot_a5.fasta` is a
  committed copy of the sequences it emits.

`build_structure_secondary_structure_annotations.py`
- Dense DSSP secondary structure from AlphaFold DB structures (the `secondary` concept set).
- In: `inputs/swissprot_a5_prefixes.txt`; downloads AlphaFold CIFs.
- Out: a processed concept dir of per-residue helix/strand/coil labels.

`build_boltz_structure_annotations.py`
- Same, but DSSP on **Boltz-1's own** predicted structures (the `boltz_secondary` set). This is the
  self-consistency control behind S7.
- In: `inputs/swissprot_a5_prefixes.txt`; runs/reads Boltz CIFs.

`build_amino_acid_concepts.py`
- One-hot residue identity (`aa:A` .. `aa:Y`) — the positive control behind Fig 1 / S1 / S9.
- In: `--source_dir processed_swissprot_a5` (`shard_*/sequences.fasta`); falls back to
  `inputs/swissprot_a5.fasta`.
- Out: `--out_dir processed_swissprot_aa/`.

`count_concept_label_stats.py`
- Per-concept label prevalence over `inputs/common_activation_proteins.txt` (the 486-protein eval
  set) — the source for `tab:dataset_label_prevalence`.
- Out: `data/sae_explore/concept_label_stats.csv`.

## 2. Activations

`get_activations.py`
- Downloads per-residue Boltz-1 activations from Cloudflare R2 / S3.
- In: `--bucket`, `--manifest` (`inputs/80proteins.txt` for the 84k training set,
  `inputs/SwissProtproteins.txt` for eval), `--layer`, `--rec`, `--layer_type`. All required.
- Out: `--out downloads_layer{N}/`.
- Needs the R2/S3 credentials issued on request (see `DATA_ACCESS.md`); there is no anonymous access.

`stage_diffusion_data.py`
- Stages diffusion activations **and** the per-seed diffusion SAE checkpoints from S3.
- Out: `downloads_diffusion_rec{R}/`, `sae_explore/diffusion_cache/rec{R}/`.
- Needs AWS credentials (the diffusion checkpoints are not on HuggingFace).

`generate_manifest.py`
- Lists protein prefixes in a bucket to build a manifest. Needs S3 access; the manifests it produces
  are already committed under `inputs/`.

`compute_common_proteins.py`
- Intersects the proteins available at *every* (layer, rec) so `n_proteins` does not drift between
  layers, making cross-layer and cross-module comparisons like-for-like.
- Out: `inputs/common_activation_proteins.txt` (486 proteins).

## 3. SAE training

`setup_train_sae.sh` — installs CUDA PyTorch wheels. Run once per machine.

`train_sae.py`
- TopK SAE, k=256, 2048 latents, trained on demeaned activations, ℓ2 = 3e-3, 3 seeds, 500k steps.

`eval_sae.py` — reconstruction/sparsity eval for one checkpoint → `eval_step_*.json`.

`sae_utils.py` — shared SAE model definition, checkpoint I/O, decoder normalisation.

`layer_analysis_utils.py`
- Downloads and caches the per-layer SAE checkpoints and eval artefacts from
  `evolve-away/Boltz1-SAEs-L2` (uses `requests`; no `huggingface_hub`, no GPU).
- `download_run_checkpoint(layer, rec)` is the entry point the rest of the pipeline uses.

## 4. Benchmarks

`loaders/` — the two shared activation loaders. `raw_activation_loader.RawActivationLoader` reads
raw per-residue activations; `sae_latent_loader.SAELatentActivationLoader` loads a checkpoint,
demeans, and encodes to latents. Imported by the benchmark, amino-acid, and direction-export
scripts. (Formerly named `auto_interp/`, which had nothing to do with auto-interpretability.)

`benchmark_f1.py` — the domain-level F1 metric (per-residue precision × per-domain recall) and the
best-of-features threshold sweep, plus the circular-shift permutation null.

`embeddings_concepts_evaluation.py` — the vectorised precision/recall/F1 engine `benchmark_f1` and
the amino-acid scripts sit on. Despite the name, this is core scoring machinery, not an embeddings
experiment.

`linear_probe.py` — grouped-by-protein K-fold logistic probes, out-of-fold scoring.

`run_layer_benchmark.py`
- The workhorse. For one (layer, rec, seed) scores all four readouts — probe-raw, probe-SAE,
  SAE-1feat, neuron-1feat — against a concept set, with permutation nulls.
- Out: `sae_explore/layer_sweep_rec{R}/benchmark/layer{L}/{set}_seed{S}_benchmark.json`.
- `--layer_type {pairformer,diffusion}` selects the stack.

`run_parallel_benchmark.py` — fans `run_layer_benchmark.py` out over layers/seeds.

`aggregate_seed_benchmark.py`
- Averages the per-seed JSONs into `*_agg.json`. SAE-derived numbers get mean + std across the 3
  seeds; probe-raw and neuron baselines are seed-independent and asserted to match.

`augment_benchmark.py` / `run_parallel_augment.py`
- Adds the probe-SAE readout (logistic probe on alive SAE latents) to existing benchmark JSONs —
  the upper bound shown in S5.

## 5. Derived analyses

`compute_precision_recall.py` — re-scores the SAE/neuron winners and the held-out probe to emit
precision and recall separately → `data/sae_explore/precision_recall{,_diffusion}.csv` (Fig 4).

`bootstrap_concept_f1.py` — cluster bootstrap over proteins (2000 resamples, fixed operating point)
→ `data/sae_explore/concept_f1_ci.csv`. Feeds the Fig 3A error bars and S11.

`amino_acid_sanity_check.py` — per-layer amino-acid identity across SAE-1feat / neuron-1feat /
probe → `data/sae_explore/amino_acid_sanity/results.jsonl` (Fig 1, S1).

`amino_acid_cross_seed.py` — the same, across seeds, for the diffusion stack →
`amino_acid_sanity/cross_seed_results_diffusion.jsonl` (S9).

`check_probe_homology.py`
- Sequence-identity leakage control for the Limitations section. All-vs-all BLOSUM62 global
  alignment over the 486 eval proteins; reports what fraction of held-out proteins have a >30%
  identity neighbour in their training fold.
- In: `inputs/common_activation_proteins.txt` + sequences (shards, else
  `inputs/swissprot_a5.fasta`).
- Out: `data/homology/probe_homology_{report.txt,matrix.npy,pairs.csv}`.

## 6. Steering bridge

`export_probe_direction.py` / `export_sae_direction.py`
- Fit the steering directions used by [`../steering/`](../steering/): the held-out logistic-probe
  weight mapped back to raw-activation space, and the decoder column of the highest-F1 latent.
- Both write to `../steering/directions/` by default. Protocol:
  [`STEERING_METHODS.md`](STEERING_METHODS.md) §2.

`validate_directions_f1.py`
- Re-scores the exact exported directions for held-out F1 on the 97 steered proteins — the
  vector-validity check in the appendix (helix 0.83, strand 0.78, coil 0.88).

## 7. Figures and tables

`_build_paper_a_figures.py` / `_build_paper_a_supplementary.py`
- Generate `paper_a_figures.ipynb` / `paper_a_supplementary.ipynb`. Edit the **generators**, not the
  notebooks — direct notebook edits are wiped on regen.
- Read `data/sae_explore/` and write `figures/`.

`make_tables.py` — `tab:headline` and `tab:dataset_label_prevalence` as CSV + LaTeX →
`data/tables/`.

## 8. Orchestration

| script | does |
|---|---|
| `run_full_workflow.sh` | download activations → train 3 seeds → eval. One layer per invocation. |
| `run_analysis_workflow.sh` | annotate → download → stage SAEs → benchmark → aggregate → upload. |
| `run_train_and_analyse_workflow.sh` | chains the two above. |
| `run_boltz_secondary_analysis.sh` | the Boltz-own-DSSP variant of the benchmark (S7). |
