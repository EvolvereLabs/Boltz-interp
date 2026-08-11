> **Relocation note.** This doc came from the original `boltz_causal_intervention` repo. In this suite, paths written `boltz_causal_intervention/` → `steering/` and `SwissProt_annotations/` → `decodability/`. Steering directions now default to `../steering/directions/`. See the top-level `README.md`.

# Boltz-1 Causal Steering: Methods and Results

Reproduction-grade documentation for the causal steering study on Boltz-1. The question:
are the Pairformer trunk's single-representation (`s`) features **causally used** by the
diffusion module (sufficiency / necessity), or merely decodable? The flagship concept is
secondary-structure **helix**, read out as dense DSSP helix fraction on the predicted structure.

This document is self-contained: an agent or human can reproduce the whole pipeline from it.
Where CLI flags appear they are taken verbatim from the scripts. The GPU fan-out was driven by an
internal launcher that is not part of this release; the per-stage commands it invoked are given
below, so the pipeline can be reproduced on any GPU host.

---

## 1. Overview

- **Model.** Boltz-1 (`boltz1_conf.ckpt`), run through the `nutz_and_boltz` fork (branch
  `feature/update-boltz`), instrumented with TensorLens. The fork's forward is never edited;
  every intervention is a runtime forward-hook / monkeypatch installed before `trainer.predict`
  and removed after (`boltz_causal/hooks.py`).
- **Steering direction.** A unit vector `u` in the raw activation space of one representation
  (trunk single-repr `s_trunk` at PairformerLayer 47, or the diffusion token repr at
  DiffusionTransformerLayer 22). Fit two ways: a supervised logistic **probe** direction and an
  unsupervised **SAE-latent** direction.
- **Held-out design (why).** The activation/annotation pool (`common_activation_proteins.txt`,
  486 proteins) **is** the pool the probe was fit on. Steering with a direction fit on the very
  proteins you then steer is circular. So the pool is split into a TRAIN set (fit the direction)
  and a disjoint HELD-OUT set (run the steering test). All headline results are held-out.
- **Readout.** DSSP helix fraction (`pydssp`, C3 secondary structure, fraction of residues
  assigned `H`), plus CA-RMSD to baseline and mean CA pLDDT (Boltz stores per-atom confidence in
  the mmCIF B-factor column). See `boltz_causal/structural_readout.py`.

The pipeline reruns the full Boltz forward on each sequence for every condition - nothing is
staged/cached at run time. Precomputed activations are used only to *fit* the directions (step 2).

---

## 2. Pipeline (in order)

Run from the repo root `boltz_causal_intervention/`. Steps 2 (direction fitting) run in the
sibling `SwissProt_annotations/` repo because they need the staged per-layer activations there.
The EC2 orchestration in `run_steering_analysis.sh` runs steps 3-8 unattended.

### Step 1 - Stratified train / held-out split

`scripts/split_probe_proteins.py` splits the 486-protein pool by SwissProt helix content into
three bins (`poor` 0.00-0.15, `moderate` 0.15-0.55, `rich` 0.55-1.01) and takes a `--frac`
fraction of each bin into TRAIN, the rest into HELD-OUT. Deterministic (seeded); asserts the two
sets are disjoint.

```bash
python scripts/split_probe_proteins.py \
    --pool ../SwissProt_annotations/common_activation_proteins.txt \
    --annotations uniprot_annotations.tsv --frac 0.8 --seed 0 \
    --train_out probe_train_ids.txt --heldout_out probe_heldout_ids.txt
```

| flag | default | meaning |
|---|---|---|
| `--pool` | `../SwissProt_annotations/common_activation_proteins.txt` | 486-protein activation pool |
| `--annotations` | `uniprot_annotations.tsv` | SwissProt TSV; helix fraction = sum of `HELIX a..b` spans / `Length` |
| `--frac` | `0.8` | training fraction per bin |
| `--seed` | `0` | RNG seed |
| `--train_out` | `probe_train_ids.txt` | 389 proteins (fit the direction) |
| `--heldout_out` | `probe_heldout_ids.txt` | 97 proteins (steer these) |

Outputs: `probe_train_ids.txt` (389), `probe_heldout_ids.txt` (97). The held-out list is committed
to the repo so it travels via git to the EC2 box.

### Step 2 - Fit the steering directions (on the TRAIN split)

Two exporters, both in `../SwissProt_annotations/`, both writing into the SAME
`.npz` (default `../boltz_causal_intervention/directions/disulfide_helix_directions.npz`), keyed
`{concept}@{where}` with a companion `{concept}@{where}.mean`. Both restrict fitting to the
training split via `--train_ids` (this is what makes the run held-out).

**(a) Probe direction** - `export_probe_direction.py`. Fits one L2 logistic probe per
`(concept, layer)` on `StandardScaler`-scaled raw activations, then maps the probe weight back to
raw-activation space and L2-normalises:

```
scale features:  z = (x - mu) / sigma ;  p(y=1) = sigmoid(w . z + b)
raw-space direction:  d = w / sigma   (elementwise)
unit direction:       u = d / ||d||
```

`LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, solver="lbfgs")`,
`StandardScaler`. The scaler mean `mu` is exported as `{concept}@{where}.mean` and used for
mean-centred projection at steering time (see section 3). `helix`/`strand` use the **dense DSSP**
labels (`secondary_structure:helix` / `:strand`), not SwissProt's sparse SS annotation.

```bash
# in ../SwissProt_annotations
python export_probe_direction.py --concepts helix,strand \
    --trunk_layers 47 --include_diffusion \
    --train_ids ../boltz_causal_intervention/probe_train_ids.txt \
    --out ../boltz_causal_intervention/directions/disulfide_helix_directions.npz
```

| flag | default | meaning |
|---|---|---|
| `--concepts` | `disulfide_bond,helix` | comma-separated concept names |
| `--trunk_layers` | `10,16,24,32,47` | pairformer layers to fit (key `trunk_L{n}`) |
| `--include_diffusion` | off | also fit each concept in diffusion space (`@diffusion`, layer 22, rec 199) - required for E2/E3 |
| `--train_ids` | none (fit all) | protein-ID file to fit on (held-out steering) |
| `--max_proteins` | `0` (all) | cap fitted proteins |
| `--out` | `../boltz_causal_intervention/directions/disulfide_helix_directions.npz` | output npz |

A matched-norm `random@trunk_L{last}` control is also written, reusing the reference concept's
mean so the random control is mean-centred identically.

**(b) SAE-latent direction** - `export_sae_direction.py`. Derives a helix direction from the
trunk-L47 TopK SAE (2048 latents) instead of a supervised probe. Default `--select f1` ranks
latents by per-latent helix F1 (the same statistic `run_layer_benchmark` reports, per
protein-domain, best-threshold); default sub-mode is **top-1** (`--k` unset), i.e. the single
highest-F1 latent, and the steering direction is that latent's decoder column, L2-normalised. The
SAE's training demean vector is exported as the `.mean` companion.

```bash
# in ../SwissProt_annotations
python export_sae_direction.py --layer 47 --select f1 --n_perm 0 \
    --train_ids ../boltz_causal_intervention/probe_train_ids.txt \
    --out_concept helix_sae
```

| flag | default | meaning |
|---|---|---|
| `--concept` | `helix` | source label (DSSP helix) |
| `--out_concept` | `helix_sae` | output key concept name; use `helix_sae_f1set` for the F1-above-null set |
| `--layer` | `47` | pairformer trunk layer |
| `--select` | `f1` | `f1` (discrimination) or `crs` (concept relevance score) |
| `--k` | none = 1 | force top-N latents (ignored if a threshold mode is set) |
| `--f1_above_null` | off | select all latents with F1 above the permutation-null 95th pct |
| `--f1_min` | none | select all latents with F1 >= cutoff |
| `--n_perm` | `200` | permutations for the F1 null (must be > 0 for `--f1_above_null`; use `0` for plain top-1) |
| `--seed` | `1` | SAE seed run |
| `--train_ids` | none | fit-on-train protein file |

For the F1-above-null variant re-run with `--f1_above_null --n_perm 200 --out_concept
helix_sae_f1set`. The result reported below (E5) is the top-1 latent: **latent 1987, helix F1 =
0.723**.

`np.savez` cannot append, so `export_sae_direction.py` reloads all existing keys and re-saves them
alongside the two new SAE keys - run it AFTER `export_probe_direction.py` on the same npz.

### Step 3 - Build per-protein inputs

`scripts/build_inputs.py` fetches, for each held-out protein, its Boltz YAML from
`s3://aas-processed-data-us/SwissProtAnnotation5/yaml/{id}.yaml[.gz]` and the per-entity CSV MSA
from `.../msa/{id}_{entity}.csv[.gz]`, un-gzips them, rewrites the YAML `msa:` field to the
absolute local CSV path, and writes `inputs/{id}.yaml`. Requires boto3 + AWS creds. Runs are
offline afterwards (`use_msa_server=False`).

```bash
python scripts/build_inputs.py --proteins probe_heldout_ids.txt --out-dir inputs --limit 40
```

| flag | default | meaning |
|---|---|---|
| `--pairs` | `disulfide_pairs.json` | default run set = keys of this JSON |
| `--proteins` | none | ID file (overrides `--pairs`); used here with `probe_heldout_ids.txt` |
| `--out-dir` | `inputs` | YAML output dir |
| `--msa-dir` | `inputs/msa` | MSA output dir |
| `--limit` | `0` (all) | cap proteins |

### Step 4 - Baseline helix survey (the true gradient)

`scripts/select_protein.py` runs a fast **baseline-only** Boltz prediction (no interventions) for
every built input, measures dense DSSP helix fraction + mean pLDDT, and caches to
`outputs/select/helix_survey.json`. SwissProt HELIX annotations under-report, so this DSSP survey
on the actual baseline structure is the authoritative helix gradient the steering battery selects
proteins from. Re-runs skip cached proteins (`--force` to recompute).

```bash
python scripts/select_protein.py --input_dir inputs --sampling_steps 30 --output_dir outputs/select
```

| flag | default | meaning |
|---|---|---|
| `--input_dir` | `inputs` | built YAMLs |
| `--sampling_steps` | `30` | low is fine for ranking (geometry converges early) |
| `--annotations` | `uniprot_annotations.tsv` | SwissProt helix hint column |
| `--limit` | `0` (all) | cap proteins |
| `--force` | off | recompute cached |

Observed: 40 proteins surveyed, DSSP helix range **0.06-0.95**.

### Step 5 - E1 / E5: trunk steering battery (sufficiency + specificity)

`scripts/probe_batch.py` is the core steering battery. For each protein it runs baseline + an
additive dose sweep of the concept direction and a matched-norm random direction, plus `alpha=1`
ablations, all **noise-matched** (see section 3). Doses are per-protein-calibrated multipliers
`k in {1,2,4,8,16}` of the baseline projection scale, so results aggregate by `k` (a
scale-invariant x-axis), not absolute alpha. Rows stream to `outputs/{output_dir}/batch_{site}.jsonl`.

```bash
# E1 probe direction, trunk site:
python scripts/probe_batch.py --from_survey outputs/select/helix_survey.json \
    --helix_min 0.25 --helix_max 0.85 --site trunk \
    --sampling_steps 50 --max_proteins 40 --output_dir outputs/probe_batch

# E5 SAE latent, trunk site (only differs by --concept and --output_dir):
python scripts/probe_batch.py --from_survey outputs/select/helix_survey.json \
    --helix_min 0.25 --helix_max 0.85 --site trunk --concept helix_sae \
    --sampling_steps 50 --max_proteins 40 --output_dir outputs/probe_batch_helix_sae
```

| flag | default | meaning |
|---|---|---|
| `--from_survey` | `outputs/select/helix_survey.json` | survey to draw proteins from |
| `--proteins` | none | comma-separated IDs (overrides survey) |
| `--helix_min` / `--helix_max` | `0.25` / `0.85` | DSSP helix band to select |
| `--max_proteins` | `8` (workflow passes 40) | cap proteins |
| `--site` | `trunk` | `trunk` (s_trunk @ L47) or `diffusion` (token repr @ L22) |
| `--concept` | `helix` | direction key: `helix`, `helix_sae`, `helix_sae_f1set`, `strand` |
| `--diffusion_multilayer` | off | inject at ALL diffusion layers (E2b), not just L22 |
| `--sampling_steps` | `50` | steering diffusion steps |
| `--directions` | config default | override npz path |
| `--output_dir` | `outputs/probe_batch` | jsonl output dir |

Doses `k`: `MULTIPLIERS = (1, 2, 4, 8, 16)`. Per protein the additive strength is
`alpha = k * mean|(x-mean).u|` measured on the baseline `s_trunk` (or diffusion repr). The random
direction uses `rand_seed = i` (protein index).

### Step 6 - E2: diffusion single-layer steering

Same script, `--site diffusion`. Injects the `@diffusion` direction at DiffusionTransformerLayer
22 only (fires once per sampling step). Requires `helix@diffusion` in the npz.

```bash
python scripts/probe_batch.py --from_survey outputs/select/helix_survey.json \
    --helix_min 0.25 --helix_max 0.85 --site diffusion \
    --sampling_steps 50 --max_proteins 40 --output_dir outputs/probe_batch
```

### Step 6b - E2b: diffusion multi-layer steering

`--site diffusion --diffusion_multilayer` injects at every diffusion layer each sampling step
(single-layer diffusion steering is likely underpowered per DiT steering literature). Separate
output dir so it does not clobber the single-layer file.

```bash
python scripts/probe_batch.py --from_survey outputs/select/helix_survey.json \
    --helix_min 0.25 --helix_max 0.85 --site diffusion --diffusion_multilayer \
    --sampling_steps 50 --max_proteins 40 --output_dir outputs/probe_batch_multilayer
```

### Step 7 - E3: combined multi-site ablation (necessity)

`scripts/probe_combined.py` ablates (`alpha=1`, mean-centred projection) the helix direction at
the trunk conditioning ALONE, the diffusion module ALONE, and BOTH at once in a single forward
(`register_interventions`), measuring DSSP helix change. Tests whether necessity emerges only when
a redundantly encoded feature is removed everywhere. Requires `helix@diffusion` in the npz.

```bash
python scripts/probe_combined.py --from_survey outputs/select/helix_survey.json \
    --helix_min 0.35 --helix_max 0.95 --sampling_steps 50 --max_proteins 40 \
    --output_dir outputs/probe_combined
```

| flag | default | meaning |
|---|---|---|
| `--helix_min` / `--helix_max` | `0.35` / `0.95` | needs helix present to remove (most-helical first) |
| `--max_proteins` | `8` (workflow passes 40) | cap proteins |
| `--sampling_steps` | `50` | diffusion steps |

Sites: `trunk_only`, `diffusion_only`, `both`.

### Step 8 - Orchestration

`run_steering_analysis.sh` runs steps 3-7 (build_inputs, survey, E1, E2, E2b, E3, E5 for each SAE
variant present in the npz) on an EC2 GPU box, each stage failure-isolated with `aws s3 sync` after
each, resumable from the S3 survey cache. Launched via `boltz_sae_server`'s
`deploy_per_layer.ps1` (`-Config confgs/boltz_causal.json -WorkflowScript
run_steering_analysis.sh -StackBaseName boltz-steer -SelfDestruct`). Key env tunables:
`MAX_PROTEINS=40`, `SAMPLING_STEPS=50`, `SURVEY_STEPS=30`, `HELIX_MIN/MAX=0.25/0.85`,
`RUN_DIFFUSION=1`, `RUN_COMBINED=1`.

---

## 3. Steering math

Source: `boltz_causal/directions.py`, `boltz_causal/steering.py`, `boltz_causal/hooks.py`.

Given an activation tensor `x` (shape `(..., N, D)`, `N` = residues, `D` = features), a unit
direction `u` (`(D,)`), and the probe/SAE feature mean `mean` (`(D,)`):

- **Ablation (necessity / control):**
  ```
  x' = x - alpha * ((x - mean) . u) * u
  ```
  Mean-centring is essential. A few activation dims have huge magnitude, so `u.mean` dominates
  `u.x`; a raw projection would remove a constant offset (a random vector does the same - no
  specificity), not the concept. Mean-centring matches the probe/SAE geometry (the probe logit is
  `d_raw . (x - mean)`). `alpha=1` fully removes the component; `alpha in (0,1)` is graded.
- **Addition (sufficiency):**
  ```
  x' = x + alpha * u
  ```

- **Per-protein dose calibration.** Each protein has its own activation scale, so on the baseline
  pass `capture_scale` records `proj = mean |(x - mean) . u|` (the concept fluctuation scale, not
  the huge-dim offset) at the target site. The additive dose is then `alpha = k * proj` for
  `k in {1,2,4,8,16}`, and results aggregate by `k`. Using a multiple of `proj` (rather than of
  the full RMS norm) keeps the nudge on-manifold - full-RMS additions collapse pLDDT.

- **Injection sites.**
  - `trunk_output` (target = `s_trunk`): monkeypatch `structure_module.sample` to transform the
    `s_trunk` kwarg the diffusion module reads (fork `model.py:352`). This is the L47 conditioning.
  - `trunk_depth` (target = intermediate PairformerLayer): forward-hook on `PairformerLayer_{L}`
    output `(s, z)`, perturbing `s` only; can be pinned to the final recycle
    (`trunk_depth_last_recycle_only=True`).
  - `diffusion` (target = DiffusionTransformerLayer output `a`): forward-hook at
    `DIFFUSION_PROBE_LAYER = 22` (the layer the diffusion direction was fit on), fires once per
    sampling step; `layer=None` = all diffusion layers (E2b).

  Architectural note: in the fork's `PairformerLayer`, `z` (pair) updates only from `z`, and `s`
  (single) updates from `z`, so `s` never feeds `z`. Single-rep steering therefore reaches the
  output only through the final `s_trunk` conditioning; it cannot directly move pairwise geometry.

### Critical controls

1. **Matched diffusion noise.** `torch.manual_seed(cfg.seed)` (and `cuda.manual_seed_all`) is
   re-set immediately before EVERY prediction (baseline and every condition), so all conditions
   denoise from *identical* noise. Without this the per-condition RNG advance means each condition
   draws independent noise and the "effect" is confounded with a different noise sample. (This was
   a real bug that invalidated earlier numbers.)
2. **Matched-norm random direction.** For every additive dose the concept direction is paired with
   a random unit direction scaled to the same raw-space norm and mean-centred identically. The
   reported effect is the *paired* concept-add minus random-add. A real effect must beat this
   matched-norm baseline (specificity).
3. **pLDDT as on-manifold monitor.** Mean CA pLDDT is tracked for every condition. If a dose pushes
   the representation off-manifold, pLDDT collapses; a valid steering effect must hold while pLDDT
   stays stable.

---

## 4. Model and run parameters

From `boltz_causal/config.py` (`ExperimentConfig`) and the workflow. **The steering runs override
`sampling_steps` to 50** (survey uses 30) via the `--sampling_steps` CLI flag.

| parameter | value | notes |
|---|---|---|
| `recycling_steps` | `1` | matches the recycle the directions were fit at (final recycle == rec1 == fit space) |
| `sampling_steps` | `50` (steering) / `30` (survey) | config default is 200; runs pass `--sampling_steps` |
| `step_scale` | `1.638` | diffusion step scale |
| `diffusion_samples` | `1` | one sample per prediction |
| `seed` | `0` | re-seeded before every prediction (control 1) |
| `accelerator` / `devices` | `gpu` / `1` | |
| `use_msa_server` | `False` | offline; MSAs fetched in step 3 |
| `DIFFUSION_PROBE_LAYER` | `22` | diffusion injection/fit layer |
| checkpoint | `boltz1_conf.ckpt` | downloaded into `~/.boltz` if `checkpoint_path` empty |
| model fork | `nutz_and_boltz` (`feature/update-boltz`) + TensorLens | |
| instance | `g6.2xlarge`, region `eu-west-2` | EC2 |
| disulfide bond cutoff | `2.5` A | baseline "formed" filter (disulfide arm) |

`n_recycles = recycling_steps + 1 = 2` is passed to the hook registrar so depth ablation can be
pinned to the final recycle pass.

---

## 5. Results

All headline numbers are **held-out** (direction fit on the 389 TRAIN proteins, steering measured
on the disjoint HELD-OUT proteins), aggregated over the DSSP helix band. Unless noted, the effect
size is the **paired concept-add minus random-add at dose k=16** helix fraction. `SE` = standard
error of the paired difference.

| Exp | Direction / site | Effect (paired, k=16) | Verdict |
|---|---|---|---|
| **E1** | probe helix, trunk (`s_trunk` @ L47) | **+0.018 +/- 0.005** helix fraction (3.5 SE) | sufficiency: specific, monotone in dose, on-manifold (pLDDT ~80) |
| **E5** | SAE latent 1987 (F1 0.72), trunk | **+0.012 +/- 0.005** | sufficiency: specific + monotone, slightly WEAKER than the probe |
| **E2** | probe helix, diffusion single-layer (L22) | ~+0.002 (ns) | null |
| **E2b** | probe helix, diffusion all-layers | technical failure (NaN) | untested (over-injection) |
| **E3** | helix ablation: trunk / diffusion / both | all ~0.000 | necessity: NULL - not necessary at single-rep level |

### E1 - trunk probe direction (sufficiency)

- Paired concept-add minus random-add at k=16: **+0.018 +/- 0.005** helix fraction (~3.5 SE).
- Monotone in dose (larger `k` -> larger effect), and opposite sign to the matched-norm random
  vector (specificity holds; random-add ~0).
- pLDDT stays ~80 across doses -> the steer stays on-manifold.
- **Cross-reference (parallel run).** This +0.018 is the 34-protein subset on the 0.25-0.85 helix
  band from the early single-instance run. The later 6-concept parallel run (section 7) fits the
  identical probe but steers **all 97** held-out proteins over the FULL 0-1 gradient and reports
  **+0.039 [+0.025, +0.056]** for helix. The difference is not a model discrepancy: the diffusion
  noise is seed-matched and deterministic (control 1), so it is entirely the protein set (the full 97
  include many more low-pLDDT proteins, which steer much more strongly) plus the per-run re-drawn
  matched-norm random control (`random_like` reseeded by protein index). See section 7 for the full
  multi-concept table and the corrected numbers.
- **Confidence dependence.** The effect is ~4x larger in low-pLDDT proteins (**+0.038**, n=10,
  baseline pLDDT < 75) than high-pLDDT (**+0.010**, n=24). `corr(d_helix, baseline pLDDT) = -0.48`;
  `corr(d_helix, baseline helix) ~ 0`. Interpretation: steering bites hardest where fold
  commitment is weak, consistent with the paper's claim.

### E5 - trunk SAE latent (sufficiency, interpretable)

- Top-1 helix-F1 latent (latent **1987**, F1 = **0.723**) decoder column as the direction.
- Paired effect at k=16: **+0.012 +/- 0.005** - specific and monotone, but slightly WEAKER than
  the supervised probe. The interpretable single feature is causally *sufficient*, not superior:
  it recapitulates the probe effect.

### E2 / E2b - diffusion localization

- **E2 (single-layer, L22):** null (~+0.002, ns). Single-layer diffusion steering is likely
  underpowered.
- **E2b (all diffusion layers):** technical failure - over-injection at every layer drives NaN.
  Multi-layer diffusion causality remains **untested**; a follow-up needs per-step or late-step-only
  directions and a gentler multi-layer dose.

### E3 - combined ablation (necessity)

- trunk-only, diffusion-only, and both-at-once helix ablation all give mean `d_helix ~ 0.000`
  (RMSD small), even on 0.75-0.95-helical proteins. Helix is robustly reconstructed regardless.
- Verdict: helix is **NOT necessary** at the single-representation level - it is redundantly
  encoded. (This also confirmed that earlier large negative "ablation" effects were off-manifold
  offset removal, fixed by mean-centring.)

### Bottom line

Trunk single-representation helix is causally **SUFFICIENT** (specific, dose-dependent,
on-manifold, and strongest where confidence is low) but **NOT NECESSARY** (single-site ablation
does nothing; the feature is redundantly maintained). The interpretable SAE latent recapitulates
the probe's sufficiency effect, slightly weaker.

---

## 6. Data and artifact locations

**Directions npz** (fit on TRAIN split):
`s3://aas-processed-data-us/SwissProtAnnotation5/causal/disulfide_helix_directions.npz`
(local: `directions/disulfide_helix_directions.npz`). Keys include
`helix@trunk_L47`, `helix@diffusion`, `strand@trunk_L47`, `helix_sae@trunk_L47`,
`random@trunk_L47`, each with a `.mean` companion; per-key fit metadata in `__meta__`.

**Results** under `s3://activations-at-scale-artefacts/boltz-causal/steering/outputs/`:

| file | content |
|---|---|
| `select/helix_survey.json` | baseline DSSP helix + pLDDT per protein (the gradient) |
| `probe_batch/batch_trunk.jsonl` | E1 (trunk probe) per-protein rows |
| `probe_batch/batch_diffusion.jsonl` | E2 (diffusion single-layer) rows |
| `probe_batch_multilayer/batch_diffusion.jsonl` | E2b (all diffusion layers) rows |
| `probe_batch_helix_sae/batch_trunk.jsonl` | E5 (SAE top-1 latent, trunk) rows |
| `probe_batch_helix_sae_f1set/batch_trunk.jsonl` | E5 (SAE F1-above-null set) rows, if produced |
| `probe_combined/combined.jsonl` | E3 per-site ablation rows |

The aggregated specificity curves are printed to the stage logs (`~/workflow_layer_0.log` on the
instance), not to a file.

### jsonl row schema

`probe_batch.py` rows (one per condition per protein):

| field | meaning |
|---|---|
| `protein` | UniProt ID |
| `condition` | e.g. `baseline`, `helix_add_k16`, `rand_add_k4`, `helix_ablate_a1` |
| `kind` | one of `helix_add`, `rand_add`, `helix_ablate`, `rand_ablate` (absent on the baseline row) |
| `mult` | dose multiplier `k` |
| `d_helix` | helix fraction change vs baseline (the effect size) |
| `helix` | absolute DSSP helix fraction under the condition |
| `base_helix` | baseline DSSP helix fraction |
| `plddt` | mean CA pLDDT under the condition (on-manifold monitor) |
| `base_plddt` | baseline mean CA pLDDT |
| `ca_rmsd` | Kabsch CA-RMSD to baseline (A) |

`probe_combined.py` rows use `site` (`baseline`, `trunk_only`, `diffusion_only`, `both`) in place
of `condition`/`kind`/`mult`, plus `helix`, `d_helix`, `ca_rmsd`, `plddt`.

---

## 7. Secondary-structure steering: probe vs SAE, all three states (parallel run)

Section 5 established sufficiency for a single concept (helix) with a single instance running every
condition sequentially. This section extends the test to all three DSSP secondary-structure states
(**helix, strand, coil**), each fit two ways (supervised **probe** and unsupervised **single SAE
latent**), and reports specificity + a full confusion matrix. It is a distinct run and is reported
separately; the earlier E1/E5 numbers in section 5 are not overwritten.

### 7.1 Method delta from the section-5 run

The earlier design ran one instance through every concept in sequence (~24h wall-clock). This run
**shards one concept per GPU instance** so 6 instances finish in roughly 1/6 the wall-clock.

- **Sharding.** `run_steering_shard.sh` (repo root) selects a concept from `SHARD` (equivalently
  `LAYER`, the env var `boltz_sae_server`'s `server_setup.sh` passes through). The 6 shards are
  `SHARD 0..5 = helix / helix_sae / strand / strand_sae / coil / coil_sae`. Launch with
  `deploy_per_layer.ps1 ... -WorkflowScript run_steering_shard.sh -LayerNumbers 0,1,2,3,4,5`.
- **Full gradient.** Unlike section 5's 0.25-0.85 helix band, each shard steers over the FULL
  gradient (`--helix_min 0.0 --helix_max 1.0`) across all **97** held-out proteins, so the confusion
  matrix is checkable for every concept regardless of its baseline SS content.
- **Per-concept S3 prefixes, run-tagged.** Each shard writes to its own date-tagged prefix
  `s3://activations-at-scale-artefacts/boltz-causal/steering/shards/20260713/<concept>/outputs/`, so
  shards cannot collide and successive runs cannot alias stale files. A 7th shard, `random_null`,
  holds the random-direction null distribution (19 matched-norm directions x 97 proteins @ k=16).

**Fixes recorded on this run:**

1. **Unique per-run / per-shard S3 prefix.** `aws s3 sync` skips files whose size is unchanged. On a
   reused prefix this had kept **stale CIFs** on disk that were decoupled from the freshly written
   `jsonl` (the structure files and the metrics disagreed). A distinct prefix per run (and per shard)
   removes the aliasing.
2. **Shared survey cache.** The slow baseline DSSP survey (step 4) is run once and cached at
   `.../steering/survey_cache/helix_survey.json`; every shard restores it and skips the survey.
3. **Export merge asymmetry.** `export_probe_direction.py` **OVERWRITES** the npz (a single-concept
   export clobbers the others), so per-concept probe exports must be **merged from S3** before
   upload. `export_sae_direction.py` reloads and **merges** existing keys, so SAE exports are safe to
   append. This asymmetry is the reason probe directions are re-merged and SAE ones are not.

### 7.2 Directions

- **Probe** (per concept): supervised L2 logistic direction, `coef_ / scale_` mapped back to raw
  activation space and mean-centred, exactly as section 2(a) - one each for helix / strand / coil.
- **SAE** (per concept): the single top-F1 latent's decoder column, as section 2(b). The latents and
  their **SAE-1feat F1** (single best latent) are:

  | concept | SAE latent | SAE-1feat F1 |
  |---|---|---|
  | helix | 1987 | 0.72 |
  | coil | 1144 | 0.71 |
  | strand | 1972 | 0.47 |

  **What this F1 is (clarification).** This is the paper's **SAE-1feat** score: the single best latent,
  taken as the max over 2048 latents x percentile thresholds, i.e. a *monosemanticity* statistic
  computed by `benchmark_f1.score_best_f1_per_concept`. It is LOWER than, and distinct from, the
  paper's held-out **probe-raw** F1 (0.79-0.90, a *decodability* statistic). The gap is expected -
  one interpretable latent should not match a full supervised probe - it is **not a discrepancy**.

### 7.3 Results

All numbers are the **full 97 held-out** set, paired **concept-add minus random-add at dose k=16**,
on-manifold (pLDDT stable across doses). 95% CI = paired-protein bootstrap (B=10000).

> **`d_target` bug (must read).** The stored `d_target` field is correct only for the probe shards.
> For the SAE shards it defaulted to `d_helix` (verified: `d_target == d_helix` in 1164/1164 rows for
> helix_sae, strand_sae AND coil_sae). So `strand_sae`/`coil_sae` own-state effects must be read from
> `d_strand`/`d_coil`, never `d_target`. All numbers below are recomputed from the correct per-SS
> fields (`scripts/analyze_full97.py`). This is why the earlier `strand_sae +0.032` and
> `coil_sae -0.007` numbers were wrong: the first was the helix effect mislabeled, the second was noise.

**Table 1 - specificity on the concept's OWN DSSP state** (from `d_helix`/`d_strand`/`d_coil`):

| Direction | own-state effect (k=16) | 95% CI | verdict |
|---|---|---|---|
| coil (probe) | **+0.061** | [+0.045, +0.079] | steers (strongest lever) |
| helix (probe) | +0.039 | [+0.025, +0.056] | steers |
| helix_sae | +0.029 | [+0.020, +0.038] | steers |
| strand (probe) | -0.001 | [-0.003, +0.001] | null |
| strand_sae | +0.001 | [-0.001, +0.003] | null |
| coil_sae | +0.001 | [-0.005, +0.007] | null |

**Table 2 - confusion matrix**, mean concept-add effect at k=16 (raw). Rows = steered concept, columns
= measured DSSP change. Diagonal (own state) is bold. (`steering/data/confusion_full97.csv` also carries
the random-controlled *paired* version.)

| steered \ measured | d_helix | d_strand | d_coil |
|---|---|---|---|
| helix | **+0.033** | -0.000 | -0.033 |
| helix_sae | **+0.026** | -0.001 | -0.026 |
| strand | -0.022 | **+0.001** | +0.021 |
| strand_sae | +0.026 | **+0.002** | -0.027 |
| coil | -0.065 | -0.000 | **+0.066** |
| coil_sae | -0.001 | +0.000 | **+0.001** |

### 7.4 Findings

1. **Helix and coil steer specifically and trade off.** Steering helix up drives coil down and vice
   versa (Table 2: helix row +0.033 / -0.033; coil row -0.065 / +0.066). Coil is the **strongest
   single-representation causal lever** (+0.061 on its own state). pLDDT stays stable, so this is
   a genuine helix<->loop conversion, not a confidence collapse.
2. **Strand is NOT inducible by single-rep steering.** The strand probe does not move its own state
   (-0.001, null) and `strand_sae` actually behaves as a weak *helix* lever (+0.026 d_helix / -0.027
   d_coil / +0.002 d_strand). Beta sheets are non-local - a strand needs partner strands to pair with -
   so a single-residue, single-representation nudge cannot conjure one. This is consistent with
   strand's low SAE-1feat F1 (0.47): the strand concept is distributed, not carried by one latent.
3. **The supervised probe beats the single SAE latent as a causal lever in every case.** For helix
   both work but probe > SAE (+0.039 vs +0.029); for coil the probe steers strongly (+0.061) while
   `coil_sae` is a **dead lever** (+0.001, null); the `strand_sae` latent does not move strand either.
   Monosemanticity (a single interpretable latent) is therefore **not** the same as maximal causality:
   the distributed supervised direction is the better lever, and for coil/strand the single latent
   carries no usable causal signal at all.
4. **Every working lever clears the random-direction null; the nulls sit inside it.**
   (`random_null` shard = 19 matched-norm random directions x 97 proteins @ k=16; fig
   `null_distribution_full97`.) Own-SS population effect as a z-score against the 19-direction null:
   helix +6.2z, helix_sae +5.0z, coil +11.5z (all above all 19 random directions, 100th percentile);
   strand +0.3z, strand_sae +1.0z, coil_sae -0.1z (all inside the null cloud). With 19 directions the
   empirical one-sided p floor is 1/20 = 0.05, so z is the informative statistic.
5. **Even on strand-rich proteins, strand steering builds no strand.** The held-out set is strand-poor
   (max baseline strand 0.47; 10 proteins > 0.30, 2 > 0.40). Restricting to those 10, strand-add moves
   d_strand ~0 (paired -0.0006) while converting helix->coil (e.g. P18467 d_helix -0.044 / d_coil
   +0.051). So the strand result is not merely an under-powered benchmark: the single-rep strand nudge
   *cannot* build strand, only push helix->coil. (`steering/data/high_strand_full97.csv`.)
6. **Bidirectional example (protein A6NI15).** A single held-out protein steered both ways:
   baseline helix 0.528 / coil 0.472 -> helix-steered 0.705 / 0.295 -> coil-steered 0.264 / 0.736
   (41 residues gained helix under helix-steer; 51 gained coil under coil-steer). CIFs verified
   current via DSSP-vs-jsonl match. See `structure_bidir_3d_chimera`.
7. **The trade-off is geometric: a dominant helix<->coil axis.** The trunk probe directions are
   strongly anti-parallel for helix vs coil (cosine **-0.69**) but near-orthogonal to strand
   (helix-strand -0.44, strand-coil -0.24). So helix and coil occupy roughly opposite ends of ONE
   dominant single-representation axis; steering along it converts between the two states (explaining
   Table 2's helix<->coil anti-correlation and the A6NI15 bidirectional result). Strand lies off this
   axis, matching its non-inducibility. NOTE: helix and coil are separately-fit directions that
   happen to be anti-parallel - not the same vector negated.

### 7.5 Figures

Under `steering/figures/`:

| figure | content |
|---|---|
| `fig7_confusion_matrix` | Table 2 as a heatmap (steered concept x measured DSSP change) |
| `fig9_confidence_by_pLDDT` | low vs high pLDDT bars + per-protein scatter |
| `null_distribution_full97` | real concept effect vs the 19-direction random null |
| `structure_bidir_3d_chimera` | A6NI15 Chimera ribbons, coil-steered / baseline / helix-steered |

Plus the two steering supplementary panels from the earlier single-instance run
(`make_paper_plots.py`): `fig1_sufficiency_dose_response`, `fig4_necessity_ablation`.

Table 1's own-state effects and their 95% CIs are **CSV-only** deliverables — no figure is
rendered for them. `scripts/make_tables.py` joins them into the manuscript's `tab:steering`
(LaTeX + CSV under `steering/data/tables/`). They come from
`data/own_state_effect_full97.csv` (`analyze_full97.py`)
and `data/fig10_bootstrap_ci.csv` (`make_bootstrap_ci.py`). `analyze_full97.py` also writes
`confusion_full97.csv`, `plddt_dependence_full97.csv` and `high_strand_full97.csv`.

### 7.6 Data locations

Per-concept shard outputs (run-tagged prefix, no cross-shard or cross-run collision):

```
s3://activations-at-scale-artefacts/boltz-causal/steering/shards/20260713/<concept>/outputs/
    probe_batch_<concept>/batch_trunk.jsonl
```

for `<concept>` in `helix, helix_sae, strand, strand_sae, coil, coil_sae, random_null` (1261 rows
each for the 6 concepts; 3104 for random_null). Local mirror: `shard4_20260713/*.jsonl` (re-pull from
S3 - local scratchpad does not persist across sessions). Row schema is identical to section 6's
`probe_batch.py` schema, with `d_strand` / `d_coil` (and `strand` / `coil` absolute fractions)
alongside `d_helix` so the confusion matrix is recoverable from any shard's rows; `random_null` rows
add `rand_idx` (1..19) and use `kind = rand_null`. **Read own-state effects from `d_helix`/`d_strand`/
`d_coil`, not `d_target` (bugged for SAE shards - see 7.3).**
