# Data & artifact access

This repository ships the **small summary/result data** needed to regenerate every figure
(`decodability/data/`, `steering/data/`; total ≈ 15 MB). The **large raw artifacts** — per-residue
Boltz-1 activations and the trained SAE checkpoints — are hosted externally and pulled on demand by
the analysis scripts. Some are public; some require credentials we issue on request. This document is
the single source of truth for where they live and how to get them.

| Artifact | Size | Host | Access | Pulled by |
|---|---|---|---|---|
| Per-residue Boltz-1 activations (train + eval) | ~TB / hundreds of GB | Cloudflare R2, backed by AWS S3 | **credentials on request** | `decodability/get_activations.py`, `stage_diffusion_data.py` |
| Trained SAE checkpoints (Pairformer trunk) | ~GB | HuggingFace | **public, no auth** | `decodability/layer_analysis_utils.py:download_run_checkpoint` |
| Trained SAE checkpoints (diffusion module) | ~GB | AWS S3 → local cache | **credentials on request** | `decodability/stage_diffusion_data.py` |
| Steering directions (`.npz`) | 40 KB | committed (`steering/directions/`) | **in this repo** | `boltz_causal.directions` |

Only the trunk SAE checkpoints and the committed steering directions need no arrangement with us.
**Every figure and table in the manuscript can be rebuilt from what is already committed here** (§3) —
the credentialed artifacts are needed only to redo the upstream activation extraction, SAE training,
or benchmark sweeps from scratch.

---

## 1. Activations — Cloudflare R2 (credentials on request)

The activations live in AWS S3 and are served to external users through a Cloudflare R2 mirror. The
R2 bucket is **not world-readable**: access is granted per requester. Request it from the
corresponding author of the accompanying paper, stating which layers and recycles you need — the full
set is hundreds of GB, so a scoped subset is usually faster for everyone.

You will be issued three things: the **R2 endpoint** (or the account ID it is derived from), an
**access key pair**, and the **bucket name** to pass as `--bucket`.

### Credentials

Copy `decodability/.env.example` to `decodability/.env` and fill in the values you were issued —
these are *our* R2 credentials scoped to you, not credentials from your own Cloudflare account:

```dotenv
CF_ACCOUNT_ID=          # account ID we give you; auto-constructs the endpoint URL
R2_ACCESS_KEY_ID=       # scoped access key we issue
R2_SECRET_ACCESS_KEY=   # its secret
# or, instead of CF_ACCOUNT_ID, the endpoint directly:
# R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
```

`get_activations.py` resolves the endpoint from `R2_ENDPOINT_URL`, else from `CF_ACCOUNT_ID`
(→ `https://<account>.r2.cloudflarestorage.com`), and authenticates with the `R2_*` keys. When those
are set it prints *"Using Cloudflare R2 storage backend"* and every download goes through R2. With no
credentials configured it logs a warning and every request fails — there is no anonymous access path.

> **Bucket / key layout.** The R2 bucket mirrors the S3 layout key-for-key, so the same `--bucket`
> and manifest arguments work against either backend. For reference, the internal S3 layout is:
>
> | content | S3 bucket | notes |
> |---|---|---|
> | training activations (84k proteins) | `boltz-1-activations` | — |
> | eval activations (SwissProt set) | `swissprot-annotated-proteins-activations` | prefix `SwissProtAnnotation5/` |
>
> Both are private. Use the R2 bucket name you were issued, not these.

### Download example

`get_activations.py` builds its CLI from `download_activations()` via `tapify`, so the flags are that
function's parameters: `--bucket`, `--manifest`, `--layer_type` (`pairformer` | `diffusion`),
`--layer`, `--rec`, `--out`. All but `--out` are required. Substitute the bucket name you were issued:

```bash
cd decodability
# Pairformer trunk, layer 20, recycle 1, for the SwissProt eval set:
python get_activations.py \
    --bucket THE_BUCKET_YOU_WERE_ISSUED \
    --manifest inputs/SwissProtproteins.txt \
    --layer_type pairformer \
    --layer 20 --rec 1 \
    --out downloads_layer20
```

> **Why a manifest is required.** The R2 mirror is fronted by Cloudflare Sippy, which serves `GET`
> but not `LIST`. So a client cannot enumerate the bucket; it must be told which keys to fetch.
> `generate_manifest.py` produces these listings and needs AWS access, so manifests are generated on
> our side and shipped in `decodability/inputs/` (see the files listed there) or sent with your
> credentials.

Per-protein key layout is `{protein_prefix}/{layer_folder}/output_{rec}.npz`. Activations land in the
`downloads_layer{N}/` (Pairformer) and `downloads_diffusion_rec{R}/` (diffusion) directories that the
benchmark and probe scripts expect (see `docs/DECODABILITY_SCRIPT_MAP.md`).

---

## 2. SAE checkpoints

The **Pairformer trunk** SAEs are public and need no credentials — released in the HuggingFace
collection **<https://huggingface.co/collections/evolve-away/boltz-saes>**. `layer_analysis_utils.py`
downloads them automatically (no `huggingface_hub` or GPU required — it fetches with `requests`):

| stack | recycle | Location | Access |
|---|---|---|---|
| Pairformer trunk | rec 1 (main text) | `evolve-away/Boltz1-SAEs-L2` | public |
| Pairformer trunk | rec 0 | `evolve-away/Boltz1-SAEs-L2-rec0` | public |
| Diffusion module | steps 0/50/199 … | `s3://boltz-saes-l2/diffusion/rec{R}/layer{N}/` | private — credentials on request |

The **diffusion-module** SAEs are not published: they sit in a private S3 bucket and are staged by
`stage_diffusion_data.py`, which uses the standard boto3 credential chain. Request access alongside
the activations (§1).

Recipe: TopK SAE, `k=256`, 2048 latents, trained on **demeaned** activations, L2 = 3e-3, 3 seeds,
500k steps.

```python
# decodability/layer_analysis_utils.py
from layer_analysis_utils import download_run_checkpoint, repo_for_rec
ckpt = download_run_checkpoint(layer=47, rec=1)   # pulls from evolve-away/Boltz1-SAEs-L2
```

Diffusion SAE checkpoints + activations are staged together:

```bash
cd decodability
python stage_diffusion_data.py --layers 4,14,22 --recs 0,50,199 --seeds 1,2,3 --n_proteins 160
# needs the issued AWS creds (env vars or ~/.aws/credentials) for s3://boltz-saes-l2/diffusion/...
```

---

## 3. What is already committed (no download needed)

- `decodability/data/` — all benchmark aggregates (`*_agg.json`), per-seed SS benchmarks,
  precision/recall CSVs, bootstrap CIs (`concept_f1_ci.csv`), label prevalence
  (`concept_label_stats.csv`), and the amino-acid sanity JSONLs. Enough to rebuild every
  decodability figure and table.
- `steering/data/shard4_20260713/` — the 6-concept + `random_null` steering shard rows
  (`*.jsonl`) and the A6NI15 CIFs.
- `steering/data/section5_20260708/` — the section-5 single-instance rows behind the dose-response
  and necessity-ablation figures (`probe_batch/batch_{trunk,diffusion}.jsonl`,
  `probe_batch_helix_sae/batch_trunk.jsonl`, `probe_combined/combined.jsonl`; 351 KB total).
- `steering/data/*.csv` — frozen plotted values for the steering figures.
- `steering/directions/disulfide_helix_directions.npz` — the fitted steering directions. Despite
  the filename it holds the helix / strand / coil probe and SAE directions used in the manuscript.

Together these are enough to rebuild **every** figure and table except the Chimera structure panel,
which needs a GUI render step (see the top-level `README.md`).

### Not committed (build locally)

- **`processed_swissprot_a5/`** — the processed SwissProt shards
  (`shard_*/annotations.npz`, `shard_*/sequences.fasta`, `concept_vocabulary.json`). Most of the
  label and benchmark pipeline reads these. Build them with:

  ```bash
  cd decodability
  python swissprot_annotation_pipeline.py \
      --input_ids inputs/swissprot_a5_ids.txt \
      --output_dir processed_swissprot_a5
  ```

  This hits the UniProt REST API and takes a while. Two scripts that need only the *sequences* —
  `check_probe_homology.py` and `build_amino_acid_concepts.py` — fall back to the committed
  `decodability/inputs/swissprot_a5.fasta` (the same 5,000-protein pool) when the shards are
  absent, so the leakage control and the AA concept set run without this step.
