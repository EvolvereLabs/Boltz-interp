# Mirroring the HF collection for double-blind review

The SAE checkpoints live in a public HuggingFace collection owned by the
`evolve-away` namespace. That namespace names us, so linking it from a
double-blind submission breaks anonymity. This note covers making an anonymous
mirror that reviewers can pull from.

`tools/mirror_hf_collection_anon.py` does the copying. Everything it *cannot* do
is listed under "Manual steps" — read that part; the script is the easy half.

## What the script does

For each `model` / `dataset` item in the source collection it:

1. Downloads a full snapshot.
2. Rewrites identifying strings in text files (`README.md`, `config.json`,
   `stats.jsonl`, …): `evolve-away` → the anonymous namespace,
   `Evolvere Biosciences` / `EvolvereLabs` → `Anonymous`, and the GitHub URL of
   this repo → a placeholder. E-mail addresses and W&B run URLs are redacted
   outright.
3. Scans the binaries (`.pt`, `.npy`, …) for the same strings and **reports**
   them without patching — rewriting bytes inside a pickle would corrupt the
   checkpoint.
4. Creates the target repo and uploads the scrubbed snapshot as **one fresh
   commit** authored by the anonymous account.
5. Rebuilds the collection on the anonymous account and adds each mirrored repo.

Step 4 is the reason this is a script and not `git clone && git push`: a git push
carries every original commit, and every commit carries a real name and e-mail.

## Running it

```bash
pip install "huggingface_hub>=0.34"

export HF_TOKEN_DST=hf_...          # WRITE token, from the anonymous account
# export HF_TOKEN_SRC=hf_...        # only if a source repo is private

# 1. Dry run: lists what would be mirrored, downloads nothing.
python tools/mirror_hf_collection_anon.py \
    --src-collection evolve-away/boltz-saes \
    --dst-namespace <anon-account> \
    --dry-run

# 2. For real.
python tools/mirror_hf_collection_anon.py \
    --src-collection evolve-away/boltz-saes \
    --dst-namespace <anon-account> \
    --extra-sub boltz-saes-l2=anon-sae-bucket \
    --execute
```

Notes on the invocation:

- **`--src-collection` wants the slug exactly as the browser shows it** after
  `/collections/`. HF slugs usually carry a trailing id
  (`evolve-away/boltz-saes-6712ab…`); if the short form 404s, copy the full one.
- **`--extra-sub OLD=NEW`** is repeatable, for anything the defaults miss. The
  internal bucket names (`boltz-saes-l2`, `aas-processed-data-us`,
  `activations-at-scale-artefacts`) are *not* substituted by default, because
  blindly rewriting a path can invalidate a recorded config — pass them
  explicitly once you have decided what they should read.
- Budget roughly **1.5 GB per SAE repo** (25 layers × 3 seeds × 18.9 MB), so
  ~3 GB of download and re-upload for the two current repos.
- The script refuses to run if `HF_TOKEN_SRC == HF_TOKEN_DST`, which would
  silently produce a mirror owned by the identified account.

## Manual steps the script cannot do

1. **Create the anonymous account.** Sign it up with an e-mail that is not a
   work address, and do not add it to the `evolve-away` organisation — HF shows
   org membership on the public profile, which would undo the whole exercise.
   Do not reuse a personal HF account either; its existing repos and likes are
   public.
2. **Deal with flagged binaries.** If the report says a `.pt` embeds a bucket
   path or `evolve-away`, the training config was pickled into the checkpoint.
   Re-save those checkpoints with the offending fields stripped, then re-run the
   mirror. Uploading them as-is leaks through a `strings` call on the download.
3. **Repoint the code you submit — and mind the repo *names*.**
   `decodability/layer_analysis_utils.py` hardcodes `evolve-away/Boltz1-SAEs-L2`
   and `…-rec0` in `REPO_ID` / `REC_REPO_IDS`, and the top-level `README.md`
   links the `evolve-away` collection. An anonymous code drop has to point at
   the mirror instead, or a reviewer following the SAE download deanonymises the
   submission in one click.

   A find-and-replace of the namespace alone is **not** enough. The collection
   holds three repos:

   | in the collection            | what the code asks for |
   | ---------------------------- | ---------------------- |
   | `Boltz1-SAEs-L2-rec1`        | `Boltz1-SAEs-L2`       |
   | `Boltz1-SAEs-L2-rec0`        | `Boltz1-SAEs-L2-rec0`  |
   | `Boltz1-SAEs-L2-Diffusion`   | (not referenced)       |

   The rec-1 names differ. `evolve-away/Boltz1-SAEs-L2` presumably still
   resolves because HF redirects a renamed repo from its old name — but the
   mirror is a *brand-new* repo called `…-rec1`, and new repos carry no such
   redirect. So `REPO_ID` and `REC_REPO_IDS[1]` need the name changed as well as
   the namespace, or the anonymous code 404s on every rec-1 download.

   Separately, `download_run_checkpoint` still raises
   `"there is no HF fallback"` for `layer_type != "pairformer"`, which is now
   stale: `Boltz1-SAEs-L2-Diffusion` exists. A reviewer trying to reproduce the
   diffusion results will hit that error even though the weights are published.
4. **Scrub `LICENSE`.** It reads `Copyright (c) 2026 Evolvere Biosciences`. Keep
   MIT, but the copyright line has to go for the review copy.
5. **Do not link back.** The anonymous repos must not reference the real GitHub
   repo, and the real repo should not gain a link to the anonymous mirror until
   after the review closes.

## Before you hand over the link

- Read the report's **NEEDS A HUMAN LOOK** section end to end.
- Open each mirrored repo's *Files* tab and its **commit history** — confirm one
  commit, authored by the anonymous account.
- `git log --format='%an %ae'` on a fresh clone of a mirrored repo should show
  only the anonymous identity.
- Check the model card renders without the org name, and that the collection
  title and description carry no author or affiliation.
- Confirm the repos are **public**; reviewers cannot read private ones.

## After the review

Delete the anonymous repos and collection, or leave them and add the real
attribution — but do not quietly convert the anonymous account into an
attributed one while other submissions still cite it.
