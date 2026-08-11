#!/usr/bin/env python3
"""Fetch per-protein Boltz inputs (YAML + MSA) from S3 for the causal runs.

The intervention runs **rerun Boltz from scratch** on each sequence (no activation staging), so all
they need is one YAML + its MSA per protein. Both already exist for the SwissProt validation set:

    yaml:  s3://aas-processed-data-us/SwissProtAnnotation5/yaml/{id}.yaml
    msa:   s3://aas-processed-data-us/SwissProtAnnotation5/msa/{...}.a3m

For each protein this script:
  1. downloads ``{id}.yaml``,
  2. reads its ``msa:`` reference(s), downloads the matching ``.a3m`` from the msa prefix,
  3. rewrites ``msa:`` to the **absolute** local a3m path (process_inputs resolves it relative to
     cwd, so absolute is safest), and writes the YAML into the runner's input dir.

Protein set defaults to the keys of ``disulfide_pairs.json`` (the exact run set). Requires boto3 and
AWS creds (``~/.aws/credentials``). Run offline afterwards — the runner uses ``use_msa_server=False``.

    python scripts/build_inputs.py --pairs disulfide_pairs.json --out-dir inputs
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import yaml

S3_YAML_PREFIX = "s3://aas-processed-data-us/SwissProtAnnotation5/yaml"
S3_MSA_PREFIX = "s3://aas-processed-data-us/SwissProtAnnotation5/msa"


def _split_s3(uri: str) -> tuple[str, str]:
    rest = uri.removeprefix("s3://")
    bucket, _, key = rest.partition("/")
    return bucket, key


class S3:
    """Thin boto3 wrapper (download + prefix-list) so we can fall back to a fuzzy MSA lookup."""

    def __init__(self) -> None:
        try:
            import boto3  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise SystemExit("boto3 required: pip install boto3 (and configure ~/.aws/credentials)") from e
        self.client = boto3.client("s3")

    def download(self, uri: str, dest: Path) -> bool:
        bucket, key = _split_s3(uri)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(bucket, key, str(dest))
            return True
        except Exception:  # noqa: BLE001
            return False

    def list_keys(self, prefix_uri: str) -> list[str]:
        bucket, prefix = _split_s3(prefix_uri)
        keys: list[str] = []
        token = None
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = self.client.list_objects_v2(**kw)
            keys.extend(obj["Key"] for obj in resp.get("Contents", []))
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return keys


def _fetch_first(s3: S3, uris: list[str], dest_dir: Path) -> Path | None:
    """Download the first URI that exists (S3 objects here are often gzipped, so we try .gz too).
    Returns the local path (still gzipped if the key was .gz)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    for uri in uris:
        if uri in seen:
            continue
        seen.add(uri)
        local = dest_dir / Path(uri).name
        if local.exists() or s3.download(uri, local):
            return local
    return None


def _read_maybe_gz(path: Path) -> str:
    """Read text from a plain or .gz file."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return fh.read()
    return path.read_text(encoding="utf-8")


def _plain(raw: Path, out_dir: Path) -> Path:
    """Materialise an un-gzipped copy of ``raw`` in ``out_dir`` (strips a .gz suffix)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if raw.suffix == ".gz":
        plain = out_dir / raw.with_suffix("").name  # strip .gz -> e.g. X.csv
        with gzip.open(raw, "rb") as f_in, open(plain, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        plain = out_dir / raw.name
        if raw.resolve() != plain.resolve():
            shutil.copy(raw, plain)
    return plain


def fetch_msa_csv(s3: S3, protein_id: str, entity: int, msa_dir: Path) -> Path | None:
    """Download the per-entity CSV MSA (``{id}_{entity}.csv.gz`` on S3), return a plain local .csv.

    Boltz's fork reads CSV MSAs (colabfold-style); the SwissProt pipeline stores one per protein
    entity as ``{id}_{entity}.csv.gz``. Mirrors aas-boltz-runner/prepare_yaml.py, which injects the
    csv path into ``sequences[i].protein.msa``.
    """
    names = [f"{protein_id}_{entity}.csv.gz", f"{protein_id}_{entity}.csv"]
    if entity == 0:  # some are stored without the _0 suffix
        names += [f"{protein_id}.csv.gz", f"{protein_id}.csv"]
    raw = _fetch_first(s3, [f"{S3_MSA_PREFIX}/{n}" for n in names], msa_dir / "_raw")
    return _plain(raw, msa_dir) if raw is not None else None


def build_one(s3: S3, protein_id: str, out_dir: Path, msa_dir: Path) -> str:
    """Download the YAML and inject the per-entity CSV MSA path(s). Returns a status string."""
    raw = _fetch_first(
        s3,
        [f"{S3_YAML_PREFIX}/{protein_id}.yaml.gz", f"{S3_YAML_PREFIX}/{protein_id}.yaml"],
        out_dir / "_raw",
    )
    if raw is None:
        return "no-yaml"
    doc = yaml.safe_load(_read_maybe_gz(raw))

    proteins = [e["protein"] for e in doc.get("sequences", []) if isinstance(e.get("protein"), dict)]
    if not proteins:
        return "no-protein-entity"

    # inject one CSV MSA per protein entity (matches prepare_yaml.py; monomers have entity 0 only)
    for i, prot in enumerate(proteins):
        csv = fetch_msa_csv(s3, protein_id, i, msa_dir)
        if csv is None:
            return f"no-msa(entity {i})"
        prot["msa"] = str(csv.resolve())  # absolute → resolves regardless of runner cwd

    (out_dir / f"{protein_id}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch per-protein Boltz YAML+MSA from S3.")
    ap.add_argument("--pairs", default="disulfide_pairs.json", help="run set = keys of this JSON")
    ap.add_argument("--proteins", default=None, help="alternative: newline/space-separated ID file")
    ap.add_argument("--out-dir", default="inputs")
    ap.add_argument("--msa-dir", default="inputs/msa")
    ap.add_argument("--limit", type=int, default=0, help="cap number of proteins (0 = all)")
    args = ap.parse_args()

    if args.proteins:
        ids = Path(args.proteins).read_text().split()
    else:
        ids = list(json.loads(Path(args.pairs).read_text()).keys())
    if args.limit:
        ids = ids[: args.limit]

    out_dir = Path(args.out_dir)
    msa_dir = Path(args.msa_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    s3 = S3()
    counts: dict[str, int] = {}
    for pid in ids:
        status = build_one(s3, pid, out_dir, msa_dir)
        counts[status.split("(")[0]] = counts.get(status.split("(")[0], 0) + 1
        print(f"{pid}: {status}")
    print(f"\nDone: {counts}  ->  {out_dir}")
    if counts.get("ok", 0) + counts.get("ok-no-msa", 0) == 0:
        raise SystemExit("No inputs built — check S3 prefixes / credentials.")


if __name__ == "__main__":
    main()
