#!/usr/bin/env python3
"""Mirror a HuggingFace collection to an anonymous account for double-blind review.

Copies every model/dataset repo in a source collection into a target namespace,
scrubbing author-identifying strings on the way, then rebuilds the collection
itself on the target account.

Why not ``git clone`` + ``git push``: that preserves commit authorship -- real
names and e-mail addresses in every commit -- which defeats the point. This
script uploads each repo as a single fresh commit made by the anonymous token's
own account, so the mirrored history carries no author identity at all.

Usage -- dry run first; it lists what would happen and downloads nothing:

    pip install "huggingface_hub>=0.34"
    export HF_TOKEN_SRC=hf_...   # optional, only for private source repos
    export HF_TOKEN_DST=hf_...   # WRITE token for the anonymous account
    python tools/mirror_hf_collection_anon.py \
        --src-collection evolve-away/boltz-saes \
        --dst-namespace anon-submission-1234 \
        --dry-run

Re-run with ``--execute`` to create and populate the mirror. Read the
"NEEDS A HUMAN LOOK" section of the report before handing the link to anyone.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:  # pragma: no cover
    sys.exit("pip install 'huggingface_hub>=0.34' first")

# --- What counts as identifying -------------------------------------------- #
# A replacement of None means "substitute the target namespace". Order does not
# matter here: build_subs sorts longest-needle-first so that, say,
# "EvolvereLabs/Boltz-interp" is rewritten before the bare "Evolvere" matches.
DEFAULT_SUBS: list[tuple[str, str | None]] = [
    ("https://github.com/EvolvereLabs/Boltz-interp", "<anonymised: code repo>"),
    ("github.com/EvolvereLabs/Boltz-interp", "<anonymised: code repo>"),
    ("EvolvereLabs/Boltz-interp", "<anonymised: code repo>"),
    ("Evolvere Biosciences", "Anonymous"),
    ("EvolvereLabs", "Anonymous"),
    ("Evolvere", "Anonymous"),
    ("evolve-away", None),
]

# Pure identity leaks with no functional meaning: redacted outright in text files.
AUTO_REDACT_PATTERNS: list[tuple[str, str, str]] = [
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<anonymised: e-mail>", "e-mail"),
    (
        r"https?://[A-Za-z0-9.-]*wandb\.ai/[A-Za-z0-9_./-]+",
        "<anonymised: W&B run>",
        "W&B run URL",
    ),
]

# Leaks we report but refuse to auto-rewrite, because a blind substitution would
# corrupt a path or silently change a recorded result. Anything AUTO_REDACT
# already handled is gone by the time these run; they stay as a safety net.
FLAG_PATTERNS: list[tuple[str, str]] = [
    (r"s3://[A-Za-z0-9._/-]+", "internal S3 bucket path"),
    (r"r2://[A-Za-z0-9._/-]+", "internal R2 bucket path"),
    (r"https?://[A-Za-z0-9.-]*wandb\.ai/[A-Za-z0-9_./-]+", "W&B run URL (leaks entity)"),
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "e-mail address"),
    (r"arxiv\.org/abs/\d{4}\.\d{4,5}", "arXiv link (may deanonymise)"),
    (r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", "DOI (may deanonymise)"),
]
# Only these run against binary payloads; the e-mail/DOI/arXiv patterns produce
# constant false positives against packed float data.
BINARY_FLAG_PATTERNS = FLAG_PATTERNS[:3]

# Extensions we are willing to rewrite in place. Everything else is opaque
# bytes: scanned for leaks, never patched.
TEXT_SUFFIXES = {
    ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".py", ".sh",
    ".csv", ".tsv", ".cfg", ".toml", ".ini", ".rst",
}
# Binary payloads worth scanning; pickled configs routinely keep bucket paths.
SCAN_BINARY_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".npy", ".npz", ".pkl"}
MAX_BINARY_SCAN_BYTES = 256 * 1024 * 1024


@dataclass
class Report:
    """Everything the operator needs to eyeball before going public."""

    created: list[str] = field(default_factory=list)
    patched: list[str] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def render(self) -> str:
        sections = [
            ("Repos mirrored", self.created),
            ("Files rewritten", self.patched),
            ("NEEDS A HUMAN LOOK", self.flagged),
            ("Skipped", self.skipped),
        ]
        lines: list[str] = []
        for title, rows in sections:
            lines.append(f"\n=== {title} ({len(rows)}) ===")
            if rows:
                lines.extend(f"  {row}" for row in rows)
            else:
                lines.append("  (none)")
        return "\n".join(lines)


def build_subs(dst_namespace: str, extra: list[str]) -> list[tuple[str, str]]:
    subs = [(old, dst_namespace if new is None else new) for old, new in DEFAULT_SUBS]
    for item in extra:
        if "=" not in item:
            sys.exit(f"--extra-sub needs OLD=NEW, got {item!r}")
        old, new = item.split("=", 1)
        if not old:
            sys.exit("--extra-sub needs a non-empty OLD")
        subs.append((old, new))
    return sorted(subs, key=lambda kv: -len(kv[0]))


def scrub_tree(root: Path, subs: list[tuple[str, str]], report: Report, label: str) -> None:
    """Rewrite identifying strings in text files; flag leaks we will not touch."""
    needles = [old for old, _ in subs]
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if ".cache" in rel.parts or ".git" in rel.parts:
            continue
        suffix = path.suffix.lower()

        if suffix in TEXT_SUFFIXES:
            try:
                original = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                report.flagged.append(f"{label}:{rel} -- undecodable text file, review by hand")
                continue
            patched = original
            for old, new in subs:
                patched = patched.replace(old, new)
            redactions: list[str] = []
            for pattern, placeholder, why in AUTO_REDACT_PATTERNS:
                patched, hits = re.subn(pattern, placeholder, patched)
                if hits:
                    redactions.append(f"{hits} {why}(s)")
            if patched != original:
                path.write_text(patched, encoding="utf-8")
                detail = f" -- redacted {', '.join(redactions)}" if redactions else ""
                report.patched.append(f"{label}:{rel}{detail}")
            for pattern, why in FLAG_PATTERNS:
                for hit in sorted(set(re.findall(pattern, patched))):
                    report.flagged.append(f"{label}:{rel} -- {why}: {hit}")
            continue

        if suffix in SCAN_BINARY_SUFFIXES:
            size = path.stat().st_size
            if size > MAX_BINARY_SCAN_BYTES:
                report.flagged.append(
                    f"{label}:{rel} -- {size / 1e6:.0f} MB binary, too big to scan; "
                    "confirm by hand that it embeds no bucket paths or usernames"
                )
                continue
            blob = path.read_bytes()
            for needle in needles:
                if needle.encode() in blob:
                    report.flagged.append(
                        f"{label}:{rel} -- binary embeds {needle!r}; cannot be "
                        "auto-patched, re-save the artefact without it"
                    )
            for pattern, why in BINARY_FLAG_PATTERNS:
                for hit in sorted(set(re.findall(pattern.encode(), blob))):
                    report.flagged.append(
                        f"{label}:{rel} -- binary embeds {why}: {hit.decode(errors='replace')}"
                    )


def mirror_repo(
    api: HfApi,
    src_id: str,
    repo_type: str,
    dst_namespace: str,
    subs: list[tuple[str, str]],
    workdir: Path,
    report: Report,
    *,
    src_token: str | None,
    dst_token: str | None,
    execute: bool,
    private: bool,
) -> str:
    """Download, scrub and re-upload one repo. Returns the target repo id."""
    name = src_id.split("/")[-1]
    dst_id = f"{dst_namespace}/{name}"

    if not execute:
        report.created.append(f"{repo_type} {src_id} -> {dst_id} (dry run)")
        return dst_id

    local = workdir / repo_type / name
    if local.exists():
        shutil.rmtree(local)
    local.mkdir(parents=True)

    print(f"  downloading {src_id} ...", flush=True)
    # With local_dir set, huggingface_hub writes real files (not cache symlinks),
    # so scrubbing rewrites the copy we upload and never poisons the shared cache.
    snapshot_download(
        repo_id=src_id,
        repo_type=repo_type,
        local_dir=str(local),
        token=src_token,
    )
    # snapshot_download leaves bookkeeping behind; never upload it.
    shutil.rmtree(local / ".cache", ignore_errors=True)

    scrub_tree(local, subs, report, label=name)

    print(f"  creating {dst_id} ...", flush=True)
    api.create_repo(
        repo_id=dst_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
        token=dst_token,
    )
    print(f"  uploading {dst_id} ...", flush=True)
    api.upload_folder(
        repo_id=dst_id,
        repo_type=repo_type,
        folder_path=str(local),
        token=dst_token,
        commit_message="Anonymous mirror for double-blind review",
    )
    report.created.append(f"{repo_type} {src_id} -> {dst_id}")
    return dst_id


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--src-collection",
        required=True,
        help="collection slug exactly as it appears in the URL after /collections/",
    )
    ap.add_argument("--dst-namespace", required=True, help="the anonymous HF user or org")
    ap.add_argument("--dst-title", default="Boltz-1 sparse autoencoders (anonymous)")
    ap.add_argument(
        "--dst-description",
        default="Anonymous mirror of the SAE checkpoints accompanying this submission, "
        "provided for double-blind review.",
    )
    ap.add_argument("--workdir", default=".hf_anon_mirror", help="scratch dir for downloads")
    ap.add_argument("--extra-sub", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument(
        "--private",
        action="store_true",
        help="create private repos; reviewers cannot read these, so normally leave it off",
    )
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True)
    group.add_argument("--execute", dest="dry_run", action="store_false")
    args = ap.parse_args()

    src_token = os.environ.get("HF_TOKEN_SRC") or os.environ.get("HF_TOKEN")
    dst_token = os.environ.get("HF_TOKEN_DST")
    if not args.dry_run and not dst_token:
        print(
            "HF_TOKEN_DST must hold a write token for the anonymous account.",
            file=sys.stderr,
        )
        return 2
    if src_token and src_token == dst_token:
        print(
            "HF_TOKEN_SRC and HF_TOKEN_DST are the same token -- the mirror would be "
            "owned by the identified account. Use a token from the anonymous account.",
            file=sys.stderr,
        )
        return 2

    api = HfApi()
    subs = build_subs(args.dst_namespace, args.extra_sub)
    report = Report()
    workdir = Path(args.workdir)

    print(f"Reading collection {args.src_collection} ...")
    try:
        collection = api.get_collection(args.src_collection, token=src_token)
    except Exception as exc:
        print(f"Could not read the collection: {exc}", file=sys.stderr)
        print(
            "HF collection slugs usually carry a trailing id, e.g.\n"
            "  evolve-away/boltz-saes-6712ab34cd56ef7890abcdef\n"
            "Copy the slug verbatim from the browser URL after /collections/.",
            file=sys.stderr,
        )
        return 1
    print(f"  title: {collection.title!r}   items: {len(collection.items)}")

    mirrored: list[tuple[str, str]] = []
    for item in collection.items:
        kind = item.item_type
        if kind in {"model", "dataset"}:
            dst_id = mirror_repo(
                api, item.item_id, kind, args.dst_namespace, subs, workdir, report,
                src_token=src_token, dst_token=dst_token,
                execute=not args.dry_run, private=args.private,
            )
            mirrored.append((dst_id, kind))
        else:
            report.skipped.append(
                f"{kind} {item.item_id} -- not mirrored; a {kind} entry points back at "
                "the identified authors, so leave it out of the anonymous collection"
            )

    if args.dry_run:
        report.skipped.append("collection itself -- dry run, nothing was created")
    else:
        print(f"Creating collection on {args.dst_namespace} ...")
        dst_collection = api.create_collection(
            title=args.dst_title,
            namespace=args.dst_namespace,
            description=args.dst_description,
            private=False,
            exists_ok=True,
            token=dst_token,
        )
        for dst_id, kind in mirrored:
            api.add_collection_item(
                collection_slug=dst_collection.slug,
                item_id=dst_id,
                item_type=kind,
                exists_ok=True,
                token=dst_token,
            )
        report.created.append(
            f"collection -> https://huggingface.co/collections/{dst_collection.slug}"
        )

    print(report.render())
    if report.flagged:
        print(
            f"\n{len(report.flagged)} item(s) need a human look before you hand the link "
            "to reviewers -- see NEEDS A HUMAN LOOK above."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
