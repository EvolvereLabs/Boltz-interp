#!/usr/bin/env python3
"""
Generate a manifest file listing all protein prefixes from an S3 bucket.

This script requires AWS S3 access. The generated manifest file can be shared
with users who only have R2 credentials, allowing them to download via Sippy
using get_activations.py --manifest.

Example:
  python generate_manifest.py \
    --bucket boltz-1-activations \
    --prefix boltz-scalable-data/boltz1024_3000 \
    --output proteins.txt

Example (flat layout):
  python generate_manifest.py \
    --bucket swissprot-annotated-proteins-activations \
    --prefix SwissProtAnnotation5/activations \
    --output proteins.txt \
    --flat
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

import boto3
from botocore.config import Config
from tap import tapify

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def create_s3_client(region_name: str | None = None) -> "S3Client":
    """
    Create an AWS S3 client for listing objects.

    Uses default boto3 credential resolution (env vars, ~/.aws/credentials, IAM role).

    Args:
        region_name: AWS region name (e.g., "eu-west-2"). If None, uses boto3 default.

    Returns:
        Configured boto3 S3 client.
    """
    cfg = Config(
        max_pool_connections=10,
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )

    client_kwargs = {"config": cfg}
    if region_name:
        client_kwargs["region_name"] = region_name
        logger.info(f"Using AWS region: {region_name}")

    return boto3.client("s3", **client_kwargs)


def list_shard_prefixes(s3: "S3Client", bucket: str, prefix: str) -> Iterator[str]:
    """
    List all shard folders under the given prefix.

    Args:
        s3: Boto3 S3 client.
        bucket: S3 bucket name.
        prefix: Base prefix (e.g., "boltz-scalable-data/boltz1024_3000").

    Yields:
        Shard prefixes like "boltz-scalable-data/boltz1024_3000/shard_0/".
    """
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    logger.info(f"Listing shards under: s3://{bucket}/{prefix}")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            shard_prefix = cp["Prefix"]
            # Only yield directories that look like shards (shard_N).
            folder_name = shard_prefix.rstrip("/").split("/")[-1]
            if folder_name.startswith("shard_"):
                logger.debug(f"Found shard: {shard_prefix}")
                yield shard_prefix


def list_protein_prefixes_in_shard(s3: "S3Client", bucket: str, shard_prefix: str) -> Iterator[str]:
    """
    List all protein ID folders within a shard's activations directory.

    Args:
        s3: Boto3 S3 client.
        bucket: S3 bucket name.
        shard_prefix: Shard prefix (e.g., "prefix/shard_0/").

    Yields:
        Protein prefixes like "prefix/shard_0/activations/1a3j/".
    """
    activations_prefix = f"{shard_prefix}activations/"
    logger.debug(f"Listing proteins under: {activations_prefix}")

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=activations_prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            yield cp["Prefix"]


def list_protein_prefixes_flat(s3: "S3Client", bucket: str, prefix: str) -> Iterator[str]:
    """
    List all protein ID folders directly under the given prefix (flat layout).

    Args:
        s3: Boto3 S3 client.
        bucket: S3 bucket name.
        prefix: Base prefix containing protein folders directly.

    Yields:
        Protein prefixes like "SwissProtAnnotation5/activations/A0A0A7HFE1/".
    """
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    logger.info(f"Listing proteins under: s3://{bucket}/{prefix}")
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            yield cp["Prefix"]


def generate_manifest(
    bucket: str,
    output: str,
    prefix: str = "boltz-scalable-data/boltz1024_3000",
    flat: bool = False,
    limit: int = 0,
    region: str | None = None,
) -> int:
    """
    Generate a manifest file listing all protein prefixes from S3.

    Args:
        bucket: S3 bucket name.
        output: Path to output manifest file.
        prefix: Base prefix containing protein folders (or shard folders if not flat).
        flat: If True, protein folders are directly under prefix (no shard subfolders).
        limit: Only list first N proteins (0 = no limit).
        region: AWS region name (e.g., "eu-west-2"). If None, uses boto3 default.

    Returns:
        Exit code (0 for success).
    """
    logger.info(f"Generating manifest from s3://{bucket}/{prefix}")
    logger.info(f"Layout: {'flat' if flat else 'sharded'}")
    logger.info(f"Output file: {output}")

    # Create S3 client.
    s3 = create_s3_client(region_name=region)

    # Setup output file.
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def iterate_proteins() -> Iterator[str]:
        """Iterate over all protein prefixes."""
        if flat:
            logger.info("Using flat layout (proteins directly under prefix)")
            yield from list_protein_prefixes_flat(s3, bucket, prefix)
        else:
            logger.info("Using sharded layout (prefix/shard_X/activations/protein_id/)")
            for shard_prefix in list_shard_prefixes(s3, bucket, prefix):
                shard_name = shard_prefix.rstrip("/").split("/")[-1]
                logger.info(f"Processing {shard_name}...")
                yield from list_protein_prefixes_in_shard(s3, bucket, shard_prefix)

    # Write manifest file.
    protein_count = 0
    with open(output_path, "w") as f:
        # Write header comments.
        f.write(f"# Protein manifest generated from s3://{bucket}/{prefix}\n")
        f.write(f"# Layout: {'flat' if flat else 'sharded'}\n")
        f.write(f"# Use with: python get_activations.py --manifest {output}\n")
        f.write("#\n")

        for protein_prefix in iterate_proteins():
            f.write(f"{protein_prefix}\n")
            protein_count += 1

            if protein_count % 1000 == 0:
                logger.info(f"Listed {protein_count} proteins...")

            # Check limit.
            if limit and protein_count >= limit:
                logger.info(f"Reached limit of {limit} proteins, stopping.")
                break

    logger.info(f"Manifest saved to: {output_path}")

    # Print summary.
    print("\n=== Manifest Generated ===")
    print(f"File: {output_path}")
    print(f"Proteins listed: {protein_count}")
    print(f"Source: s3://{bucket}/{prefix}")
    print("\nTo download using this manifest:")
    print("  python get_activations.py \\")
    print(f"    --bucket {bucket} \\")
    print(f"    --manifest {output} \\")
    print("    --layer_type pairformer \\")
    print("    --layer 20 \\")
    print("    --rec 0 \\")
    print("    --out ./downloads")

    return 0


if __name__ == "__main__":
    raise SystemExit(tapify(generate_manifest))
