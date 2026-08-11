#!/usr/bin/env python3
"""
Download activation files from S3/R2 using a manifest file.

This script downloads activation files for proteins listed in a manifest file.
It supports both AWS S3 and Cloudflare R2 (via Sippy).

For R2 with Sippy:
  Sippy only works with GET operations, not LIST. Use generate_manifest.py
  (requires AWS access) to create a manifest file, then share it with R2 users
  who can download using this script.

Layer types:
  - Pairformer: "PairformerLayer_<N> s from PairformerLayer"
  - Diffusion:  "DiffusionTransformerLayer_<N> from DiffusionTransformer
                 from DiffusionModule from AtomDiffusion"

Example:
  python get_activations.py \\
    --bucket boltz-1-activations \\
    --manifest proteins.txt \\
    --layer_type pairformer \\
    --layer 20 \\
    --rec 0 \\
    --out ./downloads
"""

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, ConnectionClosedError, EndpointConnectionError
from dotenv import load_dotenv
from tap import tapify

# Load environment variables from .env file (if present).
# This ensures R2 credentials persist across terminal sessions.
load_dotenv()

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Error codes that warrant a retry.
RETRIABLE_ERROR_CODES = {
    "SlowDown",
    "Throttling",
    "ThrottlingException",
    "RequestTimeout",
    "InternalError",
    "ServiceUnavailable",
}

# Supported layer types for activation downloads.
VALID_LAYER_TYPES = {"pairformer", "diffusion"}


def get_layer_folder_name(layer_type: str, layer_num: int) -> str:
    """
    Construct the folder name for a given layer type and number.

    Args:
        layer_type: The type of layer (pairformer or diffusion).
        layer_num: The layer number.

    Returns:
        The folder name string matching the S3 structure.
    """
    if layer_type == "pairformer":
        return f"PairformerLayer_{layer_num} s from PairformerLayer"
    elif layer_type == "diffusion":
        return (
            f"DiffusionTransformerLayer_{layer_num} from DiffusionTransformer "
            f"from DiffusionModule from AtomDiffusion"
        )
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def get_r2_endpoint_url(account_id: str | None = None) -> str | None:
    """
    Construct Cloudflare R2 endpoint URL from account ID.

    Args:
        account_id: Cloudflare account ID. If None, reads from CF_ACCOUNT_ID env var.

    Returns:
        R2 endpoint URL, or None if no account ID available.
    """
    account_id = account_id or os.environ.get("CF_ACCOUNT_ID")
    if account_id:
        return f"https://{account_id}.r2.cloudflarestorage.com"
    return None


def create_s3_client(
    endpoint_url: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    max_pool_connections: int = 50,
) -> "S3Client":
    """
    Create an S3-compatible client, supporting both AWS S3 and Cloudflare R2.

    Credential resolution order:
    1. Explicit parameters (access_key_id, secret_access_key)
    2. R2-specific env vars (R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY) if endpoint_url is R2
    3. Standard AWS env vars (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    4. AWS credentials file / IAM role (boto3 default chain)

    Args:
        endpoint_url: Custom S3-compatible endpoint (e.g., Cloudflare R2).
                      If None, checks R2_ENDPOINT_URL env var, then CF_ACCOUNT_ID.
        access_key_id: Access key ID. If None, uses env vars.
        secret_access_key: Secret access key. If None, uses env vars.
        max_pool_connections: HTTP connection pool size for parallel downloads.

    Returns:
        Configured boto3 S3 client.
    """
    # Resolve endpoint URL: explicit > R2_ENDPOINT_URL env > CF_ACCOUNT_ID env.
    if endpoint_url is None:
        endpoint_url = os.environ.get("R2_ENDPOINT_URL") or get_r2_endpoint_url()

    # Determine if we're using R2 (affects credential resolution and region).
    is_r2 = endpoint_url and "r2.cloudflarestorage.com" in endpoint_url

    # Log which storage backend is being used (helps catch misconfiguration).
    if is_r2:
        logger.info(msg="Using Cloudflare R2 storage backend")
    else:
        logger.info(
            msg="Using AWS S3 storage backend (no R2 endpoint configured). "
            "Set CF_ACCOUNT_ID or R2_ENDPOINT_URL in .env to use R2."
        )

    # R2 requires specific region names (auto, wnam, enam, weur, eeur, apac, oc).
    # AWS region names like "eu-west-2" are invalid for R2.
    region_name = "auto" if is_r2 else None

    # Resolve credentials: explicit > R2 env vars (if R2) > AWS env vars > boto3 default.
    if access_key_id is None and is_r2:
        access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
    if secret_access_key is None and is_r2:
        secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")

    # Build client config.
    cfg = Config(
        max_pool_connections=max_pool_connections,
        s3={
            # R2 doesn't support some S3 features; disable them to avoid errors.
            "addressing_style": "path" if is_r2 else "auto",
        },
        # Disable checksum validation that R2 may not support.
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    )

    # Build client kwargs.
    client_kwargs = {"config": cfg}

    if endpoint_url:
        client_kwargs["endpoint_url"] = endpoint_url
        logger.info(f"Using custom endpoint: {endpoint_url}")

    # Set region for R2 (required).
    if region_name:
        client_kwargs["region_name"] = region_name
        logger.info(f"Using region: {region_name}")

    if access_key_id and secret_access_key:
        client_kwargs["aws_access_key_id"] = access_key_id
        client_kwargs["aws_secret_access_key"] = secret_access_key
        logger.info(msg="Using explicit credentials (R2 or provided)")
    elif is_r2:
        logger.warning(
            msg="R2 endpoint detected but no R2 credentials found. "
            "Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY environment variables."
        )

    return boto3.client("s3", **client_kwargs)


def is_missing_key_error(e: ClientError) -> bool:
    """Check if the error indicates a missing (or unreadable) S3 key.

    Note on the 403 / "Forbidden" case: when the caller lacks `s3:ListBucket`
    on the bucket, S3 cannot reveal whether a missing key is a 404 vs a
    permission denial, so it returns 403 for both cases. We treat 403 the
    same as 404 here so the `.npy.gz` -> `.npz` format fallback in
    `download_one` still fires when only the alternative format exists. If
    *both* formats are genuinely 403, the caller will surface a "missing"
    rather than an "error", which is still the correct user-facing signal.
    """
    code = e.response.get("Error", {}).get("Code", "")
    return code in {"404", "NoSuchKey", "NotFound", "403", "Forbidden", "AccessDenied"}


def should_retry(e: Exception) -> bool:
    """Determine if an exception warrants a retry."""
    if isinstance(e, EndpointConnectionError | ConnectionClosedError):
        return True
    if isinstance(e, ClientError):
        code = e.response.get("Error", {}).get("Code", "")
        if code in RETRIABLE_ERROR_CODES:
            return True
        status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        return isinstance(status, int) and status >= 500
    return False


def download_one(
    s3: "S3Client",
    bucket: str,
    protein_prefix: str,
    layer_folder_name: str,
    rec: int,
    out_dir: Path,
    overwrite: bool = False,
    dry_run: bool = False,
    max_retries: int = 5,
) -> tuple[str, str, str]:
    """
    Download a single activation file for a protein.

    Tries both .npy.gz and .npz formats, downloading whichever exists.

    Args:
        s3: Boto3 S3 client.
        bucket: S3 bucket name.
        protein_prefix: Full prefix to the protein folder.
        layer_folder_name: Name of the layer folder.
        rec: Recycle number for 'pairformer' layer, or diffusion step for 'diffusion' layer (downloads output_<rec>.npy.gz or output_<rec>.npz).
        out_dir: Local output directory.
        overwrite: Whether to overwrite existing files.
        dry_run: If True, only print what would be downloaded.
        max_retries: Maximum number of retry attempts.

    Returns:
        Tuple of (status, protein_id, detail) where status is one of:
        "downloaded", "skipped", "missing", "error", "would_download".
    """
    # Extract protein ID from the prefix.
    protein_id = protein_prefix.rstrip("/").split("/")[-1]

    # Try both file formats: .npy.gz first, then .npz as fallback.
    file_formats = [".npy.gz", ".npz"]

    for file_ext in file_formats:
        key = f"{protein_prefix}{layer_folder_name}/output_{rec}{file_ext}"
        rel_path = Path(protein_id) / layer_folder_name / f"output_{rec}{file_ext}"
        dest = out_dir / rel_path

        # Skip if file exists and we're not overwriting.
        if dest.exists() and not overwrite:
            logger.info(f"Skipping {protein_id}: as the path {str(rel_path)} already exists")
            return ("skipped", protein_id, str(rel_path))

        # Dry run mode: check if file exists in S3.
        if dry_run:
            try:
                s3.head_object(Bucket=bucket, Key=key)
                return ("would_download", protein_id, f"s3://{bucket}/{key} -> {rel_path}")
            except ClientError as e:
                if is_missing_key_error(e):
                    # Try next format.
                    continue
                # Other error, report it.
                error_code = e.response.get("Error", {}).get("Code", "ClientError")
                return ("error", protein_id, f"{error_code}: {e}")

        # Create destination directory.
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Attempt download with exponential backoff retry.
        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                s3.download_file(bucket, key, str(dest))
                return ("downloaded", protein_id, str(rel_path))
            except ClientError as e:
                if is_missing_key_error(e):
                    # File not found with this extension, try next format.
                    break
                if attempt < max_retries and should_retry(e):
                    logger.debug(f"Retry {attempt}/{max_retries} for {protein_id}: {e}")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                error_code = e.response.get("Error", {}).get("Code", "ClientError")
                return ("error", protein_id, f"{error_code}: {e}")
            except Exception as e:
                if attempt < max_retries and should_retry(e):
                    logger.debug(f"Retry {attempt}/{max_retries} for {protein_id}: {e}")
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return ("error", protein_id, repr(e))

    # Neither format found.
    base_key = f"{protein_prefix}{layer_folder_name}/output_{rec}"
    return ("missing", protein_id, f"{base_key}[.npy.gz or .npz]")


def download_activations(
    bucket: str,
    manifest: str,
    layer_type: str,
    layer: int,
    rec: int,
    out: str = "./downloads",
    workers: int = 16,
    limit: int = 0,
    overwrite: bool = False,
    dry_run: bool = False,
    endpoint_url: str | None = None,
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
) -> int:
    """
    Download protein activations using a manifest file.

    Downloads .npy.gz and .npz activation files for a specific layer type and number
    for proteins listed in the manifest file.

    Args:
        bucket: S3/R2 bucket name.
        manifest: Path to manifest file containing protein prefixes (one per line).
        layer_type: Type of layer to download: "pairformer" or "diffusion".
        layer: Layer number to download (e.g., 20).
        rec: Recycle number for 'pairformer' layer, or diffusion step for 'diffusion' layer (downloads output_<rec>.npy.gz or output_<rec>.npz).
        out: Local output directory.
        workers: Number of parallel download threads.
        limit: Only process first N proteins (0 = no limit).
        overwrite: Overwrite local files if they already exist.
        dry_run: Only print what would be downloaded.
        endpoint_url: Custom S3-compatible endpoint URL (e.g., R2).
        access_key_id: Access key ID for custom endpoints.
        secret_access_key: Secret access key for custom endpoints.

    Returns:
        Exit code (0 for success).
    """
    # Validate layer type.
    layer_type = layer_type.lower()
    if layer_type not in VALID_LAYER_TYPES:
        logger.error(f"Invalid layer_type: {layer_type}. Must be one of: {VALID_LAYER_TYPES}")
        return 1

    # Validate manifest file exists.
    manifest_path = Path(manifest).expanduser().resolve()
    if not manifest_path.exists():
        logger.error(f"Manifest file not found: {manifest_path}")
        return 1

    logger.info(f"Layer type: {layer_type}")
    logger.info(f"Layer number: {layer}")
    rec_label = "Recycle number" if layer_type == "pairformer" else "Diffusion step"
    logger.info(f"{rec_label}: {rec}")
    logger.info(f"Manifest file: {manifest_path}")

    # Setup output directory.
    out_dir = Path(out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # Construct the layer folder name based on type.
    layer_folder_name = get_layer_folder_name(layer_type, layer)
    logger.info(f"Layer folder pattern: {layer_folder_name}")

    # Create S3/R2 client with appropriate credentials and connection pool.
    s3 = create_s3_client(
        endpoint_url=endpoint_url,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        max_pool_connections=max(10, workers * 2),
    )

    # Import threading utilities.
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    # Counters for summary.
    total_seen = 0
    counts = {"downloaded": 0, "skipped": 0, "missing": 0, "error": 0, "would_download": 0}
    errors: list[tuple[str, str]] = []

    # Control inflight tasks to manage memory.
    max_inflight = max(1, workers * 3)
    futures: set = set()

    def submit(protein_prefix: str):
        """Submit a download task to the executor."""
        return executor.submit(
            download_one,
            s3,
            bucket,
            protein_prefix,
            layer_folder_name,
            rec,
            out_dir,
            overwrite,
            dry_run,
        )

    missing_proteins: list[tuple[str, str]] = []

    def process_completed(done_futures):
        """Process completed futures and update counters."""
        for fut in done_futures:
            status, protein_id, detail = fut.result()
            counts[status] += 1
            if status == "error":
                errors.append((protein_id, detail))
            elif status == "downloaded":
                logger.debug(f"Downloaded: {protein_id}")
            elif status == "missing":
                missing_proteins.append((protein_id, detail))
                logger.debug(f"Missing: {protein_id} - {detail}")

    def iterate_proteins_from_manifest():
        """Iterate over protein prefixes from manifest file."""
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Ensure prefix ends with /
                    if not line.endswith("/"):
                        line += "/"
                    yield line

    # Process all proteins from manifest.
    logger.info("Reading proteins from manifest file...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for protein_prefix in iterate_proteins_from_manifest():
            total_seen += 1

            # Check limit.
            if limit and total_seen > limit:
                logger.info(f"Reached limit of {limit} proteins, stopping.")
                break

            # Submit download task.
            futures.add(submit(protein_prefix))

            # Process completed tasks if we have too many inflight.
            if len(futures) >= max_inflight:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                process_completed(done)

        # Process remaining futures.
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            process_completed(done)

    # Print summary.
    print("\n=== Summary ===")
    print(f"Layer type: {layer_type}")
    print(f"Layer folder: {layer_folder_name}")
    print(f"Protein folders processed: {total_seen}")
    for k in ["downloaded", "skipped", "missing", "error", "would_download"]:
        print(f"{k:>14}: {counts[k]}")

    # Print errors if any.
    if errors:
        print("\nFirst few errors:")
        for protein_id, detail in errors[:10]:
            print(f"  {protein_id}: {detail}")

    # Print missing proteins if any.
    if missing_proteins:
        print(f"\nMissing files ({len(missing_proteins)} proteins):")
        for protein_id, detail in missing_proteins[:10]:
            print(f"  {protein_id}: {detail}")
        if len(missing_proteins) > 10:
            print(f"  ... and {len(missing_proteins) - 10} more")

    print(f"\nLocal output root: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(tapify(download_activations))
