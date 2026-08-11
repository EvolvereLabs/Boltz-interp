#!/usr/bin/env python3
"""Shared utilities for SAE training and evaluation."""

from __future__ import annotations

import gzip
import logging
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class TopKSAE(torch.nn.Module):
    """Simple Top-K sparse autoencoder.

    Args:
        input_dim: Input activation dimension.
        latent_dim: Latent dimension for the SAE.
        k: Number of active latent features per example.
        tie_weights: If True, reuse encoder weights for decoding.

    Attributes:
        input_dim: Input activation dimension.
        latent_dim: Latent dimension for the SAE.
        k: Number of active latent features per example.
        tie_weights: If True, reuse encoder weights for decoding.
    """

    def __init__(self, input_dim: int, latent_dim: int, k: int, tie_weights: bool) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError("k must be > 0")
        if k > latent_dim:
            raise ValueError("k must be <= latent_dim")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.k = k
        self.tie_weights = tie_weights

        # Encoder and decoder weights are small by default for stable early training.
        self.encoder = torch.nn.Linear(input_dim, latent_dim, bias=True)
        self.decoder = torch.nn.Linear(latent_dim, input_dim, bias=True)
        torch.nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.encoder.bias)
        torch.nn.init.normal_(self.decoder.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs into sparse latent activations using Top-K.

        Args:
            x: Input activations with shape (batch, input_dim).

        Returns:
            Sparse latent activations with shape (batch, latent_dim).
        """
        pre_act = torch.relu(self.encoder(x))
        values, indices = torch.topk(pre_act, k=self.k, dim=-1)
        sparse = torch.zeros_like(pre_act)
        sparse.scatter_(dim=-1, index=indices, src=values)
        return sparse


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch RNGs for reproducibility.

    Args:
        seed: Random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_weight_columns(weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize weight columns to unit norm.

    Args:
        weight: Weight matrix with shape (out_features, in_features).
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized weight matrix.
    """
    norms = torch.norm(weight, dim=0, keepdim=True).clamp_min(eps)
    return weight / norms


def normalize_weight_rows(weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize weight rows to unit norm.

    Args:
        weight: Weight matrix with shape (out_features, in_features).
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized weight matrix.
    """
    norms = torch.norm(weight, dim=1, keepdim=True).clamp_min(eps)
    return weight / norms


def get_decoder_weight(model: TopKSAE, normalize_decoder: bool) -> torch.Tensor:
    """Return decoder weight, optionally normalized.

    Args:
        model: SAE model.
        normalize_decoder: Whether to normalize decoder weights.

    Returns:
        Decoder weight tensor.
    """
    if model.tie_weights:
        weight = model.encoder.weight
        if normalize_decoder:
            return normalize_weight_rows(weight)
        return weight
    weight = model.decoder.weight
    if normalize_decoder:
        return normalize_weight_columns(weight)
    return weight


def list_activation_files(
    activations_dir: Path,
    layer_subfolder: str,
    rec: int,
    max_proteins: int | None,
) -> list[Path]:
    """List activation files under the activation directory.

    Args:
        activations_dir: Root directory containing protein activation folders.
        layer_subfolder: Subfolder name for the layer to train on.
        rec: Recycle number (pairformer) or diffusion step.
        max_proteins: Optional cap on proteins loaded for quick tests.

    Returns:
        List of activation file paths.
    """
    files: list[Path] = []
    protein_dirs = sorted(p for p in activations_dir.iterdir() if p.is_dir())
    if max_proteins is not None:
        protein_dirs = protein_dirs[:max_proteins]

    for protein_dir in protein_dirs:
        layer_dir = protein_dir / layer_subfolder
        if not layer_dir.exists():
            logger.debug(msg=f"Skipping missing layer dir: {layer_dir}")
            continue
        npz_path = layer_dir / f"output_{rec}.npz"
        npy_gz_path = layer_dir / f"output_{rec}.npy.gz"
        if npz_path.exists():
            files.append(npz_path)
            continue
        if npy_gz_path.exists():
            files.append(npy_gz_path)
            continue
        logger.debug(msg=f"No activation file found in: {layer_dir}")

    return files


def load_activation(path: Path) -> np.ndarray:
    """Load a single activation file into a 2D array (tokens x dim).

    Args:
        path: Path to the activation file (.npz or .npy.gz).

    Returns:
        Activation array with shape (tokens, dim).

    Raises:
        ValueError: If the file contains no arrays or has an unexpected shape.
    """
    suffixes = path.suffixes
    if suffixes[-2:] == [".npy", ".gz"]:
        with gzip.open(path, "rb") as f:
            arr = np.load(f)
    elif suffixes[-1] == ".npy":
        arr = np.load(path)
    else:
        with np.load(path) as data:
            if len(data.files) == 0:
                raise ValueError(f"No arrays found in {path}")
            arr = data[data.files[0]]

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D activations from {path}, got shape {arr.shape}")
    return arr.astype(np.float32, copy=False)


def compute_activation_mean(
    files: list[Path],
    input_dim: int,
    max_tokens_for_mean: int | None = None,
) -> np.ndarray:
    """Compute a per-dimension mean vector from activation files.

    Args:
        files: Activation file paths.
        input_dim: Expected activation dimension.
        max_tokens_for_mean: Optional cap on processed tokens.

    Returns:
        Mean vector with shape (input_dim,) and dtype float32.
    """
    if not files:
        raise ValueError("No activation files provided to compute_activation_mean")
    if input_dim <= 0:
        raise ValueError("input_dim must be > 0")
    if max_tokens_for_mean is not None and max_tokens_for_mean <= 0:
        raise ValueError("max_tokens_for_mean must be > 0 when provided")

    sum_x = np.zeros(input_dim, dtype=np.float64)
    total_tokens = 0
    token_cap = max_tokens_for_mean

    for path in files:
        arr = load_activation(path)
        if arr.shape[1] != input_dim:
            raise ValueError(
                f"Input dim mismatch in {path}: expected {input_dim}, got {arr.shape[1]}"
            )
        if token_cap is None:
            chunk = arr
        else:
            remaining = token_cap - total_tokens
            if remaining <= 0:
                break
            chunk = arr[:remaining]
        if chunk.size == 0:
            continue
        sum_x += chunk.sum(axis=0, dtype=np.float64)
        total_tokens += chunk.shape[0]
        if token_cap is not None and total_tokens >= token_cap:
            break

    if total_tokens == 0:
        raise ValueError("No tokens available to compute activation mean")

    mean_vec = sum_x / float(total_tokens)
    return mean_vec.astype(np.float32, copy=False)


def apply_demean(
    x: np.ndarray | torch.Tensor,
    mean_vec: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Subtract a per-dimension mean vector from activations.

    Args:
        x: Activation matrix with shape (tokens, dim).
        mean_vec: Mean vector with shape (dim,).

    Returns:
        Demeaned activations with same type as input.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x to be 2D, got shape {tuple(x.shape)}")
    if mean_vec.ndim != 1:
        raise ValueError(f"Expected mean_vec to be 1D, got shape {tuple(mean_vec.shape)}")
    if x.shape[1] != mean_vec.shape[0]:
        raise ValueError(
            f"Dimension mismatch: x has dim {x.shape[1]} but mean_vec has dim {mean_vec.shape[0]}"
        )
    return x - mean_vec


def iter_activation_batches(
    files: list[Path],
    batch_size: int,
    rng: np.random.Generator,
    shuffle_files: bool,
    shuffle_within_file: bool,
) -> Iterator[np.ndarray]:
    """Yield mini-batches of activations indefinitely.

    Args:
        files: List of activation file paths.
        batch_size: Mini-batch size (tokens).
        rng: NumPy RNG for shuffling.
        shuffle_files: Shuffle file order each epoch.
        shuffle_within_file: Shuffle tokens within each activation file.

    Yields:
        Mini-batches of activations with shape (batch, dim).
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    file_indices = list(range(len(files)))
    while True:
        if shuffle_files:
            rng.shuffle(file_indices)
        for file_idx in file_indices:
            path = files[file_idx]
            activations = load_activation(path)
            indices = np.arange(activations.shape[0])
            if shuffle_within_file:
                rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                if len(batch_indices) == 0:
                    continue
                yield activations[batch_indices]


def resolve_device(device: str | None) -> torch.device:
    """Resolve a torch device from a user-provided string.

    Args:
        device: Optional device string override.

    Returns:
        Torch device to use for training.
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
