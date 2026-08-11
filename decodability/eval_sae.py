#!/usr/bin/env python3
"""Evaluate SAE reconstruction error and feature usage."""

# from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tap import tapify

from sae_utils import (
    TopKSAE,
    apply_demean,
    get_decoder_weight,
    list_activation_files,
    load_activation,
    resolve_device,
    set_seed,
)

# Configure logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for SAE evaluation.

    Attributes:
        checkpoint: Path to the training checkpoint.
        activations_dir: Directory containing downloaded activation folders.
        layer_subfolder: Subfolder name for the layer to evaluate on.
        rec: Recycle number (pairformer) or diffusion step.
        input_dim: Activation feature dimension.
        latent_dim: SAE latent size.
        k: Top-K active features per example.
        batch_size: Mini-batch size (tokens).
        max_batches: Optional cap on batches for quick eval.
        max_proteins: Optional cap on proteins loaded for quick eval.
        device: Torch device string override.
        tie_weights: Use encoder weights for the decoder.
        pre_encoder_bias: Subtract decoder bias from inputs before encoding.
        normalize_decoder: Normalize decoder weight columns to unit norm.
        demean_embeddings: Subtract per-dimension mean before evaluation.
        mean_vector_file: Optional mean vector file used for demeaning.
        seed: Random seed.
        out_path: Optional path to write JSON output.
        activation_file_count: Number of activation files found for evaluation.
    """

    checkpoint: str
    activations_dir: str
    layer_subfolder: str
    rec: int
    input_dim: int
    latent_dim: int
    k: int
    batch_size: int
    max_batches: int | None
    max_proteins: int | None
    device: str | None
    tie_weights: bool
    pre_encoder_bias: bool
    normalize_decoder: bool
    demean_embeddings: bool
    mean_vector_file: str | None
    seed: int
    out_path: str | None
    activation_file_count: int


def load_config_from_run(checkpoint_path: Path) -> dict:
    """Load config.json next to the checkpoint if it exists."""
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r") as f:
        return json.load(f)


def safe_div(numerator: float, denominator: float) -> float | None:
    """Safely divide two floats, returning None if the denominator is zero.

    Args:
        numerator: Value to divide.
        denominator: Value to divide by.

    Returns:
        The division result or None if denominator is zero.
    """
    if denominator == 0:
        return None
    return numerator / denominator


def eval_sae(
    checkpoint: str,
    activations_dir: str | None = None,
    layer_subfolder: str | None = None,
    rec: int | None = None,
    input_dim: int | None = None,
    latent_dim: int | None = None,
    k: int | None = None,
    batch_size: int = 2048,
    max_batches: int | None = None,
    max_proteins: int | None = None,
    device: str | None = None,
    tie_weights: bool | None = None,
    pre_encoder_bias: bool | None = None,
    normalize_decoder: bool | None = None,
    demean_embeddings: bool | None = None,
    mean_vector_path: str | None = None,
    seed: int = 7,
    out_path: str | None = None,
) -> None:
    """Evaluate a trained SAE on activation files.

    Args:
        checkpoint: Path to the training checkpoint (.pt).
        activations_dir: Directory containing downloaded activation folders.
        layer_subfolder: Subfolder name for the layer to evaluate on.
        rec: Recycle number (pairformer) or diffusion step.
        input_dim: Activation feature dimension.
        latent_dim: SAE latent size.
        k: Top-K active features per example.
        batch_size: Mini-batch size (tokens).
        max_batches: Optional cap on batches for quick eval.
        max_proteins: Optional cap on proteins loaded for quick eval.
        device: Torch device string override.
        tie_weights: Use encoder weights for the decoder.
        pre_encoder_bias: Subtract decoder bias from inputs before encoding.
        normalize_decoder: Normalize decoder weight columns to unit norm.
        demean_embeddings: Subtract per-dimension mean before evaluation.
        mean_vector_path: Optional override path for mean vector (.npy).
        seed: Random seed.
        out_path: Optional path to write JSON output.
    """
    logger.info(msg="Starting SAE evaluation")
    set_seed(seed)

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = load_config_from_run(checkpoint_path)
    activations_dir = activations_dir or config.get("activations_dir")
    layer_subfolder = layer_subfolder or config.get("layer_subfolder")
    rec = rec if rec is not None else config.get("rec")
    input_dim = input_dim if input_dim is not None else config.get("input_dim")
    latent_dim = latent_dim if latent_dim is not None else config.get("latent_dim")
    k = k if k is not None else config.get("k")
    tie_weights = tie_weights if tie_weights is not None else config.get("tie_weights", False)
    pre_encoder_bias = (
        pre_encoder_bias if pre_encoder_bias is not None else config.get("pre_encoder_bias", False)
    )
    normalize_decoder = (
        normalize_decoder
        if normalize_decoder is not None
        else config.get("normalize_decoder", False)
    )
    demean_embeddings = (
        demean_embeddings
        if demean_embeddings is not None
        else config.get("demean_embeddings", False)
    )

    mean_vector_file = config.get("mean_vector_file")
    resolved_mean_vector_path = mean_vector_path
    if resolved_mean_vector_path is None and mean_vector_file is not None:
        resolved_mean_vector_path = str(checkpoint_path.parent / mean_vector_file)

    if activations_dir is None or layer_subfolder is None:
        raise ValueError("activations_dir and layer_subfolder must be provided")
    if rec is None or input_dim is None or latent_dim is None or k is None:
        raise ValueError("rec, input_dim, latent_dim, and k must be provided")

    eval_config = EvalConfig(
        checkpoint=str(checkpoint_path),
        activations_dir=activations_dir,
        layer_subfolder=layer_subfolder,
        rec=int(rec),
        input_dim=int(input_dim),
        latent_dim=int(latent_dim),
        k=int(k),
        batch_size=batch_size,
        max_batches=max_batches,
        max_proteins=max_proteins,
        device=device,
        tie_weights=bool(tie_weights),
        pre_encoder_bias=bool(pre_encoder_bias),
        normalize_decoder=bool(normalize_decoder),
        demean_embeddings=bool(demean_embeddings),
        mean_vector_file=mean_vector_file,
        seed=seed,
        out_path=out_path,
        activation_file_count=0,
    )

    activations_path = Path(activations_dir)
    files = list_activation_files(activations_path, layer_subfolder, int(rec), max_proteins)
    if not files:
        raise FileNotFoundError("No activation files found for evaluation")
    logger.info(msg=f"Found {len(files)} activation files")
    eval_config = EvalConfig(
        **{
            **eval_config.__dict__,
            "activation_file_count": len(files),
        }
    )

    device_obj = resolve_device(device)
    logger.info(msg=f"Using device: {device_obj}")

    model = TopKSAE(
        input_dim=int(input_dim),
        latent_dim=int(latent_dim),
        k=int(k),
        tie_weights=bool(tie_weights),
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.to(device_obj)
    model.eval()

    mean_vec_np: np.ndarray | None = None
    mean_vec_tensor: torch.Tensor | None = None
    if eval_config.demean_embeddings:
        if resolved_mean_vector_path is None:
            raise ValueError(
                "demean_embeddings=True but no mean vector path found in config or argument"
            )
        mean_path = Path(resolved_mean_vector_path)
        if not mean_path.exists():
            raise FileNotFoundError(f"Mean vector not found: {mean_path}")
        mean_vec_np = np.load(mean_path).astype(np.float32, copy=False)
        if mean_vec_np.ndim != 1 or mean_vec_np.shape[0] != int(input_dim):
            raise ValueError(
                f"Mean vector shape mismatch: expected ({input_dim},), got {mean_vec_np.shape}"
            )
        mean_vec_tensor = torch.from_numpy(mean_vec_np).to(device_obj)
        logger.info(msg=f"Loaded mean vector for demeaning: {mean_path}")

    total_mse = 0.0
    total_elements = 0
    total_batches = 0
    active_feature_frac_sum = 0.0
    feature_counts = torch.zeros(int(latent_dim), dtype=torch.long, device=device_obj)

    sum_x = np.zeros(int(input_dim), dtype=np.float64)
    sum_x2 = np.zeros(int(input_dim), dtype=np.float64)
    total_tokens = 0

    with torch.no_grad():
        for path in files:
            activations = load_activation(path)
            if activations.shape[1] != int(input_dim):
                raise ValueError(
                    "Input dim mismatch in "
                    f"{path}: expected {input_dim}, got {activations.shape[1]}"
                )
            indices = np.arange(activations.shape[0])
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                if len(batch_indices) == 0:
                    continue
                batch_np = activations[batch_indices]
                batch = torch.from_numpy(batch_np).to(device_obj)
                if mean_vec_tensor is not None:
                    batch = apply_demean(batch, mean_vec_tensor)
                    batch_np = apply_demean(batch_np, mean_vec_np)

                input_batch = batch
                if eval_config.pre_encoder_bias:
                    enc_input = input_batch - model.decoder.bias
                else:
                    enc_input = input_batch
                sparse = model.encode(enc_input)
                decoder_weight = get_decoder_weight(model, eval_config.normalize_decoder)
                if model.tie_weights:
                    recon = torch.matmul(sparse, decoder_weight) + model.decoder.bias
                else:
                    recon = torch.matmul(sparse, decoder_weight.T) + model.decoder.bias

                mse = torch.nn.functional.mse_loss(recon, input_batch, reduction="sum")
                total_mse += float(mse.item())
                total_elements += int(input_batch.numel())
                total_batches += 1

                active = (sparse > 0).sum(dim=0)
                feature_counts += active
                active_feature_frac_sum += float((active > 0).float().mean().item())

                sum_x += batch_np.sum(axis=0)
                sum_x2 += (batch_np**2).sum(axis=0)
                total_tokens += batch_np.shape[0]

                if max_batches is not None and total_batches >= max_batches:
                    break
            if max_batches is not None and total_batches >= max_batches:
                break

    mean_mse = total_mse / max(total_elements, 1)
    mean_x = sum_x / max(total_tokens, 1)
    mean_x2 = sum_x2 / max(total_tokens, 1)
    rms = float(np.sqrt(mean_x2.mean()))
    mse_zero = float(mean_x2.mean())
    mse_mean = float((mean_x2 - mean_x**2).mean())
    active_feature_frac = active_feature_frac_sum / max(total_batches, 1)
    fraction_features_active = float((feature_counts > 0).float().mean().item())

    results = {
        "checkpoint": str(checkpoint_path),
        "mean_mse": mean_mse,
        "input_rms": rms,
        "baseline_mse_zero": mse_zero,
        "baseline_mse_mean": mse_mean,
        "relative_mse_zero": safe_div(mean_mse, mse_zero),
        "relative_mse_mean": safe_div(mean_mse, mse_mean),
        "active_feature_frac": active_feature_frac,
        "fraction_features_active": fraction_features_active,
        "total_batches": total_batches,
        "total_tokens": total_tokens,
        "config": eval_config.__dict__,
    }

    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        logger.info(msg=f"Wrote eval results to {out_file}")
    else:
        logger.info(msg=json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(tapify(eval_sae))
