#!/usr/bin/env python3
"""
Train a Top-K sparse autoencoder (SAE) on Boltz activation files.

Example:
    python train_sae.py \
        --activations_dir ./downloads \
        --layer_subfolder "PairformerLayer_20 s from PairformerLayer" \
        --rec 1 \
        --input_dim 384 \
        --latent_dim 2048 \
        --k 32 \
        --batch_size 2048 \
        --max_steps 20000 \
        --run_name "pairformer20_topk32"
"""

# from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tap import tapify

from sae_utils import (
    TopKSAE,
    apply_demean,
    compute_activation_mean,
    get_decoder_weight,
    iter_activation_batches,
    list_activation_files,
    load_activation,
    normalize_weight_columns,
    normalize_weight_rows,
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
class TrainConfig:
    """Configuration for SAE training.

    Attributes:
        activations_dir: Directory containing downloaded activation folders.
        layer_subfolder: Subfolder name for the layer to train on.
        rec: Recycle number (pairformer) or diffusion step.
        input_dim: Activation feature dimension.
        latent_dim: SAE latent size.
        k: Top-K active features per example.
        batch_size: Mini-batch size (tokens).
        max_steps: Number of optimization steps.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        optimizer_weight_decay: Effective AdamW weight decay used by the optimizer.
        weight_l2: Explicit L2 penalty on encoder/decoder weights.
        grad_clip: Optional max gradient norm for clipping.
        grad_projection: Project gradients to preserve decoder directions.
        seed: Random seed.
        log_every: Steps between logging.
        save_every: Steps between checkpoints.
        max_proteins: Optional cap on proteins loaded for quick tests.
        shuffle_files: Shuffle file order each epoch.
        shuffle_within_file: Shuffle tokens within each activation file.
        use_amp: Enable automatic mixed precision on CUDA.
        run_name: Optional name for the run (used in output paths).
        out_dir: Base directory to store outputs.
        device: Torch device string override.
        tie_weights: Use encoder weights for the decoder.
        tied_init: Initialize decoder weights from encoder weights.
        demean_embeddings: Subtract per-dimension mean from activations.
        mean_vector_path: Optional path to an existing mean vector (.npy).
        max_tokens_for_mean: Optional token cap when computing mean vector.
        mean_vector_file: Run-local file name for saved mean vector.
        pre_encoder_bias: Subtract decoder bias from inputs before encoding.
        normalize_decoder: Normalize decoder weight columns to unit norm.
        activation_file_count: Number of activation files found for training.
        resume_from_checkpoint: Optional checkpoint path used to resume training.
    """

    activations_dir: str
    layer_subfolder: str
    rec: int
    input_dim: int
    latent_dim: int
    k: int
    batch_size: int
    max_steps: int
    lr: float
    weight_decay: float
    optimizer_weight_decay: float
    weight_l2: float
    grad_clip: float | None
    grad_projection: bool
    seed: int
    log_every: int
    save_every: int
    max_proteins: int | None
    shuffle_files: bool
    shuffle_within_file: bool
    use_amp: bool
    run_name: str | None
    out_dir: str
    device: str | None
    tie_weights: bool
    tied_init: bool
    demean_embeddings: bool
    mean_vector_path: str | None
    max_tokens_for_mean: int | None
    mean_vector_file: str | None
    pre_encoder_bias: bool
    normalize_decoder: bool
    activation_file_count: int
    resume_from_checkpoint: str | None


def write_json(path: Path, payload: dict) -> None:
    """Write JSON payload to disk with stable formatting.

    Args:
        path: Output JSON path.
        payload: Data to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def train_sae(
    activations_dir: str,
    layer_subfolder: str,
    rec: int,
    input_dim: int,
    latent_dim: int,
    k: int,
    batch_size: int,
    max_steps: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    weight_l2: float = 0.0,
    grad_clip: float | None = None,
    grad_projection: bool = False,
    seed: int = 7,
    log_every: int = 500,
    save_every: int = 5000,
    max_proteins: int | None = None,
    shuffle_files: bool = True,
    shuffle_within_file: bool = True,
    use_amp: bool = False,
    run_name: str | None = None,
    out_dir: str = "sae_runs",
    device: str | None = None,
    tie_weights: bool = False,
    tied_init: bool = False,
    demean_embeddings: bool = False,
    mean_vector_path: str | None = None,
    max_tokens_for_mean: int | None = None,
    pre_encoder_bias: bool = False,
    normalize_decoder: bool = False,
    resume_from_checkpoint: str | None = None,
) -> None:
    """
    Train a Top-K SAE on activation files.

    Args:
        activations_dir: Directory containing downloaded activation folders.
        layer_subfolder: Subfolder name for the layer to train on.
        rec: Recycle number (pairformer) or diffusion step.
        input_dim: Activation feature dimension.
        latent_dim: SAE latent size.
        k: Top-K active features per example.
        batch_size: Mini-batch size (tokens).
        max_steps: Number of optimization steps.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        weight_l2: Explicit L2 penalty on encoder/decoder weights.
        grad_clip: Optional max gradient norm for clipping.
        grad_projection: Project gradients to preserve decoder directions.
        seed: Random seed.
        log_every: Steps between logging.
        save_every: Steps between checkpoints.
        max_proteins: Optional cap on proteins loaded for quick tests.
        shuffle_files: Shuffle file order each epoch.
        shuffle_within_file: Shuffle tokens within each activation file.
        use_amp: Enable automatic mixed precision on CUDA.
        run_name: Optional name for the run (used in output paths).
        out_dir: Base directory to store outputs.
        device: Torch device string override.
        tie_weights: Use encoder weights for the decoder.
        tied_init: Initialize decoder weights from encoder weights.
        demean_embeddings: Subtract per-dimension mean from activations.
        mean_vector_path: Optional path to an existing mean vector (.npy).
        max_tokens_for_mean: Optional token cap when computing mean vector.
        pre_encoder_bias: Subtract decoder bias from inputs before encoding.
        normalize_decoder: Normalize decoder weight columns to unit norm.
        resume_from_checkpoint: Optional checkpoint path to resume from.
    """
    logger.info(msg="Starting SAE training")
    set_seed(seed)

    optimizer_weight_decay = 0.0 if weight_l2 > 0 else weight_decay
    config = TrainConfig(
        activations_dir=activations_dir,
        layer_subfolder=layer_subfolder,
        rec=rec,
        input_dim=input_dim,
        latent_dim=latent_dim,
        k=k,
        batch_size=batch_size,
        max_steps=max_steps,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_weight_decay=optimizer_weight_decay,
        weight_l2=weight_l2,
        grad_clip=grad_clip,
        grad_projection=grad_projection,
        seed=seed,
        log_every=log_every,
        save_every=save_every,
        max_proteins=max_proteins,
        shuffle_files=shuffle_files,
        shuffle_within_file=shuffle_within_file,
        use_amp=use_amp,
        run_name=run_name,
        out_dir=out_dir,
        device=device,
        tie_weights=tie_weights,
        tied_init=tied_init,
        demean_embeddings=demean_embeddings,
        mean_vector_path=mean_vector_path,
        max_tokens_for_mean=max_tokens_for_mean,
        mean_vector_file="mean_vector.npy" if demean_embeddings else None,
        pre_encoder_bias=pre_encoder_bias,
        normalize_decoder=normalize_decoder,
        activation_file_count=0,
        resume_from_checkpoint=resume_from_checkpoint,
    )

    activations_path = Path(activations_dir)
    if not activations_path.exists():
        raise FileNotFoundError(f"Activations directory not found: {activations_path}")

    files = list_activation_files(activations_path, layer_subfolder, rec, max_proteins)
    if not files:
        raise FileNotFoundError(
            "No activation files found. Check layer_subfolder/rec or download activations."
        )
    logger.info(msg=f"Found {len(files)} activation files")
    config = TrainConfig(
        **{
            **asdict(config),
            "activation_file_count": len(files),
        }
    )
    sample = load_activation(files[0])
    if sample.shape[1] != input_dim:
        raise ValueError(
            f"Input dim mismatch: expected {input_dim}, found {sample.shape[1]} in {files[0]}"
        )

    run_label = run_name or f"sae_{layer_subfolder.replace(' ', '_')}_rec{rec}_{int(time.time())}"
    run_dir = Path(out_dir) / run_label
    run_dir.mkdir(parents=True, exist_ok=True)

    mean_vec_np: np.ndarray | None = None
    if demean_embeddings:
        if mean_vector_path:
            mean_path = Path(mean_vector_path)
            if not mean_path.exists():
                raise FileNotFoundError(f"mean_vector_path not found: {mean_path}")
            mean_vec_np = np.load(mean_path).astype(np.float32, copy=False)
            if mean_vec_np.ndim != 1 or mean_vec_np.shape[0] != input_dim:
                raise ValueError(
                    "mean_vector shape mismatch: expected "
                    f"({input_dim},), got {tuple(mean_vec_np.shape)}"
                )
            logger.info(msg=f"Loaded mean vector from {mean_path}")
        else:
            logger.info(msg="Computing activation mean vector for demeaning")
            mean_vec_np = compute_activation_mean(
                files=files,
                input_dim=input_dim,
                max_tokens_for_mean=max_tokens_for_mean,
            )
        saved_mean_path = run_dir / "mean_vector.npy"
        np.save(saved_mean_path, mean_vec_np)
        logger.info(msg=f"Saved mean vector to {saved_mean_path}")

    write_json(run_dir / "config.json", asdict(config))
    logger.info(msg=f"Run directory: {run_dir}")

    device_obj = resolve_device(device)
    logger.info(msg=f"Using device: {device_obj}")

    model = TopKSAE(input_dim=input_dim, latent_dim=latent_dim, k=k, tie_weights=tie_weights)
    model.to(device_obj)

    if weight_l2 > 0 and weight_decay != 0:
        logger.info(
            msg=(
                "Overriding weight_decay to 0.0 because "
                f"weight_l2={weight_l2} is set"
            )
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=optimizer_weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device_obj.type == "cuda")
    if grad_clip is not None and grad_clip > 0:
        logger.info(msg=f"Using gradient clipping with max_norm={grad_clip}")
    if grad_projection:
        logger.info(msg="Using gradient projection to preserve decoder directions")

    stats_path = run_dir / "stats.jsonl"
    feature_counts = torch.zeros(latent_dim, dtype=torch.long, device=device_obj)
    mean_vec_tensor: torch.Tensor | None = None
    if mean_vec_np is not None:
        mean_vec_tensor = torch.from_numpy(mean_vec_np).to(device_obj)

    start_step = 0
    if resume_from_checkpoint:
        resume_path = Path(resume_from_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume_from_checkpoint not found: {resume_path}")
        logger.info(msg=f"Resuming training from checkpoint {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device_obj)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        feature_counts = checkpoint.get(
            "feature_counts",
            torch.zeros(latent_dim, dtype=torch.long),
        ).to(device_obj)
        start_step = int(checkpoint.get("step", 0))
        if start_step >= max_steps:
            raise ValueError(
                f"resume checkpoint step {start_step} is already >= max_steps {max_steps}"
            )
    elif tied_init:
        with torch.no_grad():
            model.decoder.weight.copy_(model.encoder.weight.T)
        logger.info(msg="Applied tied initialization to decoder weights")

    rng = np.random.default_rng(seed)
    batch_iter = iter_activation_batches(files, batch_size, rng, shuffle_files, shuffle_within_file)

    logger.info(msg="Beginning optimization loop")
    stats_mode = "a" if start_step > 0 else "w"
    with stats_path.open(stats_mode) as stats_file:
        for step in range(start_step + 1, max_steps + 1):
            batch_np = next(batch_iter)
            batch = torch.from_numpy(batch_np).to(device_obj)
            if mean_vec_tensor is not None:
                batch = apply_demean(batch, mean_vec_tensor)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp and device_obj.type == "cuda"):
                input_batch = batch
                if pre_encoder_bias:
                    enc_input = input_batch - model.decoder.bias
                else:
                    enc_input = input_batch
                sparse = model.encode(enc_input)
                decoder_weight = get_decoder_weight(model, normalize_decoder)
                if model.tie_weights:
                    recon = torch.matmul(sparse, decoder_weight) + model.decoder.bias
                else:
                    recon = torch.matmul(sparse, decoder_weight.T) + model.decoder.bias
                mse_loss = torch.nn.functional.mse_loss(recon, input_batch)
                if weight_l2 > 0:
                    l2_penalty = (
                        model.encoder.weight.pow(2).sum()
                        + model.decoder.weight.pow(2).sum()
                    )
                    loss = mse_loss + (weight_l2 * l2_penalty)
                else:
                    l2_penalty = None
                    loss = mse_loss

            scaler.scale(loss).backward()
            if grad_clip is not None and grad_clip > 0 or grad_projection:
                scaler.unscale_(optimizer)
                if grad_projection:
                    if model.tie_weights:
                        weight = model.encoder.weight
                        grad = weight.grad
                        if grad is not None:
                            denom = weight.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8)
                            proj = (grad * weight).sum(dim=1, keepdim=True) / denom
                            grad -= proj * weight
                    else:
                        weight = model.decoder.weight
                        grad = weight.grad
                        if grad is not None:
                            denom = weight.pow(2).sum(dim=0, keepdim=True).clamp_min(1e-8)
                            proj = (grad * weight).sum(dim=0, keepdim=True) / denom
                            grad -= proj * weight
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if normalize_decoder:
                with torch.no_grad():
                    if model.tie_weights:
                        model.encoder.weight.data = normalize_weight_rows(model.encoder.weight.data)
                    else:
                        model.decoder.weight.data = normalize_weight_columns(
                            model.decoder.weight.data
                        )

            with torch.no_grad():
                active = (sparse > 0).sum(dim=0)
                feature_counts += active
                active_frac = (active > 0).float().mean().item()

            if step % log_every == 0 or step == 1:
                stats = {
                    "step": step,
                    "loss": float(loss.item()),
                    "mse_loss": float(mse_loss.item()),
                    "active_feature_frac": active_frac,
                }
                if l2_penalty is not None:
                    stats["l2_penalty"] = float(l2_penalty.item())
                stats_file.write(json.dumps(stats) + "\n")
                stats_file.flush()
                logger.info(
                    msg=(
                        f"Step {step}/{max_steps} mse={mse_loss.item():.6f} "
                        f"loss={loss.item():.6f} active_frac={active_frac:.4f}"
                    )
                )

            if step % save_every == 0 or step == max_steps:
                checkpoint = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "step": step,
                    "feature_counts": feature_counts.cpu(),
                }
                ckpt_path = run_dir / f"checkpoint_step_{step}.pt"
                torch.save(checkpoint, ckpt_path)
                logger.info(msg=f"Saved checkpoint: {ckpt_path}")

    logger.info(msg="Training complete")


if __name__ == "__main__":
    raise SystemExit(tapify(train_sae))
