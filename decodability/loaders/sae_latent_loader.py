"""SAE-aware activation loader for auto-interpretability workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embeddings_concepts_evaluation import EmbeddingLoader
from eval_sae import load_config_from_run
from sae_utils import TopKSAE, apply_demean, list_activation_files, load_activation, resolve_device

LOGGER = logging.getLogger(__name__)


def load_sae_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TopKSAE, dict[str, Any]]:
    """Load a trained TopKSAE and its adjacent run config.

    Args:
        checkpoint_path: Path to ``checkpoint_step_*.pt``.
        device: Torch device used for inference.

    Returns:
        Loaded SAE model and the run config loaded from ``config.json``.

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
        KeyError: If required shape fields are missing from the config.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")

    cfg = load_config_from_run(checkpoint_path)
    input_dim = int(cfg["input_dim"])
    latent_dim = int(cfg["latent_dim"])
    k = int(cfg["k"])
    tie_weights = bool(cfg.get("tie_weights", False))

    model = TopKSAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        k=k,
        tie_weights=tie_weights,
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    LOGGER.info(
        msg=(
            f"Loaded SAE checkpoint={checkpoint_path} input_dim={input_dim} "
            f"latent_dim={latent_dim} k={k}"
        )
    )
    return model, cfg


def resolve_mean_vector(
    checkpoint_path: Path,
    cfg: dict[str, Any],
    mean_vector_path: str | None = None,
) -> np.ndarray | None:
    """Resolve and load the training-set mean vector if the SAE was demeaned.

    Args:
        checkpoint_path: Path to the SAE checkpoint.
        cfg: Run config loaded from the checkpoint directory.
        mean_vector_path: Optional explicit override for the mean vector.

    Returns:
        Mean vector as ``float32``, or ``None`` when the run did not use demeaning.

    Raises:
        FileNotFoundError: If the resolved mean vector path is missing.
        ValueError: If demeaning is enabled but no mean path can be resolved.
    """
    if not bool(cfg.get("demean_embeddings", False)):
        return None

    resolved = mean_vector_path
    if resolved is None and cfg.get("mean_vector_file"):
        resolved = str(checkpoint_path.parent / str(cfg["mean_vector_file"]))
    if resolved is None:
        raise ValueError("demean_embeddings=True but no mean vector path could be resolved.")

    mean_path = Path(resolved)
    if not mean_path.exists():
        raise FileNotFoundError(f"Mean vector not found: {mean_path}")

    mean_vec = np.load(mean_path).astype(np.float32, copy=False)
    expected_dim = int(cfg["input_dim"])
    if mean_vec.ndim != 1 or mean_vec.shape[0] != expected_dim:
        raise ValueError(f"Mean vector shape mismatch: expected ({expected_dim},), got {mean_vec.shape}")
    LOGGER.info(msg=f"Loaded SAE mean vector from {mean_path}")
    return mean_vec


class SAELatentActivationLoader(EmbeddingLoader):
    """Load raw Boltz activations and return sparse SAE latent activations.

    The preprocessing mirrors ``train_sae.py`` and ``eval_sae.py``:
    raw activations are optionally demeaned with the run-local mean vector, the
    decoder bias is optionally subtracted before encoding, and ``model.encode``
    applies ReLU + Top-K sparsity.
    """

    def __init__(
        self,
        embeddings_dir: str | Path,
        subfolder: str,
        rec: int,
        sae_checkpoint: str | Path,
        device: str | None = None,
        mean_vector_path: str | None = None,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.subfolder = subfolder
        self.rec = int(rec)
        self.checkpoint_path = Path(sae_checkpoint)
        self.device = resolve_device(device)
        self.model, self.cfg = load_sae_from_checkpoint(self.checkpoint_path, self.device)
        self.mean_vec = resolve_mean_vector(self.checkpoint_path, self.cfg, mean_vector_path)
        self.input_dim = int(self.cfg["input_dim"])
        self.pre_encoder_bias = bool(self.cfg.get("pre_encoder_bias", False))
        self._available: list[str] | None = None

    def _path_for(self, protein_id: str) -> Path:
        """Return the raw activation file path for one protein."""
        layer_dir = self.embeddings_dir / protein_id / self.subfolder
        npz_path = layer_dir / f"output_{self.rec}.npz"
        if npz_path.exists():
            return npz_path
        npy_gz_path = layer_dir / f"output_{self.rec}.npy.gz"
        if npy_gz_path.exists():
            return npy_gz_path
        raise FileNotFoundError(f"No output_{self.rec}.npz/.npy.gz under {layer_dir}")

    def load(self, protein_id: str) -> np.ndarray | None:
        """Load one protein and return ``(seq_len, latent_dim)`` SAE latents."""
        try:
            activation_path = self._path_for(protein_id)
        except FileNotFoundError:
            LOGGER.debug(msg=f"Missing activation file for {protein_id}")
            return None

        raw = load_activation(activation_path)
        if raw.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dim mismatch for {protein_id}: expected {self.input_dim}, got {raw.shape[1]}"
            )
        if self.mean_vec is not None:
            raw = apply_demean(raw, self.mean_vec)

        batch = torch.from_numpy(raw).to(self.device)
        if self.pre_encoder_bias:
            batch = batch - self.model.decoder.bias
        with torch.no_grad():
            latents = self.model.encode(batch)
        return latents.cpu().numpy()

    def list_available(self) -> list[str]:
        """List protein IDs with raw activation files available."""
        if self._available is None:
            files = list_activation_files(self.embeddings_dir, self.subfolder, self.rec, None)
            self._available = sorted({path.parent.parent.name for path in files})
        return self._available
