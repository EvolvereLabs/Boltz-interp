"""Raw-activation loader: the single-neuron baseline for concept-F1 benchmarks.

This mirrors :class:`loaders.sae_latent_loader.SAELatentActivationLoader` but
returns the *raw* Boltz pairformer activations (one column per neuron) instead of
SAE latents. It therefore lets the existing :class:`ConceptEvaluator` machinery
score every individual neuron against a concept set, giving a like-for-like
"best single neuron" baseline to compare against the SAE's "best single latent".

The raw activation files are exactly the ones the SAE loader reads
(``downloads_layer{N}/<id>/<subfolder>/output_{rec}.npz``) -- no extra downloads
are needed to run the single-neuron baseline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from embeddings_concepts_evaluation import EmbeddingLoader
from sae_utils import list_activation_files, load_activation

LOGGER = logging.getLogger(__name__)


class RawActivationLoader(EmbeddingLoader):
    """Load raw Boltz activations (``(seq_len, n_neurons)``) for single-neuron F1.

    Args:
        embeddings_dir: Root directory of per-protein activation folders.
        subfolder: Layer subfolder name, e.g. ``"PairformerLayer_20 s from PairformerLayer"``.
        rec: Recycle/output index of the activation file to load.
        expected_dim: Optional sanity check on the neuron dimension.
    """

    def __init__(
        self,
        embeddings_dir: str | Path,
        subfolder: str,
        rec: int,
        expected_dim: int | None = None,
    ) -> None:
        self.embeddings_dir = Path(embeddings_dir)
        self.subfolder = subfolder
        self.rec = int(rec)
        self.expected_dim = expected_dim
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
        """Load one protein and return its raw ``(seq_len, n_neurons)`` activations."""
        try:
            activation_path = self._path_for(protein_id)
        except FileNotFoundError:
            LOGGER.debug(msg=f"Missing activation file for {protein_id}")
            return None

        raw = load_activation(activation_path)
        if self.expected_dim is not None and raw.shape[1] != self.expected_dim:
            raise ValueError(
                f"Neuron dim mismatch for {protein_id}: expected {self.expected_dim}, "
                f"got {raw.shape[1]}"
            )
        return raw

    def list_available(self) -> list[str]:
        """List protein IDs with raw activation files available."""
        if self._available is None:
            files = list_activation_files(self.embeddings_dir, self.subfolder, self.rec, None)
            self._available = sorted({path.parent.parent.name for path in files})
        return self._available
