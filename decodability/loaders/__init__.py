"""Shared activation loaders for the Paper A decodability pipeline.

- ``raw_activation_loader`` — load raw Boltz-1 per-residue activations for probing.
- ``sae_latent_loader`` — load an SAE checkpoint, demean, and encode activations to latents.

Both are reused across the benchmark, amino-acid-sanity, and precision/recall scripts.
"""

__version__ = "0.1.0"
