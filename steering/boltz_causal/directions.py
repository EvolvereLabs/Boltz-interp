"""Intervention directions and the projection math.

The two operations we ever apply to an activation tensor ``x`` (shape ``(..., N, D)`` where ``N`` is
the token/residue axis and ``D`` the feature axis), along a **unit** direction ``u`` (shape ``(D,)``):

- **ablate** (necessity / control):  ``x' = x - alpha * (x . u) u``  — projects the concept out.
- **add**    (sufficiency):          ``x' = x + alpha * u``          — writes the concept in.

Both can be restricted to a subset of residues (``residue_idx``) so we perturb only the
concept-relevant positions (e.g. the annotated cysteines) rather than the whole chain.

Directions come from ``SwissProt_annotations/export_probe_direction.py`` (held-out probe-raw weight
vector mapped to raw-activation space, ``coef_ / scaler.scale_``, then L2-normalised), plus a
random matched-norm control and, for the sufficiency variant, the SAE disulfide decoder row.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

# Set BOLTZ_CAUSAL_DEBUG=1 to print, at every hook fire, how much the intervention actually
# perturbed the activation (relative delta norm + mean projection coefficient). This is the number
# that tells "the hook did nothing" apart from "the hook fired but the effect is small".
_DEBUG = bool(int(os.environ.get("BOLTZ_CAUSAL_DEBUG", "0") or "0"))

Target = Literal["trunk_output", "trunk_depth", "diffusion"]
Mode = Literal["ablate", "add"]


def unit(direction: np.ndarray | Tensor) -> Tensor:
    """Return ``direction`` as an L2-normalised 1-D float tensor."""
    t = torch.as_tensor(np.asarray(direction), dtype=torch.float32).reshape(-1)
    norm = torch.linalg.vector_norm(t)
    if norm == 0:
        raise ValueError("Cannot normalise a zero direction vector.")
    return t / norm


def _apply_on_residues(x: Tensor, delta: Tensor, residue_idx: Tensor | None) -> Tensor:
    """Return ``x + delta`` everywhere, or only at ``residue_idx`` on the token axis (dim -2)."""
    if residue_idx is None:
        return x + delta
    residue_idx = residue_idx.to(x.device)  # model loads on CPU; activations are on GPU at hook time
    out = x.clone()
    out[..., residue_idx, :] = x[..., residue_idx, :] + delta[..., residue_idx, :]
    return out


def ablate_direction(
    x: Tensor,
    u: Tensor,
    alpha: float = 1.0,
    residue_idx: Tensor | None = None,
    mean: Tensor | None = None,
) -> Tensor:
    """Project the concept component out of ``x``: ``x - alpha ((x-mean).u) u``.

    ``alpha=1`` fully removes the component; ``alpha in (0,1)`` is a graded ablation (Panel G).

    ``mean`` (the probe's feature mean) makes this project the *mean-centred* activation, matching the
    probe geometry (logit = d_raw.(x-mean)). Without it, a few huge-magnitude activation dims make
    ``u.mean`` dominate ``u.x`` and the ablation removes a constant offset, not the concept. ``mean=None``
    keeps the legacy raw projection for old direction files.
    """
    u = u.to(dtype=x.dtype, device=x.device)
    xc = x if mean is None else x - mean.to(dtype=x.dtype, device=x.device)
    coeff = torch.tensordot(xc, u, dims=([-1], [0]))  # (..., N)
    delta = -alpha * coeff.unsqueeze(-1) * u  # (..., N, D)
    return _apply_on_residues(x, delta, residue_idx)


def add_direction(
    x: Tensor,
    u: Tensor,
    alpha: float = 1.0,
    residue_idx: Tensor | None = None,
) -> Tensor:
    """Write the concept in: ``x + alpha * u`` (sufficiency variant, dose = ``alpha``).

    ``alpha`` is in raw-activation units. A convenient scale is a multiple of the concept's typical
    positive projection magnitude; callers can pre-scale ``alpha`` accordingly.
    """
    u = u.to(dtype=x.dtype, device=x.device)
    delta = alpha * u.expand(*x.shape[:-1], u.shape[0])  # broadcast to (..., N, D)
    return _apply_on_residues(x, delta, residue_idx)


@dataclass
class InterventionSpec:
    """One intervention to apply during a single Boltz forward pass.

    Attributes:
        concept: Concept name (e.g. ``"disulfide_bond"``, ``"helix"``, ``"random"``).
        direction: Unit direction in raw-activation space, shape ``(D,)``. ``D`` must match the
            activation the intervention targets (trunk single-repr for trunk_* targets; diffusion
            token-repr for the diffusion target).
        target: Where to apply — the trunk output conditioning, an intermediate trunk layer, or
            inside the diffusion transformer.
        mode: ``"ablate"`` (project out) or ``"add"`` (write in).
        alpha: Strength. 1.0 = full ablation; a sweep over (0,1] is the graded panel.
        layer: For ``trunk_depth`` the PairformerLayer index; for ``diffusion`` an optional single
            layer index (``None`` = all diffusion layers). Ignored for ``trunk_output``.
        residue_key: Which residues to perturb, resolved per protein at registration time
            (e.g. ``"cys"`` = annotated cysteines, ``"all"``). Indices themselves are passed to the
            hook registrar, not stored here.
    """

    concept: str
    direction: Tensor
    target: Target
    mode: Mode = "ablate"
    alpha: float = 1.0
    layer: int | None = None
    residue_key: str = "concept"
    mean: Tensor | None = None  # probe feature mean -> mean-centred projection (see ablate_direction)

    def __post_init__(self) -> None:
        self.direction = unit(self.direction)
        if self.mean is not None:
            self.mean = torch.as_tensor(np.asarray(self.mean), dtype=torch.float32).reshape(-1)
        if self.target == "trunk_depth" and self.layer is None:
            raise ValueError("trunk_depth intervention requires a `layer` (PairformerLayer index).")

    def apply(self, x: Tensor, residue_idx: Tensor | None) -> Tensor:
        """Apply this spec to an activation tensor ``x`` at the given residues."""
        if self.mode == "ablate":
            out = ablate_direction(x, self.direction, self.alpha, residue_idx, self.mean)
        else:
            out = add_direction(x, self.direction, self.alpha, residue_idx)
        if _DEBUG:
            self._log_effect(x, out, residue_idx)
        return out

    def _log_effect(self, x: Tensor, out: Tensor, residue_idx: Tensor | None) -> None:
        """Print the realized perturbation magnitude (only affected residues)."""
        u = self.direction.to(dtype=x.dtype, device=x.device)
        xc = x if self.mean is None else x - self.mean.to(dtype=x.dtype, device=x.device)
        if residue_idx is not None:
            ridx = residue_idx.to(x.device)
            xr, outr, xcr = x[..., ridx, :], out[..., ridx, :], xc[..., ridx, :]
            scope = f"{len(ridx)} res"
        else:
            xr, outr, xcr = x, out, xc
            scope = "all res"
        xn = torch.linalg.vector_norm(xr)
        dn = torch.linalg.vector_norm(outr - xr)
        rel = float(dn / xn) if float(xn) > 0 else float("nan")
        coeff = torch.tensordot(xcr, u, dims=([-1], [0])).abs().mean()
        print(
            f"[causal] {self.concept}/{self.target}/{self.mode} a={self.alpha} "
            f"L={self.layer} [{scope}] |dx|/|x|={rel:.4f} mean|x.u|={float(coeff):.3f} "
            f"|x|={float(xn):.1f}",
            flush=True,
        )


@dataclass
class DirectionBank:
    """Directions exported by ``export_probe_direction.py``, keyed ``{concept}@{where}``.

    ``where`` is ``trunk_L{n}`` for pairformer layer n, or ``diffusion`` for the diffusion-space fit.
    """

    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)
    means: dict[str, np.ndarray] = field(default_factory=dict)

    def get(self, concept: str, where: str) -> np.ndarray:
        key = f"{concept}@{where}"
        if key not in self.vectors:
            raise KeyError(f"No direction {key!r}; available: {sorted(self.vectors)}")
        return self.vectors[key]

    def get_mean(self, concept: str, where: str) -> np.ndarray | None:
        """Feature mean for ``{concept}@{where}`` (for mean-centred projection), or None if absent
        (legacy direction files without exported means)."""
        return self.means.get(f"{concept}@{where}")

    def random_like(self, concept: str, where: str, seed: int = 0) -> np.ndarray:
        """A random direction matched in norm to ``{concept}@{where}`` (specificity control S1)."""
        ref = self.get(concept, where)
        rng = np.random.default_rng(seed)
        r = rng.standard_normal(ref.shape).astype(np.float32)
        r = r / np.linalg.norm(r)
        return r * float(np.linalg.norm(ref))


def load_directions(path: str | Path) -> DirectionBank:
    """Load a ``.npz`` produced by ``export_probe_direction.py`` into a :class:`DirectionBank`."""
    data = np.load(Path(path), allow_pickle=True)
    meta = {}
    vectors = {}
    means = {}
    for key in data.files:
        if key == "__meta__":
            meta = data[key].item()
        elif key.endswith(".mean"):
            means[key[: -len(".mean")]] = np.asarray(data[key], dtype=np.float32)
        else:
            vectors[key] = np.asarray(data[key], dtype=np.float32)
    return DirectionBank(vectors=vectors, meta=meta, means=means)
