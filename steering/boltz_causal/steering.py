"""Baseline activation-scale capture for steering calibration.

Shared by the steering scripts (probe_batch, probe_combined). Lives in the installed ``boltz_causal``
package so scripts never import each other -- ``python scripts/probe_batch.py`` puts ``scripts/`` on
sys.path, not the repo root, so a ``from scripts.x import y`` would fail on the EC2 box.
"""

from __future__ import annotations

import torch

from .config import DIFFUSION_PROBE_LAYER
from .hooks import diffusion_layers


def proj_stats(x: torch.Tensor, u: torch.Tensor, box: dict, mean: torch.Tensor | None = None) -> None:
    """Accumulate per-token RMS norm and mean |(x-mean) . u| projection onto the (unit) direction.
    Demeaning matches the probe geometry, so the additive calibration tracks the concept fluctuation
    scale rather than the huge-dim offset."""
    u = (u / torch.linalg.vector_norm(u)).to(dtype=x.dtype, device=x.device)
    xc = x if mean is None else x - mean.to(dtype=x.dtype, device=x.device)
    box.setdefault("_rms", []).append(float(torch.linalg.vector_norm(x, dim=-1).mean()))
    box.setdefault("_proj", []).append(float(torch.tensordot(xc, u, dims=([-1], [0])).abs().mean()))


def finalize(box: dict) -> None:
    box["rms"] = sum(box["_rms"]) / len(box["_rms"]) if box.get("_rms") else float("nan")
    box["proj"] = sum(box["_proj"]) / len(box["_proj"]) if box.get("_proj") else float("nan")


def capture_scale(model, direction: torch.Tensor, site: str, mean: torch.Tensor | None = None) -> dict:
    """Record, on the next (baseline) pass and without modifying anything, the activation scale where
    the intervention will act: the per-token RMS norm and the mean |(x-mean) . u| projection onto the
    concept direction. The projection is the right scale for an *additive* nudge -- a small multiple
    writes the concept in without swamping the representation (multiples of the full RMS are an
    off-manifold wrecking ball: pLDDT collapses). Captures at s_trunk for the trunk site, or the
    diffusion probe layer's token repr for the diffusion site (averaged over sampling steps).

    Returns a dict; call ``box["_restore"]()`` after the baseline pass to remove the hook and populate
    ``box["rms"]`` / ``box["proj"]``."""
    box: dict = {}
    if site == "trunk":
        sm = model.structure_module
        original = sm.sample

        def capture_sample(*args, **kwargs):
            s = kwargs.get("s_trunk")
            if s is not None:
                proj_stats(s, direction, box, mean)
            return original(*args, **kwargs)

        sm.sample = capture_sample  # type: ignore[method-assign]
        box["_restore"] = lambda: (setattr(sm, "sample", original), finalize(box))[-1]
    else:
        layer = diffusion_layers(model)[DIFFUSION_PROBE_LAYER]

        def capture_hook(_m, _i, out):
            proj_stats(out, direction, box, mean)  # DiffusionTransformerLayer returns token repr `a`

        h = layer.register_forward_hook(capture_hook)
        box["_restore"] = lambda: (h.remove(), finalize(box))[-1]
    return box
