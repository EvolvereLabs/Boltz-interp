"""Register the three intervention types on a loaded Boltz-1 model.

We never edit the ``nutz_and_boltz`` fork's forward. Everything here is a runtime hook or
monkeypatch, installed before ``trainer.predict`` and removed afterwards via the returned handle.

Confirmed anchors in the fork (branch ``feature/update-boltz``):
  - ``Boltz1.forward`` calls ``self.structure_module.sample(s_trunk=s, z_trunk=z, ...)``
    at ``model/model.py:352`` — ``s_trunk`` is a kwarg → C1 trunk-output = monkeypatch on ``sample``.
  - ``PairformerLayer`` returns ``(s, z)``            (``model/modules/trunk.py:488``)  → C1 depth sweep.
  - ``DiffusionTransformerLayer`` returns ``a``       (``model/modules/transformers.py:132``) → C2.

All three carry TensorLens / activation-checkpoint / torch.compile wrappers. We locate the *inner*
compute module by class name, so hooks fire regardless of wrapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor, nn

from .directions import _DEBUG, InterventionSpec


# --------------------------------------------------------------------------------------------------
# module lookup (robust to TensorLens / compile / checkpoint wrappers)
# --------------------------------------------------------------------------------------------------
def find_modules_by_classname(model: nn.Module, class_name: str) -> list[nn.Module]:
    """Return every submodule whose *own* class name equals ``class_name``, in registration order."""
    return [m for _, m in model.named_modules() if type(m).__name__ == class_name]


def pairformer_layers(model: nn.Module) -> list[nn.Module]:
    """Ordered list of ``PairformerLayer`` compute modules (index == trunk depth)."""
    return find_modules_by_classname(model, "PairformerLayer")


def diffusion_layers(model: nn.Module) -> list[nn.Module]:
    """Ordered list of ``DiffusionTransformerLayer`` compute modules."""
    return find_modules_by_classname(model, "DiffusionTransformerLayer")


# --------------------------------------------------------------------------------------------------
# handles
# --------------------------------------------------------------------------------------------------
@dataclass
class InterventionHandle:
    """Bundle of installed hooks / patches; call :meth:`remove` to fully restore the model."""

    _removers: list[Callable[[], None]] = field(default_factory=list)

    def add(self, remover: Callable[[], None]) -> None:
        self._removers.append(remover)

    def absorb(self, other: "InterventionHandle") -> None:
        """Take over another handle's removers (for bundling multi-site interventions)."""
        self._removers.extend(other._removers)
        other._removers = []

    def remove(self) -> None:
        for r in reversed(self._removers):
            r()
        self._removers.clear()

    def __enter__(self) -> "InterventionHandle":
        return self

    def __exit__(self, *exc: object) -> None:
        self.remove()


# --------------------------------------------------------------------------------------------------
# C1 — trunk-output conditioning (monkeypatch structure_module.sample)
# --------------------------------------------------------------------------------------------------
def register_trunk_output(
    model: nn.Module,
    spec: InterventionSpec,
    residue_idx: Tensor | None,
) -> InterventionHandle:
    """Project the direction out of the ``s_trunk`` kwarg diffusion reads (Condition 1, at L47)."""
    if spec.target != "trunk_output":
        raise ValueError(f"expected target='trunk_output', got {spec.target!r}")

    sm = model.structure_module
    original_sample = sm.sample

    def patched_sample(*args: object, **kwargs: object):  # noqa: ANN202
        if "s_trunk" in kwargs and kwargs["s_trunk"] is not None:
            kwargs["s_trunk"] = spec.apply(kwargs["s_trunk"], residue_idx)
        return original_sample(*args, **kwargs)

    sm.sample = patched_sample  # type: ignore[method-assign]

    handle = InterventionHandle()
    handle.add(lambda: setattr(sm, "sample", original_sample))
    return handle


# --------------------------------------------------------------------------------------------------
# C1 depth sweep — intermediate trunk layer (forward hook on PairformerLayer_{L})
# --------------------------------------------------------------------------------------------------
def register_trunk_depth(
    model: nn.Module,
    spec: InterventionSpec,
    residue_idx: Tensor | None,
    n_recycles: int | None = None,
    last_recycle_only: bool = False,
) -> InterventionHandle:
    """Project out of the single-repr ``s`` at PairformerLayer ``spec.layer`` output; it then
    propagates through the remaining trunk (Condition 1, depth-sweep variant).

    RECYCLE SEMANTICS: the trunk runs the whole layer stack ``recycling_steps + 1`` times, so this
    hook fires **once per recycle iteration**. Two matched choices:
      - ``last_recycle_only=False`` (default): ablate at layer L on *every* recycle pass — the concept
        can never be represented at depth L (a stronger, thorough ablation).
      - ``last_recycle_only=True`` with ``n_recycles`` (= recycling_steps + 1): ablate only on the
        final pass, so the trunk forms normally through recycling and the concept is removed just as
        the final conditioning (rec = n_recycles-1) is produced — the tightest analog to the rec-1
        direction the steering vector was fit on. Falls back to all-passes if the call count differs.
    """
    if spec.target != "trunk_depth":
        raise ValueError(f"expected target='trunk_depth', got {spec.target!r}")

    layers = pairformer_layers(model)
    if not 0 <= spec.layer < len(layers):  # type: ignore[operator]
        raise IndexError(f"PairformerLayer index {spec.layer} out of range (0..{len(layers) - 1}).")
    layer = layers[spec.layer]  # type: ignore[index]

    state = {"calls": 0}

    def hook(_module: nn.Module, _inp: tuple, out: tuple):  # noqa: ANN202
        # PairformerLayer returns (s, z); perturb s only.
        state["calls"] += 1
        # Apply on the final recycle pass AND any later call, never "== n_recycles" exactly: if the
        # layer's true call count ever differs from n_recycles (recycle semantics, confidence pass,
        # etc.), an exact match would silently skip on *every* pass → a no-op depth ablation. Using
        # ">=" degrades gracefully to "apply on the last pass we see" instead.
        skip = last_recycle_only and n_recycles is not None and state["calls"] < n_recycles
        if _DEBUG:
            print(f"[causal] trunk_depth L={spec.layer} call#{state['calls']} "
                  f"n_recycles={n_recycles} {'SKIP' if skip else 'APPLY'}", flush=True)
        if skip:
            return out  # earlier recycle pass; wait for the final one
        s, z = out
        return spec.apply(s, residue_idx), z

    h = layer.register_forward_hook(hook)
    handle = InterventionHandle()
    handle.add(h.remove)
    return handle


# --------------------------------------------------------------------------------------------------
# C2 — diffusion activations only (forward hook on every DiffusionTransformerLayer)
# --------------------------------------------------------------------------------------------------
def register_diffusion(
    model: nn.Module,
    spec: InterventionSpec,
    residue_idx: Tensor | None,
) -> InterventionHandle:
    """Project out of the diffusion token-repr ``a``, conditioning left intact (Condition 2, the
    discriminating control).

    LAYER MATCHING: ``spec.layer`` should be the diffusion layer the direction was fit on
    (DIFFUSION_PROBE_LAYER = 22, the module output the correlational study probed) — the direction is
    a property of that layer's activation space, so we project **only there**. Applying it at other
    diffusion layers would inject off-distribution noise, not a cleaner control. ``spec.layer=None``
    falls back to all layers (not recommended; kept only for ad-hoc use).

    STEP SEMANTICS: ``sample()`` loops over sampling steps and each step runs the diffusion
    transformer, so at the matched layer this hook fires **once per sampling step** — the concept is
    removed from that layer's representation across the whole denoising trajectory (the direction was
    fit at the final step, rec199)."""
    if spec.target != "diffusion":
        raise ValueError(f"expected target='diffusion', got {spec.target!r}")

    layers = diffusion_layers(model)
    if spec.layer is not None:
        layers = [layers[spec.layer]]

    def hook(_module: nn.Module, _inp: tuple, out: Tensor):  # noqa: ANN202
        # DiffusionTransformerLayer returns the token repr `a`.
        return spec.apply(out, residue_idx)

    handle = InterventionHandle()
    for layer in layers:
        h = layer.register_forward_hook(hook)
        handle.add(h.remove)
    return handle


# --------------------------------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------------------------------
def register_intervention(
    model: nn.Module,
    spec: InterventionSpec,
    residue_idx: Tensor | None,
    *,
    n_recycles: int | None = None,
    last_recycle_only: bool = False,
) -> InterventionHandle:
    """Install ``spec`` on ``model`` and return a handle. Baseline (``alpha == 0``) is a no-op.

    ``n_recycles`` / ``last_recycle_only`` only affect ``trunk_depth`` (they pin the depth-sweep
    ablation to the final recycle iteration so it matches the rec-1 direction); ignored otherwise.
    """
    if spec.alpha == 0.0 and spec.mode == "ablate":
        return InterventionHandle()  # α=0 ablation == identity → clean baseline pass
    if spec.target == "trunk_output":
        return register_trunk_output(model, spec, residue_idx)
    if spec.target == "trunk_depth":
        return register_trunk_depth(
            model, spec, residue_idx, n_recycles=n_recycles, last_recycle_only=last_recycle_only
        )
    if spec.target == "diffusion":
        return register_diffusion(model, spec, residue_idx)
    raise ValueError(f"unknown target {spec.target!r}")


def register_interventions(
    model: nn.Module,
    items: list[tuple[InterventionSpec, Tensor | None]],
    *,
    n_recycles: int | None = None,
    last_recycle_only: bool = False,
) -> InterventionHandle:
    """Install several interventions in ONE forward pass and return a single combined handle.

    ``items`` is a list of ``(spec, residue_idx)`` pairs. This is what enables multi-site ablation
    (E3): e.g. project the helix direction out of BOTH the trunk conditioning and the diffusion token
    repr simultaneously, to test whether necessity emerges only when a redundantly-encoded feature is
    removed from every site at once. Removing the returned handle restores all of them.
    """
    combined = InterventionHandle()
    for spec, residue_idx in items:
        combined.absorb(register_intervention(
            model, spec, residue_idx, n_recycles=n_recycles, last_recycle_only=last_recycle_only
        ))
    return combined


def residue_indices(positions: list[int], device: torch.device | str = "cpu") -> Tensor:
    """Build a long index tensor for the token axis from 0-based residue positions."""
    return torch.as_tensor(positions, dtype=torch.long, device=device)
