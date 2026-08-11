"""Unit tests for the projection math (no GPU / no Boltz needed)."""

from __future__ import annotations

import numpy as np
import torch

from boltz_causal.directions import (
    DirectionBank,
    InterventionSpec,
    ablate_direction,
    add_direction,
    unit,
)


def test_unit_normalises():
    u = unit(np.array([3.0, 0.0, 4.0]))
    assert torch.allclose(torch.linalg.vector_norm(u), torch.tensor(1.0), atol=1e-6)


def test_full_ablation_removes_component():
    torch.manual_seed(0)
    u = unit(torch.tensor([1.0, 0.0, 0.0]))
    x = torch.randn(2, 5, 3)
    out = ablate_direction(x, u, alpha=1.0)
    coeff = torch.tensordot(out, u, dims=([-1], [0]))
    assert torch.allclose(coeff, torch.zeros_like(coeff), atol=1e-5)


def test_graded_ablation_is_partial():
    u = unit(torch.tensor([0.0, 1.0, 0.0]))
    x = torch.randn(4, 3)
    before = torch.tensordot(x, u, dims=([-1], [0]))
    after = torch.tensordot(ablate_direction(x, u, alpha=0.5), u, dims=([-1], [0]))
    assert torch.allclose(after, 0.5 * before, atol=1e-5)


def test_residue_mask_only_touches_selected():
    u = unit(torch.tensor([1.0, 0.0]))
    x = torch.randn(1, 4, 2)
    idx = torch.tensor([1, 3])
    out = ablate_direction(x, u, alpha=1.0, residue_idx=idx)
    # untouched residues identical
    assert torch.allclose(out[:, 0], x[:, 0])
    assert torch.allclose(out[:, 2], x[:, 2])
    # touched residues have zero component
    coeff = torch.tensordot(out[:, idx], u, dims=([-1], [0]))
    assert torch.allclose(coeff, torch.zeros_like(coeff), atol=1e-5)


def test_add_writes_direction():
    u = unit(torch.tensor([0.0, 0.0, 1.0]))
    x = torch.zeros(3)
    out = add_direction(x, u, alpha=2.0)
    assert torch.allclose(out, 2.0 * u, atol=1e-6)


def test_spec_normalises_and_dispatches():
    spec = InterventionSpec("disulfide_bond", np.array([2.0, 0.0]), "trunk_output", alpha=1.0)
    assert torch.allclose(torch.linalg.vector_norm(spec.direction), torch.tensor(1.0), atol=1e-6)
    x = torch.randn(3, 2)
    out = spec.apply(x, None)
    coeff = torch.tensordot(out, spec.direction, dims=([-1], [0]))
    assert torch.allclose(coeff, torch.zeros_like(coeff), atol=1e-5)


def test_random_like_matches_norm():
    ref = np.array([3.0, 4.0, 0.0], dtype=np.float32)  # norm 5
    bank = DirectionBank(vectors={"disulfide_bond@trunk_L47": ref})
    r = bank.random_like("disulfide_bond", "trunk_L47", seed=1)
    assert np.isclose(np.linalg.norm(r), 5.0, atol=1e-5)
