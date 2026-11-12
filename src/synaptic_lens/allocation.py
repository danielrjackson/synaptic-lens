"""Detached allocation policies and antithetic optimizer mechanisms.

Study schemas, controller architectures and learner orchestration stay in adapters.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


def normalized_multipliers(
    scores: torch.Tensor,
    coefficients: torch.Tensor,
    *,
    floor: float = 0.1,
    log_clip: float = math.log(4.0),
) -> torch.Tensor:
    """Positive, coefficient-weighted-mean-one multipliers from clipped scores."""
    if (
        scores.ndim != 1
        or coefficients.shape != scores.shape
        or scores.numel() == 0
        or not bool(torch.isfinite(scores).all())
        or not bool(torch.isfinite(coefficients).all())
        or bool((coefficients <= 0).any())
        or not 0 < floor < 1
        or log_clip <= 0
    ):
        raise ValueError("allocator score or averaging coefficients are invalid")
    coefficient_sum = coefficients.sum()
    if not torch.isclose(coefficient_sum, coefficient_sum.new_tensor(1.0), rtol=1e-5, atol=1e-6):
        raise ValueError("original update coefficients must sum to one")
    raw = scores.clamp(-log_clip, log_clip).exp()
    multipliers = floor + (1.0 - floor) * raw / torch.sum(coefficients * raw)
    return multipliers


def normalized_direction(
    controller: nn.Module, *, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, ...]:
    """Draw one global-L2-normalized direction before evaluating its two branches."""
    parameters = tuple(controller.parameters())
    directions = tuple(
        torch.randn(
            parameter.shape, dtype=parameter.dtype, device=parameter.device, generator=generator
        )
        for parameter in parameters
    )
    norm = torch.sqrt(sum(direction.square().sum() for direction in directions))
    if not bool(torch.isfinite(norm)) or norm.item() == 0:
        raise ValueError("finite-difference direction is invalid")
    return tuple(direction / norm for direction in directions)


def antithetic_scores(
    controller: nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    direction: tuple[torch.Tensor, ...],
    perturbation_scale: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate + and - policies using the exact externally retained direction."""
    parameters = tuple(controller.parameters())
    if len(direction) != len(parameters) or any(
        delta.shape != parameter.shape or delta.device != parameter.device
        for parameter, delta in zip(parameters, direction, strict=True)
    ):
        raise ValueError("finite-difference direction does not match controller parameters")
    originals = tuple(parameter.detach().clone() for parameter in parameters)
    with torch.no_grad():
        try:
            for parameter, original, delta in zip(parameters, originals, direction, strict=True):
                parameter.copy_(original + perturbation_scale * delta)
            score_plus = controller(**inputs).detach()
            for parameter, original, delta in zip(parameters, originals, direction, strict=True):
                parameter.copy_(original - perturbation_scale * delta)
            score_minus = controller(**inputs).detach()
        finally:
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
    return score_plus, score_minus


def antithetic_adam_update(
    controller: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    direction: tuple[torch.Tensor, ...],
    reward_plus: float,
    reward_minus: float,
    perturbation_scale: float = 0.01,
) -> None:
    """Apply a controller Adam step using rewards for this exact prior direction."""
    if (
        not math.isfinite(reward_plus)
        or not math.isfinite(reward_minus)
        or perturbation_scale <= 0
        or not math.isfinite(perturbation_scale)
    ):
        raise ValueError("finite-difference rewards and scale must be finite")
    parameters = tuple(controller.parameters())
    if len(direction) != len(parameters) or any(
        delta.shape != parameter.shape or delta.device != parameter.device
        for parameter, delta in zip(parameters, direction, strict=True)
    ):
        raise ValueError("finite-difference direction does not match controller parameters")
    gradient = -(reward_plus - reward_minus) / (2.0 * perturbation_scale)
    optimizer.zero_grad(set_to_none=True)
    for parameter, delta in zip(parameters, direction, strict=True):
        parameter.grad = gradient * delta
    optimizer.step()


def policy_scores(
    policy: Literal["uniform", "sqrt_loss", "learned"],
    *,
    unreduced_loss: torch.Tensor,
    learned_scores: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return log-weights; normalization and the floor are shared by all policies."""
    if unreduced_loss.ndim != 1 or not bool(torch.isfinite(unreduced_loss).all()):
        raise ValueError("per-query loss must be a finite vector")
    if policy == "uniform":
        return torch.zeros_like(unreduced_loss).detach()
    if policy == "sqrt_loss":
        return 0.5 * torch.log(unreduced_loss.detach().clamp_min(1e-12))
    if policy == "learned" and learned_scores is not None:
        if learned_scores.shape != unreduced_loss.shape or not bool(
            torch.isfinite(learned_scores).all()
        ):
            raise ValueError("learned controller scores must align with query losses")
        return learned_scores.detach()
    raise ValueError("pilot policy or learned controller scores are invalid")
