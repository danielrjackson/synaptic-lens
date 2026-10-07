"""Exact affine/ReLU target records, independent of learned reader dimensions.

Hidden unit IDs establish correspondence only. They are never model features.
This initial adapter rejects unsupported modules instead of dropping them.
"""

import hashlib
from dataclasses import dataclass, replace

import torch
from torch import nn


@dataclass(frozen=True)
class ParameterLocation:
    owner: int
    name: str
    shape: tuple[int, ...]
    coordinate: tuple[int, ...]


@dataclass(frozen=True)
class ScalarMaskMetadata:
    """Audit/query metadata, never learned numeric input features."""

    missing: bool = True
    corruption_applied: bool = False
    operator_version: str = "single-scalar-mask-v1"

    def __post_init__(self):
        if type(self.missing) is not bool or type(self.corruption_applied) is not bool:
            raise ValueError("Mask flags must be explicit booleans")
        if not isinstance(self.operator_version, str) or not self.operator_version:
            raise ValueError("Mask operator version must be explicit")


@dataclass(frozen=True)
class AffineRecords:
    widths: tuple[int, ...]
    weights: tuple[torch.Tensor, ...]
    biases: tuple[torch.Tensor | None, ...]
    relu: tuple[bool, ...]
    weight_present: tuple[torch.Tensor, ...]
    bias_present: tuple[torch.Tensor | None, ...]
    schema: str = "affine-relu-scalar-v1"

    @property
    def record_count(self) -> int:
        return sum(self.widths) + sum(w.numel() for w in self.weights)

    @property
    def parameter_count(self) -> int:
        return sum(w.numel() for w in self.weights) + sum(
            b.numel() for b in self.biases if b is not None
        )

    @property
    def complete(self) -> bool:
        return all(bool(p.all()) for p in self.weight_present) and all(
            bool(p.all()) for p in self.bias_present if p is not None
        )

    def identity(self) -> str:
        """Bind values, missing markers, structure and preprocessing version."""
        digest = hashlib.sha256()
        digest.update(repr((self.schema, self.widths, self.relu)).encode())
        for group in (self.weights, self.biases, self.weight_present, self.bias_present):
            for tensor in group:
                if tensor is None:
                    digest.update(b"absent")
                else:
                    value = tensor.detach().cpu().contiguous()
                    digest.update(repr((str(value.dtype), tuple(value.shape))).encode())
                    digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def masked(self, location: ParameterLocation) -> "AffineRecords":
        """Remove a scalar before deriving any reader features; keep its location."""
        if location.owner < 0 or location.owner >= len(self.weights):
            raise ValueError("Unknown parameter owner")
        if location.name not in ("weight", "bias"):
            raise ValueError("Unsupported parameter name")
        values = self.weights if location.name == "weight" else self.biases
        flags = self.weight_present if location.name == "weight" else self.bias_present
        source = values[location.owner]
        if source is None or tuple(source.shape) != location.shape:
            raise ValueError("Parameter shape does not match source records")
        if len(location.coordinate) != source.ndim or any(
            i < 0 or i >= size for i, size in zip(location.coordinate, source.shape)
        ):
            raise ValueError("Invalid scalar coordinate")
        new_values = list(values)
        new_flags = list(flags)
        new_values[location.owner] = source.clone()
        new_flags[location.owner] = flags[location.owner].clone()
        new_values[location.owner][location.coordinate] = 0
        new_flags[location.owner][location.coordinate] = False
        if location.name == "weight":
            return replace(self, weights=tuple(new_values), weight_present=tuple(new_flags))
        return replace(self, biases=tuple(new_values), bias_present=tuple(new_flags))

    def evaluate(self, inputs: torch.Tensor, intervention=None) -> torch.Tensor:
        """Reconstruct supported target semantics, optionally replacing one unit.

        intervention is (layer, unit, replacement); layer 0 is the first affine
        output, after its optional activation. This is adapter validation, not Lens.
        """
        if not self.complete:
            raise ValueError("Cannot evaluate an incomplete target")
        if inputs.shape[-1] != self.widths[0]:
            raise ValueError("Input width mismatch")
        if intervention is not None:
            layer, unit, _ = intervention
            if not 0 <= layer < len(self.weights) or not 0 <= unit < self.widths[layer + 1]:
                raise ValueError("Unknown intervention unit")
        result = inputs
        for layer, (weight, bias, activation) in enumerate(
            zip(self.weights, self.biases, self.relu)
        ):
            result = nn.functional.linear(result, weight, bias)
            if activation:
                result = result.relu()
            if intervention is not None and layer == intervention[0]:
                result = result.clone()
                result[..., intervention[1]] = intervention[2]
        return result


def affine_relu_records(model: nn.Sequential) -> AffineRecords:
    """Copy every numeric parameter from a flat Sequential of Linear/ReLU.

    Sharing is rejected by this initial adapter until an explicit sharing adapter
    exists. No pointers to unread parameters or silent unsupported operations.
    """
    if type(model) is not nn.Sequential:
        raise ValueError("Expected a flat Sequential affine/ReLU target")
    weights, biases, relu, widths = [], [], [], []
    seen = set()
    for module in model:
        if type(module) is nn.Linear:
            for parameter in module.parameters(recurse=False):
                if id(parameter) in seen:
                    raise ValueError("Shared parameters need an explicit sharing adapter")
                seen.add(id(parameter))
                if not bool(torch.isfinite(parameter).all()):
                    raise ValueError("Nonfinite target parameter")
            if widths and widths[-1] != module.in_features:
                raise ValueError("Disconnected affine dimensions")
            if not widths:
                widths.append(module.in_features)
            widths.append(module.out_features)
            weights.append(module.weight.detach().clone())
            biases.append(None if module.bias is None else module.bias.detach().clone())
            relu.append(False)
        elif type(module) is nn.ReLU:
            if not relu or relu[-1]:
                raise ValueError("ReLU must follow exactly one affine operation")
            relu[-1] = True
        else:
            raise ValueError(f"Unsupported target operation: {type(module).__name__}")
    if not weights:
        raise ValueError("Target has no affine operations")
    return AffineRecords(
        tuple(widths), tuple(weights), tuple(biases), tuple(relu),
        tuple(torch.ones_like(w, dtype=torch.bool) for w in weights),
        tuple(None if b is None else torch.ones_like(b, dtype=torch.bool) for b in biases),
    )
