"""Root-preserving preparation for the published SYN88 trajectory reader.

This module performs no artifact I/O, optimization or holdout selection. Callers
supply manifest-verified numeric arrays in rule/age order from syn88_targets.
"""
from dataclasses import dataclass

import torch

from .syn88_targets import AGES, RULES
from .targets import addition_suite
from .trajectory import canonicalize, normalize_parts


@dataclass(frozen=True)
class TrajectoryPopulation:
    """One partition: roots, two paired rules, eight ages, then coefficients."""

    root_ids: tuple[int, ...]
    weights: torch.Tensor
    addition_correct: torch.Tensor

    def __post_init__(self):
        ids = self.root_ids
        if not isinstance(ids, tuple) or not ids or any(type(i) is not int or not 0 <= i < 2304 for i in ids):
            raise ValueError("Expected nonempty SYN88 root IDs as an integer tuple")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate root IDs within partition")
        if self.weights.shape != (len(ids), len(RULES), len(AGES), 1299):
            raise ValueError("Expected weights [roots, 2 rules, 8 ages, 1299 coefficients]")
        if self.weights.dtype != torch.float32 or not torch.isfinite(self.weights).all():
            raise ValueError("Expected finite float32 target coefficients")
        if self.addition_correct.shape != (len(ids), len(RULES), len(AGES), 100):
            raise ValueError("Expected addition labels [roots, 2 rules, 8 ages, 100 queries]")
        if not ((self.addition_correct == 0) | (self.addition_correct == 1)).all():
            raise ValueError("Expected binary addition labels")
        if self.weights.device != self.addition_correct.device:
            raise ValueError("Weights and labels must share a device")
        if not torch.equal(self.weights[:, 0, 0], self.weights[:, 1, 0]):
            raise ValueError("Paired rules must share birth weights")


def require_disjoint_roots(*populations):
    """Keep all ages and paired rules from a root in exactly one partition."""
    seen = set()
    for population in populations:
        if seen.intersection(population.root_ids):
            raise ValueError("Root overlap across partitions")
        seen.update(population.root_ids)


def canonical_parts(population):
    """Preserve historical float64 canonicalization and verify native labels."""
    # Revalidate because frozen dataclasses do not make tensor storage immutable.
    population.__post_init__()
    _, query, answers = addition_suite()
    device = population.weights.device
    query, answers = query.to(device), answers.to(device)
    incoming, hidden_bias, edges, output_bias, activation = canonicalize(
        population.weights, query)
    logits = torch.einsum("...qn,...nk->...qk", activation, edges) + output_bias.unsqueeze(-2)
    raw = population.weights.double()
    native_hidden = torch.relu(torch.einsum("qj,...nj->...qn", query.double(),
        raw[..., :640].reshape(*raw.shape[:-1], 32, 20)) + raw[..., 640:672].unsqueeze(-2))
    native_edges = raw[..., 672:1280].reshape(*raw.shape[:-1], 19, 32).transpose(-1, -2)
    native_logits = torch.einsum("...qn,...nk->...qk", native_hidden, native_edges) + raw[..., 1280:1299].unsqueeze(-2)
    if (native_logits - logits).abs().max() > 2e-6:
        raise ValueError("Canonicalization changed native target logits")
    if not torch.equal(native_logits.argmax(-1).eq(answers), population.addition_correct.bool()):
        raise ValueError("Saved addition labels disagree with native target predictions")
    return {"incoming": incoming, "hidden_bias": hidden_bias, "edges": edges,
            "output_bias": output_bias, "activation": activation,
            "labels": population.addition_correct.float(), "queries": query,
            "answers": answers, "root_ids": population.root_ids}


def _fit_train_normalization(train_parts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Fit shared train-only normalization over the trajectory union."""
    incoming = train_parts["incoming"]
    hidden_bias = train_parts["hidden_bias"]
    edges = train_parts["edges"]
    out_bias = train_parts["output_bias"]
    activation = train_parts["activation"]
    feature = torch.cat((incoming, hidden_bias.unsqueeze(-1)), -1)
    return {
        # Match SYN-82's global per-coordinate normalization: aggregate over
        # roots, rules, ages, and neurons, leaving the 20 input coordinates
        # plus hidden bias. Do not normalize each neuron slot separately.
        "feature_mean": feature.mean((0, 1, 2, 3)),
        "feature_scale": feature.std((0, 1, 2, 3), unbiased=False).clamp_min(1e-6),
        "edge_mean": edges.mean().reshape(1),
        "edge_scale": edges.std(unbiased=False).clamp_min(1e-6).reshape(1),
        "bias_mean": out_bias.mean().reshape(1),
        "bias_scale": out_bias.std(unbiased=False).clamp_min(1e-6).reshape(1),
        "activation_mean": activation.mean().reshape(1),
        "activation_scale": activation.std(unbiased=False).clamp_min(1e-6).reshape(1),
    }

def prepare_training_validation(train, validation):
    """Fit normalization on TRAIN only; return features without merging roots.

    Returned feature arrays preserve [root, rule, age, ...]. Features are cast to
    float32 only after float64 normalization, as in the historical implementation.
    No test population is accepted or used to select normalization.
    """
    require_disjoint_roots(train, validation)
    train_parts, validation_parts = canonical_parts(train), canonical_parts(validation)
    norm = _fit_train_normalization(train_parts)
    def pack(parts):
        features = {key: value.float() for key, value in normalize_parts(parts, norm).items()}
        features.update({key: parts[key] for key in ("labels", "queries", "answers", "root_ids")})
        return features
    return pack(train_parts), pack(validation_parts), norm
