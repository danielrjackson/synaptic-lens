"""Mean-risk observers of MNIST target networks; Lens never receives images."""

from dataclasses import dataclass

import torch

from .observer import fit_lens, layer_statistics, metrics, predict_lens

# Named tensor order from the hash-pinned published d32-L2 coordinate schema.
HISTORICAL_RECIPE = {
    "fit_roots": [1007, 1017, 1027, 1037, 1047, 1057],
    "validation_roots": [1067, 1077],
    "development_roots": [1087, 1097],
    "epochs": [0, 1, 5, 20, 100],
    "updates": [0, 196, 980, 3920, 19600],
    "fit_query_count": 240,
    "validation_query_count": 120,
    "development_query_count": 120,
    "observer_epochs": 200,
    "observer_seed": 7,
    "official_mnist_test_used": False,
    "scope": "Recipe metadata; original query IDs and measured risk labels are not bundled",
}

MNIST_LAYOUT = (
    ("blocks.0.attention.in_proj_bias", 0, 96),
    ("blocks.0.attention.in_proj_weight", 96, 3168),
    ("blocks.0.attention.out_proj.bias", 3168, 3200),
    ("blocks.0.attention.out_proj.weight", 3200, 4224),
    ("blocks.0.attention_norm.bias", 4224, 4256),
    ("blocks.0.attention_norm.weight", 4256, 4288),
    ("blocks.0.feedforward.0.bias", 4288, 4352),
    ("blocks.0.feedforward.0.weight", 4352, 6400),
    ("blocks.0.feedforward.2.bias", 6400, 6432),
    ("blocks.0.feedforward.2.weight", 6432, 8480),
    ("blocks.0.feedforward_norm.bias", 8480, 8512),
    ("blocks.0.feedforward_norm.weight", 8512, 8544),
    ("blocks.1.attention.in_proj_bias", 8544, 8640),
    ("blocks.1.attention.in_proj_weight", 8640, 11712),
    ("blocks.1.attention.out_proj.bias", 11712, 11744),
    ("blocks.1.attention.out_proj.weight", 11744, 12768),
    ("blocks.1.attention_norm.bias", 12768, 12800),
    ("blocks.1.attention_norm.weight", 12800, 12832),
    ("blocks.1.feedforward.0.bias", 12832, 12896),
    ("blocks.1.feedforward.0.weight", 12896, 14944),
    ("blocks.1.feedforward.2.bias", 14944, 14976),
    ("blocks.1.feedforward.2.weight", 14976, 17024),
    ("blocks.1.feedforward_norm.bias", 17024, 17056),
    ("blocks.1.feedforward_norm.weight", 17056, 17088),
    ("classifier.bias", 17088, 17098),
    ("classifier.weight", 17098, 17418),
    ("cls_token", 17418, 17450),
    ("final_norm.bias", 17450, 17482),
    ("final_norm.weight", 17482, 17514),
    ("patch_projection.bias", 17514, 17546),
    ("patch_projection.weight", 17546, 18058),
    ("position_embedding", 18058, 19658),
)


def target_layout():
    """Return fresh ranges in the published 19,658-coordinate order."""
    return [{"start": start, "stop": stop} for _, start, stop in MNIST_LAYOUT]


@dataclass(frozen=True)
class ObserverPartition:
    root_ids: tuple[int, ...]
    weights: torch.Tensor
    mean_risk: torch.Tensor

    def validate(self):
        if (
            not isinstance(self.root_ids, tuple)
            or not self.root_ids
            or any(type(i) is not int or i < 0 for i in self.root_ids)
        ):
            raise ValueError("Explicit nonnegative root IDs required for every state")
        if (
            self.weights.shape != (len(self.root_ids), 19658)
            or self.weights.dtype != torch.float32
            or not torch.isfinite(self.weights).all()
        ):
            raise ValueError(
                "Expected finite float32 ordered target weights [states, 19658], never images"
            )
        if (
            self.mean_risk.shape != (len(self.root_ids),)
            or self.mean_risk.dtype != torch.float32
            or not torch.isfinite(self.mean_risk).all()
            or (self.mean_risk < 0).any()
        ):
            raise ValueError(
                "Expected nonnegative measured mean risk for every target state"
            )
        if self.weights.device.type != "cpu" or self.mean_risk.device.type != "cpu":
            raise ValueError("This bounded observer workflow is CPU-only")


def validate_layout(layout):
    if len(layout) != 32:
        raise ValueError("Expected 32 target parameter tensors")
    for entry, (name, start, stop) in zip(layout, MNIST_LAYOUT, strict=True):
        if (
            type(entry["start"]) is not int
            or type(entry["stop"]) is not int
            or (entry["start"], entry["stop"]) != (start, stop)
        ):
            raise ValueError("Layout differs from published named-tensor ordering")
        if "parameter_name" in entry and entry["parameter_name"] != name:
            raise ValueError("Layout parameter name differs from published schema")


def features(partition, *, mode, layout):
    partition.validate()
    validate_layout(layout)
    if mode == "raw":
        return partition.weights
    if mode == "statistics":
        return layer_statistics(partition.weights, layout)
    raise ValueError("Expected raw or statistics observer")


def fit_mean_risk(train, validation, *, layout, mode="statistics", epochs=200, seed=7):
    """Train on scalar measured risk; select only on validation roots.

    Caller-supplied risk labels must share a documented evaluation suite. This
    helper does not generate them, read images, infer historical splits or open
    the official MNIST test set. It does not reproduce the archived side study.
    """
    train.validate()
    validation.validate()
    if set(train.root_ids).intersection(validation.root_ids):
        raise ValueError("Observer roots overlap across train and validation")
    if type(epochs) is not int or epochs < 1:
        raise ValueError("Positive epoch count required")
    x = features(train, mode=mode, layout=layout)
    vx = features(validation, mode=mode, layout=layout)
    with torch.random.fork_rng(devices=[]):
        bundle = fit_lens(
            x, train.mean_risk, vx, validation.mean_risk, epochs=epochs, seed=seed
        )
    bundle.update(
        schema="mnist-direct-weight-lens-bundle-v1",
        observer_mode=mode,
        training_roots=sorted(set(train.root_ids)),
        selection_roots=sorted(set(validation.root_ids)),
    )
    report = {
        "train_roots": sorted(set(train.root_ids)),
        "validation_roots": sorted(set(validation.root_ids)),
        "train_states": len(train.root_ids),
        "validation_states": len(validation.root_ids),
        "selected_epoch": bundle["best_epoch"],
        "historical_metrics_reproduced": False,
        "output": "one mean-cross-entropy estimate per target state, not digit class",
        "validation": metrics(validation.mean_risk, predict_lens(bundle, vx)),
    }
    return bundle, report


def evaluate_mean_risk(bundle, development, *, layout):
    """Report state errors on disjoint development roots, without selection."""
    development.validate()
    if "training_roots" not in bundle or "selection_roots" not in bundle:
        raise ValueError(
            "Explicit training/selection provenance required for development evaluation"
        )
    used = set(bundle["training_roots"]) | set(bundle["selection_roots"])
    if used.intersection(development.root_ids):
        raise ValueError("Development roots overlap training or selection")
    x = features(development, mode=bundle["observer_mode"], layout=layout)
    with torch.inference_mode():
        prediction = predict_lens(bundle, x)
    return {
        "metrics": metrics(development.mean_risk, prediction),
        "root_count": len(set(development.root_ids)),
        "state_count": len(development.root_ids),
        "scope": "state-level errors; neighboring states are not independent roots",
    }
