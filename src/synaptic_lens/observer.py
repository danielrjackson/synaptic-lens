"""
Weights-only regression and deliberately strong, cheap comparison models.

An observer can predict loss by recognizing training progress without identifying
an addition algorithm. Baselines make that possible shortcut visible. None of
the weights-only models receives a digit pair, step, run ID, or measured loss as
input. The step baseline is explicitly privileged and reported separately.
"""

import math
from copy import deepcopy

import torch
from torch import nn


def split_runs(count, seed=20260915):
    """Partition *runs*, not checkpoints. Validation selects; test only reports."""
    if count < 10:
        raise ValueError("At least 10 independent runs are required.")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    n_train, n_val = int(count * 0.6), int(count * 0.2)
    return {
        "train": order[:n_train],
        "validation": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }


def checkpoint_sample(last_step, count=32):
    """
    Keep more detail early in learning; retain all checkpoints on disk.

    This rule uses step indices, not labels. It is chosen before looking at test
    losses. Primary metrics weight these selected snapshots equally within each run.
    """
    indices = torch.logspace(0, math.log10(last_step + 1), count).round().long() - 1
    return torch.unique(torch.cat((torch.tensor([0, last_step]), indices))).sort().values


def standardize_fit(x):
    # This MUST be fit on training runs only. Fitting on the full population
    # would allow evaluation data to influence the observer's preprocessing.
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, scale


def layer_statistics(weights, layout):
    """
    Six permutation-invariant scalar statistics per tensor (24 in our MLP).

    These cheap features test whether weight scale alone explains observer success.
    They cannot describe which input connects to which hidden neuron.
    """
    features = []
    for entry in layout:
        block = weights[..., entry["start"] : entry["stop"]]
        features.extend(
            (
                block.mean(-1),
                block.std(-1, unbiased=False),
                block.square().mean(-1).sqrt(),
                block.abs().mean(-1),
                block.amin(-1),
                block.amax(-1),
            )
        )
    return torch.stack(features, dim=-1)


class Lens(nn.Module):
    """A small nonlinear observer; input coordinates are target parameters."""

    def __init__(self, features, width=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, width), nn.ReLU(), nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def fit_lens(train_x, train_y, val_x, val_y, *, epochs=200, seed=7):
    """
    Optimize on training runs; select checkpoint using validation MSE only.

    Standardizing labels makes the optimization scale easier to handle. A predicted
    loss is clipped to zero only after converting back to its original units.
    """
    torch.manual_seed(seed)
    mean, scale = standardize_fit(train_x)
    y_mean = train_y.mean()
    y_scale = train_y.std(unbiased=False).clamp_min(1e-6)
    x, vx = (train_x - mean) / scale, (val_x - mean) / scale
    y = (train_y - y_mean) / y_scale
    model = Lens(train_x.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    rng = torch.Generator().manual_seed(seed + 1)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(x), generator=rng)
        for batch in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[batch]) - y[batch]).square().mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = (model(vx) * y_scale + y_mean).clamp_min(0)
            validation_mse = float((prediction - val_y).square().mean())
            train_mse = float(
                ((model(x) * y_scale + y_mean).clamp_min(0) - train_y).square().mean()
            )
        history.append(
            {"epoch": epoch + 1, "train_mse": train_mse, "validation_mse": validation_mse}
        )
        if validation_mse < best_loss:
            best_loss, best_state, best_epoch = (
                validation_mse,
                deepcopy(model.state_dict()),
                epoch + 1,
            )
    model.load_state_dict(best_state)
    return {
        "state_dict": best_state,
        "mean": mean,
        "scale": scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "features": train_x.shape[-1],
        "best_epoch": best_epoch,
        "seed": seed,
        "history": history,
    }


@torch.no_grad()
def predict_lens(bundle, x):
    model = Lens(bundle["features"])
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    # Batch inference also handles larger saved populations without huge buffers.
    original_shape = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    result = [
        (
            (model((b - bundle["mean"]) / bundle["scale"]) * bundle["y_scale"]) + bundle["y_mean"]
        ).clamp_min(0)
        for b in flat.split(1024)
    ]
    return torch.cat(result).reshape(original_shape)


def fit_ridge(train_x, train_y, val_x, val_y):
    """Linear regression with shrinkage; select its strength on validation runs."""
    mean, scale = standardize_fit(train_x)
    x = torch.cat(((train_x - mean) / scale, torch.ones(len(train_x), 1)), dim=1).double()
    vx = torch.cat(((val_x - mean) / scale, torch.ones(len(val_x), 1)), dim=1).double()
    best = None
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
        penalty = torch.eye(x.shape[1], dtype=torch.float64) * alpha
        penalty[-1, -1] = 0  # Do not shrink the intercept toward zero.
        coefficients = torch.linalg.solve(x.T @ x + penalty, x.T @ train_y.double())
        mse = float(((vx @ coefficients).clamp_min(0) - val_y).square().mean())
        if best is None or mse < best[0]:
            best = (mse, coefficients, alpha)
    return {"mean": mean, "scale": scale, "coefficients": best[1].float(), "alpha": best[2]}


def predict_ridge(bundle, x):
    standardized = (x - bundle["mean"]) / bundle["scale"]
    return (standardized @ bundle["coefficients"][:-1] + bundle["coefficients"][-1]).clamp_min(0)


def metrics(truth, prediction):
    error = prediction - truth
    mse = float(error.square().mean())
    variance = float((truth - truth.mean()).square().mean())
    return {
        "mae": float(error.abs().mean()),
        "rmse": math.sqrt(mse),
        "r2": 1 - mse / variance if variance > 1e-12 else None,
    }


def paired_run_bootstrap(truth, prediction, baseline, seed=99):
    """
    Resample whole runs; never pretend neighboring checkpoints are independent.

    Positive differences mean the Lens has LOWER error than the baseline. This
    interval describes this finite test population, not architecture-wide certainty.
    """
    differences = (baseline - truth).abs().mean(1) - (prediction - truth).abs().mean(1)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(len(differences), (2000, len(differences)), generator=generator)
    means = differences[indices].mean(1)
    return {
        "mae_improvement": float(differences.mean()),
        "ci95": torch.quantile(means, torch.tensor([0.025, 0.975])).tolist(),
    }
