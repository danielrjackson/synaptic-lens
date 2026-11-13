"""Portable SYN88 optimization and development scoring; no private run ledger.

Historical mechanics are preserved. A bounded call is not a reproduction of the
historical full population, common stopping decision, or published test metrics.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .syn88_targets import AGES
from .trajectory import OutputCentered
from .trajectory_data import prepare_training_validation

QUERY_CHUNK = 10
HISTORICAL_RECIPE = {
    "arms": ["trajectory", "checkpoint25"],
    "seeds": [7, 17, 27],
    "train_roots": 1536,
    "validation_roots": 256,
    "historical_test_roots": 512,
    "batch_roots": 24,
    "query_chunk": 10,
    "max_updates": 20000,
    "validation_every": 1000,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "schedule_seed_base": 880088,
    "selection": "minimum full_trajectory_macro_loss on validation",
    "scope": "fixed-budget portable runner; historical common stopping/recovery orchestration not replayed",
}


def make_schedule(root_count, updates, batch_roots, seeds=(7, 17, 27)):
    """Draw root positions without replacement, sharing an age across both rules."""
    if any(type(n) is not int or n <= 0 for n in (root_count, updates, batch_roots)):
        raise ValueError("Positive integer root/update/batch counts required")
    if batch_roots > root_count or root_count > 2304:
        raise ValueError("Batch exceeds available roots or SYN88 population")
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Nonnegative integer seeds required")
    roots = np.empty((len(seeds), updates, batch_roots), dtype=np.int16)
    ages = np.empty_like(roots)
    for si, seed in enumerate(seeds):
        rng = np.random.default_rng(880088 + seed)
        for step in range(updates):
            roots[si, step] = rng.choice(root_count, batch_roots, replace=False)
            ages[si, step] = rng.integers(0, len(AGES), size=batch_roots)
    return {"roots": roots, "ages": ages}


def loss_and_backward(
    model,
    x,
    edge,
    bias,
    activation,
    queries,
    answers,
    labels,
    rules,
    arm: str,
    query_chunk: int = 10,
):
    """Exact complete-batch BCE plus same-query, same-rule root ranking."""
    if type(query_chunk) is not int or query_chunk <= 0:
        raise ValueError("Positive integer query chunk required")
    supports = []
    for rule in (0, 1):
        rows = (rules == rule).nonzero(as_tuple=False).flatten()
        y = labels[rows]
        supports.append(((y[:, None] > 0.5) & (y[None, :] < 0.5)).sum((0, 1)) > 0)
    active_rules = sum(bool(s.any()) for s in supports)
    totals = {"objective": 0.0, "bce": 0.0, "ranking": 0.0}
    for lo in range(0, 100, query_chunk):
        hi = min(lo + query_chunk, 100)
        logit = model(
            x, edge, bias, queries[lo:hi], activation[:, lo:hi], answers[lo:hi], arm
        )
        y = labels[:, lo:hi]
        bce = (
            F.binary_cross_entropy_with_logits(logit, y, reduction="sum")
            / labels.numel()
        )
        ranking = logit.new_zeros(())
        for ix, rule in enumerate((0, 1)):
            rows = (rules == rule).nonzero(as_tuple=False).flatten()
            z, lab = logit[rows], y[rows]
            delta = z[:, None] - z[None, :]
            mask = (lab[:, None] > 0.5) & (lab[None, :] < 0.5)
            counts = mask.sum((0, 1))
            eligible = supports[ix][lo:hi]
            if eligible.any():
                per_query = (F.softplus(-delta) * mask).sum((0, 1))[eligible] / counts[
                    eligible
                ]
                ranking = ranking + per_query.sum() / (
                    int(supports[ix].sum()) * active_rules
                )
        objective = bce + ranking
        objective.backward()
        totals["objective"] += float(objective.detach())
        totals["bce"] += float(bce.detach())
        totals["ranking"] += float(ranking.detach())
    return totals


def select_batch(
    data: dict, roots: np.ndarray, ages: np.ndarray, arm: str, device: torch.device
):
    ix = torch.as_tensor(roots, dtype=torch.long, device=device)
    stage = torch.as_tensor(
        ages if arm == "trajectory" else np.full(len(roots), 5),
        dtype=torch.long,
        device=device,
    )
    # Both rules for a selected root remain together and share its age draw.
    x = data["neurons"][ix, :, stage].reshape(-1, 32, 21)
    edge = data["edges"][ix, :, stage].reshape(-1, 32, 19)
    bias = data["bias"][ix, :, stage].reshape(-1, 19)
    act = data["activation"][ix, :, stage].reshape(-1, 100, 32)
    labels = data["labels"][ix, :, stage].reshape(-1, 100)
    return (
        x,
        edge,
        bias,
        act,
        labels,
        torch.tensor([0, 1], device=device).repeat(len(roots)),
    )


@torch.no_grad()
def predict(model, data: dict, arm: str, *, chunk_states: int = 16) -> np.ndarray:
    if type(chunk_states) is not int or chunk_states <= 0:
        raise ValueError("Positive integer state chunk required")
    n, _, stages = data["neurons"].shape[:3]
    flat_x = data["neurons"].reshape(n * 2 * stages, 32, 21)
    flat_e = data["edges"].reshape(n * 2 * stages, 32, 19)
    flat_b = data["bias"].reshape(n * 2 * stages, 19)
    flat_a = data["activation"].reshape(n * 2 * stages, 100, 32)
    logits = np.empty((n * 2 * stages, 100), dtype=np.float32)
    model.eval()
    for start in range(0, len(flat_x), chunk_states):
        stop = min(start + chunk_states, len(flat_x))
        vals = []
        for qlo in range(0, 100, QUERY_CHUNK):
            qhi = min(qlo + QUERY_CHUNK, 100)
            scores = model.score_outputs(
                flat_x[start:stop],
                flat_e[start:stop],
                flat_b[start:stop],
                data["queries"][qlo:qhi],
                flat_a[start:stop, qlo:qhi],
            )
            idx = data["answers"][qlo:qhi].view(1, -1, 1).expand(stop - start, -1, -1)
            expected = scores.gather(-1, idx).squeeze(-1)
            rival = scores.scatter(-1, idx, float("-inf")).amax(-1)
            expected = expected - rival
            vals.append(expected)
        logits[start:stop] = torch.cat(vals, 1).float().cpu().numpy()
    return logits.reshape(n, 2, stages, 100)


def val_metrics(logits: np.ndarray, labels: np.ndarray) -> dict:
    logits, labels = np.asarray(logits), np.asarray(labels)
    if (
        logits.shape != labels.shape
        or logits.ndim != 4
        or logits.shape[1:] != (2, 8, 100)
        or len(logits) == 0
    ):
        raise ValueError("Expected nonempty root x rule x age x query arrays")
    if not np.isfinite(logits).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Expected finite logits and binary labels")
    z = torch.from_numpy(logits).double()
    y = torch.as_tensor(labels).double()
    cells = []
    by_rule_stage = {}
    for rule in range(2):
        for stage, age in enumerate(AGES):
            zz, yy = z[:, rule, stage], y[:, rule, stage]
            bce = F.binary_cross_entropy_with_logits(zz, yy)
            delta = zz[:, None, :] - zz[None, :, :]
            mask = (yy[:, None, :] > 0.5) & (yy[None, :, :] < 0.5)
            counts = mask.sum((0, 1))
            supported = counts > 0
            if supported.any():
                per_query = (F.softplus(-delta) * mask).sum((0, 1))[supported] / counts[
                    supported
                ]
                ranking = per_query.mean()
                supported_queries = int(supported.sum())
            else:
                ranking = torch.tensor(0.0, dtype=torch.float64)
                supported_queries = 0
            objective = bce + ranking
            cells.append(float(objective))
            by_rule_stage[f"rule{rule}-update{age}"] = {
                "macro_objective": float(objective),
                "bce": float(bce),
                "ranking": float(ranking),
                "supported_queries": supported_queries,
                "prevalence": float(yy.mean()),
            }
    probs = z.sigmoid()
    return {
        "full_trajectory_macro_loss": float(np.mean(cells)),
        "cells": by_rule_stage,
        "checkpoint25_macro_loss": float(
            np.mean(
                [by_rule_stage[f"rule{r}-update25"]["macro_objective"] for r in (0, 1)]
            )
        ),
        "checkpoint25_bce": float(
            np.mean([by_rule_stage[f"rule{r}-update25"]["bce"] for r in (0, 1)])
        ),
        "checkpoint25_ranking": float(
            np.mean([by_rule_stage[f"rule{r}-update25"]["ranking"] for r in (0, 1)])
        ),
        "accuracy": float(((probs >= 0.5) == y).double().mean()),
        "brier": float(((probs - y) ** 2).mean()),
        "support": int(y.numel()),
        "positive": int(y.sum()),
        "prevalence": float(y.mean()),
    }


def train_trajectory(
    train,
    validation,
    *,
    updates=1,
    batch_roots=2,
    seed=7,
    arm="trajectory",
    validation_every=1000,
):
    """Fit one fixed-budget CPU cell and select on validation, never test.

    For the historical configuration use HISTORICAL_RECIPE explicitly. This
    function does not resume inference exports or recreate private common-stop
    orchestration. Supplied populations must already be manifest verified.
    """
    if arm not in ("trajectory", "checkpoint25"):
        raise ValueError("Unknown training arm")
    if type(validation_every) is not int or validation_every < 1:
        raise ValueError("Positive validation interval required")
    schedule = make_schedule(len(train.root_ids), updates, batch_roots, (seed,))
    if train.weights.device.type != "cpu" or validation.weights.device.type != "cpu":
        raise ValueError("This portable runner currently supports CPU only")
    training, development, norm = prepare_training_validation(train, validation)
    history = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = OutputCentered()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        best_loss, best_update, selected = float("inf"), 0, None
        for update in range(updates + 1):
            if update:
                x, edge, bias, activation, labels, rules = select_batch(
                    training,
                    schedule["roots"][0, update - 1],
                    schedule["ages"][0, update - 1],
                    arm,
                    torch.device("cpu"),
                )
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss_and_backward(
                    model,
                    x,
                    edge,
                    bias,
                    activation,
                    training["queries"],
                    training["answers"],
                    labels,
                    rules,
                    "margin",
                    QUERY_CHUNK,
                )
                optimizer.step()
            if update == 0 or update % validation_every == 0 or update == updates:
                metric = val_metrics(
                    predict(model, development, arm), development["labels"].numpy()
                )
                history.append({"update": update, "validation": metric})
                if metric["full_trajectory_macro_loss"] < best_loss:
                    best_loss, best_update = (
                        metric["full_trajectory_macro_loss"],
                        update,
                    )
                    selected = deepcopy(model.state_dict())
        model.load_state_dict(selected, strict=True)
    return (
        model.eval(),
        norm,
        {
            "selected_update": best_update,
            "completed_updates": updates,
            "arm": arm,
            "seed": seed,
            "history": history,
            "train_root_ids": list(train.root_ids),
            "validation_root_ids": list(validation.root_ids),
            "historical_metrics_reproduced": False,
        },
    )


def save_inference_export(path, model, norm):
    """Write the existing numeric-only inference format without replacing files."""
    arrays = {
        "model/" + key: value.detach().cpu().numpy()
        for key, value in model.state_dict().items()
    }
    arrays.update(
        {"norm/" + key: value.detach().cpu().numpy() for key, value in norm.items()}
    )
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
