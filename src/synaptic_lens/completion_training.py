"""Train a position-free missing-neuron reader from root-isolated populations."""

import copy
import io

import numpy as np
import torch

from .checkpoints import verified_bytes
from .neuron_completion import (
    REPRESENTATION_VERSION,
    NoSortLens,
    normalize_neurons,
    permute_neurons,
)
from .syn88_targets import AGES, split_roots


def training_partitions(fit_count=64, development_count=16):
    """Only audited fitting/development roots; consumed final cohorts stay unused."""
    if not 1 <= fit_count <= 768 or not 1 <= development_count <= 128:
        raise ValueError("Require 1..768 fitting and 1..128 development roots")
    train = split_roots()["train"]
    return {
        "fit": (train[:256] + train[640:1152])[:fit_count],
        "development": train[384:512][:development_count],
    }


def parse_final_weights(weights):
    if weights.shape != (2, 8, 1299) or weights.dtype != np.float32:
        raise ValueError("Expected float32 [2 rules,8 ages,1299 parameters]")
    if not np.isfinite(weights).all():
        raise ValueError("Nonfinite target weights")
    v = torch.from_numpy(weights[:, -1].copy())
    raw = torch.cat(
        (
            v[:, :640].reshape(2, 32, 20),
            v[:, 640:672, None],
            v[:, 672:1280].reshape(2, 19, 32).transpose(1, 2),
        ),
        -1,
    )
    return raw, v[:, 1280:].clone()


def load_population(ids, files):
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Require unique nonempty root IDs")
    raw, bias = [], []
    for root in ids:
        entry = files[f"syn88/population/root-{root:04d}.npz"]
        with np.load(
            io.BytesIO(verified_bytes(entry["path"], entry["sha256"])),
            allow_pickle=False,
        ) as data:
            if int(data["root_index"]) != root or not np.array_equal(
                data["ages"], AGES
            ):
                raise ValueError("Target root or age identity mismatch")
            r, b = parse_final_weights(data["weights"])
            raw.append(r)
            bias.append(b)
    return {"roots": list(ids), "raw": torch.cat(raw), "bias": torch.cat(bias)}


def _validate_population(pop):
    n = len(pop["roots"])
    if n == 0 or len(set(pop["roots"])) != n:
        raise ValueError("Unique nonempty roots required")
    for name, shape in (("raw", (2 * n, 32, 40)), ("bias", (2 * n, 19))):
        v = pop[name]
        if v.shape != shape or v.dtype != torch.float32 or not torch.isfinite(v).all():
            raise ValueError("Invalid population tensor")


@torch.no_grad()
def all_slot_predictions(model, values, bias, chunk=128):
    predictions = []
    for start in range(0, len(values) * 32, chunk):
        flat = torch.arange(
            start, min(start + chunk, len(values) * 32), device=values.device
        )
        predictions.append(model(values[flat // 32], bias[flat // 32], flat % 32))
    return torch.cat(predictions).reshape(-1, 32, 40)


def train_completion(
    fit,
    development,
    *,
    updates=2000,
    batch_size=128,
    validation_every=200,
    seed=161120,
    device="cuda",
    width=256,
    blocks=4,
    heads=8,
):
    """Fit-only statistics; select on development all-slot standardized MSE.

    The default architecture matches the released 3.19M-parameter reader. Small
    dimensions are exposed for explicit fixture tests, not benchmark comparisons.
    """
    _validate_population(fit)
    _validate_population(development)
    if set(fit["roots"]) & set(development["roots"]):
        raise ValueError("Root partitions overlap")
    if min(updates, batch_size, validation_every) < 1:
        raise ValueError("Positive training budgets required")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly select CPU for a smoke test")
    torch.manual_seed(seed)
    model = NoSortLens(width, blocks, heads, positional=False).to(dev)
    norm, _ = normalize_neurons(fit["raw"])
    dn, _ = normalize_neurons(development["raw"])
    mean, std = (
        norm.mean((0, 1), keepdim=True),
        norm.std((0, 1), keepdim=True).clamp_min(1e-5),
    )
    bm, bs = (
        fit["bias"].mean(0, keepdim=True),
        fit["bias"].std(0, keepdim=True).clamp_min(1e-5),
    )
    x, b = ((norm - mean) / std).to(dev), ((fit["bias"] - bm) / bs).to(dev)
    dx, db = ((dn - mean) / std).to(dev), ((development["bias"] - bm) / bs).to(dev)
    mask_rng = torch.Generator(device=dev).manual_seed(seed + 10000)
    perm_rng = torch.Generator(device=dev).manual_seed(seed + 20000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best, selected, history = float("inf"), None, []
    for step in range(1, updates + 1):
        ids = torch.randint(len(x), (batch_size,), generator=mask_rng, device=dev)
        physical = torch.randint(32, (batch_size,), generator=mask_rng, device=dev)
        for _ in range(100):
            keys = torch.randint(
                0, 2**63 - 1, (batch_size, 32), generator=perm_rng, device=dev
            )
            ordered, perm = keys.sort(1)
            if not bool((ordered[:, 1:] == ordered[:, :-1]).any()):
                break
        else:
            raise RuntimeError("Repeated permutation-key collision")
        values, missing = permute_neurons(x[ids], physical, perm)
        model.train()
        loss = (model(values, b[ids], missing) - x[ids, physical]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % validation_every == 0 or step == updates:
            model.eval()
            value = float((all_slot_predictions(model, dx, db) - dx).square().mean())
            if not np.isfinite(value):
                raise FloatingPointError("Nonfinite development loss")
            history.append(
                {
                    "step": step,
                    "development_mse": value,
                    "training_mse": float(loss.detach()),
                }
            )
            if value < best:
                best, selected_step = value, step
                selected = copy.deepcopy(
                    {k: v.cpu() for k, v in model.state_dict().items()}
                )
    checkpoint = {
        "state_dict": selected,
        "positional": False,
        "config": {
            "representation_version": REPRESENTATION_VERSION,
            "width": width,
            "blocks": blocks,
            "heads": heads,
        },
        "mean": mean,
        "std": std,
        "bias_mean": bm,
        "bias_std": bs,
    }
    return checkpoint, {
        "seed": seed,
        "updates": updates,
        "selected_step": selected_step,
        "fit_roots": fit["roots"],
        "development_roots": development["roots"],
        "history": history,
        "selection": "development all-slot standardized MSE",
        "scope": "training walkthrough; no fresh benchmark evaluation",
    }
