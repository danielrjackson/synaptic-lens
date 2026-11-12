"""A weights-only observer that respects the target's hidden-neuron symmetry.

Rather than assigning significance to 'neuron 7', describe every hidden neuron
by its incoming weights, bias, and outgoing weights. Apply the SAME learned
encoder to each description, then average. Renumbering the hidden neurons only
reorders the items being averaged, leaving the Lens prediction unchanged.

Input digit coordinates and output sum coordinates KEEP their meanings. We do
not make Lens invariant to scrambling these: that changes the capability.
This is a deliberately small structural bias, not a claim of discovered circuits.
"""

from copy import deepcopy

import torch
from torch import nn


def unpack(weights, layout):
    blocks = {
        item["name"]: weights[..., item["start"] : item["stop"]].reshape(
            *weights.shape[:-1], *item["shape"]
        )
        for item in layout
    }
    neurons = torch.cat(
        (
            blocks["hidden.weight"],
            blocks["hidden.bias"].unsqueeze(-1),
            blocks["output.weight"].transpose(-1, -2),
        ),
        dim=-1,
    )
    return neurons, blocks["output.bias"]


class StructuredLens(nn.Module):
    """Shared neuron encoder + averaging + a final loss regression head."""

    def __init__(self):
        super().__init__()
        self.neuron_encoder = nn.Sequential(
            nn.Linear(40, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU()
        )
        self.readout = nn.Sequential(nn.Linear(64 + 19, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, neurons, output_bias):
        summary = self.neuron_encoder(neurons).mean(dim=-2)
        return self.readout(torch.cat((summary, output_bias), dim=-1)).squeeze(-1)


def fit_structured(tx, ty, vx, vy, layout, epochs=150, seed=7):
    torch.manual_seed(seed)
    neurons, bias = unpack(tx, layout)
    vn, vb = unpack(vx, layout)
    # Pool over examples AND hidden-neuron positions when fitting scales. Per-
    # neuron-position normalization would accidentally break our invariance.
    nm = neurons.mean(dim=(0, 1))
    ns = neurons.std(dim=(0, 1), unbiased=False).clamp_min(1e-6)
    bm, bs = bias.mean(0), bias.std(0, unbiased=False).clamp_min(1e-6)
    ym, ys = ty.mean(), ty.std(unbiased=False).clamp_min(1e-6)
    n, b = (neurons - nm) / ns, (bias - bm) / bs
    vn, vb = (vn - nm) / ns, (vb - bm) / bs
    y = (ty - ym) / ys
    model = StructuredLens()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    rng = torch.Generator().manual_seed(seed + 1)
    best, state, best_epoch, history = float("inf"), None, 0, []
    for epoch in range(epochs):
        model.train()
        for batch in torch.randperm(len(tx), generator=rng).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(n[batch], b[batch]) - y[batch]).square().mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = (model(vn, vb) * ys + ym).clamp_min(0)
            error = float((prediction - vy).square().mean())
            train_error = float(((model(n, b) * ys + ym).clamp_min(0) - ty).square().mean())
        history.append({"epoch": epoch + 1, "train_mse": train_error, "validation_mse": error})
        if error < best:
            best, state, best_epoch = error, deepcopy(model.state_dict()), epoch + 1
    return {
        "state_dict": state,
        "neuron_mean": nm,
        "neuron_scale": ns,
        "bias_mean": bm,
        "bias_scale": bs,
        "y_mean": ym,
        "y_scale": ys,
        "layout": layout,
        "seed": seed,
        "best_epoch": best_epoch,
        "history": history,
    }


@torch.no_grad()
def predict_structured(bundle, x):
    model = StructuredLens()
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    shape = x.shape[:-1]
    values = []
    for batch in x.reshape(-1, x.shape[-1]).split(512):
        n, b = unpack(batch, bundle["layout"])
        score = model(
            (n - bundle["neuron_mean"]) / bundle["neuron_scale"],
            (b - bundle["bias_mean"]) / bundle["bias_scale"],
        )
        values.append((score * bundle["y_scale"] + bundle["y_mean"]).clamp_min(0))
    return torch.cat(values).reshape(shape)
