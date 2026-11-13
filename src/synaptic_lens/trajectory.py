"""Frozen SYN-82 output-centered objective with SYN-86 canonical inputs."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class OutputCentered(nn.Module):
    """Per-neuron shared reader, output-edge reader, and output aggregation."""

    def __init__(self):
        super().__init__()
        self.neuron = nn.Sequential(nn.Linear(42, 64), nn.ReLU(),
                                    nn.Linear(64, 16), nn.ReLU())
        self.edge = nn.Sequential(nn.Linear(17, 32), nn.ReLU(),
                                  nn.Linear(32, 8), nn.ReLU())
        self.score = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 1))
        self.calibration_raw_scale = nn.Parameter(torch.tensor(0.54132485))
        self.calibration_shift = nn.Parameter(torch.zeros(()))

    def score_outputs(self, neurons, edge_w, out_bias, query, activation):
        states, n_neurons, _ = neurons.shape
        n_queries = query.shape[0]
        qx = query[None, :, None, :].expand(states, n_queries, n_neurons, 20)
        ax = activation.unsqueeze(-1)
        base = torch.cat((neurons[:, None].expand(states, n_queries, n_neurons, 21),
                          qx, ax), -1)
        h = self.neuron(base)
        ew = edge_w[:, None].expand(states, n_queries, n_neurons, 19).unsqueeze(-1)
        edge_input = torch.cat((h.unsqueeze(3).expand(states, n_queries, n_neurons, 19, 16), ew), -1)
        edge_summary = self.edge(edge_input).mean(dim=2)
        bias = out_bias[:, None, :, None].expand(states, n_queries, 19, 1)
        raw = self.score(torch.cat((edge_summary, bias), -1)).squeeze(-1)
        return F.softplus(self.calibration_raw_scale) * raw + self.calibration_shift

    def forward(self, neurons, edge_w, out_bias, query, activation, answers,
                arm: str):
        scores = self.score_outputs(neurons, edge_w, out_bias, query, activation)
        idx = answers.long().view(1, -1, 1).expand(scores.shape[0], -1, -1)
        expected = scores.gather(-1, idx).squeeze(-1)
        if arm == "expected-score":
            return expected
        rival = scores.scatter(-1, idx, float("-inf")).amax(-1)
        return expected - rival


def canonicalize(weights: torch.Tensor, query: torch.Tensor):
    """Apply frozen incoming max-abs neuron scaling in float64."""
    w = weights.double()
    hidden_bias = w[..., 640:672]
    incoming = w[..., :640].reshape(*w.shape[:-1], 32, 20)
    output_w = w[..., 672:1280].reshape(*w.shape[:-1], 19, 32).transpose(-1, -2)
    output_bias = w[..., 1280:1299]
    scale = torch.cat((incoming.abs(), hidden_bias.abs().unsqueeze(-1)), -1).amax(-1)
    nonzero = scale > 0
    denom = torch.where(nonzero, scale, torch.ones_like(scale))
    ci = incoming / denom.unsqueeze(-1)
    cb = hidden_bias / denom
    activation = F.relu(torch.einsum("qj,...nj->...qn", query.double(), incoming)
                        + hidden_bias.unsqueeze(-2)) / denom.unsqueeze(-2)
    ce = output_w * scale.unsqueeze(-1)
    ci = torch.where(nonzero.unsqueeze(-1), ci, torch.zeros_like(ci))
    cb = torch.where(nonzero, cb, torch.zeros_like(cb))
    activation = torch.where(nonzero.unsqueeze(-2), activation, torch.zeros_like(activation))
    ce = torch.where(nonzero.unsqueeze(-1), ce, torch.zeros_like(ce))
    return ci, cb, ce, output_bias, activation


def normalize_parts(parts: dict[str, torch.Tensor], norm: dict[str, torch.Tensor]):
    feature = torch.cat((parts["incoming"], parts["hidden_bias"].unsqueeze(-1)), -1)
    return {
        "neurons": (feature - norm["feature_mean"]) / norm["feature_scale"],
        "edges": (parts["edges"] - norm["edge_mean"]) / norm["edge_scale"],
        "bias": (parts["output_bias"] - norm["bias_mean"]) / norm["bias_scale"],
        "activation": (parts["activation"] - norm["activation_mean"]) / norm["activation_scale"],
    }
