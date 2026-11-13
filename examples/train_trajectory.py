"""Small CPU smoke: random target states, not historical result reproduction."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch
from synaptic_lens.targets import addition_suite
from synaptic_lens.trajectory_checkpoint import load_model
from synaptic_lens.trajectory_data import TrajectoryPopulation
from synaptic_lens.trajectory_training import save_inference_export, train_trajectory


def fixture(root_ids, seed):
    g = torch.Generator().manual_seed(seed)
    weights = torch.randn(len(root_ids), 2, 8, 1299, generator=g) * 0.1
    weights[:, 1, 0] = weights[:, 0, 0]
    raw = weights.double()
    _, q, answers = addition_suite()
    hidden = torch.relu(
        torch.einsum(
            "qj,...nj->...qn",
            q.double(),
            raw[..., :640].reshape(*raw.shape[:-1], 32, 20),
        )
        + raw[..., 640:672].unsqueeze(-2)
    )
    edge = raw[..., 672:1280].reshape(*raw.shape[:-1], 19, 32).transpose(-1, -2)
    labels = (
        (
            torch.einsum("...qn,...nk->...qk", hidden, edge)
            + raw[..., 1280:1299].unsqueeze(-2)
        )
        .argmax(-1)
        .eq(answers)
    )
    return TrajectoryPopulation(tuple(root_ids), weights, labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.updates <= 10:
        parser.error(
            "smoke budget is 1 through 10 updates; use library API for explicit larger recipes"
        )
    torch.set_num_threads(1)
    model, norm, report = train_trajectory(
        fixture((0, 1), 1), fixture((2,), 2), updates=args.updates
    )
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "inference.npz"
        save_inference_export(path, model, norm)
        load_model(path, hashlib.sha256(path.read_bytes()).hexdigest())
    print(
        json.dumps(
            {"scope": "synthetic CPU smoke; not historical accuracy", **report},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
