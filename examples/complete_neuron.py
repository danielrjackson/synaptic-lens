"""Hide and reconstruct one adder neuron with the reviewed local inference export."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

from synaptic_lens.checkpoints import load_verified
from synaptic_lens.neuron_completion import (
    OrganicCheckpointReader,
    prepare_organic_completion,
)


def answers(neurons, bias):
    pairs = torch.cartesian_prod(torch.arange(10), torch.arange(10))
    x = torch.cat(
        (
            torch.nn.functional.one_hot(pairs[:, 0], 10),
            torch.nn.functional.one_hot(pairs[:, 1], 10),
        ),
        -1,
    ).float()
    return (
        (x @ neurons[:, :20].T + neurons[:, 20]).relu() @ neurons[:, 21:] + bias
    ).argmax(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--neuron", type=int, choices=range(32), default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads(
        (
            Path(__file__).resolve().parents[1] / "docs/neuron-completion-manifest.json"
        ).read_text()
    )
    files = manifest["files"]
    reader = OrganicCheckpointReader(
        args.artifacts / "reader.pt", files["reader.pt"]["sha256"], args.device
    )
    target = load_verified(
        args.artifacts / "example-target.pt", files["example-target.pt"]["sha256"]
    )
    raw, bias = target["neurons"], target["output_bias"]
    view = prepare_organic_completion(raw, bias, args.neuron)
    restored = view.predict(reader)
    damaged = raw.clone()
    damaged[args.neuron] = 0
    expected = torch.cartesian_prod(torch.arange(10), torch.arange(10)).sum(-1)
    report = {
        "root": target["root"],
        "rule": target["rule"],
        "physical_neuron": args.neuron,
        "device": args.device,
        "external_true_scale_retained": True,
        "actual_weights": raw[args.neuron].tolist(),
        "predicted_weights": restored[args.neuron].tolist(),
    }
    for label, values in (
        ("original", raw),
        ("masked", damaged),
        ("restored", restored),
    ):
        predicted = answers(values, bias)
        report[label] = {
            "correct_of_100": int((predicted == expected).sum()),
            "answers": predicted.tolist(),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
