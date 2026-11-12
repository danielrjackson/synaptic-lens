"""Generate the *targets* that our observer will learn to recognize.

There are two learning problems here. An adder learns from digit pairs. Lens
learns from saved adders. The labels for Lens are measured by executing each
adder, but neither the digits nor those measurements are Lens input features.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


@dataclass(frozen=True)
class TargetConfig:
    runs: int = 100
    steps: int = 200
    hidden: int = 32
    learning_rate: float = 0.02
    seed: int = 1729

    def validate(self):
        if self.runs < 10 or self.steps < 2 or self.hidden < 1:
            raise ValueError("Use at least 10 runs, 2 steps, and 1 hidden unit.")
        if not 0 < self.learning_rate < 1:
            raise ValueError("Target learning rate must be between 0 and 1.")


def addition_suite():
    """Return all 100 ordered pairs, their 20 input indicators, and 19 labels.

    One-hot encoding represents identity, not magnitude: digit 3 has one 1 in
    position 3 and nine zeros. Concatenating two such vectors gives 20 inputs.
    Labels remain integer sums; CrossEntropyLoss expects these class indices.
    """
    pairs = torch.cartesian_prod(torch.arange(10), torch.arange(10))
    inputs = F.one_hot(pairs, num_classes=10).float().reshape(100, 20)
    return pairs, inputs, pairs.sum(dim=1)


class Adder(nn.Module):
    """20 -> hidden ReLU units -> 19 scores, one for every possible sum."""

    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = nn.Linear(20, hidden)
        self.output = nn.Linear(hidden, 19)

    def forward(self, inputs):
        # Raw scores are called logits. Cross entropy includes log-softmax, so
        # applying softmax here would duplicate that operation and change loss.
        return self.output(F.relu(self.hidden(inputs)))


def parameter_layout(model):
    """Save offsets so a flat vector can always be mapped back to its layers."""
    layout, start = [], 0
    for name, param in model.named_parameters():
        stop = start + param.numel()
        layout.append({"name": name, "shape": list(param.shape), "start": start, "stop": stop})
        start = stop
    return layout


def restore_adder(weights, hidden):
    """Reconstruct a target for auditing labels, never as part of Lens input."""
    model = Adder(hidden)
    vector_to_parameters(weights.detach().clone(), model.parameters())
    return model.eval()


def generate_population(config, output):
    """Record initialization and EVERY optimizer step, with reproducible seeds.

    The fixed test suite is also the adder's training set in this first experiment.
    Thus labels measure acquired proficiency on this suite, not input generalization.
    Lens generalization is instead measured across independent target training runs.
    """
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "population.pt").exists():
        raise FileExistsError("Population exists; choose a new output directory.")
    pairs, inputs, labels = addition_suite()
    layout = parameter_layout(Adder(config.hidden))
    n_params = layout[-1]["stop"]
    shape = (config.runs, config.steps + 1)
    # Preallocation avoids keeping multiple copies of a long checkpoint list.
    weights = torch.empty(*shape, n_params)
    loss = torch.empty(*shape)
    accuracy = torch.empty(*shape)
    problem_loss = torch.empty(*shape, 100)
    predictions = torch.empty(*shape, 100, dtype=torch.int16)
    seeds = torch.arange(config.seed, config.seed + config.runs)
    for run, seed in enumerate(seeds.tolist()):
        torch.manual_seed(seed)
        model = Adder(config.hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        for step in range(config.steps + 1):
            # Evaluating does not backpropagate or change weights. Step zero is
            # the untrained network; checkpoint k follows exactly k updates.
            with torch.no_grad():
                scores = model(inputs)
                individual = F.cross_entropy(scores, labels, reduction="none")
                guesses = scores.argmax(dim=1)
                weights[run, step] = parameters_to_vector(model.parameters())
                loss[run, step] = individual.mean()
                accuracy[run, step] = (guesses == labels).float().mean()
                problem_loss[run, step] = individual
                predictions[run, step] = guesses.to(torch.int16)
            if step == config.steps:
                break
            # The four essential training operations: clear old gradients,
            # predict, differentiate the loss, then update the parameters.
            optimizer.zero_grad(set_to_none=True)
            objective = F.cross_entropy(model(inputs), labels)
            objective.backward()
            optimizer.step()
        if (run + 1) % 10 == 0 or run == 0:
            print(
                f"Targets {run + 1}/{config.runs}: final loss={loss[run, -1]:.4f}, "
                f"accuracy={accuracy[run, -1]:.1%}",
                flush=True,
            )
    manifest = {
        "config": asdict(config),
        "layout": layout,
        "torch_version": str(torch.__version__),
        "suite": "All 100 ordered digit pairs; also target training data",
        "parameter_count": n_params,
        "format_version": 1,
    }
    torch.save(
        {
            "weights": weights,
            "loss": loss,
            "accuracy": accuracy,
            "problem_loss": problem_loss,
            "predictions": predictions,
            "pairs": pairs,
            "labels": labels,
            "seeds": seeds,
            "steps": torch.arange(config.steps + 1),
            "manifest": manifest,
        },
        output / "population.pt",
    )
    (output / "population.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output / "population.pt"
