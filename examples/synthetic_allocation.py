"""Run with `python examples/synthetic_allocation.py` after editable installation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from synaptic_lens.allocation import normalized_multipliers, policy_scores


def main():
    losses = torch.tensor([0.01, 0.25, 1.0], device="cpu")
    scores = policy_scores("sqrt_loss", unreduced_loss=losses)
    coefficients = torch.full_like(losses, 1 / len(losses))
    weights = normalized_multipliers(scores, coefficients)
    assert torch.isclose((weights * coefficients).sum(), torch.tensor(1.0))
    print("Synthetic CPU allocation weights:", weights.tolist())


if __name__ == "__main__":
    main()
