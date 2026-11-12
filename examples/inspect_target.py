"""Small CPU-only numeric-coverage example; no model downloads or training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn
from synaptic_lens.target_records import affine_relu_records


def main():
    torch.manual_seed(7)
    target = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    records = affine_relu_records(target)
    inputs = torch.randn(5, 3)
    torch.testing.assert_close(records.evaluate(inputs), target(inputs), rtol=0, atol=0)
    print(f"Exact reconstruction verified: {records.parameter_count} parameters, {len(inputs)} inputs.")


if __name__ == "__main__":
    main()
