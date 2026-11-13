"""Reproduce SYN88 target lifetimes without loading pickle checkpoints.

This is the training recipe, not a promise of bit-identical CUDA regeneration.
Use the published manifest-bound population to reproduce historical evaluations.
"""
from pathlib import Path
import argparse
import json
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector
from .targets import Adder, addition_suite

AGES = (0, 1, 2, 5, 10, 25, 50, 100)
RULES = ("addition", "alternate")


def split_roots():
    order = torch.randperm(2304, generator=torch.Generator().manual_seed(880088)).tolist()
    return {"train": order[:1536], "validation": order[1536:1792], "test": order[1792:]}


def generate_root(root_index, device="cpu"):
    """Return numeric NPZ fields for one paired root, with historical seeds."""
    if not isinstance(root_index, int) or not 0 <= root_index < 2304:
        raise ValueError("root_index must be in [0, 2304)")
    seed = 492232 + root_index
    pairs, inputs, answers = addition_suite()
    inputs = inputs.to(device)
    weights, correct = [], []
    for rule in RULES:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        if torch.device(device).type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = Adder(hidden=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        labels = (answers if rule == "addition" else (2*pairs[:, 0]+pairs[:, 1]+1)%19).to(device)
        rule_weights, rule_correct = [], []
        for update in range(101):
            if update in AGES:
                model.eval()
                with torch.no_grad():
                    rule_weights.append(parameters_to_vector(model.parameters()).cpu().numpy().copy())
                    rule_correct.append(model(inputs).argmax(-1).cpu().eq(answers).numpy().astype(np.uint8))
                model.train()
            if update == 100:
                break
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(inputs), labels).backward()
            optimizer.step()
        weights.append(rule_weights)
        correct.append(rule_correct)
    return {"weights": np.asarray(weights, dtype=np.float32),
            "addition_correct": np.asarray(correct, dtype=np.uint8),
            "root_index": np.asarray(root_index, dtype=np.int64),
            "root_seed": np.asarray(seed, dtype=np.int64),
            "ages": np.asarray(AGES, dtype=np.int64)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1, help="2304 for the complete population")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.start < 0 or args.count < 1 or args.start + args.count > 2304:
        parser.error("requested roots must be within [0, 2304)")
    args.output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: never replace an existing population or partial result.
    with (args.output / "root-splits.json").open("x", encoding="utf-8") as f:
        json.dump(split_roots(), f, indent=2)
    for index in range(args.start, args.start + args.count):
        payload = generate_root(index, args.device)
        with (args.output / f"root-{index:04d}.npz").open("xb") as f:
            np.savez_compressed(f, **payload)
        print(f"saved root {index}", flush=True)


if __name__ == "__main__":
    main()
