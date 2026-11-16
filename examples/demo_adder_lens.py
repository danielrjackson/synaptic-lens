"""Show a pretrained SYN88 Lens reading addition-network lifetimes on CPU."""
from pathlib import Path
import argparse
import io
import json
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from synaptic_lens.checkpoints import verified_bytes
from synaptic_lens.trajectory_checkpoint import load_model, predict

CHECKPOINT_SHA256 = "20cf06bf7599c457b56e42c381c7876ba2119985a98bb0a36a8633e3cffd614f"


def summarize(logits, labels):
    """Summarize the supplied labeled sample with stable probability math."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.shape != labels.shape or not np.isfinite(logits).all():
        raise ValueError("Predictions and labels must have equal finite shapes")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Expected binary addition_correct labels")
    probabilities = np.exp(-np.logaddexp(0.0, -logits))
    return {
        "queries": int(labels.size),
        "lens_accuracy": float(np.mean((logits >= 0) == labels)),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "target_addition_accuracy": float(np.mean(labels)),
        "mean_predicted_correctness": float(np.mean(probabilities)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Published syn88/lens-trajectory-seed-27.npz")
    parser.add_argument("--targets", type=Path, required=True,
                        help="Published SYN88 numeric root bundle")
    parser.add_argument("--targets-sha256", required=True)
    args = parser.parse_args()
    payload = verified_bytes(args.targets, args.targets_sha256)
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        weights = data["weights"].copy()
        labels = data["addition_correct"].copy()
        ages = data["ages"].copy()
    if weights.shape != (2, 8, 1299) or labels.shape != (2, 8, 100):
        raise ValueError("Expected one published root with two rules and eight ages")
    if ages.shape != (8,) or not np.array_equal(ages, [0, 1, 2, 5, 10, 25, 50, 100]):
        raise ValueError("Unexpected SYN88 checkpoint ages")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    model, norm = load_model(args.checkpoint, CHECKPOINT_SHA256)
    logits = predict(model, norm, weights)
    rows = []
    for rule in range(2):
        for stage, age in enumerate(ages):
            rows.append({"training_rule": "addition" if rule == 0 else "alternate",
                         "target_update": int(age),
                         **summarize(logits[rule, stage], labels[rule, stage])})
    print(json.dumps({
        "model": "SYN88 trajectory seed 27, update 17000",
        "selection": "Lowest saved validation objective among the three released trajectory seeds",
        "scope": "CPU inference on the supplied root; sample-level descriptive metrics",
        "input": "Canonical target weights, arithmetic queries, query-aligned hidden activations",
        "summary": summarize(logits, labels), "by_rule_and_age": rows,
    }, indent=2))


if __name__ == "__main__":
    main()
