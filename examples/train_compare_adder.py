"""Train an Adder Lens from released targets and compare it with the featured Lens."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "examples")]
from download_models import download_release
from demo_adder_lens import summarize
from synaptic_lens.checkpoints import verified_bytes
from synaptic_lens.trajectory_checkpoint import load_model, predict
from synaptic_lens.trajectory_data import TrajectoryPopulation, require_disjoint_roots
from synaptic_lens.syn88_targets import split_roots
from synaptic_lens.trajectory_training import save_inference_export, train_trajectory

LENS_FILE = "syn88/lens-trajectory-seed-27.npz"
AGES = [0, 1, 2, 5, 10, 25, 50, 100]


def root_partitions():
    """Small fixed subsets of the historical root-isolated partitions."""
    roots = split_roots()
    return {"train": roots["train"][:8], "validation": roots["validation"][:2], "comparison": roots["test"][:2]}


def load_population(ids, files):
    weights, labels = [], []
    for root in ids:
        entry = files[f"syn88/population/root-{root:04d}.npz"]
        with np.load(io.BytesIO(verified_bytes(entry["path"], entry["sha256"])), allow_pickle=False) as data:
            if int(data["root_index"]) != root or not np.array_equal(data["ages"], AGES):
                raise ValueError("Target identity or training ages disagree with the requested root")
            weights.append(torch.from_numpy(data["weights"].copy()))
            labels.append(torch.from_numpy(data["addition_correct"].copy()))
    return TrajectoryPopulation(tuple(ids), torch.stack(weights), torch.stack(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("runs/adder-comparison"))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.updates <= 20000:
        parser.error("--updates must be between 1 and 20000")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install the CUDA profile or select --device cpu")
    if args.output.exists():
        parser.error("--output already exists; choose a new directory to keep previous runs")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    partitions = root_partitions()
    options = {"cache_dir": args.cache_dir, "local_files_only": args.local_files_only}
    names = [f"syn88/population/root-{root:04d}.npz" for ids in partitions.values() for root in ids]
    print("Downloading 12 target roots and the featured Lens...", file=sys.stderr)
    targets = download_release("synaptic-lens-targets-adder", names, **options)
    reference = download_release("synaptic-lens-adder", [LENS_FILE], **options)
    populations = {key: load_population(ids, targets["files"]) for key, ids in partitions.items()}
    require_disjoint_roots(*populations.values())
    print(f"Training for {args.updates} updates on {args.device}...", file=sys.stderr)
    model, norm, training = train_trajectory(
        populations["train"], populations["validation"], updates=args.updates,
        batch_roots=2, validation_every=min(25, args.updates), device=args.device,
    )
    # Comparison roots only enter after fitting and validation selection finish.
    print("Comparing readers on the two reserved roots...", file=sys.stderr)
    sample = populations["comparison"]
    lens = reference["files"][LENS_FILE]
    featured, featured_norm = load_model(lens["path"], lens["sha256"])
    logits = {
        "your_lens": predict(model, norm, sample.weights),
        "featured_lens": predict(featured.to(args.device), featured_norm, sample.weights),
    }
    labels = sample.addition_correct.numpy()
    report = {
        "device": args.device, "root_partitions": partitions, "training": training,
        "comparison": {key: summarize(value, labels) for key, value in logits.items()},
        "comparison_scope": "Two fixed historical test roots; all rules, ages and 100 digit pairs. Your Lens trains on eight roots; the featured Lens trained on 1536.",
        "target_revision": targets["revision"], "lens_revision": reference["revision"],
        "target_files": targets["files"], "featured_lens_sha256": lens["sha256"],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output / "your-lens.npz"
    save_inference_export(checkpoint, model, norm)
    report["your_lens_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # Verify the persisted model, rather than reporting only the in-memory result.
    restored, restored_norm = load_model(checkpoint, report["your_lens_sha256"])
    restored_logits = predict(restored.to(args.device), restored_norm, sample.weights)
    np.testing.assert_allclose(restored_logits, logits["your_lens"], atol=1e-6, rtol=1e-6)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["root", "rule", "target_update", "a", "b", "target_correct", "your_probability", "featured_probability"])
        probs = {k: np.exp(-np.logaddexp(0, -v)) for k, v in logits.items()}
        for r, root in enumerate(sample.root_ids):
            for rule in range(2):
                for aidx, age in enumerate(AGES):
                    for q in range(100):
                        writer.writerow([root, rule, age, q // 10, q % 10, int(labels[r, rule, aidx, q]),
                                         float(probs["your_lens"][r, rule, aidx, q]),
                                         float(probs["featured_lens"][r, rule, aidx, q])])
    print(json.dumps({"output": str(args.output), "comparison": report["comparison"], "selected_update": training["selected_update"]}, indent=2))


if __name__ == "__main__":
    main()
