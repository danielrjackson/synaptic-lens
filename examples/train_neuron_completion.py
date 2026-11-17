"""Train a missing-neuron Lens from public, hash-verified target roots."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from download_models import download_release

from synaptic_lens.completion_training import (
    load_population,
    train_completion,
    training_partitions,
)
from synaptic_lens.neuron_completion import (
    OrganicCheckpointReader,
    prepare_organic_completion,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--fit-roots", type=int, default=64)
    parser.add_argument("--development-roots", type=int, default=16)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory; previous runs are preserved")
    if not 1 <= args.updates <= 20000:
        parser.error("--updates must be between 1 and 20000")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; select --device cpu explicitly for a smoke run")
    try:
        partitions = training_partitions(args.fit_roots, args.development_roots)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    names = [
        f"syn88/population/root-{r:04d}.npz" for ids in partitions.values() for r in ids
    ]
    release = download_release(
        "synaptic-lens-targets-adder",
        names,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    pops = {k: load_population(ids, release["files"]) for k, ids in partitions.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoint, report = train_completion(
        pops["fit"], pops["development"], updates=args.updates, device=args.device
    )
    path = args.output / "your-reader.pt"
    torch.save(checkpoint, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    # Exercise the saved checkpoint through the same public inference wrapper.
    reader = OrganicCheckpointReader(path, digest, args.device)
    raw, bias = pops["development"]["raw"][0], pops["development"]["bias"][0]
    view = prepare_organic_completion(raw, bias, 0)
    restored = view.predict(reader)
    zero = raw.clone()
    zero[0] = 0
    pairs = torch.cartesian_prod(torch.arange(10), torch.arange(10))
    inputs = torch.cat(
        [torch.nn.functional.one_hot(pairs[:, i], 10) for i in range(2)], -1
    ).float()
    comparison = {}
    for name, neurons in (("original", raw), ("zero", zero), ("your_reader", restored)):
        answers = (
            (inputs @ neurons[:, :20].T + neurons[:, 20]).relu() @ neurons[:, 21:]
            + bias
        ).argmax(-1)
        comparison[name] = {
            "correct_of_100": int((answers == pairs.sum(-1)).sum()),
            "answers": answers.tolist(),
        }
    report.update(
        {
            "checkpoint_sha256": digest,
            "target_revision": release["revision"],
            "target_hashes": {n: release["files"][n]["sha256"] for n in names},
            "example": {
                "root": partitions["development"][0],
                "rule": "addition",
                "physical_neuron": 0,
                "selection": "first development target; no score ranking",
                "external_true_scale_retained": True,
                "actual_weights": raw[0].tolist(),
                "predicted_weights": restored[0].tolist(),
                "comparison": comparison,
            },
        }
    )
    (args.output / "training-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "neurons": raw,
            "output_bias": bias,
            "root": partitions["development"][0],
            "rule": "addition",
        },
        args.output / "example-target.pt",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_step": report["selected_step"],
                "development_example": comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
