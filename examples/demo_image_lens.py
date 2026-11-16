"""Download and run the released MNIST weight-statistics Lens on CUDA or CPU."""
from pathlib import Path
import argparse
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from download_models import download_release
from synaptic_lens.checkpoints import load_milestone, predict_mnist

LENS_FILE = "lens-tensor_statistics.pt"
TARGET_FILE = "mnist-d32-l2-root-1007/states/state-update-00196.pt"
SCHEMA_FILE = "coordinate-schema.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; install the CUDA profile or select --device cpu")
    options = {"cache_dir": args.cache_dir, "local_files_only": args.local_files_only}
    lens_release = download_release("synaptic-lens-mnist", [LENS_FILE], **options)
    target_release = download_release("synaptic-lens-targets-mnist", [TARGET_FILE, SCHEMA_FILE], **options)
    lens = lens_release["files"][LENS_FILE]
    target = target_release["files"][TARGET_FILE]
    schema = target_release["files"][SCHEMA_FILE]
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    bundle = load_milestone("mnist-statistics", lens["path"], inference_export=True)
    prediction = predict_mnist("mnist-statistics", bundle, target["path"],
                               target["sha256"], schema["path"], device=args.device)
    print(json.dumps({
        "lens": "MNIST tensor-statistics Lens",
        "device": args.device,
        "target": TARGET_FILE,
        "input": "192 statistics computed from the target network's weights",
        "predicted_mean_cross_entropy_nats": prediction,
        "lens_revision": lens_release["revision"],
        "target_revision": target_release["revision"],
        "lens_sha256": lens["sha256"],
        "target_sha256": target["sha256"],
        "coordinate_schema_sha256": schema["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
