"""Predict all 100 addition queries from a verified local SYN88 export."""
from pathlib import Path
import argparse
import io
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from synaptic_lens.trajectory_checkpoint import load_model, predict
from synaptic_lens.checkpoints import verified_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--targets-sha256', required=True)
    args = parser.parse_args()
    target_bytes = verified_bytes(args.targets, args.targets_sha256)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    model, norm = load_model(args.checkpoint, args.checkpoint_sha256)
    with np.load(io.BytesIO(target_bytes), allow_pickle=False) as source:
        weights = source['weights'].copy()
    logits = predict(model, norm, weights)
    print(json.dumps({'shape': list(logits.shape), 'logits': logits.tolist(),
                     'scope': 'local inference, not a full study rerun; query-aligned activations are inputs'}))


if __name__ == '__main__':
    main()
