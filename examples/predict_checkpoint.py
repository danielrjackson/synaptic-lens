"""Run local, audited historical Lens checkpoints. Does not download artifacts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import argparse
import json
import torch
from torch import nn
from synaptic_lens.checkpoints import CHECKPOINTS, load_milestone, predict_mnist
from synaptic_lens.target_records import affine_relu_records
from synaptic_lens.transformer_lens import LensQuery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--milestone', choices=sorted(CHECKPOINTS), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--inference-export', action='store_true', help='Use separately hash-pinned inference-only export, not original checkpoint')
    parser.add_argument('--target', type=Path)
    parser.add_argument('--target-sha256')
    parser.add_argument('--coordinate-schema', type=Path)
    args = parser.parse_args()
    if args.milestone.startswith('mnist-') and not all((args.target,args.target_sha256,args.coordinate_schema)):
        parser.error('MNIST requires --target, --target-sha256 and --coordinate-schema')
    if args.milestone.startswith('syn53-') and any((args.target,args.target_sha256,args.coordinate_schema)):
        parser.error('SYN53 example uses a synthetic adder; target-file import is not implemented')
    torch.set_num_threads(1)
    model = load_milestone(args.milestone, args.checkpoint, inference_export=args.inference_export)
    if args.milestone.startswith('mnist-'):
        result = {'predicted_mean_cross_entropy': predict_mnist(args.milestone,model,args.target,args.target_sha256,args.coordinate_schema),
                  'scope': 'scalar target risk, not digit prediction or a reproduced study metric'}
    else:
        torch.manual_seed(7)
        target = nn.Sequential(nn.Linear(20,32),nn.ReLU(),nn.Linear(32,19))
        cases = torch.zeros(3,20)
        pairs = [(0,0),(2,3),(9,9)]
        for i,(a,b) in enumerate(pairs):
            cases[i,a]=1;cases[i,10+b]=1
        with torch.inference_mode():
            response=model.predict(model.encode((affine_relu_records(target),)),LensQuery(cases))
        if response.status != 'supported' or response.values is None:
            raise ValueError(f'Unsupported prediction: {response.status}: {response.reason}')
        result={'queries':pairs,'measurements':response.measurements,'values':response.values.tolist(),
                'scope':'trained Lens on synthetic random adder; not historical study reproduction'}
    print(json.dumps({'milestone':args.milestone,**result},indent=2))

if __name__ == '__main__':
    main()
