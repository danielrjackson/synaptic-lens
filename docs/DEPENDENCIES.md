# Current dependency environment

Validated upgrade snapshot: 2026-11-16, Windows x64.

- CPython 3.15.2 (standard GIL build).
- uv 0.13.8; the project checks this version before resolution/synchronization.
- PyTorch 2.15.0+cu134, CUDA runtime 13.4, NumPy 2.5.5.
- pytest 9.2.0, setuptools 84.2.1, optional huggingface-hub 2.4.0.
- `uv.lock` records all resolved transitive versions and archive hashes.

The versions were resolved from PyPI and the official PyTorch CUDA 13.4 index.
The explicitly named PyTorch index supplies torch; ordinary dependencies resolve
from PyPI. The lock uses CUDA wheels on Windows/Linux and the regular PyPI wheel
on other systems. Local validation covers Windows x64 and the NVIDIA RTX 4090.
The Python compatibility floor remains 3.13; `.python-version` selects the current
3.15.2 interpreter for maintained development. Other profiles need separate testing.

## Setup

Install uv 0.13.8 from its official release/package source, then run:

```powershell
uv python install 3.15.2
uv sync --locked --extra test --extra hub
uv run --locked --extra test pytest
uv run --locked python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
```

The PyTorch wheel includes its matching CUDA runtime. A compatible NVIDIA driver
is required; compiling CUDA extensions additionally needs a suitable toolkit.
This upgrade does not replace system drivers or toolkits. The validation machine
uses NVIDIA driver 621.24. Use a fresh environment to preserve existing work.

The small synthetic fixture scripts intentionally retain CPU execution. The
pretrained Adder walkthrough defaults to CUDA and supports `--device cpu`.
For a separate CPU-only package environment, install the official CPU torch wheel
first, then the checkout with pip. Keep that environment separate from the CUDA
lock; `uv sync` restores the locked CUDA selection.

## Updating

Review official release and compatibility information, update `.python-version`,
intentional package floors and the uv requirement, then use `uv lock --upgrade`.
Review the lock diff, synchronize a fresh environment, run all core tests, examples
and the bounded CUDA parity check. Retain historical model artifacts and scientific
results unchanged; a modern software environment is separately validated.

SymPy 1.14.0 constrains mpmath to <1.4, so the lock selects mpmath 1.3.0 even
though 1.4.1 is available. This is a dependency compatibility constraint.

Official sources: [Python](https://www.python.org/downloads/),
[PyTorch installation](https://pytorch.org/get-started/locally/),
[PyTorch CUDA wheels](https://download.pytorch.org/whl/cu134/torch/),
[uv](https://docs.astral.sh/uv/), [PyPI](https://pypi.org/).

## Validation receipt

The fresh isolated Python 3.15.2 CUDA environment passed 136 core tests and six
subtests, including the new CUDA trajectory parity test on the
RTX 4090, comparing CPU and GPU predictions with rtol=1e-4 and atol=1e-5.
A Python 3.15 linecache deprecation warning was emitted by one existing fixture.
No training experiment, historical metric or published model was changed.

The pinned public SYN88 root-0000 walkthrough also completed on CUDA: 1,600
predictions, 99.625% sample accuracy and Brier 0.0030306472. This is a software
smoke check on one sample, separate from the historical benchmark.
