# Synaptic Lens

Research software for asking what a neural network's weights reveal about its behavior. The project builds structured representations of a target network and trains separate readers to answer questions about that target.

## Research highlights

The featured Adder Lens predicted target-network correctness with **99.6167% accuracy** across 819,200 queries from 512 independent target-root families in its saved evaluation. It reads target weights, arithmetic queries and query-aligned hidden activations. See the [Adder walkthrough](docs/ADDER_EXAMPLE.md) for the model, benchmark and runnable example.

Other studies explore weights-only competence prediction, query-based attention readers and allocation of training effort. A structured observer outperformed parameter-statistics baselines on output-rewired addition networks, and regional attention improved probability estimates. [Selected research results](docs/RESULTS.md) include the comparisons, unsuccessful approaches and evaluation methods.

## What is here

- Raw-weight and neuron-invariant learned observers, train-only normalization, validation selection, statistics-ridge baselines and whole-run uncertainty estimates.
- Deterministic training-prefix continuation with model, optimizer and sampler snapshots.
- Records of supported affine/ReLU networks, with explicit parameter coverage and unsupported-input checks.
- Paired attention readers over network records, separating target encoding from queried outputs.
- A value-blind masked-scalar reader and local graph-context control.
- Detached allocation policies, antithetic optimizer updates, ranking/AUC and table export.
- Small deterministic tests for numeric coverage, masking, permutations, queries and reconstruction contracts.

## Try the pretrained readers

The [pretrained Adder walkthrough](docs/ADDER_EXAMPLE.md) downloads the featured
SYN88 reader and one target root, verifies their hashes, and reports predictions
across its training ages. [Model downloads](docs/DOWNLOADS.md) cover all four
published Lens and target releases, with pinned revisions and cache reuse.

The [image-recognition walkthrough](docs/IMAGE_EXAMPLE.md) runs the released MNIST
tensor-statistics Lens. It reads a classifier’s saved parameters and predicts
its mean classification loss. Both walkthroughs default to CUDA.

[Train your own Adder Lens](docs/TRAINING.md#train-an-adder-lens-and-compare-it-with-the-released-model)
downloads trained targets, fits the same reader architecture and compares your
model with the featured Lens on the same queries. It saves your checkpoint,
metrics and a spreadsheet-friendly prediction table.

[Complete a missing adder neuron](docs/NEURON_COMPLETION.md) downloads a trained
position-free Lens and an example target. Hide any neuron, inspect its predicted
weights, and compare the target's answers before and after reconstruction. This
is a separate weight-completion task from the correctness reader above.

## Install the current CUDA environment

The maintained environment uses Python 3.15.2, uv 0.13.8 and PyTorch 2.15.0
with CUDA 13.4. The lock includes the runtime, test and optional Hub dependencies.
See [dependencies and validation](docs/DEPENDENCIES.md) for platform requirements,
CPU options and upgrade details.

```powershell
uv sync --locked --extra test --extra hub
uv run --locked python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run --locked --extra test pytest
```

## Try the small CPU fixtures

These self-contained examples run on CPU:

```powershell
uv run --locked python examples/train_observer.py
uv run --locked python examples/inspect_target.py
uv run --locked python examples/synthetic_allocation.py
```

The learned-observer example generates small addition networks, keeps whole target runs in separate train/validation/test partitions, and fits raw/structured observers against statistics-ridge and train-mean controls. It uses a fixed small CPU budget and automatically removes temporary targets.

The record-inspection example constructs a tiny random network locally, records its parameters, and checks that evaluating those records exactly reproduces the network's outputs. The inputs are generated locally.

The allocation example demonstrates normalized allocation weights computed from synthetic losses. [Public scope and validation](docs/PUBLIC_SCOPE.md) describes the included interfaces and test coverage.

To run the selected core tests from the repository root:

```sh
python -m pytest tests/core -q -o pythonpath=src
```

For platform-specific PyTorch installation options, see the [official installation guide](https://pytorch.org/get-started/locally/).

## Trained Lens milestones

The [milestone catalog](docs/milestones/README.md) covers arithmetic readers and weights-only observers of image-recognition targets. Each page describes the reader's inputs, trained checkpoints, results and inference commands. Download the checkpoints and target populations from [Hugging Face](docs/ARTIFACTS.md).

See [training and development workflows](docs/TRAINING.md) for the bounded
trajectory optimizer, root-isolated MNIST mean-risk fitting API and exact release
manifest references.

## Models and reproducibility

Model files are hosted in four [public Hugging Face repositories](docs/ARTIFACTS.md). The pretrained walkthrough uses released artifacts; the small CPU fixtures and core unit tests generate their own inputs. Each milestone documents its checkpoints, data requirements and evaluation protocol.

## Project and history

Daniel Jackson develops Synaptic Lens as an independent research project. AI tools assist engineering and curation. This repository presents selected development milestones and reusable research software. Selected historical commits retain original source authorship and author dates; curation is recorded separately.

[Selected references](docs/REFERENCES.md) describe the research behind the reader design and the algorithms and software used here. A [BibTeX bibliography](docs/references.bib) is included.

## License and citation

The source code uses the [AI Pubs research-use RAIL-S license, Version 0.1, March 2, 2023](LICENSE.md). Its permitted purpose is academic or research use only, subject to the full license and use restrictions. Retain the attribution in [NOTICE](NOTICE).

The companion [AI Pubs research-use RAIL-M license, Version 0.1](LICENSE-MODEL.md) applies only to model files expressly designated in a release manifest or model card. Software, documentation, datasets and third-party materials retain their respective terms; each model release identifies the files covered by RAIL-M.

Third-party dependencies and separately licensed materials retain their own licenses. Earlier copies remain subject to the terms under which they were provided.

If you use the software in research, please cite this repository and the revision used. [CITATION.cff](CITATION.cff) provides citation metadata.
