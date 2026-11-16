# Synaptic Lens

Research software for asking what a neural network's weights reveal about its behavior. The project builds structured representations of a target network and trains separate readers to answer questions about that target.

## Research highlights

The work spans weights-only competence prediction, query-based attention readers, network activations, and controlled allocation of training effort. [Selected research results](docs/RESULTS.md) present the evidence and contrary findings: a structured observer outperformed strong shortcut baselines in a narrow output-rewiring study; regional attention improved probability estimates without establishing reliable query localization; learned allocation showed mixed gains at substantial probe cost.

## What is here

- Raw-weight and neuron-invariant learned observers, train-only normalization, validation selection, statistics-ridge baselines and whole-run uncertainty estimates.
- Deterministic training-prefix continuation with model, optimizer and sampler snapshots.
- Records of supported affine/ReLU networks, with explicit parameter coverage and unsupported-input checks.
- Paired attention readers over network records, separating target encoding from queried outputs.
- A value-blind masked-scalar reader and local graph-context control.
- Detached allocation policies, antithetic optimizer updates, ranking/AUC and table export.
- Small deterministic tests for numeric coverage, masking, permutations, queries and reconstruction contracts.

## Try a pretrained Adder Lens

The [pretrained Adder walkthrough](docs/ADDER_EXAMPLE.md) downloads the featured
SYN88 reader and one target root, verifies their hashes, and reports predictions
across its training ages. [Model downloads](docs/DOWNLOADS.md) cover all four
published Lens and target releases, with pinned revisions and cache reuse.

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

The following self-contained fixtures keep their small CPU budget:

```powershell
uv run --locked python examples/train_observer.py
uv run --locked python examples/inspect_target.py
uv run --locked python examples/synthetic_allocation.py
```

The learned-observer example generates small addition networks, keeps whole target runs in separate train/validation/test partitions, and fits raw/structured observers against statistics-ridge and train-mean controls. It uses a fixed small CPU budget and automatically removes temporary targets.

The record-inspection example constructs a tiny random network locally, records its parameters, and checks that evaluating those records exactly reproduces the network's outputs. This CPU-only example checks record construction and output reconstruction using locally generated inputs.

The allocation example demonstrates normalized allocation weights computed from synthetic losses. [Public scope and validation](docs/PUBLIC_SCOPE.md) describe the included interfaces, intentional omissions and tested environment.

To run the selected core tests from the repository root:

```sh
python -m pytest tests/core -q -o pythonpath=src
```

For platform-specific PyTorch installation options, see the [official installation guide](https://pytorch.org/get-started/locally/).

## Trained Lens milestones

The [milestone catalog](docs/milestones/README.md) separates arithmetic-target
readers (including the later activation-assisted trajectory reader) from weights-only observers of image-recognition targets. It documents
selected trained checkpoints, exact input contracts, saved comparisons (including
negative results), and hash-verified local inference commands. Selected checkpoints and target populations are now available on Hugging Face;
see [published artifacts](docs/ARTIFACTS.md) for exact releases and limitations.
Image-fed Lens experiments are excluded.

See [training and development workflows](docs/TRAINING.md) for the bounded
trajectory optimizer, root-isolated MNIST mean-risk fitting API and exact release
manifest references.

## Models and reproducibility

Model files are hosted in four [public Hugging Face repositories](docs/ARTIFACTS.md). The example and core unit tests generate their own small inputs and do not depend on those artifacts. Reproducing trained-model research findings requires the relevant checkpoint, dataset and protocol. The small runnable example exercises the software workflow.

## Project and history

Daniel Jackson develops Synaptic Lens as an independent research project. AI tools assist engineering and curation. This repository presents selected development milestones and reusable research software. Selected historical commits retain original source authorship and author dates; curation is recorded separately.

[Selected references](docs/REFERENCES.md) explain which sources informed the reader design and which algorithms/software the project uses. They distinguish design motivation from reproduction or validated capability. A [BibTeX bibliography](docs/references.bib) is included.

## License and citation

The source code uses the [AI Pubs research-use RAIL-S license, Version 0.1, March 2, 2023](LICENSE.md). Its permitted purpose is academic or research use only, subject to the full license and use restrictions. Retain the attribution in [NOTICE](NOTICE).

The companion [AI Pubs research-use RAIL-M license, Version 0.1](LICENSE-MODEL.md) applies only to model files expressly designated in a release manifest or model card. Software, documentation, datasets and third-party materials retain their respective terms; each model release identifies the files covered by RAIL-M.

Third-party dependencies and separately licensed materials retain their own licenses. Earlier copies remain subject to the terms under which they were provided.

If you use the software in research, please cite this repository and the revision used. [CITATION.cff](CITATION.cff) provides citation metadata.
