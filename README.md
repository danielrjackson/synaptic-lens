# Synaptic Lens

Research software for asking what a neural network's weights reveal about its behavior. The project builds structured representations of a target network and trains separate readers to answer questions about that target.

## Research highlights

The work spans weights-only competence prediction, query-based attention readers, and controlled allocation of training effort. [Selected research results](docs/RESULTS.md) present the evidence and contrary findings: a structured observer outperformed strong shortcut baselines in a narrow output-rewiring study; regional attention improved probability estimates without establishing reliable query localization; learned allocation showed mixed gains at substantial probe cost. These are historical controlled studies, not broad interpretability claims.

## What is here

- Raw-weight and neuron-invariant learned observers, train-only normalization, validation selection, statistics-ridge baselines and whole-run uncertainty estimates.
- Deterministic training-prefix continuation with model, optimizer and sampler snapshots.
- Exact records of supported affine/ReLU networks, with explicit parameter coverage and unsupported-input checks.
- Paired attention readers over network records, separating target encoding from queried outputs.
- A value-blind masked-scalar reader and local graph-context control.
- Detached allocation policies, antithetic optimizer updates, ranking/AUC and table export.
- Small deterministic tests for numeric coverage, masking, permutations, queries and reconstruction contracts.

These are research components, not a general-purpose interpreter for arbitrary models. Implementing a query interface or passing a unit test does not establish learned predictive accuracy, causal localization or generalization. Attention weights alone are not explanations.

## Try a small CPU example

Use Python 3.13 or later in a virtual environment. Install a CPU-compatible PyTorch build for your platform and pytest. For a default installation:

```sh
python -m pip install -e ".[test]"
python examples/train_observer.py
python examples/inspect_target.py
python examples/synthetic_allocation.py
```

The learned-observer example generates small addition networks, keeps whole target runs in separate train/validation/test partitions, and fits raw/structured observers against statistics-ridge and train-mean controls. It uses a fixed small CPU budget and automatically removes temporary targets. It reports whatever wins, including baselines: it is an executable workflow, not reproduction of the historical study scores.

The record-inspection example constructs a tiny random network locally, records its parameters, and checks that evaluating those records exactly reproduces the network's outputs. It does not train a reader or demonstrate learned prediction. No GPU, downloaded model, account, external service or model checkpoint is required.

The allocation example computes normalized allocation weights from synthetic losses; it does not establish improved training performance. [Public scope and validation](docs/PUBLIC_SCOPE.md) describe the included interfaces, intentional omissions and tested environment.

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
manifest references. These mechanisms do not claim historical result reproduction.

## Models and reproducibility

Model files are hosted separately in four [public Hugging Face repositories](docs/ARTIFACTS.md), not in this Git repository. The example and core unit tests generate their own small inputs and do not depend on those artifacts. Reproducing trained-model research findings requires the relevant checkpoint, dataset and protocol; the runnable example must not be mistaken for that reproduction.

## Project and history

Daniel Jackson develops Synaptic Lens as an independent research project. AI tools assist engineering and curation; that assistance does not independently validate research claims. This repository contains a deliberately selected development history and a curated distribution, not a complete record of every experiment or development step. Selected historical commits retain original source authorship and author dates; curation is recorded separately.

[Selected references](docs/REFERENCES.md) explain which sources informed the reader design and which algorithms/software the project uses. They distinguish design motivation from reproduction or validated capability. A [BibTeX bibliography](docs/references.bib) is included.

## License and citation

The source code uses the unmodified [AI Pubs research-use RAIL-S license, Version 0.1, March 2, 2023](LICENSE.md). Its permitted purpose is academic or research use only, subject to the full license and use restrictions. This is not a general noncommercial-use license; it does not categorically prohibit ICE or all weapons uses. Retain the attribution in [NOTICE](NOTICE).

The companion unmodified [AI Pubs research-use RAIL-M license, Version 0.1](LICENSE-MODEL.md) applies only to model files expressly designated in a release manifest or model card. It does not license complementary software or documentation, datasets, or third-party materials. Its inclusion here does not license absent checkpoints. These are use-restricted licenses, not OSI-approved open-source licenses.

Third-party dependencies and separately licensed materials retain their own licenses. Earlier copies remain subject to the terms under which they were provided.

If you use the software in research, please cite this repository and the revision used. [CITATION.cff](CITATION.cff) provides citation metadata; this citation request is not an additional license condition.
