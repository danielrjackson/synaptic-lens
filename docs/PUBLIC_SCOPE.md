# Public scope and validation

This distribution collects reusable research components.
The current integration includes exact affine/ReLU records, paired readers,
value-blind scalar reconstruction, reader configuration, detached allocation
policies, antithetic Adam updates, descriptive ranking/AUC, and CSV output.
It also includes raw/structured weights-only observers, small target generation,
whole-run fitting/evaluation and deterministic training-prefix continuation.

The source modules retain their `synaptic_lens.<module>` import names.
`ReaderConfig` and `DedicatedBoundaryReaderConfig` retain the serialization
identity `synaptic_lens.transformer_lens`, even though their definitions live in
`reader_config.py`. Compatibility is checked for the documented checkpoint identities. Load only
trusted artifacts with verified provenance.

Tests cover numeric records, token/query contracts, boundary attention,
masking, configuration serialization, policy detachment, allocation gradients,
optimizer-state updates, ranking ties and output bytes. The optimizer fixture is
an ordinary small neural network. The small CPU
examples need no stored model or dataset. Allocation weights demonstrate the policy mechanism. Training-performance
measurements are reported separately in the research results.

## Deliberate omissions

Study orchestration, private working notes, agent coordination, operational
supervision and storage management are outside this small distribution. So are
unselected study checkpoint adapters, pretrained weights, dataset caches and
training outputs. Selected hash-bound inference adapters are included for SYN53, SYN88 and the direct-weight MNIST comparison; see the [milestone catalog](milestones/README.md). These adapters do not include the weights or reproduce complete study evaluations. The public core tests do not certify omitted study runners.
No claims of trained-model accuracy, fresh-model generalization or causal
interpretation follow from these software checks. Historical numerical evidence
and its specific limitations are separately curated in [Selected results](RESULTS.md).

## Reproducing software checks

Install a compatible PyTorch build first if your platform needs a specific
CPU/GPU wheel, then install this checkout with `python -m pip install -e ".[test]"`.
Run `python -m pytest`, `python examples/train_observer.py`, `python examples/inspect_target.py`, and
`python examples/synthetic_allocation.py` from the repository root.

The current environment uses Python 3.15.2, PyTorch 2.15.0+cu134 and NumPy
2.5.5 on Windows x64. The CUDA dependency profile and pinned development tooling
are documented in [Dependencies](DEPENDENCIES.md). Historical software checks
used earlier Python/PyTorch environments; their original evidence is retained.

The milestone integration passed 95 core tests on Python 3.13/PyTorch 2.14 CPU. Independently reviewed focused tests passed as well. Local real-artifact smoke checks exercised all four stripped SYN53/MNIST exports and the SYN88 four-root fixture. These checks validate loader and inference paths.
