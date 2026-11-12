# Public scope and validation

This distribution selects reusable components, not an entire research workspace.
The current integration includes exact affine/ReLU records, paired readers,
value-blind scalar reconstruction, reader configuration, detached allocation
policies, antithetic Adam updates, descriptive ranking/AUC, and CSV output.

The source modules retain their `synaptic_lens.<module>` import names.
`ReaderConfig` and `DedicatedBoundaryReaderConfig` retain the serialization
identity `synaptic_lens.transformer_lens`, even though their definitions live in
`reader_config.py`. This does not promise compatibility with every experimental
checkpoint or make untrusted pickle files safe to load.

Tests cover numeric records, token/query contracts, boundary attention,
masking, configuration serialization, policy detachment, allocation gradients,
optimizer-state updates, ranking ties and output bytes. The optimizer fixture is
an ordinary small neural network, not a study-specific controller. The two CPU
examples need no stored model or dataset. Allocation weights are a mechanism
demonstration, not evidence of improved training performance.

## Deliberate omissions

Study orchestration, private working notes, agent coordination, operational
supervision and storage management are outside this small distribution. So are
historical study checkpoint adapters, pretrained weights, dataset caches and
training outputs. The public core tests do not certify omitted study runners.
No claims of trained-model accuracy, fresh-model generalization or causal
interpretation follow from these software checks.

## Reproducing software checks

Install a compatible PyTorch build first if your platform needs a specific
CPU/GPU wheel, then install this checkout with `python -m pip install -e ".[test]"`.
Run `python -m pytest`, `python examples/inspect_target.py`, and
`python examples/synthetic_allocation.py` from the repository root.

The current validation used Python 3.13.3 and PyTorch 2.14.0, on CPU under
Windows. A CUDA-enabled wheel was installed, but these checks did not use a GPU.
Other platforms and older PyTorch versions have not been independently checked
for this curated distribution. The dependency floor conservatively reflects the
validated version, not a claim that older versions cannot work.
