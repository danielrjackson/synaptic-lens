# Complete a missing adder neuron

This package reconstructs a hidden neuron's 20 incoming weights,
one bias and 19 outgoing weights. Neurons stay in their original order. Each
visible neuron is normalized independently; the missing neuron's true scale is
retained outside Lens and used when inserting the prediction back into the target.
The reader has no positional embeddings and was trained with random permutations.

## Download and run

Use the maintained CUDA environment described in the README. No factory package,
training corpus or research repository is needed for inference.

```sh
uv sync --locked --extra hub
uv run --locked --extra hub python scripts/download_completion.py
uv run --locked python examples/complete_neuron.py --artifacts artifacts/neuron-completion
```

For downloaded artifacts in another location:

```sh
python examples/complete_neuron.py --artifacts PATH_TO_ARTIFACTS --neuron 0
python examples/complete_neuron.py --artifacts PATH_TO_ARTIFACTS --neuron 7 --device cpu
```

CUDA is the default. CPU is an explicit option; floating-point results can differ
slightly by backend. The example checks the pinned SHA-256 before restricted
checkpoint loading. It prints the actual and predicted weights and all 100 answers
before masking, after masking and after reconstruction. Try any neuron from 0 to 31.
The selected target is the first saved evaluation target, root 896, addition rule.
No score ranking was used. Its full original weights are supplied so you can compare
predictions and retain the true reconstruction scale; the reader sees only the masked view.

## Model and results

The release uses the preselected seed 161120, selected at step 4400 by
development error. Its 3,185,448-parameter reader has four Transformer blocks,
width 256 and eight heads. The manifest identifies both the original selected
checkpoint and stripped inference export. Exporting removes training configuration
and source paths; model tensors and normalization statistics are unchanged.

The three-seed experiment averaged 99.2766% restored-answer accuracy across both
rules, all 32 hidden neurons and 128 evaluation roots, compared with 98.2528% for
zero replacement. This aggregate is not a score for this one checkpoint or example.
Learned controls had similar accuracy. The position-free readers changed zero answers
under the tested permutations (819,200 comparisons per seed).

This task predicts missing weights. The earlier 99.6167% Adder Lens result measures
a different task; these percentages are not an improvement comparison.

## Published artifacts

The [manifest](neuron-completion-manifest.json) pins the model and example target
at Hugging Face revision `9d8bf578fd456d7475afa00e627d13178c24faab` in
[Synaptic Lens Adder](https://huggingface.co/danielrjackson/synaptic-lens-adder/tree/9d8bf578fd456d7475afa00e627d13178c24faab/neuron-completion-v1).
The downloader uses the standard Hub cache with an exact revision, verifies each
file's size and SHA-256, and never falls back to `main`.

Models use the existing AI Pubs research-use RAIL-M license in LICENSE-MODEL.md;
code uses LICENSE.md. Research provenance is identified by commit and hashes.

## Training scope

This walkthrough runs pretrained completion inference. The existing
`train_compare_adder.py` trains the separate correctness-prediction reader.
[Train your own missing-neuron Lens](TRAIN_NEURON_COMPLETION.md) downloads
root-isolated public targets, fits the same position-free architecture and reloads
your selected checkpoint for a hands-on reconstruction comparison. It uses fitting
and development roots; neither consumed evaluation cohort is used.
