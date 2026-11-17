# Train your own missing-neuron Lens

This walkthrough trains the same position-free Transformer architecture as the
released completion reader: 3,185,448 parameters, width 256, four blocks and eight
attention heads. It learns to predict a hidden neuron's 20 incoming weights, bias
and 19 outgoing weights from the other neurons and output biases.

Install the repository's CUDA and Hub dependencies using the main installation
instructions. From the checkout:

```sh
uv run --locked --extra hub python examples/train_neuron_completion.py --output runs/my-completion
```

The script downloads 64 fitting and 16 development roots from the pinned public
Adder target release, verifies their hashes, and uses their age-100
weights. Both arithmetic rules remain together within each root. These are subsets
of the audited experiment's fitting and development partitions; neither consumed
final cohort is used. Partitioning precedes random permutation augmentation.

The default is 2,000 updates on CUDA. Each update samples physical missing neurons
independently of their weights and applies fresh random permutations to whole
neuron bundles. There is no weight-based ordering. Normalization statistics come
only from fitting roots. Selection uses development error averaged over all 32
missing-neuron positions. The wrapper retains the true missing-neuron scale outside
Lens for reconstruction into the original target.

For a short, explicit CPU plumbing check using public targets:

```sh
uv run --locked --extra hub python examples/train_neuron_completion.py --output runs/cpu-smoke --device cpu --updates 2 --fit-roots 1 --development-roots 1
```

This still uses the full architecture. The automated unit test uses a tiny synthetic
fixture to test training and reload quickly. Neither is a quality benchmark.

## Inspect your result

The output directory contains:

- `your-reader.pt`: selected model and fit-only statistics, compatible with
  `OrganicCheckpointReader`.
- `training-report.json`: selected update, development curve, target revision and
  hashes, checkpoint SHA-256, and the first development target's original, zeroed
  and reconstructed answers. It includes original and predicted weights.
- `example-target.pt`: that development example in original coordinates.

The script reloads the saved reader with its computed hash before producing the
comparison. This comparison uses the first development target and neuron 0, with
no score ranking. It is a hands-on development example, not an independent test.

To use the saved reader programmatically:

```python
import json
from pathlib import Path
from synaptic_lens.checkpoints import load_verified
from synaptic_lens.neuron_completion import OrganicCheckpointReader, prepare_organic_completion

run = Path("runs/my-completion")
report = json.loads((run / "training-report.json").read_text())
reader = OrganicCheckpointReader(run / "your-reader.pt", report["checkpoint_sha256"])
# Supply your own float32 neurons [32,40] and output_bias [19].
# view = prepare_organic_completion(neurons, output_bias, missing=7)
# reconstructed = view.predict(reader)
```

Increase `--fit-roots` up to 768, `--development-roots` up to 128, or `--updates`
(up to 20,000) for a longer run. Choose a new output directory each time. This entry
point provides fixed-budget training with development-set selection;
it does not recreate the three-seed audited benchmark or claim its published score.
There is no automatic resume or GPU sensor monitoring in this standalone script.
