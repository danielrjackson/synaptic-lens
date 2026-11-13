# Research milestones

Synaptic Lens is a research program about reading a network's parameters to
predict its behavior. The two tracks below separate **what the target does**
from **what Lens receives**. A target may classify images without its observer
receiving any images.

| Track | Selected milestone | Lens input | Prediction |
| --- | --- | --- | --- |
| [Arithmetic targets](adder-boundary-attention.md) | Matched boundary-attention comparison (SYN53) | Target network records and explicit query metadata | Query-level behavioral estimates |
| [Arithmetic trajectories](adder-trajectory.md) | Lifetime trajectory reader (SYN88) | Canonical target weights, queries and query-aligned hidden activations | Correctness logits across training ages |
| [Image-recognition targets](mnist-direct-weights.md) | Direct-weight MNIST comparison | Target parameters, or statistics of those parameters | Mean cross-entropy of a fixed target evaluation suite |

These are different problems and architectures, not interchangeable versions
of one universal pretrained model. Image-fed Lens experiments are not part of
this release.

## Code, checkpoints and evidence

Three things should be distinguished:

1. **Model implementation:** the code defining an observer and its inputs.
2. **Trained checkpoint:** parameters fitted in a particular study, with its
   input schema and normalization.
3. **Research result:** an evaluation on a specified population and protocol.

The repository provides model implementations and runnable software examples.
The milestone pages identify selected historical checkpoints and their audited
compatibility. [Published artifacts](../ARTIFACTS.md) links the released SYN88
and MNIST checkpoints and target populations, and two SYN53 development-endpoint
exports. The sealed SYN53 holdout is not included. Inference commands use downloaded
local files and manifest hashes; they do not automatically download models.

The historical metrics are saved results, not a new reproduction. A successful
checkpoint load or synthetic fixture demonstrates software compatibility, not
the original study's accuracy. A useful complete reproduction additionally
needs the matched target population, query/split manifests, labels and evaluator.
Each page identifies what is and is not established.

## Distribution and rights

The [software license](../../LICENSE.md) and [attribution notice](../../NOTICE)
apply to the software distributed in this repository. They do not silently
license absent checkpoint files, target datasets or code from another project.
The linked artifact releases carry their own terms, source provenance and
integrity manifests. Third-party materials retain their applicable notices. No rights to redistribute MNIST
images or external target-generation code are granted by these milestone pages.

For earlier arithmetic studies and the allocation comparison, see
[Selected research results](../RESULTS.md). The milestone catalog is selective,
not a claim that every research experiment is a supported release.

## Inference-only packaging

The checkpoint example also accepts `--inference-export`. This explicitly selects a separate hash-pinned export identity rather than treating modified bytes as the historical checkpoint. The exported models preserve the model tensors and required configuration/normalization, record the original checkpoint hash, and omit optimizer, RNG, sampler or unnecessary training-history fields. They cannot resume training. Published SYN53 and MNIST inference exports are linked in the artifact catalog; use this flag with those exports. This option does not itself grant redistribution rights.
