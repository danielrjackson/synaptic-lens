# Research milestones

Synaptic Lens trains readers to predict a target network's behavior from its
parameters and, in some experiments, its activations. The table lists each
reader's inputs and prediction task.

| Track | Selected milestone | Lens input | Prediction |
| --- | --- | --- | --- |
| [Arithmetic targets](adder-boundary-attention.md) | Matched boundary-attention comparison (SYN53) | Target network records and explicit query metadata | Query-level behavioral estimates |
| [Arithmetic trajectories](adder-trajectory.md) | Lifetime trajectory reader (SYN88) | Canonical target weights, queries and query-aligned hidden activations | Correctness logits across training ages |
| [Image-recognition targets](mnist-direct-weights.md) | Direct-weight MNIST comparison | Target parameters, or statistics of those parameters | Mean cross-entropy of a fixed target evaluation suite |

Each track has its own prediction problem, architecture and input contract. Image-fed Lens experiments are not part of
this release.

## Code, checkpoints and evidence

Each milestone documents:

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

The milestone pages report saved historical metrics. To reproduce an evaluation, use its matched target population, query/split manifests, labels and evaluator. Software checks cover checkpoint loading and inference compatibility.

## Distribution and rights

The [software license](../../LICENSE.md) and [attribution notice](../../NOTICE)
apply to the software distributed in this repository. The linked artifact releases carry their own terms, source provenance and
integrity manifests. MNIST images, external target-generation code and other
third-party materials retain their own licenses and notices.

For earlier arithmetic studies and the allocation comparison, see
[Selected research results](../RESULTS.md). The milestone catalog covers selected supported releases.

## Inference-only packaging

The checkpoint example also accepts `--inference-export`. This selects the hash-pinned inference export. The exported models preserve the model tensors and required configuration/normalization, record the original checkpoint hash, and omit optimizer, RNG, sampler or unnecessary training-history fields. They cannot resume training. Published SYN53 and MNIST inference exports are linked in the artifact catalog; use this flag with those exports.
