# Published model artifacts

Code lives in this GitHub repository; checkpoints, target populations and their
integrity/provenance manifests live in the following public Hugging Face repos.
Use a pinned revision and verify the manifest SHA-256 values for repeatable downloads. These releases were verified against the staged payload hashes.

| Repository | Published contents | Verified revision |
| --- | --- | --- |
| [Adder Lens](https://huggingface.co/danielrjackson/synaptic-lens-adder) | Three SYN88 inference-only Lens checkpoints under `syn88/`, plus two SYN53 development-endpoint exports under `syn53/` | `78680e8ccac56ebbfe58eb8d0a1d6e4924f0bc78` |
| [Adder targets](https://huggingface.co/danielrjackson/synaptic-lens-targets-adder) | 2,304 historical SYN88 root bundles under `syn88/population/`, plus a separate smoke fixture | `bf84ab005319635d866dbaccf54ae29ea9e15197` |
| [MNIST Lens](https://huggingface.co/danielrjackson/synaptic-lens-mnist) | Two inference-only Lens checkpoints: raw-weight and parameter-statistics observers | `1acd95aaffda8fc148aa362a0573f72d65dddea3` |
| [MNIST targets](https://huggingface.co/danielrjackson/synaptic-lens-targets-mnist) | 300 target checkpoints across 12 trajectories, coordinate schema and trajectory metadata | `f422682c970bdf64c8b30d24c2cf7bac8ce3ef6c` |

## Newer releases

The table above pins the original study exports. Later models have their own
release identities and instructions:

- [Missing-neuron Adder Lens](NEURON_COMPLETION.md): a position-free Transformer
  that reconstructs a hidden neuron. The [completion manifest](neuron-completion-manifest.json)
  pins the reader and example target at Adder revision
  `9d8bf578fd456d7475afa00e627d13178c24faab`. Use
  `scripts/download_completion.py` through the walkthrough.
- [Custom-image targets](https://huggingface.co/danielrjackson/synaptic-lens-targets-custom-images/tree/81f312e0f32e00fdbf02e027c711ef21044d82ee):
  ten trained image classifiers with a CUDA-default prediction example. The model
  card describes the custom tasks, model files and commands. These are the target
  classifiers that Lens studies. The MNIST Lens readers remain in the table above.

## Using the files

Read each model card and manifest first. Download the selected files locally and
use their published hashes with the matching inference example:

- [SYN53 boundary-attention interface](milestones/adder-boundary-attention.md),
  with `predict_checkpoint.py --inference-export`. Its generated-fixture example
  checks loader and inference compatibility.
- [SYN88 trajectory interface](milestones/adder-trajectory.md), with
  [`predict_trajectory.py`](../examples/predict_trajectory.py). The adder-target
  repo also supplies a numeric-only NumPy quickstart that verifies native labels.
- [MNIST direct-weight interface](milestones/mnist-direct-weights.md), with
  [`predict_checkpoint.py`](../examples/predict_checkpoint.py) and
  `--inference-export` for the stripped Lens exports. Use the target manifest's
  hash and matching coordinate schema.

Inference-only exports omit optimizer/RNG state and cannot resume training.
The [SYN88 target generator](milestones/syn88-generation.md) implements the target
training recipe. Lens optimization and evaluation use separate workflows.

## Scope and limitations

The SYN88 population has 1,536 training, 256 validation and 512 historical test
roots. Each root contains two rules at eight ages: 36,864 recorded states, with
34,560 distinct float32 tensor hashes because paired birth states are shared.
Keep every state from a root in its assigned partition. The historical test results have already been examined. The separate four-root smoke fixture supports a quick compatibility check;
full-population evaluation uses all assigned roots.

The SYN53 capacity-control and regional-input-output development endpoints are
published as inference-only exports. The sealed early SYN53 holdout and complete
SYN53 evaluation bundle are excluded; these two checkpoints do not enable a full
reproduction of that study.

MNIST target checkpoints contain trained classifier parameters. No raw MNIST image dataset,
external pretrained weights or image-fed Lens experiments are included. Consult
the cards for data-source and preprocessing provenance. The two MNIST Lens
checkpoints read weights or weight statistics, while SYN88 also uses queries and
query-aligned hidden activations. Each model requires its documented input contract.
Load, smoke and label checks validate the artifact interfaces. Research metric
reproduction additionally requires the study population and evaluation protocol.

## Terms

Model files expressly designated by their cards/manifests use the unmodified
AI Pubs research-use **RAIL-M 0.1**; accompanying software uses **RAIL-S 0.1**.
These permit academic/research use subject to their full restrictions.
Third-party datasets and other materials retain their respective terms.
See each repository's license files and notices for the actual terms.
