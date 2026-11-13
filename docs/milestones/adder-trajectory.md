# Arithmetic targets: lifetime trajectory reader (SYN88)

This later adder milestone follows the [boundary-attention comparison](adder-boundary-attention.md), but uses a different architecture and input contract. It reads a 20-input, 32-hidden-unit ReLU, 19-output target network across training ages, and estimates whether that target answers each of 100 ordered digit-pair addition queries correctly.

## What the reader receives

The inputs are canonicalized incoming weights and hidden biases, outgoing edges and output biases, the arithmetic query, and **query-aligned hidden ReLU activations**. This is not a weights-only observer. Target output logits are not inputs. The output is the expected-answer score minus the strongest rival score; its sigmoid is the fitted correctness probability.

Target coefficients are flattened in this order: hidden weight (640), hidden bias (32), output weight (608), output bias (19). Preprocessing uses float64 max-absolute incoming-weight/bias canonicalization, saved TRAIN-only normalization, then float32 inference. A zero incoming neuron is handled explicitly.

## Selected inference exports

These are numeric-only NPZ exports, not training-resume checkpoints. They omit optimizer and random-generator state and load with `allow_pickle=False`. The three checkpoints and complete historical target population are publicly hosted on Hugging Face; see [published artifacts](../ARTIFACTS.md).

| Training seed | Selected update | Export bytes | Export SHA-256 |
| --- | ---: | ---: | --- |
| 7 | 19,000 | 26,282 | `73786c9e025070b6884240430222b74dea7c5ee1422ee8d7c409462c604aee26` |
| 17 | 17,000 | 26,282 | `4cee5c66bfb8189a2b4b58821c251035da8cc424dd73a0c67247cae935f6d3b3` |
| 27 | 17,000 | 26,282 | `20cf06bf7599c457b56e42c381c7876ba2119985a98bb0a36a8633e3cffd614f` |

The paired smoke fixture contains four previously evaluated TEST roots, both rules, eight ages and 100 queries (6,400 predictions per model). Its SHA-256 is `24f0f4fd207652076197113e8e2156c2f0da67f267f6dffb95b4e41e397c1e67`. These are not fresh holdout roots. The saved export verification records CPU agreement with the original CPU implementation and bitwise reproduction of the corresponding archived GPU predictions with float32 matmul precision `highest`. CPU predictions differ slightly from archived GPU predictions (maximum absolute logit differences 0.003174, 0.001465 and 0.001953 respectively). Hardware paths must not be silently equated.

## Local inference

`examples/predict_trajectory.py` accepts a local numeric checkpoint, a local NPZ containing `weights`, and independently obtained SHA-256 values for both. Run `python examples/predict_trajectory.py --help` for the interface. It performs CPU inference and prints logits in the target array's original leading dimensions followed by 100 queries ordered `(0,0), (0,1), ..., (9,9)`. It neither downloads models nor reports these fixture predictions as evaluation metrics.

The library APIs are `load_model` and `predict` in `synaptic_lens.trajectory_checkpoint`. If moving the loaded model to CUDA, set `torch.set_float32_matmul_precision('highest')`; lower-precision TF32 changes this checkpoint's predictions. The predictor rejects CUDA inference with a different precision setting.

This small inference interface is not a portable training runner or a reproduction of the complete study. Full evaluation additionally requires all target roots, split and query manifests, labels, and the matching evaluator. Model implementation, selected checkpoint, smoke compatibility, and empirical generalization are separate claims. Artifact distribution requires explicit terms and provenance in addition to this repository's software license.

## Target generation recipe

See [SYN88 generation and evaluation](syn88-generation.md) for the paired-rule
training recipe, numeric target format and root-disjoint population protocol.
This target generator is separate from training the Lens reader itself.
