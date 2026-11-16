# Image-recognition targets: direct-weight MNIST comparison

## What Lens sees

The target networks classify MNIST digits. **Lens does not receive digit
images.** This milestone asks whether their parameters predict their mean
cross-entropy on a fixed evaluation suite.

The audited historical data path is:

```text
target checkpoint -> ordered parameter vector -> normalization -> Lens -> predicted mean loss
                  -> per-tensor statistics  -> normalization -> Lens -> predicted mean loss

images + labels   -> target classifier       -> measured mean loss (supervision/evaluation)
```

The raw-weight observer receives 19,658 parameter coordinates. The statistics
observer receives 192 statistics of those parameters. Images are used only by
the target classifier to generate supervision/evaluation labels and to check
data splits. Neither observer receives pixels during fitting or prediction.
Scalar measured losses supply the behavioral training supervision.

The output is **one mean-risk estimate per target state** over the fixed
evaluation suite.

## Selected models and provenance

The implementation uses the public `Lens`, `fit_lens`, `layer_statistics` and
`predict_lens` functions in `synaptic_lens.observer`. Its frozen source SHA-256
is `82ed9223874fa81ed743bf83470caa075eba4183b5be4f580bd19303b87983d1`.
The public observer retains that architecture and feature construction, with
optional device selection added for CUDA inference.

The following sizes and hashes identify the **historical original bundles**,
with separate identities for the downloadable inference exports. Export hashes
are recorded in the Hugging Face release manifest; exports retain the filenames.

| Checkpoint | Parameters | Selected epoch | Bytes |
| --- | ---: | ---: | ---: |
| `lens-full_raw_weights.pt` | 1,260,289 | 1 | 5,209,509 |
| `lens-tensor_statistics.pt` | 14,465 | 47 | 70,645 |

SHA-256 identifiers:

- Raw weights: `6011ca7cf3df54282eecef05aebc057e17fcc4a846ba61683e988c07b02a1111`
- Tensor statistics: `797a2834e33ed27835cb65054d1e4d6b44cf48c09ea3b125f96cf4d9b6c843c4`

**The two inference-only Lens exports and 300 target checkpoints are now [published on Hugging Face](../ARTIFACTS.md).** The original training-resume checkpoints are not distributed.
Both hashes match the saved result inventory. Both bundles passed restricted
`weights_only=True` loading, strict state-dictionary loading into the public
model class and CPU inference on a hash-verified real target checkpoint.
For root 1007 at epoch 1, the raw observer predicted 0.958178 nats and the
statistics observer 1.475826 nats; both matched the archived predictions within
1e-6. This validates the loader and inference path on one target.

Each observer was trained for 200 epochs on 30 target states. Ten validation
states selected the saved model; ten development states from two target roots
were scored afterward. Development labels were not used for model selection.
States from one root are correlated, so ten states do not represent ten
independent target runs. The official test set was never opened for this study.

## Saved development comparison

These are historical recorded metrics. Errors are in
cross-entropy nats; lower is better.

| Method | MAE | RMSE |
| --- | ---: | ---: |
| Constant training mean | 0.83429 | 0.93545 |
| Raw-weight Lens | 0.88615 | 0.97737 |
| Tensor-statistics Lens | 0.17140 | 0.21724 |
| Privileged training-age diagnostic | 0.04324 | 0.05613 |

The raw-weight observer had higher error than the constant baseline. Parameter
statistics reduced error substantially; training age provided the strongest
diagnostic. Training age uses privileged metadata beyond the observers' inputs.
These saved development results motivate further root-disjoint confirmation.

## Local inference and reproduction limits

The [image Lens walkthrough](../IMAGE_EXAMPLE.md) downloads and runs the featured
tensor-statistics reader on CUDA. For the lower-level inference interface,
after downloading the matching inference-only Lens export, target checkpoint
and coordinate schema, use:

```sh
python examples/predict_checkpoint.py --milestone mnist-raw --inference-export --checkpoint /path/to/lens-full_raw_weights.pt --target /path/to/target.pt --target-sha256 TARGET_SHA256 --coordinate-schema /path/to/coordinates.json
```

Replace each path and hash placeholder with the verified artifact value. For
the statistics observer, select `mnist-statistics` and its checkpoint. The
command requires a real target checkpoint and coordinate schema and runs local
inference. Assess prediction accuracy against the target's measured outcome.

Inference requires the exact coordinate ordering, tensor shapes and feature
construction used by training, together with the bundle's saved feature and
label normalization. Preserve the complete input schema to reconstruct
that contract. A representative target checkpoint and matching schema should
accompany a prediction example.

Recreating the comparison additionally requires the target population,
root/epoch splits, evaluation-query identifiers, label-generation procedure
and matching evaluator. The original runner had platform-specific operational
dependencies; porting that runner remains separate work.

The linked model cards distinguish the weights-only observer's license
from target-model, generator and dataset terms. No image data or external
generator code is relicensed by this page. This milestone deliberately excludes
the separate image-fed Lens experiments.

The coordinate-schema file is pinned by the loader to SHA-256
`36e6ab0e0a61bf01246a6c303fc7d0a96ef8cc916fec4ce297412908a042a1c4`.
Legacy target metadata uses PyTorch's string-like `TorchVersion` class; the loader
allows only that reviewed type within a scoped restricted-load context. It never
falls back to unrestricted pickle. Only use trusted artifacts with independently
verified hashes. Hash checking alone does not establish a file's trustworthiness.
