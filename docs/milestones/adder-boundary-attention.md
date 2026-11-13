# Arithmetic targets: boundary-attention comparison

## Research question

Does directing added attention branches toward a target network's input and
output regions improve a reader's behavioral predictions compared with giving
the same extra capacity access to all valid tokens?

This milestone, SYN53, compares two continuations of the same trained reader:

- **Capacity control:** the added branches can attend to all valid tokens.
- **Regional input/output:** the added branches are restricted to the
  corresponding boundary regions.

Both have 732,293 parameters and continue for 12,800 updates with matching data
order, labels and exposure. Their initial CPU FP32 outputs matched exactly.
The final checkpoint counter is 115,200 total updates, not 115,200 newly added
updates. Encoded states already carry global context, so regional attention is
an access bias, not isolation of raw boundary information.

## Interface

The model consumes records of supported affine/ReLU target networks and
explicit query metadata. It is not an image-fed observer. The public classes
are `DedicatedBoundaryReaderConfig` and `TransformerLens` in
`synaptic_lens.transformer_lens`; target-record construction is in
`synaptic_lens.target_records`.

This is a query-level reader. It should not be confused with the
[MNIST scalar-risk observer](mnist-direct-weights.md), which has a different
architecture, feature schema and prediction target.

## Selected checkpoint identities

The historical endpoint filename is `pass-09-complete.pt` for each arm.
Two inference-only development-endpoint exports are now [published on Hugging Face](../ARTIFACTS.md).
The table below identifies **historical original bundles**, not downloadable
exports: export bytes and hashes differ and are recorded in the release manifest.

| Arm | Bytes | SHA-256 |
| --- | ---: | --- |
| `all_valid_capacity_control` | 8,911,871 | `9dd0baffc515d481fafb62a33a1189e1526c21f1648515755da7caa26a357bd9` |
| `regional_input_output` | 8,911,871 | `119c4139203a19d2a9a134ebc568e8b2b5f534c34450b9cb0737a0b18713eb85` |

Both files were read on CPU with PyTorch's restricted `weights_only=True`
loader. Their configurations constructed the public model classes and their
model state dictionaries loaded with `strict=True`. Both reported 115,200
completed updates and 732,293 model parameters. This establishes compatibility
of those model states, not full training-resume or evaluation reproducibility.

## What the experiment found

The saved development evaluation contains 6,400 target states from 16 families.

| Metric | Capacity control | Regional |
| --- | ---: | ---: |
| Within-animal AUROC | 0.60477 | 0.60708 |
| Pass Brier score | 0.03210 | 0.02733 |
| Correct-answer-probability MAE | 0.05107 | 0.03378 |
| Paired-clone AUROC | 0.51869 | 0.50350 |

The probability estimates improved, but the within-animal AUROC difference
interval includes zero. Paired-clone discrimination became worse. This supports
better probability estimation in this comparison, **not a reliable gain in
query localization**. There was one matched training fork, not independent-seed
replication or new holdout access. See the
[full comparison and uncertainty](../RESULTS.md#2-better-probabilities-are-not-necessarily-better-localization)
and [machine-readable results](../results/boundary-attention.json).

## Reproduction boundary

Download the matching inference-only exports, then run from the repository root:

```sh
python examples/predict_checkpoint.py --milestone syn53-regional --inference-export --checkpoint /path/to/syn53/regional-input-output.pt
python examples/predict_checkpoint.py --milestone syn53-control --inference-export --checkpoint /path/to/syn53/capacity-control.pt
```

The default input is a deterministic, randomly initialized synthetic adder.
These commands run the saved reader on a generated fixture; they do not score
the historical development population or measure the reader's accuracy. There
is no checkpoint download and no training. Use the checkpoint belonging to the
selected arm, not another file with the same basename.

The selected checkpoint alone is insufficient to recreate the table. A complete
evaluation bundle must bind the target records, query definitions, population
splits, labels, metric definitions and evaluation script to the historical
protocol. Those complete study inputs are not distributed here. Generated toy
records can exercise the interface but are not substitutes for that evaluation.

Future inference packaging should load only the model state and configuration
needed for prediction, rather than requiring optimizer state, sampler state or
the original operational training environment. Preserve the original artifact
hashes as provenance if a separate inference-only format is exported.
