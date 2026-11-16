# Training and development workflows

## Train an Adder Lens and compare it with the released model

From the repository root:

```sh
uv sync --locked --extra hub
uv run --locked --extra hub python examples/train_compare_adder.py
```

This downloads the featured Lens and twelve target-root bundles, verifies their
hashes, and trains the same output-centered Lens architecture on CUDA. Eight
roots are used for fitting, two for validation selection and two for comparison.
These are fixed subsets of the original training, validation and test partitions.
Both target training rules and every age stay with their root.

The default is 100 updates. The example writes three files to
`runs/adder-comparison`:

- `your-lens.npz`: your trained reader, including its normalization.
- `comparison.json`: both readers' accuracy and Brier score on exactly the same
  3,200 comparison queries, training history, root IDs and artifact hashes.
- `predictions.csv`: one row per digit pair, rule and age, with the target's
  correctness and each reader's predicted probability. Open it in a spreadsheet
  to inspect where they agree or differ.

**Reading the comparison:** higher correctness-prediction accuracy is better;
lower Brier score means the probabilities are closer to the actual outcomes.
The featured reader trained on 1,536 roots. Your eight-root run is a small
training exercise using the same architecture. Its measured results appear
beside the featured model's results for these two comparison roots.

To train longer while keeping the first run:

```sh
uv run --locked --extra hub python examples/train_compare_adder.py --updates 1000 --output runs/adder-1000
```

Use validation history to choose your training budget. Once you inspect the
comparison results, those roots serve as a development comparison; use fresh,
unexamined roots for a later final evaluation.

Add `--local-files-only` to repeat a run using cached downloads. Add
`--cache-dir PATH` to choose the cache, or `--device cpu` for the optional CPU
path. Each run requires a new output directory to preserve earlier models.

## Lifetime adder reader API

Run the bounded CPU smoke after installing this repository:

```console
python examples/train_trajectory.py --updates 1
```

It creates random synthetic target states, keeps paired rules and all ages with
one root, fits one optimizer update, selects using a separate development root,
and round-trips a numeric-only inference export. This is a software smoke check
with synthetic inputs.

`trajectory_data.TrajectoryPopulation` accepts manifest-verified float32 weights
`[roots, 2, 8, 1299]` and binary addition-correct labels `[roots, 2, 8, 100]`.
The two rules use the order in `syn88_targets.RULES`; ages use `AGES`. Root IDs
must be explicit and disjoint. Paired births must match. Independent native
forward calculations check the saved labels and canonicalization. TRAIN alone
fits normalization in float64; features become float32 afterward.

`trajectory_training.train_trajectory` exposes a fixed-budget CUDA/CPU runner (select `device="cuda"`) for
trajectory and checkpoint25 arms. The latter always trains on age 25; evaluation
still covers all eight ages. `HISTORICAL_RECIPE` records three seeds, batch size
24 roots, AdamW learning rate 0.001/weight decay 0.01, ten-query chunks, validation
every 1,000 updates and a maximum of 20,000 updates. The common historical stopping
decision, interleaving across six cells and recovery ledger are not replayed.
Running the full recipe requires a substantially larger compute budget than
the default smoke example.
Validation selects the minimum full-trajectory macro objective; no test data is
accepted by the fitting API. Saved historical test results have already been examined. Use the released population to compare historical
models with their recorded artifact identities.

The earlier SYN53 development endpoints remain available through the existing
boundary-reader loader and milestone documentation. The complete continuation and its sealed holdout remain in the research archive.

## Network-only MNIST mean-risk observers

`mnist_observer.ObserverPartition` contains ordered target weights, one scalar
measured mean risk per state and an explicit root ID per state. It does not
accept pixels. `fit_mean_risk` supports raw 19,658-weight inputs and 192 features
(six statistics for each of 32 tensors). Use the verified coordinate schema via
`checkpoints.flatten_target`; never infer order from a checkpoint dictionary.

Keep every state from a target root in the same observer partition. The
published target corpus FIT/SELECT/CONFIRM split is distinct from the observer
side-study split. This API requires the caller to supply that choice explicitly;
it does not silently turn target partitions into historical observer partitions.
Risk labels must refer to a documented, fixed evaluation suite. Validation alone
selects the epoch; target-risk normalization and feature normalization are fitted
on training roots. The 200-epoch setting matches the existing observer mechanism,
with reproduction depending on the archived study's data and protocol.

The recovered observer recipe uses roots 1007–1057 (six roots stepping by ten)
for fitting, 1067/1077 for validation, and 1087/1097 for development. Each uses
epochs 0, 1, 5, 20 and 100: 30/10/10 states. The target corpus instead assigns
1067/1077 to FIT and 1087/1097 to SELECT. These are deliberately different splits.
Risk labels are means over 240 fitting query IDs and 120 separate validation or
development query IDs, not automatically over the entire 10,000-image split.
`HISTORICAL_RECIPE` records this grid and seed seven/200 observer epochs; the
original query-ID manifests and measured labels are not bundled, so complete
historical metric reproduction remains unverified. `evaluate_mean_risk` requires
development roots disjoint from both fitting and selection roots; its error
summary counts states and independent roots separately.

For released-model CPU inference, use `examples/predict_checkpoint.py --help`
and the MNIST milestone instructions. A hash-pinned statistics observer and root
1007 at epoch one predicted 1.47582555 nats in the current CPU compatibility check,
matching the rounded archived prediction. Only 155,994 checkpoint bytes were
read from the public release; no images were used. This is one loader/inference
check. The historical raw-weight Lens error exceeded the constant baseline;
the milestone page preserves the complete comparison.

## Artifact identities and required source data

[releases.json](releases.json) pins the four public repositories, manifest hashes
and selected file hashes. This compact index points to the pinned manifests containing all target
entries. Optimizer/RNG state was stripped from
inference exports. The historical MNIST query-ID manifests and measured labels are still needed
for a download–train–compare image walkthrough. The released image example
currently covers downloading and inference; the Adder example above covers
the complete training and comparison workflow.
