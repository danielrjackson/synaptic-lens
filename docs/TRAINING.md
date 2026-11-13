# Training and development workflows

## Lifetime adder reader

Run the bounded CPU smoke after installing this repository:

```console
python examples/train_trajectory.py --updates 1
```

It creates random synthetic target states, keeps paired rules and all ages with
one root, fits one optimizer update, selects using a separate development root,
and round-trips a numeric-only inference export. It does not reproduce research
accuracy or resume a released inference checkpoint.

`trajectory_data.TrajectoryPopulation` accepts manifest-verified float32 weights
`[roots, 2, 8, 1299]` and binary addition-correct labels `[roots, 2, 8, 100]`.
The two rules use the order in `syn88_targets.RULES`; ages use `AGES`. Root IDs
must be explicit and disjoint. Paired births must match. Independent native
forward calculations check the saved labels and canonicalization. TRAIN alone
fits normalization in float64; features become float32 afterward.

`trajectory_training.train_trajectory` exposes a fixed-budget CPU runner for
trajectory and checkpoint25 arms. The latter always trains on age 25; evaluation
still covers all eight ages. `HISTORICAL_RECIPE` records three seeds, batch size
24 roots, AdamW learning rate 0.001/weight decay 0.01, ten-query chunks, validation
every 1,000 updates and a maximum of 20,000 updates. The common historical stopping
decision, interleaving across six cells and recovery ledger are not replayed.
Running the full recipe is substantial work, not the default smoke example.
Validation selects the minimum full-trajectory macro objective; no test data is
accepted by the fitting API. Saved historical test results are already examined,
not a new blind benchmark. Use the released population to compare historical
models, rather than pretending newly generated tensors are byte-identical.

The earlier SYN53 development endpoints remain available through the existing
boundary-reader loader and milestone documentation. A portable reproduction of
that complete continuation is not claimed by this trajectory runner. Its sealed
holdout is excluded.

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
not a claim that any caller-supplied dataset reproduces the archived study.

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
check, not an evaluation rerun. Raw-weight Lens did not beat the constant
baseline in the historical comparison; do not omit that negative result.

## Artifact identities and remaining gates

[releases.json](releases.json) pins the four public repositories, manifest hashes
and selected file hashes. It is a compact index, not a duplicate of every target
entry; the pinned manifest gives all files. Optimizer/RNG state was stripped from
inference exports. Full historical side-study query/label reconstruction and
SYN53 continuation remain separate provenance gates. Broader vision semantic
repairs and later study claims are not promoted by these APIs.
