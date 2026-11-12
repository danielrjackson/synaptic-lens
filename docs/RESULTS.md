# Selected research results

Synaptic Lens studies whether a separate network can predict another network's
behavior from its parameters. These selected studies show the progression from
competence prediction to query-specific readers and controlled training
interventions. They include failures and contrary diagnostics, not just favorable
aggregate scores. All target networks in these examples are small arithmetic
networks; none of this establishes arbitrary-model interpretation.

The tables below summarize **historical saved evaluations**, not a fresh rerun.
The accompanying JSON files contain the reported numeric summaries.
[Provenance](results/provenance.json) identifies the source reports by SHA-256;
the original complete checkpoints and execution records are not distributed here.
These hashes identify records, not independently reproducible experiments on
their own. The runnable examples exercise the code without pretending to
reproduce these numerical findings.

## 1. A structural observer beats shortcuts in a controlled population

An initial flattened-weight observer lost badly to simple parameter statistics:
MAE 0.3615 versus 0.0229 for statistics and 0.0183 for training step. Rather than
hide that result, the next comparison introduced output-label permutations that
change behavior while preserving tensor statistics.

The structural observer encodes each hidden neuron's incoming weights, bias and
outgoing weights with a shared network, pools those encodings, and combines them
with output biases. Hidden-neuron renumbering does not change its prediction.
It sees weights, not arithmetic examples, at inference. The target architecture
is fixed at 20-32-19 with ReLU and 1,299 parameters.

A saved observer was evaluated without retraining on **40 fresh target
initializations**. All 100 ordered single-digit addition cases were also target
training examples. Output rewiring broadens this controlled population; it does
not turn the task into arithmetic input generalization.

| Evaluation condition | Step baseline | Tensor statistics MLP | Raw weights | Structured observer |
| --- | ---: | ---: | ---: | ---: |
| Familiar output shifts | 3.1074 | 3.1121 | 2.2947 | 0.3297 |
| Reserved output shifts | 2.4313 | 2.4340 | 2.5252 | 0.4909 |
| Arbitrary output permutations | 1.3963 | 1.2585 | 2.3448 | 0.9010 |

Values are MAE in predicted cross-entropy nats; lower is better. Compare methods
within a row, because loss distributions differ across rows. The step baseline
is privileged metadata, not a weights-only observer.

[Full aggregate metrics and run-bootstrap intervals](results/structured-observer-metrics.json)
retain RMSE, R-squared and comparisons against the step baseline. Checkpoints
from one target run are correlated; uncertainty resamples whole runs. There was
one fitted structured observer, not independent observer-seed replication.
Architecture, capacity, normalization and optimization also differed. The result
supports useful prediction in this narrow population, not causal localization,
identified internal algorithms, or broad neural-network capability recognition.

## 2. Better probabilities are not necessarily better localization

A later attention reader compared two added input/output branches. The
**capacity control** could attend to all valid tokens; the **regional arm** was
restricted to input/output-associated regions. Both continued the same checkpoint
for 12,800 updates, with matching parameters, data order, labels and exposure.
Their initial CPU FP32 outputs matched exactly. Both had 732,293 parameters.
Post-encoder states already contain global context, so this is a regional-access
bias, not isolated raw boundary information.

The endpoint used 6,400 DEV states across 16 families. Intervals are pointwise
95% paired whole-family bootstrap intervals from one matched training fork.

| Metric | Capacity control | Regional | Difference interval |
| --- | ---: | ---: | --- |
| Within-animal AUROC | 0.60477 | 0.60708 | [-0.00242, +0.00702] |
| Balanced accuracy | 96.09% | 96.43% | [-0.07, +0.84] percentage points |
| Pass Brier | 0.03210 | 0.02733 | [-0.00813, -0.00184] |
| Equal-family calibration error | 0.03080 | 0.01581 | [-0.02051, -0.00955] |
| Correct-answer-probability MAE | 0.05107 | 0.03378 | [-0.02334, -0.01146] |

The probability estimates improved, but query discrimination did not reliably
improve: AUROC's interval includes zero. On paired-clone views AUROC **fell** from
0.51869 to 0.50350, with difference interval [-0.02391, -0.00741]. The control's
matched-false-positive-rate swap envelope equaled or exceeded treatment at every
attainable budget. Rare early/partial correct answers remained missed.

AUROC is defined only for mixed correct/wrong states: 2,106 of the 6,400 original
DEV states. Homogeneous states are not filled with a pooled surrogate. These
results support better probability estimation in this run, **not reliable
query-localization gains**. There was no new holdout access or independent-seed
replication. [Machine-readable table](results/boundary-attention.json).

## 3. Learned training allocation: mixed gains, substantial cost

The allocation pilot compared uniform, square-root-current-loss, and learned
weighting. At the fixed 25,600 accepted-update endpoint, learned allocation had
the best AUC and Brier, while square-root-loss had higher threshold accuracy.
The diagnostic panel contained 142 states from 16 families, with 100 queries per
state. DEV did not drive controller training or continuation; HOLD was not read.

| Policy | Accepted updates | Within-animal AUC | Brier | Accuracy at 0.5 | Mean-state probability MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform | 25,600 | 0.4655 | 0.2470 | 72.11% | 0.2601 |
| Square-root loss | 25,600 | 0.5129 | 0.1773 | 88.99% | 0.2488 |
| Learned | 25,600 | 0.5764 | 0.1485 | 83.42% | 0.1372 |
| Longer uniform | 128,000 | 0.6799 | 0.0649 | 92.46% | 0.0558 |

Learned allocation required **91,648 total learner updates**, including 66,048
discarded counterfactual probes, for its 25,600 accepted updates. Matching
accepted updates does not match compute. Longer uniform is an approximate-cost
comparison, not an exact matched-compute control. All arms missed all **537 rare
positive cases** in the three reported subgroups. This is one-seed descriptive
evidence, not a general learned-policy advantage.

Subsequent selected-anchor fitting also illustrates the transfer problem: in a
six-family development-exposed check, the final joint adaptation retained
231/232 trained anchors, but broader mixed-state AUC fell from 0.597 to 0.535
and mean BCE rose from 0.332 to 1.326 versus the original checkpoint. Good anchor
fit is not broader-family discrimination. These six correlated families are not
a pristine final holdout. [Machine-readable summaries](results/allocation.json).

## What the portfolio demonstrates

The contribution here is a research process as well as code: invariant
representations, family-aware evaluation, matched controls, preselected
endpoints, explicit accounting of training/probe cost, and diagnostics that
expose misleading aggregate success. Generalization, reliable failure-pattern
recognition and causal interpretation remain open questions. No novelty or
state-of-the-art claim is made.
