"""Small CPU contracts for the public learned-observer workflow."""
from pathlib import Path
import runpy

import pytest
import torch
from torch.nn.utils import parameters_to_vector

from synaptic_lens.observer import fit_lens, layer_statistics, split_runs, standardize_fit
from synaptic_lens.structured import StructuredLens, unpack
from synaptic_lens.targets import Adder, TargetConfig, addition_suite, generate_population, parameter_layout, restore_adder


def test_run_partitions_are_disjoint_complete_and_repeatable():
    split = split_runs(20)
    assert [len(split[k]) for k in ("train", "validation", "test")] == [12, 4, 4]
    all_ids = torch.cat(list(split.values()))
    assert sorted(all_ids.tolist()) == list(range(20))
    assert all(torch.equal(v, split_runs(20)[k]) for k, v in split.items())
    with pytest.raises(ValueError):
        split_runs(9)


def test_training_only_normalization_and_validation_selection():
    torch.manual_seed(8)
    tx = torch.randn(8, 3)
    ty = torch.arange(8).float() / 8
    vx = torch.randn(3, 3) + 100
    vy = torch.tensor([0.3, 0.4, 0.5])
    bundle = fit_lens(tx, ty, vx, vy, epochs=3)
    mean, scale = standardize_fit(tx)
    assert torch.equal(bundle["mean"], mean)
    assert torch.equal(bundle["scale"], scale)
    best = min(bundle["history"], key=lambda row: row["validation_mse"])
    assert bundle["best_epoch"] == best["epoch"]


def test_structured_observer_is_invariant_to_hidden_renumbering():
    torch.manual_seed(4)
    target = Adder()
    vector = parameters_to_vector(target.parameters()).detach()[None]
    layout = parameter_layout(target)
    neurons, bias = unpack(vector, layout)
    model = StructuredLens().eval()
    permutation = torch.randperm(neurons.shape[-2])
    torch.testing.assert_close(model(neurons, bias), model(neurons[:, permutation], bias),
                               rtol=1e-6, atol=1e-6)
    assert layer_statistics(vector, layout).shape == (1, 24)


def test_generated_labels_match_reconstructed_target(tmp_path):
    config = TargetConfig(runs=10, steps=2, hidden=4)
    path = generate_population(config, tmp_path)
    data = torch.load(path, weights_only=True)
    _, inputs, labels = addition_suite()
    target = restore_adder(data["weights"][0, 2], config.hidden)
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(target(inputs), labels)
    torch.testing.assert_close(loss, data["loss"][0, 2])
    assert data["weights"].shape[:2] == (10, 3)
    with pytest.raises(FileExistsError):
        generate_population(config, tmp_path)


def test_cpu_workflow_is_repeatable_without_private_artifacts():
    script = Path(__file__).resolve().parents[2] / "examples/train_observer.py"
    run_demo = runpy.run_path(str(script))["run_demo"]
    first = run_demo(runs=10, steps=2, epochs=2)
    second = run_demo(runs=10, steps=2, epochs=2)
    assert first == second
    assert set(first["test_metrics"]) == {"train_mean", "statistics_ridge", "raw_weights", "structured_weights"}
    assert first["device"] == "cpu"
    # No superiority assertion: this tiny run tests the workflow, not the hypothesis.
    assert all(row["mae"] >= 0 for row in first["test_metrics"].values())
