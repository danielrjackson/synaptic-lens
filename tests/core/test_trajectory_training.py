"""Optimization/reference checks use only tiny synthetic tensors."""

import hashlib

import numpy as np
import pytest
import torch
from synaptic_lens.targets import addition_suite
from synaptic_lens.trajectory_checkpoint import load_model
from synaptic_lens.trajectory_training import (
    loss_and_backward,
    make_schedule,
    save_inference_export,
    train_trajectory,
)
from test_trajectory_data import population
from torch.nn import functional as F


def test_loss_and_gradient_match_unchunked_reference():
    class Logits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.z = torch.nn.Parameter(torch.linspace(-2, 2, 400).reshape(4, 100))

        def forward(self, x, edge, bias, queries, activation, answers, arm):
            indices = queries[:, :10].argmax(-1) * 10 + queries[:, 10:].argmax(-1)
            return self.z[:, indices]

    model = Logits()
    _, queries, answers = addition_suite()
    labels = torch.tensor([[0.0], [1.0], [1.0], [0.0]]).expand(4, 100)
    rules = torch.tensor([0, 1, 0, 1])
    z = model.z.detach().clone().requires_grad_()
    reference = F.binary_cross_entropy_with_logits(z, labels)
    for rule in (0, 1):
        zz, yy = z[rules == rule], labels[rules == rule]
        mask = (yy[:, None] > 0.5) & (yy[None, :] < 0.5)
        reference = (
            reference + (F.softplus(-(zz[:, None] - zz[None, :])) * mask).sum() / 200
        )
    reference.backward()
    result = loss_and_backward(
        model,
        None,
        None,
        None,
        torch.empty(4, 100, 1),
        queries,
        answers,
        labels,
        rules,
        "margin",
        7,
    )
    assert result["objective"] == pytest.approx(float(reference.detach()), abs=1e-6)
    torch.testing.assert_close(model.z.grad, z.grad, atol=1e-7, rtol=1e-6)


def test_schedule_preserves_historical_rng_order_and_pairs():
    actual = make_schedule(1536, 2, 24)
    for si, seed in enumerate((7, 17, 27)):
        rng = np.random.default_rng(880088 + seed)
        for update in range(2):
            np.testing.assert_array_equal(
                actual["roots"][si, update], rng.choice(1536, 24, replace=False)
            )
            np.testing.assert_array_equal(
                actual["ages"][si, update], rng.integers(0, 8, size=24)
            )
    with pytest.raises(ValueError):
        make_schedule(1, 1, 2)


def test_one_update_is_deterministic_and_numeric_export_loads(tmp_path):
    # Three distinct root IDs; all ages/rules remain with their root.
    train_one, train_two = population(0), population(1, 3)
    from synaptic_lens.trajectory_data import TrajectoryPopulation

    train = TrajectoryPopulation(
        (0, 1),
        torch.cat((train_one.weights, train_two.weights)),
        torch.cat((train_one.addition_correct, train_two.addition_correct)),
    )
    val = population(2, 2)
    first, norm, report = train_trajectory(train, val)
    second, _, other = train_trajectory(train, val)
    assert report == other and report["completed_updates"] == 1
    assert not report["historical_metrics_reproduced"]
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    path = tmp_path / "model.npz"
    save_inference_export(path, first, norm)
    loaded, loaded_norm = load_model(
        path, hashlib.sha256(path.read_bytes()).hexdigest()
    )
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
    assert set(loaded_norm) == set(norm)
    with pytest.raises(FileExistsError):
        save_inference_export(path, first, norm)


@pytest.mark.parametrize("chunk", [0, -1, True, 1.5])
def test_invalid_chunks_refuse_before_work(chunk):
    from synaptic_lens.trajectory_training import predict

    with pytest.raises(ValueError, match="Positive integer state chunk"):
        predict(None, {}, "trajectory", chunk_states=chunk)
    with pytest.raises(ValueError, match="Positive integer query chunk"):
        loss_and_backward(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "margin",
            query_chunk=chunk,
        )


def test_development_metric_support_and_unsupported_ranking():
    from synaptic_lens.trajectory_training import val_metrics

    z = np.zeros((1, 2, 8, 100), dtype=np.float32)
    result = val_metrics(z, np.zeros_like(z, dtype=np.uint8))
    assert result["support"] == 1600 and result["positive"] == 0
    assert result["full_trajectory_macro_loss"] == pytest.approx(np.log(2))
    assert result["brier"] == 0.25
    assert all(
        cell["supported_queries"] == 0 and cell["ranking"] == 0
        for cell in result["cells"].values()
    )
    with pytest.raises(ValueError, match="nonempty"):
        val_metrics(z[:0], z[:0])
