"""Synthetic CPU regressions for extracted mechanisms and legacy identities."""

import io
import pickle
import runpy
import sys
from pathlib import Path

import pytest
import torch

from synaptic_lens import allocation, reader_config, transformer_lens
from synaptic_lens.ranking import average_ranks, binary_auc
from synaptic_lens.tabular import write_csv


def test_reader_config_legacy_pickle_and_checkpoint_identity():
    assert reader_config.ReaderConfig is transformer_lens.ReaderConfig
    assert (
        reader_config.DedicatedBoundaryReaderConfig
        is transformer_lens.DedicatedBoundaryReaderConfig
    )
    for cls in (reader_config.ReaderConfig, reader_config.DedicatedBoundaryReaderConfig):
        config = cls(width=8, heads=2, blocks=1)
        assert cls.__module__ == "synaptic_lens.transformer_lens"
        assert pickle.loads(pickle.dumps(config)) == config
        model = transformer_lens.TransformerLens(config)
        stream = io.BytesIO()
        torch.save({"config": config, "state": model.state_dict()}, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False, map_location="cpu")
        clone = transformer_lens.TransformerLens(restored["config"])
        clone.load_state_dict(restored["state"], strict=True)
        assert all(torch.equal(v, clone.state_dict()[k]) for k, v in model.state_dict().items())
    assert not any(name.startswith("synaptic_lens._impl.") for name in sys.modules)


def test_normalization_outputs_and_gradients_match_frozen_expression():
    scores = torch.tensor([-3.0, -0.2, 0.7, 4.0], requires_grad=True)
    coefficients = torch.tensor([0.1, 0.2, 0.3, 0.4])
    raw = scores.clamp(-allocation.math.log(4.0), allocation.math.log(4.0)).exp()
    expected = 0.1 + 0.9 * raw / (coefficients * raw).sum()
    actual = allocation.normalized_multipliers(scores, coefficients)
    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.autograd.grad(actual.square().sum(), scores, retain_graph=True)[0],
        torch.autograd.grad(expected.square().sum(), scores)[0],
    )
    assert torch.isclose((coefficients * actual).sum(), torch.tensor(1.0))


@pytest.mark.parametrize("policy", ["uniform", "sqrt_loss", "learned"])
def test_policy_detaches(policy):
    loss = torch.tensor([0.1, 0.2], requires_grad=True)
    scores = allocation.policy_scores(policy, unreduced_loss=loss, learned_scores=loss)
    assert not scores.requires_grad


def test_antithetic_optimizer_matches_original_and_restores_on_failure():
    # Generic fixture: public tests do not depend on a historical study adapter.
    def make_controller():
        return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))

    torch.manual_seed(24)
    old = make_controller()
    new = make_controller()
    new.load_state_dict(old.state_dict())
    direction = allocation.normalized_direction(new, generator=torch.Generator().manual_seed(5))
    old_optimizer = torch.optim.Adam(old.parameters(), lr=0.001, foreach=False)
    new_optimizer = torch.optim.Adam(new.parameters(), lr=0.001, foreach=False)
    kwargs = dict(direction=direction, reward_plus=0.4, reward_minus=-0.1)
    # Frozen pre-extraction reward-ascent gradient: -(0.4 - -0.1)/(2 * 0.01).
    old_optimizer.zero_grad(set_to_none=True)
    for parameter, delta in zip(old.parameters(), direction, strict=True):
        parameter.grad = -25.0 * delta
    old_optimizer.step()
    allocation.antithetic_adam_update(new, new_optimizer, **kwargs)
    assert all(torch.equal(a, b) for a, b in zip(old.parameters(), new.parameters(), strict=True))
    for old_state, new_state in zip(
        old_optimizer.state.values(), new_optimizer.state.values(), strict=True
    ):
        assert all(torch.equal(old_state[k], new_state[k]) for k in old_state)
    before = [parameter.detach().clone() for parameter in new.parameters()]
    with pytest.raises(TypeError):
        allocation.antithetic_scores(new, {}, direction=direction)
    assert all(torch.equal(a, b) for a, b in zip(before, new.parameters(), strict=True))


def test_ranking_ties_and_unsupported_auc():
    scores = torch.tensor([3.0, 1.0, 3.0, 2.0])
    assert torch.equal(average_ranks(scores), torch.tensor([2.5, 0, 2.5, 1], dtype=torch.float64))
    positive = torch.tensor([True, False, False, False])
    assert binary_auc(scores, positive) == pytest.approx(5 / 6)
    assert binary_auc(scores, torch.ones(4, dtype=torch.bool)) is None
    assert binary_auc(scores, torch.zeros(4, dtype=torch.bool)) is None
    assert binary_auc(torch.ones(4), positive) == 0.5


def test_csv_exact_bytes(tmp_path):
    path = tmp_path / "table.csv"
    write_csv(path, [{"label": "a,b", "value": 0.5}, {"label": "é", "value": 1}])
    assert path.read_bytes() == 'label,value\n"a,b",0.5\né,1\n'.encode("utf-8")


def test_synthetic_cpu_example_entrypoint(capsys):
    root = Path(__file__).resolve().parents[2]
    runpy.run_path(str(root / "examples/synthetic_allocation.py"), run_name="__main__")
    assert "Synthetic CPU allocation weights:" in capsys.readouterr().out
