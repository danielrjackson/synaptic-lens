"""Synthetic contracts; no historical artifacts or research fitting."""
import pytest
import torch
from synaptic_lens.targets import addition_suite
from synaptic_lens.trajectory import normalize_parts
from synaptic_lens.trajectory_data import (
    TrajectoryPopulation,
    canonical_parts,
    prepare_training_validation,
)


def population(root, seed=1):
    g = torch.Generator().manual_seed(seed)
    weights = torch.randn(1, 2, 8, 1299, generator=g) * .1
    weights[:, 1, 0] = weights[:, 0, 0]
    _, query, answers = addition_suite()
    # Independent target forward, without reader canonicalization.
    flat = weights.double().reshape(-1, 1299)
    hidden = torch.relu(query.double() @ flat[:, :640].reshape(-1, 32, 20).transpose(-1, -2)
                        + flat[:, None, 640:672])
    logits = hidden @ flat[:, 672:1280].reshape(-1, 19, 32).transpose(-1, -2) + flat[:, None, 1280:1299]
    labels = logits.argmax(-1).eq(answers).reshape(1, 2, 8, 100)
    return TrajectoryPopulation((root,), weights, labels)


def test_historical_transform_order_and_validation_independence():
    train, validation = population(0), population(1, 2)
    x, v, norm = prepare_training_validation(train, validation)
    raw = canonical_parts(train)
    feature = torch.cat((raw['incoming'], raw['hidden_bias'].unsqueeze(-1)), -1)
    expected_norm = {'feature_mean': feature.mean((0, 1, 2, 3)),
                     'feature_scale': feature.std((0, 1, 2, 3), unbiased=False).clamp_min(1e-6)}
    for name, key in [('edge', 'edges'), ('bias', 'output_bias'), ('activation', 'activation')]:
        expected_norm[name + '_mean'] = raw[key].mean().reshape(1)
        expected_norm[name + '_scale'] = raw[key].std(unbiased=False).clamp_min(1e-6).reshape(1)
    expected = normalize_parts(raw, expected_norm)
    for key in norm:
        torch.testing.assert_close(norm[key], expected_norm[key], rtol=0, atol=0)
    for key in expected:
        torch.testing.assert_close(x[key], expected[key].float(), rtol=0, atol=0)
    _, _, other_norm = prepare_training_validation(train, population(2, 3))
    for key in norm:
        torch.testing.assert_close(norm[key], other_norm[key], rtol=0, atol=0)
    assert x['neurons'].shape == (1, 2, 8, 32, 21)
    assert v['activation'].shape == (1, 2, 8, 100, 32)
    assert x['root_ids'] == (0,) and v['root_ids'] == (1,)


def test_root_overlap_rejected_before_canonicalization():
    with pytest.raises(ValueError, match='Root overlap'):
        prepare_training_validation(population(0), population(0, 2))


def test_invalid_saved_labels_and_mutated_birth_rejected():
    data = population(0)
    data.addition_correct.logical_not_()
    with pytest.raises(ValueError, match='Saved addition labels'):
        canonical_parts(data)
    data = population(0)
    data.weights[0, 1, 0, 0] += 1
    with pytest.raises(ValueError, match='birth weights'):
        canonical_parts(data)


@pytest.mark.parametrize('ids', [(True,), (2304,), (0, 0)])
def test_invalid_root_identity_rejected(ids):
    data = population(0)
    with pytest.raises(ValueError, match='root IDs|Duplicate'):
        TrajectoryPopulation(ids, data.weights, data.addition_correct)


def test_canonicalization_must_preserve_native_logits(monkeypatch):
    import synaptic_lens.trajectory_data as module
    original = module.canonicalize
    def broken(*args):
        incoming, bias, edges, output_bias, activation = original(*args)
        return incoming, bias, edges + 1, output_bias, activation
    monkeypatch.setattr(module, 'canonicalize', broken)
    with pytest.raises(ValueError, match='changed native target logits'):
        canonical_parts(population(0))
