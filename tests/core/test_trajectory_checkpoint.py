"""Inference contracts for the numeric-only trajectory milestone."""
import hashlib
import numpy as np
import pytest
import torch
from synaptic_lens.trajectory import OutputCentered, canonicalize
from synaptic_lens.trajectory_checkpoint import load_model, predict


def archive(tmp_path, invalid=False):
    torch.manual_seed(5)
    model = OutputCentered()
    arrays = {'model/' + k: v.detach().numpy() for k, v in model.state_dict().items()}
    for name, size in [('feature', 21), ('edge', 1), ('bias', 1), ('activation', 1)]:
        arrays['norm/' + name + '_mean'] = np.zeros(size, dtype=np.float64)
        arrays['norm/' + name + '_scale'] = np.ones(size, dtype=np.float64)
    if invalid:
        arrays['norm/feature_scale'][0] = 0
    path = tmp_path / 'checkpoint.npz'
    np.savez(path, **arrays)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_numeric_checkpoint_and_prediction(tmp_path):
    path, digest = archive(tmp_path)
    model, norm = load_model(path, digest)
    output = predict(model, norm, torch.zeros(2, 1299))
    assert output.shape == (2, 100)
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output[0], output[1])


def test_integrity_checked_before_loading(tmp_path):
    path, _ = archive(tmp_path)
    with pytest.raises(ValueError, match='SHA-256'):
        load_model(path, '0' * 64)


def test_zero_normalization_rejected(tmp_path):
    path, digest = archive(tmp_path, invalid=True)
    with pytest.raises(ValueError, match='feature_scale'):
        load_model(path, digest)


@pytest.mark.parametrize('weights', [torch.zeros(1298), torch.empty(0, 1299), torch.full((1299,), float('nan'))])
def test_invalid_targets_rejected(tmp_path, weights):
    path, digest = archive(tmp_path)
    model, norm = load_model(path, digest)
    with pytest.raises(ValueError):
        predict(model, norm, weights)


def test_canonicalization_preserves_relu_scale_equivalence():
    torch.manual_seed(3)
    original = torch.randn(1, 1299, dtype=torch.float64)
    scaled = original.clone()
    scaled[:, :672] *= 2
    scaled[:, 672:1280] /= 2
    query = torch.nn.functional.one_hot(torch.tensor([[2, 3]]), 10).reshape(1, 20)
    for left, right in zip(canonicalize(original, query), canonicalize(scaled, query)):
        torch.testing.assert_close(left, right)
