import hashlib

import pytest
import torch

from synaptic_lens.neuron_completion import (
    REPRESENTATION_VERSION,
    NoSortLens,
    OrganicCheckpointReader,
    prepare_organic_completion,
)


def test_hidden_values_do_not_affect_reader():
    raw = torch.randn(32, 40)
    bias = torch.randn(19)
    a = prepare_organic_completion(raw, bias, 7)
    raw[7] = float("nan")
    b = prepare_organic_completion(raw, bias, 7)
    assert torch.equal(a.reader_input, b.reader_input)
    assert torch.equal(a.output_bias, b.output_bias)


def test_normalization_inverse_preserves_observed():
    raw = torch.randn(32, 40)
    bias = torch.randn(19)
    view = prepare_organic_completion(raw, bias, 7)
    normalized = raw[7].double().clone()
    scale = view.external_original_scale
    normalized[:21] /= scale
    normalized[21:] *= scale
    restored = view.reinsert_prediction(normalized.float())
    torch.testing.assert_close(restored, raw)
    assert torch.equal(restored[:7], raw[:7])


def test_checkpoint_hash_and_permutation(tmp_path):
    torch.set_num_threads(1)
    model = NoSortLens(16, 1, 2).eval()
    saved = {
        "config": {
            "representation_version": REPRESENTATION_VERSION,
            "width": 16,
            "blocks": 1,
            "heads": 2,
        },
        "positional": False,
        "state_dict": model.state_dict(),
        "mean": torch.zeros(1, 1, 40),
        "std": torch.ones(1, 1, 40),
        "bias_mean": torch.zeros(1, 19),
        "bias_std": torch.ones(1, 19),
    }
    path = tmp_path / "reader.pt"
    torch.save(saved, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    reader = OrganicCheckpointReader(path, digest, "cpu")
    raw = torch.randn(32, 40)
    bias = torch.randn(19)
    permutation = torch.randperm(32)
    a = prepare_organic_completion(raw, bias, 7)
    b = prepare_organic_completion(
        raw[permutation], bias, int((permutation == 7).nonzero()[0])
    )
    torch.testing.assert_close(
        reader(a.reader_input, a.output_bias),
        reader(b.reader_input, b.output_bias),
        atol=1e-5,
        rtol=1e-5,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        OrganicCheckpointReader(path, "0" * 64, "cpu")
