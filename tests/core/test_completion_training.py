import hashlib

import numpy as np
import pytest
import torch

from synaptic_lens.completion_training import (
    parse_final_weights,
    train_completion,
    training_partitions,
)
from synaptic_lens.neuron_completion import (
    OrganicCheckpointReader,
    normalize_neurons,
    prepare_organic_completion,
)
from synaptic_lens.syn88_targets import split_roots


def test_partitions_never_use_consumed_final():
    p = training_partitions(768, 128)
    roots = split_roots()["train"]
    assert not set(p["fit"]) & set(p["development"])
    assert not set(p["fit"] + p["development"]) & set(roots[512:640] + roots[1152:1280])


def test_final_age_layout():
    v = np.arange(2 * 8 * 1299, dtype=np.float32).reshape(2, 8, 1299)
    raw, bias = parse_final_weights(v)
    assert raw[1, 7, 3] == v[1, 7, 7 * 20 + 3]
    assert raw[0, 7, 20] == v[0, 7, 640 + 7]
    assert raw[1, 7, 21 + 4] == v[1, 7, 672 + 4 * 32 + 7]
    assert bias[1, 8] == v[1, 7, 1280 + 8]
    with pytest.raises(ValueError):
        parse_final_weights(v.astype(np.float64))


def test_cpu_fixture_training_reload_and_fit_only_stats(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(71)

    def pop(root):
        return {
            "roots": [root],
            "raw": torch.randn(2, 32, 40),
            "bias": torch.randn(2, 19),
        }

    fit, dev = pop(1), pop(2)
    checkpoint, report = train_completion(
        fit,
        dev,
        updates=2,
        batch_size=2,
        validation_every=1,
        device="cpu",
        width=16,
        blocks=1,
        heads=2,
    )
    expected, _ = normalize_neurons(fit["raw"])
    assert torch.equal(checkpoint["mean"], expected.mean((0, 1), keepdim=True))
    assert report["selected_step"] in (1, 2)
    assert all(torch.isfinite(v).all() for v in checkpoint["state_dict"].values())
    p = tmp_path / "reader.pt"
    torch.save(checkpoint, p)
    reader = OrganicCheckpointReader(
        p, hashlib.sha256(p.read_bytes()).hexdigest(), "cpu"
    )
    raw, bias = dev["raw"][0], dev["bias"][0]
    view = prepare_organic_completion(raw, bias, 7)
    restored = view.predict(reader)
    assert torch.isfinite(restored).all()
    assert torch.equal(restored[:7], raw[:7]) and torch.equal(restored[8:], raw[8:])
    changed = raw.clone()
    changed[7] = float("nan")
    hidden = prepare_organic_completion(changed, bias, 7)
    assert torch.equal(
        reader(view.reader_input, bias), reader(hidden.reader_input, bias)
    )
    with pytest.raises(ValueError, match="overlap"):
        train_completion(fit, fit, updates=1, device="cpu")
