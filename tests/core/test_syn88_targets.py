# Target generation is an explicitly optional companion-package capability.
import pytest

pytest.importorskip("synaptic_lens_target_model_factory")

import numpy as np
import pytest
import torch
from synaptic_lens.syn88_targets import AGES, generate_root, split_roots
from synaptic_lens.targets import addition_suite, restore_adder


def test_partitions_are_root_disjoint():
    splits = split_roots()
    assert [len(splits[k]) for k in ("train", "validation", "test")] == [1536, 256, 512]
    assert sorted(sum(splits.values(), [])) == list(range(2304))


def test_generated_root_and_native_labels(tmp_path):
    torch.set_num_threads(1)
    result = generate_root(0)
    assert result["weights"].shape == (2, 8, 1299)
    assert result["addition_correct"].shape == (2, 8, 100)
    assert result["ages"].tolist() == list(AGES)
    np.testing.assert_array_equal(result["weights"][0, 0], result["weights"][1, 0])
    _, inputs, answers = addition_suite()
    for rule in range(2):
        for age in range(8):
            model = restore_adder(torch.from_numpy(result["weights"][rule, age]), 32)
            with torch.no_grad():
                labels = model(inputs).argmax(-1).eq(answers).numpy()
            np.testing.assert_array_equal(labels, result["addition_correct"][rule, age])
    path = tmp_path / "root.npz"
    np.savez_compressed(path, **result)
    with np.load(path, allow_pickle=False) as loaded:
        for key, value in result.items():
            np.testing.assert_array_equal(loaded[key], value)
    again = generate_root(0)
    np.testing.assert_array_equal(result["weights"], again["weights"])


def test_invalid_root():
    with pytest.raises(ValueError):
        generate_root(2304)
