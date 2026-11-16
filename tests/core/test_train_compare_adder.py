"""Walkthrough contracts: real root partitions, verified input, persisted output."""
import importlib.util
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/train_compare_adder.py"
spec = importlib.util.spec_from_file_location("train_compare_adder", EXAMPLE)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def test_partitions_are_fixed_disjoint_subsets():
    from synaptic_lens.syn88_targets import split_roots
    chosen = example.root_partitions()
    official = split_roots()
    assert chosen["train"] == official["train"][:8]
    assert chosen["validation"] == official["validation"][:2]
    assert chosen["comparison"] == official["test"][:2]
    assert len(set(sum(chosen.values(), []))) == 12


def test_root_bundle_identity_is_checked(tmp_path):
    path = tmp_path / "root.npz"
    np.savez(path, root_index=3, ages=example.AGES)
    entry = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="identity"):
        example.load_population([2], {"syn88/population/root-0002.npz": entry})
    entry["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA"):
        example.load_population([2], {"syn88/population/root-0002.npz": entry})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_training_roundtrip(tmp_path):
    from test_trajectory_data import population
    from synaptic_lens.trajectory_data import TrajectoryPopulation
    train = TrajectoryPopulation((0, 1), torch.cat([population(0).weights, population(1).weights]),
                                 torch.cat([population(0).addition_correct, population(1).addition_correct]))
    torch.set_float32_matmul_precision("highest")
    model, norm, report = example.train_trajectory(train, population(2), device="cuda")
    assert next(model.parameters()).device.type == "cuda"
    assert report["device"] == "cuda"
    path = tmp_path / "lens.npz"
    example.save_inference_export(path, model, norm)
    restored, restored_norm = example.load_model(path, hashlib.sha256(path.read_bytes()).hexdigest())
    expected = example.predict(model, norm, population(2).weights)
    actual = example.predict(restored.cuda(), restored_norm, population(2).weights)
    np.testing.assert_allclose(expected, actual, atol=1e-6)
