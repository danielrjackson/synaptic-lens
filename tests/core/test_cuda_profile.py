"""Bounded CUDA parity check for the public trajectory reader."""
import numpy as np
import pytest
import torch
from test_trajectory_checkpoint import archive
from synaptic_lens.trajectory_checkpoint import load_model, predict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_trajectory_cpu_cuda_parity(tmp_path):
    prior = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        path, digest = archive(tmp_path)
        model, norm = load_model(path, digest)
        weights = torch.randn(1, 1299, generator=torch.Generator().manual_seed(42)) * 0.1
        cpu = predict(model, norm, weights)
        gpu = predict(model.cuda(), norm, weights)
        np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-5)
    finally:
        torch.set_float32_matmul_precision(prior)
