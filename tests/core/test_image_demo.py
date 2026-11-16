"""Device parity for the image Lens inference path; synthetic test weights."""
import pytest
import torch
from synaptic_lens.observer import Lens, predict_lens


def bundle():
    torch.manual_seed(7)
    return {"features": 192, "state_dict": Lens(192).state_dict(),
            "mean": torch.zeros(192), "scale": torch.ones(192),
            "y_mean": torch.tensor(1.0), "y_scale": torch.tensor(0.3)}


def test_cpu_explicit_matches_default_and_preserves_bundle():
    saved = bundle()
    x = torch.randn(3, 192)
    with torch.inference_mode():
        torch.testing.assert_close(predict_lens(saved, x), predict_lens(saved, x, device="cpu"))
    assert saved["mean"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_cuda_prediction_matches_cpu():
    saved = bundle()
    x = torch.randn(3, 192)
    with torch.inference_mode():
        cpu = predict_lens(saved, x)
        cuda = predict_lens(saved, x, device="cuda")
    assert cuda.device.type == "cuda"
    torch.testing.assert_close(cpu, cuda.cpu(), rtol=1e-5, atol=1e-6)
    assert saved["mean"].device.type == "cpu"
