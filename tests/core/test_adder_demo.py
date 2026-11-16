import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("demo_adder_lens", Path(__file__).resolve().parents[2] / "examples/demo_adder_lens.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_summary_extreme_logits_and_accuracy():
    result = module.summarize(np.array([-1000., 1000.]), np.array([0, 1]))
    assert result["lens_accuracy"] == 1.
    assert result["brier"] == 0.
    assert result["queries"] == 2
    assert result["mean_predicted_correctness"] == .5


@pytest.mark.parametrize("logits,labels", [([1], [0, 1]), ([float('nan')], [0]), ([1], [2])])
def test_summary_rejects_invalid_inputs(logits, labels):
    with pytest.raises(ValueError):
        module.summarize(logits, labels)
