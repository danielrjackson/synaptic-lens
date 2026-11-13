import pytest
import torch
from synaptic_lens.mnist_observer import (
    ObserverPartition,
    features,
    fit_mean_risk,
    target_layout,
)


def fixture(root, seed=1):
    g = torch.Generator().manual_seed(seed)
    return ObserverPartition(
        (root, root), torch.randn(2, 19658, generator=g), torch.tensor([0.2, 0.8])
    )


def layout():
    return target_layout()


def test_root_leakage_rejected_and_images_not_accepted():
    with pytest.raises(ValueError, match="overlap"):
        fit_mean_risk(fixture(1), fixture(1, 2), layout=layout(), epochs=1)
    images = ObserverPartition((1, 2), torch.zeros(2, 1, 28, 28), torch.ones(2))
    with pytest.raises(ValueError, match="never images"):
        features(images, mode="raw", layout=layout())


def test_statistics_order_and_training_only_normalization():
    train, validation = fixture(1), fixture(2, 2)
    x = features(train, mode="statistics", layout=layout())
    block = train.weights[:, : layout()[0]["stop"]]
    torch.testing.assert_close(
        x[:, :6],
        torch.stack(
            (
                block.mean(-1),
                block.std(-1, unbiased=False),
                block.square().mean(-1).sqrt(),
                block.abs().mean(-1),
                block.amin(-1),
                block.amax(-1),
            ),
            -1,
        ),
    )
    assert x.shape == (2, 192)
    bundle, report = fit_mean_risk(train, validation, layout=layout(), epochs=1)
    torch.testing.assert_close(bundle["mean"], x.mean(0))
    other, _ = fit_mean_risk(train, fixture(3, 3), layout=layout(), epochs=1)
    torch.testing.assert_close(bundle["mean"], other["mean"], rtol=0, atol=0)
    assert report["train_roots"] == [1] and report["validation_roots"] == [2]
    assert not report["historical_metrics_reproduced"]


def test_layout_and_nonfinite_supervision_fail_closed():
    data = fixture(1)
    with pytest.raises(ValueError, match="32"):
        features(data, mode="statistics", layout=[])
    data.mean_risk[0] = float("nan")
    with pytest.raises(ValueError, match="mean risk"):
        data.validate()


def test_arbitrary_contiguous_ranges_are_not_published_tensor_statistics():
    wrong = [
        {"start": i * 614, "stop": (i + 1) * 614 if i < 31 else 19658}
        for i in range(32)
    ]
    with pytest.raises(ValueError, match="published named-tensor ordering"):
        features(fixture(1), mode="statistics", layout=wrong)


def test_development_evaluation_requires_unused_roots():
    from synaptic_lens.mnist_observer import evaluate_mean_risk

    bundle, _ = fit_mean_risk(fixture(1), fixture(2, 2), layout=layout(), epochs=1)
    with pytest.raises(ValueError, match="Development roots overlap"):
        evaluate_mean_risk(bundle, fixture(2), layout=layout())
    result = evaluate_mean_risk(bundle, fixture(3, 3), layout=layout())
    assert result["root_count"] == 1 and result["state_count"] == 2
