import pytest
import torch
from torch import nn

from synaptic_lens.target_records import ParameterLocation, affine_relu_records


def test_complete_coverage_and_intervention_reconstruction():
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 19))
    records = affine_relu_records(model)
    assert records.record_count == 1319
    assert records.parameter_count == 1299
    inputs = torch.randn(12, 20)
    torch.testing.assert_close(records.evaluate(inputs), model(inputs), rtol=0, atol=0)
    hidden = model[1](model[0](inputs))
    for replacement in (0., float(hidden[:, 3].detach().max())):
        changed = hidden.clone()
        changed[:, 3] = replacement
        torch.testing.assert_close(
            records.evaluate(inputs, (0, 3, replacement)), model[2](changed), rtol=0, atol=0
        )


def test_variable_depth_and_absent_bias():
    model = nn.Sequential(nn.Linear(3, 5, bias=False), nn.ReLU(), nn.Linear(5, 2),
                          nn.ReLU(), nn.Linear(2, 4))
    records = affine_relu_records(model)
    assert records.widths == (3, 5, 2, 4)
    assert records.biases[0] is None
    inputs = torch.randn(8, 3)
    torch.testing.assert_close(records.evaluate(inputs), model(inputs), rtol=0, atol=0)


@pytest.mark.parametrize('name,shape,coordinate', [('weight', (2, 3), (1, 2)),
                                                  ('bias', (2,), (1,))])
def test_masking_is_value_blind_and_distinguishes_zero(name, shape, coordinate):
    model = nn.Sequential(nn.Linear(3, 2))
    getattr(model[0], name).data[coordinate] = 17.
    first = affine_relu_records(model)
    location = ParameterLocation(0, name, shape, coordinate)
    masked = first.masked(location)
    getattr(model[0], name).data[coordinate] = -23.
    assert masked.identity() == affine_relu_records(model).masked(location).identity()
    getattr(model[0], name).data[coordinate] = 0.
    zero = affine_relu_records(model)
    assert masked.identity() != zero.identity()
    assert first.complete and not masked.complete
    with pytest.raises(ValueError, match='incomplete'):
        masked.evaluate(torch.zeros(1, 3))


def test_unsupported_and_sharing_fail_explicitly():
    with pytest.raises(ValueError, match='Unsupported'):
        affine_relu_records(nn.Sequential(nn.Linear(3, 2), nn.Sigmoid()))
    shared = nn.Linear(3, 3)
    with pytest.raises(ValueError, match='Shared'):
        affine_relu_records(nn.Sequential(shared, nn.ReLU(), shared))


def test_records_are_snapshot_and_preserve_numeric_dtype():
    model = nn.Sequential(nn.Linear(3, 2)).double()
    records = affine_relu_records(model)
    identity = records.identity()
    with torch.no_grad():
        model[0].weight.add_(1)
    assert records.identity() == identity
    assert records.weights[0].dtype == torch.float64
    assert affine_relu_records(model).identity() != identity


def test_consistent_hidden_renaming_preserves_intervention_maps():
    torch.manual_seed(13)
    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
    renamed = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))
    permutation = torch.tensor([3, 0, 4, 1, 2])
    with torch.no_grad():
        renamed[0].weight.copy_(model[0].weight[permutation])
        renamed[0].bias.copy_(model[0].bias[permutation])
        renamed[2].weight.copy_(model[2].weight[:, permutation])
        renamed[2].bias.copy_(model[2].bias)
    first, second = affine_relu_records(model), affine_relu_records(renamed)
    inputs = torch.randn(9, 3)
    torch.testing.assert_close(first.evaluate(inputs), second.evaluate(inputs))
    for new_unit, old_unit in enumerate(permutation.tolist()):
        torch.testing.assert_close(
            first.evaluate(inputs, (0, old_unit, 0.)),
            second.evaluate(inputs, (0, new_unit, 0.)),
        )


def test_adapter_rejects_subclass_with_changed_semantics():
    class DifferentLinear(nn.Linear):
        def forward(self, inputs):
            return super().forward(inputs).square()

    with pytest.raises(ValueError, match='Unsupported'):
        affine_relu_records(nn.Sequential(DifferentLinear(3, 2)))
