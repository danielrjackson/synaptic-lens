from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from synaptic_lens.scalar_lens import (
    ScalarLens,
    masked_scalar,
    matched_scalar_readers,
    scalar_tokens,
)
from synaptic_lens.target_records import (
    ParameterLocation,
    ScalarMaskMetadata,
    affine_relu_records,
)
from synaptic_lens.transformer_lens import LensQuery, ReaderConfig, TransformerLens


def target(hidden=4):
    return nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 2))


def config(**options):
    return ReaderConfig(case_width=3, width=16, heads=4, blocks=2, max_tokens=128, **options)


def locations():
    return (
        ParameterLocation(0, "weight", (4, 3), (2, 1)),
        ParameterLocation(0, "bias", (4,), (2,)),
        ParameterLocation(1, "weight", (2, 4), (1, 2)),
        ParameterLocation(1, "bias", (2,), (1,)),
    )


def query(location, metadata=ScalarMaskMetadata(), **options):
    return LensQuery(
        mode="reconstruction",
        scope="single_scalar",
        operation="predict_missing",
        parameter_references=(location,),
        mask_metadata=(metadata,),
        measurement_bundle=("parameter_value",),
        **options,
    )


@pytest.mark.parametrize("location", locations())
@pytest.mark.parametrize("local", [False, True])
def test_four_type_location_decoding_and_coverage(location, local):
    records = affine_relu_records(target())
    reader = ScalarLens(
        config(relation_conditioned=True),
        local=local,
        means=(1.0, 2.0, 3.0, 4.0),
        scales=(2.0,) * 4,
    )
    encoded = reader.encode((masked_scalar(records, location),))
    result = reader.predict(encoded, query(location))
    type_id = 2 * location.owner + int(location.name == "bias")
    assert result.status == "supported" and result.values.shape == (1, 1)
    torch.testing.assert_close(result.values, 1 + type_id + 2 * result.normalized_values)
    assert result.coverage == ((records.parameter_count - 1, records.parameter_count),)
    if local:
        assert result.context_coverage[0][0] < records.parameter_count - 1
    else:
        assert result.context_coverage == result.coverage
    assert result.uncertainty is None and result.learned_capability == "unvalidated"
    result.normalized_values.square().mean().backward()
    assert reader.feature_projection.weight.grad.abs().sum() > 0
    assert reader.scalar_head[0].weight.grad.abs().sum() > 0
    assert reader.blocks[0].relation_bias.grad.abs().sum() > 0


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("location", locations())
def test_hidden_label_and_corrupted_numeric_value_cannot_change_features_or_prediction(
    local, location
):
    torch.manual_seed(8)
    clean = target()
    other = deepcopy(clean)
    first = masked_scalar(affine_relu_records(clean), location)
    reader = ScalarLens(config(relation_conditioned=True), local=local)
    a = reader.encode((first,))
    expected = reader.predict(a, query(location)).values
    for hidden_value in (0.0, -12345.0, 12345.0):
        with torch.no_grad():
            getattr(other[2 * location.owner], location.name)[location.coordinate] = hidden_value
        changed = masked_scalar(affine_relu_records(other), location)
        corrupted = masked_scalar(
            affine_relu_records(other), location, ScalarMaskMetadata(corruption_applied=True)
        )
        assert first.identity() == changed.identity()
        assert first.identity() != corrupted.identity()
        for records in (changed, corrupted):
            encoded = reader.encode((records,))
            torch.testing.assert_close(
                a.tokens[0].features, encoded.tokens[0].features, rtol=0, atol=0
            )
            torch.testing.assert_close(a.states, encoded.states, rtol=0, atol=0)
            actual = reader.predict(encoded, query(location, records.mask_metadata)).values
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_real_zero_and_single_missing_contract_and_query_metadata():
    model = target()
    location = locations()[1]
    with torch.no_grad():
        model[0].bias[2] = 0
    complete = affine_relu_records(model)
    assert complete.complete
    reader = ScalarLens(config())
    with pytest.raises(ValueError, match="declared mask"):
        reader.encode((complete,))
    masked = masked_scalar(complete, location)
    assert not masked.complete and masked.biases[0][2] == 0
    with pytest.raises(ValueError, match="exactly one"):
        masked_scalar(masked, locations()[0])
    with pytest.raises(ValueError, match="declare a missing"):
        masked_scalar(complete, location, ScalarMaskMetadata(missing=False))
    encoded = reader.encode((masked,))
    with pytest.raises(ValueError, match="location/flags"):
        reader.predict(encoded, query(locations()[0]))
    with pytest.raises(ValueError, match="location/flags"):
        reader.predict(encoded, query(location, ScalarMaskMetadata(corruption_applied=True)))
    with pytest.raises(ValueError, match="behavior settings"):
        reader.predict(encoded, query(location, cases=torch.zeros(1, 3)))
    assert reader.predict(encoded, replace(query(location), task="unknown")).status == "unsupported"
    base = TransformerLens(config())
    assert base.predict(base.encode((masked,)), LensQuery(torch.zeros(1, 3))).status == "partial"


def test_local_counts_and_no_context_outside_selected_graph():
    large = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 19))
    records = affine_relu_records(large)
    for loc, expected in (
        (ParameterLocation(0, "weight", (32, 20), (3, 2)), 141),
        (ParameterLocation(0, "bias", (32,), (3,)), 79),
        (ParameterLocation(1, "weight", (19, 32), (2, 3)), 141),
        (ParameterLocation(1, "bias", (19,), (2,)), 65),
    ):
        tokens = scalar_tokens(
            masked_scalar(records, loc),
            device="cpu",
            dtype=torch.float32,
            max_tokens=4096,
            local=True,
        )
        assert len(tokens.features) == expected
        assert tokens.sources[tokens.units :].min() >= 0
        assert tokens.destinations[tokens.units :].max() < tokens.units
    small = target()
    changed = deepcopy(small)
    loc = locations()[1]
    # b1 unit2 context excludes edges into other hidden units and their biases.
    with torch.no_grad():
        changed[0].weight[0, 0] += 999
        changed[0].bias[0] += 999
    reader = ScalarLens(config(relation_conditioned=True), local=True)
    a, b = (reader.encode((masked_scalar(affine_relu_records(m), loc),)) for m in (small, changed))
    torch.testing.assert_close(a.states, b.states, rtol=0, atol=0)


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("location", locations())
def test_scalar_prediction_commutes_with_hidden_renaming(local, location):
    torch.manual_seed(9)
    model = target()
    renamed = deepcopy(model)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        renamed[0].weight.copy_(model[0].weight[permutation])
        renamed[0].bias.copy_(model[0].bias[permutation])
        renamed[2].weight.copy_(model[2].weight[:, permutation])
    coordinate = list(location.coordinate)
    if location.owner == 0:
        coordinate[0] = int(torch.where(permutation == coordinate[0])[0][0])
    elif location.name == "weight":
        coordinate[1] = int(torch.where(permutation == coordinate[1])[0][0])
    new_location = replace(location, coordinate=tuple(coordinate))
    reader = ScalarLens(config(relation_conditioned=True), local=local)
    with torch.no_grad():
        for block in reader.blocks:
            block.relation_bias.normal_()
    a = reader.predict(
        reader.encode((masked_scalar(affine_relu_records(model), location),)), query(location)
    ).values
    b = reader.predict(
        reader.encode((masked_scalar(affine_relu_records(renamed), new_location),)),
        query(new_location),
    ).values
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


def test_dynamic_width_padding_batch_query_indices_and_chunk_gradients():
    torch.manual_seed(4)
    location = locations()[0]
    other_location = ParameterLocation(1, "bias", (2,), (0,))
    records = (
        masked_scalar(affine_relu_records(target()), location),
        masked_scalar(affine_relu_records(target(hidden=2)), other_location),
    )
    reader = ScalarLens(replace(config(), width=24), local=True)
    batch = reader.encode(records)
    q = replace(
        query(location),
        parameter_references=(location, other_location, location),
        mask_metadata=(ScalarMaskMetadata(),) * 3,
        target_indices=torch.tensor([0, 1, 0]),
    )
    before = reader.predict(batch, q, chunk_size=10)
    after = reader.predict(batch, q, chunk_size=1)
    torch.testing.assert_close(before.values, after.values, rtol=1e-6, atol=1e-6)
    parameters = tuple(reader.parameters())
    a = torch.autograd.grad(before.normalized_values.square().mean(), parameters, retain_graph=True)
    b = torch.autograd.grad(after.normalized_values.square().mean(), parameters, retain_graph=True)
    for first, second in zip(a, b):
        torch.testing.assert_close(first, second, rtol=2e-4, atol=2e-6)
    for index, loc in enumerate((location, other_location)):
        alone = reader.predict(reader.encode((records[index],)), query(loc))
        torch.testing.assert_close(
            before.values[index : index + 1], alone.values, rtol=1e-6, atol=1e-6
        )
    with pytest.raises(ValueError, match="Batched targets"):
        reader.predict(batch, query(location))


def test_scalar_cache_flags_mask_scope_encoder_and_current_head():
    reader = ScalarLens(config()).eval()
    records = affine_relu_records(target())
    location = locations()[0]
    first = masked_scalar(records, location)
    cache = {}
    with torch.no_grad():
        encoded = reader.encode((first,), cache)
        assert reader.encode((first,), cache).states.data_ptr() == encoded.states.data_ptr()
        before = reader.predict(encoded, query(location)).values
        reader.scalar_head[-1].bias.add_(1)
        assert reader.encode((first,), cache).states.data_ptr() == encoded.states.data_ptr()
        torch.testing.assert_close(reader.predict(encoded, query(location)).values, before + 1)
        for metadata in (
            ScalarMaskMetadata(corruption_applied=True),
            ScalarMaskMetadata(operator_version="other-v2"),
        ):
            changed = reader.encode((masked_scalar(records, location, metadata),), cache)
            assert changed.states.data_ptr() != encoded.states.data_ptr()
        another_mask = reader.encode((masked_scalar(records, locations()[1]),), cache)
        assert another_mask.states.data_ptr() != encoded.states.data_ptr()
        reader.feature_projection.weight.add_(0.1)
        with pytest.raises(ValueError, match="Encoder changed"):
            reader.predict(encoded, query(location))
    reader.train()
    with pytest.raises(ValueError, match="Detached inference cache"):
        reader.encode((first,), cache)
    reader.eval()
    with torch.no_grad():
        current_cached = reader.encode((first,), cache)
    with pytest.raises(ValueError, match="Detached inference encoding"):
        reader.predict(current_cached, query(location))


def test_global_and_local_encoder_cache_namespaces_remain_separate():
    readers = matched_scalar_readers(config())
    records = masked_scalar(affine_relu_records(target()), locations()[0])
    cache = {}
    with torch.no_grad():
        global_encoding = readers["R"].eval().encode((records,), cache)
        local_encoding = readers["L"].eval().encode((records,), cache)
    assert global_encoding.cache_key != local_encoding.cache_key
    assert len(cache) == 2
    assert global_encoding.states.shape[1] > local_encoding.states.shape[1]


def test_initial_active_parameter_counts_matching_and_no_legacy_heads():
    readers = matched_scalar_readers(ReaderConfig())
    assert {arm: sum(p.numel() for p in m.parameters()) for arm, m in readers.items()} == {
        "R": 530985,
        "C": 530945,
        "L": 530985,
    }
    assert not hasattr(readers["R"], "baseline_head")
    assert not hasattr(readers["R"], "readout_attention")
    for name, value in readers["C"].state_dict().items():
        for arm in ("R", "L"):
            torch.testing.assert_close(value, readers[arm].state_dict()[name], rtol=0, atol=0)
    for arm in ("R", "L"):
        assert all(block.relation_bias.eq(0).all() for block in readers[arm].blocks)
