from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import torch
from torch import nn

from synaptic_lens.target_records import ParameterLocation, affine_relu_records
from synaptic_lens.transformer_lens import (
    DedicatedBoundaryReaderConfig,
    LensQuery,
    ReaderConfig,
    TransformerLens,
    boundary_attention_masks,
    matched_readers,
    tokenize_target,
)


def config(**options):
    return ReaderConfig(case_width=3, width=16, heads=4, blocks=2, max_tokens=128, **options)


def dedicated_config(**options):
    return DedicatedBoundaryReaderConfig(
        case_width=3, width=16, heads=4, blocks=2, max_tokens=128, **options
    )


def target(hidden=4):
    return nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, 2))


def role_query(cases, references=None):
    return LensQuery(
        cases,
        mode="intervention",
        scope="individual_unit",
        operation="clamp",
        unit_references=references or tuple((0, i % 4) for i in range(len(cases))),
        directions=torch.tensor([-1, 1] * ((len(cases) + 1) // 2))[: len(cases)],
        maxima=torch.ones(len(cases)),
    )


def test_full_detail_tokens_numeric_coverage_and_missing_value_blindness():
    model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 19))
    records = affine_relu_records(model)
    token = tokenize_target(records, device="cpu", dtype=torch.float32, max_tokens=1319)
    assert token.features.shape == (1319, 12)
    assert token.units == 71
    assert token.observed_parameters == token.numeric_parameters == 1299
    assert token.sources[71:91].tolist() == list(range(20))
    assert token.destinations[71:91].tolist() == [20] * 20
    location = ParameterLocation(0, "weight", (32, 20), (3, 2))
    masked = records.masked(location)
    changed = deepcopy(model)
    changed[0].weight.data[3, 2] = 10000
    other = affine_relu_records(changed).masked(location)
    a = tokenize_target(masked, device="cpu", dtype=torch.float32, max_tokens=1319)
    b = tokenize_target(other, device="cpu", dtype=torch.float32, max_tokens=1319)
    torch.testing.assert_close(a.features, b.features, rtol=0, atol=0)
    assert a.observed_parameters == 1298 and not a.complete
    with pytest.raises(ValueError, match="no truncation"):
        tokenize_target(records, device="cpu", dtype=torch.float32, max_tokens=1318)


def test_matched_initial_function_and_common_parameters():
    torch.manual_seed(11)
    records = affine_relu_records(target())
    content, relation = matched_readers(config(), seed=7)
    cases = torch.randn(7, 3)
    for query in (LensQuery(cases), role_query(cases)):
        a = content.predict(content.encode((records,)), query).values
        b = relation.predict(relation.encode((records,)), query).values
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    for name, value in content.state_dict().items():
        torch.testing.assert_close(value, relation.state_dict()[name], rtol=0, atol=0)
    assert sum(p.numel() for p in relation.parameters()) > sum(
        p.numel() for p in content.parameters()
    )


def test_baseline_raw_logits_support_stable_training_without_changing_outputs():
    records = affine_relu_records(target())
    reader = TransformerLens(config())
    with torch.no_grad():
        reader.baseline_head.weight.zero_()
        reader.baseline_head.bias.fill_(-1000)
    response = reader.predict(reader.encode((records,)), LensQuery(torch.zeros(2, 3)))
    assert response.baseline_logits.eq(-1000).all()
    assert response.values.eq(0).all()
    assert (
        reader.predict(reader.encode((records,)), role_query(torch.zeros(2, 3))).baseline_logits
        is None
    )


@pytest.mark.parametrize("relation", [False, True])
def test_hidden_renaming_preserves_predictions_with_remapped_queries(relation):
    torch.manual_seed(9)
    model = target()
    renamed = deepcopy(model)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        renamed[0].weight.copy_(model[0].weight[permutation])
        renamed[0].bias.copy_(model[0].bias[permutation])
        renamed[2].weight.copy_(model[2].weight[:, permutation])
    reader = TransformerLens(config(relation_conditioned=relation))
    if relation:
        with torch.no_grad():
            for block in reader.blocks:
                block.relation_bias.normal_()
    cases = torch.randn(4, 3)
    first = reader.encode((affine_relu_records(model),))
    second = reader.encode((affine_relu_records(renamed),))
    torch.testing.assert_close(
        reader.predict(first, LensQuery(cases)).values,
        reader.predict(second, LensQuery(cases)).values,
    )
    before = role_query(cases)
    after = role_query(
        cases, tuple((0, int(torch.where(permutation == i)[0][0])) for i in range(4))
    )
    torch.testing.assert_close(
        reader.predict(first, before).values, reader.predict(second, after).values
    )


@pytest.mark.parametrize("relation", [False, True])
def test_padding_and_batch_isolation(relation):
    torch.manual_seed(2)
    first = affine_relu_records(target())
    second = affine_relu_records(
        nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 2))
    )
    reader = TransformerLens(config(relation_conditioned=relation))
    cases = torch.randn(6, 3)
    batch = reader.encode((first, second))
    query = LensQuery(cases, target_indices=torch.tensor([0, 1, 0, 1, 0, 1]))
    predicted = reader.predict(batch, query).values
    for index, records in enumerate((first, second)):
        isolated = reader.predict(reader.encode((records,)), LensQuery(cases[index::2])).values
        torch.testing.assert_close(predicted[index::2], isolated)


def test_chunking_preserves_full_map_predictions_and_encoder_gradients():
    torch.manual_seed(8)
    first = TransformerLens(config(relation_conditioned=True))
    second = deepcopy(first)
    records = affine_relu_records(target())
    cases = torch.randn(9, 3)
    baseline, intervention = LensQuery(cases), role_query(cases)
    predictions = []
    for reader, chunk in ((first, 20), (second, 2)):
        encoded = reader.encode((records,))
        base = reader.predict(encoded, baseline, chunk_size=chunk).values
        roles = reader.predict(encoded, intervention, chunk_size=chunk).values
        (base.square().mean() + roles.square().mean()).backward()
        predictions.append((base, roles))
    for a, b in zip(predictions[0], predictions[1]):
        torch.testing.assert_close(a, b)
    for (name, a), (_, b) in zip(first.named_parameters(), second.named_parameters()):
        assert a.grad is not None, name
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-4, atol=2e-6)
    assert first.feature_projection.weight.grad.abs().sum() > 0
    assert first.blocks[0].relation_bias.grad.abs().sum() > 0


def test_inference_cache_content_weights_mask_and_decoder_invalidation():
    reader = TransformerLens(config()).eval()
    records = affine_relu_records(target())
    cache = {}
    query = LensQuery(torch.randn(3, 3))
    with torch.no_grad():
        first = reader.encode((records,), cache)
        same = reader.encode((records,), cache)
        assert first.states.data_ptr() == same.states.data_ptr()
        torch.testing.assert_close(
            reader.predict(first, query).values,
            reader.predict(reader.encode((records,)), query).values,
        )
        reader.baseline_head.bias.add_(1)
        assert reader.encode((records,), cache).states.data_ptr() == first.states.data_ptr()
        reader.feature_projection.weight.add_(0.1)
        with pytest.raises(ValueError, match="Encoder changed"):
            reader.predict(first, query)
        new = reader.encode((records,), cache)
        assert new.cache_key != first.cache_key
        masked = records.masked(ParameterLocation(0, "bias", (4,), (1,)))
        partial = reader.encode((masked,), cache)
        assert partial.cache_key != new.cache_key
        response = reader.predict(partial, query)
        assert response.status == "partial" and response.values is None
        assert response.coverage == ((records.parameter_count - 1, records.parameter_count),)
    with pytest.raises(ValueError, match="inference cache"):
        reader.encode((records,), cache)
    reader.train()
    with pytest.raises(ValueError, match="inference encoding"):
        reader.predict(new, query)


def test_encoder_update_invalidates_live_encoding_and_no_zero_output_rule():
    reader = TransformerLens(config())
    records = affine_relu_records(target())
    encoded = reader.encode((records,))
    query = role_query(torch.zeros(2, 3))
    with torch.no_grad():
        reader.intervention_head.weight.zero_()
        reader.intervention_head.bias.fill_(1)
    response = reader.predict(encoded, replace(query, maxima=torch.zeros(2)))
    assert response.uncertainty is None and response.learned_capability == "unvalidated"
    torch.testing.assert_close(response.values, torch.ones(2, 3), rtol=0, atol=0)
    optimizer = torch.optim.AdamW(reader.parameters(), lr=0.001)
    reader.predict(encoded, LensQuery(torch.ones(2, 3))).values.sum().backward()
    optimizer.step()
    with pytest.raises(ValueError, match="Encoder changed"):
        reader.predict(encoded, query)


def test_unknown_queries_and_invalid_reference_are_explicit():
    reader = TransformerLens(config())
    encoded = reader.encode((affine_relu_records(target()),))
    query = LensQuery(torch.ones(1, 3))
    for unsupported in (
        replace(query, task="unknown"),
        replace(query, scope="set"),
        replace(query, dose=torch.zeros(1)),
        replace(query, schema="future"),
        replace(query, measurement_bundle=("baseline_margin",)),
    ):
        assert reader.predict(encoded, unsupported).status == "unsupported"
    roles = role_query(query.cases, ((0, 900),))
    assert reader.predict(encoded, roles).status == "unsupported"
    with pytest.raises(ValueError, match="positive"):
        reader.predict(encoded, query, chunk_size=0)


def test_uncached_detached_encoding_is_not_a_training_shortcut():
    reader = TransformerLens(config())
    records = affine_relu_records(target())
    with torch.no_grad():
        encoded = reader.encode((records,))
    with pytest.raises(ValueError, match="encoder gradients"):
        reader.predict(encoded, LensQuery(torch.ones(1, 3)))


def test_target_content_and_depth_invalidate_encoding_cache():
    reader = TransformerLens(config()).eval()
    model = target()
    cache = {}
    with torch.no_grad():
        original = reader.encode((affine_relu_records(model),), cache)
        model[0].weight.add_(0.5)
        changed = reader.encode((affine_relu_records(model),), cache)
        deeper = affine_relu_records(
            nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
        )
        extended = reader.encode((deeper,), cache)
        assert len({original.cache_key, changed.cache_key, extended.cache_key}) == 3
        assert not torch.equal(original.states, changed.states)


def test_future_query_shapes_are_representable_but_not_claimed_supported():
    reader = TransformerLens(config())
    records = affine_relu_records(target())
    encoded = reader.encode((records,))
    future = LensQuery(
        mode="parameter_reconstruction",
        scope="parameter",
        parameter_references=(ParameterLocation(0, "bias", (4,), (1,)),),
    )
    assert reader.predict(encoded, future).status == "unsupported"
    grouped = LensQuery(torch.ones(1, 3), scope="unit_set", unit_sets=(((0, 1), (0, 2)),))
    assert reader.predict(encoded, grouped).status == "unsupported"
    with pytest.raises(ValueError, match="schema"):
        reader.encode((replace(records, schema="future-target-v2"),))


def test_common_binding_retains_neuron_correspondence_with_equal_biases():
    torch.manual_seed(19)
    model = target()
    with torch.no_grad():
        model[0].bias.zero_()
        model[0].weight.copy_(
            torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        )
        model[2].weight.fill_(1)
        model[2].bias.zero_()
    reader = TransformerLens(config(relation_conditioned=False))
    encoded = reader.encode((affine_relu_records(model),))
    # Equal biases and equal incoming sums must not erase distinct source assignments.
    assert not torch.allclose(encoded.states[0, 3], encoded.states[0, 4])


def test_dedicated_boundary_masks_follow_roles_and_edge_incidence():
    records = affine_relu_records(target())
    reader = TransformerLens(dedicated_config())
    encoded = reader.encode((records,))
    input_mask, output_mask = boundary_attention_masks(
        encoded.tokens[0], encoded.valid[0]
    )

    # Three inputs, four hidden units, two outputs, then row-major edge tokens.
    assert torch.where(input_mask)[0].tolist() == list(range(3)) + list(range(9, 21))
    assert torch.where(output_mask)[0].tolist() == list(range(7, 9)) + list(range(21, 29))
    assert not input_mask[21:].logical_and(output_mask[21:]).any()


def test_dedicated_boundary_masks_overlap_for_no_hidden_layer_edges():
    records = affine_relu_records(nn.Sequential(nn.Linear(3, 2)))
    reader = TransformerLens(dedicated_config())
    encoded = reader.encode((records,))
    input_mask, output_mask = boundary_attention_masks(
        encoded.tokens[0], encoded.valid[0]
    )
    edge_start = encoded.tokens[0].units

    assert input_mask[edge_start:edge_start + 6].all()
    assert output_mask[edge_start:edge_start + 6].all()
    assert (input_mask & output_mask).sum().item() == 6


def test_dedicated_attention_preserves_legacy_checkpoint_function_and_state_names():
    torch.manual_seed(41)
    config_legacy = config()
    legacy = TransformerLens(config_legacy).eval()
    expanded = TransformerLens(DedicatedBoundaryReaderConfig(**asdict(config_legacy))).eval()
    loaded = expanded.load_state_dict(legacy.state_dict(), strict=False)
    assert not loaded.unexpected_keys
    assert set(loaded.missing_keys) == {
        name for name, _ in expanded.named_parameters()
        if name.startswith(("input_attention.", "output_attention.",
                            "input_residual.", "output_residual."))
    }
    assert all(parameter.requires_grad for parameter in expanded.parameters())

    records = affine_relu_records(target())
    cases = torch.randn(9, 3)
    for query in (LensQuery(cases), role_query(cases)):
        original = legacy.predict(legacy.encode((records,)), query)
        matched = expanded.predict(expanded.encode((records,)), query)
        torch.testing.assert_close(matched.baseline_logits, original.baseline_logits,
                                   rtol=1e-5, atol=1e-6) if query.mode == "baseline" else None
        torch.testing.assert_close(matched.values, original.values, rtol=1e-5, atol=1e-6)


def test_dedicated_attention_staged_gradient_flow_reaches_attention_after_residual_update():
    torch.manual_seed(42)
    reader = TransformerLens(dedicated_config())
    optimizer = torch.optim.AdamW(reader.parameters(), lr=1e-3)
    records = affine_relu_records(target())
    query = LensQuery(torch.randn(5, 3))

    reader.predict(reader.encode((records,)), query).values.square().mean().backward()
    assert reader.input_residual.weight.grad.abs().sum() > 0
    assert reader.output_residual.weight.grad.abs().sum() > 0
    assert reader.input_attention.in_proj_weight.grad.abs().sum() == 0
    assert reader.output_attention.in_proj_weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    reader.predict(reader.encode((records,)), query).values.square().mean().backward()
    assert reader.input_attention.in_proj_weight.grad.abs().sum() > 0
    assert reader.output_attention.in_proj_weight.grad.abs().sum() > 0


def test_dedicated_attention_masks_exclude_padding_and_missing_boundary_is_explicit():
    reader = TransformerLens(dedicated_config())
    small = affine_relu_records(nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 2)))
    large = affine_relu_records(
        nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 2))
    )
    encoded = reader.encode((small, large))
    input_mask, output_mask = boundary_attention_masks(
        encoded.tokens[0], encoded.valid[0]
    )
    assert not input_mask[encoded.valid[0].logical_not()].any()
    assert not output_mask[encoded.valid[0].logical_not()].any()

    missing_features = encoded.tokens[0].features.clone()
    missing_features[:, 2] = 0
    missing_tokens = replace(encoded.tokens[0], features=missing_features)
    missing = replace(encoded, tokens=(missing_tokens, encoded.tokens[1]))
    response = reader.predict(
        missing, LensQuery(torch.randn(1, 3), target_indices=torch.tensor([0]))
    )
    assert response.status == "unsupported"
    assert response.reason == "Missing required input-unit region"

    malformed_sources = encoded.tokens[0].sources.clone()
    malformed_sources[-1] = encoded.tokens[0].features.shape[0]
    malformed_tokens = replace(encoded.tokens[0], sources=malformed_sources)
    malformed = replace(encoded, tokens=(malformed_tokens, encoded.tokens[1]))
    response = reader.predict(
        malformed, LensQuery(torch.randn(1, 3), target_indices=torch.tensor([0]))
    )
    assert response.status == "unsupported"
    assert response.reason == "Boundary edge has missing endpoint incidence"
