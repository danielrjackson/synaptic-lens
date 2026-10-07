"""Single-scalar masked reconstruction on the reusable affine target encoder.

The local control selects its graph before any endpoint binding. Audit flags
affect cache/query identity but never token features or the learned decoder.
Implemented computation does not establish a trained reconstruction skill.
"""

import hashlib
from dataclasses import dataclass, fields

import torch
from torch import nn

from .target_records import AffineRecords, ParameterLocation, ScalarMaskMetadata
from .transformer_lens import (
    LensQuery,
    LensResponse,
    ReaderConfig,
    TargetTokens,
    TransformerLens,
    tokenize_target,
)

SCALAR_SCHEMA = "single-scalar-readout-v1"


@dataclass(frozen=True)
class ScalarRecords(AffineRecords):
    location: ParameterLocation | None = None
    mask_metadata: ScalarMaskMetadata | None = None

    def identity(self):
        return hashlib.sha256(
            repr((super().identity(), SCALAR_SCHEMA, self.location, self.mask_metadata)).encode()
        ).hexdigest()


def single_missing_location(records: AffineRecords) -> ParameterLocation:
    missing = []
    for name, values, flags in (
        ("weight", records.weights, records.weight_present),
        ("bias", records.biases, records.bias_present),
    ):
        for owner, (value, present) in enumerate(zip(values, flags)):
            if value is None:
                continue
            for coordinate in (~present).nonzero().tolist():
                missing.append(
                    ParameterLocation(owner, name, tuple(value.shape), tuple(coordinate))
                )
    if len(missing) != 1:
        raise ValueError("Scalar reconstruction requires exactly one missing parameter")
    return missing[0]


def masked_scalar(
    records: AffineRecords,
    location: ParameterLocation,
    metadata: ScalarMaskMetadata = ScalarMaskMetadata(),
) -> ScalarRecords:
    """Keep the supervision label outside this value-blind reader input."""
    if not metadata.missing:
        raise ValueError("Reconstruction metadata must declare a missing scalar")
    masked = records.masked(location)
    if single_missing_location(masked) != location:
        raise ValueError("Declared location differs from missing parameter")
    values = {field.name: getattr(masked, field.name) for field in fields(AffineRecords)}
    return ScalarRecords(**values, location=location, mask_metadata=metadata)


@dataclass(frozen=True)
class ScalarTokens(TargetTokens):
    location: ParameterLocation | None = None
    mask_metadata: ScalarMaskMetadata | None = None
    original_indices: tuple[int, ...] = ()
    context_observed_parameters: int = 0


@dataclass(frozen=True)
class ScalarResponse(LensResponse):
    normalized_values: torch.Tensor | None = None
    context_coverage: tuple[tuple[int, int], ...] = ()


def location_tokens(widths, location):
    """Global correspondence indices; never numeric model features."""
    destination = sum(widths[: location.owner + 1]) + location.coordinate[0]
    if location.name == "bias":
        return destination, None, destination
    source = sum(widths[: location.owner]) + location.coordinate[1]
    previous_edges = sum(a * b for a, b in zip(widths, widths[1 : location.owner + 1]))
    edge = sum(widths) + previous_edges + location.coordinate[0] * widths[location.owner]
    return edge + location.coordinate[1], source, destination


def scalar_type(records, location):
    if len(records.widths) != 3 or any(b is None for b in records.biases):
        raise ValueError("Scalar regimen supports two affine operations with both biases")
    if location.owner not in (0, 1) or location.name not in ("weight", "bias"):
        raise ValueError("Unsupported scalar parameter type")
    return 2 * location.owner + int(location.name == "bias")


def scalar_tokens(records: ScalarRecords, *, device, dtype, max_tokens, local):
    if not isinstance(records, ScalarRecords) or records.mask_metadata is None:
        raise ValueError("Scalar encoding requires declared mask metadata")
    location = single_missing_location(records)
    if location != records.location or not records.mask_metadata.missing:
        raise ValueError("Scalar input mask metadata does not match missing location")
    scalar_type(records, location)
    tokens = tokenize_target(records, device=device, dtype=dtype, max_tokens=max_tokens)
    count = tokens.features.shape[0]
    indices = torch.arange(count, device=device)
    if local:
        _, source, destination = location_tokens(tokens.widths, location)
        query_units = torch.tensor(
            [destination] if source is None else [source, destination], device=device
        )
        edges = torch.arange(tokens.units, count, device=device)
        retained = edges[
            torch.isin(tokens.sources[edges], query_units)
            | torch.isin(tokens.destinations[edges], query_units)
        ]
        units = torch.cat((query_units, tokens.sources[retained], tokens.destinations[retained]))
        units = units.unique(sorted=True)
        indices = torch.cat((units, retained))
        remap = torch.full((tokens.units,), -1, device=device, dtype=torch.long)
        remap[units] = torch.arange(len(units), device=device)
        sources = tokens.sources[indices].clone()
        destinations = tokens.destinations[indices].clone()
        sources[len(units) :] = remap[sources[len(units) :]]
        destinations[len(units) :] = remap[destinations[len(units) :]]
        unit_count = len(units)
    else:
        sources, destinations, unit_count = tokens.sources, tokens.destinations, tokens.units
    visible = tokens.features[indices]
    observed_here = int(
        (visible[:, 5] * (visible[:, 1] + visible[:, 0] * visible[:, 4])).sum().item()
    )
    return ScalarTokens(
        tokens.features[indices],
        sources,
        destinations,
        tokens.owners[indices],
        unit_count,
        tokens.widths,
        tokens.complete,
        tokens.target_identity,
        tokens.numeric_parameters,
        tokens.observed_parameters,
        location,
        records.mask_metadata,
        tuple(indices.tolist()),
        observed_here,
    )


class ScalarLens(TransformerLens):
    def __init__(self, config: ReaderConfig, *, local=False, means=(0.0,) * 4, scales=(1.0,) * 4):
        self.local = bool(local)
        super().__init__(config)
        means = torch.as_tensor(means, dtype=torch.float64).clone()
        scales = torch.as_tensor(scales, dtype=torch.float64).clone()
        if means.shape != (4,) or scales.shape != (4,):
            raise ValueError("Four frozen training-type constants required")
        if not bool(torch.isfinite(means).all() & torch.isfinite(scales).all()):
            raise ValueError("Training constants must be finite")
        if bool((scales <= 0).any()):
            raise ValueError("Training scales must be positive")
        self.register_buffer("type_means", means)
        self.register_buffer("type_scales", scales)

    def _initialize_readout(self):
        """No inactive behavior heads consume the scalar initialization stream."""
        width = self.config.width
        self.task_vector = nn.Parameter(torch.zeros(width))
        self.scalar_head = nn.Sequential(
            nn.Linear(4 * width + 8, width), nn.GELU(), nn.Linear(width, 1)
        )

    def _tokenize(self, records, *, device, dtype):
        return scalar_tokens(
            records,
            device=device,
            dtype=dtype,
            max_tokens=self.config.max_tokens,
            local=self.local,
        )

    def encoder_identity(self):
        return hashlib.sha256(
            repr((super().encoder_identity(), SCALAR_SCHEMA, self.local)).encode()
        ).hexdigest()

    def predict(self, encoded, query: LensQuery, *, chunk_size=256):
        self._validate_encoding(encoded)
        if (
            query.schema != "lens-query-v1"
            or query.task != "addition-v1"
            or query.mode != "reconstruction"
            or query.scope != "single_scalar"
            or query.operation != "predict_missing"
            or query.measurement_bundle != ("parameter_value",)
        ):
            return ScalarResponse("unsupported", None, (), "Unimplemented scalar query")
        if (
            query.cases is not None
            or query.case_ids
            or query.unit_references
            or query.unit_sets
            or query.directions is not None
            or query.maxima is not None
            or query.dose is not None
            or query.hints
            or query.operation_settings
        ):
            raise ValueError("Scalar query contains behavior settings or unimplemented hints")
        count = len(query.parameter_references)
        if not count or len(query.mask_metadata) != count or chunk_size <= 0:
            raise ValueError("Explicit scalar location and mask metadata required per query")
        if query.target_indices is None:
            if len(encoded.tokens) != 1:
                raise ValueError("Batched targets require explicit target indices")
            indices = [0] * count
        else:
            index = query.target_indices
            if index.dtype != torch.long or index.shape != (count,):
                raise ValueError("Invalid scalar target indices")
            indices = index.tolist()
        decoder_inputs, types = [], []
        for target, location, metadata in zip(
            indices, query.parameter_references, query.mask_metadata
        ):
            if not 0 <= target < len(encoded.tokens):
                raise ValueError("Invalid scalar target index")
            token = encoded.tokens[target]
            if not isinstance(token, ScalarTokens):
                raise ValueError("Scalar decoder requires scalar encoding")
            if location != token.location or metadata != token.mask_metadata:
                raise ValueError("Query location/flags differ from encoded mask")
            q, source, destination = location_tokens(token.widths, location)
            lookup = {original: local for local, original in enumerate(token.original_indices)}
            state = encoded.states[target]
            zq, zd = state[lookup[q]], state[lookup[destination]]
            zs = torch.zeros_like(zq) if source is None else state[lookup[source]]
            type_id = 2 * location.owner + int(location.name == "bias")
            type_features = state.new_zeros(4)
            type_features[type_id] = 1
            shape = location.shape
            # The denominator32 is a declared reference scale, not an input limit.
            descriptors = state.new_tensor(
                [
                    shape[0] / 32,
                    shape[1] / 32 if len(shape) == 2 else 0,
                    len(shape) / 2,
                ]
            )
            descriptors = torch.cat((descriptors, token.features[lookup[destination], 7:8]))
            decoder_inputs.append(
                torch.cat((zq, zs, zd, self.task_vector, type_features, descriptors))
            )
            types.append(type_id)
        normalized = torch.cat(
            [
                self.scalar_head(torch.stack(decoder_inputs[start : start + chunk_size]))
                for start in range(0, count, chunk_size)
            ]
        )
        type_ids = torch.tensor(types, device=normalized.device)
        decoded = (
            self.type_means[type_ids, None].to(normalized.dtype)
            + self.type_scales[type_ids, None].to(normalized.dtype) * normalized
        )
        coverage = tuple(
            (encoded.tokens[i].observed_parameters, encoded.tokens[i].numeric_parameters)
            for i in sorted(set(indices))
        )
        return ScalarResponse(
            "supported",
            decoded,
            ("parameter_value",),
            coverage=coverage,
            information="local_masked_weights_and_declared_task"
            if self.local
            else "masked_weights_and_declared_task",
            normalized_values=normalized,
            context_coverage=tuple(
                (
                    encoded.tokens[i].context_observed_parameters,
                    encoded.tokens[i].numeric_parameters,
                )
                for i in sorted(set(indices))
            ),
        )


def matched_scalar_readers(config: ReaderConfig, *, seed=25, means=(0.0,) * 4, scales=(1.0,) * 4):
    from dataclasses import replace

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        content = ScalarLens(
            replace(config, relation_conditioned=False), means=means, scales=scales
        )
        relation = ScalarLens(
            replace(config, relation_conditioned=True), means=means, scales=scales
        )
        local = ScalarLens(
            replace(config, relation_conditioned=True), local=True, means=means, scales=scales
        )
    for reader in (relation, local):
        missing, unexpected = reader.load_state_dict(content.state_dict(), strict=False)
        if unexpected or any(not name.endswith("relation_bias") for name in missing):
            raise ValueError("Matched scalar reader parameter mismatch")
    return {"R": relation, "C": content, "L": local}
