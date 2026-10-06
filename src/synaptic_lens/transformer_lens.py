"""Full-detail affine target reader with explicit query and coverage contracts.

Both attention variants share token features, endpoint binding and readout.
These interfaces describe supported computation, not demonstrated learned skills.
"""

import hashlib
from dataclasses import asdict, dataclass

import torch
from torch import nn

from .target_records import AffineRecords, ParameterLocation

TOKEN_SCHEMA = "affine-full-detail-features-v1"
RELATIONS = (
    "edge_to_source",
    "source_to_edge",
    "edge_to_destination",
    "destination_to_edge",
    "same_operation",
)


@dataclass(frozen=True)
class ReaderConfig:
    case_width: int = 20
    width: int = 128
    heads: int = 4
    blocks: int = 2
    feedforward_multiplier: int = 4
    max_tokens: int = 4096
    relation_conditioned: bool = False

    def __post_init__(self):
        if (
            min(
                self.case_width,
                self.width,
                self.heads,
                self.blocks,
                self.feedforward_multiplier,
                self.max_tokens,
            )
            <= 0
        ):
            raise ValueError("Reader dimensions and budgets must be positive")
        if self.width % self.heads:
            raise ValueError("Reader width must be divisible by head count")


@dataclass(frozen=True)
class TargetTokens:
    features: torch.Tensor
    sources: torch.Tensor
    destinations: torch.Tensor
    owners: torch.Tensor
    units: int
    widths: tuple[int, ...]
    complete: bool
    target_identity: str
    numeric_parameters: int
    observed_parameters: int


def tokenize_target(records: AffineRecords, *, device, dtype, max_tokens: int) -> TargetTokens:
    """All units then row-major scalar connections; no hidden-index features.

    Twelve features: unit/edge/input/output, bias-exists, value-present,
    numeric value, ReLU, layer fraction, semantic input/output coordinate,
    log layer width. Numeric features are calculated AFTER hiding missing values.
    """
    if records.schema != "affine-relu-scalar-v1":
        raise ValueError("Unsupported target record schema")
    if records.record_count > max_tokens:
        raise ValueError("Target exceeds full-detail token budget; no truncation")
    offsets = [0]
    for width in records.widths:
        offsets.append(offsets[-1] + width)
    units = offsets[-1]
    features = torch.zeros(records.record_count, 12, device=device, dtype=dtype)
    sources = torch.full((records.record_count,), -1, device=device, dtype=torch.long)
    destinations = sources.clone()
    owners = sources.clone()
    depth = len(records.weights)
    for layer, width in enumerate(records.widths):
        rows = slice(offsets[layer], offsets[layer + 1])
        features[rows, 0] = 1
        features[rows, 2] = float(layer == 0)
        features[rows, 3] = float(layer == depth)
        features[rows, 8] = layer / depth
        features[rows, 11] = torch.tensor(float(width), device=device, dtype=dtype).log1p()
        if layer == 0 or layer == depth:
            coordinate = torch.arange(width, device=device, dtype=dtype) / max(1, width - 1)
            features[rows, 9 if layer == 0 else 10] = coordinate
        if layer:
            owners[rows] = layer - 1
            features[rows, 7] = float(records.relu[layer - 1])
            bias = records.biases[layer - 1]
            if bias is not None:
                present = records.bias_present[layer - 1].to(device=device)
                features[rows, 4] = 1
                features[rows, 5] = present.to(dtype=dtype)
                features[rows, 6] = torch.where(present, bias.to(device=device, dtype=dtype), 0)
    start = units
    for layer, weight in enumerate(records.weights):
        count = weight.numel()
        rows = slice(start, start + count)
        present = records.weight_present[layer].to(device=device).flatten()
        features[rows, 1] = 1
        features[rows, 5] = present.to(dtype=dtype)
        features[rows, 6] = torch.where(present, weight.to(device=device, dtype=dtype).flatten(), 0)
        features[rows, 8] = (layer + 1) / depth
        owners[rows] = layer
        sources[rows] = (
            torch.arange(records.widths[layer], device=device).repeat(records.widths[layer + 1])
            + offsets[layer]
        )
        destinations[rows] = (
            torch.arange(records.widths[layer + 1], device=device).repeat_interleave(
                records.widths[layer]
            )
            + offsets[layer + 1]
        )
        start += count
    observed = sum(int(p.sum()) for p in records.weight_present) + sum(
        int(p.sum()) for p in records.bias_present if p is not None
    )
    return TargetTokens(
        features,
        sources,
        destinations,
        owners,
        units,
        records.widths,
        records.complete,
        records.identity(),
        records.parameter_count,
        observed,
    )


def relation_features(tokens: TargetTokens) -> torch.Tensor:
    """Directed incidence and same affine operation, without numeric labels."""
    count = tokens.features.shape[0]
    relations = torch.zeros(
        len(RELATIONS), count, count, dtype=torch.bool, device=tokens.features.device
    )
    edges = torch.arange(tokens.units, count, device=tokens.features.device)
    source, destination = tokens.sources[edges], tokens.destinations[edges]
    relations[0, edges, source] = True
    relations[1, source, edges] = True
    relations[2, edges, destination] = True
    relations[3, destination, edges] = True
    owners = tokens.owners
    relations[4] = (owners[:, None] == owners[None, :]) & (owners[:, None] >= 0)
    return relations


class ReaderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.norm_attention = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.norm_feedforward = nn.LayerNorm(config.width)
        self.feedforward = nn.Sequential(
            nn.Linear(config.width, config.width * config.feedforward_multiplier),
            nn.GELU(),
            nn.Linear(config.width * config.feedforward_multiplier, config.width),
        )
        self.relation_bias = (
            nn.Parameter(torch.zeros(config.heads, len(RELATIONS)))
            if config.relation_conditioned
            else None
        )

    def forward(self, states, valid, relations):
        batch, count, width = states.shape
        qkv = (
            self.qkv(self.norm_attention(states))
            .reshape(batch, count, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        mask = states.new_zeros(batch, self.heads, count, count)
        if self.relation_bias is not None:
            mask = mask + torch.einsum(
                "hr,brij->bhij", self.relation_bias, relations.to(dtype=states.dtype)
            )
        mask = mask.masked_fill(~valid[:, None, None, :], float("-inf"))
        attended = nn.functional.scaled_dot_product_attention(*qkv.unbind(0), attn_mask=mask)
        states = states + self.projection(attended.transpose(1, 2).reshape(batch, count, width))
        states = states + self.feedforward(self.norm_feedforward(states))
        return states.masked_fill(~valid[..., None], 0)


@dataclass(frozen=True)
class EncodedTargets:
    states: torch.Tensor
    valid: torch.Tensor
    tokens: tuple[TargetTokens, ...]
    encoder_versions: tuple
    cache_key: str | None = None


@dataclass(frozen=True)
class QueryHint:
    kind: str
    values: torch.Tensor
    available: torch.Tensor


@dataclass(frozen=True)
class LensQuery:
    cases: torch.Tensor | None = None
    task: str = "addition-v1"
    mode: str = "baseline"
    unit_references: tuple[tuple[int, int], ...] = ()
    directions: torch.Tensor | None = None
    maxima: torch.Tensor | None = None
    target_indices: torch.Tensor | None = None
    scope: str = "whole_target"
    dose: torch.Tensor | None = None
    case_ids: tuple[str, ...] = ()
    schema: str = "lens-query-v1"
    operation: str = "none"
    measurement_bundle: tuple[str, ...] = ()
    parameter_references: tuple[ParameterLocation, ...] = ()
    unit_sets: tuple[tuple[tuple[int, int], ...], ...] = ()
    operation_settings: tuple[tuple[str, float], ...] = ()
    hints: tuple[QueryHint, ...] = ()


@dataclass(frozen=True)
class LensResponse:
    status: str
    values: torch.Tensor | None
    measurements: tuple[str, ...]
    reason: str | None = None
    uncertainty: torch.Tensor | None = None
    uncertainty_status: str = "unvalidated"
    learned_capability: str = "unvalidated"
    information: str = "weights_and_declared_maximum"
    coverage: tuple[tuple[int, int], ...] = ()


class TransformerLens(nn.Module):
    def __init__(self, config: ReaderConfig):
        super().__init__()
        self.config = config
        self.feature_projection = nn.Linear(12, config.width)
        self.source_binding = nn.Linear(config.width, config.width, bias=False)
        self.destination_binding = nn.Linear(config.width, config.width, bias=False)
        self.incoming_binding = nn.Linear(config.width, config.width, bias=False)
        self.outgoing_binding = nn.Linear(config.width, config.width, bias=False)
        self.blocks = nn.ModuleList(ReaderBlock(config) for _ in range(config.blocks))
        self.encoder_norm = nn.LayerNorm(config.width)
        self.query_projection = nn.Linear(config.case_width + 5, config.width)
        self.selected_projection = nn.Linear(config.width, config.width, bias=False)
        self.readout_attention = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(config.width, config.width), nn.GELU())
        self.baseline_head = nn.Linear(config.width, 2)
        self.intervention_head = nn.Linear(config.width, 3)

    def _encoder_parameters(self):
        prefixes = (
            "feature_projection.",
            "source_binding.",
            "destination_binding.",
            "incoming_binding.",
            "outgoing_binding.",
            "blocks.",
            "encoder_norm.",
        )
        return [(name, p) for name, p in self.named_parameters() if name.startswith(prefixes)]

    def _versions(self):
        return tuple((name, id(p), p._version) for name, p in self._encoder_parameters())

    def encoder_identity(self):
        digest = hashlib.sha256(repr((TOKEN_SCHEMA, asdict(self.config))).encode())
        for name, p in self._encoder_parameters():
            value = p.detach().cpu().contiguous()
            digest.update(repr((name, str(value.dtype), tuple(value.shape))).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def encode(self, records: tuple[AffineRecords, ...], cache=None) -> EncodedTargets:
        if not records:
            raise ValueError("Empty target batch")
        parameter = self.feature_projection.weight
        if any(r.widths[0] != self.config.case_width for r in records):
            raise ValueError("Target input width differs from configured case semantics")
        key = None
        if cache is not None:
            if self.training or torch.is_grad_enabled():
                raise ValueError(
                    "Detached inference cache cannot be used during training/gradients"
                )
            key = hashlib.sha256(
                repr(
                    (
                        self.encoder_identity(),
                        tuple(r.identity() for r in records),
                        str(parameter.device),
                    )
                ).encode()
            ).hexdigest()
            if key in cache:
                saved = cache[key]
                return EncodedTargets(
                    saved.states, saved.valid, saved.tokens, self._versions(), key
                )
        tokens = tuple(
            tokenize_target(
                r, device=parameter.device, dtype=parameter.dtype, max_tokens=self.config.max_tokens
            )
            for r in records
        )
        count = max(t.features.shape[0] for t in tokens)
        features = parameter.new_zeros(len(tokens), count, 12)
        valid = torch.zeros(len(tokens), count, device=parameter.device, dtype=torch.bool)
        relations = (
            torch.zeros(
                len(tokens), len(RELATIONS), count, count, device=parameter.device, dtype=torch.bool
            )
            if self.config.relation_conditioned
            else None
        )
        for index, token in enumerate(tokens):
            size = token.features.shape[0]
            features[index, :size] = token.features
            valid[index, :size] = True
            if self.config.relation_conditioned:
                relations[index, :, :size, :size] = relation_features(token)
        states = self.feature_projection(features)
        bound = states.clone()
        for index, token in enumerate(tokens):
            size = token.features.shape[0]
            edges = slice(token.units, size)
            edge_context = nn.functional.gelu(
                states[index, edges]
                + self.source_binding(states[index, token.sources[edges]])
                + self.destination_binding(states[index, token.destinations[edges]])
            )
            incoming = states.new_zeros(token.units, self.config.width).index_add(
                0, token.destinations[edges], edge_context
            )
            outgoing = states.new_zeros(token.units, self.config.width).index_add(
                0, token.sources[edges], edge_context
            )
            in_degree = torch.bincount(token.destinations[edges], minlength=token.units).clamp_min(
                1
            )
            out_degree = torch.bincount(token.sources[edges], minlength=token.units).clamp_min(1)
            unit_context = (
                states[index, : token.units]
                + self.incoming_binding(incoming / in_degree[:, None])
                + self.outgoing_binding(outgoing / out_degree[:, None])
            )
            bound[index, : token.units] = unit_context
            bound[index, edges] = (
                states[index, edges]
                + self.source_binding(unit_context[token.sources[edges]])
                + self.destination_binding(unit_context[token.destinations[edges]])
            )
        for block in self.blocks:
            bound = block(bound, valid, relations)
        encoded = EncodedTargets(self.encoder_norm(bound), valid, tokens, self._versions(), key)
        if cache is not None:
            encoded = EncodedTargets(encoded.states.detach(), valid, tokens, self._versions(), key)
            cache[key] = encoded
        return encoded

    def predict(self, encoded: EncodedTargets, query: LensQuery, *, chunk_size=256):
        if encoded.encoder_versions != self._versions():
            raise ValueError("Encoder changed; recompute target encoding")
        if encoded.cache_key is not None and (self.training or torch.is_grad_enabled()):
            raise ValueError("Detached inference encoding cannot supply training gradients")
        if (
            torch.is_grad_enabled()
            and any(p.requires_grad for _, p in self._encoder_parameters())
            and not encoded.states.requires_grad
        ):
            raise ValueError("Detached target states cannot supply encoder gradients")
        if (
            query.schema != "lens-query-v1"
            or query.task != "addition-v1"
            or query.mode not in ("baseline", "intervention")
        ):
            return LensResponse("unsupported", None, (), "Unimplemented task or query mode")
        scope = "whole_target" if query.mode == "baseline" else "individual_unit"
        operation = "none" if query.mode == "baseline" else "clamp"
        if query.scope != scope or query.operation != operation or query.dose is not None:
            return LensResponse("unsupported", None, (), "Unimplemented scope or information level")
        if query.parameter_references or query.unit_sets or query.operation_settings or query.hints:
            return LensResponse("unsupported", None, (), "Unimplemented query extension")
        if chunk_size <= 0:
            raise ValueError("Query chunk size must be positive")
        if query.cases is None:
            raise ValueError("Behavior prediction requires semantic case encodings")
        cases = query.cases.to(device=encoded.states.device, dtype=encoded.states.dtype)
        if cases.ndim != 2 or cases.shape[1] != self.config.case_width or not len(cases):
            raise ValueError("Invalid case encodings")
        if not bool(torch.isfinite(cases).all()):
            raise ValueError("Nonfinite case encoding")
        count = len(cases)
        if query.case_ids and len(query.case_ids) != count:
            raise ValueError("Case identities do not match case encodings")
        if query.target_indices is None:
            indices = torch.zeros(count, device=cases.device, dtype=torch.long)
            if len(encoded.tokens) != 1:
                raise ValueError("Batched targets require explicit query target indices")
        else:
            indices = query.target_indices.to(device=cases.device)
            if (
                indices.dtype != torch.long
                or indices.shape != (count,)
                or bool(((indices < 0) | (indices >= len(encoded.tokens))).any())
            ):
                raise ValueError("Invalid query target indices")
        coverage = tuple(
            (encoded.tokens[i].observed_parameters, encoded.tokens[i].numeric_parameters)
            for i in indices.unique().tolist()
        )
        if any(not encoded.tokens[i].complete for i in indices.unique().tolist()):
            return LensResponse(
                "partial", None, (), "Target numeric coverage is incomplete", coverage=coverage
            )
        controls = cases.new_zeros(count, 5)
        selected = encoded.states.new_zeros(count, self.config.width)
        if query.mode == "baseline":
            if query.unit_references or query.directions is not None or query.maxima is not None:
                raise ValueError("Baseline query contains intervention settings")
            controls[:, 0] = 1
            measurements = ("pass_probability", "correct_answer_probability")
        else:
            if (
                len(query.unit_references) != count
                or query.directions is None
                or query.maxima is None
            ):
                raise ValueError("Incomplete intervention settings")
            directions, maxima = query.directions.to(cases.device), query.maxima.to(cases.device)
            if (
                directions.shape != (count,)
                or maxima.shape != (count,)
                or not bool(((directions == -1) | (directions == 1)).all())
                or not bool(torch.isfinite(maxima).all())
                or bool((maxima < 0).any())
            ):
                raise ValueError("Invalid direction or natural maximum")
            positions = []
            for target, (layer, unit) in zip(indices.tolist(), query.unit_references):
                widths = encoded.tokens[target].widths
                if not 0 <= layer < len(widths) - 2 or not 0 <= unit < widths[layer + 1]:
                    return LensResponse("unsupported", None, (), "Unknown hidden unit reference")
                positions.append(sum(widths[: layer + 1]) + unit)
            selected = encoded.states[indices, torch.tensor(positions, device=cases.device)]
            controls[:, 1] = 1
            controls[:, 2] = (directions == -1).to(cases.dtype)
            controls[:, 3] = (directions == 1).to(cases.dtype)
            controls[:, 4] = maxima.to(cases.dtype)
            measurements = ("probability_change", "pass_change", "margin_change")
        if query.measurement_bundle and query.measurement_bundle != measurements:
            return LensResponse("unsupported", None, (), "Unimplemented measurement selector")
        queries = self.query_projection(torch.cat((cases, controls), dim=-1))
        queries = queries + self.selected_projection(selected)
        values = []
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            readouts = queries.new_zeros(stop - start, self.config.width)
            target_indices = indices[start:stop]
            for target in target_indices.unique().tolist():
                local = torch.where(target_indices == target)[0]
                attended, _ = self.readout_attention(
                    queries[start:stop][local][None, :, :],
                    encoded.states[target : target + 1],
                    encoded.states[target : target + 1],
                    key_padding_mask=~encoded.valid[target : target + 1],
                    need_weights=False,
                )
                readouts[local] = attended[0]
            result = self.readout(queries[start:stop] + readouts)
            values.append(
                self.baseline_head(result).sigmoid()
                if query.mode == "baseline"
                else self.intervention_head(result)
            )
        return LensResponse(
            "supported",
            torch.cat(values),
            measurements,
            coverage=coverage,
            information="weights_only"
            if query.mode == "baseline"
            else "weights_and_declared_maximum",
        )


def matched_readers(config: ReaderConfig, seed=7):
    """Copy all common tensors; zero relation terms give a matched initial function."""
    from dataclasses import replace

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        content = TransformerLens(replace(config, relation_conditioned=False))
        relation = TransformerLens(replace(config, relation_conditioned=True))
        loaded = relation.load_state_dict(content.state_dict(), strict=False)
        expected = {f"blocks.{i}.relation_bias" for i in range(config.blocks)}
        if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
            raise ValueError("Reader pair does not have the declared common parameter structure")
    return content, relation
