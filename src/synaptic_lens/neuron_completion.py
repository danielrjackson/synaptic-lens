"""Full-neuron completion in original coordinates, without weight-based ordering."""

from dataclasses import dataclass

import torch
from torch import nn

from .checkpoints import load_verified

REPRESENTATION_VERSION = "organic-masked-relu-maxabs-v1"


def normalize_neurons(raw):
    """Pointwise ReLU normalization; no neuron affects another's scale or position."""
    if raw.shape[-2:] != (32, 40) or raw.dtype != torch.float32:
        raise ValueError("Expected float32 [...,32,40] neuron bundles")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("Nonfinite observed neuron parameters")
    values = raw.double().clone()
    scales = values[..., :21].abs().amax(-1)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    values[..., :21] /= scales[..., None]
    values[..., 21:] *= scales[..., None]
    return values.to(raw.dtype), scales


def permute_neurons(values, missing, permutations):
    """Apply supplied value-independent permutations and track the same physical mask."""
    if values.ndim != 3 or values.shape[1:] != (32, 40):
        raise ValueError("Expected [B,32,40] values")
    if permutations.shape != values.shape[:2] or missing.shape != values.shape[:1]:
        raise ValueError("Invalid permutation or mask shape")
    if permutations.dtype != torch.long or missing.dtype != torch.long:
        raise ValueError("Indices must be int64")
    if bool(((permutations < 0) | (permutations >= 32)).any()):
        raise ValueError("Invalid neuron index")
    counts = torch.zeros_like(permutations).scatter_add(
        1, permutations, torch.ones_like(permutations)
    )
    if not bool((counts == 1).all()):
        raise ValueError("Every physical neuron must appear once")
    if bool(((missing < 0) | (missing >= 32)).any()):
        raise ValueError("Invalid missing neuron")
    result = values.gather(1, permutations[..., None].expand(-1, -1, 40))
    new_missing = (permutations == missing[:, None]).long().argmax(-1)
    return result, new_missing


class NoSortLens(nn.Module):
    """Set-transformer reader; optional positional control uses the same capacity."""

    def __init__(self, width=256, blocks=4, heads=8, positional=False):
        super().__init__()
        self.project = nn.Linear(60, width)
        self.positions = (
            nn.Parameter(torch.randn(32, width) * 0.02) if positional else None
        )
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            width * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.body = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 40))

    def forward(self, values, bias, missing):
        rows = torch.arange(len(values), device=values.device)
        visible = values.clone()
        visible[rows, missing] = 0
        observed = torch.ones(
            *values.shape[:2], 1, device=values.device, dtype=values.dtype
        )
        observed[rows, missing] = 0
        tokens = torch.cat((visible, observed, bias[:, None, :].expand(-1, 32, -1)), -1)
        embedded = self.project(tokens)
        if self.positions is not None:
            embedded = embedded + self.positions
        return self.output(self.body(embedded)[rows, missing])


@dataclass
class OrganicCompletionView:
    reader_input: torch.Tensor
    output_bias: torch.Tensor
    original_neurons: torch.Tensor
    original_missing: int
    external_original_scale: torch.Tensor
    representation_version: str = REPRESENTATION_VERSION

    def decode_prediction(self, prediction):
        scale = self.external_original_scale.double()
        if prediction.shape != (40,) or not bool(torch.isfinite(prediction).all()):
            raise ValueError("Expected finite 40-value prediction")
        if not bool(torch.isfinite(scale)) or float(scale) <= 0:
            raise ValueError(
                "Valid externally retained original scale required for decoding"
            )
        replacement = prediction.double().clone()
        replacement[:21] *= scale
        replacement[21:] /= scale
        return replacement.to(self.original_neurons.dtype)

    def reinsert_prediction(self, prediction):
        restored = self.original_neurons.clone()
        restored[self.original_missing] = self.decode_prediction(prediction)
        return restored

    def predict(self, reader):
        if (
            getattr(reader, "representation_version", REPRESENTATION_VERSION)
            != self.representation_version
        ):
            raise ValueError("Reader representation mismatch")
        return self.reinsert_prediction(reader(self.reader_input, self.output_bias))


def prepare_organic_completion(raw, output_bias, missing):
    """Hide first, normalize observed bundles, retain original coordinates and decoder scale.

    Changing the hidden bundle cannot change reader inputs. Its true normalization
    scale is external reconstruction metadata, never a reader input.
    """
    if (
        raw.shape != (32, 40)
        or raw.dtype != torch.float32
        or output_bias.shape != (19,)
        or output_bias.dtype != torch.float32
        or not 0 <= missing < 32
    ):
        raise ValueError("Expected original adder and valid physical missing index")
    observed = raw.clone()
    observed[missing] = 0
    if not bool(torch.isfinite(observed).all()) or not bool(
        torch.isfinite(output_bias).all()
    ):
        raise ValueError("Observed parameters must be finite")
    normalized, _ = normalize_neurons(observed)
    flag = torch.ones(32, 1, dtype=raw.dtype, device=raw.device)
    flag[missing] = 0
    scale = raw[missing, :21].double().abs().max()
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return OrganicCompletionView(
        torch.cat((normalized, flag), -1),
        output_bias.clone(),
        raw.clone(),
        missing,
        scale,
    )


class OrganicCheckpointReader:
    """Actual saved-checkpoint adapter for the no-sort wrapper."""

    representation_version = REPRESENTATION_VERSION

    def __init__(self, checkpoint, sha256, device="cuda"):
        saved = load_verified(checkpoint, sha256)
        config = saved["config"]
        if config.get("representation_version") != REPRESENTATION_VERSION:
            raise ValueError("Unsupported checkpoint representation")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; explicitly choose device='cpu' for CPU inference"
            )
        self.model = (
            NoSortLens(
                config["width"],
                config["blocks"],
                config["heads"],
                positional=saved["positional"],
            )
            .to(device)
            .eval()
        )
        self.model.load_state_dict(saved["state_dict"], strict=True)
        self.model.requires_grad_(False)
        if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            raise ValueError("Nonfinite model weights")
        self.mean, self.std = saved["mean"].to(device), saved["std"].to(device)
        self.bias_mean, self.bias_std = (
            saved["bias_mean"].to(device),
            saved["bias_std"].to(device),
        )
        for value, shape in (
            (self.mean, (1, 1, 40)),
            (self.std, (1, 1, 40)),
            (self.bias_mean, (1, 19)),
            (self.bias_std, (1, 19)),
        ):
            if value.shape != shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid normalization statistics")
        if not bool((self.std > 0).all()) or not bool((self.bias_std > 0).all()):
            raise ValueError("Normalization scales must be positive")

    def __call__(self, masked_view, bias):
        if masked_view.shape != (32, 41) or bias.shape != (19,):
            raise ValueError("Invalid masked input shape")
        flag = masked_view[:, -1]
        if not bool(((flag == 0) | (flag == 1)).all()) or int((flag == 0).sum()) != 1:
            raise ValueError("Exactly one missing neuron required")
        missing = torch.where(flag == 0)[0].to(self.device)
        with torch.no_grad():
            prediction = self.model(
                (masked_view[:, :40].to(self.device)[None] - self.mean) / self.std,
                (bias.to(self.device)[None] - self.bias_mean) / self.bias_std,
                missing,
            )
            result = prediction * self.std.squeeze(0) + self.mean.squeeze(0)
        return result[0].to(masked_view.device)
