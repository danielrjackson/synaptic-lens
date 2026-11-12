"""CPU-only, hash-bound inference for selected historical milestone checkpoints.

No downloads and no unrestricted pickle fallback. A supplied hash establishes
file identity, not trust: obtain files and expected hashes from trusted sources.
"""
from pathlib import Path
import hashlib
import io
import json
import re
import torch
from .observer import Lens, layer_statistics, predict_lens
from .transformer_lens import DedicatedBoundaryReaderConfig, TransformerLens

CHECKPOINTS = {
    "mnist-raw": "6011ca7cf3df54282eecef05aebc057e17fcc4a846ba61683e988c07b02a1111",
    "mnist-statistics": "797a2834e33ed27835cb65054d1e4d6b44cf48c09ea3b125f96cf4d9b6c843c4",
    "syn53-control": "9dd0baffc515d481fafb62a33a1189e1526c21f1648515755da7caa26a357bd9",
    "syn53-regional": "119c4139203a19d2a9a134ebc568e8b2b5f534c34450b9cb0737a0b18713eb85",
}
MNIST_SCHEMA_SHA256 = "36e6ab0e0a61bf01246a6c303fc7d0a96ef8cc916fec4ce297412908a042a1c4"


def verified_bytes(path, sha256, *, max_bytes=32 * 1024 * 1024):
    """Bound reads and hash the exact bytes later parsed, avoiding a second read."""
    if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
        raise ValueError("Expected SHA-256 must be 64 lowercase hex characters")
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise ValueError("Expected a regular non-symlink file")
    with p.open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Artifact exceeds size limit")
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Artifact SHA-256 mismatch")
    return data


def load_verified(path, sha256, *, legacy_torch_version=False):
    """Restricted CPU loading; only legacy target metadata permits TorchVersion."""
    data = verified_bytes(path, sha256)
    # Reviewed PyTorch str subclass, with no constructor/deserialization hooks.
    # Context scoped: no persistent allowlist and no arbitrary checkpoint globals.
    from torch.torch_version import TorchVersion
    allowed = [TorchVersion] if legacy_torch_version else []
    with torch.serialization.safe_globals(allowed):
        return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def load_milestone(name, path):
    """Return a frozen inference model (SYN53) or normalized observer bundle."""
    if name not in CHECKPOINTS:
        raise ValueError("Unknown milestone")
    saved = load_verified(path, CHECKPOINTS[name])
    if name.startswith("syn53-"):
        model = TransformerLens(DedicatedBoundaryReaderConfig(**saved["model_config"]))
        model.load_state_dict(saved["model"], strict=True)
    else:
        if saved.get("schema") != "mnist-direct-weight-lens-bundle-v1":
            raise ValueError("Unsupported observer bundle")
        expected = 19658 if name == "mnist-raw" else 192
        if saved["features"] != expected:
            raise ValueError("Observer feature count mismatch")
        model = Lens(expected)
        model.load_state_dict(saved["state_dict"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite model weights")
    model.eval().requires_grad_(False)
    return model if name.startswith("syn53-") else saved


def flatten_target(state, schema):
    """Reconstruct the original ordered MNIST target-parameter feature vector."""
    rows = schema["parameters"]
    if schema.get("parameter_count") != 19658 or len(rows) != 32:
        raise ValueError("Unsupported MNIST coordinate layout")
    pieces, layout, names, cursor = [], [], set(), 0
    for row in rows:
        name = row["parameter_name"]
        start, stop = row["raw_flat_index_range"]
        if (name in names or start != cursor or stop <= start
                or stop - start != row["coordinate_count"]
                or row["dtype"] != "torch.float32"):
            raise ValueError("Invalid coordinate coverage")
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                or list(value.shape) != row["shape"] or value.numel() != stop - start
                or not torch.isfinite(value).all()):
            raise ValueError("Target tensor does not match schema")
        pieces.append(value.detach().cpu().reshape(-1))
        layout.append({"start": start, "stop": stop})
        cursor = stop
        names.add(name)
    if cursor != 19658 or set(state) != names:
        raise ValueError("Incomplete coordinate coverage")
    return torch.cat(pieces).unsqueeze(0), layout


def predict_mnist(name, bundle, target_path, target_sha256, schema_path):
    """Predict scalar mean CE from target weights, never from images or labels."""
    if name not in ("mnist-raw", "mnist-statistics"):
        raise ValueError("Expected MNIST milestone")
    schema = json.loads(verified_bytes(schema_path, MNIST_SCHEMA_SHA256, max_bytes=1024*1024))
    target = load_verified(target_path, target_sha256, legacy_torch_version=True)
    if target.get("coordinate_schema_sha256") != schema["schema_sha256"]:
        raise ValueError("Target/schema identity mismatch")
    x, layout = flatten_target(target["model_state"], schema)
    if name == "mnist-statistics":
        x = layer_statistics(x, layout)
    with torch.inference_mode():
        prediction = predict_lens(bundle, x)
    if not torch.isfinite(prediction).all():
        raise ValueError("Nonfinite prediction")
    return float(prediction.item())
