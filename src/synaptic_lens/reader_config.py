"""Reader dimensions and validated switches, separate from tensor execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReaderConfig:
    case_width: int = 20
    width: int = 128
    heads: int = 4
    blocks: int = 2
    feedforward_multiplier: int = 4
    max_tokens: int = 4096
    relation_conditioned: bool = False
    dose_input: bool = False

    def __post_init__(self):
        if not isinstance(self.dose_input, bool):
            raise ValueError("Dose input configuration must be boolean")
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
class DedicatedBoundaryReaderConfig(ReaderConfig):
    """Versioned opt-in config; the legacy ReaderConfig serialization stays stable."""

    dedicated_boundary_attention: bool = True
    boundary_attention_mode: str = "regional"

    def __post_init__(self):
        super().__post_init__()
        if (
            not isinstance(self.dedicated_boundary_attention, bool)
            or not self.dedicated_boundary_attention
        ):
            raise ValueError("Dedicated boundary attention must be enabled in the expanded config")
        if self.boundary_attention_mode not in ("regional", "all_valid"):
            raise ValueError("Boundary attention mode must be regional or all_valid")


# Keep existing pickle/checkpoint globals and a single class identity.
ReaderConfig.__module__ = "synaptic_lens.transformer_lens"
DedicatedBoundaryReaderConfig.__module__ = "synaptic_lens.transformer_lens"
