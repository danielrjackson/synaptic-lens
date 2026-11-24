"""Compatibility entry point for the companion factory's SYN88 generator.

Core Lens inference/training imports need no factory dependency. Generation is
provided by the optional ``targets`` extra. Published root IDs are static metadata.
"""

from importlib import import_module

from ._syn88_population import ROOT_SPLITS

AGES = (0, 1, 2, 5, 10, 25, 50, 100)
RULES = ("addition", "alternate")


def _factory():
    try:
        module = import_module("synaptic_lens_target_model_factory.arithmetic.syn88")
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("synaptic_lens_target_model_factory"):
            raise ImportError(
                "SYN88 generation requires the companion target factory; install the synaptic-lens targets extra"
            ) from error
        raise
    if tuple(module.AGES) != AGES or tuple(module.RULES) != RULES:
        raise ValueError("Companion factory uses an incompatible SYN88 protocol")
    return module


def split_roots():
    """Return fresh lists for the fixed published population partitions."""
    return {name: list(roots) for name, roots in ROOT_SPLITS}


def generate_root(root_index, device="cpu"):
    return _factory().generate_root(root_index, device=device)


def main():
    return _factory().main()


if __name__ == "__main__":
    main()
