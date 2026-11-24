import types

import pytest

from synaptic_lens import syn88_targets


def test_existing_entry_points_delegate_to_one_factory(monkeypatch):
    calls = []
    module = types.SimpleNamespace(
        AGES=syn88_targets.AGES,
        RULES=syn88_targets.RULES,
        generate_root=lambda root, device: (
            calls.append((root, device)) or {"root": root}
        ),
        main=lambda: "factory-cli",
    )
    monkeypatch.setattr(syn88_targets, "import_module", lambda _: module)
    assert syn88_targets.generate_root(2, "cpu") == {"root": 2}
    assert calls == [(2, "cpu")] and syn88_targets.main() == "factory-cli"


def test_missing_factory_is_explicit_and_does_not_break_lens_imports(monkeypatch):
    def missing(_):
        raise ModuleNotFoundError(
            "missing companion", name="synaptic_lens_target_model_factory"
        )

    monkeypatch.setattr(syn88_targets, "import_module", missing)
    with pytest.raises(ImportError, match="companion target factory"):
        syn88_targets.generate_root(0)
    roots = syn88_targets.split_roots()
    assert [len(roots[k]) for k in ("train", "validation", "test")] == [1536, 256, 512]
    assert sorted(sum(roots.values(), [])) == list(range(2304))
    roots["train"].clear()
    assert len(syn88_targets.split_roots()["train"]) == 1536

    from synaptic_lens.trajectory_data import TrajectoryPopulation

    assert TrajectoryPopulation is not None


def test_protocol_drift_rejected_before_generation(monkeypatch):
    monkeypatch.setattr(
        syn88_targets,
        "import_module",
        lambda _: types.SimpleNamespace(AGES=(1,), RULES=("other",)),
    )
    with pytest.raises(ValueError, match="incompatible SYN88 protocol"):
        syn88_targets.generate_root(0)
