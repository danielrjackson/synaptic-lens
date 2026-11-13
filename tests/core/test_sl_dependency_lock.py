"""Lock checks must reject stale or incomplete identities before installing."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def checker(tmp_path):
    spec = importlib.util.spec_from_file_location("lock_check", ROOT / "scripts/locked.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    req = tmp_path / "requirements"
    req.mkdir()
    for source in (ROOT / "requirements").iterdir():
        if source.is_file():
            (req / source.name).write_bytes(source.read_bytes())
    (tmp_path / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    module.ROOT = tmp_path
    module.REQ = req
    return module


def rebind(module, name):
    path = module.REQ / "integrity.json"
    manifest = json.loads(path.read_text())
    manifest["sha256"][name] = hashlib.sha256((module.ROOT / name).read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))


def test_lock_profile_valid_and_pyproject_newline_independent(tmp_path):
    module = checker(tmp_path)
    module.check()
    p = tmp_path / "pyproject.toml"
    p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
    module.check()


def test_lock_changed_archive_hash_is_rejected(tmp_path):
    module = checker(tmp_path)
    p = module.REQ / "windows-cpu.txt"
    p.write_text(p.read_text().replace("--hash=sha256:", "--hash=sha256:0", 1))
    with pytest.raises(ValueError, match="integrity mismatch"):
        module.check()


def test_lock_requires_hash_even_if_manifest_is_refreshed(tmp_path):
    module = checker(tmp_path)
    p = module.REQ / "windows-cpu.txt"
    p.write_text("pytest==9.2.0\n")
    rebind(module, "requirements/windows-cpu.txt")
    with pytest.raises(ValueError, match="SHA-256"):
        module.check()


def test_lock_missing_transitive_pin_is_rejected(tmp_path):
    module = checker(tmp_path)
    p = module.REQ / "windows-cpu.in"
    p.write_text(p.read_text().replace("filelock==4.2.10\n", ""))
    rebind(module, "requirements/windows-cpu.in")
    with pytest.raises(ValueError, match="pins and resolved lock differ"):
        module.check()


def test_locked_run_uses_installed_package():
    name = os.environ.get("SL_LOCKED_PACKAGE")
    if name:
        package = __import__(name)
        assert Path(package.__file__).is_relative_to(Path(sys.prefix))


@pytest.mark.parametrize("change", ["repository", "revision", "dirty", "lock"])
def test_companion_rejects_wrong_identity_before_build(tmp_path, monkeypatch, change):
    module = checker(tmp_path)
    pin = json.loads((module.REQ / "companion.json").read_text())
    companion = tmp_path / "factory"
    (companion / "requirements").mkdir(parents=True)
    (companion / "requirements/integrity.json").write_bytes(b"wrong identity")
    values = {("remote", "get-url", "origin"): pin["repository"],
              ("rev-parse", "HEAD"): pin["commit"],
              ("rev-parse", "HEAD^{tree}"): pin["tree"],
              ("status", "--porcelain"): ""}
    if change == "repository":
        values[("remote", "get-url", "origin")] = "https://example.invalid/other.git"
    elif change == "revision":
        values[("rev-parse", "HEAD")] = "0" * 40
    elif change == "dirty":
        values[("status", "--porcelain")] = " M changed.py"
    monkeypatch.setattr(module.subprocess, "check_output", lambda command, **kwargs: values[tuple(command[3:])])
    monkeypatch.setattr(module, "run", lambda *args, **kwargs: pytest.fail("Unverified companion executed"))
    with pytest.raises(ValueError):
        module.verify_companion(companion)


def test_pinned_build_excludes_ignored_source_and_stale_build(tmp_path, monkeypatch):
    module = checker(tmp_path)
    repo = tmp_path / "factory-source"
    repo.mkdir()
    def git(*args):
        return module.subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    git("init", "--quiet")
    (repo / ".gitignore").write_text("build/\nsrc/ignored.py\n")
    (repo / "src").mkdir()
    (repo / "src/tracked.py").write_text("value = 1\n")
    git("add", ".gitignore", "src/tracked.py")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "Tracked source fixture")
    (repo / "build/lib").mkdir(parents=True)
    (repo / "build/lib/stale.py").write_text("stale = True\n")
    (repo / "src/ignored.py").write_text("poison = True\n")
    assert git("status", "--porcelain") == ""
    (module.REQ / "companion.json").write_text(json.dumps({"commit": git("rev-parse", "HEAD")}))
    seen = []
    def build(python, snapshot):
        assert (snapshot / "src/tracked.py").read_text() == "value = 1\n"
        assert not (snapshot / "src/ignored.py").exists()
        assert not (snapshot / "build").exists()
        seen.append(snapshot)
    monkeypatch.setattr(module, "install_wheel", build)
    module.install_pinned_wheel("unused-python", repo)
    assert len(seen) == 1 and not seen[0].exists()
    assert (repo / "build/lib/stale.py").exists()
