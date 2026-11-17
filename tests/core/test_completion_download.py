import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "download_completion",
    Path(__file__).resolve().parents[2] / "scripts/download_completion.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_pinned_verified_download(tmp_path, monkeypatch):
    payload = tmp_path / "payload"
    payload.write_bytes(b"model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "huggingface": {
                    "repo_id": "test/model",
                    "subfolder": "v1",
                    "revision": "a" * 40,
                },
                "files": {
                    "reader.pt": {
                        "bytes": 5,
                        "sha256": hashlib.sha256(b"model").hexdigest(),
                    }
                },
            }
        )
    )
    monkeypatch.setattr(module, "MANIFEST", manifest)
    calls = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return payload

    result = module.download(tmp_path / "out", local_files_only=True, fetch=fetch)
    assert result["revision"] == "a" * 40
    assert calls[0]["filename"] == "v1/reader.pt"
    assert calls[0]["local_files_only"] is True
    assert (tmp_path / "out/reader.pt").read_bytes() == b"model"
    payload.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="identity"):
        module.download(tmp_path / "bad", fetch=fetch)
    assert not (tmp_path / "bad/reader.pt").exists()


@pytest.mark.parametrize("revision", [None, "main", "", "a" * 39])
def test_reject_unpinned_revision(tmp_path, monkeypatch, revision):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"huggingface": {"revision": revision}, "files": {}})
    )
    monkeypatch.setattr(module, "MANIFEST", manifest)

    def fetch(**kwargs):
        raise AssertionError("Network must not be reached")

    with pytest.raises(ValueError, match="immutable"):
        module.download(tmp_path / "out", fetch=fetch)
