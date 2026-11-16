"""Download explicitly selected, hash-verified model exports from Hugging Face.

This script uses the Hub's versioned cache. It never imports downloaded code or
loads checkpoints. See docs/DOWNLOADS.md for installation and examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Callable

CATALOG = Path(__file__).resolve().parents[1] / "docs" / "releases.json"


def _name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Invalid release filename: {value!r}")
    if value.startswith("/") or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError(f"Invalid release filename: {value!r}")
    if PurePosixPath(value).as_posix() != value:
        raise ValueError(f"Invalid release filename: {value!r}")
    return value


def _digest(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


def verify(path: str | Path, expected: str, size: int | None = None) -> Path:
    """Hash all bytes before the caller uses the cached file."""
    # Hub snapshots may contain symlinks to content-addressed blobs.
    # Return the verified regular blob path for strict local checkpoint readers.
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected a regular cached file: {path}")
    expected = _digest(expected)
    if size is not None and path.stat().st_size != size:
        raise ValueError(f"Size mismatch: {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {path}; cache file left unchanged")
    return path


def catalog() -> dict:
    return json.loads(CATALOG.read_text(encoding="utf-8"))


def download_release(
    release_name: str,
    filenames: list[str] | None = None,
    *,
    all_files: bool = False,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    downloader: Callable | None = None,
) -> dict:
    """Fetch metadata and selected exports; an empty selection only lists exports.

    ``downloader`` is an injection point for offline tests. The default is the
    standard huggingface_hub.hf_hub_download implementation.
    """
    release = catalog()[release_name]
    revision = release["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Release revision must be a full commit SHA")
    if all_files and filenames:
        raise ValueError("Choose filenames or all_files, not both")
    if downloader is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError(
                "Install huggingface_hub in a download environment; see docs/DOWNLOADS.md"
            ) from error
        downloader = hf_hub_download

    def fetch(name: str) -> Path:
        return Path(downloader(
            repo_id=release["repository"], filename=_name(name),
            revision=revision, repo_type="model", token=False,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        ))

    manifest_path = verify(fetch(release["manifest_file"]), release["manifest_sha256"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exports = {}
    for entry in manifest["exports"]:
        name = _name(entry["file"])
        _digest(entry["sha256"])
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            raise ValueError(f"Invalid byte count: {name}")
        if name in exports:
            raise ValueError(f"Duplicate manifest filename: {name}")
        exports[name] = entry
    selected = list(exports) if all_files else list(dict.fromkeys(filenames or []))
    for name in selected:
        _name(name)
        if name not in exports:
            raise ValueError(f"File is absent from the pinned export manifest: {name}")

    # Preserve the release's model card and independently hash-bound license files.
    metadata = {"README.md": str(fetch("README.md"))}
    for key in ("model_license", "documentation_license"):
        entry = manifest[key]
        metadata[entry["file"]] = str(verify(fetch(entry["file"]), entry["sha256"]))
    supplementary = {
        "synaptic-lens-targets-adder": ["syn88/population/manifest.json"],
        "synaptic-lens-targets-mnist": ["trajectories.json"],
    }
    for name in supplementary.get(release_name, []):
        metadata[name] = str(fetch(name))
    files = {}
    for name in selected:
        entry = exports[name]
        path = verify(fetch(name), entry["sha256"], entry["bytes"])
        files[name] = {"path": str(path), "sha256": entry["sha256"], "bytes": entry["bytes"]}
    return {
        "repo_id": release["repository"], "revision": revision,
        "manifest": str(manifest_path), "metadata": metadata, "files": files,
        "available_files": [
            {"file": name, "bytes": item["bytes"], "sha256": item["sha256"]}
            for name, item in exports.items()
        ] if not selected else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=sorted(catalog()))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--file", action="append", dest="filenames", help="Exact export path; repeat for several files")
    selection.add_argument("--all", action="store_true", help="Explicitly download every manifest-listed export")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true", help="Use only existing cached files")
    args = parser.parse_args()
    if args.release is None:
        if args.filenames or args.all:
            parser.error("--file and --all require --release")
        print(json.dumps({name: {"repo_id": value["repository"], "revision": value["revision"]}
                          for name, value in catalog().items()}, indent=2))
        return
    try:
        result = download_release(args.release, args.filenames, all_files=args.all,
                                  cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Download failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
