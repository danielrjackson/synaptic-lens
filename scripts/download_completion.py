"""Download the pinned neuron-completion release and verify every payload."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "docs/neuron-completion-manifest.json"


def download(output, *, cache_dir=None, local_files_only=False, fetch=None):
    if fetch is None:
        from huggingface_hub import hf_hub_download

        fetch = hf_hub_download
    manifest = json.loads(MANIFEST.read_text())
    hub = manifest["huggingface"]
    if (
        not isinstance(hub.get("revision"), str)
        or re.fullmatch(r"[0-9a-f]{40}", hub["revision"]) is None
    ):
        raise ValueError("A complete immutable Hub revision is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, identity in manifest["files"].items():
        source = Path(
            fetch(
                repo_id=hub["repo_id"],
                filename=hub["subfolder"] + "/" + name,
                revision=hub["revision"],
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        )
        payload = source.read_bytes()
        if (
            len(payload) != identity["bytes"]
            or hashlib.sha256(payload).hexdigest() != identity["sha256"]
        ):
            raise ValueError("Artifact identity mismatch: " + name)
        destination = output / name
        if destination.exists() and destination.read_bytes() != payload:
            raise ValueError(
                "Refusing to overwrite different existing artifact: " + name
            )
        if destination.resolve() != source.resolve():
            destination.write_bytes(payload)
        result[name] = str(destination.resolve())
    shutil.copyfile(MANIFEST, output / "manifest.json")
    return {"revision": hub["revision"], "files": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/neuron-completion")
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            download(
                args.output,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
