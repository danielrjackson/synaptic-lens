# Download released models from Hugging Face

The four published model repositories separate Lens readers from target networks.
Each checkout pins their complete Hugging Face commit IDs and export-manifest
SHA-256 values in [releases.json](releases.json).

| Use | Release name |
| --- | --- |
| Arithmetic Lens readers | `synaptic-lens-adder` |
| Image-network Lens readers | `synaptic-lens-mnist` |
| Arithmetic target population | `synaptic-lens-targets-adder` |
| MNIST target trajectories | `synaptic-lens-targets-mnist` |

## Install the download client

Use a separate small environment for downloads. The downloaded paths can be
passed to inference running in your existing PyTorch environment. This keeps
optional network tooling separate from the training environment. The maintained
CUDA environment also includes it with `uv sync --locked --extra hub`.

```powershell
py -3.15 -m venv .venv-hub
.\.venv-hub\Scripts\python.exe -m pip install "huggingface_hub==2.4.0"
.\.venv-hub\Scripts\python.exe scripts/download_models.py
```

On Linux/macOS, use `python3 -m venv .venv-hub` and
`.venv-hub/bin/python` for the same commands. Hub 2.4.0 was tested on Windows;
the install command fixes the client version and resolves its dependencies through
pip. The download client is independent of the model-training environment.

## Select files explicitly

List a release's available exports and fetch its small metadata files:

```powershell
.\.venv-hub\Scripts\python.exe scripts/download_models.py --release synaptic-lens-adder
```

Download one Lens checkpoint and one target root:

```powershell
.\.venv-hub\Scripts\python.exe scripts/download_models.py --release synaptic-lens-adder --file syn88/lens-trajectory-seed-27.npz
.\.venv-hub\Scripts\python.exe scripts/download_models.py --release synaptic-lens-targets-adder --file syn88/population/root-0000.npz
```

For MNIST, select the reader and coordinate schema that match your task:

```powershell
.\.venv-hub\Scripts\python.exe scripts/download_models.py --release synaptic-lens-mnist --file lens-full_raw_weights.pt
.\.venv-hub\Scripts\python.exe scripts/download_models.py --release synaptic-lens-targets-mnist --file coordinate-schema.json
```

List the MNIST target release to choose a checkpoint path from a named trajectory.
Repeat `--file` for multiple exports. `--all` explicitly downloads every export
listed in a release manifest, including the full population for a target release.
Review the listing's byte counts before requesting a whole population.

Each command prints JSON with the pinned revision, verified local file paths,
SHA-256 digests and byte counts. Copy the paths and hashes into the matching
inference example's arguments. A single root is useful for inspecting predictions;
research comparisons use the full evaluation protocol and root-level partitions.

## Cache, integrity and provenance

The helper uses the standard `huggingface_hub.hf_hub_download` function. Repeated
requests reuse its version-aware cache. `--cache-dir PATH` chooses a cache location;
`--local-files-only` reuses an already populated cache without a network request.
Keep cached files unchanged. Verified model paths resolve the Hub snapshot links
to their regular cache blobs, so they work with the strict local checkpoint readers.

All requests use a complete release commit ID and anonymous access (`token=False`).
The helper verifies the export manifest against the digest committed in this
repository, then verifies each selected model's size and SHA-256. It also downloads
the model card and hash-verifies the two license files. Target population/trajectory
metadata is fetched from the same pinned commit. Model cards and supplementary
metadata are revision-pinned; the export and license files additionally carry
independently checked SHA-256 values. A failed check stops the command and leaves
cached bytes intact for inspection. The helper performs file downloads and hashing;
checkpoint loading is handled by the project's matching inference APIs.

Read the release model card, licenses and provenance before use. Model exports
carry the release's Research-use RAIL-M terms; accompanying authored software and
documentation carry Research-use RAIL-S terms. Dataset and other third-party terms
remain with their respective owners. Preserve the metadata when redistributing
an authorized copy. MNIST targets here are trained-network checkpoints.

## Standard Hub alternatives

Hugging Face also provides `hf download` for the command line,
`hf_hub_download` for individual files, and `snapshot_download` for selected
patterns or complete repositories. Use a full `--revision` / `revision` commit ID
from `releases.json`. The helper above adds this project's manifest and payload
checks to the standard client. See the official
[download guide](https://huggingface.co/docs/huggingface_hub/guides/download)
and [cache guide](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

## Offline tests

The helper's unit tests use local fixtures and an injected download function:

```powershell
python -m unittest discover -s tests -p test_hub_downloads.py
```

For Synaptic Lens, use `-s tests/core`. These tests check revision pinning,
selection, hash/size failures, safe relative paths, and anonymous/offline options.
