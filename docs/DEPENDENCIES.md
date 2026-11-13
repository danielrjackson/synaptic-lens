# Locked Windows CPU environment

The maintained reproducibility profile is Windows x64, CPython 3.13 (validated
locally with 3.13.3), CPU PyTorch 2.15.0 and NumPy 2.5.4. It is separate from
historical scientific environments and does not establish byte-exact historical
initialization or reproduction of published training scores. Other Python,
accelerator and operating-system profiles are not covered by these locks.

`requirements/windows-cpu.in` intentionally pins every direct and transitive
runtime, test and build dependency. `windows-cpu.txt` contains archive SHA-256
hashes from package-index metadata. `installer.txt` similarly pins uv 0.7.4.
`integrity.json` binds the input/output bytes and LF-normalized pyproject metadata.
Hashes establish artifact identity, not a package's safety or authorship.

From a fresh checkout in PowerShell, with Windows x64 Python 3.13 installed:

```powershell
if (Test-Path .venv-installer) { throw "Choose a new installer environment path" }
py -3.13 -m venv .venv-installer
.\.venv-installer\Scripts\python.exe -m pip install --require-hashes --only-binary=:all: -r requirements/installer.txt
.\.venv-installer\Scripts\python.exe scripts/locked.py install
.\.venv-installer\Scripts\python.exe scripts/locked.py test
```

The bootstrap uses its own fresh environment rather than installing uv into a
shared Python. The script prefers uv beside that bootstrap interpreter.
The installer refuses an existing environment directory; choose a new one with
`--environment PATH` to preserve other environments. It uses hash-required,
wheel-only exact synchronization, builds this checkout's wheel with the locked
setuptools 84.2.1 backend, and installs that local wheel without resolving more
dependencies or downloading an isolated backend. The local project wheel is bound
to the reviewed Git source rather than included among third-party archive hashes.
Tests verify installed package location and dependency versions before the bounded
CPU suite. No source or dataset download, model upload or GPU execution occurs.
Only package wheels are downloaded. The Python distribution and initial pip are
bootstrap trust inputs; this repository does not claim to hash-lock the OS or Python.

The `Locked Windows CPU validation` GitHub workflow is manual-only, one Windows
job, with actions pinned by commit. It runs this same installation and test path.
No hosted run is required or launched for routine edits. Core Lens installation
does not require the companion factory. Tests requiring that optional companion
run in the separately reviewed combined checkout, after the factory is available;
a core-only CI pass must not be reported as combined validation.

## Optional pinned factory

For a fully locked combined install, use the [exact companion source gate](FACTORY_INTEGRATION.md).
Its profile adds Pillow to the same core dependency versions. The factory source
is pinned by Git commit/tree and its reviewed integrity manifest. It remains
private pending a separate release decision, so ordinary public core installation
continues to work independently. Never replace the pinned Git source with an
unverified local generator copy.

## Reviewed updates

Do not regenerate against unconstrained newest dependencies. Edit the intentional
pins, assess the affected packages and then regenerate with uv 0.7.4:

```powershell
uv pip compile requirements/windows-cpu.in --python-version 3.13 --python-platform windows --generate-hashes --no-build --no-header --no-annotate -o requirements/windows-cpu.txt
uv pip compile requirements/installer.in --python-version 3.13 --python-platform windows --generate-hashes --no-build --no-header --no-annotate -o requirements/installer.txt
```

Refresh `integrity.json` hashes for changed inputs, outputs and normalized
pyproject, review the exact lock diff, then repeat a fresh locked installation,
the relevant CPU suites and an independent exact-diff review. Unchanged inputs
must regenerate identical output. Requirement/manifest files use LF for stable
regeneration; maintained Windows source and documentation use CRLF. Frozen
scientific evidence and original license/source bytes retain their existing policy.

The conventional `pip install -e ".[test]"` route follows permissive package
compatibility bounds and is not a locked reproduction path. Do not use its success
as evidence that the frozen profile installs correctly.
