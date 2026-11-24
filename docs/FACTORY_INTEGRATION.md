# Canonical target-factory integration

The companion `synaptic-lens-target-model-factory` package owns SYN88 generation,
label maps and target lineage. SL's existing generation API and CLI are lazy
compatibility forwarders. For existing population readers, Lens bundles the fixed
published root lists as data, checked for exact parity with the factory. Core
Lens imports and workflows do not require the factory.

The optional `targets` extra is bound to the reviewed factory Git revision
`aec5e0b9558bbecc654504d91aedabf124901bc6`, tree
`1d4798eb1b2843d5945ee00aab0feb097f8de3ef`, rather than an assumed PyPI release.
This revision is publicly reachable. Its Windows CPU profile currently pins NumPy
2.5.4, so it cannot pass the exact combined gate with Lens's NumPy 2.5.5 floor.
The combined commands below are reserved for a compatible, reviewed factory pin.
Core installation does not require the factory. For the current locked CUDA
environment, install generation support with `uv sync --locked --extra targets`
(and add `--extra test` to run tests).

## Hash-locked combined installation

First follow the isolated installer bootstrap in [dependency setup](DEPENDENCIES.md).
Once this Lens checkout pins a compatible, reviewed factory revision, clone
the public factory and check out that exact commit:

```powershell
git clone https://github.com/danielrjackson/synaptic-lens-target-model-factory.git ../synaptic-lens-target-model-factory
$factoryRevision = (Get-Content requirements/companion.json -Raw | ConvertFrom-Json).commit
git -C ../synaptic-lens-target-model-factory checkout --detach $factoryRevision
.\.venv-installer\Scripts\python.exe scripts/locked.py install --environment .venv-combined --factory-checkout ../synaptic-lens-target-model-factory
.\.venv-installer\Scripts\python.exe scripts/locked.py test --environment .venv-combined --factory-checkout ../synaptic-lens-target-model-factory
```

Use a fresh clone/path; preserve any existing worktree. `requirements/companion.json`
binds repository URL, source commit/tree and the factory lock-integrity manifest.
The installer rejects a different source revision, dirty checkout or lock identity
before creating an environment. It verifies both lock profiles, installs the
factory's hash-required transitive Windows CPU profile (a superset of core), and
builds the companion from a fresh tracked Git archive of the verified commit,
excluding ignored source/build residue. It builds both local wheels with the
locked backend without dependency resolution.
The combined test command exercises both installed packages with source-path
injection disabled and runs every optional generator test. No large training,
model download or raw MNIST access is required.

The conventional optional-extra installer may fetch this exact Git source, but
it is not the frozen dependency/build path above. Core-only manual CI intentionally
omits the optional companion; a core-only pass is not combined release validation.
Missing factory installation gives an explicit error; incompatible age/rule
metadata is rejected before generation. Existing generator arguments and numeric
formats are preserved.
