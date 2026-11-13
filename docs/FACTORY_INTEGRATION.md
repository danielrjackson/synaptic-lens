# Canonical target-factory integration

The companion `synaptic-lens-target-model-factory` package owns SYN88 generation,
root splits, label maps and target lineage. SL's existing
`synaptic_lens.syn88_targets` module and CLI are lazy compatibility forwarders.
Core Lens imports and workflows do not require the factory.

The optional `targets` extra is bound to the reviewed factory Git revision
`aec5e0b9558bbecc654504d91aedabf124901bc6`, tree
`1d4798eb1b2843d5945ee00aab0feb097f8de3ef`, rather than an assumed PyPI release.
[Factory draft PR #1](https://github.com/danielrjackson/synaptic-lens-target-model-factory/pull/1)
contains that source; its main currently holds only the reviewed metadata baseline.
The factory remains private: an authorized clone can run the combined gate, while
public users can use SL core without it. Public companion availability and both
follow-up merges remain separate release decisions. No private repository access
is required for SL core. Do not describe optional generation as anonymously
installable until the factory visibility/release gate is resolved.

## Hash-locked combined installation

First follow the isolated installer bootstrap in [dependency setup](DEPENDENCIES.md).
Obtain the factory using your existing authorized Git access, without adding
credentials to commands, manifests or Git URLs:

```powershell
git clone https://github.com/danielrjackson/synaptic-lens-target-model-factory.git ../synaptic-lens-target-model-factory
git -C ../synaptic-lens-target-model-factory checkout --detach aec5e0b9558bbecc654504d91aedabf124901bc6
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
omits the private companion; a core-only pass is not combined release validation.
Missing factory installation gives an explicit error; incompatible age/rule
metadata is rejected before generation. Existing generator arguments and numeric
formats are preserved.
