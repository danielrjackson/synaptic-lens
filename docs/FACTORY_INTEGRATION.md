# Canonical target-factory integration

The companion `synaptic-lens-target-model-factory` package owns the SYN88
generator, root split implementation, label maps and target lineage manifests.
SL keeps the existing `synaptic_lens.syn88_targets` module and CLI as thin lazy
forwarders. Ages/rule names remain immutable protocol metadata, not a second
implementation. Core Lens imports and workflows do not require the factory.

The optional `targets` extra requires factory version 0.1.0. Install the reviewed
factory source or wheel first, then install SL with `.[test,targets]`. This does
not assume a PyPI publication. The existing public generator commands keep their
arguments and numeric format. Missing companion installation gives an explicit
error; incompatible age/rule metadata is rejected before generation.

This transition branch must not be integrated until the factory repository has
an approved public bootstrap and an independently reviewed installable source
revision. Pin that approved revision in installation instructions. The current
reviewed factory content is available as a local wheel for combined-package
validation; no private repository access is a supported public runtime dependency.

Default core tests may omit optional generator tests when the companion is absent;
the combined release gate installs both packages and runs every generator test.
Forwarding, missing-dependency and protocol-drift contracts remain core tests.
