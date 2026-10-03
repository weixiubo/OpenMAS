# Third-party notices

Project-owned code is licensed under Apache-2.0. External dependencies and
resources retain their respective licenses.

## Dependencies and models

Runtime dependencies are declared in
[code/core/pyproject.toml](code/core/pyproject.toml); experiment dependencies
are declared in [code/experiments/pyproject.toml](code/experiments/pyproject.toml).
Optional inference uses separately obtained model checkpoints. Model sources
and recorded identities are listed in [model_sources.json](docs/model_sources.json)
and the [resource catalog](code/core/src/openagora/resource_catalog.json).
Model weights are external to the distributions and governed by their model
licenses and access terms.

## Research materials

The experiment package includes the EGOR-v10 outcome records and verification
resources described in the [reproduction guide](reproduce/README.md). Their
manifests preserve the original evidence identities and attribution.
Full benchmark images and datasets are obtained separately from their
publishers under the applicable dataset terms. Reference implementations
used by tests are identified in [tests/reference/MANIFEST.json](tests/reference/MANIFEST.json).
