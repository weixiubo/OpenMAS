# Contributing

Use the root Python 3.11 workspace. Both packages are editable members of the
environment created by `uv sync --locked`. Run checks from the repository root:

```sh
uv run --no-sync ruff check code scripts examples
uv run --no-sync ruff format --check code scripts examples
uv run --no-sync mypy --config-file pyproject.toml
uv run --no-sync python scripts/check_structure.py
uv run --no-sync pytest
uv run --no-sync python scripts/check_historical.py
uv build --all-packages
uv run --no-sync python scripts/check_release.py
uv run --no-sync python scripts/check_sdist.py
```

Keep runtime mechanisms independent of configuration and provider setup.
Changes to stage boundaries, resources, persistence or source bindings should
include focused regression and failure-recovery tests. Generation prompts
remain answer-blind; resumed stages preserve original task indices.

Maintained modules have a 600-line limit, functions a 120-line limit and
McCabe complexity a limit of 15. Regression source retains its original bytes
and is verified by SHA-256. Algorithm or protocol changes receive a distinct
experiment identity and retain the original evidence for comparison.

Keep datasets, model weights, credentials and complete run outputs in the
configured resource locations. Default tests use offline inputs.

Before a source export, review the file set and update `PUBLIC_MANIFEST.json`.
`scripts/export_public.py` verifies each file's identity. The release checks
inspect both distributions and execute installed commands outside the source
tree. The sdist check reconstructs the workspace from both source distributions
and the root metadata, runs the complete suite in a fresh locked environment,
and compares rebuilt wheel contents. Dependency packages must be cached for
this offline check. The root and both package licenses must agree.
