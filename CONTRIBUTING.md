# Contributing

Use the root workspace on Python 3.11: `uv sync --locked`. Both packages are
editable members of one environment. Run these checks from the repository root:

```bash
uv run --no-sync ruff check code scripts examples
uv run --no-sync ruff format --check code scripts examples
uv run --no-sync mypy --config-file pyproject.toml
uv run --no-sync python scripts/check_structure.py
uv run --no-sync pytest
uv run --no-sync python scripts/check_historical.py
uv build --all-packages
uv run --no-sync python scripts/check_release.py
```

The maintained library and entry-point limits are 120 lines per function,
600 lines per module and cyclomatic complexity 15. There are currently no named
exceptions. Tests may have longer scenario setup. Frozen source is checked by
SHA-256 and is never reformatted. Add focused failure/recovery tests when changing
stage boundaries, resources, persistence or source bindings.

Do not put datasets, weights, real credentials or complete runs in this tree.
Use `doctor` and the resource catalog to describe missing prerequisites. Default
tests must stay offline. Keep generation prompts answer-blind, preserve original
task indices when resuming, and use a new run when input/config/source identity
changes. Changes to an algorithm or preregistered criterion require an explicitly
new experiment and claim; editing old hashes is not a migration strategy.

Before release, regenerate `PUBLIC_MANIFEST.json` only after reviewing new public
files, then test `scripts/export_public.py`. Building artifacts does not publish
them. The root `LICENSE` and both distribution licenses must agree.

## Source distribution validation

The two library sdists must include their complete package tests and fixtures.
The root workspace metadata and shared reference drivers come from the public
source export. After `uv build --all-packages`, run
`uv run --no-sync python scripts/check_sdist.py`. It reconstructs that workspace
using the sdists, creates a fresh environment from the root lock, runs the full
suite, rebuilds both wheels, and checks installed commands outside the sources.
