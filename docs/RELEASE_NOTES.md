# Release notes

- A root uv workspace manages the runtime and experiment packages.
- `doctor`, `demo`, `run --config` and `reproduce` provide consistent entrypoints.
- Runtime recording, prompt construction, parsing, model inputs and experiment
  analysis have explicit component boundaries.
- Stage recovery checks source, configuration, inputs and task-index ordering.
- Both distributions include license notices; the experiment package includes
  the EGOR-v10 evidence verification resources.

See [migration](MIGRATION.md) for API changes and
[validation](VALIDATION.md) for the test and distribution checks.
