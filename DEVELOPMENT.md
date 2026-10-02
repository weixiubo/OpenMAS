# Development

Use the repository-root Python 3.11 uv workspace: `uv sync --locked`.
Each repository owns its `.venv`; no sibling repository or previous virtual
environment is required. See [contributing](CONTRIBUTING.md) for validation,
[resources](docs/RESOURCES.md) for local configuration, and
[migration](docs/MIGRATION.md) for the old-to-new entry mapping.

Optional groups: `inference` for local models and `services` for OpenAI clients.
Default tests and demos are offline. Store full datasets, models and experiment
outputs under your configurable research roots, outside this Git repository.
