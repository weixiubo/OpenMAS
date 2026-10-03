# Development

OpenMAS uses Python 3.11 and a root uv workspace containing `openagora` and
`openagora-experiments`:

```sh
uv sync --locked
uv run --no-sync pytest
```

The `inference` dependency group supplies local-model libraries; `services`
supplies OpenAI clients. Install the groups required by the selected provider.
Default tests and examples operate offline.

See [contributing](CONTRIBUTING.md) for engineering and distribution checks,
[resources](docs/RESOURCES.md) for data and model configuration, and
[migration](docs/MIGRATION.md) for API and command changes.
