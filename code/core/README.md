# OpenAGORA core

OpenAGORA provides negotiation-driven multi-agent orchestration with typed
tasks, runtime reconfiguration, cache policies, SQLite/JSONL recording and
replay. The distribution installs the `openagora` command.

```sh
openagora doctor
openagora demo
openagora run --config /path/to/experiment.yaml
```

The demo uses a deterministic mock provider. EGOR evidence verification also
uses `openagora-experiments`, included in the repository's Python 3.11 uv
workspace. For development, run `uv sync --locked` at the workspace root.
Workspace documentation describes provider dependencies, resource configuration
and API migration.
