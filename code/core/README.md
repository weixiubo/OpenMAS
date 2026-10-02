# OpenAGORA core

Negotiation-driven multi-agent orchestration with typed tasks, runtime
reconfiguration, cache policy, SQLite/JSONL recording and replay.

Use the repository-root uv workspace on Python 3.11: `uv sync --locked`.
The standalone distribution also installs the `openagora` command.

```bash
openagora doctor
openagora demo
openagora run --config /path/to/experiment.yaml
```

The demo uses a mock provider and needs no external resources. Reproduction of
EGOR evidence additionally requires `openagora-experiments` (included by the root
workspace). See the root README for resource layout and optional dependency groups.
The API is version 0.1; see the migration guide for changed import locations.
