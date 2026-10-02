# Migration

| Previous entry or location | Current entry |
| --- | --- |
| pip installation of two editable packages plus requirements-dev.lock | `uv sync --locked` from the root workspace |
| `python examples/egor_smoke.py` | Kept as the two-support rule example; `openagora demo` exercises runtime and cache |
| `python reproduce/verify.py` | `openagora reproduce`, also available in installed wheels |
| `doctor --config ...` creating a SQLite database | Read-only `doctor`; configuration is optional |
| `code/core/configs/default.yaml` | `configs/runtime.yaml` for the maintained mock entry |
| `code/experiments/scripts/run_egor_stage.py` | `openagora run --config configs/egor-stage.yaml` |
| Version-labelled experiment scripts/configurations | Supported stages use run configs; unused wrappers removed |
| `egor.protocol.PilotTask` | `egor.task_model.PilotTask` |
| `egor.protocol.build_*` / `parse_*` | `egor.prompts.build_*` / `egor.parsing.parse_*` |
| Private tensor/image helpers in `egor.local_provider` | `egor.model_inputs` |
| `.openagora` or experiment results in source directories | Configurable `~/research/runs/OpenMAS` defaults |

Versioned experiment wrappers without a supported current entry are removed.
The supported EGOR stages use explicit run configurations. Workflow regression
tests use only their required original inputs under `tests/reference`, against
the maintained experiment library; the independent workflow check also tests
the reference producer. The CLI, workspace lock and public exporter are the
single maintained installation and execution path.

Existing standalone YAML files resolve explicit relative paths against their own
directory. New stage runs use a new run identity; they cannot append new-source
results to old evidence. The original 800-row evidence and confirmation gate
remain unchanged and are checked by `openagora reproduce`.

See [MATERIALS.md](MATERIALS.md) for material ownership and deletion rules.
