# Migration

| Previous entry or location | Current entry |
| --- | --- |
| Two editable installs and `requirements-dev.lock` | `uv sync --locked` from the root workspace |
| `python examples/egor_smoke.py` | Same two-support example; `openagora demo` covers runtime and caching |
| `python reproduce/verify.py` | `openagora reproduce`, including installed distributions |
| `doctor --config ...` with database creation | Read-only `doctor`, with optional configuration |
| `code/core/configs/default.yaml` | `configs/runtime.yaml` |
| `code/experiments/scripts/run_egor_stage.py` | `openagora run --config configs/egor-stage.yaml` |
| Version-specific experiment wrappers | Explicit stage configurations |
| `egor.protocol.PilotTask` | `egor.task_model.PilotTask` |
| `egor.protocol.build_*` / `parse_*` | `egor.prompts.build_*` / `egor.parsing.parse_*` |
| Tensor/image helpers in `egor.local_provider` | `egor.model_inputs` |
| Source-local `.openagora` and experiment outputs | Configurable run root, defaulting to `~/research/runs/OpenMAS` |

Run development commands at the workspace root to use the shared lock and
checks. Relative paths in standalone YAML configurations resolve from the
configuration directory. Assign a new run identity when source, inputs or
configuration change.

`TEST_MIGRATION.json` maps regression scenarios. The 800-row EGOR-v10
evidence retains its original identities and is verified independently by
`openagora reproduce`.
