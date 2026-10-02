# Migration from the pre-refactor public tree

Baseline: `812f3e25997b35e41d4f38453fb49b50f0b8a40b`.

| Previous entry or location | Current entry |
| --- | --- |
| pip installation of two editable packages plus requirements-dev.lock | `uv sync --locked` from the root workspace |
| `python examples/egor_smoke.py` | Kept as the two-support rule example; `openagora demo` exercises runtime and cache |
| `python reproduce/verify.py` | `openagora reproduce`, also available in installed wheels |
| `doctor --config ...` creating a SQLite database | Read-only `doctor`; configuration is optional |
| `code/core/configs/default.yaml` | `configs/runtime.yaml` for the maintained mock entry |
| `code/experiments/scripts/run_egor_stage.py` | `openagora run --config configs/egor-stage.yaml` |
| Version-labelled experiment scripts/configurations | Immutable `reproduce/frozen/openmas-812f3e2.tar.gz`; see its SOURCE.json index |
| `egor.protocol.PilotTask` | `egor.task_model.PilotTask` |
| `egor.protocol.build_*` / `parse_*` | `egor.prompts.build_*` / `egor.parsing.parse_*` |
| Private tensor/image helpers in `egor.local_provider` | `egor.model_inputs` |
| `.openagora` or experiment results in source directories | Configurable `~/research/runs/OpenMAS` defaults |

Existing standalone runtime YAML files still resolve explicit relative paths
against their own directory. Prefer resource placeholders in new public configs.
Use a new run directory for the new stage runner; old stage manifests intentionally
cannot authorize appending new-source results to old evidence.

Every original test case remains represented. Workflow test files retain their
assertions and resolve old scripts through a verified temporary extraction;
the compatibility import is test-only and absent from wheels. The 69 workflow
scenarios also run with the untouched source via `scripts/check_historical.py`.
The GAIA pipeline fixture retains its algorithm settings while resolving output
paths in a temporary directory. Fixed-seed engine references retain their
original bytes; comparisons omit wall-clock timestamps and random UUIDs.

The old `运行.py`, core demo/smoke scripts, duplicate public-tree exporter and
root pip lock are also frozen in that snapshot. Use `openagora`, the root uv.lock
and `scripts/export_public.py`; there is one maintained install/export path.
