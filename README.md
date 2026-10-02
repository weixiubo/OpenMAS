# OpenMAS

OpenMAS contains **OpenAGORA**, a multi-agent runtime, and **EGOR**, evidence
repair and candidate-blind overwrite experiments. The two Python packages are
`openagora` and `openagora-experiments`, managed in one uv workspace.

## Install and check

Use Linux/WSL and Python 3.11. From this repository:

```bash
uv sync --locked
uv run --no-sync openagora doctor
uv run --no-sync openagora demo
uv run --no-sync openagora reproduce
```

The default environment, tests and demos use no model API and download no model
weights or datasets. `demo` executes a constructed mock task and checks cache
reuse. Its metrics are synthetic, not an empirical model benchmark. Add
`--output /path/to/empty-demo-run` to retain its artifacts.

`reproduce` verifies the packaged EGOR-v10 file hashes and recomputes outcomes
from all 800 task rows. The primary confirmation **failed** its preregistered
confidence-interval and overwrite-breadth gates. This negative result is
preserved. The reported bootstrap interval is hash-bound recorded evidence;
the arithmetic verifier does not regenerate it or rerun model inference.

## Run an explicit experiment

```bash
uv run --no-sync openagora run --config configs/runtime.yaml
uv run --no-sync openagora doctor --config configs/egor-stage.yaml
# After restoring the required pinned model and task images:
uv sync --locked --group inference
uv run --no-sync openagora run --config configs/egor-stage.yaml
```

`configs/runtime.yaml` runs the core with a mock provider. `egor-stage.yaml`
shows an explicit real-model baseline stage. Supported stages are `baseline`,
`verifier` (answer-only transport), `scheduler`, `specialist` and `synthesis`.
Set a fresh `run_id` for a new experiment. A resumed stage preserves the original
task-index seed schedule and refuses changed inputs, configuration or source.
Optional inference dependencies are locked but GPU/model execution is a
separate validation task; installing them does not establish a research result.
OpenAI service integration is installed separately with `--group services`.

## Resource locations

```text
~/project/OpenMAS/            source and .venv
~/research/data/OpenMAS/      external datasets and task manifests
~/research/models/           explicitly restored model revisions
~/research/runs/OpenMAS/      databases, records and experiment outputs
~/research/archives/          original sources and recovery material
```

Resource roots resolve in this order: command options (`--data-dir`,
`--models-dir`, `--runs-dir`), `OPENMAS_DATA_DIR` / `OPENMAS_MODELS_DIR` /
`OPENMAS_RUNS_DIR` / `OPENMAS_ARCHIVES_DIR`, then
`~/.config/research/OpenMAS.json`, then the layout above. `XDG_CONFIG_HOME` and
`OPENMAS_LOCAL_CONFIG` can relocate the local configuration. A local JSON
example is in [resources](docs/RESOURCES.md). Public experiment paths support
`{data}`, `{models}`, `{runs}` and `{archives}`; other relative paths resolve
against the configuration file, independent of the working directory.

No package import creates directories. `doctor` is read-only and never creates
a database, loads a model or contacts a service. Use `doctor --require <id>` to
require a catalog resource. Missing optional resources do not prevent the
offline demo. See the [resource catalog](code/core/src/openagora/resource_catalog.json).

## Research and maintenance scope

- Maintained: runtime negotiation/execution/caching, explicit local-provider
  boundary, EGOR evidence parsing, planning, two-support decisions and configured
  stage execution; CAGO and QA/GAIA analysis remain library APIs.
- Frozen: previous experiment orchestration/configuration and sealed evidence.
  The source snapshot and original checksums are under `reproduce/frozen`.
  Historical workflow tests run against current library functions through a
  test-only import adapter and separately against their untouched source.
- Not established by this cleanup: fresh GPU experiments, full dataset recovery,
  model numerical equivalence across dependency versions, or new scientific claims.

Run `uv run --no-sync pytest` for all migrated scenarios and new contracts.
See [architecture](docs/ARCHITECTURE.md), [contributing](CONTRIBUTING.md),
[migration](docs/MIGRATION.md), [validation](docs/VALIDATION.md),
[reproduction](reproduce/README.md) and [third-party notices](THIRD_PARTY.md).
