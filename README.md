# OpenMAS

OpenMAS contains the **OpenAGORA** multi-agent runtime and the **EGOR**
candidate-blind evidence repair and overwrite methods. It is one repository
with two installable Python packages: `openagora` and `openagora-experiments`.

## Install

Use Python 3.11 or later on Linux/WSL. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.lock
python -m pip install -e code/core -e code/experiments
```

The lock records the tested development dependencies. Model execution is
optional; the examples and core tests do not download model weights or use an
API key. Configure a local model path or a provider endpoint explicitly for
real-model experiments. Keep downloaded weights in your user cache.

## Run and test

```bash
openagora doctor --config code/core/configs/default.yaml
python examples/egor_smoke.py
python reproduce/verify.py
python -m pytest -q code/core/tests code/experiments/tests
```

`examples/egor_smoke.py` exercises the two-support authorization rule with
fixed, constructed evidence. It is a functional example, not a benchmark.

## Research status

The current scientific line is **EGOR-v10**, candidate-blind two-support
overwrite. Its sealed 800-task confirmation **failed** the preregistered
strict confidence-interval and overwrite-breadth criteria. This result is
retained transparently in `reproduce/egor-v10`; the verifier checks its frozen
hashes and recomputes counts from the task rows. Historical v8 positive results
are not the current primary claim. Refactoring does not constitute a new model
run or a new confirmation experiment.

## Layout

- `code/core`: runtime, providers, storage, typed evidence and repair planner.
- `code/experiments`: EGOR execution, selection, verification and analysis.
- `examples`: small offline examples.
- `reproduce`: frozen result checks and reproduction requirements.
- `docs`: architecture, migration and public model-source records.

See [architecture](docs/ARCHITECTURE.md), [reproduction](reproduce/README.md),
[migration](docs/MIGRATION.md) and [third-party notices](THIRD_PARTY.md).
