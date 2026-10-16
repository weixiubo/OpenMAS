# OpenMAS

OpenMAS provides **OpenAGORA**, a multi-agent runtime, and **EGOR**, an
experimental framework for evidence repair and candidate-blind overwrite
decisions. The `openagora` and `openagora-experiments` packages share one
Python 3.11 workspace.

## Research artifacts

The [artifact guide](docs/ARTIFACTS.md) describes the recorded experiments,
manuscript sources, datasets and historical source versions in the
[research archive](https://github.com/weixiubo/OpenMAS/releases/tag/research-artifacts-2026-10-17).
Downloads are public and include file-level integrity verification.

## Installation

From the repository root on Linux or WSL:

```sh
uv sync --locked
uv run --no-sync openagora doctor
uv run --no-sync openagora demo
uv run --no-sync openagora reproduce
```

`demo` runs a deterministic mock task and verifies cache reuse. Use
`--output /path/to/demo-run` to retain its records. The default environment
supports development, offline examples and evidence verification.

## Experiment execution

```sh
uv run --no-sync openagora run --config configs/runtime.yaml
uv sync --locked --group inference
uv run --no-sync openagora doctor --config configs/egor-stage.yaml
uv run --no-sync openagora run --config configs/egor-stage.yaml
```

`configs/runtime.yaml` selects the mock provider. `configs/egor-stage.yaml`
specifies a local-model baseline stage and requires the configured model
checkpoint and task images. EGOR supports `baseline`, `verifier`, `scheduler`,
`specialist` and `synthesis` stages. The verifier stage uses answer-only
transport. OpenAI client integration is available through `--group services`.

Give each experiment a distinct `run_id`. Stage recovery preserves the
original task-index seed schedule and checks source, input and configuration
identities before continuing.

## Configuration and resources

Source and the project environment reside in the checkout. Default resource
locations are:

```text
~/research/data/OpenMAS/           datasets and task manifests
~/research/data/OpenMAS/evidence/  external evidence
~/research/models/                 model checkpoints
~/research/runs/OpenMAS/            databases and experiment records
```

Resource roots resolve from command-line options, `OPENMAS_*_DIR` environment
variables, `~/.config/research/OpenMAS.json`, then defaults. Experiment paths
support `{data}`, `{models}`, `{runs}` and `{evidence}`; relative paths resolve
from the configuration file. See [resources](docs/RESOURCES.md) for overrides.

`doctor` inspects configuration and resource availability. Use
`doctor --require <id>` for a required catalog item or `doctor --config FILE`
for an experiment's preflight checks.

## Research workflows

The runtime provides negotiation, execution, caching, recording and replay.
The experiment package provides EGOR evidence parsing, planning, two-support
decisions and stage execution, together with CAGO and QA/GAIA analysis APIs.

`reproduce` verifies the packaged EGOR-v10 evidence and recomputes paired
outcomes from 800 task rows. The [reproduction guide](reproduce/README.md)
describes the recorded results, preregistered decision criteria and verifier
outputs. New experiments record their own source and input identities.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) and [repository layout](docs/MATERIALS.md)
- [Development](DEVELOPMENT.md) and [contributing](CONTRIBUTING.md)
- [Migration](docs/MIGRATION.md) and [release notes](docs/RELEASE_NOTES.md)
- [Validation](docs/VALIDATION.md) and [third-party notices](THIRD_PARTY.md)

Project code is licensed under Apache-2.0.
