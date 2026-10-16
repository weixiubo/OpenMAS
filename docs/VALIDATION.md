# Validation

The offline suite covers runtime behavior, experiment analysis, configuration,
resources and persistence. Run the commands in [CONTRIBUTING.md](../CONTRIBUTING.md).

| Check | Coverage |
| --- | --- |
| Main suite | 239 cases, including 213 original scenarios |
| Reference workflows | 69 additional cases against original producer modules |
| Runtime comparisons | Cold/warm cache with negotiation enabled/disabled; results, metrics, decisions and event order |
| Stage lifecycle | Interruption, resume, task-index seeds, changed inputs, invalid scheduler references and resource checks |
| Static checks | Ruff, boundary type checking, 600-line modules, 120-line functions and McCabe complexity 15 |
| Coverage gates | Core at least 85%; experiments at least 80% |

The recorded 2026-10-03 coverage run reached 88.24% for the core and 83.44%
for experiments. Runtime comparisons normalize wall-clock timestamps and
generated UUIDs. `TEST_MIGRATION.json` records scenario and assertion mappings.

## Distribution verification

Release checks inspect licenses and resources in both wheels and source
distributions. Installed commands execute outside the source tree. The sdist
check reconstructs the root workspace, creates a fresh locked environment,
runs the complete suite and compares rebuilt wheel contents. Source exports
are checked against `PUBLIC_MANIFEST.json`.

## Scientific evidence

`openagora reproduce` checks the EGOR-v10 file identities and the paired
outcomes of all 800 task rows. The [reproduction guide](../reproduce/README.md)
reports the registered confirmation criteria and resulting counts. Model
experiments use the explicit model and dataset setup in [RESOURCES.md](RESOURCES.md).

## Artifact restoration checks

Ten artifact restoration tests cover Unicode paths, missing and corrupt parts,
repeat execution, modified-file protection, upstream assets, optional
models and independent Git history with reference and link protection. The suite counts above include these cases. The restoration and
research-material checks for the distributed archive edition are recorded in the
[research archive](https://github.com/weixiubo/OpenMAS/releases/tag/research-artifacts-2026-10-17).
