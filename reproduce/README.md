# Reproduction

Run `python reproduce/verify.py` for a CPU-only, network-free check of the
sealed EGOR-v10 task rows and gate. It verifies both SHA-256 bindings before
computing the number of tasks and the paired outcome counts. The original
negative gate remains a negative result.

For a new experiment, install the two packages, obtain the referenced datasets
and model weights under their upstream terms, and pass explicit data, model,
and output locations to `code/experiments/scripts/run_egor_stage.py`. Its
`--help` documents the stage inputs. Configurations are under
`code/experiments/configs`. Use a new output directory; frozen evidence is not
an execution target. Full model reruns require additional model dependencies
and compute and have not been run during this code cleanup.

Original source snapshots, complete research records and reproduction-only
internal checks are preserved in the local private archive. This public
verification path does not depend on those private files.

## Refactored installation

`uv run --no-sync openagora reproduce` verifies the same sealed files from an
installed experiments package. Original evidence bytes and sidecar hashes remain
unchanged. Version-labelled workflow scripts/configurations mentioned in older
research notes now live at their original paths inside
`frozen/openmas-812f3e2.tar.gz`; `frozen/SOURCE.json` indexes their checksums.
`python scripts/check_historical.py` verifies and runs the archived workflow tests
with their original library source. It does not call models or download data.
New model runs use `openagora run --config`; see `docs/MIGRATION.md`.
