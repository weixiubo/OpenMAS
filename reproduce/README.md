# EGOR-v10 evidence reproduction

```sh
uv run --no-sync openagora reproduce
```

The command verifies the packaged EGOR-v10 SHA-256 bindings and recomputes
paired outcomes from all 800 task rows. `python reproduce/verify.py` provides
the repository entrypoint. Verification runs on the CPU using the supplied
records and requires no network access.

## Recorded confirmation result

The panel contains 5 wins, 1 loss and 794 ties, with a quality delta of 0.005
and 7 accepted overwrites. The primary confirmation did not meet its
preregistered confidence-interval and overwrite-breadth criteria. The verifier
checks the recorded bootstrap interval through its evidence bindings;
outcome arithmetic is recomputed from the task rows.

## Experiment execution

New model experiments use `openagora run --config`, with
`configs/egor-stage.yaml` as a starting configuration. Supply the model
checkpoint, task inputs and output location described in
[resources](../docs/RESOURCES.md), and assign a distinct run ID. The configured
runtime records the source and inputs used by that experiment.

Workflow regression comparisons run through
`uv run --no-sync python scripts/check_historical.py`. Their original producer
modules and hashes are listed in `tests/reference/MANIFEST.json`.

The current public edition omits recorded model-message bodies. Its integrity
indexes bind the published records; recorded statistical outcomes are retained.
