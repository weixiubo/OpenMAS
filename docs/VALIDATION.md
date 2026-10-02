# Validation report — 2026-10-02

Baseline: `812f3e25997b35e41d4f38453fb49b50f0b8a40b` (213 collected tests).
The AST audit in `TEST_MIGRATION.json` maps every original named test and verifies
that its original assertion expressions are retained. Parameterized cases remain.

## Run successfully

- 229 maintained-suite tests, including all 213 original scenarios and 15 new
  resource/CLI/behavior/recovery tests.
- 69 historical workflow scenarios against the immutable original source;
  these are additional verification, not added to the migrated-case count.
- Exact semantic comparison of four pre-refactor engine executions: cold/warm
  cache with negotiation enabled/disabled. Task results, metrics, cache decisions
  and ordered event payloads match, excluding wall-clock timestamps and UUIDs.
- Ruff checks and formatting; strict types on the core and stage/config boundary.
- All maintained package and utility modules satisfy 600 lines, functions satisfy
  120 lines and complexity 15, with no blanket exceptions.
- Core coverage 86.91% (required 85%); experiments coverage 83.30% (required 80%)
  before the final added CLI preflight case. Both original thresholds passed.
- Two wheels and two sdists include the Apache-2.0 license and required resources.
  Wheels installed into a fresh environment run doctor, a real mock task with
  cache reuse, and 800-row evidence verification from an unrelated directory.
- New stage contracts cover interruption/resume, original-index seeds, changed
  inputs/configuration, missing images/checksums and invalid scheduler references.

## Frozen results

The EGOR-v10 evidence bytes and all existing sidecar hashes are unchanged.
All 800 rows reproduce 5 wins, 1 loss, 794 ties, a 0.005 quality delta and 7
accepted overwrites. The primary scientific gate remains false. The bootstrap
interval is recorded, hash-verified evidence rather than a recomputed interval.

## Resource-dependent validation outstanding

Fresh model inference, GPU device behavior and the complete image datasets have
not been restored or rerun. Optional dependency groups are resolved in uv.lock;
their existence is not a numerical-equivalence claim. Use the resource catalog
and doctor before restoring a pinned model and starting an explicit new run.

A second source directory rebuilt its own environment from uv.lock and passed
the maintained tests, historical tests and demo. Setup downloaded a few missing
small development wheels; subsequent sync and all test execution were offline.

No model or remote experiment service was called and no repository or research material was published.
