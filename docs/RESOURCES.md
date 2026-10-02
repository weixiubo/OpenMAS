# Resources and local configuration

The installable catalog is `code/core/src/openagora/resource_catalog.json`.
Each record describes purpose, source, revision, license status, expected location,
available checksums and recovery instructions. Missing model/dataset resources
are optional for offline development. `present_unverified` means existence was
checked, not content identity. Unknown checksums are explicitly recorded as null;
do not substitute a guessed value.

Example private file `~/.config/research/OpenMAS.json`:

```json
{
  "paths": {"data": "~/research/data/OpenMAS", "models": "~/research/models", "runs": "~/research/runs/OpenMAS"},
  "resources": {"qwen3-vl-8b": "~/research/models/Qwen3-VL-8B-Instruct"}
}
```

An individual catalog path can also be overridden with
`OPENMAS_RESOURCE_QWEN3_VL_8B`. This changes diagnostics; set the experiment's
`model_path` explicitly (usually `{models}/Qwen3-VL-8B-Instruct`) for execution.
Resource-root CLI options override environment and local JSON values. Public
experiment configuration is the final authority for the files an experiment uses.

The original Qwen3-VL file checksums are recorded in `docs/model_sources.json`.
Full image datasets require their publishers' access and redistribution terms.
Neither tests nor demos restore these automatically. GPU/Transformers inference
is optional and has not been validated by the offline suite.

The former research archive was relocated at directory level to
`~/research/archives`; the old `~/project/_research_archive` path is a compatibility
symlink. The private relocation index and original Git bundles live under
`~/research/archives/refactor-20261002-baseline`. That index is outside the public
repository and records actual machine paths; historical manifests were not edited.
