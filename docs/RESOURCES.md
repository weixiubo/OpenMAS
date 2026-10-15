# Resources and local configuration

The installed [resource catalog](../code/core/src/openagora/resource_catalog.json)
records purposes, sources, revisions, licenses, locations, known checksums and
acquisition instructions. Offline examples and verification use packaged inputs.

Resource roots resolve from command-line options, environment variables,
local JSON, then defaults. The variables are `OPENMAS_DATA_DIR`,
`OPENMAS_MODELS_DIR`, `OPENMAS_RUNS_DIR` and `OPENMAS_EVIDENCE_DIR`.
Local configuration uses `$XDG_CONFIG_HOME/research/OpenMAS.json`, normally
`~/.config/research/OpenMAS.json`; `OPENMAS_LOCAL_CONFIG` selects another file.

```json
{
  "paths": {
    "data": "~/research/data/OpenMAS",
    "models": "~/research/models",
    "runs": "~/research/runs/OpenMAS",
    "evidence": "~/research/data/OpenMAS/evidence"
  },
  "resources": {"qwen3-vl-8b": "~/research/models/Qwen3-VL-8B-Instruct"}
}
```

`OPENMAS_RESOURCE_QWEN3_VL_8B` overrides that catalog item's diagnostic path.
Experiment execution uses the explicit `model_path` in its configuration,
such as `{models}/Qwen3-VL-8B-Instruct`. Public paths support `{data}`,
`{models}`, `{runs}` and `{evidence}`; other relative paths resolve from the
configuration file.

Local inference requires the `inference` dependency group, a selected model
checkpoint and task images. [model_sources.json](model_sources.json) contains
recorded Qwen3-VL file identities. Obtain image datasets and model files under
their upstream terms, then run `openagora doctor --config FILE` before the
configured stage. OpenAI clients use the separate `services` group.

`doctor` reports `present_unverified` for a resource whose presence is known
but whose content has no matching catalog checksum. A null checksum denotes
unspecified identity. Experiment manifests bind the actual selected inputs.
Diagnostics inspect files and configuration without loading models or starting
services.

## Preserved handoff resources

The [handoff guide](HANDOFF.md) provides restored local research inputs,
resource bindings, exact download identities and historical limitations.
Use those bindings for retained inputs and the setup below for new external
models or services.
