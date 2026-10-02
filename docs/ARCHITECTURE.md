# Architecture

`openagora` owns domain models, negotiation, runtime execution, cache policy,
storage and providers. `OpenAGORAEngine` coordinates the algorithm;
`core.recording.RunRecording` owns manifest creation, ordered audit events,
result persistence, completion and failure. Configuration loading and resource
preflight live at the application boundary. Core mechanisms receive validated
objects; they do not inspect sibling repositories or choose a model service.

`openagora_experiments` owns experiment-specific evidence, planners, source
selection, numerical summaries and EGOR stages. EGOR task schemas are in
`task_model`, answer-blind prompt rendering in `prompts`, strict output parsing
in `parsing`, tensor/image preparation in `model_inputs`, and provider lifecycle
in `local_provider`. `egor.job.StageConfig` is the configured run boundary;
`egor.catalog` validates scheduler templates before inference.

CAGO separates task-source loading, outcome validation, ridge standardization,
normal-equation construction, policy decisions and reporting. EGOR source
cross-validation separates held-source fitting, operating-point summaries and
grid selection. Deterministic split selection separates eligibility, ranking
and interleaved image-unique allocation. The original arithmetic and ordering
are retained and exercised by the migrated tests.

The public CLI offers `doctor`, `demo`, `run --config` and `reproduce`, plus the
runtime's `replay`, `eval`, `manifest verify`, `export` and `template validate`.
CLI modules adapt files/options into domain objects. The runtime never imports
the CLI. Optional experiment imports occur only when selected by a command.

New stage manifests bind source hashes, experiment parameters and input hashes.
They use a new identity, separate from sealed historical experiments. Old
tools/configurations are byte-preserved in the public historical archive;
none is silently relabelled as a run of the refactored implementation.
