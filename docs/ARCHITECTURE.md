# Architecture

`openagora` owns domain models, negotiation, execution, caching, storage and
providers. `OpenAGORAEngine` coordinates the runtime;
`core.recording.RunRecording` manages manifests, ordered audit events,
persisted results, completion and failure. Application boundaries resolve
configuration and resources before passing validated objects to the core.

`openagora_experiments` owns experiment-specific evidence, planners, source
selection, numerical summaries and EGOR stages:

| Module | Responsibility |
| --- | --- |
| `egor.task_model` | Task schemas |
| `egor.prompts` | Answer-blind prompt construction |
| `egor.parsing` | Structured output parsing |
| `egor.model_inputs` | Tensor and image preparation |
| `egor.local_provider` | Provider lifecycle |
| `egor.job.StageConfig` | Configured stage execution |
| `egor.catalog` | Scheduler-template validation |

CAGO separates task loading, outcome validation, ridge standardization,
normal-equation construction, policy decisions and reporting. EGOR
cross-validation separates held-source fitting, operating-point summaries
and grid selection. Deterministic split selection applies eligibility,
ranking and interleaved image-unique allocation in a defined order.

The CLI exposes `doctor`, `demo`, `run --config` and `reproduce`, together
with `replay`, `eval`, `manifest verify`, `export` and `template validate`.
CLI modules adapt files and options into domain objects. Optional experiment
imports occur when a command selects them.

Stage manifests bind source, parameter and input identities. Regression
comparisons use original producer modules listed in
`tests/reference/MANIFEST.json`. Scientific evidence is packaged separately
from executable runtime logic and verified by the reproduction command.
