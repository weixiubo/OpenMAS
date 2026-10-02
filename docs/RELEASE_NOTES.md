# Local release preparation

This refactor introduces a root uv workspace and portable `doctor`, `demo`,
`run --config` and `reproduce` commands. Both distributions include the Apache-2.0
license. Offline reproduction evidence is included in the experiments wheel.

The runtime recording lifecycle, prompt/parser/model-input boundaries and
experiment fitting/selection/reporting phases are separated. Stage recovery
binds the original task order, inputs, configuration and new source identity.
Required historical workflows are retained as hash-verified regression inputs.

No scientific gate has been changed. EGOR-v10 remains a failed primary
confirmation; the immutable EGOR-v10 evidence remains authoritative. This is a
local release candidate, not a GitHub publication or a new model experiment.
