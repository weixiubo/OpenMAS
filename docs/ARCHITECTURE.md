# Architecture

`openagora` owns runtime interfaces, provider adapters, storage and the typed
evidence grammar. `mechanisms.evidence_types` defines immutable contracts;
`mechanisms.evidence_gap_repair` plans bounded capability recruitment.

`openagora_experiments.egor` owns the research policies. `evidence` normalizes
generated answers and validates evidence structure. `verification` and
`verification_v9` implement the explicit v8/v9/v10 authorization rules.
The original imports remain available as compatibility exports. The shared
functions accept no reference answers or realized experiment outcomes.

Execution and model providers produce records; planning and authorization
consume them; scoring and statistical analysis remain separate. The final v10
rule uses two heterogeneous specialist corroborators. Earlier variants remain
only where they support current comparisons and regression checks.

Machine-specific launch queues and manuscript-only contracts are preserved in
the private research archive, outside the installable runtime and public CI.
