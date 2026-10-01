# Development

Install the locked development environment and both editable packages as
documented in README.md. Keep generated files under `outputs/` and per-project
environments under `.venv/`. Package APIs and scientific result schemas are
compatibility boundaries. Add a regression case when changing policy logic.

The runtime and experiment suite are separate packages; providers and
environment-specific I/O do not belong in evidence contracts or policy math.
Frozen research artifacts have their own source identity and must not be
rewritten to claim results for refactored code.
